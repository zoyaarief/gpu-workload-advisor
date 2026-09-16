#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kMaxMatrixSize = 4096;
constexpr int kMaxRepeats = 50;
constexpr int kDefaultRepeats = 5;

struct Measurement {
    std::string implementation;
    bool available = false;
    std::vector<double> trial_latencies_ms;
    bool correct = false;
    // Wall-clock window covering every trial, used to align GPU telemetry.
    double started_unix_ms = 0.0;
    double finished_unix_ms = 0.0;
    std::string error;
};

struct RowRange {
    int first_row;
    int row_count;
};

Measurement unavailable(const std::string& implementation, const std::string& reason) {
    Measurement measurement;
    measurement.implementation = implementation;
    measurement.error = reason;
    return measurement;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Owns one device allocation so failed trials cannot leak GPU memory.
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const { return data_; }

    void release() {
        check_cuda(cudaFree(data_), "cudaFree");
        data_ = nullptr;
    }

private:
    float* data_ = nullptr;
};

std::string json_escape(const std::string& value) {
    std::string escaped;
    for (char character : value) {
        if (character == '\\' || character == '"') {
            escaped += '\\';
        }
        escaped += character;
    }
    return escaped;
}

template <typename Function>
double elapsed_ms(Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double unix_time_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

// Times `run` once per trial. `check` runs outside the timed region and must
// pass on every trial for the measurement to count as correct.
template <typename Run, typename Check>
Measurement measure(const std::string& implementation, int repeats, Run&& run, Check&& check) {
    Measurement measurement;
    measurement.implementation = implementation;
    measurement.available = true;
    measurement.correct = true;
    measurement.started_unix_ms = unix_time_ms();
    for (int trial = 0; trial < repeats; ++trial) {
        measurement.trial_latencies_ms.push_back(elapsed_ms(run));
        measurement.correct = check() && measurement.correct;
    }
    measurement.finished_unix_ms = unix_time_ms();
    return measurement;
}

// Reports the team size of a real parallel region. If the build dropped the
// OpenMP compiler flag, the pragmas are ignored and this returns 1.
int openmp_team_size() {
    int team_size = 1;
#pragma omp parallel
    {
#pragma omp single
        team_size = omp_get_num_threads();
    }
    return team_size;
}

void multiply_host(const std::vector<float>& left,
                   const std::vector<float>& right,
                   std::vector<float>& output,
                   int size,
                   bool use_openmp) {
    std::fill(output.begin(), output.end(), 0.0F);

    auto multiply_row = [&](int row) {
        for (int inner = 0; inner < size; ++inner) {
            const float left_value = left[row * size + inner];
            for (int column = 0; column < size; ++column) {
                output[row * size + column] +=
                    left_value * right[inner * size + column];
            }
        }
    };

    if (use_openmp) {
#pragma omp parallel for schedule(static)
        for (int row = 0; row < size; ++row) {
            multiply_row(row);
        }
    } else {
        for (int row = 0; row < size; ++row) {
            multiply_row(row);
        }
    }
}

// Multiplies `row_count` rows of the left matrix by the full right matrix.
// Row indices are local to the slice the device received.
__global__ void matrix_multiply_kernel(const float* left_rows,
                                       const float* right,
                                       float* output_rows,
                                       int row_count,
                                       int size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= row_count || column >= size) {
        return;
    }

    float value = 0.0F;
    for (int inner = 0; inner < size; ++inner) {
        value += left_rows[row * size + inner] * right[inner * size + column];
    }
    output_rows[row * size + column] = value;
}

bool approximately_equal(const std::vector<float>& expected,
                         const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        return false;
    }

    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float tolerance = 1.0e-3F + 1.0e-3F * std::abs(expected[index]);
        if (std::abs(expected[index] - actual[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

std::vector<RowRange> partition_rows(int size, int parts) {
    std::vector<RowRange> ranges;
    int first_row = 0;
    for (int part = 0; part < parts; ++part) {
        const int row_count = size / parts + (part < size % parts ? 1 : 0);
        ranges.push_back({first_row, row_count});
        first_row += row_count;
    }
    return ranges;
}

// Allocation, both transfers, the kernel, and deallocation are all timed.
void multiply_on_device(int device,
                        const RowRange& rows,
                        const std::vector<float>& left,
                        const std::vector<float>& right,
                        std::vector<float>& output,
                        int size) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");

    const std::size_t columns = static_cast<std::size_t>(size);
    const std::size_t offset = static_cast<std::size_t>(rows.first_row) * columns;
    const std::size_t slice_bytes =
        static_cast<std::size_t>(rows.row_count) * columns * sizeof(float);
    const std::size_t right_bytes = right.size() * sizeof(float);

    DeviceBuffer device_left(slice_bytes);
    DeviceBuffer device_right(right_bytes);
    DeviceBuffer device_output(slice_bytes);
    check_cuda(cudaMemcpy(device_left.get(), left.data() + offset, slice_bytes,
                          cudaMemcpyHostToDevice),
               "copy left rows to GPU");
    check_cuda(cudaMemcpy(device_right.get(), right.data(), right_bytes,
                          cudaMemcpyHostToDevice),
               "copy right to GPU");

    const dim3 threads(16, 16);
    const dim3 blocks((size + threads.x - 1) / threads.x,
                      (rows.row_count + threads.y - 1) / threads.y);
    matrix_multiply_kernel<<<blocks, threads>>>(
        device_left.get(), device_right.get(), device_output.get(), rows.row_count, size);
    check_cuda(cudaGetLastError(), "CUDA kernel launch");
    check_cuda(cudaDeviceSynchronize(), "CUDA kernel execution");
    check_cuda(cudaMemcpy(output.data() + offset, device_output.get(), slice_bytes,
                          cudaMemcpyDeviceToHost),
               "copy result to CPU");

    device_left.release();
    device_right.release();
    device_output.release();
}

// Splits the output rows evenly across devices. Each device gets its slice of
// the left matrix and a full copy of the right matrix; devices never talk to
// each other, and each writes a disjoint range of `output`.
void multiply_on_devices(int device_count,
                         const std::vector<float>& left,
                         const std::vector<float>& right,
                         std::vector<float>& output,
                         int size) {
    const std::vector<RowRange> ranges = partition_rows(size, device_count);
    if (device_count == 1) {
        multiply_on_device(0, ranges[0], left, right, output, size);
        return;
    }

    std::vector<std::string> errors(device_count);
    std::vector<std::thread> workers;
    workers.reserve(device_count);
    try {
        for (int device = 0; device < device_count; ++device) {
            workers.emplace_back([&, device] {
                try {
                    multiply_on_device(device, ranges[device], left, right, output, size);
                } catch (const std::exception& error) {
                    errors[device] = "GPU " + std::to_string(device) + ": " + error.what();
                }
            });
        }
    } catch (...) {
        for (std::thread& worker : workers) {
            worker.join();
        }
        throw;
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (const std::string& error : errors) {
        if (!error.empty()) {
            throw std::runtime_error(error);
        }
    }
}

Measurement measure_cuda(const std::string& implementation,
                         int device_count,
                         int repeats,
                         const std::vector<float>& left,
                         const std::vector<float>& right,
                         const std::vector<float>& expected,
                         std::vector<float>& output,
                         int size) {
    try {
        // Initialize each CUDA context before timing. Allocations and transfers
        // remain inside every timed trial.
        for (int device = 0; device < device_count; ++device) {
            check_cuda(cudaSetDevice(device), "cudaSetDevice");
            check_cuda(cudaFree(nullptr), "CUDA context initialization");
        }
        return measure(
            implementation, repeats,
            [&] { multiply_on_devices(device_count, left, right, output, size); },
            [&] { return approximately_equal(expected, output); });
    } catch (const std::exception& error) {
        return unavailable(implementation, error.what());
    }
}

void print_measurement(const Measurement& measurement) {
    std::cout << "{\"implementation\":\"" << measurement.implementation << "\",";
    std::cout << "\"available\":" << (measurement.available ? "true" : "false") << ",";
    if (measurement.available) {
        std::cout << "\"latency_ms\":" << median(measurement.trial_latencies_ms) << ",";
        std::cout << "\"trial_latencies_ms\":[";
        for (std::size_t index = 0; index < measurement.trial_latencies_ms.size(); ++index) {
            std::cout << (index == 0 ? "" : ",") << measurement.trial_latencies_ms[index];
        }
        std::cout << "],";
        std::cout << "\"correct\":" << (measurement.correct ? "true" : "false") << ",";
        std::cout << "\"started_unix_ms\":" << measurement.started_unix_ms << ",";
        std::cout << "\"finished_unix_ms\":" << measurement.finished_unix_ms << ",";
        std::cout << "\"error\":null";
    } else {
        std::cout << "\"latency_ms\":null,\"trial_latencies_ms\":[],\"correct\":null,";
        std::cout << "\"started_unix_ms\":null,\"finished_unix_ms\":null,";
        std::cout << "\"error\":\"" << json_escape(measurement.error) << "\"";
    }
    std::cout << "}";
}

int parse_bounded_integer(const char* argument, int maximum, const std::string& name) {
    std::size_t parsed_characters = 0;
    const std::string text(argument);
    const int value = std::stoi(text, &parsed_characters);
    if (parsed_characters != text.size() || value < 1 || value > maximum) {
        throw std::invalid_argument(
            name + " must be an integer from 1 to " + std::to_string(maximum));
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: benchmark_runner <matrix_size> [repeats]\n";
        return EXIT_FAILURE;
    }

    try {
        const int size = parse_bounded_integer(argv[1], kMaxMatrixSize, "matrix size");
        const int repeats =
            argc == 3 ? parse_bounded_integer(argv[2], kMaxRepeats, "repeats")
                      : kDefaultRepeats;
        const std::size_t element_count =
            static_cast<std::size_t>(size) * static_cast<std::size_t>(size);

        std::vector<float> left(element_count);
        std::vector<float> right(element_count);
        std::vector<float> cpu_output(element_count);
        std::vector<float> openmp_output(element_count);
        std::vector<float> cuda_output(element_count);
        std::vector<float> multi_gpu_output(element_count);

        for (std::size_t index = 0; index < element_count; ++index) {
            left[index] = static_cast<float>((index % 101) + 1) / 101.0F;
            right[index] = static_cast<float>((index % 97) + 1) / 97.0F;
        }

        // The sequential CPU result is the correctness reference.
        const Measurement cpu = measure(
            "cpu", repeats,
            [&] { multiply_host(left, right, cpu_output, size, false); },
            [] { return true; });

        const Measurement openmp = measure(
            "openmp", repeats,
            [&] { multiply_host(left, right, openmp_output, size, true); },
            [&] { return approximately_equal(cpu_output, openmp_output); });

        int device_count = 0;
        const cudaError_t count_status = cudaGetDeviceCount(&device_count);
        std::string no_device_reason;
        if (count_status != cudaSuccess) {
            device_count = 0;
            no_device_reason = cudaGetErrorString(count_status);
        } else if (device_count == 0) {
            no_device_reason = "No CUDA-capable GPU was detected";
        }

        const Measurement cuda =
            device_count >= 1
                ? measure_cuda("cuda", 1, repeats, left, right, cpu_output, cuda_output, size)
                : unavailable("cuda", no_device_reason);

        Measurement multi_gpu;
        if (device_count < 2) {
            multi_gpu = unavailable(
                "cuda_multi_gpu",
                "multi-GPU run requires at least two CUDA devices; found " +
                    std::to_string(device_count));
        } else if (size < device_count) {
            multi_gpu = unavailable(
                "cuda_multi_gpu",
                "matrix size must be at least the GPU count to split rows across GPUs");
        } else {
            multi_gpu = measure_cuda("cuda_multi_gpu", device_count, repeats, left, right,
                                     cpu_output, multi_gpu_output, size);
        }

        std::string gpu_name = "unavailable";
        if (device_count > 0) {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
                gpu_name = properties.name;
            }
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\"matrix_size\":" << size << ",";
        std::cout << "\"repeats\":" << repeats << ",";
        std::cout << "\"data_type\":\"float32\",";
        std::cout << "\"transfer_included\":true,";
        std::cout << "\"cuda_context_warmup_excluded\":true,";
        std::cout << "\"gpu_name\":\"" << json_escape(gpu_name) << "\",";
        std::cout << "\"gpu_count\":" << device_count << ",";
        std::cout << "\"openmp_threads\":" << openmp_team_size() << ",";
        std::cout << "\"measurements\":[";
        print_measurement(cpu);
        std::cout << ",";
        print_measurement(openmp);
        std::cout << ",";
        print_measurement(cuda);
        std::cout << ",";
        print_measurement(multi_gpu);
        std::cout << "]}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
