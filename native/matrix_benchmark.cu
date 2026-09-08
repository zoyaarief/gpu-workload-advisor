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
#include <vector>

namespace {

constexpr int kMaxMatrixSize = 4096;

struct Measurement {
    std::string implementation;
    bool available;
    double latency_ms;
    bool correct;
    std::string error;
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

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

__global__ void matrix_multiply_kernel(const float* left,
                                       const float* right,
                                       float* output,
                                       int size) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= size || column >= size) {
        return;
    }

    float value = 0.0F;
    for (int inner = 0; inner < size; ++inner) {
        value += left[row * size + inner] * right[inner * size + column];
    }
    output[row * size + column] = value;
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

Measurement run_cuda(const std::vector<float>& left,
                     const std::vector<float>& right,
                     const std::vector<float>& expected,
                     std::vector<float>& output,
                     int size) {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        const std::string reason = count_status == cudaSuccess
                                       ? "No CUDA-capable GPU was detected"
                                       : cudaGetErrorString(count_status);
        return {"cuda", false, 0.0, false, reason};
    }

    float* device_left = nullptr;
    float* device_right = nullptr;
    float* device_output = nullptr;
    const std::size_t bytes = left.size() * sizeof(float);

    try {
        // Initialize the CUDA context before timing. Allocations and both transfers
        // remain inside the timed region.
        check_cuda(cudaFree(nullptr), "CUDA context initialization");

        const double latency = elapsed_ms([&] {
            check_cuda(cudaMalloc(&device_left, bytes), "cudaMalloc(left)");
            check_cuda(cudaMalloc(&device_right, bytes), "cudaMalloc(right)");
            check_cuda(cudaMalloc(&device_output, bytes), "cudaMalloc(output)");
            check_cuda(cudaMemcpy(device_left, left.data(), bytes, cudaMemcpyHostToDevice),
                       "copy left to GPU");
            check_cuda(cudaMemcpy(device_right, right.data(), bytes, cudaMemcpyHostToDevice),
                       "copy right to GPU");

            const dim3 threads(16, 16);
            const dim3 blocks((size + threads.x - 1) / threads.x,
                              (size + threads.y - 1) / threads.y);
            matrix_multiply_kernel<<<blocks, threads>>>(
                device_left, device_right, device_output, size);
            check_cuda(cudaGetLastError(), "CUDA kernel launch");
            check_cuda(cudaDeviceSynchronize(), "CUDA kernel execution");
            check_cuda(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost),
                       "copy result to CPU");

            check_cuda(cudaFree(device_left), "cudaFree(left)");
            device_left = nullptr;
            check_cuda(cudaFree(device_right), "cudaFree(right)");
            device_right = nullptr;
            check_cuda(cudaFree(device_output), "cudaFree(output)");
            device_output = nullptr;
        });
        return {"cuda", true, latency, approximately_equal(expected, output), ""};
    } catch (const std::exception& error) {
        if (device_left != nullptr) cudaFree(device_left);
        if (device_right != nullptr) cudaFree(device_right);
        if (device_output != nullptr) cudaFree(device_output);
        return {"cuda", false, 0.0, false, error.what()};
    }
}

void print_measurement(const Measurement& measurement) {
    std::cout << "{\"implementation\":\"" << measurement.implementation << "\",";
    std::cout << "\"available\":" << (measurement.available ? "true" : "false") << ",";
    if (measurement.available) {
        std::cout << "\"latency_ms\":" << measurement.latency_ms << ",";
        std::cout << "\"correct\":" << (measurement.correct ? "true" : "false") << ",";
        std::cout << "\"error\":null";
    } else {
        std::cout << "\"latency_ms\":null,\"correct\":null,";
        std::cout << "\"error\":\"" << json_escape(measurement.error) << "\"";
    }
    std::cout << "}";
}

int parse_size(const char* argument) {
    std::size_t parsed_characters = 0;
    const std::string text(argument);
    const int size = std::stoi(text, &parsed_characters);
    if (parsed_characters != text.size() || size < 1 || size > kMaxMatrixSize) {
        throw std::invalid_argument("matrix size must be an integer from 1 to 4096");
    }
    return size;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: benchmark_runner <matrix_size>\n";
        return EXIT_FAILURE;
    }

    try {
        const int size = parse_size(argv[1]);
        const std::size_t element_count =
            static_cast<std::size_t>(size) * static_cast<std::size_t>(size);

        std::vector<float> left(element_count);
        std::vector<float> right(element_count);
        std::vector<float> cpu_output(element_count);
        std::vector<float> openmp_output(element_count);
        std::vector<float> cuda_output(element_count);

        for (std::size_t index = 0; index < element_count; ++index) {
            left[index] = static_cast<float>((index % 101) + 1) / 101.0F;
            right[index] = static_cast<float>((index % 97) + 1) / 97.0F;
        }

        const double cpu_latency = elapsed_ms(
            [&] { multiply_host(left, right, cpu_output, size, false); });
        const Measurement cpu{"cpu", true, cpu_latency, true, ""};

        const double openmp_latency = elapsed_ms(
            [&] { multiply_host(left, right, openmp_output, size, true); });
        const Measurement openmp{"openmp", true, openmp_latency,
                                 approximately_equal(cpu_output, openmp_output), ""};

        const Measurement cuda =
            run_cuda(left, right, cpu_output, cuda_output, size);

        std::string gpu_name = "unavailable";
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
                gpu_name = properties.name;
            }
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\"matrix_size\":" << size << ",";
        std::cout << "\"data_type\":\"float32\",";
        std::cout << "\"transfer_included\":true,";
        std::cout << "\"cuda_context_warmup_excluded\":true,";
        std::cout << "\"gpu_name\":\"" << json_escape(gpu_name) << "\",";
        std::cout << "\"openmp_threads\":" << omp_get_max_threads() << ",";
        std::cout << "\"measurements\":[";
        print_measurement(cpu);
        std::cout << ",";
        print_measurement(openmp);
        std::cout << ",";
        print_measurement(cuda);
        std::cout << "]}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
