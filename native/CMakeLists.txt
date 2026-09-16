cmake_minimum_required(VERSION 3.24)

project(gpu_matrix_benchmark LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_executable(benchmark_runner matrix_benchmark.cu)
target_compile_features(benchmark_runner PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(benchmark_runner PRIVATE OpenMP::OpenMP_CXX Threads::Threads)
# OpenMP::OpenMP_CXX only adds its compile flag to CXX sources. This file is a
# CUDA source, so nvcc must forward the flag to the host compiler explicitly;
# without it the OpenMP pragmas are silently ignored.
target_compile_options(benchmark_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
)

set_target_properties(benchmark_runner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
