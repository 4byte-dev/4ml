cmake_minimum_required(VERSION 3.14)
project(ml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Tensor engine library
add_library(tensor_engine STATIC
    src/tensor/cpu_engine.cpp
)
target_include_directories(tensor_engine PUBLIC src)

# Linear regression library
add_library(linear_regression STATIC
    src/supervised/regression/linear_regression.cpp
)
target_include_directories(linear_regression PUBLIC src)
target_link_libraries(linear_regression PUBLIC tensor_engine)

# --- Executables ---

# Tensor benchmark
add_executable(tensor_benchmark
    src/tensor/tensor_benchmark.cpp
)
target_link_libraries(tensor_benchmark tensor_engine)

# Linear regression example
add_executable(linear_regression_example
    src/supervised/regression/linear_regression_example.cpp
)
target_link_libraries(linear_regression_example linear_regression)

# Linear regression benchmark
add_executable(linear_regression_benchmark
    src/supervised/regression/linear_regression_benchmark.cpp
)
target_link_libraries(linear_regression_benchmark linear_regression)
