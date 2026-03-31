cmake_minimum_required(VERSION 3.14)
project(ml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_library(tensor_engine STATIC
    src/tensor/cpu_engine.cpp
)
target_include_directories(tensor_engine PUBLIC src)

add_library(frame STATIC
    src/frame/frame.h
)
target_include_directories(frame PUBLIC src)
target_link_libraries(frame PUBLIC tensor_engine)

add_executable(tensor_benchmark
    src/tensor/tensor_benchmark.cpp
)
target_link_libraries(tensor_benchmark tensor_engine)

add_executable(frame_example
    src/frame/frame_example.cpp
)
target_link_libraries(frame_example frame)

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tensor_test src/tensor/tensor_test.cpp)
target_link_libraries(tensor_test tensor_engine GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(tensor_test)

add_executable(frame_test src/frame/frame_test.cpp)
target_link_libraries(frame_test frame GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(frame_test)
