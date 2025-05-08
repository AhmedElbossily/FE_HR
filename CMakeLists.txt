cmake_minimum_required(VERSION 3.11.3)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -pthread")
#set(CMAKE_BUILD_TYPE Debug)

project(mFree LANGUAGES CUDA CXX)

enable_language(CUDA)

#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})
#link_directories(${CUDA_LIBRARY_DIRS})

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED)

set(CUDA_TARGET_ARCH compute_70)
set(CUDA_ARCHITECTURES "70")

#find_package (Threads REQUIRED)
#find_package(Thrust REQUIRED CONFIG)
#thrust_create_target(Thrust)
#Set CUDA flags

file(GLOB project_SRCS src/*.cpp src/Benchmarks/*.cpp src/Benchmarks/*.cu src/*.cu)

#include_directories(/usr/local/cuda/targets/x86_64-linux/include)
#link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

include_directories(/usr/local/cuda-11.0/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-11.0/targets/x86_64-linux/lib)

add_executable(Mfree ${project_SRCS})
#target_compile_options(Mfree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)

set_property(TARGET Mfree PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(Mfree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
target_link_libraries(Mfree )
set_target_properties(Mfree PROPERTIES CUDA_ARCHITECTURES "70")
