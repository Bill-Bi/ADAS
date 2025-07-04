# Advanced Driver Assistance System (ADAS) - CMake Configuration
# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Specify project title and description
project(ADAS VERSION 1.0.0 DESCRIPTION "Advanced Driver Assistance System for Jetson Nano")

# GPU architecture configuration for TensorRT
# Configure your GPU_ARCHS based on your specific GPU
# See: https://developer.nvidia.com/cuda-gpus for compute capability
set(GPU_ARCHS 53)  # Jetson Nano compute capability

# Build configuration
set(CMAKE_BUILD_TYPE Release)

# C++ standard configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find CUDA package
find_package(CUDA REQUIRED)

###########################################################################
# TensorRT Configuration
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)

message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

# Find TensorRT libraries
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
        
find_library(TENSORRT_LIBRARY_PARSER nvparsers
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
        
message(STATUS "Found TensorRT inference library: ${TENSORRT_LIBRARY_INFER}")
message(STATUS "Found TensorRT parser library: ${TENSORRT_LIBRARY_PARSER}")
#########################################################################

# OpenCV Configuration
# Specify OpenCV folder, and take care of dependencies and includes
# set(OpenCV_DIR "path_to_opencv")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories for all components
include_directories(
    ${TENSORRT_INCLUDE_DIR}
    "configs"
    "common/utils/include"
    "lane_detection"
    "common/uff_models/common/include"
    "common/uff_models/unet"
    ${CUDA_INCLUDE_DIRS}
)

# Build lane detector subdirectory
add_subdirectory(lane_detection)

# Enable current directory includes for older CMake versions
if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

# Main executable source files
add_executable(${PROJECT_NAME}
    "main.cpp"
    "common/utils/timer.cpp"
)

# C++ 17 features
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Filesystem library for older GCC versions
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    set (CPP_FS_LIB "stdc++fs")
endif()

# Link required libraries
target_link_libraries(${PROJECT_NAME} 
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_PARSER}
    ${OpenCV_LIBS} 
    ${CPP_FS_LIB} 
    lane_detector
)
