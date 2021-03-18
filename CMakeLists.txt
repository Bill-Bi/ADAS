# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Specify project title
project(ADAS)

# Setup for TensorRT
set(GPU_ARCHS 53)  ## config your GPU_ARCHS,See [here](https://developer.nvidia.com/cuda-gpus) for finding what maximum compute capability your specific GPU supports.

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)

###########################################################################
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        # HINTS  $ENV{TRT_RELEASE}
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)

       
        
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
        # HINTS  $ENV{TRT_RELEASE}
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
        
find_library(TENSORRT_LIBRARY_PARSER nvparsers
        # HINTS  $ENV{TRT_RELEASE}
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
        
message(STATUS "Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")
#########################################################################

# Specify OpenCV folder, and take care of dependencies and includes
#set(OpenCV_DIR "path_to_opencv")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(
    ${TENSORRT_INCLUDE_DIR}
    "configs"
    "common/utils/include"

    "lane_detection"
    "common/uff_models/common/include"
    "common/uff_models/unet"
    
    ${CUDA_INCLUDE_DIRS}
)

# Build lane detector
add_subdirectory(lane_detection)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

# add required source, header, ui and resource files
add_executable(${PROJECT_NAME}

    "main.cpp"

    "common/utils/timer.cpp"
)


# Use C++ 17
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    set (CPP_FS_LIB "stdc++fs")
endif()

# Link required libs
target_link_libraries(${PROJECT_NAME} 
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_PARSER}
    ${OpenCV_LIBS} 
    ${CPP_FS_LIB} 
    lane_detector
)
