# Advanced Driver Assistance System (ADAS) - Jetson Nano Implementation

An advanced driving assistance system based on NVIDIA Jetson Nano embedded computer with comprehensive capabilities for lane detection, traffic sign recognition, and object detection on roadways.

## Features

- **Lane Detection**: Real-time lane boundary detection and tracking
- **Traffic Sign Recognition**: Detection and classification of traffic signs
- **Object Detection**: Vehicle and pedestrian detection
- **Lane Departure Warning**: Vehicle position monitoring relative to lane markings
- **Performance Optimized**: TensorRT acceleration for real-time processing

## Development Environment and Build Requirements

### System Requirements
* CMake >= 3.10
* OpenCV with CUDA >= 4.0.1
* C++ 17 Compiler
* CUDA 10.1
* TensorRT 5
* NVIDIA Jetson Nano (recommended)

## Setup Instructions for Linux - Ubuntu 18.04/20.04

### 1. Download Required Data

#### Pre-trained Models
Download the pre-trained neural network models from [Google Drive](https://drive.google.com/drive/u/0/folders/1STGPyK7os0lNbR1ZOlT-CD54BnEs1lSg) and place them in the `models/` folder.

#### Test Data
Download test images and videos from [Google Drive](https://drive.google.com/drive/u/0/folders/1SRPS_FjzGYO56YbtnBSw4UE1ZDSfohUQ) and place them in the root folder of this project.

### 2. GPU Architecture Configuration

Update the `GPU_ARCHS` parameter in `CMakeLists.txt` to match your GPU architecture:

- **Jetson Nano**: `GPU_ARCHS = 53`
- **Other GPUs**: Check your GPU's compute capability at [NVIDIA GPU Compute Capabilities](https://developer.nvidia.com/cuda-gpus)

For detailed information on GPU architectures:
- [How to get NVIDIA graphics card GPU_ARCHS for Linux](https://www.programmersought.com/article/28125950847/)
- [Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

### 3. Compilation and Execution

#### Build the Project
```bash
cd <project_directory>
mkdir build
cd build
cmake ../
make
```

#### Run the Application
```bash
./ADAS
```

## Installation Help

If you encounter issues with the standard installation procedures, refer to these helpful resources:

### CUDA 10.1 Installation
[Installing CUDA 10.1 on Ubuntu 20.04](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0)

### OpenCV with CUDA Installation
[How to install OpenCV 4.2.0 with CUDA 10.0 in Ubuntu distro 18.04](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)

## Project Structure

```
ADAS/
├── main.cpp                 # Main application entry point
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── LICENSE                 # Apache 2.0 license
├── configs/                # Configuration files
├── common/                 # Common utilities and models
├── lane_detection/         # Lane detection implementation
├── lane_detection_NN/      # Neural network training scripts
├── models/                 # Pre-trained model files
└── input_folder/           # Test images and videos
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
