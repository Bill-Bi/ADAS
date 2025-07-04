# Advanced Driver Assistance System using Jetson Nano
Advanced Driving Assistance System based on Jetson Nano embedded computer with the capability to detect lane lines on the road. This repository contains source code for Jetson Nano, and the source code for the lane detection neural network training.

## Development Environment And Build

### Requirements:
* CMake >= 3.10
* OpenCV with CUDA >= 4.0.1
* C++ 17 Compiler
* CUDA 10.1
* TensorRT 5

## Setup for Linux - Ubuntu 18.04/20.04

### Models and test data
* Download models [here](https://drive.google.com/drive/u/0/folders/1STGPyK7os0lNbR1ZOlT-CD54BnEs1lSg) and place into the models folder
* Download data [here](https://drive.google.com/drive/u/0/folders/1SRPS_FjzGYO56YbtnBSw4UE1ZDSfohUQ) and place into the root folder of this project

### Compile and Run
* Update `GPU_ARCHS`: Modify `GPU_ARCHS` in `CMakeLists.txt` to match your GPU. For Jetson Nano, `GPU_ARCHS = 53`. Read more in following posts:
[How to get NVIDIA graphics card GPU_ARCHS for Linux](https://www.programmersought.com/article/28125950847/)
[Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
* Compile
`cd <project directory>`
`mkdir build`
`cd build`
`cmake ../`
`make`
* Run
`./ADAS`

### Helpful websites for requirements installations (If original installations do not work)
* Install CUDA 10.1:
[Installing CUDA 10.1 on Ubuntu 20.04](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0)

* Install OpenCV with CUDA
[How to install OpenCV 4.2.0 with CUDA 10.0 in Ubuntu distro 18.04](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
