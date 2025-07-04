# Models Directory

This directory contains pre-trained neural network models for the Advanced Driver Assistance System (ADAS).

## Model Files

### Lane Detection Models
- `lane_segmentation_384x384.uff` - UFF format lane segmentation model (384x384 input)
- `lane_segmentation_384x384.engine` - TensorRT optimized engine file for lane detection

### Traffic Sign Recognition Models
- `traffic_sign_classifier.onnx` - ONNX format traffic sign classification model
- `traffic_sign_classifier.engine` - TensorRT optimized engine for traffic sign detection

### Object Detection Models
- `vehicle_detector.uff` - UFF format vehicle detection model
- `pedestrian_detector.uff` - UFF format pedestrian detection model

## Model Specifications

All models are optimized for Jetson Nano and support:
- FP16 precision for improved performance
- Batch size of 1 for real-time inference
- CUDA acceleration via TensorRT

## Usage

Models should be placed in this directory before running the ADAS application. The system will automatically load the appropriate models based on configuration settings.

## Model Sources

These models are trained using the training scripts in the `lane_detection_NN/` directory and converted to TensorRT format for optimal performance on embedded systems. 