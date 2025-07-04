# Input Folder

This directory contains test images and video files used for testing the Advanced Driver Assistance System (ADAS).

## Contents

### Test Images
The folder contains various road scene images in JPG format that are used for:
- Lane detection testing
- Traffic sign recognition testing
- Object detection validation
- System performance evaluation

### Image Specifications
- **Format**: JPG
- **Resolution**: Various (typically 1920x1080 or higher)
- **Content**: Road scenes with lane markings, traffic signs, and vehicles
- **Usage**: Test images for the lane detection and object recognition algorithms

### Video Files
- `lane_line.mp4` - Test video for lane detection demonstration
- Additional video files may be added for comprehensive testing

## Usage

These files are automatically loaded by the ADAS application when running in test mode. The system processes these images to validate the performance of:

1. **Lane Detection**: Identifying and tracking lane boundaries
2. **Traffic Sign Recognition**: Detecting and classifying traffic signs
3. **Object Detection**: Identifying vehicles, pedestrians, and other objects
4. **Lane Departure Warning**: Monitoring vehicle position relative to lane markings

## File Naming Convention

Images follow a unique naming pattern with alphanumeric identifiers to ensure proper organization and avoid conflicts during testing.

## Adding New Test Data

To add new test images:
1. Place JPG files in this directory
2. Ensure images contain relevant road scenes
3. Use descriptive filenames if possible
4. Test with the ADAS application to verify compatibility

## Performance Testing

These images are used for:
- Accuracy validation
- Performance benchmarking
- Algorithm optimization
- System reliability testing 