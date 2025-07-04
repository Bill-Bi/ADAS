// ============================================================================
// Utility Functions Header
// ============================================================================
// Provides common utility functions for image processing and system operations

#if !defined(UTILITY_H)
#define UTILITY_H

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <QImage>

namespace ml_cam {

    // Draw text label on image with background rectangle
    void setLabel(cv::Mat& im, const std::string label, const cv::Point & origin);
    
    // Get user's home directory path
    std::string getHomePath();

    // Convert OpenCV Mat to Qt QImage
    QImage Mat2QImage(cv::Mat const& src);
    
    // Convert Qt QImage to OpenCV Mat
    cv::Mat QImage2Mat(QImage const& src);

    // Overlay one image onto another at specified position
    void placeOverlay(cv::Mat &image, const cv::Mat &overlay, int x, int y);

}

#endif // UTILITY_H
