// ============================================================================
// Camera Model Utilities Header
// ============================================================================
// Provides utility structures and functions for camera calibration and transformation

#if !defined(UTILS_H)
#define UTILS_H

#include <opencv2/opencv.hpp>

// Four corner points structure for perspective transformation
// Used for camera calibration and bird's eye view transformation
class FourPoints {
   public:
    cv::Point2f top_left;      // Top-left corner point
    cv::Point2f top_right;     // Top-right corner point
    cv::Point2f bottom_right;  // Bottom-right corner point
    cv::Point2f bottom_left;   // Bottom-left corner point

    // Default constructor
    FourPoints() {}

    // Constructor with all four corner points
    FourPoints(cv::Point2f tl, cv::Point2f tr, cv::Point2f br, cv::Point2f bl)
        : top_left(tl), top_right(tr), bottom_right(br), bottom_left(bl) {}

    // Convert to vector of points (clockwise order)
    std::vector<cv::Point2f> to_vector() {
        return std::vector<cv::Point2f>({top_left, top_right, bottom_right, bottom_left});
    }

    // Convert to vector of points with scaling applied
    std::vector<cv::Point2f> to_vector(float scale_x, float scale_y) {
        return std::vector<cv::Point2f>({
            cv::Point2f(top_left.x * scale_x, top_left.y * scale_y), 
            cv::Point2f(top_right.x * scale_x, top_right.y * scale_y), 
            cv::Point2f(bottom_right.x * scale_x, bottom_right.y * scale_y),  
            cv::Point2f(bottom_left.x * scale_x, bottom_left.y * scale_y)});
    }
};

#endif  // UTILS_H
