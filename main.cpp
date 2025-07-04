#include <fstream>
#include <opencv2/opencv.hpp>

#include "filesystem_include.h"
#include "lane_detector.h"

using namespace std;
using namespace cv;

// Global variables for lane detection processing
static cv::Mat input_image;
static cv::Mat lane_line_mask;
static cv::Mat detected_line_image;
static cv::Mat reduced_line_image;
static cv::Mat output_image;
static bool lane_departure_warning;
static bool lane_detection_complete = false;

void laneDetectionThread() {
    LaneDetector lane_detector;
    while (true) {
        std::vector<LaneLine> detected_lines = lane_detector.detectLaneLines(
            input_image, lane_line_mask, detected_line_image, 
            reduced_line_image, lane_departure_warning);
        lane_detection_complete = true;
    }
}

int main(int argc, char** argv) { 
    // Start lane detection in separate thread
    std::thread lane_detection_thread(laneDetectionThread);
    lane_detection_thread.detach();
    
    // Configuration paths
    std::string input_folder_path = "../input_folder";
    std::string output_folder_path = "output_folder";
    std::string video_file_path = "../lane_line.mp4";
    
    // Display window setup
    cv::namedWindow("ADAS Camera View", cv::WINDOW_AUTOSIZE);
    
    // Image blending parameters
    double alpha = 0.6;
    double beta = 1.0 - alpha;
    
    VideoCapture video_capture(video_file_path);
    cv::Mat processed_line_image;
    
    // Display dimensions
    int display_height = 500;
    int display_width = 900;
    
    while (true) {
        video_capture >> input_image;
        
        if (lane_detection_complete) {            
            cv::addWeighted(input_image, alpha, reduced_line_image, beta, 0.0, output_image);
            // cv::resize(output_image, output_image, cv::Size(display_width, display_height));
            // cv::imshow("ADAS Camera View", output_image);
        } else {
            output_image = input_image;
            // cv::imshow("ADAS Camera View", input_image);
        }
        
        cv::resize(output_image, output_image, cv::Size(display_width, display_height));
        cv::imshow("ADAS Camera View", output_image);
        
        int key_code = cv::waitKey(30) & 0xff; 
        if (key_code == 27) break; // ESC key to exit
    }
    return 0;
}