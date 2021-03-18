#include <fstream>
#include <opencv2/opencv.hpp>

#include "filesystem_include.h"
#include "lane_detector.h"

using namespace std;
using namespace cv;


static cv::Mat input_img;
static cv::Mat lane_line_mask;
static cv::Mat detected_line_img;
static cv::Mat reduced_line_img;
static cv::Mat dst;
static bool lane_departure;
static bool lane_detected = false;

void laneDetectionThread(){
    LaneDetector lane_detector;
    while(true) {
        std::vector<LaneLine> detected_lines = lane_detector.detectLaneLines(input_img, lane_line_mask, detected_line_img, reduced_line_img, lane_departure);
        lane_detected = true;
    }
}

int main(int argc, char** argv) { 
    std::thread lane_detection_thread(laneDetectionThread);
    lane_detection_thread.detach();
    std::string input_folder = "../input_folder";
    std::string output_folder = "output_folder";
    std::string videoFilePath ="../lane_line.mp4";
    cv::namedWindow("CSI Camera", cv::WINDOW_AUTOSIZE);
    double alpha = 0.8;
    double beta = 1.0 - alpha;
    VideoCapture cap(videoFilePath);
    cv::Mat update_line_img;
    while(true){
        cap >> input_img;
        if (lane_detected){            
            cv::addWeighted(input_img, alpha, reduced_line_img, beta, 0.0, dst);
            cv::imshow("CSI Camera",dst);
        }else{
            cv::imshow("CSI Camera",input_img);
        }
        
        int keycode = cv::waitKey(30) & 0xff ; 
        if (keycode == 27) break ;
    }
    return 0;
}