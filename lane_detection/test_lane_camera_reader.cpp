#include <fstream>
#include <opencv2/opencv.hpp>

#include "filesystem_include.h"
#include "lane_detector.h"

using namespace std;
using namespace cv;

std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main(int argc, char** argv) {


    int capture_width = 1280 ;
    int capture_height = 720 ;
    int display_width = 1280 ;
    int display_height = 720 ;
    int framerate = 60 ;
    int flip_method = 0 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    std::string output_folder = "output_folder";
    std::string videoFilePath ="../lane_line.mp4";
    fs::create_directory(output_folder);

    LaneDetector lane_detector;
    cv::Mat lane_line_mask;
    cv::Mat detected_line_img;
    cv::Mat reduced_line_img;
    cv::Mat dst;
    bool lane_departure;
    using namespace std::chrono;
    double total_time = 0;
    size_t n_images = 0;
    double alpha = 0.5;
    double beta = 1.0 - alpha;
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }

    for (int frameNum = 0; frameNum < cap.get(CAP_PROP_FRAME_COUNT); frameNum++) {
        n_images += 1;
        cv::Mat input_img;
        cap.read(img);
        auto start = high_resolution_clock::now();

        //cv::Mat output_mask = lane_detector.getLaneMask(input_img);
        std::vector<LaneLine> detected_lines = lane_detector.detectLaneLines(input_img, lane_line_mask, detected_line_img, reduced_line_img, lane_departure);

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        long long int execution_time = duration.count();
        double milliseconds = 0.001 * execution_time;
        total_time += milliseconds;

        // cv::Mat output_img = cv::Mat::zeros(input_img.size(), CV_8UC1);
        // output_img.setTo(Scalar(255), output_mask > 0.5);
        cv::addWeighted(input_img, alpha, reduced_line_img, beta, 0.0, dst);

        // std::string output_img_path = (fs::path(output_folder) / to_string(frameNum)).string();
        std::string output_img_path = output_folder + to_string(static_cast<long long>(frameNum))+ ".jpg";
        // cv::imwrite(output_img_path, output_img);
        cv::imwrite(output_img_path, dst);
    }
    cout << "Avg. time: " << total_time / n_images << " ms" << endl;

    return 0;
}