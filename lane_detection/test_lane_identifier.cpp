#include <fstream>
#include <opencv2/opencv.hpp>

#include "filesystem_include.h"
#include "lane_detector.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    std::string input_folder = "../input_folder";
    std::string output_folder = "output_folder";
    fs::create_directory(output_folder);

    cout << "Input folder: " << input_folder << endl;

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
    for (auto& p : fs::recursive_directory_iterator(input_folder)) {
        if (p.path().extension() == ".png" || p.path().extension() == ".jpg") {
            n_images += 1;
            std::string img_path = p.path().string();
            cout << n_images << " " << img_path << endl;
            cv::Mat input_img = cv::imread(img_path);
            auto start = high_resolution_clock::now();

            //cv::Mat output_mask = lane_detector.getLaneMask(input_img);
             std::vector<LaneLine> detected_lines = lane_detector.detectLaneLines(input_img, lane_line_mask, detected_line_img, reduced_line_img, lane_departure);
            //std::vector<LaneLine> detected_lines = lane_detector.detectLaneLines(input_img, lane_departure);

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            long long int execution_time = duration.count();
            double milliseconds = 0.001 * execution_time;
            total_time += milliseconds;

            // cv::Mat output_img = cv::Mat::zeros(input_img.size(), CV_8UC1);
            // output_img.setTo(Scalar(255), output_mask > 0.5);
            cv::addWeighted(input_img, alpha, reduced_line_img, beta, 0.0, dst);

            std::string output_img_path = (fs::path(output_folder) / p.path().filename()).string();
            // cv::imwrite(output_img_path, output_img);
            cv::imwrite(output_img_path, dst);
        }
    }
    cout << "Average time: " << total_time / n_images << " ms" << endl;

    return 0;
}