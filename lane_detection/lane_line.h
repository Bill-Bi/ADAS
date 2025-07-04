#ifndef LANE_LINE_H
#define LANE_LINE_H

#include <opencv2/opencv.hpp>

enum lane_line_category { LeftLaneLine, RightLaneLine, OtherLaneLine };

struct LaneLine {
    cv::Vec4i line;
    lane_line_category type;
    LaneLine(cv::Vec4i line, lane_line_category type) : line(line), type(type) {}
};

#endif