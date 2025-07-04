#include "utility.h"

namespace ml_cam {

// ============================================================================
// Cross-platform Environment Variable Functions
// ============================================================================

// Define Microsoft functions and data types for cross-platform compatibility
#if !defined(_WIN32)
typedef int errno_t;

// Cross-platform implementation of _dupenv_s for non-Windows systems
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/dupenv-s-wdupenv-s?view=vs-2017
error_t _dupenv_s(char **pValue, size_t *len, const char *pPath) {
    char *value;
    value = getenv(pPath);

    if (value != NULL) {
        *len = strlen(value);

        char *buffer = (char *)malloc((*len + 1) * sizeof(char));
        // Check memory allocation
        if (buffer == NULL) {  // Not enough memory
            return ENOMEM;
        }

        strncpy(buffer, value, *len);
        buffer[*len] = '\0';
        *pValue = buffer;
    } else {
        return EINVAL;
    }

    return 0;
}
#endif  // _WIN32

// ============================================================================
// Image Processing Functions
// ============================================================================

// Draw text label on image with green background rectangle
void setLabel(cv::Mat &image, const std::string label, const cv::Point &origin) {
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    cv::Size text_size =
        cv::getTextSize(label, font_face, scale, thickness, &baseline);
    cv::rectangle(image, origin + cv::Point(0, baseline),
                  origin + cv::Point(text_size.width, -text_size.height),
                  cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(image, label, origin, font_face, scale, cv::Scalar(0, 0, 0),
                thickness, 8);
}

// ============================================================================
// System Path Functions
// ============================================================================

// Get user's home directory path across different platforms
std::string getHomePath() {
    char *pValue;
    char *pValue2;
    size_t len;
    errno_t err = _dupenv_s(&pValue, &len, "HOME");
    errno_t err2;

    if (!err && pValue != NULL) {
        std::string home_path = pValue;
        free(pValue);
        return home_path;
    } else {
        err = _dupenv_s(&pValue, &len, "USERPROFILE");

        if (!err && pValue != NULL) {
            std::string user_profile = pValue;
            free(pValue);
            return user_profile;
        } else {
            err = _dupenv_s(&pValue, &len, "HOMEDRIVE");
            err2 = _dupenv_s(&pValue2, &len, "HOMEPATH");

            if (!err && !err2 && pValue != NULL && pValue2 != NULL) {
                std::string home_path = std::string(pValue) + pValue2;
                return home_path;
            }

            free(pValue);
            free(pValue2);
        }
    }

    return "";
}

// ============================================================================
// OpenCV-Qt Conversion Functions
// ============================================================================

// Convert OpenCV Mat to Qt QImage
QImage Mat2QImage(cv::Mat const &src) {
    cv::Mat temp;  // Create temporary Mat for color conversion
    cvtColor(src, temp, cv::COLOR_BGR2RGB);  // Convert BGR to RGB
    QImage dest((const uchar *)temp.data, static_cast<int>(temp.cols),
                static_cast<int>(temp.rows), static_cast<int>(temp.step),
                QImage::Format_RGB888);
    dest.bits();  // Enforce deep copy for proper memory management
    return dest;
}

// Convert Qt QImage to OpenCV Mat
cv::Mat QImage2Mat(QImage const &src) {
    cv::Mat temp(src.height(), src.width(), CV_8UC3, (uchar *)src.bits(),
                src.bytesPerLine());
    cv::Mat result;  // Deep copy for proper memory management
    cvtColor(temp, result, cv::COLOR_BGR2RGB);
    return result;
}

// ============================================================================
// Image Overlay Functions
// ============================================================================

// Overlay one image onto another at specified position
void placeOverlay(cv::Mat &image, const cv::Mat &overlay, int x, int y) {
    // Validate overlay dimensions fit within target image
    assert(x + overlay.cols < image.cols);
    assert(y + overlay.rows < image.rows);

    cv::Mat destination_roi;
    destination_roi = image(cv::Rect(x, y, overlay.cols, overlay.rows));
    overlay.copyTo(destination_roi);
}

}  // namespace ml_cam