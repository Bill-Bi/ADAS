#include "file_storage.h"

using namespace ml_cam;

// Constructor - initialize storage paths
FileStorage::FileStorage() {
    DATA_FOLDER = fs::path(getHomePath()) / DATA_FOLDER_NAME;
    PHOTO_FOLDER = DATA_FOLDER / PHOTO_FOLDER_NAME;
    VIDEO_FOLDER = DATA_FOLDER / VIDEO_FOLDER_NAME;
}

// Destructor
FileStorage::~FileStorage() {}

// Initialize storage directory structure
void FileStorage::initStorage() {
    std::cout << "Data Folder: " << getDataPath() << std::endl;
    std::cout << "Photos Folder: " << getPhotoPath() << std::endl;
    std::cout << "Videos Folder: " << getVideoPath() << std::endl;

    // ============================================================================
    // Create Photos Directory
    // ============================================================================
    if (fs::exists(getPhotoPath()) && !fs::is_directory(getPhotoPath())) {
        std::cerr << "Photos folder path is not a directory: " << getPhotoPath()
                  << std::endl;
        exit(-1);
    } else if (!fs::exists(getPhotoPath())) {
        // Create photos directory if it doesn't exist
        fs::create_directories(getPhotoPath());

        if (!fs::exists(getPhotoPath())) {
            std::cerr << "Could not create directory: " << getPhotoPath()
                      << std::endl;
            exit(-1);
        }
    }

    // ============================================================================
    // Create Videos Directory
    // ============================================================================
    if (fs::exists(getVideoPath()) && !fs::is_directory(getVideoPath())) {
        std::cerr << "Videos folder path is not a directory: " << getVideoPath()
                  << std::endl;
        exit(-1);
    } else if (!fs::exists(getVideoPath())) {
        // Create videos directory if it doesn't exist
        fs::create_directories(getVideoPath());

        if (!fs::exists(getVideoPath())) {
            std::cerr << "Could not create directory: " << getVideoPath()
                      << std::endl;
            exit(-1);
        }
    }
}

// Get data folder path
fs::path FileStorage::getDataPath() { 
    return DATA_FOLDER; 
}

// Get photos folder path
fs::path FileStorage::getPhotoPath() { 
    return PHOTO_FOLDER; 
}

// Get videos folder path
fs::path FileStorage::getVideoPath() { 
    return VIDEO_FOLDER; 
}

// Save image with timestamped filename
bool FileStorage::saveImage(const cv::Mat& image) {
    // ============================================================================
    // Generate Unique Timestamped Filename
    // ============================================================================
    // Format: YYYY-MM-DD-HH-MM-SS.milliseconds.png
    std::stringstream filename;

    auto current_time = std::chrono::system_clock::now();
    auto duration = current_time.time_since_epoch();
    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    time_t time_timestamp = std::chrono::system_clock::to_time_t(current_time);
    tm local_time = *localtime(&time_timestamp);

    // Format timestamp components with zero-padding
    filename << std::setw(4) << std::setfill('0') << local_time.tm_year + 1900
    << "-" << std::setw(2) << std::setfill('0') << local_time.tm_mon + 1 
    << "-" << std::setw(2) << std::setfill('0') << local_time.tm_mday
    << "-" << std::setw(2) << std::setfill('0') << local_time.tm_hour
    << "-" << std::setw(2) << std::setfill('0') << local_time.tm_min 
    << "-" << std::setw(2) << std::setfill('0') << local_time.tm_sec 
    << "." << milliseconds << ".png";

    fs::path file_path = getPhotoPath() / fs::path(filename.str());
    setLastSavedItem(fs::path(filename.str()));

    // ============================================================================
    // Save Image to File
    // ============================================================================
    cv::imwrite(file_path.string(), image);

    return true;
}

// Get the last saved file path
fs::path FileStorage::getLastSavedItem() {
    return last_saved_file;
}

// Set the last saved file path
void FileStorage::setLastSavedItem(fs::path path) {
    last_saved_file = path;
}