// ============================================================================
// File Storage Management Header
// ============================================================================
// Provides functionality for organizing and managing data files, photos, and videos

#if !defined(FILE_STORAGE_H)
#define FILE_STORAGE_H

#include <iostream>
#include "filesystem_include.h"
#include "utility.h"
#include <ctime>
#include <sstream>
#include <chrono>
#include <cstdlib>

namespace ml_cam {

class FileStorage {
   private:
    // Directory structure configuration
    const fs::path DATA_FOLDER_NAME = "CarSmartCam";
    const fs::path PHOTO_FOLDER_NAME = "Photos";
    const fs::path VIDEO_FOLDER_NAME = "Videos";
    
    // Full paths to storage directories
    fs::path DATA_FOLDER;
    fs::path PHOTO_FOLDER;
    fs::path VIDEO_FOLDER;

    // Track the last saved file for reference
    fs::path last_saved_file;

   public:
    // Constructor and destructor
    FileStorage();
    ~FileStorage();

    // Directory path getters
    fs::path getDataPath();
    fs::path getPhotoPath();
    fs::path getVideoPath();
    
    // Initialize storage directory structure
    void initStorage();

    // File tracking methods
    fs::path getLastSavedItem();
    void setLastSavedItem(fs::path);

    // Save image to photo directory with timestamp
    bool saveImage(const cv::Mat & img);
};

}  // namespace ml_cam

#endif // FILE_STORAGE_H
