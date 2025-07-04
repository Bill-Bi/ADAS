// ============================================================================
// Camera Model Header
// ============================================================================
// Provides camera calibration and transformation functionality for ADAS system

#ifndef CAMERA_MODEL_H
#define CAMERA_MODEL_H

#include <bits/stdc++.h>

// Qt includes for GUI components
#include <QCloseEvent>
#include <QDebug>
#include <QFileDialog>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QImage>
#include <QListWidgetItem>
#include <QMainWindow>
#include <QMessageBox>
#include <QPixmap>
#include <QShortcut>
#include <QWidget>

// Standard library includes
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <thread>

// Project includes
#include "camera_wizard.h"
#include "car_status.h"
#include "config.h"
#include "filesystem_include.h"
#include "birdview_model.h"
#include "utils.h"

// Camera model class for handling camera calibration and transformation
class CameraModel : public QObject {
    Q_OBJECT
    
    // Shared car status for real-time vehicle data
    std::shared_ptr<CarStatus> car_status;
    
    // Camera calibration wizard for interactive setup
    std::unique_ptr<CameraWizard> camera_wizard;
    
    // Bird's eye view transformation model
    BirdViewModel birdview_model;

   public:
    // Constructor with car status dependency
    explicit CameraModel(std::shared_ptr<CarStatus> car_status);
    
    // Show camera calibration wizard interface
    void showCameraWizard();
    
    // Read camera calibration file
    void readCalibFile(std::string file_path);
    
    // Get bird's eye view model for transformations
    BirdViewModel *getBirdViewModel();

   public slots:
    // Update camera model with new calibration parameters
    void updateCameraModel(
        float car_width, float carpet_width, 
        float car_to_carpet_distance, float carpet_length,
        float top_left_x, float top_left_y,
        float top_right_x, float top_right_y,
        float bottom_right_x, float bottom_right_y,
        float bottom_left_x, float bottom_left_y
    );
};

#endif // CAMERA_MODEL_H
