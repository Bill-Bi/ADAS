#ifndef CONFIG_H
#define CONFIG_H

#include "config_lane_detection.h"

// ============================================================================
// ADAS System Configuration
// ============================================================================

// Image processing configuration
#define IMG_MAX_SIZE 384

// Feature enable/disable flags
// #define DISABLE_LANE_DETECTOR
#define DISABLE_GPS_READER

// Debug and development settings
#define SMARTCAM_DEBUG

// Display and visualization options
// #define DEBUG_LANE_DETECTOR_SHOW_LINES
// #define DEBUG_LANE_DETECTOR_SHOW_LINE_MASK
// #define DEBUG_SHOW_FPS
#define SHOW_DISTANCES true
#define USE_CAN_BUS_FOR_SIMULATION_DATA true

// ============================================================================
// Traffic Sign Recognition Configuration
// ============================================================================

// Traffic sign validity and notification timing (in milliseconds)
#define MAX_SPEED_SIGN_VALID_TIME 30*60*1000        // 30 minutes
#define TIME_TO_RENOTIFY_A_SAME_TRAFFIC_SIGN 60*1000 // 1 minute
#define OVERSPEED_WARNING_AFTER_TRAFFIC_SIGN 3 * 1000 // 3 seconds
#define OVERSPEED_WARNING_INTERVAL 15 * 1000         // 15 seconds
#define SPEED_LIMIT_VALID_TIME 15 * 60 * 1000        // 15 minutes

// ============================================================================
// Collision Warning Configuration
// ============================================================================

// Collision detection parameters
#define MIN_SPEED_FOR_COLLISION_WARNING 25           // km/h
#define COLLISION_WARNING_INTERVAL 0.5 * 1000        // 500 milliseconds

// ============================================================================
// Lane Departure Warning Configuration
// ============================================================================

// Lane departure detection parameters
#define MIN_SPEED_FOR_LANE_DEPARTURE_WARNING 20      // km/h
#define LANE_DEPARTURE_WARNING_INTERVAL 1 * 1000     // 1 second

// ============================================================================
// File Path Configuration
// ============================================================================

// Data file paths
#define SMARTCAM_SIMULATION_LIST "data/sim_list.txt"
#define SMARTCAM_CAMERA_CALIB_FILE "data/camera_calib.txt"

#endif // CONFIG_H
