// ============================================================================
// Camera Device Configuration Header
// ============================================================================
// Defines camera structures and configurations for V4L (Video4Linux) devices

#if !defined(CAMERA_H)
#define CAMERA_H

#include <string>

// Camera structure for V4L device configuration
struct Camera { 
   int v4l_id;                    // V4L device identifier
   std::string identifier;        // Human-readable camera identifier
   
   // Constructor for camera initialization
   Camera(int v4l_id, std::string identifier) :
    v4l_id(v4l_id), identifier(identifier)
    {}
};

#endif // CAMERA_H