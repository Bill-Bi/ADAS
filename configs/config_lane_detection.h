#if !defined(CONFIG_LANE_DETECTION_H)
#define CONFIG_LANE_DETECTION_H

// ============================================================================
// Lane Detection Configuration Parameters
// ============================================================================

// Neural network model configuration
#define LANE_DETECTION_BATCH_SIZE 1
#define LANE_DETECTION_N_CLASSES 1
#define LANE_DETECTION_FORCE_REBUILD_ENGINE false

// Model file paths
#define LANE_DETECTION_MODEL \
    "../models/lane_segmentation_384x384.uff"
#define LANE_DETECTION_TENSORRT_PLAN \
    "../models/lane_segmentation_384x384.engine"

// Model optimization settings
#define LANE_DETECTION_USE_FP_16 true

// Input dimensions for the neural network
#define LANE_DETECTION_INPUT_WIDTH 384
#define LANE_DETECTION_INPUT_HEIGHT 384

// TensorRT node names for model inference
#define LANE_DETECTION_INPUT_NODE "data"
#define LANE_DETECTION_OUTPUT_NODE "sigmoid/Sigmoid"

#endif // CONFIG_LANE_DETECTION_H
