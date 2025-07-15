# Hand Detection Pipeline Module

The source code can be found in src/


## Module Structure

```
pipeline/
├── __init__.py          # Package initialization and imports
├── config.py            # Configuration classes and enums
├── image_ops.py         # Image loading, enhancement, and rotation operations
├── landmark_ops.py      # ROI, bbox, and cropping operations
├── processing.py        # Hand detection and processing logic
├── pipeline.py          # Main pipeline class and execution logic
└── README.md           # This documentation
```

## Components

### 1. `config.py`
- **ShiftDirection**: Enum for camera-specific shift directions
- **CameraShiftConfig**: Configuration for camera-specific bbox shifts
- **PipelineConfig**: Main configuration dataclass with all pipeline parameters

### 2. `image_ops.py` - ImageOperations Class
- `load_and_undistort_image()`: Load and undistort images using camera parameters
- `detect_hands_with_enhancement()`: Hand detection with brightness/contrast enhancement
- `try_rotation_angles()`: Try different image rotations for detection
- `transform_landmarks_back()`: Transform landmarks from rotated back to original coordinates

### 3. `landmark_ops.py` - LandmarkOperations Class
- `get_roi_points()`: Get ROI points around hand landmarks
- `compute_shifted_bbox()`: Compute camera-specific shifted bounding boxes
- `crop_to_bbox()`: Crop image to bounding box
- `crop_fixed_size()`: Crop fixed size region around wrist joint
- `draw_bbox()`: Draw bounding boxes on images
- `draw_landmarks_and_bbox()`: Draw landmarks and bounding boxes

### 4. `processing.py` - HandProcessor Class
- `process_single_hand()`: Process images with single hand detection
- `process_double_hands()`: Process images with two hands detected
- `detect_second_hand_or_retry()`: Retry mechanism for second hand detection
- `check_diff_hand()`: Check if detected hands are actually different
- `comp_shift_diff()`: Compute shift differences between hands

### 5. `pipeline.py` - HandDetectionPipeline Class
- Main pipeline orchestration
- Threading and batch processing
- Output directory management
- Logging setup
- Detector initialization
- `run()`: Single-threaded execution
- `run_multithreaded()`: Multi-threaded execution

## Usage

### Basic Usage
```python
from pipeline import PipelineConfig, HandDetectionPipeline

# Configure pipeline
config = PipelineConfig(
    input_path="data/orbbec/camera01",
    camera_path="data/",
    output_path="output/results",
    camera_name="camera01",
    model="mediapipe",
    min_detection_confidence=0.3,
    verbose=True
)

# Create and run pipeline
pipeline = HandDetectionPipeline(config)
pipeline.run(start_idx=0, end_idx=100)
```

### Multi-threaded Usage
```python
# For faster processing with multiple threads
pipeline.run_multithreaded(start_idx=0, end_idx=1000)
```