from dataclasses import dataclass
from enum import Enum


class ShiftDirection(Enum):
    LEFT_RIGHT = "left_right"                   # Original behavior (cameras 1, 5)
    UP_DOWN = "up_down"                         # For cameras 2
    DOWN_UP = "down_up"                         # For camera 3
    DIAGONAL_DOWN_LEFT = "diagonal_down_left"   # For camera 4
    DIAGONAL_UP_LEFT = "diagonal_up_left"       # For camera 6

@dataclass
class CameraShiftConfig:
    """Configuration for camera-specific bbox shifts"""
    direction: ShiftDirection
    primary_shift: int  # Main shift amount
    secondary_shift: int = 0  # Secondary shift amount for diagonal movements

    @staticmethod
    def get_camera_config(camera_name: str, base_shift: int) -> 'CameraShiftConfig':
        """Get camera-specific shift configuration"""
        configs = {
            "camera01": CameraShiftConfig(ShiftDirection.LEFT_RIGHT, base_shift),
            "camera02": CameraShiftConfig(ShiftDirection.UP_DOWN, base_shift),
            "camera03": CameraShiftConfig(ShiftDirection.DOWN_UP, base_shift),
            "camera04": CameraShiftConfig(ShiftDirection.DIAGONAL_DOWN_LEFT, base_shift),
            "camera05": CameraShiftConfig(ShiftDirection.LEFT_RIGHT, base_shift),
            "camera06": CameraShiftConfig(ShiftDirection.DIAGONAL_UP_LEFT, base_shift)
        }
        return configs.get(camera_name, CameraShiftConfig(ShiftDirection.LEFT_RIGHT, base_shift))


@dataclass
class PipelineConfig:
    """Configuration for the hand detection pipeline"""
    input_path: str = "data/orbbec/"
    camera_path: str = "data/"
    output_path: str = "preprocessed_OR_data"
    verbose: bool = False
    camera_name: str = "camera01"
    orbbec_cam: bool = True if camera_name not in ['camera05', 'camera06'] else False
    image_prefix: str = ""
    image_suffix: str = ".jpg"
    
    # Processing parameters
    crop_size: int = 256
    roi_padding: int = 20
    bbox_shift: int = 250
    
    # Model parameters
    model: str = "mediapipe"
    max_num_hands: int = 2
    min_detection_confidence: float = 0.1
    min_tracking_confidence: float = 0.5
    
    # Brightness adjustment parameters
    max_alpha: float = 2.1
    alpha_step: float = 0.3
    max_beta: int = 26
    min_beta: int = -45
    beta_step: int = 10
    normalize_to_first_lighting = True
    normalization_method = "reinhard"
    normalization_mask_margin = 0.02
    clahe_clip = 2.0
    clahe_grid = 8
    unsharp_amount = 0.0
    
    # Threading parameters
    max_workers: int = 4
    batch_size: int = 100
