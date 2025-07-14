import cv2
import numpy as np
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading
from enum import Enum
import os
import argparse

from utils.camera import load_cam_infos
from utils.image import undistort_image

from models.hand_detector import HandDetection
from models.mediapipe_detector import MediaPipeHandDetector
from models.openpose_detector import OpenPoseHandDetector


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
            "camera06": CameraShiftConfig(ShiftDirection.DIAGONAL_UP_LEFT, base_shift),
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
    
    # Threading parameters
    max_workers: int = 4
    batch_size: int = 100


class HandDetectionPipeline:
    """Clean pipeline for hand detection and cropping"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_output_dirs()
        self.setup_logging()
        self.load_camera_params()


        if self.config.model == "mediapipe":
            # Initialize detector
            self.detector = MediaPipeHandDetector(
                max_num_hands=self.config.max_num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
        elif self.config.model == "openpose":
            self.detector = OpenPoseHandDetector(
                max_num_hands=self.config.max_num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
        else:
            raise ValueError(f"{self.config.model} implementation not existant...")
        
        # Thread-safe counters
        self.processed_count = 0
        self.failed_count = 0
        self.partial_count = 0  # New counter for partial detections (single hand)
        self.lock = Lock()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_folder = os.path.join(self.config.output_path, 'log')
        logfile = os.path.join(log_folder, 'output.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
            handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_output_dirs(self):
        """Create output directories"""
        output_path = Path(self.config.output_path)
        if self.config.verbose:
            self.dirs = {
                'cropped': output_path / 'cropped',
                'bboxes': output_path / 'bboxes',
                'preds': output_path / 'preds',
                'original': output_path / 'original',
                'shifted_roi': output_path / 'shifted_roi',
                'json': output_path / 'json',
                'log': output_path / 'log'
            }
        else:
            self.dirs = {
                'cropped': output_path / 'cropped',
                'json': output_path / 'json',
                'log': output_path / 'log'
            }

        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        for hand_side in ['left', 'right']:
            for dir_name in ['cropped', 'bboxes', 'preds']:
                if dir_name in self.dirs:
                    (self.dirs[dir_name] / hand_side).mkdir(parents=True, exist_ok=True)
    
    def load_camera_params(self):
        """Load camera calibration parameters"""
        try:
            cam_infos = load_cam_infos(Path(self.config.camera_path), orbbec=self.config.orbbec_cam)
            
            self.cam_params = cam_infos[self.config.camera_name]
        except Exception as e:
            self.logger.error(f"Failed to load camera parameters: {e}")
            raise
    
    def load_and_undistort_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load and undistort an image"""
        img = cv2.imread(img_path)
        if img is None:
            self.logger.warning(f"Failed to load image: {img_path}")
            return None
        
        if self.config.orbbec_cam:
            undistorted = undistort_image(img, self.cam_params, 'color')
        else:
            # Apply undistortion
            distortion_coeffs = np.array([
                self.cam_params['radial_params'][0],
                self.cam_params['radial_params'][1],
                *self.cam_params['tangential_params'][:2],
                self.cam_params['radial_params'][2],
                0, 0, 0
            ])
            undistorted = cv2.undistort(img, self.cam_params['intrinsics'], distortion_coeffs)
        return undistorted
    
    def comp_shift_diff(self, data: Dict[str, List[Dict]]) -> int:
        """"Computes the absolute difference in shifts between left and right hand"""
        r_shift = data['people'][0]['hand_right_shift']
        l_shift = data['people'][0]['hand_left_shift']
        x1, y1 = r_shift[0], r_shift[1] 
        x2, y2 = l_shift[0], l_shift[1]
        diff = math.sqrt((max(x1, x2) - min(x1, x2))**2 + (max(y1, y2) - min(y1, y2))**2)
        return diff

    def try_rotation_angles(self, image: np.ndarray) -> Tuple[Optional[any], int, np.ndarray]:
        """Try different image rotations for hand detection"""
        angles = [0, 90, 180, 270]
        for angle in angles:
            if angle == 0:
                img_rotated = image
            else:
                # For 90 degree rotations, use cv2's built-in functions
                if angle == 90:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            results_rotated = self.detector.detect_hands(img_rotated)
            if len(results_rotated) > 0:
                return results_rotated, angle, img_rotated
        return [], 0, image
    
    def transform_landmarks_back(self, detection: HandDetection, angle: int, rotated_image_shape: Tuple[int, int], original_image_shape: Tuple[int, int]):
        """Transform landmarks from rotated image back to original coordinate system"""
        h_rot, w_rot = rotated_image_shape[:2]
        h_orig, w_orig = original_image_shape[:2]
        
        landmarks = detection.landmarks
        for idx in range(0, landmarks.shape[0]):
            x, y, z = landmarks[idx]
            # Convert from normalized to pixel coordinates in rotated image
            x = x * w_rot
            y = y * h_rot
            
            # Transform back to original orientation
            if angle == 90:
                x_new, y_new = y, w_rot - x
            elif angle == 180:
                x_new, y_new = w_rot - x, h_rot - y
            elif angle == 270:
                x_new, y_new = h_rot - y, x
            else:  # angle == 0
                x_new, y_new = x, y
            
            # Convert back to normalized coordinates in original image
            x = x_new / w_orig
            y = y_new / h_orig

            landmarks[idx] = [x, y, z]
        detection.landmarks = landmarks
    
    def detect_hands_with_enhancement(self, image: np.ndarray) -> Tuple[Optional[any], int]:
        """Detect hands with brightness/contrast enhancement and rotation if needed"""
        # Try original image first
        rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.detector.detect_hands(rgb_image)
        
        if len(results) > 0:
            return results, 0
        
        # Try with brightness/contrast adjustments
        for alpha in np.arange(0, self.config.max_alpha, self.config.alpha_step):
            for beta in range(self.config.min_beta, self.config.max_beta, self.config.beta_step):
                enhanced = np.clip(image.copy().astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                # Convert RGB to YUV
                if self.config.camera_name not in ['camera05', 'camera06']:
                    img_yuv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2YUV)
                    # Equalize only the Y channel (luminance)
                    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                    # Convert back to RGB
                    enhanced_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
                results = self.detector.detect_hands(enhanced_rgb)
                
                if len(results) > 0:
                    return results, 0
                
                # If enhancement alone doesn't work, try with rotation
                results_rotated, angle, rotated_img = self.try_rotation_angles(enhanced_rgb)
                if len(results_rotated) > 0:
                    # Transform landmarks back to original coordinate system
                    for detection in results_rotated:
                        self.transform_landmarks_back(
                            detection, angle, rotated_img.shape, image.shape
                        )
                    return results_rotated, angle
        
        # If enhancement fails, try rotation on original image
        results_rotated, angle, rotated_img = self.try_rotation_angles(rgb_image)
        if len(results_rotated) > 0:
            # Transform landmarks back to original coordinate system
            for detection in results_rotated:
                self.transform_landmarks_back(
                    detection, angle, rotated_img.shape, image.shape
                )
            return results_rotated, angle
        
        return [], 0
    
    def get_roi_points(self, detection: HandDetection, image_shape: Tuple[int, int]) -> np.ndarray:
        """Get ROI points around hand landmarks"""
        h, w = image_shape[:2]
        
        points = []
        for landmark in detection.landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            points.append((x, y))
        
        x_coords, y_coords = zip(*points)
        padding = self.config.roi_padding
            
        roi_points = np.array([
            [min(x_coords) - padding, min(y_coords) - padding],
            [min(x_coords) - padding, max(y_coords) + padding],
            [max(x_coords) + padding, max(y_coords) + padding],
            [max(x_coords) + padding, min(y_coords) - padding]
        ], dtype=np.int32)
        
        return roi_points
    
    def compute_shifted_bbox(self, roi: np.ndarray, hand_side: str, shift_multiplier: float = 1.1) -> np.ndarray:
        """Compute shifted bounding box based on camera configuration"""
        bbox = roi.copy()
        shift = self.config.bbox_shift
        half_shift = shift // 2
        
        # Get camera-specific shift configuration
        cam_config = CameraShiftConfig.get_camera_config(self.config.camera_name, shift)
        
        # Expand ROI in all directions with a small base expansion
        bbox[0, 0] -= shift    # top-left: left
        bbox[0, 1] -= shift         # top-left: up
        bbox[1, 0] -= shift    # bottom-left: left
        bbox[1, 1] += shift         # bottom-left: down
        bbox[2, 0] += shift    # bottom-right: right 
        bbox[2, 1] += shift         # bottom-right: down
        bbox[3, 0] += shift    # top-right: right
        bbox[3, 1] -= shift         # top-right: up
        
        # Convert shift to integer after applying multiplier
        shift = int(shift * shift_multiplier)
        
        # Apply camera-specific shifts
        if cam_config.direction == ShiftDirection.LEFT_RIGHT:
            bbox[0, 1] -= half_shift    # top-left: up
            bbox[1, 1] += half_shift    # bottom-left: down
            bbox[2, 1] += half_shift    # bottom-right: down
            bbox[3, 1] -= half_shift    # top-right: up
            
            # Original left-right behavior
            if hand_side == "left":
                bbox[:, 0] += shift  # Shift right for left hand
                # bbox[2, 0] += shift
                # bbox[3, 0] += shift
            else:
                bbox[:, 0] -= shift  # Shift left for right hand
                # bbox[0, 0] -= shift
                # bbox[1, 0] -= shift
                
        elif cam_config.direction == ShiftDirection.UP_DOWN or cam_config.direction == ShiftDirection.DOWN_UP:
            bbox[0, 0] -= half_shift    # top-left: left
            bbox[1, 0] -= half_shift    # bottom-left: left
            bbox[2, 0] += half_shift    # bottom-right: right
            bbox[3, 0] += half_shift    # top-right: right

            if cam_config.direction == ShiftDirection.UP_DOWN:
                # Vertical shift for cameras 2
                if hand_side == "left":
                    bbox[:, 1] += shift  # Shift down for left hand
                    # bbox[1, 1] += shift
                    # bbox[2, 1] += shift
                else:
                    bbox[:, 1] -= shift  # Shift up for right hand
                    # bbox[0, 1] -= shift
                    # bbox[3, 1] -= shift
            else:
                # Vertical shift for cameras 3
                if hand_side == "left":
                    bbox[:, 1] -= shift  # Shift down for left hand
                    # bbox[1, 1] -= shift
                    # bbox[2, 1] -= shift
                else:
                    bbox[:, 1] += shift  # Shift up for right hand
                    # bbox[0, 1] += shift
                    # bbox[3, 1] += shift
        elif cam_config.direction == ShiftDirection.DIAGONAL_DOWN_LEFT:
            bbox[0, 1] -= half_shift    # top-left: up
            bbox[1, 1] += half_shift    # bottom-left: down
            bbox[2, 1] += half_shift    # bottom-right: down
            bbox[3, 1] -= half_shift    # top-right: up

            # Diagonal shift for camera 4
            if hand_side == "right":
                # Shift down-left for right hand
                bbox[:, 0] -= shift     # Shift left
                bbox[:, 1] += shift     # Shift down
            else:
                # Shift up-right for left hand
                bbox[:, 0] += shift     # Shift right
                bbox[:, 1] -= shift     # Shift up
        elif cam_config.direction == ShiftDirection.DIAGONAL_UP_LEFT:
            bbox[0, 1] -= half_shift    # top-left: up
            bbox[1, 1] += half_shift    # bottom-left: down
            bbox[2, 1] += half_shift    # bottom-right: down
            bbox[3, 1] -= half_shift    # top-right: up

            # Diagonal shift for camera 6
            if hand_side == "right":
                # Shift up-left for right hand
                bbox[:, 0] -= shift     # Shift left
                bbox[:, 1] -= shift     # Shift up
            else:
                # Shift down-right for left hand
                bbox[:, 0] += shift     # Shift right
                bbox[:, 1] += shift     # Shift down
        
        return bbox
    
    def draw_bbox(self, image: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Draw bounding box on the image"""
        # Draw bounding box
        for x, y in roi:
            cv2.circle(image, (int(x), int(y)), 4, (0, 255, 255), -1)
        
        # Draw bbox lines
        for i in range(len(roi)):
            cv2.line(image, tuple(roi[i]), tuple(roi[(i+1) % len(roi)]), (255, 0, 255), 1)
        
        return image
    
    def crop_to_bbox(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop image to bounding box"""
        x_coords, y_coords = bbox[:, 0], bbox[:, 1]
        
        x_min = max(0, int(min(x_coords)))
        y_min = max(0, int(min(y_coords)))
        x_max = min(image.shape[1], int(max(x_coords)))
        y_max = min(image.shape[0], int(max(y_coords)))
        
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped, (x_min, y_min)
    
    def crop_fixed_size(self, image: np.ndarray, detection: HandDetection, 
                       base_offset: Tuple[int, int] = (0, 0),
                       current_image_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop fixed size region around wrist joint"""
        h, w = image.shape[:2]
        size = self.config.crop_size
        
        # Determine coordinate system
        if current_image_size is not None:
            curr_h, curr_w = current_image_size
        else:
            curr_h, curr_w = h, w
        
        # Get wrist coordinates (landmark[0])
        root_x_detected = int(detection.landmarks[9][0] * curr_w)
        root_y_detected = int(detection.landmarks[9][1] * curr_h)
        
        # Transform to original image coordinates
        root_x = root_x_detected + base_offset[0]
        root_y = root_y_detected + base_offset[1]
        
        # Calculate crop boundaries
        half_size = size // 2
        x_min = max(0, root_x - half_size)
        y_min = max(0, root_y - half_size)
        x_max = min(w, root_x + half_size)
        y_max = min(h, root_y + half_size)
        
        # Adjust if out of bounds
        if x_min == 0:
            x_max = min(w, size)
        if y_min == 0:
            y_max = min(h, size)
        if x_max == w:
            x_min = max(0, w - size)
        if y_max == h:
            y_min = max(0, h - size)
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        # Pad if necessary
        if cropped.shape[0] != size or cropped.shape[1] != size:
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            pad_y = (size - cropped.shape[0]) // 2
            pad_x = (size - cropped.shape[1]) // 2
            padded[pad_y:pad_y+cropped.shape[0], pad_x:pad_x+cropped.shape[1]] = cropped
            return padded, (x_min, y_min)
        
        return cropped, (x_min, y_min)

    
    def draw_landmarks_and_bbox(self, image: np.ndarray, detection: HandDetection, bbox: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Draw hand landmarks and bounding boxes on image"""
        result_img = image.copy()
        
        if detection and len(detection.landmarks) > 0:
            self.detector.draw_landmarks(detection, result_img)
        
        # Draw bounding box
        for x, y in bbox:
            cv2.circle(result_img, (int(x), int(y)), 4, (0, 255, 255), -1)
        
        # Draw bbox lines
        for i in range(len(bbox)):
            cv2.line(result_img, tuple(bbox[i]), tuple(bbox[(i+1) % len(bbox)]), (255, 0, 255), 1)
        
        # Draw ROI
        for x, y in roi:
            cv2.circle(result_img, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        # Draw ROI lines
        for i in range(len(roi)):
            cv2.line(result_img, tuple(roi[i]), tuple(roi[(i+1) % len(roi)]), (255, 0, 255), 1)
        
        return result_img
    
    def check_diff_hand(self, detection: HandDetection, current_hand_side: str, prev_detected_handside: str, cropped_img: np.ndarray, bbox_origin: np.ndarray, final_origin: np.ndarray, data: Dict[str, List[Dict]]) -> Tuple[Dict[str, List[Dict]], bool]:
        temp_keypoint_data = self.detector.get_keypoints_data(
            detection, cropped_img.shape[:2], 
            [bbox_origin, final_origin]
        )
    
        tmp_data = {"people": [{"hand_left_shift": [], "hand_left_keypoints_2d": [], "hand_left_conf": [],
                    "hand_right_shift": [], "hand_right_keypoints_2d": [], "hand_right_conf": []}]}

        # Store current data
        temp_keypoint_data[f"hand_{prev_detected_handside}_keypoints_2d"] = data["people"][0][f"hand_{prev_detected_handside}_keypoints_2d"]
        temp_keypoint_data[f"hand_{prev_detected_handside}_shift"] = data["people"][0][f"hand_{prev_detected_handside}_shift"]
        temp_keypoint_data[f"hand_{prev_detected_handside}_conf"] = data["people"][0][f"hand_{prev_detected_handside}_conf"]
        tmp_data['people'][0].update(temp_keypoint_data)

        # Check if the detected hand is actually different
        diff = self.comp_shift_diff(tmp_data)

        if diff > 175:
            return tmp_data, True
        else:
            return None, False 

    def detect_second_hand_or_retry(self, img_idx: int, image: np.ndarray, blank_img: np.ndarray, detection: HandDetection, roi: np.ndarray, data: Dict[str, List[Dict]]) -> None:
        """Detect second hand with retry mechanism for hand misclassification"""
        current_hand_side = detection.hand_type
        prev_detected_handside = current_hand_side
        shift_multiplier = 1.5

        for retry_counter in range(0, 4):
            # Attempt hand detection
            bbox = self.compute_shifted_bbox(roi.copy(), current_hand_side, shift_multiplier)
            cropped_img, bbox_origin = self.crop_to_bbox(image.copy(), bbox)
            cropped_results, _ = self.detect_hands_with_enhancement(cropped_img)
            
            if self.config.verbose:
                full_vis = self.draw_landmarks_and_bbox(image.copy(), None, bbox, roi)
                cv2.imwrite(self.dirs['shifted_roi'] / f"{img_idx}_{self.config.bbox_shift}_{retry_counter}.jpg", full_vis)

            if len(cropped_results) > 0:
                detected_detection = cropped_results[0]
                detected_hand_side = "right" if current_hand_side == "left" else "left"
                detected_detection.hand_type = detected_hand_side
                temp = data.copy()

                # Create final crop from original image using detected landmarks
                final_crop, final_origin = self.crop_fixed_size(
                        image.copy(), detected_detection, 
                        base_offset=bbox_origin, 
                        current_image_size=cropped_img.shape[:2]
                )

                if self.config.verbose:
                    cv2.imwrite(self.dirs['preds'] / detected_hand_side / f"{img_idx}.jpg", final_crop)

                if retry_counter >= 2:
                    temp["people"][0][f"hand_{current_hand_side}_keypoints_2d"] = data["people"][0][f"hand_{prev_detected_handside}_keypoints_2d"]
                    temp["people"][0][f"hand_{current_hand_side}_shift"] = data["people"][0][f"hand_{prev_detected_handside}_shift"]
                    temp["people"][0][f"hand_{current_hand_side}_conf"] = data["people"][0][f"hand_{prev_detected_handside}_conf"]
                    cv2.imwrite(self.dirs['cropped'] / current_hand_side / f"{img_idx}.jpg", blank_img)
                    tmp_data, check = self.check_diff_hand(detected_detection, prev_detected_handside, current_hand_side, cropped_img, bbox_origin, final_origin, temp)
                else:    
                    tmp_data, check = self.check_diff_hand(detected_detection, current_hand_side, prev_detected_handside, cropped_img, bbox_origin, final_origin, temp)
                if check:
                    cv2.imwrite(self.dirs['cropped'] / detected_hand_side / f"{img_idx}.jpg", final_crop)
                    return tmp_data

            if retry_counter % 2 == 0:
                shift_multiplier = 1.75
            else:
                shift_multiplier = 1.25
                prev_detected_handside = current_hand_side
                current_hand_side = "right" if current_hand_side == "left" else "left"
        
        return None
    
    def process_single_hand(self, image: np.ndarray, detection: HandDetection, img_idx: str) -> Optional[Dict[str, Any]]:
        """Process image with single hand detection"""
        self.logger.info(f"Processing {img_idx}: {detection.hand_type} hand detected")
        
        # Initialize data structure
        data = {"people": [{"hand_left_shift": [], "hand_left_keypoints_2d": [], "hand_left_conf": [], 
                           "hand_right_shift": [], "hand_right_keypoints_2d": [], "hand_right_conf": []}]}
        
        # Initial crop from original image
        blank_img, crop_origin = self.crop_fixed_size(image.copy(), detection)
        cv2.imwrite(str(self.dirs['cropped'] / str(detection.hand_type) / f"{img_idx}.jpg"), blank_img)
        

        # Get keypoints data
        keypoint_data = self.detector.get_keypoints_data(detection, image.shape[:2], [crop_origin])
        data["people"][0].update(keypoint_data)
        
        if self.config.verbose:
            # Create visualization with landmarks
            vis_img = image.copy()
            vis_img = self.detector.draw_landmarks(detection, vis_img)
        
            # Crop visualization
            cropped_vis, _ = self.crop_fixed_size(vis_img, detection)
            cv2.imwrite(self.dirs['preds'] / str(detection.hand_type) / f"{img_idx}.jpg", cropped_vis)
    
            cv2.imwrite(self.dirs['original'] / f"{img_idx}.jpg", image)
        
        # Try shifted bbox approach with retry mechanism
        roi = self.get_roi_points(detection, image.shape)
        bbox_img = self.draw_bbox(image.copy(), roi)
        if self.config.verbose:
            cv2.imwrite(self.dirs['bboxes'] / str(detection.hand_type) / f"{img_idx}.jpg", bbox_img)

        # Create a new MediaPipe instance with max_hands=1 for retry logic
        self.detector.__init__(
            max_num_hands=1, 
            min_detection_confidence=self.config.min_detection_confidence, 
            min_tracking_confidence=self.config.min_tracking_confidence
        )

        # try to detect second hand
        data = self.detect_second_hand_or_retry(img_idx, image.copy(), blank_img, detection, roi, data)
        
        self.detector.__init__(
            max_num_hands=2, 
            min_detection_confidence=self.config.min_detection_confidence, 
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        return data
    
    def process_double_hands(self, image: np.ndarray, detections: List[HandDetection], 
                           img_idx: str) -> Dict[str, Any]:
        """Process image with two hands detected"""
        self.logger.info(f"Processing {img_idx}: 2 hands detected")
        
        # Initialize data structure
        data = {"people": [{"hand_left_shift": [], "hand_left_keypoints_2d": [], "hand_left_conf": [],
                           "hand_right_shift": [], "hand_right_keypoints_2d": [], "hand_left_conf": []}]}
        
        if self.config.verbose:
            # Save original image
            cv2.imwrite(self.dirs['original'] / f"{img_idx}.jpg", image)

        if detections[0].landmarks[0][0] < detections[1].landmarks[0][0]:
            detections[0].hand_type = "left"
            detections[1].hand_type = "right"
        else:
            detections[0].hand_type = "right"
            detections[1].hand_type = "left"

        # Process both hands
        for detection in detections:
            # Crop around each hand
            blank_img, crop_origin = self.crop_fixed_size(image.copy(), detection)
            cv2.imwrite(self.dirs['cropped'] / str(detection.hand_type) / f"{img_idx}.jpg", blank_img)

            # Get keypoints
            keypoint_data = self.detector.get_keypoints_data(detection, image.shape[:2], [crop_origin])
            data["people"][0].update(keypoint_data)

            if self.config.verbose:
                # Create visualization with landmarks
                vis_img = image.copy()
                vis_img = self.detector.draw_landmarks(detection, vis_img)
                
                # Crop visualization
                cropped_vis, _ = self.crop_fixed_size(vis_img, detection)
                cv2.imwrite(self.dirs['preds'] / str(detection.hand_type) / f"{img_idx}.jpg", cropped_vis)
        
        return data
    
    def process_image_thread_safe(self, img_idx: str) -> bool:
        """Thread-safe version of process_image"""
        # Construct image path
        img_path = f"{self.config.input_path}/{self.config.image_prefix}{img_idx}{self.config.image_suffix}"
        
        # Load and undistort image
        image = self.load_and_undistort_image(img_path)
        if image is None:
            return False
        
        # Detect hands
        detections, _ = self.detect_hands_with_enhancement(image)
        if len(detections) == 0:
            self.logger.info(f"No hands detected in {img_idx}")
            return False
        
        # Process based on number of hands
        if len(detections) == 2:
            data = self.process_double_hands(image, detections, img_idx)
        elif len(detections) == 1:
            data = self.process_single_hand(image, detections[0], img_idx)
        else:
            return False
        
        if data is None:
            return False
        
        # Save JSON data
        json_path = self.dirs['json'] / f"{img_idx}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return True
    
    def process_batch(self, indices: List[int]) -> Tuple[int, int]:
        """Process a batch of images and return (processed_count, failed_count)"""
        batch_processed = 0
        batch_failed = 0
        
        for idx in indices:
            img_idx = f"{idx:06d}"
            
            try:
                if self.process_image_thread_safe(img_idx):
                    batch_processed += 1
                else:
                    batch_failed += 1
            except Exception as e:
                self.logger.error(f"Error processing {img_idx}: {e}")
                batch_failed += 1
        
        # Thread-safe update of global counters
        with self.lock:
            self.processed_count += batch_processed
            self.failed_count += batch_failed
        
        return batch_processed, batch_failed
    
    def create_batches(self, start_idx: int, end_idx: int) -> List[List[int]]:
        """Create batches of image indices"""
        all_indices = list(range(start_idx, end_idx))
        batches = []
        
        for i in range(0, len(all_indices), self.config.batch_size):
            batch = all_indices[i:i + self.config.batch_size]
            batches.append(batch)
        
        return batches
    
    def run_multithreaded(self, start_idx: int, end_idx: int):
        """Run the pipeline with multithreading"""
        self.logger.info(f"Starting multithreaded pipeline for images {start_idx:06d} to {end_idx:06d}")
        self.logger.info(f"Configuration: {self.config.max_workers} workers, batch size {self.config.batch_size}")
        
        # Create batches
        batches = self.create_batches(start_idx, end_idx)
        self.logger.info(f"Created {len(batches)} batches")
        
        # Reset counters
        self.processed_count = 0
        self.failed_count = 0
        
        # Process batches with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(self.process_batch, batch): i for i, batch in enumerate(batches)}
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_processed, batch_failed = future.result()
                    self.logger.info(f"Batch {batch_idx + 1}/{len(batches)} completed: "
                                   f"processed={batch_processed}, failed={batch_failed}")
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx + 1} failed with error: {e}")
        
        self.logger.info(f"Multithreaded pipeline completed. "
                        f"Total processed: {self.processed_count}, Total failed: {self.failed_count}")

    def process_image(self, img_idx: str) -> bool:
        """Process a single image (legacy method for backward compatibility)"""
        # Construct image path
        img_path = f"{self.config.input_path}/{self.config.image_prefix}{img_idx}{self.config.image_suffix}"
        
        # Load and undistort image
        image = self.load_and_undistort_image(img_path)
        if image is None:
            return False
        
        # Detect hands
        results, angle = self.detect_hands_with_enhancement(image)
        if len(results) == 0:
            self.logger.info(f"No hands detected in {img_idx}")
            return False
        
        if angle != 0:
            self.logger.debug(f"Hands detected in {img_idx} using {angle}° rotation")
        
        # Process based on number of hands
        if len(results) == 2:
            data = self.process_double_hands(image, results, img_idx)
        elif len(results) > 0:
            data = self.process_single_hand(image, results[0], img_idx)
        else:
            return False
        
        if data is None:
            return False
        
        # Save JSON data
        json_path = self.dirs['json'] / f"{img_idx}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return True

    def run(self, start_idx: int, end_idx: int):
        """Run the pipeline on a range of images (single-threaded)"""
        self.logger.info(f"Starting single-threaded pipeline for images {start_idx:06d} to {end_idx:06d}")
        
        processed_count = 0
        failed_count = 0
        
        for idx in range(start_idx, end_idx):
            img_idx = f"{idx:06d}"
            
            try:
                if self.process_image(img_idx):
                    processed_count += 1
                else:
                    failed_count += 1
                    self.logger.warning(f"Failed to detect 2 hands for {img_idx}")
            except Exception as e:
                self.logger.error(f"Error processing {img_idx}: {e}")
                failed_count += 1
        
        self.logger.info(f"Single-threaded pipeline completed. Processed: {processed_count}, Failed: {failed_count}")
        self.detector.release()


def main():
    """Main execution function"""
    # Get script directory
    script_dir = Path(__file__)
    
    # Configure pipeline
    model = "mediapipe"
    multi_run = False

    if multi_run:
        cameras = {
            "camera01": 0.5,
            "camera02": 0.4,
            "camera03": 0.4,
            "camera04": 0.4,
            "camera05": 0.4,
            "camera06": 0.4
        }
        min_tracking_confidence = 0.75
        run_start = time.time()
        for camera, conf in cameras.items():
            orbbec_cam = True if camera not in ['camera05', 'camera06'] else False
            # conf = 0.35

            print(camera)
            if camera in ['camera01', 'camera05', 'camera06']:
                continue
            
            # Build paths relative to script directory
            base_path = script_dir.parent.parent.parent / "data" / "input"
            output_path = script_dir.parent.parent / "output" / "test" /  model / "diff_confs_run" / camera / f"conf_{conf:.2f}"

            config = PipelineConfig(
                input_path=str(base_path / "orbbec" / camera),
                camera_path=str(base_path),
                output_path=str(output_path),
                camera_name=camera,
                orbbec_cam=orbbec_cam,
                model=model,
                min_detection_confidence=conf,
                min_tracking_confidence=min_tracking_confidence,
                crop_size=256,
                bbox_shift=150,
                max_workers=10,     # Number of threads
                batch_size=20       # Images per batch
            )
            
            # Create pipeline
            pipeline = HandDetectionPipeline(config)

            start = time.time()
            
            # Use multithreaded version
            # pipeline.run_multithreaded(start_idx=0, end_idx=200)
            
            # Alternative: use single-threaded version
            pipeline.run(start_idx=0, end_idx=0)
            
            end = time.time()
            
            elapsed_time = end - start
            pipeline.logger.info(f"Pipeline execution time: {elapsed_time:.2f} seconds")

        run_end = time.time()
        run_time = run_end - run_start
        pipeline.logger.info(f"Entire run execution time: {run_time:.2f} seconds")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cam', type=int, default=1)
        parser.add_argument('--start', type=int, default=0)
        parser.add_argument('--end', type=int, default=20)
        parser.add_argument('--verbose', type=bool, default=False)
        args = parser.parse_args()

        camera = f"camera0{args.cam}"
        orbbec_cam = True if camera not in ['camera05', 'camera06'] else False
        conf = 0.3
        min_tracking_confidence = 0.75
        print(camera)
        # Build paths relative to script directory
        base_path = script_dir.parent.parent.parent / "data"
        output_path = script_dir.parent.parent.parent / "output" / f"{model}_{conf:.2f}" / camera 

        config = PipelineConfig(
            input_path=str(base_path / "orbbec" / camera),
            camera_path=str(base_path),
            output_path=str(output_path),
            verbose=args.verbose,
            camera_name=camera,
            orbbec_cam=orbbec_cam,
            model=model,
            min_detection_confidence=conf,
            min_tracking_confidence=min_tracking_confidence,
            crop_size=256,
            bbox_shift=150,
            max_workers=10,     # Number of threads
            batch_size=20       # Images per batch
        )
        
        # Create pipeline
        pipeline = HandDetectionPipeline(config)

        start = time.time()
        
        # Use multithreaded version
        # pipeline.run_multithreaded(start_idx=0, end_idx=200)
        
        # Alternative: use single-threaded version
        pipeline.run(start_idx=args.start, end_idx=args.end)
        
        end = time.time()
        
        elapsed_time = end - start
        pipeline.logger.info(f"Pipeline execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()