import cv2
import numpy as np
from typing import Tuple, Optional

from models.hand_detector import HandDetection
from .config import CameraShiftConfig, ShiftDirection


class LandmarkOperations:
    """ROI, bounding box, and cropping operations"""
    
    def __init__(self, config):
        self.config = config
    
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
            else:
                bbox[:, 0] -= shift  # Shift left for right hand
                
        elif cam_config.direction == ShiftDirection.UP_DOWN or cam_config.direction == ShiftDirection.DOWN_UP:
            bbox[0, 0] -= half_shift    # top-left: left
            bbox[1, 0] -= half_shift    # bottom-left: left
            bbox[2, 0] += half_shift    # bottom-right: right
            bbox[3, 0] += half_shift    # top-right: right

            if cam_config.direction == ShiftDirection.UP_DOWN:
                # Vertical shift for cameras 2
                if hand_side == "left":
                    bbox[:, 1] += shift  # Shift down for left hand
                else:
                    bbox[:, 1] -= shift  # Shift up for right hand
            else:
                # Vertical shift for cameras 3
                if hand_side == "left":
                    bbox[:, 1] -= shift  # Shift down for left hand
                else:
                    bbox[:, 1] += shift  # Shift up for right hand
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
        
        # Get wrist coordinates (landmark[9])
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
    
    def draw_landmarks_and_bbox(self, image: np.ndarray, detection: HandDetection, 
                               bbox: np.ndarray, roi: np.ndarray, detector) -> np.ndarray:
        """Draw hand landmarks and bounding boxes on image"""
        result_img = image.copy()
        
        if detection and len(detection.landmarks) > 0:
            detector.draw_landmarks(detection, result_img)
        
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
