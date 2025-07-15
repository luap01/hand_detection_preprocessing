import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sys
from pathlib import Path

# Always resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent

from .hand_detector import HandDetector, HandDetection

from .openpose_impl import model
from .openpose_impl import util
from .openpose_impl.body import Body
from .openpose_impl.hand import Hand


class OpenPoseHandDetector(HandDetector):
    """OpenPose implementation of hand detector"""
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.0,
                 min_tracking_confidence: float = 0.5,
                 base_model_path: Optional[str] = None):
        self.base_model_path = base_model_path or f'{SCRIPT_DIR}/openpose_impl/model/'
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.body_estimation = None
        self.hand_estimation = None
        self.initialize()
    
    def initialize(self):
        """Initialize OpenPose body and hand detectors"""
        if self.body_estimation is None:
            # Initialize both body and hand models as done in openpose.py
            self.body_estimation = Body(str(Path(self.base_model_path) / 'body_pose_model.pth'))
            self.hand_estimation = Hand(str(Path(self.base_model_path) / 'hand_pose_model.pth'))
    
    def detect_hands(self, image: np.ndarray) -> List[HandDetection]:
        """Detect hands using OpenPose"""
        if self.body_estimation is None:
            self.initialize()
        
        # First detect body keypoints
        candidate, subset = self.body_estimation(image)
        hands_list = util.handDetect(candidate, subset, image)
        
        detections = []
        if len(hands_list) > 0:
            # Process each detected hand
            for hand_idx, (x, y, w, is_left) in enumerate(hands_list):
                if hand_idx >= self.max_num_hands:
                    break
                    
                # Extract and process hand region
                hand_roi = image[y:y+w, x:x+w, :]
                peaks, confidences = self.hand_estimation(hand_roi)
                if peaks.shape[0] != confidences.shape[0]:
                    print(f"PEAKS: {peaks.shape[0]} vs. CONF: {confidences.shape[0]}")
                # Adjust peaks to image space
                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)

                # Compute average confidence for hand detection
                valid_confidences = confidences[confidences > 0]
                avg_confidence = np.mean(valid_confidences) if len(valid_confidences) > 0 else 0.0
                print(f"AVG CONF: {avg_confidence:.2f}")
                if avg_confidence >= self.min_detection_confidence:
                    # Convert to normalized coordinates
                    h, w = image.shape[:2]
                    normalized_peaks = peaks.astype(np.float64)  # Ensure float type
                    normalized_peaks[:, 0] /= w
                    normalized_peaks[:, 1] /= h
                    normalized_peaks = np.column_stack((normalized_peaks, confidences))

                    detection = HandDetection(
                        landmarks=normalized_peaks,
                        hand_type="left" if is_left else "right",
                        confidence=avg_confidence
                    )
                    detections.append(detection)
        else:
            peaks, confidences = self.hand_estimation(image)
            # Compute average confidence for hand detection
            valid_confidences = confidences[confidences > 0]
            avg_confidence = np.mean(valid_confidences) if len(valid_confidences) > 0 else 0.0
            if avg_confidence >= self.min_detection_confidence:
                h, w = image.shape[:2]
                normalized_peaks = peaks.astype(np.float64)  # Ensure float type
                normalized_peaks[:, 0] /= w
                normalized_peaks[:, 1] /= h
                normalized_peaks = np.column_stack((normalized_peaks, confidences))
                detection = HandDetection(
                        landmarks=normalized_peaks,
                        hand_type="",
                        confidence=avg_confidence
                )
                detections.append(detection)

        return detections
    
    def get_keypoints_data(self, detection: HandDetection, image_shape: Tuple[int, int], coord_origins: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Convert OpenPose detection to keypoints data"""
        h, w = image_shape
        
        # Calculate total offset
        x_min, y_min = coord_origins[-1]
        x_orig, y_orig = 0, 0
        if len(coord_origins) > 1:
            x_orig, y_orig = coord_origins[-2][0], coord_origins[-2][1] 

        # Convert normalized coordinates to pixel coordinates
        keypoints = []
        for x, y, c in detection.landmarks:
            px = float(x * w) - x_min + x_orig
            py = float(y * h) - y_min + y_orig
            keypoints.extend([px, py, c])
        
        return {
            f"hand_{detection.hand_type}_keypoints_2d": keypoints,
            f"hand_{detection.hand_type}_shift": [x_min, y_min],
            f"hand_{detection.hand_type}_conf": [detection.confidence]
        }
    
    def draw_landmarks(self, detection: HandDetection, image: np.array):
        """Draw OpenPose landmarks on image"""
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates for visualization
        landmarks_pixel = []
        for x, y, _ in detection.landmarks:
            px = int(x * w)
            py = int(y * h)
            landmarks_pixel.append([px, py])
            
        landmarks_array = np.array(landmarks_pixel)
        util.draw_handpose(image, [landmarks_array])
        return image

    def get_hand_center_x(self, detection: HandDetection):
        """Return the average x position of all landmarks"""
        return sum([x for (x, _, _) in detection.landmarks]) / len(detection.landmarks)
    
    def release(self):
        """Release OpenPose resources"""
        if self.body_estimation is not None:
            self.body_estimation = None
            self.hand_estimation = None 