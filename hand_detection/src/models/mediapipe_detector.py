import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from typing import List, Tuple, Optional, Dict, Any

from .hand_detector import HandDetector, HandDetection


class MediaPipeHandDetector(HandDetector):
    """MediaPipe implementation of hand detector"""
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.initialize()
    
    def initialize(self):
        """Initialize MediaPipe hands"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect_hands(self, image: np.ndarray) -> List[HandDetection]:
        """Detect hands using MediaPipe"""
        if self.hands is None:
            self.initialize()
        results = self.hands.process(image)
        
        detections = []
        if results.multi_hand_landmarks:            
            # Process each detected hand
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Convert landmarks to numpy array
                points = []
                for landmark in landmarks.landmark:
                    points.append([landmark.x, landmark.y, landmark.z])
                landmarks_array = np.array(points)
                
                # Get hand type and confidence
                hand_type = handedness.classification[0].label.lower()
                confidence = handedness.classification[0].score
                
                # Create detection object
                detection = HandDetection(
                    landmarks=landmarks_array,
                    hand_type=hand_type,
                    confidence=confidence
                )
                detections.append(detection)
        
        return detections
    
    def get_keypoints_data(self, detection: HandDetection, image_shape: Tuple[int, int], coord_origins: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Convert MediaPipe detection to keypoints data"""
        h, w = image_shape
        keypoints = []
        
        # Calculate total offset
        x_min, y_min = coord_origins[-1]
        x_orig, y_orig = 0, 0
        if len(coord_origins) > 1:
            x_orig, y_orig = coord_origins[-2][0], coord_origins[-2][1] 

        # Extract keypoints
        for x, y, z in detection.landmarks:
            x = float(x * w) - x_min + x_orig
            y = float(y * h) - y_min + y_orig
            keypoints.extend([x, y, z])
        
        return {
            f"hand_{detection.hand_type}_keypoints_2d": keypoints,
            f"hand_{detection.hand_type}_shift": [x_min, y_min],
            f"hand_{detection.hand_type}_conf": [detection.confidence]
        }
    
    def draw_landmarks(self, detection: HandDetection, image: np.array):
        """Draw MediaPipe landmarks on image"""
        landmarks_array = detection.landmarks

        # Create a NormalizedLandmarkList
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        
        # Populate the landmark list
        for idx in range(landmarks_array.shape[0]):
            x, y, z = landmarks_array[idx]
            landmark = landmark_list.landmark.add()
            landmark.x = float(x)
            landmark.y = float(y)
            landmark.z = float(z)
        
        # Draw landmarks on image
        self.mp_drawing.draw_landmarks(
            image,
            landmark_list,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        return image
    
    def get_hand_center_x(self, detection: HandDetection):
        """Return the average x position of all landmarks"""
        return sum([x for (x, _, _) in detection.landmarks]) / len(detection.landmarks)

    def release(self):
        """Release MediaPipe resources"""
        if self.hands is not None:
            self.hands.close()
            self.hands = None 