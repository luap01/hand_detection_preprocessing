from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class HandDetection:
    """Class to store hand detection results"""
    landmarks: np.ndarray  # Nx3 array of landmarks (x, y, confidence)
    hand_type: str  # "left" or "right"
    confidence: float
    
    @property
    def is_left(self) -> bool:
        return self.hand_type.lower() == "left"
    
    @property
    def is_right(self) -> bool:
        return self.hand_type.lower() == "right"


class HandDetector(ABC):
    """Abstract base class for hand detectors"""
    
    @abstractmethod
    def initialize(self):
        """Initialize the model and load any necessary weights"""
        pass
    
    @abstractmethod
    def detect_hands(self, image: np.ndarray) -> Tuple[List[HandDetection], Optional[np.ndarray]]:
        """
        Detect hands in the image
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Tuple of:
            - List of HandDetection objects
            - Optional annotated image for visualization
        """
        pass
    
    @abstractmethod
    def get_keypoints_data(self, detection: HandDetection, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Convert detection to keypoints data format
        
        Args:
            detection: HandDetection object
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary with keypoints data in the format:
            {
                "hand_{left/right}_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...],
                "hand_{left/right}_shift": [x_shift, y_shift]
            }
        """
        pass

    @abstractmethod
    def draw_landmarks(self, detection: HandDetection, image: np.array) -> np.array:
        """
        Visualize keypoints on image

        Args:
            detection: HandDetection object
            image: RGB image as numpy array

        Returns:
            image: RGB image as numpy array
        """
        pass

    @abstractmethod
    def get_hand_center_x(self, detection: HandDetection):
        """
        Average x position of all landmarks

        Args:
            detection: HandDetection object
        Returns:
            position: Float value of the x position
        """
        pass
    
    def release(self):
        """Release any resources held by the model"""
        pass 