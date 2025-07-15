import cv2
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from utils.camera import load_cam_infos
from utils.image import undistort_image
from models.hand_detector import HandDetection


class ImageOperations:
    """Image loading, enhancement, and rotation operations"""
    
    def __init__(self, config):
        self.config = config
        self.cam_params = None
        self.load_camera_params()
    
    def load_camera_params(self):
        """Load camera calibration parameters"""
        try:
            cam_infos = load_cam_infos(Path(self.config.camera_path), orbbec=self.config.orbbec_cam)
            self.cam_params = cam_infos[self.config.camera_name]
        except Exception as e:
            raise Exception(f"Failed to load camera parameters: {e}")
    
    def load_and_undistort_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load and undistort an image"""
        img = cv2.imread(img_path)
        if img is None:
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
    
    def try_rotation_angles(self, image: np.ndarray, detector) -> Tuple[List, int, np.ndarray]:
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
            
            results_rotated = detector.detect_hands(img_rotated)
            if len(results_rotated) > 0:
                return results_rotated, angle, img_rotated
        return [], 0, image
    
    def transform_landmarks_back(self, detection: HandDetection, angle: int, 
                               rotated_image_shape: Tuple[int, int], 
                               original_image_shape: Tuple[int, int]):
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
    
    def detect_hands_with_enhancement(self, image: np.ndarray, detector) -> Tuple[List, int]:
        """Detect hands with brightness/contrast enhancement and rotation if needed"""
        # Try original image first
        rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = detector.detect_hands(rgb_image)
        
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
                results = detector.detect_hands(enhanced_rgb)
                
                if len(results) > 0:
                    return results, 0
                
                # If enhancement alone doesn't work, try with rotation
                results_rotated, angle, rotated_img = self.try_rotation_angles(enhanced_rgb, detector)
                if len(results_rotated) > 0:
                    # Transform landmarks back to original coordinate system
                    for detection in results_rotated:
                        self.transform_landmarks_back(
                            detection, angle, rotated_img.shape, image.shape
                        )
                    return results_rotated, angle
        
        # If enhancement fails, try rotation on original image
        results_rotated, angle, rotated_img = self.try_rotation_angles(rgb_image, detector)
        if len(results_rotated) > 0:
            # Transform landmarks back to original coordinate system
            for detection in results_rotated:
                self.transform_landmarks_back(
                    detection, angle, rotated_img.shape, image.shape
                )
            return results_rotated, angle
        
        return [], 0
