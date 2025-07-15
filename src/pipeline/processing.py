import cv2
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple

from models.hand_detector import HandDetection


class HandProcessor:
    """Main hand detection and processing logic"""
    
    def __init__(self, config, detector, image_ops, landmark_ops, logger, dirs):
        self.config = config
        self.detector = detector
        self.image_ops = image_ops
        self.landmark_ops = landmark_ops
        self.logger = logger
        self.dirs = dirs
    
    def comp_shift_diff(self, data: Dict[str, List[Dict]]) -> int:
        """Computes the absolute difference in shifts between left and right hand"""
        r_shift = data['people'][0]['hand_right_shift']
        l_shift = data['people'][0]['hand_left_shift']
        x1, y1 = r_shift[0], r_shift[1] 
        x2, y2 = l_shift[0], l_shift[1]
        diff = math.sqrt((max(x1, x2) - min(x1, x2))**2 + (max(y1, y2) - min(y1, y2))**2)
        return diff
    
    def check_diff_hand(self, detection: HandDetection, current_hand_side: str, 
                       prev_detected_handside: str, cropped_img: np.ndarray, 
                       bbox_origin: np.ndarray, final_origin: np.ndarray, 
                       data: Dict[str, List[Dict]]) -> Tuple[Dict[str, List[Dict]], bool]:
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
    
    def detect_second_hand_or_retry(self, img_idx: int, image: np.ndarray, blank_img: np.ndarray, 
                                   detection: HandDetection, roi: np.ndarray, 
                                   data: Dict[str, List[Dict]], detector_max_1) -> Optional[Dict[str, Any]]:
        """Detect second hand with retry mechanism for hand misclassification"""
        current_hand_side = detection.hand_type
        prev_detected_handside = current_hand_side
        shift_multiplier = 1.5

        for retry_counter in range(0, 4):
            # Attempt hand detection
            bbox = self.landmark_ops.compute_shifted_bbox(roi.copy(), current_hand_side, shift_multiplier)
            cropped_img, bbox_origin = self.landmark_ops.crop_to_bbox(image.copy(), bbox)
            cropped_results, _ = self.image_ops.detect_hands_with_enhancement(cropped_img, detector_max_1)
            
            if self.config.verbose:
                full_vis = self.landmark_ops.draw_landmarks_and_bbox(image.copy(), None, bbox, roi, self.detector)
                cv2.imwrite(str(self.dirs['shifted_roi'] / f"{img_idx}_{self.config.bbox_shift}_{retry_counter}.jpg"), full_vis)

            if len(cropped_results) > 0:
                detected_detection = cropped_results[0]
                detected_hand_side = "right" if current_hand_side == "left" else "left"
                detected_detection.hand_type = detected_hand_side
                temp = data.copy()

                # Create final crop from original image using detected landmarks
                final_crop, final_origin = self.landmark_ops.crop_fixed_size(
                        image.copy(), detected_detection, 
                        base_offset=bbox_origin, 
                        current_image_size=cropped_img.shape[:2]
                )

                if self.config.verbose:
                    cv2.imwrite(str(self.dirs['preds'] / detected_hand_side / f"{img_idx}.jpg"), final_crop)

                if retry_counter >= 2:
                    temp["people"][0][f"hand_{current_hand_side}_keypoints_2d"] = data["people"][0][f"hand_{prev_detected_handside}_keypoints_2d"]
                    temp["people"][0][f"hand_{current_hand_side}_shift"] = data["people"][0][f"hand_{prev_detected_handside}_shift"]
                    temp["people"][0][f"hand_{current_hand_side}_conf"] = data["people"][0][f"hand_{prev_detected_handside}_conf"]
                    cv2.imwrite(str(self.dirs['cropped'] / current_hand_side / f"{img_idx}.jpg"), blank_img)
                    tmp_data, check = self.check_diff_hand(detected_detection, prev_detected_handside, current_hand_side, cropped_img, bbox_origin, final_origin, temp)
                else:    
                    tmp_data, check = self.check_diff_hand(detected_detection, current_hand_side, prev_detected_handside, cropped_img, bbox_origin, final_origin, temp)
                if check:
                    cv2.imwrite(str(self.dirs['cropped'] / detected_hand_side / f"{img_idx}.jpg"), final_crop)
                    return tmp_data

            if retry_counter % 2 == 0:
                shift_multiplier = 1.75
            else:
                shift_multiplier = 1.25
                prev_detected_handside = current_hand_side
                current_hand_side = "right" if current_hand_side == "left" else "left"
        
        return None
    
    def process_single_hand(self, image: np.ndarray, detection: HandDetection, 
                           img_idx: str, detector_max_1) -> Optional[Dict[str, Any]]:
        """Process image with single hand detection"""
        self.logger.info(f"Processing {img_idx}: {detection.hand_type} hand detected")
        
        # Initialize data structure
        data = {"people": [{"hand_left_shift": [], "hand_left_keypoints_2d": [], "hand_left_conf": [], 
                           "hand_right_shift": [], "hand_right_keypoints_2d": [], "hand_right_conf": []}]}
        
        # Initial crop from original image
        blank_img, crop_origin = self.landmark_ops.crop_fixed_size(image.copy(), detection)
        cv2.imwrite(str(self.dirs['cropped'] / str(detection.hand_type) / f"{img_idx}.jpg"), blank_img)
        
        # Get keypoints data
        keypoint_data = self.detector.get_keypoints_data(detection, image.shape[:2], [crop_origin])
        data["people"][0].update(keypoint_data)
        
        if self.config.verbose:
            # Create visualization with landmarks
            vis_img = image.copy()
            vis_img = self.detector.draw_landmarks(detection, vis_img)
        
            # Crop visualization
            cropped_vis, _ = self.landmark_ops.crop_fixed_size(vis_img, detection)
            cv2.imwrite(str(self.dirs['preds'] / str(detection.hand_type) / f"{img_idx}.jpg"), cropped_vis)
    
            cv2.imwrite(str(self.dirs['original'] / f"{img_idx}.jpg"), image)
        
        # Try shifted bbox approach with retry mechanism
        roi = self.landmark_ops.get_roi_points(detection, image.shape)
        bbox_img = self.landmark_ops.draw_bbox(image.copy(), roi)
        if self.config.verbose:
            cv2.imwrite(str(self.dirs['bboxes'] / str(detection.hand_type) / f"{img_idx}.jpg"), bbox_img)

        # try to detect second hand
        data = self.detect_second_hand_or_retry(img_idx, image.copy(), blank_img, detection, roi, data, detector_max_1)
        
        return data
    
    def process_double_hands(self, image: np.ndarray, detections: List[HandDetection], 
                           img_idx: str) -> Dict[str, Any]:
        """Process image with two hands detected"""
        self.logger.info(f"Processing {img_idx}: 2 hands detected")
        
        # Initialize data structure
        data = {"people": [{"hand_left_shift": [], "hand_left_keypoints_2d": [], "hand_left_conf": [],
                           "hand_right_shift": [], "hand_right_keypoints_2d": [], "hand_right_conf": []}]}
        
        if self.config.verbose:
            # Save original image
            cv2.imwrite(str(self.dirs['original'] / f"{img_idx}.jpg"), image)

        if detections[0].landmarks[0][0] < detections[1].landmarks[0][0]:
            detections[0].hand_type = "left"
            detections[1].hand_type = "right"
        else:
            detections[0].hand_type = "right"
            detections[1].hand_type = "left"

        # Process both hands
        for detection in detections:
            # Crop around each hand
            blank_img, crop_origin = self.landmark_ops.crop_fixed_size(image.copy(), detection)
            cv2.imwrite(str(self.dirs['cropped'] / str(detection.hand_type) / f"{img_idx}.jpg"), blank_img)

            # Get keypoints
            keypoint_data = self.detector.get_keypoints_data(detection, image.shape[:2], [crop_origin])
            data["people"][0].update(keypoint_data)

            if self.config.verbose:
                # Create visualization with landmarks
                vis_img = image.copy()
                vis_img = self.detector.draw_landmarks(detection, vis_img)
                
                # Crop visualization
                cropped_vis, _ = self.landmark_ops.crop_fixed_size(vis_img, detection)
                cv2.imwrite(str(self.dirs['preds'] / str(detection.hand_type) / f"{img_idx}.jpg"), cropped_vis)
        
        return data
