import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

from models.mediapipe_detector import MediaPipeHandDetector
from models.openpose_detector import OpenPoseHandDetector

from .config import PipelineConfig
from .image_ops import ImageOperations
from .landmark_ops import LandmarkOperations
from .processing import HandProcessor


class HandDetectionPipeline:
    """Clean pipeline for hand detection and cropping"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_output_dirs()
        self.setup_logging()
        
        # Initialize operations modules
        self.image_ops = ImageOperations(config)
        self.landmark_ops = LandmarkOperations(config)
        
        # Initialize detectors
        self.setup_detectors()
        
        # Initialize processor
        self.processor = HandProcessor(
            config, self.detector, self.image_ops, 
            self.landmark_ops, self.logger, self.dirs
        )
        
        # Thread-safe counters
        self.processed_count = 0
        self.failed_count = 0
        self.partial_count = 0
        self.lock = Lock()
    
    def setup_detectors(self):
        """Initialize hand detection models"""
        if self.config.model == "mediapipe":
            self.detector_max_2 = MediaPipeHandDetector(
                max_num_hands=2,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
            self.detector_max_1 = MediaPipeHandDetector(
                max_num_hands=1,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
            self.detector = self.detector_max_2 if self.config.max_num_hands == 2 else self.detector_max_1
        elif self.config.model == "openpose":
            self.detector = OpenPoseHandDetector(
                max_num_hands=self.config.max_num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
        else:
            raise ValueError(f"{self.config.model} implementation not existant...")
    
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
    
    def process_image_thread_safe(self, img_idx: str) -> bool:
        """Thread-safe version of process_image"""
        # Construct image path
        img_path = f"{self.config.input_path}/{self.config.image_prefix}{img_idx}{self.config.image_suffix}"
        
        # Load and undistort image
        image = self.image_ops.load_and_undistort_image(img_path)
        if image is None:
            return False
        
        # Detect hands
        detections, _ = self.image_ops.detect_hands_with_enhancement(image, self.detector)
        if len(detections) == 0:
            self.logger.info(f"No hands detected in {img_idx}")
            return False
        
        # Process based on number of hands
        if len(detections) == 2:
            data = self.processor.process_double_hands(image, detections, img_idx)
        elif len(detections) == 1:
            data = self.processor.process_single_hand(image, detections[0], img_idx, self.detector_max_1)
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
        image = self.image_ops.load_and_undistort_image(img_path)
        if image is None:
            return False
        
        # Detect hands
        results, angle = self.image_ops.detect_hands_with_enhancement(image, self.detector)
        if len(results) == 0:
            self.logger.info(f"No hands detected in {img_idx}")
            return False
        
        if angle != 0:
            self.logger.debug(f"Hands detected in {img_idx} using {angle}° rotation")
        
        # Process based on number of hands
        if len(results) == 2:
            data = self.processor.process_double_hands(image, results, img_idx)
        elif len(results) > 0:
            data = self.processor.process_single_hand(image, results[0], img_idx, self.detector_max_1)
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
