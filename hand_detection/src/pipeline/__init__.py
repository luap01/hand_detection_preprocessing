"""
Hand Detection Pipeline Package

This package contains modular components for hand detection and preprocessing:
- config: Configuration classes and enums
- image_ops: Image loading, enhancement, and rotation operations
- landmark_ops: ROI, bbox, and cropping operations  
- processing: Hand detection and processing logic
- pipeline: Main pipeline class and execution logic
"""

from .config import PipelineConfig, CameraShiftConfig, ShiftDirection
from .pipeline import HandDetectionPipeline
from .image_ops import ImageOperations
from .landmark_ops import LandmarkOperations
from .processing import HandProcessor

__all__ = [
    'PipelineConfig',
    'CameraShiftConfig', 
    'ShiftDirection',
    'HandDetectionPipeline',
    'ImageOperations',
    'LandmarkOperations',
    'HandProcessor'
] 