"""
Background and Objectness Score Extraction for Detectron2

This module provides functionality to extract background scores and objectness scores from 
Detectron2's object detection models.
"""

from .faster_rcnn_bg import (
    FastRCNNOutputLayersWithBG,
    fast_rcnn_inference_with_bg,
    fast_rcnn_inference_single_image_with_bg
)

from .inference import (
    CustomPredictor
)

__all__ = [
    'FastRCNNOutputLayersWithBG',
    'fast_rcnn_inference_with_bg',
    'fast_rcnn_inference_single_image_with_bg',
    'CustomPredictor',
]

__version__ = '0.1.1' 