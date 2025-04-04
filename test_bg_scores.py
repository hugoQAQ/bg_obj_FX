#!/usr/bin/env python
"""
Test script to verify that background score extraction works correctly.
"""
import unittest
import torch
import numpy as np
from detectron2.structures import Boxes, Instances

# Fix import to work when run directly
try:
    from .faster_rcnn_bg import fast_rcnn_inference_single_image_with_bg
except ImportError:
    from faster_rcnn_bg import fast_rcnn_inference_single_image_with_bg

class TestBackgroundScoreExtraction(unittest.TestCase):
    def test_background_score_preservation(self):
        """Test that background scores are correctly preserved."""
        # Create test data
        boxes = torch.tensor([
            [0, 0, 10, 10],  # Box 1
            [5, 5, 15, 15],  # Box 2
            [20, 20, 30, 30]  # Box 3
        ], dtype=torch.float32)
        
        # Repeat boxes for each class (assuming 3 classes)
        num_classes = 3
        boxes = boxes.repeat(1, num_classes).view(-1, 4)
        
        # Create scores: [foreground_classes, background]
        scores = torch.tensor([
            [0.8, 0.1, 0.05, 0.05],  # Object 1 scores for classes 0,1,2 + background
            [0.2, 0.7, 0.05, 0.05],  # Object 2 scores for classes 0,1,2 + background
            [0.1, 0.1, 0.3, 0.5]     # Object 3 scores for classes 0,1,2 + background
        ], dtype=torch.float32)
        
        # Run inference
        image_shape = (100, 100)
        score_thresh = 0.3
        nms_thresh = 0.5
        topk_per_image = 100
        
        result, _ = fast_rcnn_inference_single_image_with_bg(
            boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        
        # Verify results
        self.assertTrue(hasattr(result, "bg_scores"), "Background scores attribute missing")
        self.assertEqual(len(result.bg_scores), 3, "Should have 3 background scores")
        
        # Check expected classes (highest scoring class for each object)
        expected_classes = torch.tensor([0, 1, 2])
        self.assertTrue(torch.all(result.pred_classes == expected_classes), 
                       f"Expected classes {expected_classes}, got {result.pred_classes}")
        
        # Check background scores
        expected_bg_scores = torch.tensor([0.05, 0.05, 0.5])
        self.assertTrue(torch.allclose(result.bg_scores, expected_bg_scores), 
                       f"Expected bg scores {expected_bg_scores}, got {result.bg_scores}")
        
        print("All tests passed!")

if __name__ == "__main__":
    unittest.main() 