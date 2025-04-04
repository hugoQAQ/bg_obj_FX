#!/usr/bin/env python
"""
Simple example showing how to use the background score extraction functionality.

This minimal example shows the basic usage of the CustomPredictor.
"""
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Import our custom predictor
from bg_score_extraction import CustomPredictor

# Load an image
image_path = "test.png"  # Replace with your image path
image = cv2.imread(image_path)

# Configure the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"  # Use CPU for inference (change to "cuda" for GPU)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model

# Create our custom predictor
predictor = CustomPredictor(cfg)

# Run inference
outputs = predictor(image)

# Extract background scores and objectness scores
instances = outputs["instances"]
has_bg_scores = hasattr(instances, "bg_scores")
has_obj_scores = hasattr(instances, "objectness_scores")

if has_bg_scores:
    # Print classes, foreground scores, and background scores
    for i, (cls, fg_score) in enumerate(zip(
            instances.pred_classes,
            instances.scores)):
        
        print(f"Detection {i+1}:")
        print(f"  Class ID: {cls.item()}")
        print(f"  Foreground score: {fg_score.item():.3f}")
        
        if has_bg_scores:
            bg_score = instances.bg_scores[i].item()
            print(f"  Background score: {bg_score:.3f}")
            print(f"  FG/BG ratio: {fg_score.item() / max(bg_score, 1e-6):.3f}")
        
        if has_obj_scores:
            obj_score = instances.objectness_scores[i].item()
            print(f"  Objectness score: {obj_score:.3f}")
        
        print()
else:
    print("Scores not available")

# Visualize the results
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(instances.to("cpu"))
cv2.imshow("Detections", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows() 