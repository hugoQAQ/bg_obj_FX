#!/usr/bin/env python
"""
Test script for the objectness score extraction functionality.
Tests both background and objectness score extraction.
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path

# Import Detectron2 components
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

# Import our custom predictor
from bg_score_extraction import CustomPredictor

def main():
    # Get the path to the test image
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    test_image_path = script_dir / "test.png"
    
    # If the image doesn't exist, use any available image
    if not test_image_path.exists():
        print(f"Warning: Test image not found at {test_image_path}")
        # Try to find another image
        import glob
        image_files = glob.glob(os.path.join(os.path.dirname(script_dir), "**", "*.jpg"), recursive=True)
        image_files += glob.glob(os.path.join(os.path.dirname(script_dir), "**", "*.png"), recursive=True)
        
        if not image_files:
            print("Error: No test images found. Please provide a test image.")
            return
        
        test_image_path = image_files[0]
        print(f"Using alternative image: {test_image_path}")
    
    # Load the image
    print(f"Loading image: {test_image_path}")
    image = cv2.imread(str(test_image_path))
    
    if image is None:
        print(f"Error: Failed to load image from {test_image_path}")
        return
    
    # Get configuration for the model
    print("Configuring the model...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Check for GPU availability and set the device accordingly
    if torch.cuda.is_available():
        print("Using GPU")
        cfg.MODEL.DEVICE = "cuda"
    else:
        print("Using CPU")
        cfg.MODEL.DEVICE = "cpu"
    
    # Set detection threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Create and run our custom predictor
    print("Creating predictor and running inference...")
    predictor = CustomPredictor(cfg)
    outputs = predictor(image)
    
    # Get instances and check for scores
    instances = outputs["instances"].to("cpu")
    
    # Check for available scores
    has_bg_scores = hasattr(instances, "bg_scores")
    has_obj_scores = hasattr(instances, "objectness_scores")
    
    print(f"\nTest Results:")
    print(f"Number of detections: {len(instances)}")
    print(f"Background scores available: {has_bg_scores}")
    print(f"Objectness scores available: {has_obj_scores}")
    
    # Display scores for each detection
    if len(instances) > 0:
        print("\nDetection Scores:")
        for i in range(len(instances)):
            class_id = instances.pred_classes[i].item()
            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_id]
            fg_score = instances.scores[i].item()
            
            print(f"\nDetection {i+1}: {class_name} (Class ID: {class_id})")
            print(f"  Foreground score: {fg_score:.4f}")
            
            if has_bg_scores:
                bg_score = instances.bg_scores[i].item()
                print(f"  Background score: {bg_score:.4f}")
                print(f"  FG/BG ratio: {fg_score / max(bg_score, 1e-6):.4f}")
            
            if has_obj_scores:
                obj_score = instances.objectness_scores[i].item()
                print(f"  Objectness score: {obj_score:.4f}")
                if has_bg_scores:
                    print(f"  OBJ/BG ratio: {obj_score / max(bg_score, 1e-6):.4f}")
    
    # Create visualization with annotations
    if len(instances) > 0:
        print("\nCreating visualization...")
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(instances)
        result_image = out.get_image()[:, :, ::-1]
        
        # Save the visualization
        output_path = script_dir / "test_objectness_output.jpg"
        cv2.imwrite(str(output_path), result_image)
        print(f"Visualization saved to: {output_path}")
        
        # Create a second visualization with score annotations
        if has_bg_scores or has_obj_scores:
            anno_image = result_image.copy()
            boxes = instances.pred_boxes.tensor.numpy()
            
            # Add score annotations to each box
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                text_parts = []
                if has_bg_scores:
                    bg_score = instances.bg_scores[i].item()
                    text_parts.append(f"BG:{bg_score:.3f}")
                
                if has_obj_scores:
                    obj_score = instances.objectness_scores[i].item()
                    text_parts.append(f"OBJ:{obj_score:.3f}")
                
                text = ", ".join(text_parts)
                
                # Add black background to text for better visibility
                (text_width, text_height), _ = cv2.getTextSize(text, font, 0.5, 1)
                cv2.rectangle(anno_image, (x1, y1-text_height-10), (x1+text_width, y1), (0, 0, 0), -1)
                # Add text
                cv2.putText(anno_image, text, (x1, y1-5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Save the annotated image
            anno_output_path = script_dir / "test_objectness_annotated.jpg"
            cv2.imwrite(str(anno_output_path), anno_image)
            print(f"Annotated image saved to: {anno_output_path}")
        
        # Display the result (skip in automated testing)
        # cv2.imshow("Detection Results", result_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 