#!/usr/bin/env python
"""
Demo script for background score extraction.

Usage:
    python -m bg_score_extraction.demo --input /path/to/image.jpg [--threshold 0.5] [--device cpu]
"""
import argparse
import os
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from .inference import CustomPredictor

def parse_args():
    parser = argparse.ArgumentParser(description="Background Score Extraction Demo")
    parser.add_argument(
        "--input", required=True, help="Path to input image or directory of images"
    )
    parser.add_argument(
        "--output", 
        default="output", 
        help="Directory to save output visualizations"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5, 
        help="Detection score threshold"
    )
    parser.add_argument(
        "--device", 
        default="cpu", 
        help="Device to run inference on (cpu or cuda)"
    )
    parser.add_argument(
        "--model", 
        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        help="Model to use for inference"
    )
    
    return parser.parse_args()

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    return cfg

def main():
    args = parse_args()
    setup_logger()
    
    # Set up model
    cfg = setup_cfg(args)
    predictor = CustomPredictor(cfg)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process input (single image or directory)
    if os.path.isdir(args.input):
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        image_paths = [args.input]
    
    # Process each image
    for image_path in image_paths:
        print(f"Processing {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        # Run inference
        outputs = predictor(image)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Visualize prediction
        v = Visualizer(image[:, :, ::-1], 
                      MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                      scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_image = out.get_image()
        
        # Save standard visualization
        output_path = os.path.join(args.output, f"{base_name}_detections.jpg")
        cv2.imwrite(output_path, result_image[:, :, ::-1])
        
        # Print background scores
        if hasattr(outputs["instances"], "bg_scores"):
            print("Background scores:")
            for i, (bg_score, fg_score, cls) in enumerate(zip(
                    outputs["instances"].bg_scores, 
                    outputs["instances"].scores, 
                    outputs["instances"].pred_classes)):
                print(f"  Object {i+1}: class={cls.item()}, fg_score={fg_score.item():.3f}, bg_score={bg_score.item():.3f}")
            
            # Save visualization with background scores
            anno_image = result_image.copy()
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            bg_scores = outputs["instances"].bg_scores.cpu().numpy()
            
            # Add background score annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, (box, bg_score) in enumerate(zip(boxes, bg_scores)):
                x1, y1, x2, y2 = map(int, box)
                text = f"BG: {bg_score:.3f}"
                cv2.putText(anno_image, text, (x1, y1-10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
            # Save annotated image
            bg_output_path = os.path.join(args.output, f"{base_name}_bg_scores.jpg")
            cv2.imwrite(bg_output_path, anno_image[:, :, ::-1])
        else:
            print("Background scores not available")
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 