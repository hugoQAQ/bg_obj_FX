import os
import sys
import cv2
import numpy as np
import torch

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary modules
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Import our custom module for background scores
from .faster_rcnn_bg import FastRCNNOutputLayersWithBG, fast_rcnn_inference_with_bg

# Custom predictor that uses our modified FastRCNN output layers
class CustomPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Keep a reference to the original box predictor
        self.original_box_predictor = self.model.roi_heads.box_predictor
        
        # Override the inference method of ROI heads to use our custom function
        self.model.roi_heads.box_predictor.inference = self._custom_inference
        
    def _custom_inference(self, predictions, proposals):
        """Custom inference method that preserves background scores and objectness scores."""
        boxes = self.original_box_predictor.predict_boxes(predictions, proposals)
        scores = self.original_box_predictor.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        
        # Get results with background scores
        results, kept_indices = fast_rcnn_inference_with_bg(
            boxes,
            scores,
            image_shapes,
            self.original_box_predictor.test_score_thresh,
            self.original_box_predictor.test_nms_thresh,
            self.original_box_predictor.test_topk_per_image,
        )
        
        # Add objectness scores from proposals to the results
        for i, (result, proposal, indices) in enumerate(zip(results, proposals, kept_indices)):
            if len(indices) > 0 and proposal.has("objectness_logits"):
                # Get objectness_logits for the kept proposals
                objectness_logits = proposal.objectness_logits[indices]
                # Convert logits to probabilities using sigmoid
                objectness_scores = torch.sigmoid(objectness_logits)
                result.objectness_scores = objectness_scores
        
        return results, kept_indices

# Example usage
def main():
    # Load image
    input_image_path = "test.png"
    print(f"Loading image from {input_image_path}")
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        sys.exit(1)

    # Get a configuration for an instance segmentation model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'  # Use CPU for inference
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model

    # Create predictor with our custom class
    print("Creating predictor...")
    predictor = CustomPredictor(cfg)

    # Run prediction
    print("Running inference...")
    outputs = predictor(image)

    # Visualize prediction
    print("Visualizing results...")
    v = Visualizer(image[:, :, ::-1], 
                MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = out.get_image()

    # Print out background scores
    print("Scores for detected objects:")
    if hasattr(outputs["instances"], "bg_scores"):
        has_objectness = hasattr(outputs["instances"], "objectness_scores")
        for i, (bg_score, fg_score, cls) in enumerate(zip(
                outputs["instances"].bg_scores, 
                outputs["instances"].scores, 
                outputs["instances"].pred_classes)):
            obj_score_str = ""
            if has_objectness:
                obj_score = outputs["instances"].objectness_scores[i].item()
                obj_score_str = f", objectness_score={obj_score:.3f}"
            print(f"Object {i+1}: class={cls.item()}, foreground_score={fg_score.item():.3f}, background_score={bg_score.item():.3f}{obj_score_str}")
    else:
        print("Background scores not available in the output")

    # Save the visualization result
    output_path = "output_with_bg_scores.jpg"
    cv2.imwrite(output_path, result_image[:, :, ::-1])
    print(f"Result saved to {output_path}")

    # Also save an image with background scores annotated
    if hasattr(outputs["instances"], "bg_scores"):
        # Create a copy of the visualization
        anno_image = result_image.copy()
        
        # Get the boxes and associated scores
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        bg_scores = outputs["instances"].bg_scores.cpu().numpy()
        has_objectness = hasattr(outputs["instances"], "objectness_scores")
        if has_objectness:
            obj_scores = outputs["instances"].objectness_scores.cpu().numpy()
        
        # Add background score annotations to each box
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (box, bg_score) in enumerate(zip(boxes, bg_scores)):
            x1, y1, x2, y2 = map(int, box)
            if has_objectness:
                obj_score = obj_scores[i]
                text = f"BG: {bg_score:.3f}, OBJ: {obj_score:.3f}"
            else:
                text = f"BG: {bg_score:.3f}"
            cv2.putText(anno_image, text, (x1, y1-10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        # Save the annotated image
        bg_output_path = "output_with_scores_annotations.jpg"
        cv2.imwrite(bg_output_path, anno_image[:, :, ::-1])
        print(f"Annotated image saved to {bg_output_path}")

if __name__ == "__main__":
    main()