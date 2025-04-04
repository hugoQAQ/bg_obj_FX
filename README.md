# Background and Objectness Score Extraction for Detectron2

This module provides functionality to extract and visualize background scores and objectness scores from Detectron2's Faster R-CNN object detection models. 

- **Background scores** represent the model's confidence that a detected region is actually background rather than an object.
- **Objectness scores** are from the region proposal network and indicate the model's initial confidence that a region contains an object before classification.

## Contents

- `faster_rcnn_bg.py`: Contains the custom implementation to extract background scores from Faster R-CNN predictions
- `inference.py`: Sample script demonstrating how to use the score extraction functionality

## Usage

1. Import the module and use the `CustomPredictor` class to run inference:

```python
from bg_score_extraction.inference import CustomPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Configure your model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'  # or 'cuda' for GPU
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Create the custom predictor
predictor = CustomPredictor(cfg)

# Run inference
outputs = predictor(image)

# Access background scores
if hasattr(outputs["instances"], "bg_scores"):
    bg_scores = outputs["instances"].bg_scores
    # Use the background scores as needed

# Access objectness scores
if hasattr(outputs["instances"], "objectness_scores"):
    obj_scores = outputs["instances"].objectness_scores
    # Use the objectness scores as needed
```

## How It Works

The module extends Detectron2's standard object detection pipeline by:

1. Overriding the inference method in Faster R-CNN's output layers
2. Preserving the background scores during non-maximum suppression (NMS)
3. Adding the objectness scores from the region proposal network
4. Adding both scores to the output instances

This allows you to access the foreground class scores, background scores, and objectness scores for each detected object, which can be useful for:

- Filtering out uncertain detections
- Understanding model confidence at different stages of detection
- Debugging false positives
- Thresholding based on various score ratios
- Analyzing the performance of both the region proposal network and the final classifier

## Requirements

- detectron2
- PyTorch
- OpenCV (for visualization)
- NumPy

## Example Output

When running `inference.py`, the script will:

1. Load an image from "test.png"
2. Run object detection with score extraction
3. Visualize the results and save them to:
   - `output_with_bg_scores.jpg`: Standard detection visualization
   - `output_with_scores_annotations.jpg`: Visualization with background and objectness scores annotated

Additionally, it will print the scores for each detected object to the console. 