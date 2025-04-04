# faster_rcnn_bg.py
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

def fast_rcnn_inference_with_bg(
    boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    """
    Modified version of fast_rcnn_inference that preserves background scores.
    """
    result_per_image = [
        fast_rcnn_inference_single_image_with_bg(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

def fast_rcnn_inference_single_image_with_bg(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference with background scores preserved.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    # Save background scores
    bg_scores = scores[:, -1].clone()
    
    # Process foreground scores as usual
    scores_fg = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores_fg > score_thresh  # R x K
    filter_inds = filter_mask.nonzero()
    original_indices = filter_inds[:, 0]
    
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    
    scores_fg = scores_fg[filter_mask]

    # Apply NMS
    keep = batched_nms(boxes, scores_fg, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    
    # Build result
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes[keep])
    result.scores = scores_fg[keep]
    result.pred_classes = filter_inds[keep, 1]
    
    # Add background scores
    result.bg_scores = bg_scores[original_indices][keep]
    
    return result, keep

class FastRCNNOutputLayersWithBG(FastRCNNOutputLayers):
    """
    FastRCNNOutputLayers that preserves background scores.
    """
    
    def __init__(
        self, 
        box2box_transform, 
        input_shape, 
        num_classes, 
        test_score_thresh=0.0, 
        test_nms_thresh=0.5, 
        test_topk_per_image=100,
        cls_agnostic_bbox_reg=False, 
        smooth_l1_beta=0.0, 
        box_reg_loss_type="smooth_l1",
        loss_weight=1.0
    ):
        super().__init__(
            box2box_transform,
            input_shape,
            num_classes,
            test_score_thresh,
            test_nms_thresh,
            test_topk_per_image,
            cls_agnostic_bbox_reg,
            smooth_l1_beta,
            box_reg_loss_type,
            loss_weight
        )
    
    def inference(self, predictions, proposals):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        
        # Use our custom inference function
        return fast_rcnn_inference_with_bg(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )