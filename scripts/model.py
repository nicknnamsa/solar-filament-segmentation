"""Mask R-CNN model construction, shared by train / evaluate / predict."""
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(cfg):
    m = cfg["model"]
    model = maskrcnn_resnet50_fpn(
        weights="DEFAULT" if m["pretrained"] else None,
        weights_backbone=None if not m["pretrained"] else "DEFAULT",
        min_size=m["min_size"],
        max_size=m["max_size"],
        box_detections_per_img=m["max_detections"],
        # torchvision defaults: rpn 0.7/0.3, box 0.5/0.5. A region scoring between these two thresholds is neither
        # a confirmed match nor a confirmed non-match and drops out of the loss entirely — widening the gap is a
        # "don't punish near-miss regions" margin around each annotated filament. The box head has NO such margin
        # by default (0.5 == 0.5), so anything short of a solid overlap is currently forced to count as background.
        rpn_fg_iou_thresh=m.get("rpn_fg_iou_thresh", 0.7),
        rpn_bg_iou_thresh=m.get("rpn_bg_iou_thresh", 0.3),
        box_fg_iou_thresh=m.get("box_fg_iou_thresh", 0.5),
        box_bg_iou_thresh=m.get("box_bg_iou_thresh", 0.5),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, m["num_classes"])
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, m["num_classes"])
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
