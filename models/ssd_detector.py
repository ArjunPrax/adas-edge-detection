"""
SSD300 model wrapper for ADAS object detection.
Supports CPU, MPS (Apple Silicon), and CUDA devices.
"""

import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import functional as F
from PIL import Image

# COCO class labels (91 classes, 0 = background)
COCO_LABELS = {
    0: '__background__',
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}

# ADAS-relevant classes
ADAS_CLASSES = {1: 'person', 3: 'car', 6: 'bus', 8: 'truck'}

# Color palette per class (BGR for OpenCV, RGB for PIL)
CLASS_COLORS = {
    1: (255, 87, 87),    # person — red
    3: (87, 178, 255),   # car — blue
    6: (255, 200, 87),   # bus — orange
    8: (87, 255, 159),   # truck — green
}
DEFAULT_COLOR = (200, 200, 200)


def get_device(prefer: str = "auto") -> torch.device:
    """
    Resolve the best available device.
    prefer='auto': MPS > CUDA > CPU
    prefer='cpu': always CPU (for ZCU104 ARM)
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" or (prefer == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


class SSDDetector:
    """Wrapper around torchvision SSD300-VGG16 for ADAS detection."""

    def __init__(self):
        self.model = None
        self.device = None

    def load_model(self, device: str = "auto") -> torch.device:
        """
        Load SSD300-VGG16 with pretrained COCO weights.

        Args:
            device: 'auto', 'cpu', 'mps', or 'cuda'

        Returns:
            torch.device that the model is loaded on
        """
        self.device = get_device(device)
        print(f"Loading SSD300-VGG16 on {self.device} ...")
        weights = SSD300_VGG16_Weights.DEFAULT
        self.model = ssd300_vgg16(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")
        return self.device

    def detect(self, image, threshold: float = 0.5):
        """
        Run detection on a single image.

        Args:
            image: PIL.Image or torch.Tensor (C, H, W) in [0, 1]
            threshold: confidence score threshold

        Returns:
            boxes  (N, 4) float tensor — [x1, y1, x2, y2] in pixel coords
            labels (N,)   int tensor   — COCO class IDs
            scores (N,)   float tensor — confidence scores
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before detect()")

        if isinstance(image, Image.Image):
            image = F.to_tensor(image)  # (C, H, W) float32 in [0, 1]

        image = image.to(self.device)

        with torch.no_grad():
            predictions = self.model([image])

        pred = predictions[0]
        boxes = pred['boxes'].cpu()
        labels = pred['labels'].cpu()
        scores = pred['scores'].cpu()

        # Filter by threshold
        keep = scores >= threshold
        return boxes[keep], labels[keep], scores[keep]

    def detect_batch(self, images, threshold: float = 0.5):
        """
        Run detection on a list of PIL images or tensors.

        Returns:
            list of (boxes, labels, scores) tuples
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before detect_batch()")

        tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                img = F.to_tensor(img)
            tensors.append(img.to(self.device))

        with torch.no_grad():
            predictions = self.model(tensors)

        results = []
        for pred in predictions:
            boxes = pred['boxes'].cpu()
            labels = pred['labels'].cpu()
            scores = pred['scores'].cpu()
            keep = scores >= threshold
            results.append((boxes[keep], labels[keep], scores[keep]))
        return results


def label_name(class_id: int) -> str:
    """Return human-readable class name for a COCO class ID."""
    return COCO_LABELS.get(int(class_id), f"class_{class_id}")


def label_color(class_id: int) -> tuple:
    """Return RGB color tuple for a class ID."""
    return CLASS_COLORS.get(int(class_id), DEFAULT_COLOR)
