"""
YOLO Species Detector
Loads best.pt once at import time; exposes detect_species(image_path) → list[str]
"""

import os
import torch
from pathlib import Path
from typing import List

# ── Model path ───────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).parent / "models" / "best_repacked.pt"

# ── Lazy-load the model once ─────────────────────────────────────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e

        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {_MODEL_PATH}. "
                "Place best_repacked.pt inside backend/models/"
            )

        # Patch torch.load to allow unsafe weights (required for custom .pt)
        _orig = torch.load
        torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

        print(f"[YOLO] Loading model from {_MODEL_PATH} …")
        _model = YOLO(str(_MODEL_PATH))
        print("[YOLO] Model loaded ✓")

        torch.load = _orig  # restore original

    return _model


def detect_species(image_path: str, conf_threshold: float = 0.25) -> List[str]:
    """
    Run YOLOv8 inference on *image_path*.

    Returns a deduplicated list of detected class-name strings,
    ordered by descending confidence.  Returns [] if nothing detected.
    """
    model = _get_model()

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        verbose=False,
    )

    labels: List[str] = []
    seen: set = set()

    for result in results:
        # Sort boxes by confidence descending
        if result.boxes is None:
            continue

        boxes = result.boxes
        # conf is a tensor; convert to list for sorting
        conf_list = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else list(boxes.conf)
        cls_list  = boxes.cls.tolist()  if hasattr(boxes.cls,  "tolist") else list(boxes.cls)

        # Pair and sort by confidence descending
        pairs = sorted(zip(conf_list, cls_list), key=lambda x: x[0], reverse=True)

        for conf, cls_id in pairs:
            class_name: str = result.names[int(cls_id)]
            if class_name not in seen:
                seen.add(class_name)
                labels.append(class_name)

    return labels
