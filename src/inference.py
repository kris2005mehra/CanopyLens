"""
Tree crown detection using DeepForest.

DeepForest is a pretrained RetinaNet (ResNet-50 backbone) model specifically designed
for individual tree crown detection from aerial RGB imagery. It was trained on data
from the National Ecological Observatory Network (NEON).

Reference: Weinstein et al., "DeepForest: A Python package for RGB deep learning
tree crown delineation" (Methods in Ecology and Evolution, 2020).
License: MIT
"""

import numpy as np
import pandas as pd
import os
import tempfile
import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

# Module-level model cache
_model = None


def load_model():
    """
    Load and cache the DeepForest model.

    Downloads pretrained weights on first call (~140 MB).
    Subsequent calls return the cached model.
    """
    global _model
    if _model is not None:
        return _model

    logger.info("Loading DeepForest model...")
    try:
        from deepforest import main

        _model = main.deepforest()
        _model.use_release()
        logger.info("DeepForest model loaded successfully.")
        return _model
    except Exception as e:
        logger.error(f"Failed to load DeepForest model: {e}")
        raise RuntimeError(
            f"Could not load the DeepForest tree detection model: {e}. "
            "This may be due to a network issue or missing dependencies."
        )


def detect_trees(
    image_rgb: np.ndarray,
    confidence_threshold: float = 0.3,
    patch_size: int = 800,
    patch_overlap: int = 100,
) -> pd.DataFrame:
    """
    Detect individual tree crowns in an image.

    For small images (< ~1200px), runs prediction directly.
    For larger images, tiles the image and merges results with NMS.

    Args:
        image_rgb: RGB numpy array (H, W, 3), uint8.
        confidence_threshold: Minimum confidence score to keep a detection.
        patch_size: Tile size for large images (pixels).
        patch_overlap: Overlap between tiles (pixels).

    Returns:
        DataFrame with columns: xmin, ymin, xmax, ymax, label, score
    """
    model = load_model()
    h, w = image_rgb.shape[:2]

    empty_df = pd.DataFrame(
        columns=["xmin", "ymin", "xmax", "ymax", "label", "score"]
    )

    # Choose direct prediction or tiled prediction
    if h <= patch_size * 1.5 and w <= patch_size * 1.5:
        detections = _predict_single(model, image_rgb)
    else:
        logger.info(
            f"Image is large ({w}×{h}). Using tiled prediction "
            f"(patch={patch_size}, overlap={patch_overlap})."
        )
        detections = _predict_tiled(
            model, image_rgb, patch_size, patch_overlap
        )

    if detections is None or len(detections) == 0:
        return empty_df

    # Filter by confidence
    detections = detections[detections["score"] >= confidence_threshold].copy()

    # Clip coordinates to image bounds
    detections["xmin"] = detections["xmin"].clip(lower=0)
    detections["ymin"] = detections["ymin"].clip(lower=0)
    detections["xmax"] = detections["xmax"].clip(upper=w)
    detections["ymax"] = detections["ymax"].clip(upper=h)

    # Remove degenerate boxes (zero or negative area)
    valid = (detections["xmax"] > detections["xmin"]) & (
        detections["ymax"] > detections["ymin"]
    )
    detections = detections[valid].reset_index(drop=True)

    return detections


def _predict_single(model, image_rgb: np.ndarray) -> Optional[pd.DataFrame]:
    """Run DeepForest prediction on a single image via temp file."""
    temp_path = os.path.join(tempfile.gettempdir(), "canopylens_input.png")
    try:
        Image.fromarray(image_rgb).save(temp_path)
        predictions = model.predict_image(path=temp_path, return_plot=False)
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _predict_tiled(
    model,
    image_rgb: np.ndarray,
    patch_size: int = 800,
    overlap: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Tile a large image and run predictions on each tile,
    then merge results with Non-Maximum Suppression (NMS).
    """
    h, w = image_rgb.shape[:2]
    stride = patch_size - overlap
    all_detections = []

    for y_start in range(0, h, stride):
        for x_start in range(0, w, stride):
            y_end = min(y_start + patch_size, h)
            x_end = min(x_start + patch_size, w)

            patch = image_rgb[y_start:y_end, x_start:x_end]
            ph, pw = patch.shape[:2]

            # Pad small edge patches so the model gets a consistent input size
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            # Predict on this tile
            preds = _predict_single(model, patch)

            if preds is not None and len(preds) > 0:
                preds = preds.copy()

                # Remove detections in padded area
                preds = preds[
                    (preds["xmax"] <= pw) & (preds["ymax"] <= ph)
                ]

                # Translate coordinates to full-image space
                preds["xmin"] += x_start
                preds["ymin"] += y_start
                preds["xmax"] += x_start
                preds["ymax"] += y_start

                all_detections.append(preds)

    if not all_detections:
        return None

    merged = pd.concat(all_detections, ignore_index=True)

    # Apply NMS to remove duplicate detections from overlapping tiles
    merged = _apply_nms(merged, iou_threshold=0.3)

    return merged


def _apply_nms(
    detections: pd.DataFrame, iou_threshold: float = 0.3
) -> pd.DataFrame:
    """Apply Non-Maximum Suppression to remove duplicate/overlapping detections."""
    if len(detections) == 0:
        return detections

    try:
        import torch
        from torchvision.ops import nms

        boxes = torch.tensor(
            detections[["xmin", "ymin", "xmax", "ymax"]].values,
            dtype=torch.float32,
        )
        scores = torch.tensor(detections["score"].values, dtype=torch.float32)
        keep_indices = nms(boxes, scores, iou_threshold)
        return detections.iloc[keep_indices.numpy()].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"torchvision NMS failed ({e}), using fallback NMS.")
        return _nms_fallback(detections, iou_threshold)


def _nms_fallback(
    detections: pd.DataFrame, iou_threshold: float
) -> pd.DataFrame:
    """Simple Python NMS fallback (greedy, O(n²))."""
    if len(detections) == 0:
        return detections

    dets = detections.sort_values("score", ascending=False).reset_index(drop=True)
    keep = []

    for i in range(len(dets)):
        should_keep = True
        for j in keep:
            if _iou(dets.iloc[i], dets.iloc[j]) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(i)

    return dets.iloc[keep].reset_index(drop=True)


def _iou(det_a, det_b) -> float:
    """Calculate Intersection over Union between two detections."""
    x1 = max(det_a["xmin"], det_b["xmin"])
    y1 = max(det_a["ymin"], det_b["ymin"])
    x2 = min(det_a["xmax"], det_b["xmax"])
    y2 = min(det_a["ymax"], det_b["ymax"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (det_a["xmax"] - det_a["xmin"]) * (det_a["ymax"] - det_a["ymin"])
    area_b = (det_b["xmax"] - det_b["xmin"]) * (det_b["ymax"] - det_b["ymin"])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0
