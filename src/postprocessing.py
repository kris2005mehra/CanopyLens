"""
Post-processing of tree crown detections for CanopyLens.

Handles confidence filtering, vegetation masking using the Excess Green Index (ExG),
and individual crown mask creation from bounding box detections.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def filter_detections(
    detections: pd.DataFrame,
    confidence_threshold: float = 0.3,
    min_box_pixels: int = 100,
) -> pd.DataFrame:
    """
    Filter detections by confidence score and minimum box area.

    Args:
        detections: DataFrame with xmin, ymin, xmax, ymax, label, score columns.
        confidence_threshold: Minimum confidence to keep.
        min_box_pixels: Minimum bounding box area in pixels.

    Returns:
        Filtered DataFrame.
    """
    if len(detections) == 0:
        return detections

    filtered = detections[detections["score"] >= confidence_threshold].copy()

    # Calculate box areas
    box_areas = (filtered["xmax"] - filtered["xmin"]) * (
        filtered["ymax"] - filtered["ymin"]
    )
    filtered = filtered[box_areas >= min_box_pixels]

    return filtered.reset_index(drop=True)


def compute_excess_green(image_rgb: np.ndarray) -> np.ndarray:
    """
    Compute the Excess Green Index (ExG) for vegetation detection.

    ExG = 2*g - r - b  (using normalized RGB channels)

    This index highlights green vegetation against soil, shadow, and other backgrounds.
    It is a well-established vegetation index in precision agriculture and forestry.

    Args:
        image_rgb: RGB numpy array (H, W, 3), uint8.

    Returns:
        ExG array (H, W), float64, values typically in [-1, 1].
    """
    r = image_rgb[:, :, 0].astype(np.float64)
    g = image_rgb[:, :, 1].astype(np.float64)
    b = image_rgb[:, :, 2].astype(np.float64)

    # Normalize to [0, 1] range
    total = r + g + b + 1e-10  # Avoid division by zero
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total

    # Excess Green Index
    exg = 2.0 * g_norm - r_norm - b_norm

    return exg


def create_canopy_mask(
    image_rgb: np.ndarray,
    detections: pd.DataFrame,
    exg_min_threshold: float = 0.05,
) -> np.ndarray:
    """
    Create a combined canopy mask using vegetation thresholding within detected boxes.

    For each detected bounding box:
    1. Extract the region's ExG values.
    2. Apply Otsu thresholding (or fixed threshold if Otsu fails).
    3. Mark green pixels as canopy.

    This approach gives pixel-level canopy area estimates without needing
    a separate segmentation model.

    Args:
        image_rgb: RGB numpy array.
        detections: DataFrame of tree detections.
        exg_min_threshold: Minimum ExG threshold (prevents all-canopy masks).

    Returns:
        Boolean mask (H, W) where True = canopy pixel.
    """
    h, w = image_rgb.shape[:2]
    canopy_mask = np.zeros((h, w), dtype=bool)

    if len(detections) == 0:
        return canopy_mask

    # Pre-compute ExG for the entire image (efficient — computed once)
    exg = compute_excess_green(image_rgb)

    for _, det in detections.iterrows():
        x1 = max(0, int(det["xmin"]))
        y1 = max(0, int(det["ymin"]))
        x2 = min(w, int(det["xmax"]))
        y2 = min(h, int(det["ymax"]))

        if x2 <= x1 or y2 <= y1:
            continue

        # Extract ExG for this bounding box region
        box_exg = exg[y1:y2, x1:x2]

        # Determine threshold using Otsu's method
        threshold = _otsu_threshold(box_exg, exg_min_threshold)

        # Create mask for this box
        box_mask = box_exg > threshold
        canopy_mask[y1:y2, x1:x2] |= box_mask

    return canopy_mask


def create_crown_masks(
    image_rgb: np.ndarray,
    detections: pd.DataFrame,
    exg_min_threshold: float = 0.05,
) -> List[np.ndarray]:
    """
    Create individual boolean masks for each detected crown.

    Each mask is a (H, W) boolean array where True indicates pixels
    belonging to that specific crown.

    Args:
        image_rgb: RGB numpy array.
        detections: DataFrame of tree detections.
        exg_min_threshold: Minimum ExG threshold.

    Returns:
        List of boolean masks, one per detection.
    """
    h, w = image_rgb.shape[:2]
    masks = []

    if len(detections) == 0:
        return masks

    exg = compute_excess_green(image_rgb)

    for _, det in detections.iterrows():
        x1 = max(0, int(det["xmin"]))
        y1 = max(0, int(det["ymin"]))
        x2 = min(w, int(det["xmax"]))
        y2 = min(h, int(det["ymax"]))

        crown_mask = np.zeros((h, w), dtype=bool)

        if x2 <= x1 or y2 <= y1:
            masks.append(crown_mask)
            continue

        box_exg = exg[y1:y2, x1:x2]
        threshold = _otsu_threshold(box_exg, exg_min_threshold)
        box_mask = box_exg > threshold

        # If the mask is empty (no green pixels), use the full box as fallback
        if not box_mask.any():
            box_mask = np.ones_like(box_mask, dtype=bool)

        crown_mask[y1:y2, x1:x2] = box_mask
        masks.append(crown_mask)

    return masks


def _otsu_threshold(values: np.ndarray, min_threshold: float = 0.05) -> float:
    """
    Compute Otsu threshold for ExG values within a bounding box.

    Falls back to a fixed minimum threshold if Otsu fails
    (e.g., uniform region).
    """
    try:
        from skimage.filters import threshold_otsu

        # Flatten and remove NaN/inf
        flat = values.ravel()
        flat = flat[np.isfinite(flat)]

        if len(flat) < 10:
            return min_threshold

        thresh = threshold_otsu(flat)
        # Ensure we don't use a threshold below the minimum
        return max(thresh, min_threshold)

    except (ValueError, ImportError):
        # Otsu fails on uniform images or if scikit-image is missing
        return min_threshold
