"""
Visualization of tree crown detections for CanopyLens.

Creates annotated images with semi-transparent crown masks, bounding boxes,
and side-by-side comparisons for the UI.
"""

import numpy as np
import cv2
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Color palette for individual crowns (BGR format for OpenCV)
# Using distinct, visually pleasing colors
CROWN_COLORS = [
    (76, 175, 80),    # Green
    (33, 150, 243),   # Blue
    (255, 193, 7),    # Amber
    (156, 39, 176),   # Purple
    (255, 87, 34),    # Deep Orange
    (0, 188, 212),    # Cyan
    (233, 30, 99),    # Pink
    (139, 195, 74),   # Light Green
    (63, 81, 181),    # Indigo
    (255, 152, 0),    # Orange
    (121, 85, 72),    # Brown
    (0, 150, 136),    # Teal
    (244, 67, 54),    # Red
    (3, 169, 244),    # Light Blue
    (205, 220, 57),   # Lime
]


def draw_crowns(
    image_rgb: np.ndarray,
    detections,
    canopy_mask: Optional[np.ndarray] = None,
    crown_masks: Optional[List[np.ndarray]] = None,
    show_boxes: bool = True,
    show_masks: bool = True,
    show_labels: bool = False,
    mask_alpha: float = 0.35,
    box_color: tuple = (0, 255, 0),
    box_thickness: int = 2,
) -> np.ndarray:
    """
    Draw crown detections on the image.

    Args:
        image_rgb: Original RGB image (H, W, 3).
        detections: DataFrame with xmin, ymin, xmax, ymax, score columns.
        canopy_mask: Combined canopy mask (used if crown_masks not provided).
        crown_masks: Individual crown masks for per-tree coloring.
        show_boxes: Whether to draw bounding boxes.
        show_masks: Whether to draw semi-transparent masks.
        show_labels: Whether to show tree ID labels.
        mask_alpha: Transparency of mask overlay (0=transparent, 1=opaque).
        box_color: RGB color for bounding boxes.
        box_thickness: Thickness of bounding box lines.

    Returns:
        Annotated RGB image.
    """
    overlay = image_rgb.copy()

    if show_masks:
        if crown_masks and len(crown_masks) > 0:
            # Color each crown individually
            for i, mask in enumerate(crown_masks):
                if not mask.any():
                    continue
                color = np.array(
                    CROWN_COLORS[i % len(CROWN_COLORS)], dtype=np.float64
                )
                overlay[mask] = (
                    overlay[mask].astype(np.float64) * (1 - mask_alpha)
                    + color * mask_alpha
                ).astype(np.uint8)

                # Draw contour around each crown for clarity
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contour_color = tuple(int(c * 0.8) for c in CROWN_COLORS[i % len(CROWN_COLORS)])
                cv2.drawContours(overlay, contours, -1, contour_color, 1)

        elif canopy_mask is not None and canopy_mask.any():
            # Use combined mask with single color
            color = np.array([76, 175, 80], dtype=np.float64)  # Green
            overlay[canopy_mask] = (
                overlay[canopy_mask].astype(np.float64) * (1 - mask_alpha)
                + color * mask_alpha
            ).astype(np.uint8)

    if show_boxes and detections is not None and len(detections) > 0:
        for idx, det in detections.iterrows():
            x1 = int(det["xmin"])
            y1 = int(det["ymin"])
            x2 = int(det["xmax"])
            y2 = int(det["ymax"])

            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, box_thickness)

            if show_labels:
                label = f"#{idx + 1}"
                font_scale = 0.4
                thickness = 1
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                cv2.rectangle(
                    overlay,
                    (x1, y1 - th - 4),
                    (x1 + tw + 4, y1),
                    box_color,
                    -1,
                )
                cv2.putText(
                    overlay,
                    label,
                    (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                )

    return overlay


def draw_masks_only(
    image_rgb: np.ndarray,
    canopy_mask: np.ndarray,
    mask_color: tuple = (76, 175, 80),
    background_alpha: float = 0.3,
) -> np.ndarray:
    """
    Create a visualization showing only the canopy mask
    with the original image dimmed in the background.
    """
    # Dim the background
    dimmed = (image_rgb.astype(np.float64) * background_alpha).astype(np.uint8)

    # Highlight canopy regions
    result = dimmed.copy()
    if canopy_mask.any():
        color = np.array(mask_color, dtype=np.float64)
        # Show canopy with brighter original + color tint
        result[canopy_mask] = (
            image_rgb[canopy_mask].astype(np.float64) * 0.6
            + color * 0.4
        ).astype(np.uint8)

    return result


def create_side_by_side(
    original: np.ndarray,
    annotated: np.ndarray,
    gap: int = 4,
    gap_color: tuple = (40, 40, 40),
) -> np.ndarray:
    """
    Create a side-by-side comparison image.

    Args:
        original: Original RGB image.
        annotated: Annotated RGB image (same dimensions).
        gap: Width of separator between images.
        gap_color: RGB color of the separator.

    Returns:
        Combined image with both views side by side.
    """
    h, w = original.shape[:2]

    # Create separator
    separator = np.full((h, gap, 3), gap_color, dtype=np.uint8)

    # Concatenate
    combined = np.hstack([original, separator, annotated])

    return combined


def create_detection_summary_image(
    image_rgb: np.ndarray,
    detections,
    canopy_mask: np.ndarray,
    crown_masks: Optional[List[np.ndarray]] = None,
) -> dict:
    """
    Create all visualization outputs needed for the UI.

    Returns:
        Dictionary with:
            - 'boxes_overlay': Image with bounding boxes
            - 'masks_overlay': Image with crown masks
            - 'combined_overlay': Image with both boxes and masks
            - 'masks_only': Canopy mask visualization
            - 'side_by_side': Original vs annotated comparison
    """
    # Boxes only
    boxes_overlay = draw_crowns(
        image_rgb, detections, show_boxes=True, show_masks=False
    )

    # Masks only (with contours)
    masks_overlay = draw_crowns(
        image_rgb,
        detections,
        canopy_mask=canopy_mask,
        crown_masks=crown_masks,
        show_boxes=False,
        show_masks=True,
        mask_alpha=0.4,
    )

    # Combined
    combined_overlay = draw_crowns(
        image_rgb,
        detections,
        canopy_mask=canopy_mask,
        crown_masks=crown_masks,
        show_boxes=True,
        show_masks=True,
        mask_alpha=0.3,
        box_thickness=1,
    )

    # Masks-only view (dimmed background)
    masks_only = draw_masks_only(image_rgb, canopy_mask)

    return {
        "boxes_overlay": boxes_overlay,
        "masks_overlay": masks_overlay,
        "combined_overlay": combined_overlay,
        "masks_only": masks_only,
    }
