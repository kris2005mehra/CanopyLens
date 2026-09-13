"""
Canopy metrics calculation for CanopyLens.

Computes tree count, canopy area, canopy cover percentage, and per-crown statistics.
Physical area metrics (m², hectares) are ONLY calculated when reliable
spatial resolution (GSD) is available — never fabricated.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class CanopyMetrics:
    """Complete canopy analysis metrics."""

    # Always available (pixel-based)
    tree_count: int = 0
    total_canopy_pixels: int = 0
    total_image_pixels: int = 0
    canopy_percentage_of_image: float = 0.0  # % of image area that is canopy
    mean_crown_area_pixels: float = 0.0
    median_crown_area_pixels: float = 0.0
    min_crown_area_pixels: int = 0
    max_crown_area_pixels: int = 0
    per_crown_pixels: List[int] = field(default_factory=list)

    # Physical metrics — ONLY set when GSD is available
    gsd: Optional[float] = None  # meters per pixel
    gsd_source: str = "unavailable"  # "geotiff", "user_provided", "unavailable"
    canopy_area_m2: Optional[float] = None
    canopy_area_hectares: Optional[float] = None
    analysis_area_m2: Optional[float] = None
    analysis_area_hectares: Optional[float] = None
    canopy_cover_percentage: Optional[float] = None
    mean_crown_area_m2: Optional[float] = None
    median_crown_area_m2: Optional[float] = None

    # Confidence score statistics
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0


def calculate_metrics(
    detections: pd.DataFrame,
    canopy_mask: np.ndarray,
    image_shape: tuple,
    crown_masks: Optional[List[np.ndarray]] = None,
    gsd: Optional[float] = None,
    gsd_source: str = "unavailable",
    kml_area_m2: Optional[float] = None,
) -> CanopyMetrics:
    """
    Calculate comprehensive canopy metrics.

    Args:
        detections: DataFrame with xmin, ymin, xmax, ymax, label, score.
        canopy_mask: Boolean mask (H, W) of canopy pixels.
        image_shape: (H, W) or (H, W, C) of the input image.
        crown_masks: Optional list of individual crown masks.
        gsd: Ground Sampling Distance in meters/pixel (None if unavailable).
        gsd_source: Source of GSD ("geotiff", "user_provided", "unavailable").
        kml_area_m2: Analysis area from KML boundary in m² (if available).

    Returns:
        CanopyMetrics with all calculated values.
    """
    h, w = image_shape[:2]
    total_pixels = h * w

    metrics = CanopyMetrics(
        tree_count=len(detections),
        total_image_pixels=total_pixels,
    )

    # --- Pixel-based metrics (always computed) ---

    metrics.total_canopy_pixels = int(canopy_mask.sum())
    metrics.canopy_percentage_of_image = (
        (metrics.total_canopy_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
    )

    # Per-crown pixel areas
    if crown_masks and len(crown_masks) > 0:
        per_crown = [int(m.sum()) for m in crown_masks]
        per_crown = [p for p in per_crown if p > 0]  # Remove empty masks
        metrics.per_crown_pixels = per_crown

        if per_crown:
            metrics.mean_crown_area_pixels = float(np.mean(per_crown))
            metrics.median_crown_area_pixels = float(np.median(per_crown))
            metrics.min_crown_area_pixels = int(np.min(per_crown))
            metrics.max_crown_area_pixels = int(np.max(per_crown))
    elif len(detections) > 0:
        # Fallback: estimate per-crown area from bounding boxes
        box_areas = (
            (detections["xmax"] - detections["xmin"])
            * (detections["ymax"] - detections["ymin"])
        ).astype(int)
        metrics.per_crown_pixels = box_areas.tolist()
        metrics.mean_crown_area_pixels = float(box_areas.mean())
        metrics.median_crown_area_pixels = float(box_areas.median())
        metrics.min_crown_area_pixels = int(box_areas.min())
        metrics.max_crown_area_pixels = int(box_areas.max())

    # Confidence statistics
    if len(detections) > 0 and "score" in detections.columns:
        metrics.mean_confidence = float(detections["score"].mean())
        metrics.min_confidence = float(detections["score"].min())
        metrics.max_confidence = float(detections["score"].max())

    # --- Physical metrics (ONLY when GSD is available) ---

    if gsd is not None and gsd > 0:
        metrics.gsd = gsd
        metrics.gsd_source = gsd_source

        pixel_area_m2 = gsd * gsd  # Area of one pixel in square meters

        # Canopy area
        metrics.canopy_area_m2 = metrics.total_canopy_pixels * pixel_area_m2
        metrics.canopy_area_hectares = metrics.canopy_area_m2 / 10000.0

        # Per-crown areas in m²
        if metrics.per_crown_pixels:
            crown_areas_m2 = [p * pixel_area_m2 for p in metrics.per_crown_pixels]
            metrics.mean_crown_area_m2 = float(np.mean(crown_areas_m2))
            metrics.median_crown_area_m2 = float(np.median(crown_areas_m2))

        # Analysis area and canopy cover percentage
        if kml_area_m2 is not None and kml_area_m2 > 0:
            # Use KML-defined analysis area
            metrics.analysis_area_m2 = kml_area_m2
            metrics.analysis_area_hectares = kml_area_m2 / 10000.0
            metrics.canopy_cover_percentage = (
                metrics.canopy_area_m2 / kml_area_m2 * 100
            )
        else:
            # Use full image area
            metrics.analysis_area_m2 = total_pixels * pixel_area_m2
            metrics.analysis_area_hectares = metrics.analysis_area_m2 / 10000.0
            metrics.canopy_cover_percentage = (
                metrics.canopy_area_m2 / metrics.analysis_area_m2 * 100
                if metrics.analysis_area_m2 > 0
                else None
            )
    else:
        metrics.gsd_source = "unavailable"

    return metrics


def format_area(value_m2: Optional[float]) -> str:
    """Format an area value with appropriate units."""
    if value_m2 is None:
        return "N/A — spatial resolution unavailable"

    if value_m2 >= 10000:
        return f"{value_m2 / 10000:.2f} hectares ({value_m2:,.0f} m²)"
    else:
        return f"{value_m2:,.1f} m²"


def metrics_to_dict(metrics: CanopyMetrics) -> dict:
    """Convert CanopyMetrics to a flat dictionary for CSV export."""
    return {
        "tree_count": metrics.tree_count,
        "total_canopy_pixels": metrics.total_canopy_pixels,
        "total_image_pixels": metrics.total_image_pixels,
        "canopy_pct_of_image": round(metrics.canopy_percentage_of_image, 2),
        "mean_crown_area_pixels": round(metrics.mean_crown_area_pixels, 1),
        "gsd_m_per_pixel": metrics.gsd,
        "gsd_source": metrics.gsd_source,
        "canopy_area_m2": round(metrics.canopy_area_m2, 2) if metrics.canopy_area_m2 else None,
        "canopy_area_hectares": round(metrics.canopy_area_hectares, 4) if metrics.canopy_area_hectares else None,
        "analysis_area_m2": round(metrics.analysis_area_m2, 2) if metrics.analysis_area_m2 else None,
        "canopy_cover_pct": round(metrics.canopy_cover_percentage, 2) if metrics.canopy_cover_percentage else None,
        "mean_crown_area_m2": round(metrics.mean_crown_area_m2, 2) if metrics.mean_crown_area_m2 else None,
        "mean_confidence": round(metrics.mean_confidence, 3),
    }
