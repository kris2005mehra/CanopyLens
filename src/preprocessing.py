"""
Image validation and preprocessing for CanopyLens.

Handles format checking, size validation, metadata extraction (including GeoTIFF),
and image preparation for the detection pipeline.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import io
import logging

logger = logging.getLogger(__name__)

# --- Constants ---
MAX_DIMENSION = 10000  # pixels — reject images larger than this
RESIZE_THRESHOLD = 5000  # pixels — resize images larger than this for processing
MIN_DIMENSION = 50  # pixels — reject images smaller than this
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_FILE_SIZE_MB = 200


@dataclass
class ImageMetadata:
    """Metadata extracted from an uploaded image."""
    width: int
    height: int
    format: str
    channels: int
    file_size_mb: float = 0.0
    gsd: Optional[float] = None  # Ground Sampling Distance — meters per pixel
    crs: Optional[str] = None  # Coordinate Reference System
    bounds: Optional[tuple] = None  # Geographic bounds (left, bottom, right, top)
    transform: Optional[object] = None  # Affine transform for georeferenced imagery
    gsd_source: str = "unavailable"  # "geotiff", "user_provided", "unavailable"


@dataclass
class ValidationResult:
    """Result of image validation checks."""
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Optional[ImageMetadata] = None


def validate_image(file_bytes: bytes, filename: str) -> ValidationResult:
    """
    Validate an uploaded image file.

    Checks format, dimensions, file size, and extracts basic metadata.
    Returns a ValidationResult with any warnings or errors.
    """
    errors = []
    warnings = []

    # Check file extension
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        errors.append(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return ValidationResult(is_valid=False, errors=errors)

    # Check file size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"File size ({file_size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
        return ValidationResult(is_valid=False, errors=errors)

    # Try to open and check image
    try:
        if ext in {".tif", ".tiff"}:
            metadata = _validate_geotiff(file_bytes, filename)
        else:
            metadata = _validate_standard_image(file_bytes)
    except Exception as e:
        errors.append(f"Could not read image: {str(e)}")
        return ValidationResult(is_valid=False, errors=errors)

    if metadata is None:
        errors.append("Could not extract image metadata.")
        return ValidationResult(is_valid=False, errors=errors)

    metadata.file_size_mb = file_size_mb
    metadata.format = ext.lstrip(".")

    # Validate dimensions
    if metadata.width < MIN_DIMENSION or metadata.height < MIN_DIMENSION:
        errors.append(
            f"Image too small ({metadata.width}×{metadata.height}). "
            f"Minimum dimension: {MIN_DIMENSION}px."
        )
    if metadata.width > MAX_DIMENSION or metadata.height > MAX_DIMENSION:
        errors.append(
            f"Image too large ({metadata.width}×{metadata.height}). "
            f"Maximum dimension: {MAX_DIMENSION}px."
        )

    # Warnings
    if max(metadata.width, metadata.height) > RESIZE_THRESHOLD:
        warnings.append(
            f"Large image ({metadata.width}×{metadata.height}). "
            f"It will be processed in tiles for better results."
        )

    if metadata.channels == 1:
        warnings.append(
            "Grayscale image detected. Tree crown detection works best with RGB imagery."
        )

    if metadata.gsd is not None and metadata.gsd > 2.0:
        warnings.append(
            f"Spatial resolution ({metadata.gsd:.2f} m/pixel) may be too coarse "
            f"for reliable individual tree crown detection. Best results with GSD < 1.0 m/pixel."
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid, warnings=warnings, errors=errors, metadata=metadata
    )


def _validate_standard_image(file_bytes: bytes) -> Optional[ImageMetadata]:
    """Validate and extract metadata from a standard image (PNG/JPG)."""
    img = Image.open(io.BytesIO(file_bytes))
    img.verify()  # Check integrity without loading full data

    # Re-open after verify (verify closes the file)
    img = Image.open(io.BytesIO(file_bytes))

    channels = len(img.getbands())
    return ImageMetadata(
        width=img.width,
        height=img.height,
        format=img.format or "unknown",
        channels=channels,
    )


def _validate_geotiff(file_bytes: bytes, filename: str) -> Optional[ImageMetadata]:
    """Validate and extract metadata from a GeoTIFF, including CRS and GSD."""
    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                width = dataset.width
                height = dataset.height
                channels = dataset.count

                crs = str(dataset.crs) if dataset.crs else None
                bounds = tuple(dataset.bounds) if dataset.bounds else None
                transform = dataset.transform

                # Calculate GSD from the transform
                gsd = None
                gsd_source = "unavailable"
                if transform and not transform.is_identity:
                    # Pixel size in CRS units
                    pixel_x = abs(transform.a)
                    pixel_y = abs(transform.e)

                    if crs and _is_geographic_crs(crs):
                        # CRS in degrees — estimate meters at the center latitude
                        if bounds:
                            center_lat = (bounds[1] + bounds[3]) / 2
                            gsd = _degrees_to_meters(max(pixel_x, pixel_y), center_lat)
                            gsd_source = "geotiff"
                    else:
                        # CRS already in meters (projected)
                        gsd = max(pixel_x, pixel_y)
                        gsd_source = "geotiff"

                return ImageMetadata(
                    width=width,
                    height=height,
                    format="tiff",
                    channels=channels,
                    gsd=gsd,
                    crs=crs,
                    bounds=bounds,
                    transform=transform,
                    gsd_source=gsd_source,
                )
    except ImportError:
        logger.warning("rasterio not installed — GeoTIFF metadata extraction unavailable")
        # Fall back to PIL for basic metadata
        return _validate_standard_image(file_bytes)
    except Exception as e:
        logger.warning(f"GeoTIFF metadata extraction failed: {e}")
        # Try as standard image
        return _validate_standard_image(file_bytes)


def _is_geographic_crs(crs_string: str) -> bool:
    """Check if a CRS is geographic (lat/lon in degrees)."""
    geographic_indicators = ["EPSG:4326", "WGS 84", "GCS_WGS", "geographic"]
    return any(indicator.lower() in crs_string.lower() for indicator in geographic_indicators)


def _degrees_to_meters(degrees: float, latitude: float) -> float:
    """Approximate conversion from degrees to meters at a given latitude."""
    import math
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(latitude))
    return degrees * (meters_per_degree_lat + meters_per_degree_lon) / 2


def load_image_as_rgb(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Load image bytes into an RGB numpy array.

    For GeoTIFF: reads the first 3 bands.
    For PNG/JPG: uses PIL.
    """
    ext = Path(filename).suffix.lower()

    if ext in {".tif", ".tiff"}:
        return _load_geotiff_rgb(file_bytes)
    else:
        return _load_standard_rgb(file_bytes)


def _load_standard_rgb(file_bytes: bytes) -> np.ndarray:
    """Load PNG/JPG as RGB numpy array."""
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode == "RGBA":
        # Composite onto white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def _load_geotiff_rgb(file_bytes: bytes) -> np.ndarray:
    """Load GeoTIFF as RGB numpy array (first 3 bands)."""
    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                if dataset.count >= 3:
                    # Read first 3 bands as RGB
                    r = dataset.read(1)
                    g = dataset.read(2)
                    b = dataset.read(3)
                    img = np.stack([r, g, b], axis=-1)
                elif dataset.count == 1:
                    # Grayscale — stack to 3 channels
                    band = dataset.read(1)
                    img = np.stack([band, band, band], axis=-1)
                else:
                    raise ValueError(f"Unexpected band count: {dataset.count}")

                # Handle different dtypes — normalize to uint8
                if img.dtype != np.uint8:
                    if img.max() > 255:
                        # Likely 16-bit — scale to 8-bit
                        img = ((img.astype(np.float64) / img.max()) * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                return img
    except ImportError:
        # Fall back to PIL
        return _load_standard_rgb(file_bytes)


def prepare_for_display(image: np.ndarray, max_display_dim: int = 2000) -> np.ndarray:
    """Resize image for UI display (does not affect analysis)."""
    h, w = image.shape[:2]
    if max(h, w) <= max_display_dim:
        return image

    scale = max_display_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_pil = Image.fromarray(image)
    img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img_pil)
