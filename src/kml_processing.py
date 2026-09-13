"""
KML boundary file processing for CanopyLens.

Parses KML files to extract polygon boundaries that define analysis regions.
Handles area calculation using appropriate projected CRS (UTM).
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class KMLBoundary:
    """Parsed KML boundary information."""
    name: str = "Unnamed boundary"
    description: str = ""
    polygons: list = field(default_factory=list)  # List of (lon, lat) coordinate lists
    area_sq_m: Optional[float] = None
    area_hectares: Optional[float] = None
    crs_used: Optional[str] = None
    centroid: Optional[Tuple[float, float]] = None  # (lon, lat)
    warnings: List[str] = field(default_factory=list)


def parse_kml(kml_bytes: bytes) -> KMLBoundary:
    """
    Parse a KML file and extract polygon boundaries.

    Args:
        kml_bytes: Raw bytes of the KML file.

    Returns:
        KMLBoundary with extracted polygon coordinates and calculated area.

    Raises:
        ValueError: If the KML cannot be parsed or contains no valid polygons.
    """
    try:
        from lxml import etree
    except ImportError:
        raise RuntimeError("lxml is required for KML parsing. Install with: pip install lxml")

    try:
        root = etree.fromstring(kml_bytes)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Invalid KML file — XML parsing error: {e}")

    # KML namespace
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # Try with namespace first, then without
    polygons = _extract_polygons(root, ns)
    if not polygons:
        polygons = _extract_polygons(root, ns={})

    if not polygons:
        raise ValueError(
            "No polygon geometries found in the KML file. "
            "The KML should contain at least one Polygon element defining the analysis boundary."
        )

    # Extract name and description
    name = _find_text(root, ns, ["kml:Document/kml:name", "kml:name", "Document/name", "name"])
    description = _find_text(root, ns, [
        "kml:Document/kml:description", "kml:description",
        "Document/description", "description"
    ])

    boundary = KMLBoundary(
        name=name or "Unnamed boundary",
        description=description or "",
        polygons=polygons,
    )

    # Calculate area using projected CRS
    try:
        _calculate_area(boundary)
    except Exception as e:
        boundary.warnings.append(f"Could not calculate polygon area: {e}")
        logger.warning(f"KML area calculation failed: {e}")

    return boundary


def _extract_polygons(root, ns: dict) -> List[List[Tuple[float, float]]]:
    """Extract polygon coordinate lists from KML XML."""
    from lxml import etree

    polygons = []

    # Search for Polygon elements
    if ns:
        poly_elements = root.findall(".//kml:Polygon", ns)
    else:
        # Try without namespace
        poly_elements = root.findall(".//{http://www.opengis.net/kml/2.2}Polygon")
        if not poly_elements:
            poly_elements = root.findall(".//Polygon")

    for poly_elem in poly_elements:
        coords = _extract_coordinates(poly_elem, ns)
        if coords:
            polygons.append(coords)

    return polygons


def _extract_coordinates(
    poly_elem, ns: dict
) -> Optional[List[Tuple[float, float]]]:
    """Extract coordinate tuples from a Polygon element."""
    # Try various XPath patterns for coordinates
    coord_paths = [
        ".//kml:coordinates",
        ".//{http://www.opengis.net/kml/2.2}coordinates",
        ".//coordinates",
    ]

    for path in coord_paths:
        try:
            coord_elem = poly_elem.find(path, ns) if ns else poly_elem.find(path)
            if coord_elem is not None and coord_elem.text:
                return _parse_coordinate_text(coord_elem.text.strip())
        except Exception:
            continue

    return None


def _parse_coordinate_text(
    text: str,
) -> List[Tuple[float, float]]:
    """Parse KML coordinate text into (longitude, latitude) tuples."""
    coords = []
    for item in text.split():
        parts = item.strip().split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            except ValueError:
                continue

    if len(coords) < 3:
        return []

    return coords


def _find_text(root, ns: dict, paths: List[str]) -> Optional[str]:
    """Find text content using multiple XPath alternatives."""
    for path in paths:
        try:
            elem = root.find(path, ns) if ns and "kml:" in path else root.find(path)
            if elem is not None and elem.text:
                return elem.text.strip()
        except Exception:
            continue
    return None


def _calculate_area(boundary: KMLBoundary) -> None:
    """Calculate polygon area in square meters using UTM projection."""
    if not boundary.polygons:
        return

    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import transform
        from pyproj import Transformer
    except ImportError:
        boundary.warnings.append(
            "shapely/pyproj not available — cannot calculate polygon area."
        )
        return

    total_area_m2 = 0.0
    all_lons = []
    all_lats = []

    for coord_list in boundary.polygons:
        if len(coord_list) < 3:
            continue

        # Create Shapely polygon (lon, lat)
        polygon = ShapelyPolygon(coord_list)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # Fix minor topology issues

        # Collect coordinates for centroid calculation
        for lon, lat in coord_list:
            all_lons.append(lon)
            all_lats.append(lat)

        # Determine appropriate UTM zone from polygon centroid
        centroid = polygon.centroid
        utm_epsg = _get_utm_epsg(centroid.x, centroid.y)

        # Project to UTM for accurate area calculation
        try:
            transformer = Transformer.from_crs(
                "EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True
            )
            projected = transform(transformer.transform, polygon)
            total_area_m2 += projected.area
        except Exception as e:
            boundary.warnings.append(f"UTM projection failed: {e}")
            # Fallback: approximate using Haversine-based method
            total_area_m2 += _approximate_area_m2(coord_list)

    boundary.area_sq_m = total_area_m2
    boundary.area_hectares = total_area_m2 / 10000.0
    boundary.crs_used = f"EPSG:{utm_epsg}" if total_area_m2 > 0 else None

    if all_lons and all_lats:
        boundary.centroid = (
            sum(all_lons) / len(all_lons),
            sum(all_lats) / len(all_lats),
        )


def _get_utm_epsg(lon: float, lat: float) -> int:
    """Determine the appropriate UTM zone EPSG code for a given lon/lat."""
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + utm_zone  # Northern hemisphere
    else:
        return 32700 + utm_zone  # Southern hemisphere


def _approximate_area_m2(coords: List[Tuple[float, float]]) -> float:
    """
    Approximate polygon area using the Shoelace formula
    with latitude-adjusted degree-to-meter conversion.
    This is a fallback when proper CRS projection is unavailable.
    """
    if len(coords) < 3:
        return 0.0

    # Average latitude for degree-to-meter conversion
    avg_lat = sum(lat for _, lat in coords) / len(coords)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(avg_lat))

    # Convert to approximate meters
    coords_m = [(lon * m_per_deg_lon, lat * m_per_deg_lat) for lon, lat in coords]

    # Shoelace formula
    n = len(coords_m)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords_m[i][0] * coords_m[j][1]
        area -= coords_m[j][0] * coords_m[i][1]

    return abs(area) / 2.0
