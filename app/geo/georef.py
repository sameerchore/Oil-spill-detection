"""
TideTrace Georeferencing & Coordinate Transformation Engine.
Provides 6-parameter Affine transformation, pixel-to-WGS84 mapping, CRS handling,
and robust raster georeferencing without hard dependencies.
"""

from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List
import json
import numpy as np


class AffineTransform:
    """
    Standard 6-parameter 2D Affine Transformation Matrix.
    Mapping from pixel indices (col, row) to geographic coordinates (lon, lat):
      [ lon ]   [ a  b  c ] [ col ]
      [ lat ] = [ d  e  f ] [ row ]
      [  1  ]   [ 0  0  1 ] [  1  ]
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
    ):
        self.a = float(a)  # pixel width (x resolution)
        self.b = float(b)  # row rotation
        self.c = float(c)  # upper-left x (min_lon or origin lon)
        self.d = float(d)  # col rotation
        self.e = float(e)  # pixel height (y resolution, typically negative)
        self.f = float(f)  # upper-left y (max_lat or origin lat)

        self._det = self.a * self.e - self.b * self.d
        if abs(self._det) < 1e-15:
            raise ValueError("Degenerate affine transformation matrix (determinant is zero).")

    def pixel_to_geo(self, row: Union[float, np.ndarray], col: Union[float, np.ndarray]) -> Tuple[Any, Any]:
        """Convert pixel (row, col) to geographic (lon, lat) coordinates."""
        lon = self.a * col + self.b * row + self.c
        lat = self.d * col + self.e * row + self.f
        return lon, lat

    def geo_to_pixel(self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Tuple[Any, Any]:
        """Convert geographic (lon, lat) to raster pixel (row, col) coordinates."""
        dx = lon - self.c
        dy = lat - self.f
        col = (self.e * dx - self.b * dy) / self._det
        row = (-self.d * dx + self.a * dy) / self._det
        return row, col

    @classmethod
    def from_bounds(
        cls,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        height: int,
        width: int,
    ) -> "AffineTransform":
        """
        Construct affine transform spanning a rectangular bounding box.
        Top-left is (min_lon, max_lat), bottom-right is (max_lon, min_lat).
        """
        assert height > 0 and width > 0, "Height and width must be strictly positive."
        a = (max_lon - min_lon) / float(width)
        b = 0.0
        c = float(min_lon)
        d = 0.0
        e = -(max_lat - min_lat) / float(height)
        f = float(max_lat)
        return cls(a, b, c, d, e, f)

    @classmethod
    def from_tuple(cls, transform_tuple: Tuple[float, float, float, float, float, float]) -> "AffineTransform":
        """Construct from 6-tuple (a, b, c, d, e, f)."""
        return cls(*transform_tuple)

    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return 6-tuple representation (a, b, c, d, e, f)."""
        return (self.a, self.b, self.c, self.d, self.e, self.f)

    def to_dict(self) -> Dict[str, float]:
        """Serialize transform parameters to JSON-serializable dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "f": self.f,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "AffineTransform":
        """Deserialize from dictionary."""
        return cls(data["a"], data["b"], data["c"], data["d"], data["e"], data["f"])

    def __repr__(self) -> str:
        return f"AffineTransform(a={self.a:.8f}, b={self.b:.8f}, c={self.c:.8f}, d={self.d:.8f}, e={self.e:.8f}, f={self.f:.8f})"


class GeoReference:
    """
    Spatial georeferencing container encapsulating CRS, transform, and bounding extent.
    """

    def __init__(
        self,
        transform: AffineTransform,
        crs: str = "EPSG:4326",
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        self.transform = transform
        self.crs = crs
        self.height = height
        self.width = width

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Compute bounding box (min_lon, min_lat, max_lon, max_lat) if dimensions are known."""
        if self.height is None or self.width is None:
            return None
        
        # 4 corners
        lons, lats = [], []
        for r, c in [(0, 0), (0, self.width), (self.height, 0), (self.height, self.width)]:
            lon, lat = self.transform.pixel_to_geo(r, c)
            lons.append(lon)
            lats.append(lat)
        return (min(lons), min(lats), max(lons), max(lats))

    def pixel_to_wgs84(self, row: Union[float, np.ndarray], col: Union[float, np.ndarray]) -> Tuple[Any, Any]:
        """Convert pixel row, col to WGS84 (lon, lat)."""
        return self.transform.pixel_to_geo(row, col)

    def wgs84_to_pixel(self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Tuple[Any, Any]:
        """Convert WGS84 (lon, lat) to pixel (row, col)."""
        return self.transform.geo_to_pixel(lon, lat)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize georeference properties."""
        return {
            "transform": self.transform.to_dict(),
            "crs": self.crs,
            "height": self.height,
            "width": self.width,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeoReference":
        """Deserialize from dictionary."""
        transform = AffineTransform.from_dict(data["transform"])
        return cls(
            transform=transform,
            crs=data.get("crs", "EPSG:4326"),
            height=data.get("height"),
            width=data.get("width"),
        )


def create_default_georef(
    height: int = 2048,
    width: int = 2048,
    center_lat: float = 18.925,
    center_lon: float = 72.825,
    pixel_size_deg: float = 0.0001,  # ~10 meters per pixel
) -> GeoReference:
    """
    Construct a default calibrated georeference for unreferenced SAR masks/arrays.
    Default center represents the Arabian Sea off Mumbai.
    """
    half_w = (width * pixel_size_deg) / 2.0
    half_h = (height * pixel_size_deg) / 2.0

    min_lon = center_lon - half_w
    max_lon = center_lon + half_w
    min_lat = center_lat - half_h
    max_lat = center_lat + half_h

    transform = AffineTransform.from_bounds(min_lon, min_lat, max_lon, max_lat, height, width)
    return GeoReference(transform=transform, crs="EPSG:4326", height=height, width=width)


def extract_georeference(
    file_path: Union[str, Path],
    fallback_center: Tuple[float, float] = (18.925, 72.825),
) -> GeoReference:
    """
    Extract embedded GeoTIFF georeferencing metadata, or sidecar .meta.json file.
    If unreferenced, returns a calibrated default spatial frame.
    """
    path = Path(file_path)
    
    # 1. Check sidecar metadata file
    meta_json = path.with_suffix(".meta.json")
    if meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                return GeoReference.from_dict(data)
        except Exception:
            pass

    # 2. Try rasterio if available
    try:
        import rasterio
        with rasterio.open(path) as src:
            t = src.transform
            affine = AffineTransform(t.a, t.b, t.c, t.d, t.e, t.f)
            crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
            return GeoReference(transform=affine, crs=crs_str, height=src.height, width=src.width)
    except Exception:
        pass

    # 3. Try reading TIFF tags via tifffile
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            h, w = tif.pages[0].shape[:2]
            return create_default_georef(height=h, width=w, center_lat=fallback_center[0], center_lon=fallback_center[1])
    except Exception:
        pass

    return create_default_georef(center_lat=fallback_center[0], center_lon=fallback_center[1])
