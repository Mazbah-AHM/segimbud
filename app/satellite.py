from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, UnidentifiedImageError

GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
USER_AGENT = "SegImBud/1.0"
LAYER_CANDIDATES = [
    {
        "layer": "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
        "satellite": "VIIRS NOAA-20",
        "native_resolution_m_per_px": 250.0,
    },
    {
        "layer": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "satellite": "VIIRS Suomi NPP",
        "native_resolution_m_per_px": 250.0,
    },
    {
        "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "satellite": "MODIS Terra",
        "native_resolution_m_per_px": 250.0,
    },
]


class SatelliteFetchError(RuntimeError):
    pass


def _clamp_latitude(latitude: float) -> float:
    return max(min(float(latitude), 89.8), -89.8)


def _square_bbox(latitude: float, longitude: float, area_km: float) -> tuple[float, float, float, float]:
    center_lat = _clamp_latitude(latitude)
    center_lon = float(longitude)
    half_side_km = max(float(area_km), 0.2) / 2.0

    lat_delta = half_side_km / 110.574
    lon_scale = max(math.cos(math.radians(center_lat)), 0.15)
    lon_delta = half_side_km / (111.320 * lon_scale)

    south = max(center_lat - lat_delta, -89.9)
    north = min(center_lat + lat_delta, 89.9)
    west = max(center_lon - lon_delta, -179.9)
    east = min(center_lon + lon_delta, 179.9)
    return west, south, east, north


def _candidate_dates(search_days: int) -> list[date]:
    today = datetime.now(timezone.utc).date()
    window = max(int(search_days), 1)
    return [today - timedelta(days=offset) for offset in range(window)]


def _build_wms_url(
    layer_name: str,
    target_date: date,
    bbox: tuple[float, float, float, float],
    output_size_px: int,
) -> str:
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "SRS": "EPSG:4326",
        "LAYERS": layer_name,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "TRANSPARENT": "false",
        "WIDTH": int(output_size_px),
        "HEIGHT": int(output_size_px),
        "TIME": target_date.isoformat(),
        "BBOX": ",".join(f"{value:.6f}" for value in bbox),
    }
    return f"{GIBS_WMS_URL}?{urlencode(params)}"


def _request_image(url: str) -> Image.Image | None:
    request = Request(
        url,
        headers={
            "Accept": "image/jpeg,image/png;q=0.9,*/*;q=0.1",
            "User-Agent": USER_AGENT,
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read()
    except HTTPError as exc:
        if 400 <= exc.code < 500:
            return None
        raise SatelliteFetchError("NASA GIBS could not complete the imagery request.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise SatelliteFetchError("NASA GIBS could not be reached. Try again in a moment.") from exc

    if "image/" not in content_type:
        return None

    try:
        return Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, ValueError):
        return None


def _image_has_detail(image: Image.Image) -> bool:
    array = np.asarray(image, dtype=np.uint8)
    if array.size == 0:
        return False

    grayscale = array.mean(axis=2)
    dynamic_range = float(np.max(grayscale) - np.min(grayscale))
    stddev = float(np.std(grayscale))
    non_zero_share = float(np.mean(grayscale > 3))
    return dynamic_range >= 10.0 and stddev >= 4.0 and non_zero_share >= 0.02


def fetch_recent_scene_by_coordinates(
    latitude: float,
    longitude: float,
    *,
    area_km: float = 5.0,
    search_days: int = 21,
    max_cloud_cover: float = 20.0,
    output_size_px: int = 1024,
) -> tuple[Image.Image, dict[str, Any]]:
    del max_cloud_cover

    bbox = _square_bbox(latitude, longitude, area_km)
    requested_resolution = round(max(float(area_km), 0.2) * 1000.0 / float(output_size_px), 2)

    for target_date in _candidate_dates(search_days):
        for layer in LAYER_CANDIDATES:
            request_url = _build_wms_url(layer["layer"], target_date, bbox, output_size_px)
            image = _request_image(request_url)
            if image is None or not _image_has_detail(image):
                continue

            metadata = {
                "provider": "NASA GIBS",
                "satellite": layer["satellite"],
                "collection": layer["layer"],
                "imagery_layer": layer["layer"],
                "acquired_at": target_date.isoformat(),
                "coordinates": {
                    "latitude": latitude,
                    "longitude": longitude,
                },
                "area_km": round(float(area_km), 2),
                "resolution_m_per_px": requested_resolution,
                "native_resolution_m_per_px": layer["native_resolution_m_per_px"],
                "render_strategy": "gibs-wms",
                "service_url": GIBS_WMS_URL,
                "item_url": request_url,
                "search_window_days": max(int(search_days), 1),
            }
            return image, metadata

    raise SatelliteFetchError(
        "No recent daily imagery was found for these coordinates in the requested time window. Try a larger search window or a different location."
    )
