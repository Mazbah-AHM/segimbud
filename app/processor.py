from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_IMAGE_SIZE = (1024, 1024)
IMAGE_MEAN = np.array([0.2903465, 0.31224626, 0.29810827], dtype=np.float32)
IMAGE_STD = np.array([0.16213588, 0.14422065, 0.13841338], dtype=np.float32)

CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture",
]

CLASS_NAME_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

COLOR_MAP = {
    0: [255, 255, 255],
    1: [255, 0, 0],
    2: [255, 255, 0],
    3: [0, 0, 255],
    4: [159, 129, 183],
    5: [0, 255, 0],
    6: [255, 195, 128],
}

NEUTRAL_MASK_COLOR = np.array([241, 236, 229], dtype=np.uint8)
CHANGE_BASE_COLOR = np.array([233, 231, 226], dtype=np.uint8)


def preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize(MODEL_IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()


def build_color_mask(class_mask: np.ndarray, visible_classes: Iterable[int] | None = None) -> np.ndarray:
    visible = set(visible_classes) if visible_classes is not None else set(COLOR_MAP)
    color_mask = np.broadcast_to(NEUTRAL_MASK_COLOR, (*class_mask.shape, 3)).copy()
    for class_id, color in COLOR_MAP.items():
        if class_id in visible:
            color_mask[class_mask == class_id] = color
    return color_mask.astype(np.uint8)


def postprocess(output_tensor: torch.Tensor):
    probs = torch.softmax(output_tensor, dim=1)
    conf_map, class_mask = torch.max(probs, dim=1)

    class_mask_np = class_mask.squeeze().cpu().numpy().astype(np.uint8)
    conf_map_np = conf_map.squeeze().cpu().numpy().astype(np.float32)
    color_mask = build_color_mask(class_mask_np)
    return color_mask, class_mask_np, conf_map_np


def resize_prediction_maps(
    class_mask: np.ndarray,
    conf_map: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    target_width, target_height = target_size

    mask_image = Image.fromarray(class_mask, mode="L").resize(
        (target_width, target_height),
        Image.Resampling.NEAREST,
    )
    resized_mask = np.array(mask_image, dtype=np.uint8)

    conf_image = Image.fromarray((np.clip(conf_map, 0.0, 1.0) * 255).astype(np.uint8), mode="L").resize(
        (target_width, target_height),
        Image.Resampling.BILINEAR,
    )
    resized_conf = np.array(conf_image, dtype=np.float32) / 255.0

    return resized_mask, resized_conf


def compute_stats(class_mask: np.ndarray) -> dict:
    total = int(class_mask.size)
    counts = np.bincount(class_mask.ravel(), minlength=len(CLASS_NAMES))
    stats = {}

    for class_id, name in enumerate(CLASS_NAMES):
        count = int(counts[class_id])
        stats[name] = {
            "class_id": class_id,
            "pixels": count,
            "percentage": round(count / total * 100, 2) if total else 0.0,
            "color": COLOR_MAP[class_id],
        }

    return stats


def compute_confidence_summary(
    conf_map: np.ndarray,
    low_conf_threshold: float = 0.60,
    high_conf_threshold: float = 0.85,
) -> dict:
    flat = np.clip(conf_map.astype(np.float32).reshape(-1), 0.0, 1.0)
    return {
        "mean_confidence": round(float(np.mean(flat)), 4),
        "median_confidence": round(float(np.median(flat)), 4),
        "min_confidence": round(float(np.min(flat)), 4),
        "max_confidence": round(float(np.max(flat)), 4),
        "low_conf_threshold": round(float(low_conf_threshold), 2),
        "low_confidence_percentage": round(float(np.mean(flat < low_conf_threshold) * 100), 2),
        "high_conf_threshold": round(float(high_conf_threshold), 2),
        "high_confidence_percentage": round(float(np.mean(flat >= high_conf_threshold) * 100), 2),
    }


def compute_area_stats(stats: dict, resolution_m_per_px: float | None) -> dict:
    if not resolution_m_per_px or resolution_m_per_px <= 0:
        return stats

    pixel_area_sq_m = float(resolution_m_per_px) ** 2
    pixel_area_sq_km = pixel_area_sq_m / 1_000_000
    enriched = {}

    for class_name, class_stats in stats.items():
        pixels = class_stats["pixels"]
        enriched[class_name] = {
            **class_stats,
            "area_sq_m": round(pixels * pixel_area_sq_m, 2),
            "area_sq_km": round(pixels * pixel_area_sq_km, 6),
        }

    return enriched


def build_overlay_image(
    image_rgb: np.ndarray,
    class_mask: np.ndarray,
    visible_classes: Iterable[int] | None = None,
    alpha: float = 0.55,
) -> np.ndarray:
    visible = set(visible_classes) if visible_classes is not None else set(COLOR_MAP)
    base = image_rgb.astype(np.float32)
    color_mask = build_color_mask(class_mask, visible).astype(np.float32)
    visible_pixels = np.isin(class_mask, list(visible))
    blended = base.copy()
    blended[visible_pixels] = (
        (1.0 - alpha) * base[visible_pixels] + alpha * color_mask[visible_pixels]
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_change_analysis(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    resolution_m_per_px: float | None = None,
    top_k: int = 8,
) -> dict:
    if before_mask.shape != after_mask.shape:
        after_image = Image.fromarray(after_mask, mode="L").resize(
            (before_mask.shape[1], before_mask.shape[0]),
            Image.Resampling.NEAREST,
        )
        after_mask = np.array(after_image, dtype=np.uint8)

    changed = before_mask != after_mask
    changed_pixels = int(np.sum(changed))
    total_pixels = int(before_mask.size)

    change_visual = np.broadcast_to(CHANGE_BASE_COLOR, (*before_mask.shape, 3)).copy()
    for class_id, color in COLOR_MAP.items():
        change_visual[np.logical_and(changed, after_mask == class_id)] = color

    num_classes = len(CLASS_NAMES)
    transition_ids = before_mask.astype(np.int16) * num_classes + after_mask.astype(np.int16)
    transition_matrix = np.bincount(
        transition_ids.ravel(),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    transitions = []
    for from_id in range(num_classes):
        for to_id in range(num_classes):
            if from_id == to_id:
                continue
            pixels = int(transition_matrix[from_id, to_id])
            if pixels <= 0:
                continue
            entry = {
                "from_class": CLASS_NAMES[from_id],
                "to_class": CLASS_NAMES[to_id],
                "pixels": pixels,
                "percentage_of_image": round(pixels / total_pixels * 100, 2) if total_pixels else 0.0,
            }
            if resolution_m_per_px and resolution_m_per_px > 0:
                entry["area_sq_km"] = round(
                    pixels * (float(resolution_m_per_px) ** 2) / 1_000_000,
                    6,
                )
            transitions.append(entry)

    transitions.sort(key=lambda item: item["pixels"], reverse=True)

    class_delta = {}
    before_counts = np.bincount(before_mask.ravel(), minlength=num_classes)
    after_counts = np.bincount(after_mask.ravel(), minlength=num_classes)
    for class_id, class_name in enumerate(CLASS_NAMES):
        delta_pixels = int(after_counts[class_id] - before_counts[class_id])
        class_delta[class_name] = {
            "class_id": class_id,
            "before_pixels": int(before_counts[class_id]),
            "after_pixels": int(after_counts[class_id]),
            "delta_pixels": delta_pixels,
            "delta_percentage_points": round(
                (after_counts[class_id] - before_counts[class_id]) / total_pixels * 100,
                2,
            )
            if total_pixels
            else 0.0,
            "color": COLOR_MAP[class_id],
        }
        if resolution_m_per_px and resolution_m_per_px > 0:
            class_delta[class_name]["delta_area_sq_km"] = round(
                delta_pixels * (float(resolution_m_per_px) ** 2) / 1_000_000,
                6,
            )

    summary = {
        "changed_pixels": changed_pixels,
        "changed_percentage": round(changed_pixels / total_pixels * 100, 2) if total_pixels else 0.0,
        "transitions": transitions[:top_k],
        "class_delta": class_delta,
        "change_mask": change_visual.astype(np.uint8),
    }
    if resolution_m_per_px and resolution_m_per_px > 0:
        summary["changed_area_sq_km"] = round(
            changed_pixels * (float(resolution_m_per_px) ** 2) / 1_000_000,
            6,
        )

    return summary


def generate_operational_insights(
    stats: dict,
    confidence_summary: dict,
    change_summary: dict | None = None,
) -> list[str]:
    insights: list[str] = []
    ordered = sorted(stats.items(), key=lambda item: item[1]["pixels"], reverse=True)

    if ordered:
        dominant_name, dominant_stats = ordered[0]
        insights.append(
            f"Dominant class: {dominant_name} covers {dominant_stats['percentage']}% of the analyzed scene."
        )

    active_classes = [name for name, values in stats.items() if values["percentage"] >= 1.0]
    insights.append(f"Active land-cover classes above 1%: {len(active_classes)}.")

    low_confidence = confidence_summary.get("low_confidence_percentage", 0.0)
    mean_confidence = confidence_summary.get("mean_confidence", 0.0)
    if low_confidence >= 20:
        insights.append(
            f"Confidence watch: {low_confidence}% of pixels are below the low-confidence threshold."
        )
    else:
        insights.append(
            f"Confidence profile is stable overall, with mean confidence at {mean_confidence:.2f}."
        )

    if change_summary is not None:
        changed_percentage = change_summary.get("changed_percentage", 0.0)
        insights.append(f"Detected change footprint: {changed_percentage}% of pixels differ between the two scenes.")
        transitions = change_summary.get("transitions", [])
        if transitions:
            top_transition = transitions[0]
            insights.append(
                f"Primary class transition: {top_transition['from_class']} -> {top_transition['to_class']} "
                f"across {top_transition['percentage_of_image']}% of the image."
            )

    return insights


def _tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, max(length - tile_size, 0) + 1, stride))
    last_start = max(length - tile_size, 0)
    if not positions or positions[-1] != last_start:
        positions.append(last_start)
    return positions


def _edge_weight_mask(height: int, width: int) -> np.ndarray:
    row = np.linspace(0.1, 1.0, num=max(height, 2), dtype=np.float32)
    row = np.minimum(row, row[::-1])
    row = np.clip(row, 0.18, 1.0)

    col = np.linspace(0.1, 1.0, num=max(width, 2), dtype=np.float32)
    col = np.minimum(col, col[::-1])
    col = np.clip(col, 0.18, 1.0)

    return np.outer(row[:height], col[:width]).astype(np.float32)


def _run_network_on_image(
    image: Image.Image,
    network: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    input_tensor = preprocess(image).to(device)
    with torch.inference_mode():
        output = network(input_tensor)
        probs = torch.softmax(output, dim=1)

    resized_probs = F.interpolate(
        probs,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).cpu().numpy()

    class_mask = np.argmax(resized_probs, axis=0).astype(np.uint8)
    conf_map = np.max(resized_probs, axis=0).astype(np.float32)
    return class_mask, conf_map


def run_single_inference(
    image: Image.Image,
    network: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    class_mask, conf_map = _run_network_on_image(image, network, device)
    metadata = {
        "mode": "single",
        "tile_size": MODEL_IMAGE_SIZE[0],
        "tile_overlap": 0,
        "tile_count": 1,
        "stride": MODEL_IMAGE_SIZE[0],
    }
    return class_mask, conf_map, metadata


def run_tiled_inference(
    image: Image.Image,
    network: torch.nn.Module,
    device: torch.device,
    tile_size: int = 1024,
    tile_overlap: int = 160,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if tile_size < 256:
        raise ValueError("Tile size must be at least 256 pixels.")
    if tile_overlap < 0:
        raise ValueError("Tile overlap cannot be negative.")
    if tile_overlap >= tile_size:
        raise ValueError("Tile overlap must be smaller than tile size.")

    width, height = image.size
    stride = max(tile_size - tile_overlap, 1)
    x_positions = _tile_positions(width, tile_size, stride)
    y_positions = _tile_positions(height, tile_size, stride)

    class_map = np.zeros((height, width), dtype=np.uint8)
    conf_map = np.zeros((height, width), dtype=np.float32)
    score_map = np.full((height, width), -1.0, dtype=np.float32)

    tile_count = 0
    for top in y_positions:
        for left in x_positions:
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tile = image.crop((left, top, right, bottom))
            tile_mask, tile_conf = _run_network_on_image(tile, network, device)
            weight = _edge_weight_mask(tile.height, tile.width)
            weighted_score = tile_conf * weight

            current_scores = score_map[top:bottom, left:right]
            update_mask = weighted_score > current_scores
            current_scores[update_mask] = weighted_score[update_mask]
            conf_map[top:bottom, left:right][update_mask] = tile_conf[update_mask]
            class_map[top:bottom, left:right][update_mask] = tile_mask[update_mask]
            tile_count += 1

    metadata = {
        "mode": "tiled",
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_count": tile_count,
        "stride": stride,
    }
    return class_map, conf_map, metadata


def run_segmentation(
    image: Image.Image,
    network: torch.nn.Module,
    device: torch.device,
    *,
    tile_size: int = 1024,
    tile_overlap: int = 160,
    force_tiled: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    width, height = image.size
    use_tiled = force_tiled or width > tile_size or height > tile_size
    if use_tiled:
        return run_tiled_inference(
            image,
            network,
            device,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    return run_single_inference(image, network, device)


def summarize_review(
    original_mask: np.ndarray,
    reviewed_mask: np.ndarray,
    confidence_map: np.ndarray | None = None,
    low_conf_threshold: float = 0.6,
) -> dict[str, Any]:
    if original_mask.shape != reviewed_mask.shape:
        raise ValueError("Original and reviewed masks must have the same shape.")

    changed = original_mask != reviewed_mask
    edited_pixels = int(np.sum(changed))
    total_pixels = int(original_mask.size)

    touched_class_ids = np.unique(reviewed_mask[changed]).tolist() if edited_pixels else []
    touched_classes = [CLASS_NAMES[class_id] for class_id in touched_class_ids]

    summary: dict[str, Any] = {
        "edited_pixels": edited_pixels,
        "edited_percentage": round(edited_pixels / total_pixels * 100, 2) if total_pixels else 0.0,
        "touched_classes": touched_classes,
        "corrected_stats": compute_stats(reviewed_mask),
    }

    if confidence_map is not None:
        low_conf_mask = confidence_map < low_conf_threshold
        low_conf_edited = int(np.sum(np.logical_and(changed, low_conf_mask)))
        summary["low_confidence_edited_pixels"] = low_conf_edited
        summary["low_confidence_edited_percentage"] = (
            round(low_conf_edited / edited_pixels * 100, 2) if edited_pixels else 0.0
        )

    return summary
