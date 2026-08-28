import json
from pathlib import Path

import cv2
import numpy as np

from libs.utils.highway_data import normalize_data_roots, parse_split_line, resolve_image_path

CLASSES = ('solid', 'dashed', 'joint')
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
LABEL_COLORS = {
    'solid': (0, 220, 0),
    'dashed': (0, 180, 255),
    'joint': (255, 0, 255),
    'lane': (255, 255, 0),
    'unknown': (180, 180, 180),
}


def clean_polyline(points, sort_points=True, min_points=2):
    cleaned = []
    for point in points:
        if point is None or len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        if np.isfinite(x) and np.isfinite(y):
            cleaned.append([x, y])
    if sort_points:
        cleaned = sorted(cleaned, key=lambda p: p[1], reverse=True)

    unique = []
    for point in cleaned:
        if not unique or np.linalg.norm(np.asarray(point) - np.asarray(unique[-1])) > 1e-3:
            unique.append(point)
    if len(unique) < min_points:
        return []
    return unique


def load_lanes(json_path, allowed_labels=CLASSES, min_points=2, sort_points=True):
    json_path = Path(json_path)
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    raw_lanes = data.get('lanes')
    if raw_lanes is None:
        raw_lanes = data.get('shapes', [])

    lanes = []
    for lane_idx, lane in enumerate(raw_lanes):
        label = str(lane.get('label', '')).lower()
        if label not in allowed_labels:
            raise ValueError(f'Unexpected lane label {label!r} in {json_path}')
        points = clean_polyline(
            lane.get('points', []), sort_points=sort_points, min_points=min_points
        )
        if len(points) < min_points:
            continue
        lanes.append(
            {
                'label': label,
                'points': points,
                'lane_index': lane_idx,
                'source_json': str(json_path),
            }
        )
    return lanes


def find_image_path(data_root, item, data_roots=None):
    roots, default_root_key = normalize_data_roots(data_root, data_roots)
    root_key, rel = parse_split_line(item, roots, default_root_key)
    return resolve_image_path(roots, root_key, rel)


def read_split_images(data_root, split_file, data_roots=None):
    roots, default_root_key = normalize_data_roots(data_root, data_roots)
    paths = []
    with Path(split_file).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                root_key, rel = parse_split_line(line, roots, default_root_key)
                paths.append(resolve_image_path(roots, root_key, rel))
    return paths


def resample_polyline(points, num_samples):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError('Cannot resample an empty polyline')
    if len(points) == 1:
        return np.repeat(points, num_samples, axis=0)

    deltas = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    keep = np.r_[True, seg_lens > 1e-4]
    points = points[keep]
    if len(points) == 1:
        return np.repeat(points, num_samples, axis=0)

    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(seg_lens.sum())
    if total <= 1e-4:
        return np.repeat(points[:1], num_samples, axis=0)

    cumulative = np.r_[0.0, np.cumsum(seg_lens)]
    targets = np.linspace(0.0, total, int(num_samples), dtype=np.float32)
    xs = np.interp(targets, cumulative, points[:, 0])
    ys = np.interp(targets, cumulative, points[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def lane_strip_crop(
    image,
    points,
    crop_size=(288, 128),
    strip_width=128,
    sort_points=True,
    pad_value=(0, 0, 0),
):
    crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
    if crop_h <= 1 or crop_w <= 1:
        raise ValueError(f'Invalid crop_size: {crop_size}')
    points = clean_polyline(points, sort_points=sort_points, min_points=2)
    if len(points) < 2:
        raise ValueError('lane_strip_crop needs at least two valid points')

    center = resample_polyline(points, crop_h)
    prev_pts = np.vstack([center[:1], center[:-1]])
    next_pts = np.vstack([center[1:], center[-1:]])
    tangent = next_pts - prev_pts
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm[tangent_norm < 1e-4] = 1.0
    tangent = tangent / tangent_norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    offsets = np.linspace(
        -float(strip_width) * 0.5, float(strip_width) * 0.5, crop_w, dtype=np.float32
    )
    map_x = center[:, 0:1] + normal[:, 0:1] * offsets[None, :]
    map_y = center[:, 1:2] + normal[:, 1:2] * offsets[None, :]

    crop = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=pad_value,
    )
    return crop


def effective_strip_width(
    image_width,
    strip_width=128,
    crop_mode='fixed',
    strip_reference_width=2560,
    strip_min_width=64,
    strip_max_width=192,
):
    if str(crop_mode) == 'fixed' or str(crop_mode) == 'normalized':
        return int(strip_width)
    if str(crop_mode) != 'scaled':
        raise ValueError(f'Unsupported crop_mode: {crop_mode}')
    scaled = float(strip_width) * float(image_width) / max(float(strip_reference_width), 1.0)
    return int(round(np.clip(scaled, int(strip_min_width), int(strip_max_width))))


def normalize_image_and_points(image, points, normalized_size=(1024, 544), top_crop_ratio=0.08):
    target_w, target_h = int(normalized_size[0]), int(normalized_size[1])
    if target_w <= 1 or target_h <= 1:
        raise ValueError(f'Invalid normalized_size: {normalized_size}')
    image_h, image_w = image.shape[:2]
    top_crop = int(round(float(image_h) * float(top_crop_ratio)))
    top_crop = int(np.clip(top_crop, 0, max(image_h - 2, 0)))
    cropped = image[top_crop:]
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    clean = clean_polyline(points, sort_points=False, min_points=2)
    clipped = []
    x_min, x_max = 0.0, float(image_w - 1)
    y_min, y_max = float(top_crop), float(image_h - 1)
    for start, end in zip(clean[:-1], clean[1:]):
        x0, y0 = map(float, start)
        x1, y1 = map(float, end)
        dx, dy = x1 - x0, y1 - y0
        t_min, t_max = 0.0, 1.0
        valid = True
        for p, q in (
            (-dx, x0 - x_min),
            (dx, x_max - x0),
            (-dy, y0 - y_min),
            (dy, y_max - y0),
        ):
            if abs(p) < 1e-8:
                if q < 0.0:
                    valid = False
                    break
                continue
            ratio = q / p
            if p < 0.0:
                t_min = max(t_min, ratio)
            else:
                t_max = min(t_max, ratio)
            if t_min > t_max:
                valid = False
                break
        if not valid:
            continue
        segment = [
            [x0 + t_min * dx, y0 + t_min * dy],
            [x0 + t_max * dx, y0 + t_max * dy],
        ]
        if not clipped or np.linalg.norm(np.asarray(clipped[-1]) - np.asarray(segment[0])) > 1e-4:
            clipped.append(segment[0])
        if np.linalg.norm(np.asarray(clipped[-1]) - np.asarray(segment[1])) > 1e-4:
            clipped.append(segment[1])
    scale_x = float(target_w) / max(float(image_w), 1.0)
    scale_y = float(target_h) / max(float(image_h - top_crop), 1.0)
    transformed = []
    for x, y in clipped:
        target_x = float(x) * scale_x
        target_y = (float(y) - float(top_crop)) * scale_y
        transformed.append(
            [
                float(np.clip(target_x, 0.0, target_w - 1.0)),
                float(np.clip(target_y, 0.0, target_h - 1.0)),
            ]
        )
    transformed = clean_polyline(transformed, sort_points=False, min_points=2)
    if len(transformed) < 2:
        raise ValueError('Lane has fewer than two points after top crop normalization')
    return resized, transformed


def lane_strip_crop_with_mode(
    image,
    points,
    crop_size=(288, 128),
    strip_width=128,
    crop_mode='fixed',
    strip_reference_width=2560,
    strip_min_width=64,
    strip_max_width=192,
    normalized_size=(1024, 544),
    top_crop_ratio=0.08,
    sort_points=True,
):
    work_image = image
    work_points = points
    if str(crop_mode) == 'normalized':
        work_image, work_points = normalize_image_and_points(
            image,
            points,
            normalized_size=normalized_size,
            top_crop_ratio=top_crop_ratio,
        )
    width = effective_strip_width(
        work_image.shape[1],
        strip_width=strip_width,
        crop_mode=crop_mode,
        strip_reference_width=strip_reference_width,
        strip_min_width=strip_min_width,
        strip_max_width=strip_max_width,
    )
    crop = lane_strip_crop(
        work_image,
        work_points,
        crop_size=crop_size,
        strip_width=width,
        sort_points=sort_points,
    )
    return crop, width


def crop_to_tensor(crop):
    crop = np.ascontiguousarray(crop.astype(np.float32) / 255.0)
    crop = np.transpose(crop, (2, 0, 1))
    return crop


def polyline_to_mask(points, image_shape, width=20):
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = clean_polyline(points, sort_points=False, min_points=2)
    if len(pts) < 2:
        return mask
    arr = np.asarray(pts, dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
    cv2.polylines(mask, [arr.astype(np.int32)], False, 1, thickness=int(width))
    return mask


def lane_iou(points_a, points_b, image_shape, width=20):
    mask_a = polyline_to_mask(points_a, image_shape, width=width)
    mask_b = polyline_to_mask(points_b, image_shape, width=width)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _color_for_label(label):
    value = str(label)
    for name, color in LABEL_COLORS.items():
        if name in value:
            return color
    return LABEL_COLORS['unknown']


def draw_lanes(image, lanes, thickness=2, draw_scores=True):
    canvas = image.copy()
    for lane in lanes:
        points = clean_polyline(lane.get('points', []), sort_points=False, min_points=2)
        if len(points) < 2:
            continue
        label = lane.get('label', lane.get('gt_label', 'lane'))
        color = _color_for_label(label)
        pts = np.asarray(points, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, int(thickness), lineType=cv2.LINE_AA)
        text = str(label)
        if draw_scores:
            if 'label_score' in lane:
                text += f' {float(lane["label_score"]):.2f}'
            if 'lane_score' in lane:
                text += f'/{float(lane["lane_score"]):.2f}'
        anchor = tuple(pts[0].tolist())
        cv2.putText(
            canvas,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return canvas
