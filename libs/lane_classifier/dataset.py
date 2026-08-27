
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from libs.utils.highway_data import (
    normalize_data_roots,
    parse_split_line,
    relative_to_root,
    resolve_image_path,
)

from .crop import (
    CLASSES,
    CLASS_TO_IDX,
    clean_polyline,
    crop_to_tensor,
    effective_strip_width,
    lane_iou,
    lane_strip_crop,
    load_lanes,
    normalize_image_and_points,
)


class LaneStripDataset(Dataset):
    def __init__(
        self,
        data_root=None,
        split_file=None,
        data_roots=None,
        crop_size=(288, 128),
        strip_width=128,
        min_points=2,
        sort_points=True,
        augment=False,
        prediction_dir=None,
        predicted_crop_prob=0.0,
        require_prediction=False,
        prediction_iou_threshold=0.3,
        match_width=20,
        crop_mode='fixed',
        strip_reference_width=2560,
        strip_min_width=64,
        strip_max_width=192,
        normalized_size=(1024, 544),
        top_crop_ratio=0.08,
        normal_offset_ratio=0.12,
        endpoint_truncate_ratio=0.15,
        strip_width_jitter=(0.8, 1.2),
    ):
        self.data_roots, self.default_root_key = normalize_data_roots(data_root, data_roots)
        self.data_root = self.data_roots[self.default_root_key]
        self.multi_root = len(self.data_roots) > 1
        self.split_file = Path(split_file)
        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.strip_width = int(strip_width)
        self.min_points = int(min_points)
        self.sort_points = bool(sort_points)
        self.augment = bool(augment)
        self.prediction_dir = Path(prediction_dir) if prediction_dir else None
        self.predicted_crop_prob = float(predicted_crop_prob)
        self.require_prediction = bool(require_prediction)
        self.prediction_iou_threshold = float(prediction_iou_threshold)
        self.match_width = int(match_width)
        self.crop_mode = str(crop_mode)
        if self.crop_mode not in {'fixed', 'scaled', 'normalized'}:
            raise ValueError(f'Unsupported crop_mode: {self.crop_mode}')
        self.strip_reference_width = int(strip_reference_width)
        self.strip_min_width = int(strip_min_width)
        self.strip_max_width = int(strip_max_width)
        self.normalized_size = (int(normalized_size[0]), int(normalized_size[1]))
        self.top_crop_ratio = float(top_crop_ratio)
        self.normal_offset_ratio = float(normal_offset_ratio)
        self.endpoint_truncate_ratio = float(endpoint_truncate_ratio)
        self.strip_width_jitter = (float(strip_width_jitter[0]), float(strip_width_jitter[1]))
        self.samples = self._build_samples()
        self.class_counts = self._class_counts()
        self.class_weights = self._class_weights()

    def _build_samples(self):
        samples = []
        with self.split_file.open('r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            root_key, img_rel = parse_split_line(line, self.data_roots, self.default_root_key)
            img_path = resolve_image_path(self.data_roots, root_key, img_rel)
            data_root = self.data_roots[root_key]
            json_path = img_path.with_suffix('.json')
            lanes = load_lanes(
                json_path,
                allowed_labels=CLASSES,
                min_points=self.min_points,
                sort_points=self.sort_points,
            )
            predicted_by_gt = self._matched_predictions(root_key, img_path, json_path, lanes)
            img_rel = relative_to_root(img_path, data_root)
            for gt_index, lane in enumerate(lanes):
                predicted = predicted_by_gt.get(gt_index)
                if self.require_prediction and predicted is None:
                    continue
                label = lane['label']
                samples.append(
                    {
                        'source': root_key,
                        'image_path': str(img_path),
                        'json_path': str(json_path),
                        'image': img_path.name,
                        'image_rel': img_rel,
                        'stem': img_path.stem,
                        'lane_index': lane['lane_index'],
                        'label': label,
                        'target': CLASS_TO_IDX[label],
                        'points': lane['points'],
                        'predicted_points': predicted['points'] if predicted else None,
                        'prediction_iou': predicted['iou'] if predicted else None,
                    }
                )
        if not samples:
            raise RuntimeError(f'No lane samples loaded from {self.split_file}')
        return samples

    def _prediction_path(self, root_key, img_path):
        if self.prediction_dir is None:
            return None
        candidates = []
        if self.multi_root:
            candidates.append(self.prediction_dir / f'{root_key}__{img_path.stem}.json')
        candidates.append(self.prediction_dir / f'{img_path.stem}.json')
        return next((path for path in candidates if path.exists()), None)

    def _matched_predictions(self, root_key, img_path, json_path, gt_lanes):
        pred_path = self._prediction_path(root_key, img_path)
        if pred_path is None:
            return {}
        with pred_path.open('r', encoding='utf-8') as f:
            prediction = json.load(f)
        pred_lanes = []
        for lane in prediction.get('lanes', []):
            points = clean_polyline(lane.get('points', []), sort_points=True, min_points=2)
            if len(points) >= 2:
                pred_lanes.append({'points': points})

        with json_path.open('r', encoding='utf-8') as f:
            annotation = json.load(f)
        image_shape = (
            int(annotation.get('imageHeight') or 0),
            int(annotation.get('imageWidth') or 0),
        )
        if image_shape[0] <= 0 or image_shape[1] <= 0:
            image = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f'Failed to read image: {img_path}')
            image_shape = image.shape[:2]

        candidates = []
        for gt_index, gt in enumerate(gt_lanes):
            for pred_index, pred in enumerate(pred_lanes):
                iou = lane_iou(gt['points'], pred['points'], image_shape, width=self.match_width)
                if iou >= self.prediction_iou_threshold:
                    candidates.append((float(iou), gt_index, pred_index))
        candidates.sort(reverse=True)
        used_gt, used_pred, matched = set(), set(), {}
        for iou, gt_index, pred_index in candidates:
            if gt_index in used_gt or pred_index in used_pred:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            matched[gt_index] = {'points': pred_lanes[pred_index]['points'], 'iou': iou}
        return matched

    def _class_counts(self):
        counts = np.zeros(len(CLASSES), dtype=np.int64)
        for sample in self.samples:
            counts[int(sample['target'])] += 1
        return counts

    def _class_weights(self):
        counts = self.class_counts.astype(np.float32)
        total = float(counts.sum())
        weights = np.zeros_like(counts, dtype=np.float32)
        nonzero = counts > 0
        weights[nonzero] = total / (float(len(CLASSES)) * counts[nonzero])
        return weights

    def __len__(self):
        return len(self.samples)

    def _augment_crop(self, crop):
        if random.random() < 0.5:
            crop = np.ascontiguousarray(crop[:, ::-1, :])
        if random.random() < 0.7:
            alpha = random.uniform(0.85, 1.18)
            beta = random.uniform(-18.0, 18.0)
            crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            crop = cv2.GaussianBlur(crop, (3, 3), 0)
        if random.random() < 0.2:
            height, width = crop.shape[:2]
            block_h = random.randint(max(4, height // 20), max(5, height // 6))
            block_w = random.randint(max(4, width // 8), max(5, width // 2))
            y0 = random.randint(0, max(0, height - block_h))
            x0 = random.randint(0, max(0, width - block_w))
            fill = int(random.uniform(0.15, 0.55) * 255)
            crop[y0 : y0 + block_h, x0 : x0 + block_w] = fill
        return crop

    def _effective_strip_width(self, image_width):
        return effective_strip_width(
            image_width,
            strip_width=self.strip_width,
            crop_mode=self.crop_mode,
            strip_reference_width=self.strip_reference_width,
            strip_min_width=self.strip_min_width,
            strip_max_width=self.strip_max_width,
        )

    def _normalize_image_points(self, image, points):
        return normalize_image_and_points(
            image,
            points,
            normalized_size=self.normalized_size,
            top_crop_ratio=self.top_crop_ratio,
        )

    def _jitter_points(self, points, strip_width):
        array = np.asarray(clean_polyline(points, sort_points=True, min_points=2), dtype=np.float32)
        if len(array) < 2:
            return points
        max_drop = int(len(array) * self.endpoint_truncate_ratio)
        if max_drop > 0 and len(array) - 2 * max_drop >= 2:
            drop_bottom = random.randint(0, max_drop)
            drop_top = random.randint(0, max_drop)
            end = len(array) - drop_top if drop_top else len(array)
            array = array[drop_bottom:end]

        previous = np.vstack([array[:1], array[:-1]])
        following = np.vstack([array[1:], array[-1:]])
        tangent = following - previous
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        norm[norm < 1e-4] = 1.0
        tangent /= norm
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        max_offset = float(strip_width) * self.normal_offset_ratio
        endpoints = np.asarray(
            [random.uniform(-max_offset, max_offset), random.uniform(-max_offset, max_offset)],
            dtype=np.float32,
        )
        offsets = np.linspace(endpoints[0], endpoints[1], len(array), dtype=np.float32)
        return (array + normal * offsets[:, None]).tolist()

    def _make_crop(self, sample, use_prediction=None, augment_geometry=None):
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {sample["image_path"]}')
        if use_prediction is None:
            use_prediction = (
                sample.get('predicted_points') is not None
                and random.random() < self.predicted_crop_prob
            )
        points = sample.get('predicted_points') if use_prediction else sample['points']
        if points is None:
            points = sample['points']
            use_prediction = False
        strip_width = self._effective_strip_width(image.shape[1])
        if augment_geometry is None:
            augment_geometry = self.augment
        if augment_geometry and not use_prediction:
            points = self._jitter_points(points, strip_width)
        if augment_geometry:
            strip_width = max(4, int(round(strip_width * random.uniform(*self.strip_width_jitter))))
        if self.crop_mode == 'normalized':
            image, points = self._normalize_image_points(image, points)
        crop = lane_strip_crop(
            image,
            points,
            crop_size=self.crop_size,
            strip_width=strip_width,
            sort_points=self.sort_points,
        )
        return crop, use_prediction, strip_width

    def __getitem__(self, index):
        sample = self.samples[index]
        crop, used_prediction, effective_strip_width = self._make_crop(sample)
        if self.augment:
            crop = self._augment_crop(crop)
        tensor = torch.from_numpy(crop_to_tensor(crop)).float()
        target = torch.tensor(sample['target'], dtype=torch.long)
        meta = {
            'source': sample['source'],
            'image': sample['image'],
            'image_rel': sample['image_rel'],
            'image_path': sample['image_path'],
            'json_path': sample['json_path'],
            'lane_index': sample['lane_index'],
            'label': sample['label'],
            'points': sample['points'],
            'crop_source': 'prediction' if used_prediction else 'gt',
            'effective_strip_width': int(effective_strip_width),
            'prediction_iou': sample.get('prediction_iou'),
        }
        return tensor, target, meta

    def save_debug_crops(self, out_dir, limit=80):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, sample in enumerate(self.samples[: int(limit)]):
            crop, used_prediction, effective_strip_width = self._make_crop(
                sample,
                use_prediction=sample.get('predicted_points') is not None,
                augment_geometry=False,
            )
            label = sample['label']
            text = f'{label} #{sample["lane_index"]}'
            cv2.putText(crop, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(crop, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            name = f'{index:05d}_{sample["source"]}_{label}_{sample["stem"]}_lane{sample["lane_index"]}.jpg'
            path = out_dir / name
            cv2.imwrite(str(path), crop)
            rows.append(
                {
                    **sample,
                    'debug_crop': str(path),
                    'crop_source': 'prediction' if used_prediction else 'gt',
                    'effective_strip_width': int(effective_strip_width),
                }
            )
        with (out_dir / 'index.json').open('w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return out_dir


def lane_collate(batch):
    images, labels, metas = zip(*batch)
    return {
        'images': torch.stack(images, dim=0),
        'labels': torch.stack(labels, dim=0),
        'metas': list(metas),
    }
