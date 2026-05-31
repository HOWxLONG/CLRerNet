
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

from .crop import CLASSES, CLASS_TO_IDX, crop_to_tensor, lane_strip_crop, load_lanes


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
            img_rel = relative_to_root(img_path, data_root)
            for lane in lanes:
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
                    }
                )
        if not samples:
            raise RuntimeError(f'No lane samples loaded from {self.split_file}')
        return samples

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
        return crop

    def _make_crop(self, sample):
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {sample["image_path"]}')
        crop = lane_strip_crop(
            image,
            sample['points'],
            crop_size=self.crop_size,
            strip_width=self.strip_width,
            sort_points=self.sort_points,
        )
        return crop

    def __getitem__(self, index):
        sample = self.samples[index]
        crop = self._make_crop(sample)
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
        }
        return tensor, target, meta

    def save_debug_crops(self, out_dir, limit=80):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, sample in enumerate(self.samples[: int(limit)]):
            crop = self._make_crop(sample)
            label = sample['label']
            text = f'{label} #{sample["lane_index"]}'
            cv2.putText(crop, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(crop, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            name = f'{index:05d}_{sample["source"]}_{label}_{sample["stem"]}_lane{sample["lane_index"]}.jpg'
            path = out_dir / name
            cv2.imwrite(str(path), crop)
            rows.append({**sample, 'debug_crop': str(path)})
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
