
import json
from pathlib import Path

import cv2
import numpy as np
from mmdet.registry import DATASETS
from torch.utils.data import Dataset

from libs.datasets.pipelines import Compose
from libs.utils.highway_data import (
    display_image_name,
    normalize_data_roots,
    parse_split_line,
    relative_to_root,
    resolve_image_path,
)


LANE_LABELS = ('solid', 'dashed', 'joint')


@DATASETS.register_module()
class HighwayLaneDataset(Dataset):
    """Lane dataset for flat image/json pairs used by the highway project.

    Stage 1 is class-agnostic: every valid solid/dashed/joint annotation is
    exposed to CLRerNet as the same lane class, while the original lane labels
    are kept in metainfo for debug and Stage 2 classification.
    """

    def __init__(
        self,
        data_root=None,
        data_list=None,
        pipeline=None,
        data_roots=None,
        test_mode=False,
        allowed_labels=LANE_LABELS,
        min_points=2,
        sort_points=True,
        max_samples=None,
        **kwargs,
    ):
        self.data_roots, self.default_root_key = normalize_data_roots(data_root, data_roots)
        self.data_root = self.data_roots[self.default_root_key]
        self.multi_root = len(self.data_roots) > 1
        self.data_list = data_list
        self.test_mode = test_mode
        self.allowed_labels = tuple(allowed_labels)
        self.min_points = int(min_points)
        self.sort_points = bool(sort_points)
        self.img_infos = self.parse_data_list(data_list)
        if max_samples is not None:
            self.img_infos = self.img_infos[: int(max_samples)]
        self.pipeline = Compose(pipeline)
        self.metainfo = {
            'classes': ('lane',),
            'lane_type_classes': self.allowed_labels,
            'data_roots': {key: str(path) for key, path in self.data_roots.items()},
        }
        if not self.test_mode:
            self._set_group_flag()
        print(len(self.img_infos), 'highway lane data are loaded')

    def parse_data_list(self, data_list):
        entries = []
        with open(data_list, 'r', encoding='utf-8') as f:
            for line in f:
                root_key, img_rel = parse_split_line(line, self.data_roots, self.default_root_key)
                if root_key is None:
                    continue
                img_path = resolve_image_path(self.data_roots, root_key, img_rel)
                data_root = self.data_roots[root_key]
                json_path = img_path.with_suffix('.json')
                if not json_path.exists():
                    raise FileNotFoundError(f'Missing json for {img_path}: {json_path}')
                img_rel = relative_to_root(img_path, data_root)
                json_rel = relative_to_root(json_path, data_root)
                entries.append(
                    {
                        'root_key': root_key,
                        'data_root': data_root,
                        'img_path': img_path,
                        'img_rel': img_rel,
                        'sub_img_name': display_image_name(root_key, img_rel, self.multi_root),
                        'json_path': json_path,
                        'json_rel': json_rel,
                    }
                )
        return entries

    def _set_group_flag(self):
        self.flag = np.ones(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.img_infos)

    def _read_lanes(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw_lanes = data.get('lanes')
        if raw_lanes is None:
            raw_lanes = data.get('shapes', [])

        gt_points = []
        lane_labels = []
        for lane in raw_lanes:
            label = str(lane.get('label', '')).lower()
            if label not in self.allowed_labels:
                raise ValueError(f'Unexpected lane label {label!r} in {json_path}')
            points = self._clean_points(lane.get('points', []))
            if len(points) < self.min_points:
                continue
            gt_points.append([coord for point in points for coord in point])
            lane_labels.append(label)
        return gt_points, lane_labels

    def _clean_points(self, points):
        cleaned = []
        for point in points:
            if len(point) < 2:
                continue
            x, y = float(point[0]), float(point[1])
            if np.isfinite(x) and np.isfinite(y):
                cleaned.append([x, y])
        if self.sort_points:
            cleaned = sorted(cleaned, key=lambda p: p[1], reverse=True)

        monotonic = []
        last_y = float('inf')
        for point in cleaned:
            if point[1] < last_y:
                monotonic.append(point)
                last_y = point[1]
        return monotonic

    def _prepare_img(self, idx):
        info = self.img_infos[idx]
        img = cv2.imread(str(info['img_path']))
        if img is None:
            raise FileNotFoundError(f'Failed to read image: {info["img_path"]}')
        ori_shape = img.shape
        gt_points, lane_labels = self._read_lanes(info['json_path'])
        id_classes = [1 for _ in gt_points]
        id_instances = [i + 1 for i in range(len(gt_points))]
        results = dict(
            filename=str(info['img_path']),
            sub_img_name=info['sub_img_name'],
            data_root_key=info['root_key'],
            lane_json_path=str(info['json_path']),
            img=img,
            gt_points=gt_points,
            gt_lane_labels=lane_labels,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=ori_shape,
            ori_shape=ori_shape,
            gt_masks=None,
        )
        return self.pipeline(results)

    def __getitem__(self, idx):
        if self.test_mode:
            return self._prepare_img(idx)
        while True:
            data = self._prepare_img(idx)
            if data is not None:
                return data
            idx = np.random.choice(np.where(self.flag == self.flag[idx])[0])
