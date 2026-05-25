import numpy as np
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TopCrop(BaseTransform):
    """Crop image top and remap lane points into the cropped frame.

    Points outside the crop rectangle are removed. Lanes with fewer than
    ``min_points`` remaining points are removed together with their labels.
    """

    def __init__(self, crop_ratio=0.08, crop_pixels=None, min_points=2):
        if crop_pixels is None and not (0.0 <= float(crop_ratio) < 1.0):
            raise ValueError('crop_ratio must be in [0, 1).')
        self.crop_ratio = float(crop_ratio)
        self.crop_pixels = None if crop_pixels is None else int(crop_pixels)
        self.min_points = int(min_points)

    def _crop_top(self, height):
        if self.crop_pixels is not None:
            return max(0, min(int(self.crop_pixels), height - 1))
        return max(0, min(int(round(height * self.crop_ratio)), height - 1))

    def _filter_lanes(self, results, src_w, src_h, crop_top):
        lanes = results.get('gt_points')
        if lanes is None:
            return

        labels = results.get('gt_lane_labels')
        id_classes = results.get('id_classes')
        id_instances = results.get('id_instances')
        kept_lanes = []
        kept_labels = []
        kept_classes = []
        kept_instances = []
        removed = 0

        for idx, lane in enumerate(lanes):
            coords = []
            pts = np.asarray(lane, dtype=np.float32).reshape(-1, 2)
            for x, y in pts:
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                if x < 0 or x >= src_w or y < crop_top or y >= src_h:
                    continue
                coords.extend([float(x), float(y - crop_top)])
            if len(coords) < self.min_points * 2:
                removed += 1
                continue
            kept_lanes.append(coords)
            if labels is not None:
                kept_labels.append(labels[idx])
            if id_classes is not None:
                kept_classes.append(id_classes[idx])
            if id_instances is not None:
                kept_instances.append(id_instances[idx])

        results['gt_points'] = kept_lanes
        if labels is not None:
            results['gt_lane_labels'] = kept_labels
        if id_classes is not None:
            results['id_classes'] = kept_classes
        if id_instances is not None:
            results['id_instances'] = kept_instances
        results['top_crop_removed_lanes'] = removed

    def _crop_mask(self, mask, crop_top):
        if mask is None:
            return None
        return mask[crop_top:, :]

    def transform(self, results):
        img = results['img']
        src_h, src_w = img.shape[:2]
        crop_top = self._crop_top(src_h)
        cropped = img[crop_top:, :]
        results['img'] = cropped
        results['img_shape'] = cropped.shape
        results['top_crop_px'] = crop_top
        results['top_crop_ratio'] = crop_top / float(src_h)
        results['top_crop_original_shape'] = img.shape

        self._filter_lanes(results, src_w, src_h, crop_top)
        if results.get('gt_masks') is not None:
            results['gt_masks'] = self._crop_mask(results['gt_masks'], crop_top)
        return results


@TRANSFORMS.register_module()
class LanePointFilter(BaseTransform):
    """Remove lane points outside the current image frame."""

    def __init__(self, min_points=2):
        self.min_points = int(min_points)

    def transform(self, results):
        lanes = results.get('gt_points')
        if lanes is None:
            return results

        img_h, img_w = results['img'].shape[:2]
        labels = results.get('gt_lane_labels')
        id_classes = results.get('id_classes')
        id_instances = results.get('id_instances')
        kept_lanes = []
        kept_labels = []
        kept_classes = []
        kept_instances = []
        removed = 0

        for idx, lane in enumerate(lanes):
            coords = []
            pts = np.asarray(lane, dtype=np.float32).reshape(-1, 2)
            for x, y in pts:
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                if x < 0 or x >= img_w or y < 0 or y >= img_h:
                    continue
                coords.extend([float(x), float(y)])
            if len(coords) < self.min_points * 2:
                removed += 1
                continue
            kept_lanes.append(coords)
            if labels is not None:
                kept_labels.append(labels[idx])
            if id_classes is not None:
                kept_classes.append(id_classes[idx])
            if id_instances is not None:
                kept_instances.append(id_instances[idx])

        results['gt_points'] = kept_lanes
        if labels is not None:
            results['gt_lane_labels'] = kept_labels
        if id_classes is not None:
            results['id_classes'] = kept_classes
        if id_instances is not None:
            results['id_instances'] = kept_instances
        results['lane_point_filter_removed_lanes'] = removed
        return results
