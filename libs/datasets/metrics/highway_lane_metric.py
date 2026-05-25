import cv2
import numpy as np
from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class HighwayLaneMetric(BaseMetric):
    """Simple class-agnostic lane locator metric for pipeline sanity checks."""

    def __init__(self, img_w=800, img_h=320, iou_thresholds=(0.3, 0.5), lane_width=8):
        super().__init__()
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.iou_thresholds = tuple(float(x) for x in iou_thresholds)
        self.lane_width = int(lane_width)

    def process(self, data_batch, data_samples):
        self.results.extend(data_samples)

    def compute_metrics(self, results):
        summary = {f'TP@{thr}': 0 for thr in self.iou_thresholds}
        summary.update({f'FP@{thr}': 0 for thr in self.iou_thresholds})
        summary.update({f'FN@{thr}': 0 for thr in self.iou_thresholds})
        pred_total = 0
        gt_total = 0

        for result in results:
            pred_lanes = result.get('lanes', [])
            gt_lanes = result.get('metainfo', {}).get('gt_points', [])
            pred_masks = [self._pred_to_mask(lane) for lane in pred_lanes]
            pred_masks = [mask for mask in pred_masks if mask is not None]
            gt_masks = [self._gt_to_mask(lane) for lane in gt_lanes]
            gt_masks = [mask for mask in gt_masks if mask is not None]
            pred_total += len(pred_masks)
            gt_total += len(gt_masks)
            ious = self._cross_iou(pred_masks, gt_masks)
            for thr in self.iou_thresholds:
                matches = self._greedy_match(ious, thr)
                summary[f'TP@{thr}'] += matches
                summary[f'FP@{thr}'] += len(pred_masks) - matches
                summary[f'FN@{thr}'] += len(gt_masks) - matches

        metrics = {'pred_lanes': pred_total, 'gt_lanes': gt_total}
        eps = 1e-8
        for thr in self.iou_thresholds:
            tp = summary[f'TP@{thr}']
            fp = summary[f'FP@{thr}']
            fn = summary[f'FN@{thr}']
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            metrics[f'Precision@{thr}'] = precision
            metrics[f'Recall@{thr}'] = recall
            metrics[f'F1@{thr}'] = f1
        return metrics

    def _pred_to_mask(self, lane):
        ys = np.linspace(0, 1, 72)
        try:
            xs = lane(ys)
        except Exception:
            points = np.asarray(lane, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 2:
                return None
            xs = points[:, 0]
            ys = points[:, 1]
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (xs < 1) & (ys >= 0) & (ys <= 1)
        if valid.sum() < 2:
            return None
        pts = np.stack([xs[valid] * self.img_w, ys[valid] * self.img_h], axis=1)
        return self._draw_points(pts)

    def _gt_to_mask(self, lane):
        coords = np.asarray(lane, dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] < 2:
            return None
        return self._draw_points(coords)

    def _draw_points(self, points):
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        pts = np.round(points).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, self.img_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, self.img_h - 1)
        if len(pts) < 2:
            return None
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 1, self.lane_width)
        return mask.astype(bool)

    def _cross_iou(self, pred_masks, gt_masks):
        if not pred_masks or not gt_masks:
            return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
        ious = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
        for i, pred in enumerate(pred_masks):
            for j, gt in enumerate(gt_masks):
                union = np.logical_or(pred, gt).sum()
                if union > 0:
                    ious[i, j] = np.logical_and(pred, gt).sum() / union
        return ious

    def _greedy_match(self, ious, threshold):
        if ious.size == 0:
            return 0
        matched_pred = set()
        matched_gt = set()
        pairs = []
        for i in range(ious.shape[0]):
            for j in range(ious.shape[1]):
                if ious[i, j] >= threshold:
                    pairs.append((ious[i, j], i, j))
        pairs.sort(reverse=True)
        matches = 0
        for _, i, j in pairs:
            if i in matched_pred or j in matched_gt:
                continue
            matched_pred.add(i)
            matched_gt.add(j)
            matches += 1
        return matches
