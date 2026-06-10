import argparse
import csv
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.apis import init_detector

from libs.datasets.metrics.highway_lane_metric import HighwayLaneMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run one CLRerNet forward pass and evaluate a post-processing grid.'
    )
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conf-thresholds', nargs='+', type=float, required=True)
    parser.add_argument('--nms-topks', nargs='+', type=int, required=True)
    parser.add_argument('--nms-thres', nargs='+', type=float, required=True)
    parser.add_argument('--max-samples', type=int, default=None)
    return parser.parse_args()


def make_key(conf_threshold, nms_topk, nms_thres):
    return f'conf={conf_threshold:g},topk={nms_topk},nms={nms_thres:g}'


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    cfg.test_dataloader.dataset.data_list = args.split_file
    if args.max_samples is not None:
        cfg.test_dataloader.dataset.max_samples = args.max_samples

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    device = next(model.parameters()).device

    combinations = list(product(
        sorted(set(args.conf_thresholds)),
        sorted(set(args.nms_topks)),
        sorted(set(args.nms_thres)),
    ))
    iou_thresholds = (0.3, 0.5)
    stats = {
        make_key(*combo): {
            'pred_total': 0,
            'gt_total': 0,
            **{f'tp@{thr}': 0 for thr in iou_thresholds},
        }
        for combo in combinations
    }
    metric = HighwayLaneMetric(
        img_w=int(model.bbox_head.img_w),
        img_h=int(model.bbox_head.img_h),
        iou_thresholds=iou_thresholds,
    )
    min_conf_threshold = min(args.conf_thresholds)
    max_topk = max(args.nms_topks)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    forward_seconds = 0.0
    num_images = 0

    with torch.no_grad():
        for batch_index, data_batch in enumerate(dataloader, start=1):
            data = model.data_preprocessor(data_batch, training=False)
            synchronize(device)
            start = time.perf_counter()
            features = model.extract_feat(data['inputs'])
            pred_dict = model.bbox_head(features)[-1]
            synchronize(device)
            forward_seconds += time.perf_counter() - start

            data_samples = data['data_samples']
            num_images += len(data_samples)
            for nms_thres in sorted(set(args.nms_thres)):
                test_cfg = model.bbox_head.test_cfg
                # NMS sorts by descending confidence. Running once at the
                # lowest threshold and largest top-k gives an ordered prefix
                # from which all stricter threshold/top-k variants are exact.
                test_cfg.conf_threshold = float(min_conf_threshold)
                test_cfg.nms_topk = int(max_topk)
                test_cfg.nms_thres = float(nms_thres)
                test_cfg.use_nms = True
                test_cfg.as_lanes = True
                lanes_batch, scores_batch = model.bbox_head.get_lanes(
                    pred_dict,
                    as_lanes=True,
                    extend_bottom=test_cfg.extend_bottom,
                )
                for lanes, scores, sample in zip(
                    lanes_batch, scores_batch, data_samples
                ):
                    scores_list = (
                        scores.detach().cpu().tolist()
                        if hasattr(scores, 'detach')
                        else [float(score) for score in scores]
                    )
                    pred_masks = [metric._pred_to_mask(lane) for lane in lanes]
                    valid_indices = [
                        index
                        for index, mask in enumerate(pred_masks)
                        if mask is not None
                    ]
                    valid_row = {
                        original_index: row_index
                        for row_index, original_index in enumerate(valid_indices)
                    }
                    valid_pred_masks = [
                        pred_masks[index] for index in valid_indices
                    ]
                    gt_masks = [
                        metric._gt_to_mask(lane)
                        for lane in sample.metainfo.get('gt_points', [])
                    ]
                    gt_masks = [mask for mask in gt_masks if mask is not None]
                    full_ious = metric._cross_iou(valid_pred_masks, gt_masks)

                    for conf_threshold in sorted(set(args.conf_thresholds)):
                        threshold_indices = [
                            index
                            for index, score in enumerate(scores_list)
                            if score >= conf_threshold
                        ]
                        for nms_topk in sorted(set(args.nms_topks)):
                            selected = threshold_indices[:nms_topk]
                            selected_rows = [
                                valid_row[index]
                                for index in selected
                                if index in valid_row
                            ]
                            if selected_rows:
                                ious = full_ious[
                                    np.asarray(selected_rows, dtype=np.int64), :
                                ]
                            else:
                                ious = np.zeros(
                                    (0, len(gt_masks)), dtype=np.float32
                                )
                            key = make_key(
                                conf_threshold, nms_topk, nms_thres
                            )
                            stat = stats[key]
                            stat['pred_total'] += len(selected_rows)
                            stat['gt_total'] += len(gt_masks)
                            for iou_threshold in iou_thresholds:
                                stat[f'tp@{iou_threshold}'] += (
                                    metric._greedy_match(
                                        ious, iou_threshold
                                    )
                                )
            print(
                f'batch={batch_index}/{len(dataloader)} images={num_images}',
                flush=True,
            )

    img_w = int(model.bbox_head.img_w)
    img_h = int(model.bbox_head.img_h)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == 'cuda'
        else 0.0
    )
    rows = []
    for combo in combinations:
        conf_threshold, nms_topk, nms_thres = combo
        key = make_key(*combo)
        stat = stats[key]
        metrics = {
            'pred_lanes': stat['pred_total'],
            'gt_lanes': stat['gt_total'],
        }
        eps = 1e-8
        for iou_threshold in iou_thresholds:
            tp = stat[f'tp@{iou_threshold}']
            fp = stat['pred_total'] - tp
            fn = stat['gt_total'] - tp
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            metrics[f'Precision@{iou_threshold}'] = precision
            metrics[f'Recall@{iou_threshold}'] = recall
            metrics[f'F1@{iou_threshold}'] = f1
        rows.append(
            {
                'conf_threshold': conf_threshold,
                'nms_topk': nms_topk,
                'nms_thres': nms_thres,
                **metrics,
                'num_images': num_images,
                'forward_seconds': forward_seconds,
                'ms_per_image': 1000.0 * forward_seconds / max(1, num_images),
                'peak_memory_mb': peak_memory_mb,
            }
        )

    rows.sort(
        key=lambda row: (
            row['F1@0.5'],
            row['F1@0.3'],
            -row['nms_topk'],
            -row['conf_threshold'],
        ),
        reverse=True,
    )
    report = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split_file': args.split_file,
        'img_scale': [img_w, img_h],
        'num_images': num_images,
        'forward_seconds': forward_seconds,
        'ms_per_image': 1000.0 * forward_seconds / max(1, num_images),
        'peak_memory_mb': peak_memory_mb,
        'best': rows[0],
        'results': rows,
    }
    with (out_dir / 'sweep_results.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (out_dir / 'sweep_results.csv').open(
        'w', newline='', encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report['best'], ensure_ascii=False, indent=2))
    print(f'out_dir={out_dir}')


if __name__ == '__main__':
    main()
