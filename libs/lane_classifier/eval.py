import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from libs.lane_classifier.crop import CLASSES, clean_polyline, find_image_path, lane_iou, load_lanes
    from libs.lane_classifier.infer import build_locator, draw_lanes, infer_image
    from libs.lane_classifier.model import load_classifier_checkpoint
else:
    from .crop import CLASSES, clean_polyline, find_image_path, lane_iou, load_lanes
    from .infer import build_locator, draw_lanes, infer_image
    from .model import load_classifier_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate two-stage lane detection + classification.')
    parser.add_argument('--data-root', default='dataset/lane_706_20260518')
    parser.add_argument('--split-file', default='dataset/lane_706_20260518/splits/test.txt')
    parser.add_argument('--pred-dir', default=None, help='Optional existing prediction json dir.')
    parser.add_argument('--det-config', default='configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py')
    parser.add_argument('--det-checkpoint', default='work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth')
    parser.add_argument('--cls-checkpoint', required=True)
    parser.add_argument('--out-dir', default='work_dirs/two_stage_eval/lane_706_20260518')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--det-conf-thr', type=float, default=0.35, help='Override CLRerNet pre-NMS confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=50.0, help='Override CLRerNet lane NMS threshold.')
    parser.add_argument('--nms-topk', type=int, default=8, help='Override max lanes kept by CLRerNet NMS.')
    parser.add_argument('--no-nms', action='store_true', help='Disable CLRerNet NMS.')
    parser.add_argument('--iou-thr', type=float, default=0.3)
    parser.add_argument('--match-width', type=int, default=20)
    parser.add_argument('--crop-height', type=int, default=None)
    parser.add_argument('--crop-width', type=int, default=None)
    parser.add_argument('--strip-width', type=int, default=None)
    parser.add_argument('--error-vis-limit', type=int, default=80)
    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    if not y_true:
        cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        per_class = {c: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0} for c in CLASSES}
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'per_class': per_class, 'confusion_matrix': cm.tolist()}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, name in enumerate(CLASSES):
        per_class[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
    }


def write_confusion_csv(cm, path):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\pred', *CLASSES])
        for i, row in enumerate(cm):
            writer.writerow([CLASSES[i], *row])


def greedy_match(gt_lanes, pred_lanes, image_shape, iou_thr=0.3, width=20):
    candidates = []
    for gi, gt in enumerate(gt_lanes):
        for pi, pred in enumerate(pred_lanes):
            iou = lane_iou(gt['points'], pred.get('points', []), image_shape, width=width)
            if iou >= iou_thr:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_gt, used_pred, matches = set(), set(), []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append({'gt_index': gi, 'pred_index': pi, 'iou': float(iou)})
    return matches, used_gt, used_pred


def load_prediction(pred_dir, image_path):
    pred_path = Path(pred_dir) / f'{Path(image_path).stem}.json'
    if not pred_path.exists():
        return {'image': Path(image_path).name, 'lanes': []}
    with pred_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def draw_error_case(image, gt_lanes, pred_lanes, matches, used_pred):
    gt_draw = []
    pred_draw = []
    match_by_gt = {m['gt_index']: m for m in matches}
    for gi, gt in enumerate(gt_lanes):
        if gi in match_by_gt:
            pred = pred_lanes[match_by_gt[gi]['pred_index']]
            gt_draw.append({'points': gt['points'], 'label': f'GT:{gt["label"]}'})
            pred_draw.append({**pred, 'label': f'PR:{pred.get("label", "unknown")}'})
        else:
            gt_draw.append({'points': gt['points'], 'label': f'MISS:{gt["label"]}'})
    for pi, pred in enumerate(pred_lanes):
        if pi not in used_pred:
            pred_draw.append({**pred, 'label': f'EXTRA:{pred.get("label", "unknown")}'})
    canvas = draw_lanes(image, gt_draw, thickness=7, draw_scores=False)
    canvas = draw_lanes(canvas, pred_draw, thickness=3, draw_scores=True)
    return canvas


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    pred_dir = Path(args.pred_dir) if args.pred_dir else out_dir / 'predictions'
    error_dir = out_dir / 'error_vis'
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'
    locator = None
    classifier = None
    crop_size = None
    strip_width = None
    if args.pred_dir is None:
        locator = build_locator(
            args.det_config,
            args.det_checkpoint,
            device=device,
            det_conf_thr=args.det_conf_thr,
            nms_thres=args.nms_thres,
            nms_topk=args.nms_topk,
            use_nms=not args.no_nms,
        )
        classifier, cls_meta = load_classifier_checkpoint(args.cls_checkpoint, device=device)
        crop_size = tuple(cls_meta.get('crop_size', (288, 128)))
        if args.crop_height is not None and args.crop_width is not None:
            crop_size = (args.crop_height, args.crop_width)
        strip_width = int(args.strip_width or cls_meta.get('strip_width', 128))

    image_paths = []
    with Path(args.split_file).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                image_paths.append(find_image_path(args.data_root, line))

    y_true, y_pred = [], []
    matched_gt = 0
    unmatched_gt = 0
    unmatched_pred = 0
    per_image = []
    error_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        gt_lanes = load_lanes(image_path.with_suffix('.json'))
        if args.pred_dir is None:
            pred, _ = infer_image(
                locator,
                classifier,
                image_path,
                device=device,
                score_thr=args.score_thr,
                crop_size=crop_size,
                strip_width=strip_width,
                crop_dir=None,
            )
            with (pred_dir / f'{image_path.stem}.json').open('w', encoding='utf-8') as f:
                json.dump(pred, f, ensure_ascii=False, indent=2)
        else:
            pred = load_prediction(pred_dir, image_path)
        pred_lanes = pred.get('lanes', [])
        for lane in pred_lanes:
            lane['points'] = clean_polyline(lane.get('points', []), sort_points=True, min_points=2)

        matches, used_gt, used_pred = greedy_match(
            gt_lanes,
            pred_lanes,
            image.shape,
            iou_thr=args.iou_thr,
            width=args.match_width,
        )
        matched_gt += len(matches)
        unmatched_gt += len(gt_lanes) - len(used_gt)
        unmatched_pred += len(pred_lanes) - len(used_pred)

        wrong = False
        for match in matches:
            gt = gt_lanes[match['gt_index']]
            pred_lane = pred_lanes[match['pred_index']]
            gt_idx = CLASSES.index(gt['label'])
            pred_label = pred_lane.get('label', 'unknown')
            pred_idx = CLASSES.index(pred_label) if pred_label in CLASSES else -1
            if pred_idx >= 0:
                y_true.append(gt_idx)
                y_pred.append(pred_idx)
            if pred_idx != gt_idx:
                wrong = True
        has_unmatched = (len(gt_lanes) != len(used_gt)) or (len(pred_lanes) != len(used_pred))
        if (wrong or has_unmatched) and error_count < args.error_vis_limit:
            vis = draw_error_case(image, gt_lanes, pred_lanes, matches, used_pred)
            cv2.imwrite(str(error_dir / f'{image_path.stem}.jpg'), vis)
            error_count += 1
        per_image.append(
            {
                'image': image_path.name,
                'gt': len(gt_lanes),
                'pred': len(pred_lanes),
                'matched': len(matches),
                'unmatched_gt': len(gt_lanes) - len(used_gt),
                'unmatched_pred': len(pred_lanes) - len(used_pred),
            }
        )
        print(f'{image_path.name}: gt={len(gt_lanes)} pred={len(pred_lanes)} matched={len(matches)}')

    metrics = compute_metrics(y_true, y_pred)
    report = {
        'num_images': len(image_paths),
        'iou_threshold': args.iou_thr,
        'match_width': args.match_width,
        'matched_gt': int(matched_gt),
        'unmatched_gt': int(unmatched_gt),
        'unmatched_prediction': int(unmatched_pred),
        'classification': metrics,
        'per_image': per_image,
        'pred_dir': str(pred_dir),
        'error_vis_dir': str(error_dir),
    }
    with (out_dir / 'eval_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_confusion_csv(metrics['confusion_matrix'], out_dir / 'confusion_matrix.csv')
    print('matched_gt:', matched_gt)
    print('unmatched_gt:', unmatched_gt)
    print('unmatched_prediction:', unmatched_pred)
    print('classification_accuracy:', f'{metrics["accuracy"]:.4f}')
    print('macro_f1:', f'{metrics["macro_f1"]:.4f}')
    print('out_dir:', out_dir)


if __name__ == '__main__':
    main()
