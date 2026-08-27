import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from libs.lane_classifier.crop import CLASSES
from libs.lane_classifier.infer import build_locator, classify_lanes, locate_lanes
from libs.lane_classifier.model import build_model, load_classifier_checkpoint
from libs.utils.highway_data import (
    normalize_data_roots,
    parse_data_roots_arg,
    parse_split_line,
    resolve_image_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare old and optimized Stage 2 whole-image latency.')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--data-roots', nargs='*', default=None)
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--det-config', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--legacy-cls-checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--det-conf-thr', type=float, default=0.35)
    parser.add_argument('--nms-thres', type=float, default=50.0)
    parser.add_argument('--nms-topk', type=int, default=8)
    parser.add_argument('--max-images', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    return parser.parse_args()


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def timed(device, function):
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def settings_from_meta(meta):
    return {
        'crop_size': tuple(meta.get('crop_size', (288, 128))),
        'strip_width': int(meta.get('strip_width', 128)),
        'crop_mode': str(meta.get('crop_mode', 'fixed')),
        'strip_reference_width': int(meta.get('strip_reference_width', 2560)),
        'strip_min_width': int(meta.get('strip_min_width', 64)),
        'strip_max_width': int(meta.get('strip_max_width', 192)),
        'normalized_size': tuple(meta.get('normalized_size', (1024, 544))),
        'top_crop_ratio': float(meta.get('top_crop_ratio', 0.08)),
    }


def classify_all(model, image, lanes, settings, device, temperature=1.0, classes=CLASSES):
    classify_lanes(
        model,
        image,
        [lane['points'] for lane in lanes],
        crop_settings=settings,
        device=device,
        temperature=temperature,
        classes=classes,
    )


def main():
    args = parse_args()
    roots_arg = parse_data_roots_arg(args.data_roots)
    roots, default_key = normalize_data_roots(args.data_root, roots_arg)
    images = []
    with Path(args.split_file).open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            key, relative = parse_split_line(line, roots, default_key)
            images.append(resolve_image_path(roots, key, relative))
            if len(images) >= int(args.max_images):
                break
    if not images:
        raise RuntimeError('Latency benchmark needs at least one image')

    device = args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'
    locator = build_locator(
        args.det_config,
        args.det_checkpoint,
        device=device,
        det_conf_thr=args.det_conf_thr,
        nms_thres=args.nms_thres,
        nms_topk=args.nms_topk,
        use_nms=True,
    )
    legacy, legacy_meta = load_classifier_checkpoint(args.legacy_cls_checkpoint, device=device)
    optimized = build_model(model_type='sequence_fusion').to(device).eval()
    legacy_settings = settings_from_meta(legacy_meta)
    optimized_settings = {**legacy_settings, 'crop_mode': 'scaled'}

    prepared = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        lanes = locate_lanes(locator, image_path, image, score_thr=args.score_thr)
        classify_all(
            legacy,
            image,
            lanes,
            legacy_settings,
            device,
            temperature=legacy_meta.get('temperature', 1.0),
            classes=legacy_meta.get('classes', CLASSES),
        )
        classify_all(optimized, image, lanes, optimized_settings, device)
        prepared.append((image_path, image, lanes))

    rows = []
    for image_path, image, lanes in prepared:
        _, stage1_ms = timed(
            device, lambda: locate_lanes(locator, image_path, image, score_thr=args.score_thr)
        )
        legacy_times = []
        optimized_times = []
        for repeat in range(max(int(args.repeats), 1)):
            order = ('legacy', 'optimized') if repeat % 2 == 0 else ('optimized', 'legacy')
            for name in order:
                if name == 'legacy':
                    _, elapsed = timed(
                        device,
                        lambda: classify_all(
                            legacy,
                            image,
                            lanes,
                            legacy_settings,
                            device,
                            temperature=legacy_meta.get('temperature', 1.0),
                            classes=legacy_meta.get('classes', CLASSES),
                        ),
                    )
                    legacy_times.append(elapsed)
                else:
                    _, elapsed = timed(
                        device,
                        lambda: classify_all(optimized, image, lanes, optimized_settings, device),
                    )
                    optimized_times.append(elapsed)
        rows.append(
            {
                'image': str(image_path),
                'lanes': len(lanes),
                'stage1_ms': stage1_ms,
                'legacy_stage2_ms': float(np.median(legacy_times)),
                'optimized_stage2_ms': float(np.median(optimized_times)),
            }
        )

    measured = rows
    averages = {
        key: float(np.mean([row[key] for row in measured]))
        for key in ('stage1_ms', 'legacy_stage2_ms', 'optimized_stage2_ms')
    }
    averages['legacy_total_ms'] = averages['stage1_ms'] + averages['legacy_stage2_ms']
    averages['optimized_total_ms'] = averages['stage1_ms'] + averages['optimized_stage2_ms']
    averages['total_relative_increase'] = (
        averages['optimized_total_ms'] - averages['legacy_total_ms']
    ) / max(averages['legacy_total_ms'], 1e-9)
    report = {
        'warmup_passes': 1,
        'measured_images': len(measured),
        'repeats_per_image': max(int(args.repeats), 1),
        'optimized_model_parameters': sum(parameter.numel() for parameter in optimized.parameters()),
        'averages': averages,
        'rows': rows,
        'note': 'Optimized weights are random; latency is architecture-dependent and remains representative.',
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
