import argparse
import json
from pathlib import Path

import cv2
import torch

from libs.lane_classifier.infer import build_locator, locate_lanes
from libs.utils.highway_data import (
    normalize_data_roots,
    parse_data_roots_arg,
    parse_split_line,
    resolve_image_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Freeze Stage 1 and export predicted lanes for deployment-matched Stage 2 crops.'
    )
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--data-roots', nargs='*', default=None, help='KEY=PATH entries for a multi-root split.')
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--det-config', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--det-conf-thr', type=float, default=0.35)
    parser.add_argument('--nms-thres', type=float, default=50.0)
    parser.add_argument('--nms-topk', type=int, default=8)
    parser.add_argument('--no-nms', action='store_true')
    parser.add_argument('--max-images', type=int, default=None, help='Optional smoke-test limit.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob('*.json'))
    if existing and not args.overwrite:
        raise FileExistsError(
            f'{out_dir} already contains {len(existing)} JSON files; use a new directory or --overwrite'
        )

    roots_arg = parse_data_roots_arg(args.data_roots)
    data_roots, default_root_key = normalize_data_roots(args.data_root, roots_arg)
    multi_root = len(data_roots) > 1
    entries = []
    with Path(args.split_file).open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            root_key, relative_path = parse_split_line(line, data_roots, default_root_key)
            entries.append((root_key, resolve_image_path(data_roots, root_key, relative_path)))
    if args.max_images is not None:
        entries = entries[: max(int(args.max_images), 0)]
    if not entries:
        raise RuntimeError(f'No images resolved from {args.split_file}')

    device = args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'
    locator = build_locator(
        args.det_config,
        args.det_checkpoint,
        device=device,
        det_conf_thr=args.det_conf_thr,
        nms_thres=args.nms_thres,
        nms_topk=args.nms_topk,
        use_nms=not args.no_nms,
    )
    rows = []
    for index, (root_key, image_path) in enumerate(entries, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        lanes = locate_lanes(locator, image_path, image, score_thr=args.score_thr)
        stem = f'{root_key}__{image_path.stem}' if multi_root else image_path.stem
        output_path = out_dir / f'{stem}.json'
        payload = {
            'image': image_path.name,
            'source': root_key,
            'lanes': lanes,
        }
        with output_path.open('w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        rows.append(
            {
                'source': root_key,
                'image': str(image_path),
                'prediction': str(output_path),
                'num_lanes': len(lanes),
            }
        )
        print(f'[{index}/{len(entries)}] {root_key}/{image_path.name}: {len(lanes)} lanes')

    manifest = {
        'split_file': str(args.split_file),
        'data_roots': {key: str(value) for key, value in data_roots.items()},
        'det_config': str(args.det_config),
        'det_checkpoint': str(args.det_checkpoint),
        'score_thr': float(args.score_thr),
        'det_conf_thr': float(args.det_conf_thr),
        'nms_thres': float(args.nms_thres),
        'nms_topk': int(args.nms_topk),
        'use_nms': not args.no_nms,
        'num_images': len(rows),
        'num_lanes': sum(row['num_lanes'] for row in rows),
        'predictions': rows,
    }
    with (out_dir / 'manifest.json').open('w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print('manifest:', out_dir / 'manifest.json')


if __name__ == '__main__':
    main()
