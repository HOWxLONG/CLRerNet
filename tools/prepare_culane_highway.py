import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2


IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
ALLOWED_LABELS = ('solid', 'dashed', 'joint')


def parse_args():
    parser = argparse.ArgumentParser(description='Validate highway lane labels and create deterministic splits.')
    parser.add_argument('--data-root', default='dataset/lane_706_20260518')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    return parser.parse_args()


def load_lanes(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lanes = data.get('lanes')
    schema = 'lanes'
    if lanes is None:
        lanes = data.get('shapes', [])
        schema = 'shapes'
    return lanes, schema


def is_bottom_to_top(points):
    ys = [float(p[1]) for p in points if len(p) >= 2]
    return all(ys[i + 1] < ys[i] for i in range(len(ys) - 1))


def image_shape(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return [w, h]


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)

    images_by_stem = defaultdict(list)
    for suffix in IMAGE_SUFFIXES:
        for image_path in data_root.glob(f'*{suffix}'):
            if image_path.is_file():
                images_by_stem[image_path.stem].append(image_path)
    jsons_by_stem = {p.stem: p for p in data_root.glob('*.json') if p.is_file()}

    stems = sorted(set(images_by_stem) | set(jsons_by_stem))
    valid_images = []
    report = {
        'data_root': str(data_root),
        'allowed_labels': list(ALLOWED_LABELS),
        'seed': args.seed,
        'ratios': {'train': args.train_ratio, 'val': args.val_ratio, 'test': args.test_ratio},
        'counts': {},
        'label_counts': Counter(),
        'schema_counts': Counter(),
        'shape_counts': Counter(),
        'points_per_lane': Counter(),
        'image_size_counts': Counter(),
        'errors': [],
        'warnings': [],
    }

    for stem in stems:
        images = images_by_stem.get(stem, [])
        json_path = jsons_by_stem.get(stem)
        if len(images) != 1:
            report['errors'].append({'stem': stem, 'error': 'expected_one_image', 'count': len(images)})
            continue
        if json_path is None:
            report['errors'].append({'stem': stem, 'error': 'missing_json'})
            continue
        image_path = images[0]
        lanes, schema = load_lanes(json_path)
        report['schema_counts'][schema] += 1
        shape = image_shape(image_path)
        if shape is None:
            report['errors'].append({'stem': stem, 'error': 'image_read_failed', 'image': image_path.name})
            continue
        report['image_size_counts'][f'{shape[0]}x{shape[1]}'] += 1

        sample_ok = True
        for lane_idx, lane in enumerate(lanes):
            label = str(lane.get('label', '')).lower()
            points = lane.get('points', [])
            shape_type = lane.get('shape_type', 'lanes_polyline')
            report['label_counts'][label] += 1
            report['shape_counts'][shape_type] += 1
            report['points_per_lane'][str(len(points))] += 1
            if label not in ALLOWED_LABELS:
                report['errors'].append({'stem': stem, 'lane_idx': lane_idx, 'error': 'bad_label', 'label': label})
                sample_ok = False
            if len(points) < 2:
                report['errors'].append({'stem': stem, 'lane_idx': lane_idx, 'error': 'too_few_points', 'points': len(points)})
                sample_ok = False
            if len(points) >= 2 and not is_bottom_to_top(points):
                report['warnings'].append({'stem': stem, 'lane_idx': lane_idx, 'label': label, 'warning': 'points_not_strictly_bottom_to_top'})
        if sample_ok:
            valid_images.append(image_path.relative_to(data_root).as_posix())

    rng = random.Random(args.seed)
    rng.shuffle(valid_images)
    n_total = len(valid_images)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    train = sorted(valid_images[:n_train])
    val = sorted(valid_images[n_train:n_train + n_val])
    test = sorted(valid_images[n_train + n_val:])

    split_map = {'train': train, 'val': val, 'test': test}
    for split, items in split_map.items():
        with open(out_dir / f'{split}.txt', 'w', encoding='utf-8') as f:
            for item in items:
                f.write(f'{item}\n')

    report['counts'] = {
        'images': sum(len(v) for v in images_by_stem.values()),
        'json': len(jsons_by_stem),
        'valid_pairs': n_total,
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'errors': len(report['errors']),
        'warnings': len(report['warnings']),
    }
    for key in ('label_counts', 'schema_counts', 'shape_counts', 'points_per_lane', 'image_size_counts'):
        report[key] = dict(report[key])

    report_path = out_dir / 'prepare_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report['counts'], ensure_ascii=False, indent=2))
    print(f'Wrote splits and report to {out_dir}')


if __name__ == '__main__':
    main()
