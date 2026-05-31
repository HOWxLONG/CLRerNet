
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from libs.lane_classifier.crop import CLASSES, load_lanes
from libs.utils.highway_data import resolve_image_path


def parse_args():
    parser = argparse.ArgumentParser(description='Build keyed multi-root highway lane splits.')
    parser.add_argument('--source', action='append', required=True, help='Dataset source in KEY=DATA_ROOT format.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--prefer-on-duplicate', default=None)
    return parser.parse_args()


def parse_sources(values):
    sources = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'--source must be KEY=PATH, got: {value}')
        key, path = value.split('=', 1)
        key = key.strip()
        path = Path(path.strip())
        if not key or not str(path):
            raise ValueError(f'--source must be KEY=PATH, got: {value}')
        if key in sources:
            raise ValueError(f'duplicate source key: {key}')
        sources[key] = path
    return sources


def read_split(root, split):
    split_file = root / 'splits' / f'{split}.txt'
    if not split_file.exists():
        raise FileNotFoundError(f'Missing split file: {split_file}')
    with split_file.open('r', encoding='utf-8') as f:
        return [line.strip().split()[0].lstrip('/') for line in f if line.strip()]


def image_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def label_counts(json_path):
    counts = Counter()
    for lane in load_lanes(json_path, allowed_labels=CLASSES, min_points=2, sort_points=True):
        counts[lane['label']] += 1
    return counts


def main():
    args = parse_args()
    sources = parse_sources(args.source)
    if args.prefer_on_duplicate and args.prefer_on_duplicate not in sources:
        raise ValueError(f'Unknown preferred source: {args.prefer_on_duplicate}')

    entries = []
    input_counts = {}
    for source_key, root in sources.items():
        split_counts = {}
        for split in ('train', 'val', 'test'):
            rels = read_split(root, split)
            split_counts[split] = len(rels)
            for rel in rels:
                img_path = resolve_image_path({source_key: root}, source_key, rel)
                entries.append(
                    {
                        'source': source_key,
                        'root': str(root),
                        'split': split,
                        'rel': img_path.relative_to(root).as_posix(),
                        'image': str(img_path),
                        'json': str(img_path.with_suffix('.json')),
                        'sha256': image_sha256(img_path),
                    }
                )
        input_counts[source_key] = split_counts

    by_hash = defaultdict(list)
    for entry in entries:
        by_hash[entry['sha256']].append(entry)

    keep_ids = set()
    duplicate_groups = []
    dropped = []
    for sha, group in by_hash.items():
        if len(group) == 1:
            keep_ids.add(id(group[0]))
            continue
        preferred = [entry for entry in group if entry['source'] == args.prefer_on_duplicate]
        keep = preferred[0] if preferred else group[0]
        keep_ids.add(id(keep))
        duplicate_groups.append(
            {
                'sha256': sha,
                'kept': {k: keep[k] for k in ('source', 'split', 'rel')},
                'members': [{k: entry[k] for k in ('source', 'split', 'rel')} for entry in group],
            }
        )
        for entry in group:
            if entry is not keep:
                dropped.append({k: entry[k] for k in ('source', 'split', 'rel')})

    kept = [entry for entry in entries if id(entry) in keep_ids]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_items = {split: [] for split in ('train', 'val', 'test')}
    source_split_items = defaultdict(lambda: defaultdict(list))
    split_label_counts = {split: Counter() for split in ('train', 'val', 'test')}
    for entry in kept:
        line = f'{entry["source"]}\t{entry["rel"]}'
        split_items[entry['split']].append(line)
        source_split_items[entry['source']][entry['split']].append(line)
        split_label_counts[entry['split']].update(label_counts(Path(entry['json'])))

    for split, lines in split_items.items():
        with (out_dir / f'{split}.txt').open('w', encoding='utf-8') as f:
            for line in sorted(lines):
                f.write(f'{line}\n')
    for source_key, split_map in source_split_items.items():
        for split, lines in split_map.items():
            with (out_dir / f'{split}_{source_key}.txt').open('w', encoding='utf-8') as f:
                for line in sorted(lines):
                    f.write(f'{line}\n')

    report = {
        'sources': {key: str(path) for key, path in sources.items()},
        'prefer_on_duplicate': args.prefer_on_duplicate,
        'input_counts': input_counts,
        'output_counts': {split: len(lines) for split, lines in split_items.items()},
        'output_total': sum(len(lines) for lines in split_items.values()),
        'label_counts': {split: dict(counts) for split, counts in split_label_counts.items()},
        'duplicate_groups': duplicate_groups,
        'dropped_duplicates': dropped,
    }
    with (out_dir / 'merge_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report['output_counts'], ensure_ascii=False, indent=2))
    print(f'Wrote merged splits and report to {out_dir}')


if __name__ == '__main__':
    main()
