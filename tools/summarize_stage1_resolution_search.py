import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESOLUTION_RE = re.compile(r'(?P<w>\d{4})x(?P<h>\d{3})')
SEED_RE = re.compile(r'seed(?P<seed>\d+)')
EPOCH_RE = re.compile(r'epoch[_-]?(?P<epoch>\d+)')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize fixed and grid CLRerNet resolution evaluations.'
    )
    parser.add_argument('--root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tie-threshold', type=float, default=0.003)
    return parser.parse_args()


def extract_identity(report_path, report):
    text = ' '.join(
        [
            str(report_path),
            report.get('config', ''),
            report.get('checkpoint', ''),
        ]
    )
    resolution_match = RESOLUTION_RE.search(text)
    seed_match = SEED_RE.search(text)
    epoch_match = EPOCH_RE.search(report.get('checkpoint', ''))
    if not resolution_match or not seed_match or not epoch_match:
        raise ValueError(f'Cannot identify resolution/seed/epoch: {report_path}')
    width = int(resolution_match.group('w'))
    height = int(resolution_match.group('h'))
    mode = 'grid' if '/grid/' in report_path.as_posix() else 'fixed'
    return {
        'resolution': f'{width}x{height}',
        'width': width,
        'height': height,
        'pixels': width * height,
        'seed': int(seed_match.group('seed')),
        'epoch': int(epoch_match.group('epoch')),
        'mode': mode,
    }


def combo_key(row):
    return (
        float(row['conf_threshold']),
        int(row['nms_topk']),
        float(row['nms_thres']),
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for path in sorted(root.rglob('sweep_results.json')):
        with path.open('r', encoding='utf-8') as f:
            report = json.load(f)
        identity = extract_identity(path, report)
        reports.append((identity, report, path))

    fixed_rows = []
    grid_runs = []
    for identity, report, path in reports:
        if identity['mode'] == 'fixed':
            row = {
                **identity,
                **report['best'],
                'report_path': str(path),
            }
            fixed_rows.append(row)
        else:
            grid_runs.append((identity, report, path))

    fixed_rows.sort(
        key=lambda row: (
            row['seed'],
            row['resolution'],
            row['F1@0.5'],
            row['F1@0.3'],
        )
    )
    write_csv(out_dir / 'fixed_results.csv', fixed_rows)

    grouped = defaultdict(list)
    run_lookup = {}
    for identity, report, path in grid_runs:
        run_key = (
            identity['resolution'],
            identity['seed'],
            identity['epoch'],
        )
        run_lookup[run_key] = (identity, report, path)
        for row in report['results']:
            grouped[(identity['resolution'], combo_key(row))].append(
                (identity, row, path)
            )

    aggregate_rows = []
    for (resolution, combo), values in grouped.items():
        conf_threshold, nms_topk, nms_thres = combo
        identity = values[0][0]
        aggregate_rows.append(
            {
                'resolution': resolution,
                'width': identity['width'],
                'height': identity['height'],
                'pixels': identity['pixels'],
                'num_seeds': len({value[0]['seed'] for value in values}),
                'conf_threshold': conf_threshold,
                'nms_topk': nms_topk,
                'nms_thres': nms_thres,
                'mean_Precision@0.3': mean(
                    value[1]['Precision@0.3'] for value in values
                ),
                'mean_Recall@0.3': mean(
                    value[1]['Recall@0.3'] for value in values
                ),
                'mean_F1@0.3': mean(value[1]['F1@0.3'] for value in values),
                'mean_Precision@0.5': mean(
                    value[1]['Precision@0.5'] for value in values
                ),
                'mean_Recall@0.5': mean(
                    value[1]['Recall@0.5'] for value in values
                ),
                'mean_F1@0.5': mean(value[1]['F1@0.5'] for value in values),
            }
        )

    aggregate_rows.sort(
        key=lambda row: (
            row['mean_F1@0.5'],
            row['mean_F1@0.3'],
            -row['pixels'],
        ),
        reverse=True,
    )
    write_csv(out_dir / 'grid_aggregate.csv', aggregate_rows)

    best_by_resolution = {}
    for row in aggregate_rows:
        best_by_resolution.setdefault(row['resolution'], row)
    resolution_rows = sorted(
        best_by_resolution.values(),
        key=lambda row: (
            row['mean_F1@0.5'],
            row['mean_F1@0.3'],
            -row['pixels'],
        ),
        reverse=True,
    )

    winner = None
    if resolution_rows:
        winner = resolution_rows[0]
        for candidate in resolution_rows[1:]:
            gap = winner['mean_F1@0.5'] - candidate['mean_F1@0.5']
            if gap < args.tie_threshold and candidate['pixels'] < winner['pixels']:
                winner = candidate

    final_run = None
    if winner is not None:
        selected_combo = (
            winner['conf_threshold'],
            winner['nms_topk'],
            winner['nms_thres'],
        )
        candidates = []
        for identity, report, path in grid_runs:
            if identity['resolution'] != winner['resolution']:
                continue
            matching = [
                row
                for row in report['results']
                if combo_key(row) == selected_combo
            ]
            if matching:
                candidates.append(
                    {
                        **identity,
                        **matching[0],
                        'checkpoint': report['checkpoint'],
                        'config': report['config'],
                        'report_path': str(path),
                    }
                )
        if candidates:
            final_run = max(
                candidates,
                key=lambda row: (row['F1@0.5'], row['F1@0.3']),
            )

    summary = {
        'root': str(root),
        'tie_threshold': args.tie_threshold,
        'num_reports': len(reports),
        'best_by_resolution': resolution_rows,
        'winner': winner,
        'final_run': final_run,
    }
    with (out_dir / 'resolution_search_summary.json').open(
        'w', encoding='utf-8'
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        '| resolution | seeds | conf | topk | nms | mean F1@0.3 | mean F1@0.5 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in resolution_rows:
        lines.append(
            f"| {row['resolution']} | {row['num_seeds']} | "
            f"{row['conf_threshold']:.3f} | {row['nms_topk']} | "
            f"{row['nms_thres']:.0f} | {row['mean_F1@0.3']:.4f} | "
            f"{row['mean_F1@0.5']:.4f} |"
        )
    (out_dir / 'resolution_search_summary.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
