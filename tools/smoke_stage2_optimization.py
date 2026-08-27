import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from libs.lane_classifier.dataset import LaneStripDataset, lane_collate
from libs.lane_classifier.model import build_model
from libs.utils.highway_data import parse_data_roots_arg


def parse_args():
    parser = argparse.ArgumentParser(description='Run one real Stage 2 optimization step on lane crops.')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--data-roots', nargs='*', default=None)
    parser.add_argument('--split-file', required=True)
    parser.add_argument('--prediction-dir', default=None)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-type', choices=('legacy', 'sequence_fusion'), default='sequence_fusion')
    parser.add_argument('--crop-mode', choices=('fixed', 'scaled', 'normalized'), default='scaled')
    parser.add_argument('--strip-width', type=int, default=128)
    parser.add_argument('--benchmark-iterations', type=int, default=50)
    return parser.parse_args()


def benchmark_forward(model, image, device, iterations):
    model.eval()
    iterations = max(int(iterations), 1)
    with torch.no_grad():
        for _ in range(10):
            model(image)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            for _ in range(iterations):
                model(image)
            finished.record()
            torch.cuda.synchronize()
            return float(started.elapsed_time(finished) / iterations)
        started = time.perf_counter()
        for _ in range(iterations):
            model(image)
        return float((time.perf_counter() - started) * 1000.0 / iterations)


def main():
    args = parse_args()
    roots = parse_data_roots_arg(args.data_roots)
    dataset = LaneStripDataset(
        data_root=args.data_root,
        data_roots=roots,
        split_file=args.split_file,
        prediction_dir=args.prediction_dir,
        predicted_crop_prob=0.7,
        prediction_iou_threshold=0.3,
        crop_mode=args.crop_mode,
        strip_width=args.strip_width,
        augment=True,
    )
    predicted_indices = [
        index for index, sample in enumerate(dataset.samples) if sample.get('predicted_points') is not None
    ]
    remaining_indices = [
        index for index, sample in enumerate(dataset.samples) if sample.get('predicted_points') is None
    ]
    indices = (predicted_indices + remaining_indices)[: int(args.batch_size)]
    batch = lane_collate([dataset[index] for index in indices])

    device = args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'
    model = build_model(model_type=args.model_type).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    images = batch['images'].to(device)
    labels = batch['labels'].to(device)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else 0
    example = images[:1]
    sequence_ms = benchmark_forward(model, example, device, args.benchmark_iterations)
    legacy_model = build_model(model_type='legacy').to(device)
    legacy_ms = benchmark_forward(legacy_model, example, device, args.benchmark_iterations)

    report = {
        'model_type': args.model_type,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'dataset_samples': len(dataset),
        'dataset_class_counts': dataset.class_counts.tolist(),
        'matched_prediction_samples': len(predicted_indices),
        'batch_shape': list(images.shape),
        'batch_crop_sources': [meta['crop_source'] for meta in batch['metas']],
        'effective_strip_widths': [meta['effective_strip_width'] for meta in batch['metas']],
        'loss': float(loss.detach().cpu().item()),
        'optimization_step_seconds': float(elapsed),
        'peak_cuda_memory_bytes': peak_memory,
        'logits_finite': bool(torch.isfinite(logits).all().item()),
        'classifier_forward_ms': {
            'legacy': legacy_ms,
            'sequence_fusion': sequence_ms,
            'relative_increase': (sequence_ms - legacy_ms) / max(legacy_ms, 1e-9),
            'iterations': int(args.benchmark_iterations),
            'batch_size': 1,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
