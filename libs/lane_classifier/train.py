import argparse
import csv
import json
import random
import sys
from pathlib import Path

from libs.utils.highway_data import parse_data_roots_arg

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from libs.lane_classifier.crop import CLASSES
    from libs.lane_classifier.dataset import LaneStripDataset, lane_collate
    from libs.lane_classifier.model import build_model
else:
    from .crop import CLASSES
    from .dataset import LaneStripDataset, lane_collate
    from .model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 2 lane instance classifier.')
    parser.add_argument('--data-root', default='dataset/lane_706_20260518')
    parser.add_argument('--data-roots', nargs='*', default=None, help='Optional KEY=PATH list for multi-root splits.')
    parser.add_argument('--split-root', default='dataset/lane_706_20260518/splits')
    parser.add_argument('--work-dir', default='work_dirs/lane_classifier/lane_706_20260518_stage2_strip_288x128_w128')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--crop-height', type=int, default=288)
    parser.add_argument('--crop-width', type=int, default=128)
    parser.add_argument('--strip-width', type=int, default=128)
    parser.add_argument('--model-type', choices=['legacy', 'sequence_fusion'], default='sequence_fusion')
    parser.add_argument('--train-pred-dir', default=None, help='Stage 1 predictions for mixed train crops.')
    parser.add_argument('--val-pred-dir', default=None, help='Stage 1 predictions for deployment-like validation crops.')
    parser.add_argument('--predicted-crop-prob', type=float, default=0.7)
    parser.add_argument('--prediction-iou-thr', type=float, default=0.3)
    parser.add_argument('--match-width', type=int, default=20)
    parser.add_argument('--crop-mode', choices=['fixed', 'scaled', 'normalized'], default='scaled')
    parser.add_argument('--strip-reference-width', type=int, default=2560)
    parser.add_argument('--strip-min-width', type=int, default=64)
    parser.add_argument('--strip-max-width', type=int, default=192)
    parser.add_argument('--normalized-width', type=int, default=1024)
    parser.add_argument('--normalized-height', type=int, default=544)
    parser.add_argument('--top-crop-ratio', type=float, default=0.08)
    parser.add_argument('--normal-offset-ratio', type=float, default=0.12)
    parser.add_argument('--endpoint-truncate-ratio', type=float, default=0.15)
    parser.add_argument('--strip-width-jitter', type=float, nargs=2, default=(0.8, 1.2))
    parser.add_argument(
        '--keep-unmatched-val-gt',
        action='store_true',
        help='Keep GT crops without a matched Stage 1 lane when --val-pred-dir is set.',
    )
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--class-balance', choices=['none', 'loss', 'sampler'], default='loss')
    parser.add_argument('--debug-crops', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--init-from',
        default=None,
        help='Optional classifier checkpoint used only to initialize model weights for a fresh run.',
    )
    parser.add_argument('--resume-from', default=None, help='Optional classifier checkpoint to continue training.')
    args = parser.parse_args()
    if args.init_from and args.resume_from:
        parser.error('--init-from and --resume-from are mutually exclusive')
    if not 0.0 <= args.predicted_crop_prob <= 1.0:
        parser.error('--predicted-crop-prob must be in [0, 1]')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    if len(y_true) == 0:
        cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'per_class': {},
            'confusion_matrix': cm.tolist(),
        }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for idx, name in enumerate(CLASSES):
        per_class[name] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx]),
        }
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
    }


def write_confusion_csv(cm, path):
    path = Path(path)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\pred', *CLASSES])
        for idx, row in enumerate(cm):
            writer.writerow([CLASSES[idx], *row])


def make_loader(dataset, args, train=False):
    sampler = None
    shuffle = train
    if train and args.class_balance == 'sampler':
        counts = dataset.class_counts.astype(np.float32)
        weights = [1.0 / max(counts[s['target']], 1.0) for s in dataset.samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lane_collate,
    )


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        images = batch['images'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(labels.numel())
        total += int(labels.numel())
    return total_loss / max(total, 1)


def expected_calibration_error(probs, labels, bins=15):
    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(labels)
    error = torch.zeros((), dtype=torch.float32)
    boundaries = torch.linspace(0.0, 1.0, int(bins) + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.float().mean() * (
                correct[selected].float().mean() - confidence[selected].mean()
            ).abs()
    return float(error.item())


def rows_and_metrics(logits, labels, metas, loss, temperature=1.0):
    calibrated = logits / max(float(temperature), 1e-4)
    probs = torch.softmax(calibrated, dim=1)
    pred = probs.argmax(dim=1)
    y_true = labels.tolist()
    y_pred = pred.tolist()
    rows = []
    for i, meta in enumerate(metas):
        label_idx = int(labels[i])
        pred_idx = int(pred[i])
        rows.append(
            {
                'source': meta.get('source'),
                'image': meta['image'],
                'lane_index': int(meta['lane_index']),
                'label': CLASSES[label_idx],
                'prediction': CLASSES[pred_idx],
                'confidence': float(probs[i, pred_idx]),
                'label_probs': {name: float(probs[i, j]) for j, name in enumerate(CLASSES)},
                'crop_source': meta.get('crop_source'),
                'effective_strip_width': meta.get('effective_strip_width'),
                'prediction_iou': meta.get('prediction_iou'),
            }
        )
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = float(loss)
    metrics['temperature'] = float(temperature)
    metrics['ece'] = expected_calibration_error(probs, labels)
    return metrics, rows


def fit_temperature(logits, labels):
    logits = logits.detach().float()
    labels = labels.detach().long()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0).item())


def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, all_metas = [], [], []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.numel())
            total += int(labels.numel())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_metas.extend(batch['metas'])
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    loss = total_loss / max(total, 1)
    metrics, rows = rows_and_metrics(logits, labels, all_metas, loss, temperature=1.0)
    return metrics, rows, logits, labels, all_metas


def save_checkpoint(path, model, args, epoch, metrics, temperature=1.0):
    payload = {
        'state_dict': model.state_dict(),
        'classes': CLASSES,
        'crop_size': (args.crop_height, args.crop_width),
        'strip_width': args.strip_width,
        'crop_mode': args.crop_mode,
        'strip_reference_width': args.strip_reference_width,
        'strip_min_width': args.strip_min_width,
        'strip_max_width': args.strip_max_width,
        'normalized_size': (args.normalized_width, args.normalized_height),
        'top_crop_ratio': args.top_crop_ratio,
        'model_type': args.model_type,
        'temperature': float(temperature),
        'dropout': args.dropout,
        'epoch': int(epoch),
        'metrics': metrics,
        'args': vars(args),
    }
    torch.save(payload, str(path))


def main():
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu')

    crop_size = (args.crop_height, args.crop_width)
    data_roots = parse_data_roots_arg(args.data_roots)
    dataset_data_root = None if data_roots else args.data_root
    dataset_kwargs = dict(
        data_roots=data_roots,
        crop_size=crop_size,
        strip_width=args.strip_width,
        prediction_iou_threshold=args.prediction_iou_thr,
        match_width=args.match_width,
        crop_mode=args.crop_mode,
        strip_reference_width=args.strip_reference_width,
        strip_min_width=args.strip_min_width,
        strip_max_width=args.strip_max_width,
        normalized_size=(args.normalized_width, args.normalized_height),
        top_crop_ratio=args.top_crop_ratio,
        normal_offset_ratio=args.normal_offset_ratio,
        endpoint_truncate_ratio=args.endpoint_truncate_ratio,
        strip_width_jitter=args.strip_width_jitter,
    )
    train_set = LaneStripDataset(
        dataset_data_root,
        Path(args.split_root) / 'train.txt',
        prediction_dir=args.train_pred_dir,
        predicted_crop_prob=args.predicted_crop_prob if args.train_pred_dir else 0.0,
        augment=True,
        **dataset_kwargs,
    )
    val_set = LaneStripDataset(
        dataset_data_root,
        Path(args.split_root) / 'val.txt',
        prediction_dir=args.val_pred_dir,
        predicted_crop_prob=1.0 if args.val_pred_dir else 0.0,
        require_prediction=bool(args.val_pred_dir) and not args.keep_unmatched_val_gt,
        augment=False,
        **dataset_kwargs,
    )
    if args.debug_crops > 0:
        train_set.save_debug_crops(work_dir / 'debug_crops' / 'train', args.debug_crops)
        val_set.save_debug_crops(work_dir / 'debug_crops' / 'val', min(args.debug_crops, len(val_set)))

    train_loader = make_loader(train_set, args, train=True)
    val_loader = make_loader(val_set, args, train=False)

    model = build_model(
        num_classes=len(CLASSES),
        dropout=args.dropout,
        model_type=args.model_type,
    ).to(device)
    weights = None
    if args.class_balance == 'loss':
        weights = torch.tensor(train_set.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    if args.init_from:
        checkpoint = torch.load(str(args.init_from), map_location=device)
        checkpoint_model_type = str(checkpoint.get('model_type', 'legacy'))
        if checkpoint_model_type != args.model_type:
            raise ValueError(
                f'--init-from model_type={checkpoint_model_type} does not match '
                f'--model-type={args.model_type}'
            )
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict)
        print('initialized from:', args.init_from)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.02
    )

    history = {
        'classes': CLASSES,
        'train_samples': len(train_set),
        'val_samples': len(val_set),
        'train_class_counts': train_set.class_counts.tolist(),
        'val_class_counts': val_set.class_counts.tolist(),
        'class_weights': train_set.class_weights.tolist(),
        'init_from': args.init_from,
        'model_type': args.model_type,
        'crop_mode': args.crop_mode,
        'train_pred_dir': args.train_pred_dir,
        'val_pred_dir': args.val_pred_dir,
        'epochs': [],
    }
    best_f1 = -1.0
    best_metrics = None
    start_epoch = 1
    metrics_path = work_dir / 'metrics.json'
    if args.resume_from:
        checkpoint = torch.load(str(args.resume_from), map_location=device)
        checkpoint_model_type = str(checkpoint.get('model_type', 'legacy'))
        if checkpoint_model_type != args.model_type:
            raise ValueError(
                f'--resume-from model_type={checkpoint_model_type} does not match '
                f'--model-type={args.model_type}'
            )
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict)
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        if metrics_path.exists():
            history = json.load(metrics_path.open('r', encoding='utf-8'))
            if history.get('epochs'):
                best_record = max(history['epochs'], key=lambda e: e['val']['macro_f1'])
                best_f1 = float(best_record['val']['macro_f1'])
                best_metrics = best_record['val']
        print('resumed from:', args.resume_from, 'start_epoch:', start_epoch, 'best_f1:', f'{best_f1:.4f}')

    print('train samples:', len(train_set), 'class counts:', train_set.class_counts.tolist())
    print('val samples:', len(val_set), 'class counts:', val_set.class_counts.tolist())
    print('device:', device, 'class balance:', args.class_balance)

    if start_epoch > args.epochs:
        print('resume checkpoint is already at or beyond requested epochs:', start_epoch - 1)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_rows, val_logits, val_labels, val_metas = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        record = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val': val_metrics,
            'lr': float(optimizer.param_groups[0]['lr']),
        }
        history['epochs'].append(record)
        print(
            f'epoch {epoch:03d} train_loss={train_loss:.4f} '
            f'val_loss={val_metrics["loss"]:.4f} '
            f'acc={val_metrics["accuracy"]:.4f} macro_f1={val_metrics["macro_f1"]:.4f}'
        )
        save_checkpoint(work_dir / 'last.pth', model, args, epoch, val_metrics, temperature=1.0)
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            temperature = fit_temperature(val_logits, val_labels)
            calibrated_metrics, calibrated_rows = rows_and_metrics(
                val_logits,
                val_labels,
                val_metas,
                val_metrics['loss'],
                temperature=temperature,
            )
            best_metrics = calibrated_metrics
            record['calibration'] = {
                'temperature': temperature,
                'raw_ece': val_metrics['ece'],
                'calibrated_ece': calibrated_metrics['ece'],
            }
            save_checkpoint(
                work_dir / 'best.pth',
                model,
                args,
                epoch,
                calibrated_metrics,
                temperature=temperature,
            )
            with (work_dir / 'val_predictions_best.json').open('w', encoding='utf-8') as f:
                json.dump(calibrated_rows, f, ensure_ascii=False, indent=2)
            write_confusion_csv(
                calibrated_metrics['confusion_matrix'], work_dir / 'confusion_matrix_best.csv'
            )
        with (work_dir / 'metrics.json').open('w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    if best_metrics is not None:
        write_confusion_csv(best_metrics['confusion_matrix'], work_dir / 'confusion_matrix_best.csv')
    print('best macro_f1:', f'{best_f1:.4f}', 'work_dir:', work_dir)


if __name__ == '__main__':
    main()
