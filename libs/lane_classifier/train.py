import argparse
import csv
import json
import random
import sys
from pathlib import Path

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
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--class-balance', choices=['none', 'loss', 'sampler'], default='loss')
    parser.add_argument('--debug-crops', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume-from', default=None, help='Optional classifier checkpoint to continue training.')
    return parser.parse_args()


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


def evaluate(model, loader, criterion, device):
    model.eval()
    y_true, y_pred, rows = [], [], []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            total_loss += float(loss.item()) * int(labels.numel())
            total += int(labels.numel())
            for i, meta in enumerate(batch['metas']):
                label_idx = int(labels[i].cpu())
                pred_idx = int(pred[i].cpu())
                rows.append(
                    {
                        'image': meta['image'],
                        'lane_index': int(meta['lane_index']),
                        'label': CLASSES[label_idx],
                        'prediction': CLASSES[pred_idx],
                        'confidence': float(probs[i, pred_idx].cpu()),
                    }
                )
                y_true.append(label_idx)
                y_pred.append(pred_idx)
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = total_loss / max(total, 1)
    return metrics, rows


def save_checkpoint(path, model, args, epoch, metrics):
    payload = {
        'state_dict': model.state_dict(),
        'classes': CLASSES,
        'crop_size': (args.crop_height, args.crop_width),
        'strip_width': args.strip_width,
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
    train_set = LaneStripDataset(
        args.data_root,
        Path(args.split_root) / 'train.txt',
        crop_size=crop_size,
        strip_width=args.strip_width,
        augment=True,
    )
    val_set = LaneStripDataset(
        args.data_root,
        Path(args.split_root) / 'val.txt',
        crop_size=crop_size,
        strip_width=args.strip_width,
        augment=False,
    )
    if args.debug_crops > 0:
        train_set.save_debug_crops(work_dir / 'debug_crops' / 'train', args.debug_crops)
        val_set.save_debug_crops(work_dir / 'debug_crops' / 'val', min(args.debug_crops, len(val_set)))

    train_loader = make_loader(train_set, args, train=True)
    val_loader = make_loader(val_set, args, train=False)

    model = build_model(num_classes=len(CLASSES), dropout=args.dropout).to(device)
    weights = None
    if args.class_balance == 'loss':
        weights = torch.tensor(train_set.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
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
        'epochs': [],
    }
    best_f1 = -1.0
    best_metrics = None
    start_epoch = 1
    metrics_path = work_dir / 'metrics.json'
    if args.resume_from:
        checkpoint = torch.load(str(args.resume_from), map_location=device)
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
        val_metrics, val_rows = evaluate(model, val_loader, criterion, device)
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
        save_checkpoint(work_dir / 'last.pth', model, args, epoch, val_metrics)
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            best_metrics = val_metrics
            save_checkpoint(work_dir / 'best.pth', model, args, epoch, val_metrics)
            with (work_dir / 'val_predictions_best.json').open('w', encoding='utf-8') as f:
                json.dump(val_rows, f, ensure_ascii=False, indent=2)
            write_confusion_csv(val_metrics['confusion_matrix'], work_dir / 'confusion_matrix_best.csv')
        with (work_dir / 'metrics.json').open('w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    if best_metrics is not None:
        write_confusion_csv(best_metrics['confusion_matrix'], work_dir / 'confusion_matrix_best.csv')
    print('best macro_f1:', f'{best_f1:.4f}', 'work_dir:', work_dir)


if __name__ == '__main__':
    main()
