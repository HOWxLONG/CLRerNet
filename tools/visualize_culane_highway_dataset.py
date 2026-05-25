import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmdet.registry import DATASETS


COLORS = {
    'solid': (0, 0, 255),
    'dashed': (0, 255, 0),
    'joint': (255, 0, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize HighwayLaneDataset after configured pipeline.')
    parser.add_argument('config')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    parser.add_argument('--out-dir', default='work_dirs/lane_706_20260518_debug/pipeline_vis')
    parser.add_argument('--num-samples', type=int, default=16)
    return parser.parse_args()


def build_dataset(cfg, split, num_samples):
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataloader = getattr(cfg, f'{split}_dataloader')
    dataset_cfg = dataloader.dataset.copy()
    dataset_cfg.max_samples = num_samples
    return DATASETS.build(dataset_cfg)


def tensor_to_image(tensor):
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img.copy()


def draw_lanes(img, points_list, labels):
    for idx, points in enumerate(points_list):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 2:
            continue
        label = labels[idx] if idx < len(labels) else 'lane'
        color = COLORS.get(label, (255, 255, 255))
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_i], False, color, 2)
        x, y = pts_i[0, 0]
        cv2.putText(img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg, args.split, args.num_samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for idx in range(min(args.num_samples, len(dataset))):
        data = dataset[idx]
        img = tensor_to_image(data['inputs'])
        meta = data['data_samples'].metainfo
        img = draw_lanes(img, meta.get('gt_points', []), meta.get('gt_lane_labels', []))
        name = Path(meta.get('sub_img_name', f'{idx:06d}.jpg')).stem
        out_path = out_dir / f'{idx:03d}_{name}.jpg'
        cv2.imwrite(str(out_path), img)
        written.append(str(out_path))

    print(f'Wrote {len(written)} visualization images to {out_dir}')
    for path in written[:10]:
        print(path)


if __name__ == '__main__':
    main()
