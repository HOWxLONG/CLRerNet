import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from libs.datasets.pipelines import Compose
    from libs.lane_classifier.crop import CLASSES, clean_polyline, crop_to_tensor, draw_lanes, lane_strip_crop
    from libs.lane_classifier.model import load_classifier_checkpoint
else:
    from libs.datasets.pipelines import Compose
    from .crop import CLASSES, clean_polyline, crop_to_tensor, draw_lanes, lane_strip_crop
    from .model import load_classifier_checkpoint

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def parse_args():
    parser = argparse.ArgumentParser(description='Two-stage lane locator + lane type classifier inference.')
    parser.add_argument('--input', required=True, help='Input image or folder.')
    parser.add_argument('--det-config', required=True)
    parser.add_argument('--det-checkpoint', required=True)
    parser.add_argument('--cls-checkpoint', required=True)
    parser.add_argument('--out-dir', default='work_dirs/two_stage_infer/lane_706_20260518')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--det-conf-thr', type=float, default=0.35, help='Override CLRerNet pre-NMS confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=50.0, help='Override CLRerNet lane NMS threshold.')
    parser.add_argument('--nms-topk', type=int, default=8, help='Override max lanes kept by CLRerNet NMS.')
    parser.add_argument('--no-nms', action='store_true', help='Disable CLRerNet NMS.')
    parser.add_argument('--crop-height', type=int, default=None)
    parser.add_argument('--crop-width', type=int, default=None)
    parser.add_argument('--strip-width', type=int, default=None)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--save-crops', action='store_true')
    return parser.parse_args()


def collect_images(path, recursive=False):
    path = Path(path)
    if path.is_file():
        return [path]
    pattern = '**/*' if recursive else '*'
    images = [p for p in path.glob(pattern) if p.suffix in IMAGE_SUFFIXES and p.is_file()]
    return sorted(images)


def build_locator(config, checkpoint, device, det_conf_thr=None, nms_thres=None, nms_topk=None, use_nms=None):
    model = init_detector(config, checkpoint, device=device)
    model.bbox_head.test_cfg.as_lanes = False
    if det_conf_thr is not None:
        model.bbox_head.test_cfg.conf_threshold = float(det_conf_thr)
    if nms_thres is not None:
        model.bbox_head.test_cfg.nms_thres = float(nms_thres)
    if nms_topk is not None:
        model.bbox_head.test_cfg.nms_topk = int(nms_topk)
    if use_nms is not None:
        model.bbox_head.test_cfg.use_nms = bool(use_nms)
    model.eval()
    return model


def _detector_data(model, img_path, image):
    data = dict(
        filename=str(img_path),
        sub_img_name=Path(img_path).name,
        lane_json_path=None,
        img=image,
        gt_points=[],
        gt_lane_labels=[],
        id_classes=[],
        id_instances=[],
        img_shape=image.shape,
        ori_shape=image.shape,
    )
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    return pipeline(data)


def _lane_tensor_to_points(lane, score, meta, ori_shape, model):
    arr = lane.detach().cpu().numpy() if hasattr(lane, 'detach') else np.asarray(lane)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    valid &= (arr[:, 0] >= 0.0) & (arr[:, 0] <= 1.0) & (arr[:, 1] >= 0.0) & (arr[:, 1] <= 1.0)
    arr = arr[valid]
    if len(arr) < 2:
        return None

    img_h = float(getattr(model.bbox_head, 'img_h', meta.get('img_shape', ori_shape)[0]))
    img_w = float(getattr(model.bbox_head, 'img_w', meta.get('img_shape', ori_shape)[1]))
    pad_left, pad_top, _, _ = meta.get('resize_pad_pad', (0, 0, 0, 0))
    scale = float(meta.get('resize_pad_scale', 0.0) or 0.0)
    x_model = arr[:, 0] * img_w
    y_model = arr[:, 1] * img_h
    if scale > 0:
        xs = (x_model - float(pad_left)) / scale
        ys = (y_model - float(pad_top)) / scale
    elif 'top_crop_px' in meta:
        crop_top = float(meta.get('top_crop_px', 0.0) or 0.0)
        original_shape = meta.get('top_crop_original_shape', ori_shape)
        orig_h = float(original_shape[0])
        orig_w = float(original_shape[1])
        cropped_h = max(1.0, orig_h - crop_top)
        xs = x_model / img_w * orig_w
        ys = y_model / img_h * cropped_h + crop_top
    else:
        xs = arr[:, 0] * float(ori_shape[1])
        ys = arr[:, 1] * float(ori_shape[0])
    points = np.stack([xs, ys], axis=1)
    in_img = (
        np.isfinite(points[:, 0])
        & np.isfinite(points[:, 1])
        & (points[:, 0] >= 0)
        & (points[:, 0] < ori_shape[1])
        & (points[:, 1] >= 0)
        & (points[:, 1] < ori_shape[0])
    )
    points = points[in_img]
    points = clean_polyline(points.tolist(), sort_points=True, min_points=2)
    if len(points) < 2:
        return None
    return {
        'points': [[float(x), float(y)] for x, y in points],
        'lane_score': float(score),
    }


def locate_lanes(locator, img_path, image, score_thr=0.1):
    data = _detector_data(locator, img_path, image)
    data_ = {'inputs': [data['inputs']], 'data_samples': [data['data_samples']]}
    with torch.no_grad():
        results = locator.test_step(data_)
    result = results[0]
    lanes = result['lanes']
    scores = result['scores']
    if hasattr(scores, 'detach'):
        scores_list = scores.detach().cpu().tolist()
    else:
        scores_list = [float(s) for s in scores]
    meta = result.get('metainfo', data['data_samples'].metainfo)
    outputs = []
    for lane, score in zip(lanes, scores_list):
        if float(score) < float(score_thr):
            continue
        item = _lane_tensor_to_points(lane, score, meta, image.shape, locator)
        if item is not None:
            outputs.append(item)
    return outputs


def classify_lane(classifier, image, points, crop_size, strip_width, device):
    crop = lane_strip_crop(image, points, crop_size=crop_size, strip_width=strip_width, sort_points=True)
    tensor = torch.from_numpy(crop_to_tensor(crop)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax().item())
    return CLASSES[idx], float(probs[idx].item()), crop


def infer_image(locator, classifier, img_path, device, score_thr, crop_size, strip_width, crop_dir=None):
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {img_path}')
    lanes = locate_lanes(locator, img_path, image, score_thr=score_thr)
    output_lanes = []
    for lane_idx, lane in enumerate(lanes):
        label, label_score, crop = classify_lane(
            classifier, image, lane['points'], crop_size=crop_size, strip_width=strip_width, device=device
        )
        out = {
            'points': lane['points'],
            'lane_score': float(lane['lane_score']),
            'label': label,
            'label_score': float(label_score),
        }
        output_lanes.append(out)
        if crop_dir is not None:
            crop_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_dir / f'{Path(img_path).stem}_lane{lane_idx:02d}_{label}.jpg'), crop)
    return {'image': Path(img_path).name, 'lanes': output_lanes}, image


def run_inference(
    input_path,
    det_config,
    det_checkpoint,
    cls_checkpoint,
    out_dir,
    device='cuda:0',
    score_thr=0.1,
    det_conf_thr=None,
    nms_thres=None,
    nms_topk=None,
    use_nms=True,
    crop_size=None,
    strip_width=None,
    recursive=False,
    save_crops=False,
):
    out_dir = Path(out_dir)
    pred_dir = out_dir / 'predictions'
    vis_dir = out_dir / 'visualizations'
    crop_dir = out_dir / 'crops' if save_crops else None
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    locator = build_locator(
        det_config,
        det_checkpoint,
        device=device,
        det_conf_thr=det_conf_thr,
        nms_thres=nms_thres,
        nms_topk=nms_topk,
        use_nms=use_nms,
    )
    classifier, cls_meta = load_classifier_checkpoint(cls_checkpoint, device=device)
    if crop_size is None:
        crop_size = tuple(cls_meta.get('crop_size', (288, 128)))
    if strip_width is None:
        strip_width = int(cls_meta.get('strip_width', 128))

    images = collect_images(input_path, recursive=recursive)
    if not images:
        raise FileNotFoundError(f'No images found in {input_path}')

    summary = []
    for img_path in images:
        pred, image = infer_image(
            locator,
            classifier,
            img_path,
            device=device,
            score_thr=score_thr,
            crop_size=crop_size,
            strip_width=strip_width,
            crop_dir=crop_dir,
        )
        pred_path = pred_dir / f'{img_path.stem}.json'
        with pred_path.open('w', encoding='utf-8') as f:
            json.dump(pred, f, ensure_ascii=False, indent=2)
        vis = draw_lanes(image, pred['lanes'], draw_scores=True)
        vis_path = vis_dir / f'{img_path.stem}.jpg'
        cv2.imwrite(str(vis_path), vis)
        summary.append({'image': str(img_path), 'json': str(pred_path), 'vis': str(vis_path), 'num_lanes': len(pred['lanes'])})
        print(f'{img_path.name}: {len(pred["lanes"])} lanes')

    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'
    crop_size = None
    if args.crop_height is not None and args.crop_width is not None:
        crop_size = (args.crop_height, args.crop_width)
    run_inference(
        input_path=args.input,
        det_config=args.det_config,
        det_checkpoint=args.det_checkpoint,
        cls_checkpoint=args.cls_checkpoint,
        out_dir=args.out_dir,
        device=device,
        score_thr=args.score_thr,
        det_conf_thr=args.det_conf_thr,
        nms_thres=args.nms_thres,
        nms_topk=args.nms_topk,
        use_nms=not args.no_nms,
        crop_size=crop_size,
        strip_width=args.strip_width,
        recursive=args.recursive,
        save_crops=args.save_crops,
    )


if __name__ == '__main__':
    main()
