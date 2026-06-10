# CLRerNet Highway Two-Stage Lane Detection

Author: Haoxiang Long

Email: haoxianglong@std.uestc.edu.com

This repository adapts CLRerNet to a two-stage highway lane workflow:

- Stage 1: class-agnostic lane locator. It learns lane geometry only.
- Stage 2: lane instance classifier. It classifies each lane as `solid`, `dashed`, or `joint`.

Current verified state: 2026-06-10. The active training data is the merged multi-root set `lane_706_20260518 + lane_812_20260531 + lane_349_20260609`. `dataset/culane_highway` is historical only and is not used for current split, training, or evaluation.

Artifact retention was cleaned on 2026-06-11. The server keeps the selected Stage 1 `epoch_20.pth`, the current Stage 2 `best.pth`, resolution-search reports, four final two-stage evaluations, and six final overlays. Superseded checkpoints and historical inference outputs were removed; historical metrics remain documented below.

## Latest Verified Results

### Data And Split

| Source | Image/JSON pairs | Split | Notes |
| --- | ---: | --- | --- |
| `lane_706_20260518` | 706/706 | fixed existing split: 494/105/107 | historical split kept for comparability |
| `lane_812_20260531` | 812/812 | fixed existing split: 568/121/123 | historical split kept for comparability |
| `lane_349_20260609` | 349/349 | seed 0: 244/52/53 | new split generated on 2026-06-09 |
| merged | 1865 image-level samples | 1304/278/283 | virtual multi-root split, no image copy |

Duplicate image policy: newer source wins, with priority `lane_349_20260609 > lane_812_20260531 > lane_706_20260518`.

Dropped duplicate samples:

```text
lane_706_20260518/train/outside_20250910112526_000206.jpg
lane_812_20260531/train/DsKPdq9pWzgZ1Hu3kaf0ZfLT_202510161514_1.jpg
```

Merged lane label counts:

| Split | solid | dashed | joint |
| --- | ---: | ---: | ---: |
| train | 5088 | 2151 | 1002 |
| val | 1087 | 441 | 259 |
| test | 1124 | 513 | 255 |

### Stage 1 Locator

The controlled resolution search used the same three-source split, training schedule, augmentations, warm start, and effective batch size. Candidate sizes were `1024x544`, `1088x576`, `1152x608`, `1216x640`, `1280x672`, and the existing `1280x704` endpoint. Resolution and post-processing were selected only on merged validation data.

Config:

```text
configs/clrernet/lane_706_812_349_20260609/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1024x544_topcrop8.py
```

Checkpoint:

```text
work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth
```

Selected inference parameters:

```text
conf_threshold=0.35
nms_topk=10
nms_thres=50
```

Validation ranking after per-resolution post-processing search:

| Resolution | Seeds | Best conf/topk/nms | Mean F1@0.3 | Mean F1@0.5 |
| --- | ---: | --- | ---: | ---: |
| `1024x544` | 3 | `0.35/10/50` | 0.8382 | 0.7345 |
| `1088x576` | 3 | `0.35/10/40` | 0.8355 | 0.7216 |
| `1152x608` | 1 | `0.375/10/50` | 0.8215 | 0.7057 |
| `1216x640` | 1 | `0.375/8/60` | 0.8146 | 0.6911 |
| `1280x672` | 1 | `0.375/8/60` | 0.8000 | 0.6842 |
| `1280x704` | 1 | `0.375/8/60` | 0.8103 | 0.6809 |

The selected seed 2, epoch 20 checkpoint reached validation `F1@0.3=0.8405` and `F1@0.5=0.7431`. The three-seed mean advantage over `1088x576` was `0.0129` F1@0.5, so the lower resolution won without invoking the smaller-model tie rule.

Test results:

| Split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 1557 | 1886 | 0.8992 | 0.7423 | 0.8132 | 0.8067 | 0.6660 | 0.7296 |
| lane_706 test | 580 | 745 | 0.8793 | 0.6846 | 0.7698 | 0.7621 | 0.5933 | 0.6672 |
| lane_812 test | 668 | 779 | 0.9207 | 0.7895 | 0.8500 | 0.8353 | 0.7163 | 0.7713 |
| lane_349 test | 309 | 362 | 0.8900 | 0.7597 | 0.8197 | 0.8285 | 0.7072 | 0.7630 |

Stage 1 is class-agnostic; these metrics do not evaluate `solid/dashed/joint`.

### Stage 2 Classifier

Checkpoint:

```text
work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/best.pth
```

Best epoch: 26.

| Split | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| val | 1787 | 0.9239 | 0.8812 |

Training samples: 8241. Class order: `solid,dashed,joint`.

### Two-Stage End-To-End

The final two-stage evaluation reuses the existing Stage 2 checkpoint and uses the Stage 1 parameters selected on validation: `score_thr=0.35`, `det_conf_thr=0.35`, `nms_topk=10`, `nms_thres=50`, `iou_thr=0.3`, `match_width=20`.

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Matched-lane Acc | Matched-lane Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 283 | 1414 | 478 | 143 | 0.7999 | 0.7398 |
| lane_706 test | 107 | 522 | 223 | 58 | 0.7816 | 0.7389 |
| lane_812 test | 123 | 615 | 166 | 53 | 0.8228 | 0.7550 |
| lane_349 test | 53 | 277 | 89 | 32 | 0.7834 | 0.7018 |

Two-stage classification metrics count only predictions matched to GT lanes. They are not detection F1.

Sample overlays:

```text
work_dirs/two_stage_infer/lane_706_812_349_20260609_resolution_best_samples/
```

## 1. Environment And Installation

Use the existing server environment:

```bash
ssh 190server
cd /datadisk2/longhaoxiang/CLRerNet
source /home/longhaoxiang/anaconda3/etc/profile.d/conda.sh
conda activate clrernet
```

Verified package versions on 2026-06-09:

| Package | Version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.1.0+cu121 |
| TorchVision | 0.16.0+cu121 |
| CUDA runtime reported by PyTorch | 12.1 |
| cuDNN reported by PyTorch | 8.9.2 |
| OpenCV | 4.9.0 |
| MMEngine | 0.10.5 |
| MMCV | 2.1.0 |
| MMDetection | 3.3.0 |
| scikit-learn | 1.8.0 |

Default command prefix:

```bash
PYTHONPATH=. python ...
```

Official CULane EMA warm-start checkpoint:

```text
checkpoints/clrernet_culane_dla34_ema.pth
```

New machine installation reference:

```bash
conda create -n clrernet python=3.11 -y
conda activate clrernet
pip install -r requirements.txt
pip install mmengine==0.10.5 mmcv==2.1.0 mmdet==3.3.0 opencv-python==4.9.0.80 scikit-learn==1.8.0

cd libs/models/layers/nms
python setup.py install
cd ../../../..
```

Environment check:

```bash
PYTHONPATH=. python -c "import torch, cv2, mmengine, mmcv, mmdet; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Algorithm Architecture

Stage 1 reads `solid/dashed/joint` labels but maps every valid lane to one class for CLRerNet training. Original labels are preserved in metadata for debugging and Stage 2.

Current Stage 1 pipeline:

1. Top crop with `top_crop_ratio=0.08`.
2. Resize to `1024x544`.
3. Filter lanes with fewer than 2 valid points after crop/resize.
4. Fine-tune DLA34 CLRerNet from `checkpoints/clrernet_culane_dla34_ema.pth`.
5. Validate every 5 epochs and use seed 2 `epoch_20.pth` for current evaluation.

Current Stage 1 settings:

| Setting | Value |
| --- | --- |
| Input resolution | `1024 x 544` |
| Top crop | `0.08` |
| Batch size | `4` |
| Gradient accumulation | `2` |
| Effective batch size | about `8` |
| Epochs | `25` |
| Optimizer | `AdamW(lr=5e-5, weight_decay=0.01)` |
| Warm start | `checkpoints/clrernet_culane_dla34_ema.pth` |
| NMS topk | `10` |

Stage 2 trains a lightweight CNN on GT lane strip crops. End-to-end inference crops around Stage 1 predicted lane instances and classifies each crop.

Current Stage 2 settings:

| Setting | Value |
| --- | --- |
| Classes | `solid,dashed,joint` |
| Crop size | `288 x 128` |
| Strip width | `128` |
| Batch size | `64` |
| Epochs | `30` |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=1e-4` |
| Dropout | `0.25` |
| Class balance | weighted loss |

## 3. Project Structure

```text
CLRerNet/
├── README.md
├── checkpoints/                              # local checkpoints, not tracked by Git
├── configs/clrernet/
│   ├── base_clrernet.py
│   ├── lane_706_20260518/                    # historical single-source lane_706 config
│   ├── lane_706_812_20260531/                # historical two-source config
│   └── lane_706_812_349_20260609/            # current three-source resolution-search configs
├── dataset/                                  # local datasets, not tracked by Git
│   ├── lane_706_20260518/
│   ├── lane_812_20260531/
│   ├── lane_349_20260609/
│   └── lane_706_812_349_20260609/splits/     # virtual merged split
├── docs/
│   └── culane_highway_two_stage_report.md
├── libs/
│   ├── datasets/highway_lane_dataset.py      # Stage 1 dataset, single/multi-root
│   ├── datasets/metrics/highway_lane_metric.py
│   ├── lane_classifier/crop.py
│   ├── lane_classifier/dataset.py            # Stage 2 lane crop dataset
│   ├── lane_classifier/model.py
│   ├── lane_classifier/train.py
│   ├── lane_classifier/eval.py
│   ├── lane_classifier/infer.py
│   └── utils/highway_data.py                 # multi-root split parsing
├── tools/
│   ├── prepare_culane_highway.py             # single-source validation and split
│   ├── build_highway_multiroot_splits.py     # virtual multi-root merged split
│   ├── sweep_highway_stage1_postprocess.py   # validation post-processing grid search
│   ├── summarize_stage1_resolution_search.py # aggregate seeds and select resolution
│   ├── train.py
│   └── test.py
└── work_dirs/                                # training/eval outputs, not tracked by Git
```

## 4. Dataset Structure

Each labeled batch is a flat folder containing image files and same-stem LabelMe JSON files:

```text
dataset/lane_349_20260609/
├── xxx.jpg
├── xxx.json
├── yyy.PNG
├── yyy.json
└── splits/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── prepare_report.json
```

Valid lane labels:

```text
solid
dashed
joint
```

Merged multi-root split files use two columns:

```text
dataset_key<TAB>relative_image_path
```

Current merged split files:

```text
dataset/lane_706_812_349_20260609/splits/train.txt
dataset/lane_706_812_349_20260609/splits/val.txt
dataset/lane_706_812_349_20260609/splits/test.txt
dataset/lane_706_812_349_20260609/splits/test_lane_706_20260518.txt
dataset/lane_706_812_349_20260609/splits/test_lane_812_20260531.txt
dataset/lane_706_812_349_20260609/splits/test_lane_349_20260609.txt
dataset/lane_706_812_349_20260609/splits/merge_report.json
```

## 5. Training, Evaluation, And Inference

### 5.1 Split New Data

```bash
PYTHONPATH=. python tools/prepare_culane_highway.py \
  --data-root dataset/lane_349_20260609 \
  --seed 0 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

### 5.2 Build Three-Source Split

```bash
PYTHONPATH=. python tools/build_highway_multiroot_splits.py \
  --source lane_706_20260518=dataset/lane_706_20260518 \
  --source lane_812_20260531=dataset/lane_812_20260531 \
  --source lane_349_20260609=dataset/lane_349_20260609 \
  --out-dir dataset/lane_706_812_349_20260609/splits \
  --source-priority lane_349_20260609,lane_812_20260531,lane_706_20260518
```

### 5.3 Train Stage 1

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python tools/train.py \
  configs/clrernet/lane_706_812_349_20260609/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1024x544_topcrop8.py \
  --work-dir work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2 \
  --cfg-options randomness.seed=2
```

### 5.4 Search Stage 1 Post-Processing

Run the grid on merged validation only:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python tools/sweep_highway_stage1_postprocess.py \
  configs/clrernet/lane_706_812_349_20260609/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1024x544_topcrop8.py \
  work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth \
  --split-file dataset/lane_706_812_349_20260609/splits/val.txt \
  --out-dir work_dirs/resolution_search/eval/1024x544/seed2/epoch20/grid \
  --conf-thresholds 0.25 0.30 0.325 0.35 0.375 0.40 \
  --nms-topks 6 8 10 12 \
  --nms-thres 40 50 60 \
  --device cuda:0
```

Aggregate completed runs:

```bash
PYTHONPATH=. python tools/summarize_stage1_resolution_search.py \
  --root work_dirs/resolution_search/eval \
  --out-dir work_dirs/resolution_search/summary_final \
  --tie-threshold 0.003
```

### 5.5 Test Stage 1

Merged test:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py \
  configs/clrernet/lane_706_812_349_20260609/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1024x544_topcrop8.py \
  work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth \
  --cfg-options model.test_cfg.conf_threshold=0.35 model.test_cfg.nms_topk=10 model.test_cfg.nms_thres=50
```

Per-source test:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_20260609/splits/test_lane_706_20260518.txt model.test_cfg.conf_threshold=0.35 model.test_cfg.nms_topk=10 model.test_cfg.nms_thres=50

CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_20260609/splits/test_lane_812_20260531.txt model.test_cfg.conf_threshold=0.35 model.test_cfg.nms_topk=10 model.test_cfg.nms_thres=50

CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_20260609/splits/test_lane_349_20260609.txt model.test_cfg.conf_threshold=0.35 model.test_cfg.nms_topk=10 model.test_cfg.nms_thres=50
```

### 5.6 Train Stage 2

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python -m libs.lane_classifier.train \
  --data-roots lane_706_20260518=dataset/lane_706_20260518 lane_812_20260531=dataset/lane_812_20260531 lane_349_20260609=dataset/lane_349_20260609 \
  --split-root dataset/lane_706_812_349_20260609/splits \
  --work-dir work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 4 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --crop-height 288 \
  --crop-width 128 \
  --strip-width 128 \
  --dropout 0.25 \
  --class-balance loss \
  --debug-crops 0 \
  --device cuda:0 \
  --seed 0
```

### 5.7 Two-Stage Evaluation

Primary merged test:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python -m libs.lane_classifier.eval \
  --data-roots lane_706_20260518=dataset/lane_706_20260518 lane_812_20260531=dataset/lane_812_20260531 lane_349_20260609=dataset/lane_349_20260609 \
  --split-file dataset/lane_706_812_349_20260609/splits/test.txt \
  --det-config configs/clrernet/lane_706_812_349_20260609/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_merged_s0.35_top10_nms50 \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-thres 50 \
  --nms-topk 10 \
  --iou-thr 0.3 \
  --match-width 20 \
  --device cuda:0
```

For per-source evaluation, replace `--split-file` and `--out-dir` with:

```text
dataset/lane_706_812_349_20260609/splits/test_lane_706_20260518.txt
work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_lane706_s0.35_top10_nms50

dataset/lane_706_812_349_20260609/splits/test_lane_812_20260531.txt
work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_lane812_s0.35_top10_nms50

dataset/lane_706_812_349_20260609/splits/test_lane_349_20260609.txt
work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_lane349_s0.35_top10_nms50
```

### 5.8 Inference And Visualization

Single image or folder:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input path/to/image_or_folder \
  --det-config configs/clrernet/lane_706_812_349_20260609/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/custom \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-thres 50 \
  --nms-topk 10 \
  --device cuda:0
```

The command writes:

```text
<out-dir>/predictions/*.json
<out-dir>/visualizations/*.jpg
<out-dir>/summary.json
```
