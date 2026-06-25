# CLRerNet Highway Two-Stage Lane Detection

Author: Haoxiang Long

Email: haoxianglong@std.uestc.edu.com

This repository adapts CLRerNet to a two-stage highway lane workflow:

- Stage 1: class-agnostic lane locator. It learns lane geometry only.
- Stage 2: lane instance classifier. It classifies each lane as `solid`, `dashed`, or `joint`.

Current verified state: 2026-06-24. The active training data is the merged multi-root set `lane_706_20260518 + lane_812_20260531 + lane_349_20260609 + lane_696_20260624`. `dataset/culane_highway` is historical only and is not used for current split, training, or evaluation.

Artifact retention was cleaned on 2026-06-11. The server keeps the selected three-source Stage 1/Stage 2 checkpoints because the 2026-06-24 four-source run initializes from them. The current four-source run keeps its Stage 1 `epoch_20.pth`, Stage 2 `best.pth`, five final two-stage evaluations, and eight final overlays. Superseded historical metrics remain documented below.

## Latest Verified Results

### Data And Split

| Source | Image/JSON pairs | Split | Notes |
| --- | ---: | --- | --- |
| `lane_706_20260518` | 706/706 | fixed existing split: 494/105/107 | historical split kept for comparability |
| `lane_812_20260531` | 812/812 | fixed existing split: 568/121/123 | historical split kept for comparability |
| `lane_349_20260609` | 349/349 | fixed existing split: 244/52/53 | historical split kept for comparability |
| `lane_696_20260624` | 696/696 | seed 0: 487/104/105 | zip extracted and split generated on 2026-06-24 |
| merged | 2561 image-level samples | 1791/382/388 | virtual multi-root split, no image copy |

Duplicate image policy: newer source wins, with priority `lane_696_20260624 > lane_349_20260609 > lane_812_20260531 > lane_706_20260518`.

Dropped duplicate samples:

```text
lane_706_20260518/train/outside_20250910112526_000206.jpg
lane_812_20260531/train/DsKPdq9pWzgZ1Hu3kaf0ZfLT_202510161514_1.jpg
```

Merged lane label counts:

| Split | solid | dashed | joint |
| --- | ---: | ---: | ---: |
| train | 7028 | 2841 | 1533 |
| val | 1486 | 590 | 355 |
| test | 1569 | 683 | 370 |

### Stage 1 Locator

The 2026-06-10 controlled resolution search selected `1024x544 topcrop8` as the best resolution on the three-source validation set. The 2026-06-24 four-source run keeps that resolution and fine-tunes from the selected three-source locator.

Config:

```text
configs/clrernet/lane_706_812_349_696_20260624/clrernet_lane_706_812_349_696_20260624_dla34_ema_locator_1024x544_topcrop8_finetune.py
```

Checkpoint:

```text
work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best/epoch_20.pth
```

Selected inference parameters:

```text
conf_threshold=0.40
nms_topk=8
nms_thres=40
```

Four-source validation sweep result: epoch 20 with `0.40/top8/nms40` reached `F1@0.3=0.8722` and `F1@0.5=0.7938` on merged validation. The sweep used only validation data.

Test results:

| Split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 2140 | 2614 | 0.9430 | 0.7720 | 0.8490 | 0.8682 | 0.7108 | 0.7817 |
| lane_706 test | 591 | 745 | 0.9205 | 0.7302 | 0.8144 | 0.8223 | 0.6523 | 0.7275 |
| lane_812 test | 660 | 779 | 0.9606 | 0.8139 | 0.8812 | 0.9030 | 0.7651 | 0.8284 |
| lane_349 test | 306 | 362 | 0.9379 | 0.7928 | 0.8593 | 0.8889 | 0.7514 | 0.8144 |
| lane_696 test | 583 | 728 | 0.9485 | 0.7596 | 0.8436 | 0.8645 | 0.6923 | 0.7689 |

Stage 1 is class-agnostic; these metrics do not evaluate `solid/dashed/joint`.

### Stage 2 Classifier

Checkpoint:

```text
work_dirs/lane_classifier/lane_706_812_349_696_20260624_stage2_strip_288x128_w128_finetune_from_349_best/best.pth
```

The classifier was initialized from the three-source `best.pth` with `--init-from` and trained as a fresh 20-epoch experiment on the four-source split.

Best epoch: 19.

| Split | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| val | 2431 | 0.9292 | 0.8860 |

Training samples: 11402. Class order: `solid,dashed,joint`.

### Two-Stage End-To-End

The final two-stage evaluation uses the Stage 1 parameters selected on validation: `score_thr=0.40`, `det_conf_thr=0.40`, `nms_topk=8`, `nms_thres=40`, `iou_thr=0.3`, `match_width=20`.

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Det F1@0.3 | Matched-lane Acc | Matched-lane Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 388 | 2023 | 599 | 116 | 0.8498 | 0.8057 | 0.7400 |
| lane_706 test | 107 | 545 | 200 | 46 | 0.8159 | 0.7945 | 0.7400 |
| lane_812 test | 123 | 635 | 146 | 25 | 0.8813 | 0.8409 | 0.7773 |
| lane_349 test | 53 | 288 | 78 | 18 | 0.8571 | 0.7778 | 0.6994 |
| lane_696 test | 105 | 555 | 175 | 27 | 0.8460 | 0.7910 | 0.7219 |

Two-stage classification metrics count only predictions matched to GT lanes. They are not detection F1.

Sample overlays:

```text
work_dirs/two_stage_infer/lane_706_812_349_696_20260624_best_samples/
```

### Dataset Addition Trend On Shared Test Splits

The tables below keep the test split fixed wherever history allows it. This is more meaningful than only comparing each run's merged test, because the merged test set changes whenever a new dataset is added.

Unified columns:

- Stage 1 test `F1@0.3` and `F1@0.5`.
- Two-stage detection `F1@0.3`, recomputed from `matched_gt`, `unmatched_gt`, and `unmatched_prediction`.
- Two-stage matched-lane classification accuracy and macro-F1.

`lane_706_20260518` test, available for all training-data versions:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706` | baseline | 0.6040 | 0.4962 | 0.7305 | 0.7228 | 0.6751 |
| `lane_706 + lane_812` | `lane_812_20260531` | 0.6516 | 0.5612 | 0.7711 | 0.7515 | 0.7134 |
| `lane_706 + lane_812 + lane_349` | `lane_349_20260609` | 0.7698 | 0.6672 | 0.7879 | 0.7816 | 0.7389 |
| `lane_706 + lane_812 + lane_349 + lane_696` | `lane_696_20260624` | 0.8144 | 0.7275 | 0.8159 | 0.7945 | 0.7400 |

`lane_812_20260531` test, available from the two-source run onward:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706 + lane_812` | `lane_812_20260531` | 0.7123 | 0.6359 | 0.8331 | 0.7914 | 0.7291 |
| `lane_706 + lane_812 + lane_349` | `lane_349_20260609` | 0.8500 | 0.7713 | 0.8489 | 0.8228 | 0.7550 |
| `lane_706 + lane_812 + lane_349 + lane_696` | `lane_696_20260624` | 0.8812 | 0.8284 | 0.8813 | 0.8409 | 0.7773 |

`lane_349_20260609` test, available from the three-source run onward:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706 + lane_812 + lane_349` | `lane_349_20260609` | 0.8197 | 0.7630 | 0.8207 | 0.7834 | 0.7018 |
| `lane_706 + lane_812 + lane_349 + lane_696` | `lane_696_20260624` | 0.8593 | 0.8144 | 0.8571 | 0.7778 | 0.6994 |

`lane_696_20260624` test exists only for the current four-source run, so it is the baseline for future additions:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706 + lane_812 + lane_349 + lane_696` | `lane_696_20260624` | 0.8436 | 0.7689 | 0.8460 | 0.7910 | 0.7219 |

Merged-test project-level trend is still useful for release notes, but it is not the primary comparison because each row has a different merged test set:

| Training data | Merged test images | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lane_706` | 107 | 0.6040 | 0.4962 | 0.7305 | 0.7228 | 0.6751 |
| `lane_706 + lane_812` | 230 | 0.6834 | 0.6004 | 0.8036 | 0.7732 | 0.7228 |
| `lane_706 + lane_812 + lane_349` | 283 | 0.8132 | 0.7296 | 0.8199 | 0.7999 | 0.7398 |
| `lane_706 + lane_812 + lane_349 + lane_696` | 388 | 0.8490 | 0.7817 | 0.8498 | 0.8057 | 0.7400 |

### Resolution Search Summary

The resolution search was run before adding `lane_696`, on the fixed three-source split `lane_706 + lane_812 + lane_349`. Validation `F1@0.5` was the primary selection metric and validation `F1@0.3` was secondary.

Fixed post-processing comparison, using `conf_threshold=0.35`, `nms_topk=8`, and `nms_thres=50`:

| Resolution | Seed | Epoch | pred_lanes | Val F1@0.3 | Val F1@0.5 | Forward ms/image | Peak alloc MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024x544` | 0 | 25 | 1524 | 0.8369 | 0.7311 | 99.60 | 516.35 |
| `1088x576` | 0 | 25 | 1541 | 0.8332 | 0.7178 | 51.34 | 573.80 |
| `1152x608` | 0 | 25 | 1542 | 0.8233 | 0.7037 | 102.47 | 631.03 |
| `1216x640` | 0 | 25 | 1546 | 0.8175 | 0.6861 | 99.30 | 691.70 |
| `1280x672` | 0 | 25 | 1522 | 0.8077 | 0.6796 | 100.64 | 757.17 |
| `1280x704` | 0 | 25 | 1516 | 0.8116 | 0.6760 | 99.62 | 789.76 |

Per-resolution validation tuning used:

```text
conf_threshold = [0.25, 0.30, 0.325, 0.35, 0.375, 0.40]
nms_topk       = [6, 8, 10, 12]
nms_thres      = [40, 50, 60]
```

| Resolution | Seeds | Selected conf/topk/nms | Mean Val F1@0.3 | Mean Val F1@0.5 |
| --- | ---: | --- | ---: | ---: |
| `1024x544` | 3 | `0.35/10/50` | 0.8382 | 0.7345 |
| `1088x576` | 3 | `0.35/10/40` | 0.8355 | 0.7216 |
| `1152x608` | 1 | `0.375/10/50` | 0.8215 | 0.7057 |
| `1216x640` | 1 | `0.375/8/60` | 0.8146 | 0.6911 |
| `1280x672` | 1 | `0.375/8/60` | 0.8000 | 0.6842 |
| `1280x704` | 1 | `0.375/8/60` | 0.8103 | 0.6809 |

Conclusion: within the tested candidates, `1024x544 topcrop8` was best. Higher resolution increased memory use and did not improve validation `F1@0.5`.

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
4. Fine-tune DLA34 CLRerNet from the selected three-source locator.
5. Validate every 5 epochs and use four-source `epoch_20.pth` for current evaluation.

Current Stage 1 settings:

| Setting | Value |
| --- | --- |
| Input resolution | `1024 x 544` |
| Top crop | `0.08` |
| Batch size | `4` |
| Gradient accumulation | `2` |
| Effective batch size | about `8` |
| Epochs | `20` |
| Optimizer | `AdamW(lr=5e-5, weight_decay=0.01)` |
| Warm start | `work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth` |
| NMS topk | `8` |

Stage 2 trains a lightweight CNN on GT lane strip crops. End-to-end inference crops around Stage 1 predicted lane instances and classifies each crop.

Current Stage 2 settings:

| Setting | Value |
| --- | --- |
| Classes | `solid,dashed,joint` |
| Crop size | `288 x 128` |
| Strip width | `128` |
| Batch size | `64` |
| Epochs | `20` |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=1e-4` |
| Dropout | `0.25` |
| Class balance | weighted loss |

## 3. Project Structure

```text
CLRerNet/
|-- README.md
|-- checkpoints/                              # local checkpoints, not tracked by Git
|-- configs/clrernet/
|   |-- base_clrernet.py
|   |-- lane_706_20260518/                    # historical single-source lane_706 config
|   |-- lane_706_812_20260531/                # historical two-source config
|   |-- lane_706_812_349_20260609/            # three-source resolution-search configs
|   `-- lane_706_812_349_696_20260624/        # current four-source fine-tune configs
|-- dataset/                                  # local datasets, not tracked by Git
|   |-- lane_706_20260518/
|   |-- lane_812_20260531/
|   |-- lane_349_20260609/
|   |-- lane_696_20260624.zip
|   |-- lane_696_20260624/
|   `-- lane_706_812_349_696_20260624/splits/ # current virtual merged split
|-- docs/
|   `-- culane_highway_two_stage_report.md
|-- libs/
|   |-- datasets/highway_lane_dataset.py      # Stage 1 dataset, single/multi-root
|   |-- datasets/metrics/highway_lane_metric.py
|   |-- lane_classifier/crop.py
|   |-- lane_classifier/dataset.py            # Stage 2 lane crop dataset
|   |-- lane_classifier/model.py
|   |-- lane_classifier/train.py              # supports --resume-from and --init-from
|   |-- lane_classifier/eval.py
|   |-- lane_classifier/infer.py
|   `-- utils/highway_data.py                 # multi-root split parsing
|-- tools/
|   |-- prepare_culane_highway.py             # single-source validation and split
|   |-- build_highway_multiroot_splits.py     # virtual multi-root merged split
|   |-- sweep_highway_stage1_postprocess.py   # validation post-processing grid search
|   |-- summarize_stage1_resolution_search.py # aggregate seeds and select resolution
|   |-- train.py
|   `-- test.py
`-- work_dirs/                                # training/eval outputs, not tracked by Git
```

## 4. Dataset Structure

Each labeled batch is a flat folder containing image files and same-stem LabelMe JSON files:

```text
dataset/lane_696_20260624/
|-- xxx.jpg
|-- xxx.json
|-- yyy.PNG
|-- yyy.json
`-- splits/
    |-- train.txt
    |-- val.txt
    |-- test.txt
    `-- prepare_report.json
```

Valid lane labels:

```text
solid
dashed
joint
```

Single-source split files use one column:

```text
relative_image_path
```

Merged multi-root split files use two columns:

```text
dataset_key<TAB>relative_image_path
```

Current merged split files:

```text
dataset/lane_706_812_349_696_20260624/splits/train.txt
dataset/lane_706_812_349_696_20260624/splits/val.txt
dataset/lane_706_812_349_696_20260624/splits/test.txt
dataset/lane_706_812_349_696_20260624/splits/test_lane_706_20260518.txt
dataset/lane_706_812_349_696_20260624/splits/test_lane_812_20260531.txt
dataset/lane_706_812_349_696_20260624/splits/test_lane_349_20260609.txt
dataset/lane_706_812_349_696_20260624/splits/test_lane_696_20260624.txt
dataset/lane_706_812_349_696_20260624/splits/merge_report.json
```

## 5. Training, Evaluation, And Inference

### 5.1 Extract And Split New Data

```bash
test ! -d dataset/lane_696_20260624
unzip -q dataset/lane_696_20260624.zip -d dataset

PYTHONPATH=. python tools/prepare_culane_highway.py \
  --data-root dataset/lane_696_20260624 \
  --seed 0 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

Expected `lane_696_20260624` split: `train=487`, `val=104`, `test=105`, with `696` valid image/JSON pairs.

### 5.2 Build Four-Source Split

```bash
PYTHONPATH=. python tools/build_highway_multiroot_splits.py \
  --source lane_706_20260518=dataset/lane_706_20260518 \
  --source lane_812_20260531=dataset/lane_812_20260531 \
  --source lane_349_20260609=dataset/lane_349_20260609 \
  --source lane_696_20260624=dataset/lane_696_20260624 \
  --out-dir dataset/lane_706_812_349_696_20260624/splits \
  --source-priority lane_696_20260624,lane_349_20260609,lane_812_20260531,lane_706_20260518
```

Current merged counts are `train=1791`, `val=382`, `test=388`. The merge report records two dropped duplicate samples, both from older sources.

### 5.3 Train Stage 1

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tools/train.py \
  configs/clrernet/lane_706_812_349_696_20260624/clrernet_lane_706_812_349_696_20260624_dla34_ema_locator_1024x544_topcrop8_finetune.py \
  --work-dir work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best
```

The config uses `load_from=work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth`, `img_scale=(1024,544)`, `top_crop_ratio=0.08`, `batch_size=4`, and gradient accumulation `2`.

### 5.4 Search Stage 1 Post-Processing

Run the grid on merged validation only:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tools/sweep_highway_stage1_postprocess.py \
  configs/clrernet/lane_706_812_349_696_20260624/clrernet_lane_706_812_349_696_20260624_dla34_ema_locator_1024x544_topcrop8_finetune.py \
  work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best/epoch_20.pth \
  --split-file dataset/lane_706_812_349_696_20260624/splits/val.txt \
  --out-dir work_dirs/stage1_postprocess/lane_706_812_349_696_20260624/epoch_20_val_grid \
  --conf-thresholds 0.25 0.30 0.325 0.35 0.375 0.40 \
  --nms-topks 6 8 10 12 \
  --nms-thres 40 50 60 \
  --device cuda:0
```

The selected four-source setting is `conf_threshold=0.40`, `nms_topk=8`, `nms_thres=40` from epoch 20.

### 5.5 Test Stage 1

Merged test:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tools/test.py \
  configs/clrernet/lane_706_812_349_696_20260624/clrernet_lane_706_812_349_696_20260624_dla34_ema_locator_1024x544_topcrop8_finetune.py \
  work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best/epoch_20.pth \
  --cfg-options model.test_cfg.conf_threshold=0.4 model.test_cfg.nms_topk=8 model.test_cfg.nms_thres=40
```

Per-source test:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_20260624/splits/test_lane_706_20260518.txt model.test_cfg.conf_threshold=0.4 model.test_cfg.nms_topk=8 model.test_cfg.nms_thres=40

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_20260624/splits/test_lane_812_20260531.txt model.test_cfg.conf_threshold=0.4 model.test_cfg.nms_topk=8 model.test_cfg.nms_thres=40

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_20260624/splits/test_lane_349_20260609.txt model.test_cfg.conf_threshold=0.4 model.test_cfg.nms_topk=8 model.test_cfg.nms_thres=40

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_20260624/splits/test_lane_696_20260624.txt model.test_cfg.conf_threshold=0.4 model.test_cfg.nms_topk=8 model.test_cfg.nms_thres=40
```

### 5.6 Train Stage 2

```bash
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python -m libs.lane_classifier.train \
  --data-roots lane_706_20260518=dataset/lane_706_20260518 lane_812_20260531=dataset/lane_812_20260531 lane_349_20260609=dataset/lane_349_20260609 lane_696_20260624=dataset/lane_696_20260624 \
  --split-root dataset/lane_706_812_349_696_20260624/splits \
  --work-dir work_dirs/lane_classifier/lane_706_812_349_696_20260624_stage2_strip_288x128_w128_finetune_from_349_best \
  --init-from work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/best.pth \
  --epochs 20 \
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

`--init-from` loads model weights only and starts a new experiment from epoch 1. Use `--resume-from` only when continuing an interrupted run with optimizer/history state.

### 5.7 Two-Stage Evaluation

Primary merged test:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python -m libs.lane_classifier.eval \
  --data-roots lane_706_20260518=dataset/lane_706_20260518 lane_812_20260531=dataset/lane_812_20260531 lane_349_20260609=dataset/lane_349_20260609 lane_696_20260624=dataset/lane_696_20260624 \
  --split-file dataset/lane_706_812_349_696_20260624/splits/test.txt \
  --det-config configs/clrernet/lane_706_812_349_696_20260624/clrernet_lane_706_812_349_696_20260624_dla34_ema_locator_1024x544_topcrop8_finetune.py \
  --det-checkpoint work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best/epoch_20.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_812_349_696_20260624_stage2_strip_288x128_w128_finetune_from_349_best/best.pth \
  --out-dir work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_merged_s0.40_top8_nms40 \
  --score-thr 0.4 \
  --det-conf-thr 0.4 \
  --nms-thres 40 \
  --nms-topk 8 \
  --iou-thr 0.3 \
  --match-width 20 \
  --device cuda:0
```

For per-source evaluation, replace `--split-file` and `--out-dir` with:

```text
dataset/lane_706_812_349_696_20260624/splits/test_lane_706_20260518.txt
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_lane706_s0.40_top8_nms40

dataset/lane_706_812_349_696_20260624/splits/test_lane_812_20260531.txt
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_lane812_s0.40_top8_nms40

dataset/lane_706_812_349_696_20260624/splits/test_lane_349_20260609.txt
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_lane349_s0.40_top8_nms40

dataset/lane_706_812_349_696_20260624/splits/test_lane_696_20260624.txt
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_lane696_s0.40_top8_nms40
```

### 5.8 Inference And Visualization

Single image or folder:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input path/to/image_or_folder \
  --det-config configs/clrernet/lane_706_812_349_696_20260624/clrernet_lane_706_812_349_696_20260624_dla34_ema_locator_1024x544_topcrop8_finetune.py \
  --det-checkpoint work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best/epoch_20.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_812_349_696_20260624_stage2_strip_288x128_w128_finetune_from_349_best/best.pth \
  --out-dir work_dirs/two_stage_infer/custom \
  --score-thr 0.4 \
  --det-conf-thr 0.4 \
  --nms-thres 40 \
  --nms-topk 8 \
  --device cuda:0
```

The command writes:

```text
<out-dir>/predictions/*.json
<out-dir>/visualizations/*.jpg
<out-dir>/summary.json
```
