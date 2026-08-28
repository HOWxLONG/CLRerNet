# CLRerNet Highway Two-Stage Lane Detection

Author: Haoxiang Long

Email: haoxianglong@std.uestc.edu.com

This repository adapts CLRerNet to a two-stage highway lane workflow:

- Stage 1: class-agnostic lane locator. It learns lane geometry only.
- Stage 2: lane instance classifier. It classifies each lane as `solid`, `dashed`, or `joint`.

Current verified state: 2026-07-22. The active training data is the merged multi-root set `lane_706_20260518 + lane_812_20260531 + lane_349_20260609 + lane_696_20260624 + lane_350_20260629 + lane_1149_20260722`. `dataset/culane_highway` is historical only and is not used for current split, training, or evaluation.

The current six-source run uses the previously selected `1024x544 topcrop8` resolution. Stage 1 fully fine-tunes all parameters from the official CULane EMA checkpoint; Stage 2 is initialized randomly and trained from scratch. The same-test-split tables below are the preferred comparison view.

## Latest Verified Results

### Data And Split

| Source | Image/JSON pairs | Single-source split | Retained in merged split | Notes |
| --- | ---: | --- | --- | --- |
| `lane_706_20260518` | 706/706 | fixed: 494/105/107 | 493/105/107 | one train duplicate dropped by priority |
| `lane_812_20260531` | 812/812 | fixed: 568/121/123 | 567/121/123 | one train duplicate dropped by priority |
| `lane_349_20260609` | 349/349 | fixed: 244/52/53 | 244/52/53 | historical split kept |
| `lane_696_20260624` | 696/696 | fixed: 487/104/105 | 487/104/105 | historical split kept |
| `lane_350_20260629` | 350/350 | seed 0: 244/52/54 | 244/52/54 | zip extracted and split generated on 2026-07-01 |
| `lane_1149_20260722` | 1149/1149 | seed 0: 804/172/173 | 804/172/173 | 89 accepted point-order warnings, zero errors |
| merged | 4060 retained image-level samples | - | 2839/606/615 | virtual multi-root split, no image copy |

Duplicate image policy: newer source wins, with priority `lane_1149_20260722 > lane_350_20260629 > lane_696_20260624 > lane_349_20260609 > lane_812_20260531 > lane_706_20260518`. The new `lane_1149_20260722` batch introduced no duplicate image; the two dropped samples remain the historical duplicates below.

Dropped duplicate samples:

```text
lane_706_20260518/train/outside_20250910112526_000206.jpg
lane_812_20260531/train/DsKPdq9pWzgZ1Hu3kaf0ZfLT_202510161514_1.jpg
```

Merged lane label counts:

| Split | solid | dashed | joint |
| --- | ---: | ---: | ---: |
| train | 11240 | 4414 | 2675 |
| val | 2362 | 934 | 582 |
| test | 2524 | 1053 | 578 |

### Stage 1 Locator

The 2026-06-10 controlled resolution search selected `1024x544 topcrop8` as the best resolution on the three-source validation set. The six-source run keeps that resolution and fully fine-tunes from the official CULane EMA checkpoint.

Config:

```text
configs/clrernet/lane_706_812_349_696_350_1149_20260722/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8.py
```

Checkpoint:

```text
work_dirs/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_20.pth
```

Selected inference parameters:

```text
conf_threshold=0.40
nms_topk=12
nms_thres=40
```

Six-source validation sweep result: epoch 20 with `0.40/top12/nms40` reached `F1@0.3=0.8715` and `F1@0.5=0.7963` on merged validation. The sweep used only validation data; test data was not used to select the checkpoint or post-processing parameters.

Test results:

| Split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 3448 | 4143 | 0.9353 | 0.7784 | 0.8497 | 0.8622 | 0.7176 | 0.7833 |
| lane_706 test | 607 | 745 | 0.9012 | 0.7342 | 0.8092 | 0.8089 | 0.6591 | 0.7263 |
| lane_812 test | 670 | 779 | 0.9627 | 0.8280 | 0.8903 | 0.9075 | 0.7805 | 0.8392 |
| lane_349 test | 307 | 362 | 0.9446 | 0.8011 | 0.8670 | 0.8860 | 0.7514 | 0.8132 |
| lane_696 test | 592 | 728 | 0.9459 | 0.7692 | 0.8485 | 0.8615 | 0.7005 | 0.7727 |
| lane_350 test | 298 | 340 | 0.8993 | 0.7882 | 0.8401 | 0.8255 | 0.7235 | 0.7712 |
| lane_1149 test | 974 | 1189 | 0.9394 | 0.7696 | 0.8460 | 0.8686 | 0.7115 | 0.7822 |

Stage 1 is class-agnostic; these metrics do not evaluate `solid/dashed/joint`.

### Stage 2 Classifier

Checkpoint:

```text
work_dirs/lane_classifier/lane_706_812_349_696_350_1149_20260722_stage2_strip_288x128_w128_scratch/best.pth
```

The classifier was initialized randomly and trained from scratch for 30 epochs on the six-source split.

Best epoch: 29.

| Split | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| val | 3878 | 0.9283 | 0.8880 |

Training samples: 18329. Class order: `solid,dashed,joint`.

### Stage 2 Classification Optimization (2026-08-27)

An independent 295-image diagnostic set exposed a deployment gap: the legacy classifier reached `Accuracy=0.7650` and `Macro-F1=0.7044` on Stage 1 predicted lane crops, while its GT-crop validation result was much higher. The weakest class was `joint` (`F1=0.4533`). Fixed white-pixel rules were not reliable: white-only, white-plus-yellow, and longitudinal bright/dark heuristics reached only about `0.322`, `0.334`, and `0.438` Macro-F1. A diagnostic fusion of the existing classifier with appearance/morphology features was the best tested method (`Accuracy~0.8048`, `Macro-F1~0.7285`), but this is not a frozen-test result and must not be reported as a final model metric.

The new Stage 2 implementation therefore does not hard-override labels with a white-pixel threshold. It adds:

- deployment-matched training with frozen Stage 1 predictions, using a default `70%` predicted-lane / `30%` GT-or-perturbed-GT crop mixture;
- normal offsets, endpoint truncation, strip-width jitter, occlusion, blur, and brightness augmentation;
- `fixed`, image-width `scaled`, and `1024x544 topcrop8` `normalized` strip strategies;
- a longitudinal sequence head with dilated depthwise `Conv1d` blocks and attention pooling;
- a differentiable 16-bin white/yellow/local-contrast morphology branch encoding occupancy, continuity, gaps, and transitions;
- validation-only temperature calibration and complete `label_probs` output while retaining `label` and `label_score`;
- batched classification of all lanes in one image;
- IoU `0.3/0.5` detection metrics, matched-lane classification, resolution groups, and end-to-end class-wise P/R/F1 that include misses, extras, and wrong classes.

Compatibility with legacy Stage 2 checkpoints is retained. The implementation now passes six focused tests, including top-crop boundary clipping and exclusion of lanes that are completely outside the normalized image.

#### Four-batch Stage 2 retraining (2026-08-28)

Four client batches were added as train-only sources: `lane_260727_260802`, `lane_260803_260809`, `lane_260810_260816`, and `lane_260817_260823`. Each contains 450 valid image/JSON pairs. The ten-source split has `train=4624`, `val=606`, and `test=615`; the existing six-source val/test files are byte-for-byte unchanged. The merge report records 17 duplicate groups, including nine new training copies dropped because the same image was already in the frozen old val/test set.

Stage 1 remains frozen at epoch 20 with `score/conf=0.40`, `nms_topk=12`, and `nms_thres=40`. Its predictions were used to train Stage 2 with `70%` predicted-lane crops and `30%` GT/perturbed-GT crops. Seed 0 compared four strip strategies on the fixed validation set:

| Crop strategy | Best epoch | Accuracy | Macro-F1 | solid F1 | dashed F1 | joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed 128 px | 22 | 0.9106 | 0.8666 | 0.9669 | 0.8986 | 0.7344 |
| width-scaled 64-192 px | 16 | 0.9135 | 0.8688 | 0.9664 | 0.9053 | 0.7347 |
| normalized 1024x544, 48 px | 26 | 0.9091 | 0.8653 | 0.9622 | 0.9069 | 0.7269 |
| normalized 1024x544, 64 px | 17 | 0.9129 | 0.8661 | 0.9677 | 0.9016 | 0.7290 |

The top two differ by `0.0022`, so the preregistered tie rule selected fixed 128 px because its measured classifier latency was slightly lower. Fixed 128 px was then trained with seeds 0/1/2; validation Macro-F1 was `0.8666/0.8695/0.8676` (`mean=0.8679`, population `std=0.0012`). The final checkpoint is seed 1, epoch 23:

```text
work_dirs/lane_classifier/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_p70_seed1/best.pth
```

The table below compares the old and new Stage 2 on the exact same frozen 615-image merged test. Stage 1 and its predictions are identical in both rows.

| Stage 2 | Accuracy | Macro-F1 | solid F1 | dashed F1 | joint F1 | Det F1@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| six-source legacy baseline | 0.7669 | 0.7068 | 0.8329 | 0.8422 | 0.4454 | 0.8520 |
| ten-source sequence fusion | **0.9071** | **0.8596** | **0.9598** | **0.9134** | **0.7054** | 0.8520 |

Per-source results for the new model use each unchanged source-specific test split:

| Split | Images | Det F1@0.3 | Accuracy | Macro-F1 | joint F1 | End-to-end Macro-F1@0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged | 615 | 0.8520 | 0.9071 | 0.8596 | 0.7054 | 0.7505 |
| lane_706 | 107 | 0.8166 | 0.8750 | 0.8370 | 0.6731 | 0.6955 |
| lane_812 | 123 | 0.8959 | 0.9354 | 0.9004 | 0.7865 | 0.8283 |
| lane_349 | 53 | 0.8648 | 0.8694 | 0.7968 | 0.5682 | 0.7072 |
| lane_696 | 105 | 0.8472 | 0.9000 | 0.8488 | 0.6889 | 0.7385 |
| lane_350 | 54 | 0.8464 | 0.8963 | 0.8511 | 0.6977 | 0.7475 |
| lane_1149 | 173 | 0.8454 | 0.9258 | 0.8763 | 0.7364 | 0.7587 |

On 32 fixed test images with seven repeats per image, an independent clean latency run measured `44.76 ms/image` for the old two-stage path and `46.44 ms/image` for the new path, a `3.74%` increase. Accuracy, Macro-F1, joint F1, solid/dashed non-regression, unchanged Stage 1, and latency all pass the stated acceptance criteria. The 295-image diagnostic set was not used for model, crop, threshold, or checkpoint selection. See `docs/stage2_classification_optimization.md` for the complete record and commands.

### Legacy Two-Stage End-To-End (Six-Source Baseline)

The final two-stage evaluation uses the Stage 1 parameters selected on validation: `score_thr=0.40`, `det_conf_thr=0.40`, `nms_topk=12`, `nms_thres=40`, `iou_thr=0.3`, `match_width=20`.

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Det F1@0.3 | Matched-lane Acc | Matched-lane Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 615 | 3239 | 916 | 209 | 0.8520 | 0.7669 | 0.7068 |
| lane_706 test | 107 | 552 | 193 | 55 | 0.8166 | 0.7554 | 0.7201 |
| lane_812 test | 123 | 650 | 131 | 20 | 0.8959 | 0.7938 | 0.7365 |
| lane_349 test | 53 | 291 | 75 | 16 | 0.8648 | 0.7423 | 0.6537 |
| lane_696 test | 105 | 560 | 170 | 32 | 0.8472 | 0.7571 | 0.6935 |
| lane_350 test | 54 | 270 | 70 | 28 | 0.8464 | 0.7630 | 0.6921 |
| lane_1149 test | 173 | 916 | 277 | 58 | 0.8454 | 0.7697 | 0.7027 |

Two-stage classification metrics count only predictions matched to GT lanes. They are not detection F1.

Sample overlays:

```text
work_dirs/two_stage_infer/lane_706_812_349_696_350_1149_20260722_best_samples/
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
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350` | `lane_350_20260629` | 0.7915 | 0.7009 | 0.8036 | 0.7970 | 0.7565 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350 + lane_1149` | `lane_1149_20260722` | 0.8092 | 0.7263 | 0.8166 | 0.7554 | 0.7201 |

`lane_812_20260531` test, available from the two-source run onward:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706 + lane_812` | `lane_812_20260531` | 0.7123 | 0.6359 | 0.8331 | 0.7914 | 0.7291 |
| `lane_706 + lane_812 + lane_349` | `lane_349_20260609` | 0.8500 | 0.7713 | 0.8489 | 0.8228 | 0.7550 |
| `lane_706 + lane_812 + lane_349 + lane_696` | `lane_696_20260624` | 0.8812 | 0.8284 | 0.8813 | 0.8409 | 0.7773 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350` | `lane_350_20260629` | 0.8696 | 0.8044 | 0.8698 | 0.8392 | 0.7808 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350 + lane_1149` | `lane_1149_20260722` | 0.8903 | 0.8392 | 0.8959 | 0.7938 | 0.7365 |

`lane_349_20260609` test, available from the three-source run onward:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706 + lane_812 + lane_349` | `lane_349_20260609` | 0.8197 | 0.7630 | 0.8207 | 0.7834 | 0.7018 |
| `lane_706 + lane_812 + lane_349 + lane_696` | `lane_696_20260624` | 0.8593 | 0.8144 | 0.8571 | 0.7778 | 0.6994 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350` | `lane_350_20260629` | 0.8529 | 0.7928 | 0.8478 | 0.7500 | 0.6524 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350 + lane_1149` | `lane_1149_20260722` | 0.8670 | 0.8132 | 0.8648 | 0.7423 | 0.6537 |

`lane_696_20260624` test, available from the four-source run onward:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706 + lane_812 + lane_349 + lane_696` | `lane_696_20260624` | 0.8436 | 0.7689 | 0.8460 | 0.7910 | 0.7219 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350` | `lane_350_20260629` | 0.8285 | 0.7550 | 0.8287 | 0.7601 | 0.7032 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350 + lane_1149` | `lane_1149_20260722` | 0.8485 | 0.7727 | 0.8472 | 0.7571 | 0.6935 |

`lane_350_20260629` test is available from the five-source run onward:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350` | `lane_350_20260629` | 0.8259 | 0.7595 | 0.8259 | 0.8199 | 0.7539 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350 + lane_1149` | `lane_1149_20260722` | 0.8401 | 0.7712 | 0.8464 | 0.7630 | 0.6921 |

`lane_1149_20260722` test is the new fixed-split baseline for future additions:

| Training data | Added batch | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350 + lane_1149` | `lane_1149_20260722` | 0.8460 | 0.7822 | 0.8454 | 0.7697 | 0.7027 |

Merged-test project-level trend is still useful for release notes, but it is not the primary comparison because each row has a different merged test set:

| Training data | Merged test images | Stage 1 F1@0.3 | Stage 1 F1@0.5 | Two-stage det F1@0.3 | Cls acc | Cls macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lane_706` | 107 | 0.6040 | 0.4962 | 0.7305 | 0.7228 | 0.6751 |
| `lane_706 + lane_812` | 230 | 0.6834 | 0.6004 | 0.8036 | 0.7732 | 0.7228 |
| `lane_706 + lane_812 + lane_349` | 283 | 0.8132 | 0.7296 | 0.8199 | 0.7999 | 0.7398 |
| `lane_706 + lane_812 + lane_349 + lane_696` | 388 | 0.8490 | 0.7817 | 0.8498 | 0.8057 | 0.7400 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350` | 442 | 0.8331 | 0.7601 | 0.8356 | 0.7966 | 0.7371 |
| `lane_706 + lane_812 + lane_349 + lane_696 + lane_350 + lane_1149` | 615 | 0.8497 | 0.7833 | 0.8520 | 0.7669 | 0.7068 |

On all five historical fixed test splits, adding `lane_1149_20260722` improved Stage 1 `F1@0.3`, Stage 1 `F1@0.5`, and two-stage detection `F1@0.3`. Matched-lane classification is mixed and is lower on four of the five historical splits. This does not contradict the Stage 2 GT-strip validation result (`Macro-F1=0.8880`): end-to-end classification is measured on the subset matched by the new detector, including additional difficult lanes, and therefore is not an identical classification sample set across runs.

### Resolution Search Summary

The resolution search was run before adding `lane_696` and `lane_350`, on the fixed three-source split `lane_706 + lane_812 + lane_349`. Validation `F1@0.5` was the primary selection metric and validation `F1@0.3` was secondary.

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
4. Train DLA34 CLRerNet from the official CULane EMA checkpoint.
5. Validate every 5 epochs, sweep post-processing only on merged validation, and use six-source `epoch_20.pth` for current evaluation.

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
|-- README.md
|-- checkpoints/                              # local checkpoints, not tracked by Git
|-- configs/clrernet/
|   |-- base_clrernet.py
|   |-- lane_706_20260518/                    # historical single-source lane_706 config
|   |-- lane_706_812_20260531/                # historical two-source config
|   |-- lane_706_812_349_20260609/            # three-source resolution-search configs
|   |-- lane_706_812_349_696_20260624/        # historical four-source fine-tune configs
|   |-- lane_706_812_349_696_350_20260629/    # historical five-source configs
|   `-- lane_706_812_349_696_350_1149_20260722/ # current six-source configs
|-- dataset/                                  # local datasets, not tracked by Git
|   |-- lane_706_20260518/
|   |-- lane_812_20260531/
|   |-- lane_349_20260609/
|   |-- lane_696_20260624/
|   |-- lane_350_20260629.zip
|   |-- lane_350_20260629/
|   |-- lane_1149_20260722.zip
|   |-- lane_1149_20260722/
|   `-- lane_706_812_349_696_350_1149_20260722/splits/ # current virtual merged split
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
dataset/lane_1149_20260722/
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
dataset/lane_706_812_349_696_350_1149_20260722/splits/train.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/val.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/test.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_706_20260518.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_812_20260531.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_349_20260609.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_696_20260624.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_350_20260629.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_1149_20260722.txt
dataset/lane_706_812_349_696_350_1149_20260722/splits/merge_report.json
```

## 5. Training, Evaluation, And Inference

### 5.1 Extract And Split New Data

```bash
test ! -d dataset/lane_1149_20260722
unzip -q dataset/lane_1149_20260722.zip -d dataset

PYTHONPATH=. python tools/prepare_culane_highway.py \
  --data-root dataset/lane_1149_20260722 \
  --seed 0 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

Actual `lane_1149_20260722` split: `train=804`, `val=172`, `test=173`, with `1149` valid image/JSON pairs and `errors=0`. The report contains 89 `points_not_strictly_bottom_to_top` warnings; the loader sorts lane points before use, so these samples remain valid.

### 5.2 Build Six-Source Split

```bash
PYTHONPATH=. python tools/build_highway_multiroot_splits.py \
  --source lane_706_20260518=dataset/lane_706_20260518 \
  --source lane_812_20260531=dataset/lane_812_20260531 \
  --source lane_349_20260609=dataset/lane_349_20260609 \
  --source lane_696_20260624=dataset/lane_696_20260624 \
  --source lane_350_20260629=dataset/lane_350_20260629 \
  --source lane_1149_20260722=dataset/lane_1149_20260722 \
  --out-dir dataset/lane_706_812_349_696_350_1149_20260722/splits \
  --source-priority lane_1149_20260722,lane_350_20260629,lane_696_20260624,lane_349_20260609,lane_812_20260531,lane_706_20260518
```

Current merged counts are `train=2839`, `val=606`, `test=615`. The merge report records two dropped duplicate samples, both from older sources; `lane_1149_20260722` did not introduce a new dropped duplicate.

### 5.3 Train Stage 1

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/train.py \
  configs/clrernet/lane_706_812_349_696_350_1149_20260722/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8.py \
  --work-dir work_dirs/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8_culane_pretrain
```

The config uses `load_from=checkpoints/clrernet_culane_dla34_ema.pth`, `img_scale=(1024,544)`, `top_crop_ratio=0.08`, `batch_size=4`, and gradient accumulation `2`.

### 5.4 Search Stage 1 Post-Processing

Run the grid on merged validation only:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/sweep_highway_stage1_postprocess.py \
  configs/clrernet/lane_706_812_349_696_350_1149_20260722/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8.py \
  work_dirs/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_20.pth \
  --split-file dataset/lane_706_812_349_696_350_1149_20260722/splits/val.txt \
  --out-dir work_dirs/stage1_postprocess/lane_706_812_349_696_350_1149_20260722/epoch_20_val_grid \
  --conf-thresholds 0.25 0.30 0.325 0.35 0.375 0.40 \
  --nms-topks 6 8 10 12 \
  --nms-thres 40 50 60 \
  --device cuda:0
```

The selected six-source setting is `conf_threshold=0.40`, `nms_topk=12`, `nms_thres=40` from epoch 20.

### 5.5 Test Stage 1

Merged test:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py \
  configs/clrernet/lane_706_812_349_696_350_1149_20260722/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8.py \
  work_dirs/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_20.pth
```

Per-source test:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_706_20260518.txt

CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_812_20260531.txt

CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_349_20260609.txt

CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_696_20260624.txt

CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_350_20260629.txt

CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python tools/test.py <config> <checkpoint> \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_1149_20260722.txt
```

### 5.6 Train Legacy Stage 2 (Historical Baseline)

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python -m libs.lane_classifier.train \
  --data-roots lane_706_20260518=dataset/lane_706_20260518 lane_812_20260531=dataset/lane_812_20260531 lane_349_20260609=dataset/lane_349_20260609 lane_696_20260624=dataset/lane_696_20260624 lane_350_20260629=dataset/lane_350_20260629 lane_1149_20260722=dataset/lane_1149_20260722 \
  --split-root dataset/lane_706_812_349_696_350_1149_20260722/splits \
  --work-dir work_dirs/lane_classifier/lane_706_812_349_696_350_1149_20260722_stage2_strip_288x128_w128_scratch \
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

This current command intentionally does not pass `--init-from` or `--resume-from`; Stage 2 is trained from scratch. Use `--resume-from` only when continuing an interrupted run with optimizer/history state.

The current `sequence_fusion` training command, ten-source split, prediction-crop preparation, crop ablation, and seed selection are recorded in `docs/stage2_classification_optimization.md`.

### 5.7 Two-Stage Evaluation

Primary merged test:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python -m libs.lane_classifier.eval \
  --data-roots lane_706_20260518=dataset/lane_706_20260518 lane_812_20260531=dataset/lane_812_20260531 lane_349_20260609=dataset/lane_349_20260609 lane_696_20260624=dataset/lane_696_20260624 lane_350_20260629=dataset/lane_350_20260629 lane_1149_20260722=dataset/lane_1149_20260722 \
  --split-file dataset/lane_706_812_349_696_350_1149_20260722/splits/test.txt \
  --det-config configs/clrernet/lane_706_812_349_696_350_1149_20260722/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_20.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_p70_seed1/best.pth \
  --out-dir work_dirs/two_stage_eval/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_seed1/merged \
  --score-thr 0.4 \
  --det-conf-thr 0.4 \
  --nms-thres 40 \
  --nms-topk 12 \
  --iou-thr 0.3 \
  --match-width 20 \
  --device cuda:0
```

For per-source evaluation, replace `--split-file` and `--out-dir` with:

```text
dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_706_20260518.txt
work_dirs/two_stage_eval/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_seed1/lane706

dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_812_20260531.txt
work_dirs/two_stage_eval/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_seed1/lane812

dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_349_20260609.txt
work_dirs/two_stage_eval/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_seed1/lane349

dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_696_20260624.txt
work_dirs/two_stage_eval/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_seed1/lane696

dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_350_20260629.txt
work_dirs/two_stage_eval/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_seed1/lane350

dataset/lane_706_812_349_696_350_1149_20260722/splits/test_lane_1149_20260722.txt
work_dirs/two_stage_eval/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_seed1/lane1149
```

### 5.8 Inference And Visualization

Single image or folder:

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input path/to/image_or_folder \
  --det-config configs/clrernet/lane_706_812_349_696_350_1149_20260722/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_20.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_p70_seed1/best.pth \
  --out-dir work_dirs/two_stage_infer/custom \
  --score-thr 0.4 \
  --det-conf-thr 0.4 \
  --nms-thres 40 \
  --nms-topk 12 \
  --device cuda:0
```

The command writes:

```text
<out-dir>/predictions/*.json
<out-dir>/visualizations/*.jpg
<out-dir>/summary.json
```
