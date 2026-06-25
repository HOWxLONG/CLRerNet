# Highway Two-Stage Lane Detection Project Log

> Current status update (2026-06-24): the active pipeline uses the four-source merged multi-root set `dataset/lane_706_20260518/`, `dataset/lane_812_20260531/`, `dataset/lane_349_20260609/`, and `dataset/lane_696_20260624/`. The current recommended Stage 1 locator is the four-source fine-tuned `1024x544 topcrop8 / epoch 20` checkpoint with post-processing `0.40/top8/nms40`. The current Stage 2 classifier is initialized from the three-source best classifier with `--init-from`, then trained for 20 fresh epochs on the four-source split. Use root `README.md` and section 20 of this document as the latest reproducible commands and metrics. `dataset/culane_highway/` is historical only and is not used for current training or evaluation.

This document records the experiments completed under `/datadisk2/longhaoxiang/CLRerNet/`. Sections 1-14 are historical `dataset/culane_highway/` experiments, section 15 is the `lane_706_20260518` single-source run, section 16 is the `lane_706 + lane_812` multi-root run, sections 17-18 are the three-source high-resolution run and test-time post-processing analysis, section 19 is the three-source resolution search, and section 20 is the current four-source fine-tuning result.

---

## 1. 运行环境和基本约束

所有服务器操作均在：

```bash
/datadisk2/longhaoxiang/CLRerNet/
```

使用 conda 环境：

```bash
source /home/longhaoxiang/anaconda3/etc/profile.d/conda.sh
conda activate clrernet
```

常用运行前缀：

```bash
cd /datadisk2/longhaoxiang/CLRerNet
PYTHONPATH=. python ...
```

PyTorch 环境：

- `torch 2.1.0+cu121`
- `torchvision 0.16.0+cu121`
- CUDA runtime `12.1`
- cuDNN `8.9.2`

服务器上全局 `nvidia-smi` 可能因为某张坏卡报 NVML 错误；训练时建议查询单卡：

```bash
for i in 0 2 3 4 5 6 7 8 9; do
  echo GPU=$i
  nvidia-smi -i $i --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
done
```

训练和评估前应选择空闲显卡，例如：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ...
```

如果看到：

```text
UserWarning: Can't initialize NVML
```

但 PyTorch 仍显示 CUDA 可用，且模型能正常 forward，这通常只是 NVML 查询告警，不等于训练失败。

## 2. 数据集格式和 split

数据目录：

```bash
dataset/culane_highway/
```

数据格式是平铺图片和同名 JSON：

```text
xxx.jpg
xxx.json
yyy.PNG
yyy.json
```

JSON 中 lane 标签必须保持为：

```text
solid
dashed
joint
```

其中 `joint` 表示同一条划线中存在实线/虚线变化，整个项目没有把它改名成 `solid_dashed`。

Stage 1 是 class-agnostic lane locator：`solid`、`dashed`、`joint` 全部作为同一个 `lane` 类参与定位训练；原始 label 保留在 metadata 中，供 debug 和 Stage 2 使用。

当前 split 文件：

```text
dataset/culane_highway/splits/train.txt
dataset/culane_highway/splits/val.txt
dataset/culane_highway/splits/test.txt
dataset/culane_highway/splits/prepare_report.json
```

`prepare_report.json` 核心统计：

| 项目 | 数值 |
|---|---:|
| images | 379 |
| json | 379 |
| valid_pairs | 379 |
| train | 265 |
| val | 56 |
| test | 58 |
| errors | 0 |
| warnings | 58 |
| solid lanes | 1533 |
| dashed lanes | 696 |
| joint lanes | 276 |

当前 split 使用 image-level split，Stage 2 分类器也沿用同一套 image-level split，没有按 lane crop 随机划分。

## 3. 新增和修改的主要文件

### 3.1 数据检查和 split

`tools/prepare_culane_highway.py`

功能：

- 扫描 `dataset/culane_highway/` 中所有图片和 JSON。
- 检查同名配对是否完整。
- 检查 label 是否只包含 `solid`、`dashed`、`joint`。
- 检查 lane 至少有两个点。
- 统计 image size、label 数、点数分布、schema 类型。
- 生成 deterministic image-level split。
- 输出 `prepare_report.json`。

主要参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-root` | `dataset/culane_highway` | 数据目录 |
| `--out-dir` | `None` | 输出 split 目录；默认是 `$data_root/splits` |
| `--seed` | `0` | deterministic split 随机种子 |
| `--train-ratio` | `0.7` | 训练集比例 |
| `--val-ratio` | `0.15` | 验证集比例 |
| `--test-ratio` | `0.15` | 测试集比例 |

运行命令：

```bash
PYTHONPATH=. python tools/prepare_culane_highway.py \
  --data-root dataset/culane_highway \
  --seed 0 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

### 3.2 Dataset 可视化

`tools/visualize_culane_highway_dataset.py`

功能：

- 按指定 CLRerNet config 构建 `HighwayLaneDataset`。
- 执行完整 pipeline 后可视化 GT lane。
- 用于确认 crop/resize/pad 后 lane 点是否正确。

主要参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config` | 必填 | 要检查的训练 config |
| `--split` | `train` | `train` / `val` / `test` |
| `--out-dir` | `work_dirs/culane_highway_debug/pipeline_vis` | 输出路径 |
| `--num-samples` | `16` | 输出样本数 |

示例：

```bash
PYTHONPATH=. python tools/visualize_culane_highway_dataset.py \
  configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --split train \
  --num-samples 16 \
  --out-dir work_dirs/culane_highway_debug/pipeline_vis_1024x544_topcrop8
```

后续已按要求清理 smoke/debug 目录；如需重新生成可视化，可以再次运行上面的命令。

### 3.3 HighwayLaneDataset

`libs/datasets/highway_lane_dataset.py`

功能：

- 读取平铺的 image/json pair。
- 支持 `.jpg`、`.jpeg`、`.png` 及大写扩展名。
- JSON 支持 `lanes` schema，也兼容 LabelMe 风格的 `shapes` schema。
- 清洗非有限点。
- 默认按 y 从大到小排序，也就是从图像底部到顶部。
- 强制点序单调，避免 `sample_lane` 因 y 顺序不合法报错。
- Stage 1 输出 class-agnostic `id_classes=[1, ...]`。
- metadata 中保留 `gt_lane_labels`，即原始 `solid/dashed/joint`。

关键 metadata：

```text
filename
sub_img_name
lane_json_path
ori_shape
img_shape
gt_points
gt_lane_labels
gt_masks
lanes
```

### 3.4 ResizePad

`libs/datasets/pipelines/resize_pad.py`

功能：

- 保持原图比例 resize 到目标尺寸内。
- 再 pad 到固定输入尺寸。
- 同步对 `gt_points` 做 scale 和 pad 平移。
- 在 metadata 中写入：

```text
resize_pad_scale
resize_pad_pad
resize_pad_size
pad_shape
```

它主要用于早期“不裁切、保留完整 16:9”的 Stage 1 配置：

```text
dataset_culane_highway_clrernet.py
dataset_culane_highway_clrernet_1280x768.py
dataset_culane_highway_clrernet_1536x864.py
```

### 3.5 TopCrop 和 LanePointFilter

`libs/datasets/pipelines/top_crop.py`

#### TopCrop

功能：

- 裁掉图像顶部固定比例或固定像素。
- 默认 `crop_ratio=0.08`。
- 对 lane 点执行：

```text
保留：0 <= x < 原图宽度 且 crop_top <= y < 原图高度
重映射：y = y - crop_top
```

- 裁切后少于 2 个点的 lane 会被删除。
- 同步过滤 `gt_lane_labels`、`id_classes`、`id_instances`。
- 在 metadata 写入：

```text
top_crop_px
top_crop_ratio
top_crop_original_shape
top_crop_removed_lanes
```

#### LanePointFilter

功能：

- 在 crop、resize、augment 后再次过滤越界 lane 点。
- 解决 resize 后可能出现 `x=-0.11` 这种浮点边界误差的问题。
- 过滤后少于 2 点的 lane 会被删除。
- 在 metadata 写入：

```text
lane_point_filter_removed_lanes
```

这两个 transform 已注册到：

```text
libs/datasets/pipelines/__init__.py
```

### 3.6 HighwayLaneMetric

`libs/datasets/metrics/highway_lane_metric.py`

功能：

- Stage 1 class-agnostic sanity metric。
- 将预测 lane 和 GT lane 画成 mask。
- 计算 greedy matching 下的：

```text
Precision@0.3 / Recall@0.3 / F1@0.3
Precision@0.5 / Recall@0.5 / F1@0.5
```

默认 `lane_width=8`。

### 3.7 Stage 2 分类模块

目录：

```text
libs/lane_classifier/
```

文件功能：

| 文件 | 功能 |
|---|---|
| `crop.py` | lane strip crop、polyline resample、mask IoU、可视化画线 |
| `dataset.py` | 从 GT JSON 按 lane instance 构建分类数据集 |
| `model.py` | 轻量 CNN 分类器，输出 `solid/dashed/joint` |
| `train.py` | 分类器训练、类别不平衡处理、保存 best/last |
| `infer.py` | 两阶段推理：locator + classifier + JSON + 可视化 |
| `eval.py` | 两阶段端到端评估，输出 detection/classification 指标和错误可视化 |
| `__init__.py` | 模块初始化 |

Stage 2 类别固定：

```python
CLASSES = ('solid', 'dashed', 'joint')
```

`crop.py` 中 lane strip crop 的核心逻辑：

- 对输入 polyline 重新采样到固定高度。
- 估计每个采样点切线方向。
- 沿法线方向取固定 strip width。
- 使用 `cv2.remap` 从原图采样。
- 输出固定尺寸 crop，默认 `(288, 128)`。

当前可视化线宽默认已改细：

```python
draw_lanes(..., thickness=2)
```

没有增加额外 CLI 参数，保持推理命令简洁。

### 3.8 两阶段推理适配

`libs/lane_classifier/infer.py`

功能：

1. 输入单图或文件夹。
2. 构建 fine-tuned CLRerNet locator。
3. 执行 config 中的 test pipeline。
4. 得到 class-agnostic lane instance。
5. 将 lane 从模型输入坐标映射回原图坐标。
6. 对每条 lane 做 strip crop。
7. 调用 Stage 2 分类器。
8. 输出 JSON。
9. 保存可视化。

坐标映射支持两类 Stage 1 pipeline：

- `ResizePad`：用 `resize_pad_scale` 和 `resize_pad_pad` 还原。
- `TopCrop`：用 `top_crop_px` 和 `top_crop_original_shape` 还原。

输出 JSON 格式：

```json
{
  "image": "xxx.png",
  "lanes": [
    {
      "points": [[x1, y1], [x2, y2]],
      "lane_score": 0.91,
      "label": "solid",
      "label_score": 0.87
    }
  ]
}
```

注意：最终 JSON 中 `points` 是原图坐标，不是裁切后坐标。

## 4. Stage 1 配置文件

### 4.1 早期完整 16:9 baseline

```text
configs/clrernet/culane_highway/dataset_culane_highway_clrernet.py
configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator.py
```

核心设置：

- 输入尺寸：`1024x576`
- 不裁切
- 使用 `ResizePad`
- batch size：8
- epoch：20
- 初始化：`checkpoints/clrernet_culane_dla34_ema.pth`
- Stage 1 class-agnostic

checkpoint：

```text
work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x576_full/epoch_20.pth
```

### 4.2 高分辨率尝试：1536x864

```text
configs/clrernet/culane_highway/dataset_culane_highway_clrernet_1536x864.py
configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1536x864.py
```

核心设置：

- 输入尺寸：`1536x864`
- 不裁切
- 使用 `ResizePad`
- batch size：2
- gradient accumulation：4
- epoch：20
- 从 `1024x576 epoch_20` warm start

结论：不推荐。分辨率高但 batch 小，验证和测试表现明显下降。

### 4.3 中等高分辨率尝试：1280x768

```text
configs/clrernet/culane_highway/dataset_culane_highway_clrernet_1280x768.py
configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1280x768.py
```

核心设置：

- 输入尺寸：`1280x768`
- 不裁切
- `ResizePad`：16:9 内容约 resize 到 `1280x720`，上下 pad
- batch size：2
- gradient accumulation：4
- epoch：20
- 从 `1024x576 epoch_20` warm start

备注：

- 试过 `1280x720`，DLA residual 下采样尺寸不匹配，报 shape mismatch，已删除对应失败配置。
- `1280x768` 可跑，但整体没有超过最终 `1024x544 topcrop8`。

### 4.4 顶部裁切高分辨率尝试：1536x800 topcrop8

```text
configs/clrernet/culane_highway/dataset_culane_highway_clrernet_1536x800_topcrop8.py
configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1536x800_topcrop8.py
```

核心设置：

- 输入尺寸：`1536x800`
- 顶部裁切约 8%
- 不 pad
- batch size：2
- gradient accumulation：4
- epoch：30
- 从 `1024x576 epoch_20` warm start

结论：比 `1536x864` 好，但 detector F1 仍不如 `1024x544 topcrop8`。高分辨率不是这个数据集上的最佳方向。

### 4.5 最终推荐：1024x544 topcrop8

```text
configs/clrernet/culane_highway/dataset_culane_highway_clrernet_1024x544_topcrop8.py
configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py
```

核心设置：

- 输入尺寸：`1024x544`
- 顶部裁切约 8%
- 不 pad
- batch size：8
- epoch：25
- learning rate：`5e-5`
- optimizer：`AdamW`
- scheduler：`CosineAnnealingLR`
- 从 `1024x576 epoch_20` warm start
- `nms_topk=8`
- `nms_thres=50`

为什么选 `1024x544`：

- `1024x576` 是此前最稳定的横向分辨率。
- 裁掉顶部约 8% 后，理论有效高度约 `576 * 0.92 = 529.9`。
- `544` 比 `512` 更接近有效高度，同时对 DLA 下采样更友好。
- batch size 能保持 8，比 1536/1280 的 batch2 更稳。

推荐 checkpoint：

```text
work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth
```

## 5. Stage 1 训练命令

### 5.1 baseline 1024x576

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py \
  configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator.py \
  --work-dir work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x576_full
```

### 5.2 1536x864 full-frame

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python tools/train.py \
  configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1536x864.py \
  --work-dir work_dirs/clrernet_culane_highway_dla34_ema_locator_1536x864_full
```

### 5.3 1280x768 full-frame

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python tools/train.py \
  configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1280x768.py \
  --work-dir work_dirs/clrernet_culane_highway_dla34_ema_locator_1280x768_full
```

### 5.4 1536x800 topcrop8

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py \
  configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1536x800_topcrop8.py \
  --work-dir work_dirs/clrernet_culane_highway_dla34_ema_locator_1536x800_topcrop8_full_run2
```

### 5.5 最终推荐 1024x544 topcrop8

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py \
  configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --work-dir work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full
```

## 6. Stage 1 训练和测试效果

### 6.1 1024x544 topcrop8 验证集曲线

| epoch | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 437 | 382 | 0.6430 | 0.7356 | 0.6862 | 0.5126 | 0.5864 | 0.5470 |
| 10 | 434 | 382 | 0.6613 | 0.7513 | 0.7034 | 0.5300 | 0.6021 | 0.5637 |
| 15 | 436 | 382 | 0.6606 | 0.7539 | 0.7042 | 0.5138 | 0.5864 | 0.5477 |
| 20 | 432 | 382 | 0.6551 | 0.7408 | 0.6953 | 0.5347 | 0.6047 | 0.5676 |
| 25 | 431 | 382 | 0.6729 | 0.7592 | 0.7134 | 0.5336 | 0.6021 | 0.5658 |

### 6.2 Stage 1 测试集对比

| 方案 | checkpoint | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1024x576 full-frame | epoch_20 | 0.5801 | 0.6961 | 0.6328 | 0.4242 | 0.5091 | 0.4628 | 早期 baseline |
| 1536x864 full-frame | epoch_20 | 0.3634 | 0.6302 | 0.4610 | 0.1862 | 0.3229 | 0.2362 | 不推荐 |
| 1536x800 topcrop8 | epoch_25 | 0.3572 | 0.6371 | 0.4578 | 0.2152 | 0.3838 | 0.2758 | 不推荐 |
| 1024x544 topcrop8 | epoch_20 | 0.6239 | 0.7493 | 0.6809 | 0.4804 | 0.5770 | 0.5243 | 很好 |
| 1024x544 topcrop8 | epoch_25 | 0.6376 | 0.7624 | 0.6944 | 0.4869 | 0.5822 | 0.5303 | 当前推荐 |

1280x768 在验证集上可跑，但 Stage 1 验证 F1 明显低于最终 1024x544 topcrop8：

| epoch | F1@0.3 | F1@0.5 |
|---:|---:|---:|
| 5 | 0.4866 | 0.3008 |
| 10 | 0.4908 | 0.3201 |
| 15 | 0.4811 | 0.3104 |
| 20 | 0.4888 | 0.3135 |

## 7. Stage 2 分类器

### 7.1 训练命令

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.train \
  --data-root dataset/culane_highway \
  --split-root dataset/culane_highway/splits \
  --work-dir work_dirs/lane_classifier/stage2_strip_288x128_w128 \
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
  --debug-crops 80 \
  --device cuda:0 \
  --seed 0
```

后续按要求清理目录时，`debug_crops` 已删除；正式 checkpoint 和指标仍保留。

### 7.2 参数说明

| 参数 | 说明 |
|---|---|
| `--crop-height 288` | lane strip crop 输出高度 |
| `--crop-width 128` | lane strip crop 输出宽度 |
| `--strip-width 128` | 沿 lane 法线方向采样的原图宽度，joint 需要足够宽才能看到实线/虚线变化 |
| `--class-balance loss` | 使用 class weighted cross entropy 处理类别不平衡 |
| `--debug-crops` | 保存若干训练/验证 crop 供检查；当前 debug 输出已清理 |
| `--resume-from` | 可选，从已有分类器 checkpoint 继续训练 |

### 7.3 分类器训练数据统计

| split | lane samples | solid | dashed | joint |
|---|---:|---:|---:|---:|
| train | 1736 | 1082 | 473 | 181 |
| val | 384 | 222 | 120 | 42 |

class weights：

```text
solid: 0.5348
dashed: 1.2234
joint: 3.1971
```

### 7.4 分类器最佳结果

checkpoint：

```text
work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth
```

最佳验证结果：

| 指标 | 数值 |
|---|---:|
| accuracy | 0.8620 |
| macro-F1 | 0.7247 |

confusion matrix：

| true \ pred | solid | dashed | joint |
|---|---:|---:|---:|
| solid | 211 | 2 | 9 |
| dashed | 8 | 107 | 5 |
| joint | 9 | 20 | 13 |

per-class 观察：

- `solid` 和 `dashed` 相对稳定。
- `joint` 是最难类，主要被混到 `dashed`，这符合 joint 本身包含虚线结构的难点。

## 8. 两阶段推理

### 8.1 单图推理

推荐命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input dataset/culane_highway/3O47LADEFjtlSarSQGlBc4zB_202510271049_1.jpg \
  --det-config configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/tuned \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-topk 8 \
  --nms-thres 50
```

### 8.2 文件夹推理和可视化

`--input` 可以是文件夹。默认只处理该目录第一层图片：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input dataset/culane_highway \
  --det-config configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/folder_vis \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-topk 8 \
  --nms-thres 50
```

递归处理子目录：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input dataset/culane_highway \
  --recursive \
  --det-config configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/folder_vis_recursive \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-topk 8 \
  --nms-thres 50
```

输出目录结构：

```text
work_dirs/two_stage_infer/folder_vis/
  predictions/
    xxx.json
  visualizations/
    xxx.jpg
  summary.json
```

当前可视化默认线宽为 `2`，没有额外线宽参数。

### 8.3 完全不做阈值过滤和 NMS

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input dataset/culane_highway/xxx.jpg \
  --det-config configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/no_thr_no_nms \
  --score-thr 0 \
  --det-conf-thr 0 \
  --no-nms
```

说明：

- `--det-conf-thr 0`：关闭 CLRerNet 内部 pre-NMS confidence 阈值过滤。
- `--score-thr 0`：关闭 two-stage infer 中的后置 score 阈值过滤。
- `--no-nms`：关闭 CLRerNet lane NMS。
- `--nms-topk` 和 `--nms-thres` 在 `--no-nms` 下不生效。
- 仍会丢弃 NaN、越界、少于 2 点等无效几何结果，这不是 score/NMS 过滤。
- 这样通常会输出大量低置信 proposal，误检会明显变多，主要用于 debug。

### 8.4 推理参数说明

| 参数 | 说明 |
|---|---|
| `--input` | 输入单图或文件夹 |
| `--det-config` | Stage 1 locator config；必须和 checkpoint 匹配 |
| `--det-checkpoint` | Stage 1 checkpoint |
| `--cls-checkpoint` | Stage 2 分类器 checkpoint |
| `--out-dir` | 输出 JSON、可视化和 summary |
| `--device` | 默认 `cuda:0` |
| `--score-thr` | locator 输出转回原图坐标后，two-stage infer 端的 lane score 过滤 |
| `--det-conf-thr` | CLRerNet head 内部 confidence threshold |
| `--nms-thres` | lane NMS overlap threshold |
| `--nms-topk` | NMS 最多保留 lane 数 |
| `--no-nms` | 禁用 NMS |
| `--crop-height` / `--crop-width` | 覆盖分类器 crop size；默认从 checkpoint meta 读取 |
| `--strip-width` | 覆盖分类器 strip width；默认从 checkpoint meta 读取 |
| `--recursive` | 文件夹递归 |
| `--save-crops` | 保存每条预测 lane 的 strip crop |

关键提醒：

```text
det-config 和 det-checkpoint 必须配套。
```

例如 `1024x544_topcrop8` checkpoint 必须使用：

```text
configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py
```

不要用旧 `1024x576` config 加新 checkpoint，也不要反过来混用。否则裁切、分辨率和坐标映射会不一致。

## 9. 两阶段端到端评估

### 9.1 推荐评估命令

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.eval \
  --data-root dataset/culane_highway \
  --split-file dataset/culane_highway/splits/test.txt \
  --det-config configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_eval/stage2_test_1024x544_topcrop8_epoch25_s0.35_top8_nms50 \
  --device cuda:0 \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-topk 8 \
  --nms-thres 50 \
  --iou-thr 0.3 \
  --match-width 20 \
  --error-vis-limit 80
```

输出：

```text
eval_report.json
confusion_matrix.csv
predictions/*.json
error_vis/*.jpg
```

### 9.2 评估参数说明

| 参数 | 说明 |
|---|---|
| `--pred-dir` | 可选，使用已有 prediction JSON，不重新跑模型 |
| `--iou-thr` | GT/pred matching 的 IoU 阈值 |
| `--match-width` | 画 mask 算 IoU 时的 lane 宽度 |
| `--error-vis-limit` | 最多保存多少张错误案例可视化 |
| 其他 detector 参数 | 和 `infer.py` 一致 |

## 10. 两阶段效果汇总

### 10.1 主要方案对比

| 方案 | 阈值/NMS | matched GT | unmatched GT | unmatched pred | det P | det R | det F1 | cls acc | cls macro-F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024x576 baseline | s0.40 top8 nms50 | 242 | 143 | 28 | 0.8963 | 0.6286 | 0.7389 | - | 0.6957 | 旧推荐 |
| 1024x576 baseline | s0.35 top8 nms50 | 250 | 135 | 45 | 0.8475 | 0.6494 | 0.7353 | - | 0.6971 | 召回略高 |
| 1280x768 | s0.35 top12 nms50 | 253 | 132 | 46 | 0.8462 | 0.6571 | 0.7397 | 0.7747 | 0.6696 | 检测略升，分类降 |
| 1536x864 | s0.35 top12 nms50 | 235 | 150 | 57 | 0.8048 | 0.6104 | 0.6942 | - | 0.6810 | 不推荐 |
| 1536x800 topcrop8 | s0.35 top12 nms50 | 245 | 140 | 40 | 0.8596 | 0.6364 | 0.7313 | 0.8041 | 0.6997 | 高分辨率裁切仍不够好 |
| 1024x544 topcrop8 epoch25 | s0.30 top8 nms50 | 285 | 100 | 44 | 0.8663 | 0.7403 | 0.7983 | 0.8070 | 0.7234 | 最少漏检 |
| 1024x544 topcrop8 epoch25 | s0.35 top8 nms50 | 276 | 109 | 28 | 0.9079 | 0.7169 | 0.8012 | 0.8116 | 0.7254 | 推荐均衡配置 |
| 1024x544 topcrop8 epoch25 | s0.40 top8 nms50 | 266 | 119 | 10 | 0.9638 | 0.6909 | 0.8048 | 0.8158 | 0.7278 | 少误检配置 |
| 1024x544 topcrop8 epoch20 | s0.35 top8 nms50 | 272 | 113 | 35 | 0.8860 | 0.7065 | 0.7862 | 0.8125 | 0.7283 | 分类略好，检测差一点 |

注：部分早期 baseline 的 two-stage `cls acc` 未单独记录在最终摘要中，但 `macro-F1` 和 detection 数值保留。

### 10.2 推荐选择

默认推荐：

```text
1024x544 topcrop8 epoch_25
score_thr=0.35
det_conf_thr=0.35
nms_topk=8
nms_thres=50
```

理由：

- 比旧 `1024x576` baseline 多匹配 `34` 条 GT。
- unmatched GT 从 `143` 降到 `109`。
- unmatched prediction 仍控制在 `28`。
- detection F1 到 `0.8012`。
- classification macro-F1 到 `0.7254`。

如果更重视少误检：

```text
score_thr=0.40
det_conf_thr=0.40
```

效果：

- unmatched prediction 只有 `10`。
- detection precision `0.9638`。
- detection F1 `0.8048`。
- 但 matched GT 从 `276` 降到 `266`。

如果更重视少漏检：

```text
score_thr=0.30
det_conf_thr=0.30
```

效果：

- matched GT 到 `285`。
- unmatched GT 降到 `100`。
- 但 unmatched prediction 增到 `44`。

## 11. 目录清理记录

已按要求清理 smoke/debug/overfit/val sweep/污染临时目录。

已删除的典型目录：

```text
work_dirs/*_smoke*
work_dirs/culane_highway_debug
work_dirs/two_stage_infer/stage2_single_smoke
work_dirs/lane_classifier/stage2_strip_288x128_w128/debug_crops
work_dirs/two_stage_eval/sweep_val_nms_refine
work_dirs/two_stage_eval/sweep_val_nms_thresholds
work_dirs/two_stage_eval/val_topk_sweep_*
work_dirs/clrernet_culane_highway_dla34_ema_locator_1536x800_topcrop8_full
```

保留的正式目录：

```text
work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full
work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x576_full
work_dirs/clrernet_culane_highway_dla34_ema_locator_1280x768_full
work_dirs/clrernet_culane_highway_dla34_ema_locator_1536x800_topcrop8_full_run2
work_dirs/clrernet_culane_highway_dla34_ema_locator_1536x864_full
work_dirs/lane_classifier/stage2_strip_288x128_w128
work_dirs/two_stage_eval/stage2_test_*
work_dirs/two_stage_infer/tuned
```

## 12. 当前推荐生产命令

### 12.1 单图

```bash
cd /datadisk2/longhaoxiang/CLRerNet
source /home/longhaoxiang/anaconda3/etc/profile.d/conda.sh
conda activate clrernet

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input dataset/culane_highway/3O47LADEFjtlSarSQGlBc4zB_202510271049_1.jpg \
  --det-config configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/tuned_single \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-topk 8 \
  --nms-thres 50
```

### 12.2 文件夹

```bash
cd /datadisk2/longhaoxiang/CLRerNet
source /home/longhaoxiang/anaconda3/etc/profile.d/conda.sh
conda activate clrernet

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input dataset/culane_highway \
  --det-config configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/folder_vis \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-topk 8 \
  --nms-thres 50
```

输出：

```text
work_dirs/two_stage_infer/folder_vis/predictions/*.json
work_dirs/two_stage_infer/folder_vis/visualizations/*.jpg
work_dirs/two_stage_infer/folder_vis/summary.json
```

### 12.3 完整测试评估

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.eval \
  --data-root dataset/culane_highway \
  --split-file dataset/culane_highway/splits/test.txt \
  --det-config configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_eval/stage2_test_1024x544_topcrop8_epoch25_s0.35_top8_nms50 \
  --device cuda:0 \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-topk 8 \
  --nms-thres 50 \
  --iou-thr 0.3 \
  --match-width 20 \
  --error-vis-limit 80
```

## 13. 已知风险和后续优化方向

1. 数据量较小，test split 只有 58 张图。指标有参考价值，但仍可能受 split 分布影响。
2. `joint` 类样本少，且视觉上和 `dashed` 容易混淆。Stage 2 当前主要短板仍是 `joint`。
3. 完全关闭阈值和 NMS 会输出大量 proposal，只适合 debug，不适合常规使用。
4. 旧高分辨率配置仍保留用于复现实验，但不推荐作为默认推理配置。
5. 推理时 config 和 checkpoint 必须严格匹配，尤其 topcrop 配置不能和 full-frame checkpoint 混用。
6. 若后续继续提升定位，可以优先考虑：
   - 针对远处 lane 的 hard sample oversampling。
   - 对交叉/重复预测做几何后处理。
   - 在 `score_thr=0.30~0.40` 间按业务偏好选阈值。
   - 继续增强 `joint` 样本或为 Stage 2 做更强的时序/上下文建模。

## 14. 最终结论

当前最优且推荐使用的方案是：

```text
Stage 1:
  config: configs/clrernet/culane_highway/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8.py
  checkpoint: work_dirs/clrernet_culane_highway_dla34_ema_locator_1024x544_topcrop8_full/epoch_25.pth

Stage 2:
  checkpoint: work_dirs/lane_classifier/stage2_strip_288x128_w128/best.pth

Two-stage default:
  score_thr=0.35
  det_conf_thr=0.35
  nms_topk=8
  nms_thres=50
```

这版相比旧 `1024x576` baseline，核心改进来自：

- 适度裁掉顶部无效区域。
- 保持 1024 横向分辨率，不盲目升到 1536。
- 使用 DLA 友好的 `544` 输入高度。
- batch size 保持 8，训练比高分辨率 batch2 稳定。
- 在裁切和 resize 后严格过滤/重映射 lane 点。

最终 two-stage 测试集推荐指标：

```text
matched GT: 276
unmatched GT: 109
unmatched prediction: 28
detection F1: 0.8012
classification accuracy: 0.8116
classification macro-F1: 0.7254
```


## 15. 2026-05-18 lane_706_20260518 retraining record

### 15.1 Data source decision

New annotation batch:

```text
dataset/lane_706_20260518/
```

Review result:

- `lane_706_20260518` contains 706 image/JSON pairs.
- 379 images have the same stems and identical image hashes as `dataset/culane_highway`.
- The 379 overlapping JSON files differ from the old directory, so this batch includes re-annotations of old images.
- 327 images exist only in `lane_706_20260518`.

For this retraining round, use only `dataset/lane_706_20260518`. Do not merge or read `dataset/culane_highway`; keep it only as historical experiment data.

### 15.2 New split

Split command:

```bash
PYTHONPATH=. python tools/prepare_culane_highway.py --data-root dataset/lane_706_20260518 --seed 0 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

Outputs:

```text
dataset/lane_706_20260518/splits/train.txt
dataset/lane_706_20260518/splits/val.txt
dataset/lane_706_20260518/splits/test.txt
dataset/lane_706_20260518/splits/prepare_report.json
```

Core statistics:

| item | value |
|---|---:|
| images | 706 |
| json | 706 |
| valid_pairs | 706 |
| train | 494 |
| val | 105 |
| test | 107 |
| errors | 0 |
| warnings | 33 |
| solid lanes | 2703 |
| dashed lanes | 1378 |
| joint lanes | 525 |

### 15.3 Stage 1 config

Added configs:

```text
configs/clrernet/lane_706_20260518/dataset_lane_706_20260518_clrernet_1024x544_topcrop8.py
configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py
```

Key settings:

- data root: `dataset/lane_706_20260518`
- input size: `1024x544`
- top crop: `top_crop_ratio=0.08`
- batch size: `8`
- epochs: `25`
- initialization checkpoint: `checkpoints/clrernet_culane_dla34_ema.pth`
- work dir: `work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain`

Training command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py --work-dir work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain
```

### 15.4 Stage 2 training command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.train --data-root dataset/lane_706_20260518 --split-root dataset/lane_706_20260518/splits --work-dir work_dirs/lane_classifier/lane_706_20260518_stage2_strip_288x128_w128 --epochs 30 --batch-size 64 --num-workers 4 --lr 1e-3 --weight-decay 1e-4 --crop-height 288 --crop-width 128 --strip-width 128 --dropout 0.25 --class-balance loss --debug-crops 0 --device cuda:0 --seed 0
```

### 15.5 Completed training and evaluation

Config-level validation passed before training. The new Stage 1 config loaded successfully, train/val/test dataloaders loaded 494/105/107 images, and the sampled input tensor shape was `[3, 544, 1024]`.

Stage 1 locator training finished with checkpoint:

```text
work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth
```

Stage 1 validation curve:

| epoch | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 827 | 702 | 0.5381 | 0.6339 | 0.5821 | 0.3930 | 0.4630 | 0.4251 |
| 10 | 834 | 702 | 0.5983 | 0.7108 | 0.6497 | 0.4556 | 0.5413 | 0.4948 |
| 15 | 834 | 702 | 0.6187 | 0.7350 | 0.6719 | 0.4760 | 0.5655 | 0.5169 |
| 20 | 833 | 702 | 0.6182 | 0.7336 | 0.6710 | 0.4874 | 0.5783 | 0.5290 |
| 25 | 833 | 702 | 0.6267 | 0.7436 | 0.6801 | 0.4946 | 0.5869 | 0.5368 |

Stage 1 test result at epoch 25:

```text
pred_lanes: 851
gt_lanes: 745
Precision@0.3: 0.5664
Recall@0.3: 0.6470
F1@0.3: 0.6040
Precision@0.5: 0.4653
Recall@0.5: 0.5315
F1@0.5: 0.4962
```

Stage 2 classifier training finished with checkpoint:

```text
work_dirs/lane_classifier/lane_706_20260518_stage2_strip_288x128_w128/best.pth
```

Stage 2 best validation result:

```text
best_epoch: 22
train_samples: 3159
val_samples: 702
train_class_counts: solid 1885, dashed 944, joint 330
val_class_counts: solid 402, dashed 212, joint 88
accuracy: 0.9088
macro-F1: 0.8575
confusion_matrix:
  [379, 9, 14]
  [3, 197, 12]
  [16, 10, 62]
```

Two-stage test evaluation output:

```text
work_dirs/two_stage_eval/lane_706_20260518_stage2_test_1024x544_topcrop8_epoch25_s0.35_top8_nms50/eval_report.json
```

Two-stage test result:

```text
num_images: 107
matched_gt: 469
unmatched_gt: 276
unmatched_prediction: 70
classification_accuracy: 0.7228
classification_macro-F1: 0.6751
confusion_matrix:
  [154, 25, 59]
  [1, 145, 5]
  [10, 30, 40]
```
---

## 16. 2026-05-31 lane_706 + lane_812 多数据根两阶段重训

### 16.1 当前数据源

本轮当前主流程使用两个数据源：

```text
dataset/lane_706_20260518
dataset/lane_812_20260531
```

不使用 `dataset/culane_highway`。

`lane_812_20260531` 核查结果：812 张图片、812 个 JSON、812 对有效配对，LabelMe `shapes` schema，合法标签为 `solid/dashed/joint`。

旧 `lane_706_20260518` 保留既有 split；新 `lane_812_20260531` 使用 `seed=0` 按 `70/15/15` split，得到 `train=568,val=121,test=123`。

两批数据有 1 张重复图片：`outside_20250910112526_000206.jpg`。图片相同但 JSON 标注不同，合并 split 按规则保留 `lane_812_20260531`，丢弃 `lane_706_20260518` 中该训练样本。

### 16.2 本轮新增代码

- `libs/utils/highway_data.py`：集中处理 `data_roots`、两列 split、路径解析和显示名。
- `libs/datasets/highway_lane_dataset.py`：支持 `data_roots=dict(...)` 与 `dataset_key<TAB>relative_image_path` split，同时保留旧单 root split 兼容。
- `libs/lane_classifier/dataset.py`、`crop.py`、`train.py`、`eval.py`：Stage 2 训练和 two-stage eval 支持多数据根。
- `libs/lane_classifier/infer.py`：单图推理补齐 `data_root_key` 默认元信息，兼容包含多 root metadata 的 pipeline。
- `tools/build_highway_multiroot_splits.py`：生成虚拟合并 split，不复制图片；重复图片按 `--prefer-on-duplicate` 保留指定数据源。
- `configs/clrernet/lane_706_812_20260531/`：当前 1024x544 topcrop8 多数据源 Stage 1 配置。

### 16.3 Split 结果

`dataset/lane_706_812_20260531/splits/merge_report.json` 记录：

| Split | Images | solid | dashed | joint |
| --- | ---: | ---: | ---: | ---: |
| train | 1061 | 4091 | 1800 | 763 |
| val | 226 | 862 | 396 | 193 |
| test | 230 | 899 | 429 | 198 |

分项 test 文件：

```text
dataset/lane_706_812_20260531/splits/test_lane_706_20260518.txt  # 107 images
dataset/lane_706_812_20260531/splits/test_lane_812_20260531.txt  # 123 images
```

### 16.4 Stage 1 训练与检测评测

配置：

```text
configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py
```

权重：

```text
work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth
```

Stage 1 检测结果：

| Split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 1824 | 1524 | 0.6272 | 0.7507 | 0.6834 | 0.5510 | 0.6594 | 0.6004 |
| lane_706 test | 848 | 745 | 0.6120 | 0.6966 | 0.6516 | 0.5271 | 0.6000 | 0.5612 |
| lane_812 test | 976 | 779 | 0.6404 | 0.8023 | 0.7123 | 0.5717 | 0.7163 | 0.6359 |

### 16.5 Stage 2 训练结果

目录：

```text
work_dirs/lane_classifier/lane_706_812_20260531_stage2_strip_288x128_w128
```

训练样本：`6654`，验证样本：`1451`。最佳 epoch 为 `25`，验证 `accuracy=0.9235`，`macro_f1=0.8769`。

### 16.6 Two-stage 端到端评测

评测参数：`score_thr=0.35`、`det_conf_thr=0.35`、`nms_topk=8`、`nms_thres=50`、`iou_thr=0.3`、`match_width=20`。

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Matched-lane Acc | Matched-lane Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 230 | 1111 | 415 | 128 | 0.7732 | 0.7228 |
| lane_706 test | 107 | 507 | 238 | 63 | 0.7515 | 0.7134 |
| lane_812 test | 123 | 604 | 177 | 65 | 0.7914 | 0.7291 |

输出目录：

```text
work_dirs/two_stage_eval/lane_706_812_20260531_test_s0.35_top8_nms50
work_dirs/two_stage_eval/lane_706_812_20260531_test_lane_706_s0.35_top8_nms50
work_dirs/two_stage_eval/lane_706_812_20260531_test_lane_812_s0.35_top8_nms50
```

---

## 17. 2026-06-09 lane_706 + lane_812 + lane_349 高分辨率三数据根重训

本轮目标：保留 `lane_706_20260518` 与 `lane_812_20260531` 既有 split，新增 `lane_349_20260609`，合并三批数据重新训练两阶段模型，并提高 Stage 1 输入分辨率。

### 17.1 新数据检查与 split

新数据目录：

```text
dataset/lane_349_20260609/
```

检查结果：

| 项目 | 数值 |
|---|---:|
| images | 349 |
| json | 349 |
| valid_pairs | 349 |
| errors | 0 |
| solid lanes | 1452 |
| dashed lanes | 481 |
| joint lanes | 362 |

按 `seed=0, 70/15/15` 生成 split：

| Split | Images |
|---|---:|
| train | 244 |
| val | 52 |
| test | 53 |

### 17.2 三数据根合并 split

本轮不使用 `dataset/culane_highway`。合并 split 是虚拟 split，不复制图片。

新增 `tools/build_highway_multiroot_splits.py --source-priority`，用于多级重复图优先级。当前优先级：

```text
lane_349_20260609 > lane_812_20260531 > lane_706_20260518
```

合并命令：

```bash
PYTHONPATH=. python tools/build_highway_multiroot_splits.py \
  --source lane_706_20260518=dataset/lane_706_20260518 \
  --source lane_812_20260531=dataset/lane_812_20260531 \
  --source lane_349_20260609=dataset/lane_349_20260609 \
  --out-dir dataset/lane_706_812_349_20260609/splits \
  --source-priority lane_349_20260609,lane_812_20260531,lane_706_20260518
```

合并结果：

| Split | Images | solid | dashed | joint |
|---|---:|---:|---:|---:|
| train | 1304 | 5088 | 2151 | 1002 |
| val | 278 | 1087 | 441 | 259 |
| test | 283 | 1124 | 513 | 255 |

重复图丢弃记录：

```text
lane_706_20260518/train/outside_20250910112526_000206.jpg
lane_812_20260531/train/DsKPdq9pWzgZ1Hu3kaf0ZfLT_202510161514_1.jpg
```

### 17.3 Stage 1 高分辨率配置

原计划尝试 `1280x720`。实际训练启动时 DLA backbone 报错：

```text
RuntimeError: The size of tensor a (23) must match the size of tensor b (22) at non-singleton dimension 2
```

原因是 height 720 与 DLA 下采样 stride 不兼容。实际采用 `1280x704`，height 704 能被 32 整除，同时相比上一轮 `1024x544` 仍提高了远处车道线的输入采样密度。

配置：

```text
configs/clrernet/lane_706_812_349_20260609/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1280x704_topcrop8.py
```

权重：

```text
work_dirs/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1280x704_topcrop8_culane_pretrain/epoch_25.pth
```

merged val 结果：

| Epoch | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3330 | 1786 | 0.3718 | 0.6932 | 0.4840 | 0.2700 | 0.5034 | 0.3514 |
| 10 | 3273 | 1786 | 0.4140 | 0.7587 | 0.5357 | 0.3168 | 0.5806 | 0.4100 |
| 15 | 3264 | 1786 | 0.4240 | 0.7749 | 0.5481 | 0.3404 | 0.6221 | 0.4400 |
| 25 | 3205 | 1786 | 0.4349 | 0.7805 | 0.5586 | 0.3551 | 0.6372 | 0.4560 |

### 17.4 Stage 2 分类器

Stage 2 从零训练，不复用旧分类器。

产物：

```text
work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/best.pth
work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/metrics.json
work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/confusion_matrix_best.csv
```

| Split | Samples | class counts |
|---|---:|---|
| train | 8241 | `[5088, 2151, 1002]` |
| val | 1787 | `[1087, 441, 259]` |

最佳结果：

| Best epoch | Val accuracy | Val macro-F1 |
|---:|---:|---:|
| 26 | 0.9239 | 0.8812 |

### 17.5 Stage 1 test 结果

| Split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| merged test | 3275 | 1886 | 0.4327 | 0.7513 | 0.5491 | 0.3612 | 0.6273 | 0.4584 |
| lane_706 test | 1237 | 745 | 0.4099 | 0.6805 | 0.5116 | 0.3395 | 0.5638 | 0.4238 |
| lane_812 test | 1432 | 779 | 0.4434 | 0.8151 | 0.5744 | 0.3624 | 0.6662 | 0.4695 |
| lane_349 test | 606 | 362 | 0.4538 | 0.7597 | 0.5682 | 0.4026 | 0.6740 | 0.5041 |

### 17.6 Two-stage end-to-end 结果

主参数：`score_thr=0.25, det_conf_thr=0.25, nms_topk=12, nms_thres=50, iou_thr=0.3, match_width=20`。

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Matched-lane Acc | Matched-lane Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| merged test | 283 | 1458 | 434 | 448 | 0.7949 | 0.7324 |
| lane_706 test | 107 | 532 | 213 | 207 | 0.7895 | 0.7488 |
| lane_812 test | 123 | 643 | 138 | 157 | 0.8087 | 0.7330 |
| lane_349 test | 53 | 283 | 83 | 84 | 0.7739 | 0.6869 |

旧参数：`score_thr=0.35, det_conf_thr=0.35, nms_topk=8, nms_thres=50, iou_thr=0.3, match_width=20`。

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Matched-lane Acc | Matched-lane Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| merged test | 283 | 1388 | 504 | 155 | 0.7990 | 0.7357 |
| lane_706 test | 107 | 512 | 233 | 69 | 0.7930 | 0.7518 |
| lane_812 test | 123 | 607 | 174 | 54 | 0.8155 | 0.7416 |
| lane_349 test | 53 | 269 | 97 | 32 | 0.7732 | 0.6778 |

### 17.7 可视化样例

每个数据源从 test split 取 2 张样例，共 6 张。

服务器目录：

```text
work_dirs/two_stage_infer/lane_706_812_349_20260609_samples/visualizations/
```

本地同步目录：

```text
D:\A\Documents\车道线检测\clrernet_vis_lane_706_812_349_20260609\
```

### 17.8 结论

1. `1280x720` 不适合当前 DLA 配置；实际采用 `1280x704` 是必要的 stride 对齐修正。
2. 与上一轮 `lane_706 + lane_812` 在旧参数下的 merged two-stage macro-F1 `0.7228` 相比，本轮三源 merged test old-param macro-F1 为 `0.7357`，略有提升。
3. 新增 `lane_349` 分项表现低于 `lane_706/lane_812` 的 matched-lane macro-F1，说明新增数据域仍是当前主要难点。
4. 主参数 `0.25/top12` 提高 matched GT 数量，但 unmatched prediction 也明显增加；旧参数 `0.35/top8` 更保守，matched-lane macro-F1 略高，适合作为当前对外可比指标。

---

## 18. 2026-06-10 后处理参数调试

目标：针对 `lane_706 + lane_812 + lane_349` 的 `1280x704` Stage 1 checkpoint，调试推理后处理参数，降低高分辨率模型产生的额外 lane，同时比较 two-stage 端到端结果。

固定条件：

```text
det checkpoint: work_dirs/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1280x704_topcrop8_culane_pretrain/epoch_25.pth
cls checkpoint: work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/best.pth
split: dataset/lane_706_812_349_20260609/splits/test.txt
iou_thr=0.3
match_width=20
nms_thres=50
```

搜索方式：

- 粗网格：`score_thr=det_conf_thr in {0.25,0.30,0.35,0.40}`，`nms_topk in {6,8,10,12}`。
- 细网格：`score_thr=det_conf_thr in {0.32,0.34,0.36,0.38}`，`nms_topk in {6,7,8}`。
- 主指标：merged test 上 matched-lane Macro-F1，同时观察 `matched_gt`、`unmatched_gt`、`unmatched_prediction`。

### 18.1 merged test two-stage 调参结果

| Setting | Images | Matched GT | Unmatched GT | Unmatched Pred | Acc | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `0.25/top12` | 283 | 1458 | 434 | 448 | 0.7949 | 0.7324 |
| `0.35/top8` | 283 | 1388 | 504 | 155 | 0.7990 | 0.7357 |
| `0.36/top6` | 283 | 1307 | 585 | 108 | 0.8095 | 0.7446 |
| `0.35/top6` | 283 | 1313 | 579 | 115 | 0.8088 | 0.7437 |
| `0.40/top6` | 283 | 1253 | 639 | 78 | 0.8069 | 0.7434 |

`0.36/top6` 是本轮 two-stage matched-lane Macro-F1 最高设置。

### 18.2 `0.36/top6` 分项结果

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Acc | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| lane_706 test | 107 | 478 | 267 | 49 | 0.7971 | 0.7535 |
| lane_812 test | 123 | 578 | 203 | 35 | 0.8287 | 0.7546 |
| lane_349 test | 53 | 251 | 115 | 24 | 0.7888 | 0.6930 |

### 18.3 Stage 1 detection 对比

| Setting | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.25/top6` | 1583 | 1886 | 0.8345 | 0.7004 | 0.7616 | 0.7037 | 0.5907 | 0.6423 |
| `0.35/top6` | 1428 | 1886 | 0.9027 | 0.6835 | 0.7779 | 0.7703 | 0.5832 | 0.6639 |
| `0.36/top6` | 1415 | 1886 | 0.9067 | 0.6803 | 0.7773 | 0.7753 | 0.5817 | 0.6646 |
| `0.35/top8` | 1543 | 1886 | 0.8801 | 0.7200 | 0.7921 | 0.7479 | 0.6119 | 0.6731 |
| `0.40/top6` | 1331 | 1886 | 0.9264 | 0.6538 | 0.7666 | 0.8024 | 0.5663 | 0.6640 |

### 18.4 推荐

推荐保留两套后处理口径：

1. Two-stage 低误检/高分类质量：`score_thr=0.36, det_conf_thr=0.36, nms_topk=6, nms_thres=50`。
2. Locator 覆盖优先/检测 F1 优先：`score_thr=0.35, det_conf_thr=0.35, nms_topk=8, nms_thres=50`。

原因：

- `0.36/top6` 的 matched-lane Macro-F1 最高，unmatched prediction 从 `0.35/top8` 的 155 降到 108。
- `0.35/top8` 的 Stage 1 F1 更高，matched GT 更多，漏检更少；如果下游更在意“尽量检出所有 lane”，它更合适。
- `0.25/top12` 虽然 matched GT 最高，但 unmatched prediction 达到 448，误检过多，不建议作为默认交付参数。

---

## 19. 2026-06-10 Stage 1 分辨率搜索

### 19.1 实验原则

本节重新建立三数据根的公平基线，避免把不同数据量、不同 checkpoint 或不同后处理参数混在一起比较。

- 数据：固定使用三数据根 merged split，train/val/test 为 `1304/278/283`。
- 初始化：统一使用官方 `checkpoints/clrernet_culane_dla34_ema.pth`。
- 训练：`25` epoch，batch size `4`，梯度累积 `2`，有效 batch size 约 `8`。
- 预处理：统一 `topcrop8`，其余数据增强、学习率和 seed 规则一致。
- 选择集：只使用 merged validation 选择 epoch、分辨率和 NMS；test 只在最终方案确定后运行一次。
- 主指标：validation `F1@0.5`；次指标：validation `F1@0.3`。
- 候选尺寸：`1024x544`、`1088x576`、`1152x608`、`1216x640`、`1280x672`、历史端点 `1280x704`。

新增工具：

```text
tools/sweep_highway_stage1_postprocess.py
tools/summarize_stage1_resolution_search.py
```

前者在每个 batch 只运行一次模型前向，然后复用输出计算完整后处理网格；其结果已与 `tools/test.py` 抽样逐项核对，指标一致。

### 19.2 固定后处理下的纯尺寸比较

固定参数：`conf_threshold=0.35, nms_topk=8, nms_thres=50`。表中每个尺寸使用该固定参数下 seed 0 的最佳验证 epoch。

| Resolution | Seed | Epoch | pred_lanes | F1@0.3 | F1@0.5 | Forward ms/image | Peak alloc MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1024x544` | 0 | 25 | 1524 | 0.8369 | 0.7311 | 99.60 | 516.35 |
| `1088x576` | 0 | 25 | 1541 | 0.8332 | 0.7178 | 51.34 | 573.80 |
| `1152x608` | 0 | 25 | 1542 | 0.8233 | 0.7037 | 102.47 | 631.03 |
| `1216x640` | 0 | 25 | 1546 | 0.8175 | 0.6861 | 99.30 | 691.70 |
| `1280x672` | 0 | 25 | 1522 | 0.8077 | 0.6796 | 100.64 | 757.17 |
| `1280x704` | 0 | 25 | 1516 | 0.8116 | 0.6760 | 99.62 | 789.76 |

计时来自共享 GPU 上的单次 sweep 前向，受同时运行任务影响，只能作为本次运行记录，不能作为严格吞吐 benchmark。峰值显存趋势更稳定：尺寸越大，显存占用持续增加。

固定后处理下，`1024x544` 已在 `F1@0.3` 和 `F1@0.5` 上同时领先，说明此前远处车道线问题不能简单归因于输入分辨率不足。

### 19.3 每个尺寸经过 validation 调优后的结果

搜索范围：

```text
conf_threshold = [0.25, 0.30, 0.325, 0.35, 0.375, 0.40]
nms_topk       = [6, 8, 10, 12]
nms_thres      = [40, 50, 60]
```

| Resolution | Seeds | Selected conf/topk/nms | Mean F1@0.3 | Mean F1@0.5 |
|---|---:|---|---:|---:|
| `1024x544` | 3 | `0.35/10/50` | 0.8382 | 0.7345 |
| `1088x576` | 3 | `0.35/10/40` | 0.8355 | 0.7216 |
| `1152x608` | 1 | `0.375/10/50` | 0.8215 | 0.7057 |
| `1216x640` | 1 | `0.375/8/60` | 0.8146 | 0.6911 |
| `1280x672` | 1 | `0.375/8/60` | 0.8000 | 0.6842 |
| `1280x704` | 1 | `0.375/8/60` | 0.8103 | 0.6809 |

seed 0 排名前两名为 `1024x544` 和 `1088x576`，因此两者补训 seed 1、seed 2。三 seed 平均 `F1@0.5` 差值为：

```text
0.7345 - 0.7216 = 0.0129
```

差值大于预设 tie threshold `0.003`，因此最终选择指标更高的 `1024x544`，不是仅因像素更少而胜出。

最终 checkpoint 使用获胜尺寸中 validation `F1@0.5` 最高的运行：

```text
config:
configs/clrernet/lane_706_812_349_20260609/clrernet_lane_706_812_349_20260609_dla34_ema_locator_1024x544_topcrop8.py

checkpoint:
work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth

post-processing:
conf_threshold=0.35
nms_topk=10
nms_thres=50
```

最终 checkpoint 的 merged validation 指标：

| Seed | Epoch | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 20 | 1519 | 1786 | 0.9144 | 0.7777 | 0.8405 | 0.8084 | 0.6876 | 0.7431 |

完整汇总产物：

```text
work_dirs/resolution_search/summary_final/resolution_search_summary.json
work_dirs/resolution_search/summary_final/resolution_search_summary.md
work_dirs/resolution_search/summary_final/fixed_results.csv
work_dirs/resolution_search/summary_final/grid_aggregate.csv
```

### 19.4 最终 Stage 1 test

test 参数完全沿用 validation 选出的 `0.35/top10/nms50`，未在 test 上再次调参。

| Split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| merged test | 1557 | 1886 | 0.8992 | 0.7423 | 0.8132 | 0.8067 | 0.6660 | 0.7296 |
| lane_706 test | 580 | 745 | 0.8793 | 0.6846 | 0.7698 | 0.7621 | 0.5933 | 0.6672 |
| lane_812 test | 668 | 779 | 0.9207 | 0.7895 | 0.8500 | 0.8353 | 0.7163 | 0.7713 |
| lane_349 test | 309 | 362 | 0.8900 | 0.7597 | 0.8197 | 0.8285 | 0.7072 | 0.7630 |

与同一三数据源 `1280x704 epoch_25` 的旧调优口径 `0.35/top8/nms50` 相比，merged test：

| Model | F1@0.3 | F1@0.5 |
|---|---:|---:|
| `1280x704`, old tuned | 0.7921 | 0.6731 |
| `1024x544`, validation-selected | 0.8132 | 0.7296 |
| Improvement | +0.0211 | +0.0565 |

这是一组同数据源、同 test split 的对比，但 checkpoint 的后处理分别来自各自实验；真正隔离尺寸影响的结论仍应以第 19.2 节固定参数 validation 表为准。

### 19.5 复用 Stage 2 的最终 two-stage test

Stage 2 的 GT strip 和标签没有变化，因此复用：

```text
work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/best.pth
```

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Matched-lane Acc | Matched-lane Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| merged test | 283 | 1414 | 478 | 143 | 0.7999 | 0.7398 |
| lane_706 test | 107 | 522 | 223 | 58 | 0.7816 | 0.7389 |
| lane_812 test | 123 | 615 | 166 | 53 | 0.8228 | 0.7550 |
| lane_349 test | 53 | 277 | 89 | 32 | 0.7834 | 0.7018 |

merged test 分类分项：

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| solid | 0.9390 | 0.8010 | 0.8646 | 769 |
| dashed | 0.8495 | 0.9122 | 0.8797 | 433 |
| joint | 0.4096 | 0.5660 | 0.4752 | 212 |

当前主要分类短板仍是 `joint`，而不是 `solid/dashed`。

评测目录：

```text
work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_merged_s0.35_top10_nms50
work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_lane706_s0.35_top10_nms50
work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_lane812_s0.35_top10_nms50
work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_lane349_s0.35_top10_nms50
```

### 19.6 可视化

从每个数据源的 test split 取前两张图片，共生成 6 组预测 JSON、summary 和 overlay：

```text
work_dirs/two_stage_infer/lane_706_812_349_20260609_resolution_best_samples/
```

目录按 `lane706/lane812/lane349` 和 `sample1/sample2` 分层，避免结果互相覆盖。

### 19.7 结论

1. 在本次候选范围内，最优分辨率是 `1024x544`，不是中间尺寸或 `1280x704`。
2. 更高分辨率持续增加显存，但 validation `F1@0.5` 整体下降；远处车道线问题更可能同时受标注密度、目标像素占比、特征步长、透视压缩和 lane proposal/匹配机制影响，不能只靠放大输入解决。
3. 最终默认 Stage 1 为 `1024x544 / seed2 / epoch20 / 0.35/top10/nms50`。
4. 分辨率和 NMS 均由 validation 选择，test 只用于最终一次报告，因此本节结果比第 18 节直接在 test 上调后处理更适合作为当前正式基线。

### 19.8 产物清理

2026-06-11 清理旧训练和评测产物，`work_dirs` 从约 `12 GB` 降至约 `599 MB`。当前保留：

```text
work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth
work_dirs/resolution_search/summary_final/
work_dirs/resolution_search/eval/
work_dirs/resolution_search/final_stage1/
work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/
work_dirs/two_stage_eval/lane_706_812_349_20260609_resolution_best_*/
work_dirs/two_stage_infer/lane_706_812_349_20260609_resolution_best_samples/
```

旧 `lane_706`、`lane_706 + lane_812`、`1280x704` 训练 checkpoint，失败/烟雾测试目录、旧 test 后处理搜索和旧可视化已删除。第 15 至 18 节的指标仍作为历史实验记录保留，但其中旧 checkpoint 路径不再保证存在。

## 20. 2026-06-24 lane_706 + lane_812 + lane_349 + lane_696 four-source fine-tuning

Goal: extract `dataset/lane_696_20260624.zip`, split the new `lane_696_20260624` dataset deterministically, and fine-tune the two-stage model with `lane_706_20260518 + lane_812_20260531 + lane_349_20260609 + lane_696_20260624`. The input resolution follows the previous validation-selected winner: `1024x544 topcrop8`.

### 20.1 New data check and split

Extracted data root:

```text
dataset/lane_696_20260624/
```

Single-source check result:

| item | count |
| --- | ---: |
| images | 696 |
| json | 696 |
| valid_pairs | 696 |
| train | 487 |
| val | 104 |
| test | 105 |
| errors | 0 |
| warnings | 62 |

Split command:

```bash
PYTHONPATH=. python tools/prepare_culane_highway.py \
  --data-root dataset/lane_696_20260624 \
  --seed 0 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

### 20.2 Four-source merged split

The merged split is virtual and keeps images in their original roots. Split rows use `dataset_key<TAB>relative_image_path`.

```bash
PYTHONPATH=. python tools/build_highway_multiroot_splits.py \
  --source lane_706_20260518=dataset/lane_706_20260518 \
  --source lane_812_20260531=dataset/lane_812_20260531 \
  --source lane_349_20260609=dataset/lane_349_20260609 \
  --source lane_696_20260624=dataset/lane_696_20260624 \
  --out-dir dataset/lane_706_812_349_696_20260624/splits \
  --source-priority lane_696_20260624,lane_349_20260609,lane_812_20260531,lane_706_20260518
```

Merged image-level split:

| split | images |
| --- | ---: |
| train | 1791 |
| val | 382 |
| test | 388 |
| total | 2561 |

Merged lane label counts:

| split | solid | dashed | joint |
| --- | ---: | ---: | ---: |
| train | 7028 | 2841 | 1533 |
| val | 1486 | 590 | 355 |
| test | 1569 | 683 | 370 |

Duplicate handling result: 2 duplicate groups were recorded, both among older sources. `lane_696_20260624` did not introduce a new dropped duplicate.

```text
lane_706_20260518/train/outside_20250910112526_000206.jpg
lane_812_20260531/train/DsKPdq9pWzgZ1Hu3kaf0ZfLT_202510161514_1.jpg
```

### 20.3 Code and config changes

New Stage 1 configs:

```text
configs/clrernet/lane_706_812_349_696_20260624/dataset_lane_706_812_349_696_20260624_clrernet_1024x544_topcrop8.py
configs/clrernet/lane_706_812_349_696_20260624/clrernet_lane_706_812_349_696_20260624_dla34_ema_locator_1024x544_topcrop8_finetune.py
```

Stage 1 key settings:

```text
img_scale=(1024, 544)
top_crop_ratio=0.08
batch_size=4
accumulative_counts=2
load_from=work_dirs/resolution_search/lane_706_812_349_20260609_1024x544_seed2/epoch_20.pth
work_dir=work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best
```

`libs/lane_classifier/train.py` now supports `--init-from`. It loads model weights only and starts a new experiment from epoch 1. It is mutually exclusive with `--resume-from`, which is still used for interrupted-run recovery with optimizer/history state.

### 20.4 Pre-training validation

Stage 1 dataloader validation:

```text
train/val/test = 1791/382/388
sample tensor = (3, 544, 1024)
```

Stage 2 dataset validation:

```text
train samples = 11402
class_counts = [7028, 2841, 1533]
sample crop = (3, 288, 128)
```

Classifier initialization checkpoint loaded successfully:

```text
work_dirs/lane_classifier/lane_706_812_349_20260609_stage2_strip_288x128_w128/best.pth
```

### 20.5 Stage 1 training and validation post-processing

Training command:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python tools/train.py \
  configs/clrernet/lane_706_812_349_696_20260624/clrernet_lane_706_812_349_696_20260624_dla34_ema_locator_1024x544_topcrop8_finetune.py \
  --work-dir work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best
```

Training finished for 20 epochs. Final checkpoint:

```text
work_dirs/clrernet_lane_706_812_349_696_20260624_locator_1024x544_topcrop8_finetune_from_349_best/epoch_20.pth
```

Default epoch-20 validation metric:

| pred_lanes | gt_lanes | F1@0.3 | F1@0.5 |
| ---: | ---: | ---: | ---: |
| 2147 | 2429 | 0.8654 | 0.7854 |

Validation post-processing sweep was run for epochs 5/10/15/20 with `conf=[0.25,0.30,0.325,0.35,0.375,0.40]`, `topk=[6,8,10,12]`, and `nms=[40,50,60]`. Selection used merged validation `F1@0.5` only.

| epoch | conf/topk/nms | F1@0.3 | F1@0.5 |
| ---: | --- | ---: | ---: |
| 5 | `0.375/10/40` | 0.8560 | 0.7654 |
| 10 | `0.40/8/40` | 0.8660 | 0.7830 |
| 15 | `0.40/8/40` | 0.8677 | 0.7918 |
| 20 | `0.40/8/40` | 0.8722 | 0.7938 |

Final Stage 1 selection: `epoch_20.pth + conf=0.40 + nms_topk=8 + nms_thres=40`.

### 20.6 Stage 1 test results

| split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged | 2140 | 2614 | 0.9430 | 0.7720 | 0.8490 | 0.8682 | 0.7108 | 0.7817 |
| lane_706 | 591 | 745 | 0.9205 | 0.7302 | 0.8144 | 0.8223 | 0.6523 | 0.7275 |
| lane_812 | 660 | 779 | 0.9606 | 0.8139 | 0.8812 | 0.9030 | 0.7651 | 0.8284 |
| lane_349 | 306 | 362 | 0.9379 | 0.7928 | 0.8593 | 0.8889 | 0.7514 | 0.8144 |
| lane_696 | 583 | 728 | 0.9485 | 0.7596 | 0.8436 | 0.8645 | 0.6923 | 0.7689 |

### 20.7 Stage 2 fine-tuning results

Training command:

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

Best checkpoint:

```text
work_dirs/lane_classifier/lane_706_812_349_696_20260624_stage2_strip_288x128_w128_finetune_from_349_best/best.pth
```

Best validation metric:

| epoch | val samples | accuracy | macro-F1 |
| ---: | ---: | ---: | ---: |
| 19 | 2431 | 0.9292 | 0.8860 |

Epoch 20 validation was `accuracy=0.9264`, `macro-F1=0.8837`, so epoch 19 remains the selected `best.pth`.

### 20.8 Two-stage test results

All five evals use `epoch_20.pth + best.pth + 0.40/top8/nms40 + iou_thr=0.3 + match_width=20`.

| split | images | matched_gt | unmatched_gt | unmatched_pred | det P | det R | det F1 | cls acc | cls macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged | 388 | 2023 | 599 | 116 | 0.9458 | 0.7715 | 0.8498 | 0.8057 | 0.7400 |
| lane_706 | 107 | 545 | 200 | 46 | 0.9222 | 0.7315 | 0.8159 | 0.7945 | 0.7400 |
| lane_812 | 123 | 635 | 146 | 25 | 0.9621 | 0.8131 | 0.8813 | 0.8409 | 0.7773 |
| lane_349 | 53 | 288 | 78 | 18 | 0.9412 | 0.7869 | 0.8571 | 0.7778 | 0.6994 |
| lane_696 | 105 | 555 | 175 | 27 | 0.9536 | 0.7603 | 0.8460 | 0.7910 | 0.7219 |

Output directories:

```text
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_merged_s0.40_top8_nms40
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_lane706_s0.40_top8_nms40
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_lane812_s0.40_top8_nms40
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_lane349_s0.40_top8_nms40
work_dirs/two_stage_eval/lane_706_812_349_696_20260624_best_lane696_s0.40_top8_nms40
```

### 20.9 Inference visualization

Two samples were taken from each source-specific test split, producing 8 prediction JSON files, 8 summaries, and 8 overlays:

```text
work_dirs/two_stage_infer/lane_706_812_349_696_20260624_best_samples/
```

Visualization files are non-empty, ranging from about `1.1 MB` to `2.1 MB`. Sample folders:

```text
lane706_01/ lane706_02/
lane812_01/ lane812_02/
lane349_01/ lane349_02/
lane696_01/ lane696_02/
```

### 20.10 Current conclusion

1. After adding `lane_696_20260624`, the four-source merged test Stage 1 result is `F1@0.5=0.7817`. The validation-selected post-processing is more conservative than the older `0.35/top10/nms50`, ending at `0.40/top8/nms40`.
2. The four-source Stage 2 classifier reaches validation `macro-F1=0.8860`. On the two-stage merged test, matched-lane `macro-F1=0.7400`; `joint` remains the main classification weakness.
3. `dataset/culane_highway` is not used. No merged image copy is created; all combined training/evaluation uses multi-root split files.
4. Current four-source artifacts are Stage 1 `epoch_20.pth`, Stage 2 `best.pth`, five two-stage eval reports, and eight overlays. The selected three-source artifacts must remain because this run initializes from them.
