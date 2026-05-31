# Highway 两阶段车道线检测项目记录（含历史实验）

> 当前状态更新（2026-05-31）：当前主流程使用 `dataset/lane_706_20260518/` 与 `dataset/lane_812_20260531/` 的多数据根合并训练；最新可复现命令与结果以根目录 `README.md` 及本文第 16 节为准。`dataset/culane_highway/` 仅保留为历史资料，不参与当前训练或评测。

本文档记录在 `/datadisk2/longhaoxiang/CLRerNet/` 内完成的两阶段实验过程。第 1 至 14 节保留早期以 `dataset/culane_highway/` 为数据源的历史实验，第 15 节记录上一轮 `lane_706_20260518` 单数据源实验；当前默认流程以第 16 节 `lane_706 + lane_812` 多数据根重训为准。

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
