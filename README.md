# CLRerNet Highway 两阶段车道线检测

本仓库基于 CLRerNet 扩展了高速公路场景的两阶段车道线检测与类型分类流程：

- **Stage 1：class-agnostic lane locator**，只检测车道线实例的位置，不区分线型。
- **Stage 2：lane instance classifier**，对 Stage 1 输出的每条车道线分类为 `solid`、`dashed`、`joint`。

> 当前有效实验状态（核验日期：2026-05-26）：本轮训练、验证、测试只使用 `dataset/lane_706_20260518`。`dataset/culane_highway` 仅保留为历史标注资料，不参加下述命令中的 split、训练或评测。

## 最新已验证结果

### Stage 1：车道线定位器

配置：`configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py`
权重：`work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth`

| Split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 833 | 702 | 0.6267 | 0.7436 | 0.6801 | 0.4946 | 0.5869 | 0.5368 |
| test | 851 | 745 | 0.5664 | 0.6470 | 0.6040 | 0.4653 | 0.5315 | 0.4962 |

### Stage 2：使用 GT lane instance 的分类器验证结果

权重：`work_dirs/lane_classifier/lane_706_20260518_stage2_strip_288x128_w128/best.pth`
最佳 epoch：`22`

| Split | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| val | 702 | 0.9088 | 0.8575 |

混淆矩阵（行是真值，列是预测；顺序为 `solid`, `dashed`, `joint`）：

```text
[[379,   9, 14],
 [  3, 197, 12],
 [ 16,  10, 62]]
```

### 两阶段端到端 test 结果

评测目录：`work_dirs/two_stage_eval/lane_706_20260518_stage2_test_1024x544_topcrop8_epoch25_s0.35_top8_nms50`
参数：`score_thr=0.35`、`det_conf_thr=0.35`、`nms_topk=8`、`nms_thres=50`、`IoU match threshold=0.3`。

| Test images | Matched GT lanes | Unmatched GT lanes | Unmatched predictions | Matched-lane Accuracy | Matched-lane Macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 107 | 469 | 276 | 70 | 0.7228 | 0.6751 |

此处分类 Accuracy/Macro-F1 只统计已与 GT 匹配的预测 lane，不等价于定位器的检测 F1。该目录中已保存 107 张 `test_pred_vis` 和 107 张 `test_gt_pred_vis` 可视化图片。

## 1. 项目环境与安装

### 1.1 当前服务器环境

工作目录：

```bash
cd /datadisk2/longhaoxiang/CLRerNet
source /home/longhaoxiang/anaconda3/etc/profile.d/conda.sh
conda activate clrernet
```

在 `clrernet` 环境中于 2026-05-26 实际核验的版本如下：

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
| Albumentations | 0.4.6 |
| scikit-learn | 1.8.0 |

所有本文命令均从仓库根目录执行，并显式设置 `PYTHONPATH=.`。

### 1.2 重建环境参考

本项目依赖 MMDetection 3.x 与 CUDA NMS 扩展。对于新机器，应按 GPU/CUDA 环境先安装匹配版本的 PyTorch，然后安装下述依赖；当前服务器已存在可工作的 `clrernet` 环境，不需要重复执行。

```bash
conda create -n clrernet python=3.11 -y
conda activate clrernet

# 安装与目标 CUDA 匹配的 PyTorch / TorchVision 后：
pip install -r requirements.txt
pip install mmengine==0.10.5 mmcv==2.1.0 mmdet==3.3.0 opencv-python==4.9.0.80 scikit-learn==1.8.0

cd libs/models/layers/nms
python setup.py install
cd ../../../..
```

预训练初始化权重应放置为：

```text
checkpoints/clrernet_culane_dla34_ema.pth
```

可使用以下命令检查关键依赖是否可导入：

```bash
python -c "import torch, cv2, mmengine, mmcv, mmdet; print(torch.__version__, torch.cuda.is_available())"
```

原始 CLRerNet 的 Docker 安装提示仍保存在 `docs/INSTALL.md`，但该文档不定义本项目的当前数据源和训练配置。

## 2. 整体算法架构

### 2.1 Stage 1：Class-Agnostic Lane Locator

输入是一张道路图像及其 LabelMe 标注。`HighwayLaneDataset` 读取 `solid/dashed/joint` 三种 lane label，但训练时将每条合法车道线统一映射为一个 `lane` 检测类别，原始类型仅保留在元信息中。

当前定位器流程如下：

1. 按图像高度裁去顶部 `8%` 区域，并同步变换 polyline 坐标。
2. 将图像缩放到 `1024 x 544`，过滤越界或不足两个点的车道线。
3. 使用 DLA34 backbone 的 CLRerNet 学习车道线位置，使用官方 EMA 权重 warm start。
4. 推理阶段使用 NMS，当前端到端评测采用 `conf=0.35`、`nms_thres=50`、`topk=8`。

关键训练设置：

| Setting | Value |
| --- | --- |
| Input resolution | `1024 x 544` |
| Top crop | `0.08` |
| Batch size | `8` |
| Epochs | `25` |
| Optimizer | `AdamW(lr=5e-5, weight_decay=0.01)` |
| Warm start | `checkpoints/clrernet_culane_dla34_ema.pth` |
| Validation interval | every `5` epochs |

### 2.2 Stage 2：Lane Instance Type Classifier

Stage 2 不训练新的车道线定位，而是对单条 lane 进行线型判断：

1. 训练时从 GT polyline 沿车道线法线方向重采样局部条带图像。
2. 输入 crop 尺寸为 `288 x 128`，条带宽度 `128`。
3. 轻量 CNN 由多层 `Conv-BN-ReLU` 与 pooling 组成，最终对 `solid/dashed/joint` 输出三分类概率。
4. 推理时先由 Stage 1 给出 lane polyline，再裁取对应条带供 Stage 2 分类。

当前训练设置：

| Setting | Value |
| --- | --- |
| Crop size | `288 x 128` |
| Strip width | `128` |
| Batch size | `64` |
| Epochs | `30` |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=1e-4` |
| Dropout | `0.25` |
| Class balance | weighted loss (`loss`) |
| Debug crops | `0` |

## 3. 项目目录结构与关键文件作用

```text
CLRerNet/
├── README.md
├── checkpoints/                              # 预训练初始化权重（不纳入 Git）
├── configs/clrernet/
│   ├── base_clrernet.py                      # CLRerNet 基础模型设置
│   └── lane_706_20260518/
│       ├── dataset_lane_706_20260518_clrernet_1024x544_topcrop8.py
│       └── clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py
├── dataset/                                  # 本地数据（不纳入 Git）
│   └── lane_706_20260518/
│       └── splits/
├── libs/
│   ├── datasets/
│   │   ├── highway_lane_dataset.py
│   │   ├── metrics/highway_lane_metric.py
│   │   └── pipelines/
│   │       ├── top_crop.py
│   │       ├── resize_pad.py
│   │       └── lane_formatting.py
│   └── lane_classifier/
│       ├── crop.py
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       ├── infer.py
│       └── eval.py
├── tools/
│   ├── prepare_culane_highway.py
│   ├── visualize_culane_highway_dataset.py
│   ├── train.py
│   └── test.py
├── docs/
│   └── culane_highway_two_stage_report.md    # 实验过程与最新追加记录
└── work_dirs/                                # 训练/评测产物（不纳入 Git）
```

关键文件说明：

| Path | 作用 |
| --- | --- |
| `configs/clrernet/lane_706_20260518/dataset_lane_706_20260518_clrernet_1024x544_topcrop8.py` | 指定当前唯一数据源、split、增强、`1024x544 topcrop8` dataloader 和定位评测器。 |
| `configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py` | Stage 1 主配置，定义 warm start、训练轮数、优化器、NMS 与输出目录。 |
| `libs/datasets/highway_lane_dataset.py` | 读取扁平图片/JSON 标注，将三种线型统一作为 Stage 1 的一个 lane 类。 |
| `libs/datasets/pipelines/top_crop.py` | 上裁剪和点坐标同步变换，同时清理无效 lane。 |
| `libs/datasets/metrics/highway_lane_metric.py` | Stage 1 类别无关的 lane IoU 匹配与 P/R/F1 评测。 |
| `libs/lane_classifier/crop.py` | lane 条带裁剪、标签映射、绘图与匹配 IoU 工具。 |
| `libs/lane_classifier/dataset.py` | 使用 split 与 GT lane 构造 Stage 2 实例级训练样本。 |
| `libs/lane_classifier/model.py` | 轻量三分类 CNN 和 checkpoint 加载逻辑。 |
| `libs/lane_classifier/train.py` | Stage 2 训练与最佳 checkpoint/混淆矩阵/metrics 保存。 |
| `libs/lane_classifier/infer.py` | Stage 1 + Stage 2 联合推理，保存预测 JSON 与渲染图。 |
| `libs/lane_classifier/eval.py` | 两阶段结果和 GT 匹配后的实例分类评测与错误可视化。 |
| `tools/prepare_culane_highway.py` | 校验标注并按固定 seed 生成 train/val/test split。 |
| `tools/visualize_culane_highway_dataset.py` | 检查配置 pipeline 变换后的 GT 车道线可视化。 |

## 4. 数据集结构

### 4.1 当前唯一有效数据源

```text
dataset/lane_706_20260518/
├── <image_stem>.jpg / .png / .JPG / .PNG
├── <image_stem>.json
└── splits/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── prepare_report.json
```

图片与 JSON 采用相同 stem 一一对应。JSON 使用 LabelMe `shapes` schema，每个合法 lane 至少需要两个点，label 只允许：

```text
solid
dashed
joint
```

`prepare_report.json` 中已核验的当前数据统计如下：

| Item | Count |
| --- | ---: |
| Images | 706 |
| JSON annotations | 706 |
| Valid image/JSON pairs | 706 |
| Train images | 494 |
| Val images | 105 |
| Test images | 107 |
| `solid` instances | 2703 |
| `dashed` instances | 1378 |
| `joint` instances | 525 |
| Errors | 0 |
| Warnings | 33 |

split 为 image-level 随机划分，比例 `70/15/15`，seed 为 `0`。33 条 warning 表示部分原始 polyline 点序不严格按下到上排列；加载器会排序/过滤有效点，当前训练与评测均基于此实现完成。

`dataset/culane_highway` 与当前目录存在重复图片且标注版本不同，因此它不再合入训练集，也不作为当前指标的评测来源。

## 5. 训练、评测与推理命令

以下命令均从仓库根目录运行：

```bash
cd /datadisk2/longhaoxiang/CLRerNet
source /home/longhaoxiang/anaconda3/etc/profile.d/conda.sh
conda activate clrernet
```

### 5.1 校验数据并生成 split

重新生成 split 会覆盖 `dataset/lane_706_20260518/splits/` 下的文本列表和报告，使用相同 seed 会得到可复现划分。

```bash
PYTHONPATH=. python tools/prepare_culane_highway.py \
  --data-root dataset/lane_706_20260518 \
  --seed 0 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

检查 pipeline 处理后的 GT 可视化：

```bash
PYTHONPATH=. python tools/visualize_culane_highway_dataset.py \
  configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py \
  --split test \
  --num-samples 16 \
  --out-dir work_dirs/lane_706_20260518_debug/test_gt
```

### 5.2 Stage 1 训练与测试

训练 class-agnostic 定位器：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py \
  configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py \
  --work-dir work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain
```

使用当前保留的 checkpoint 在 test split 上评测定位：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py \
  configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py \
  work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth
```

### 5.3 Stage 2 训练

Stage 2 从同一 split 的 GT lane instance 训练，不加载旧分类器权重：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.train \
  --data-root dataset/lane_706_20260518 \
  --split-root dataset/lane_706_20260518/splits \
  --work-dir work_dirs/lane_classifier/lane_706_20260518_stage2_strip_288x128_w128 \
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

主要输出包括：

```text
work_dirs/lane_classifier/lane_706_20260518_stage2_strip_288x128_w128/
├── best.pth
├── metrics.json
├── confusion_matrix_best.csv
└── val_predictions_best.json
```

### 5.4 两阶段端到端 test 评测

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.eval \
  --data-root dataset/lane_706_20260518 \
  --split-file dataset/lane_706_20260518/splits/test.txt \
  --det-config configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_20260518_stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_eval/lane_706_20260518_stage2_test_1024x544_topcrop8_epoch25_s0.35_top8_nms50 \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-thres 50 \
  --nms-topk 8 \
  --iou-thr 0.3 \
  --match-width 20 \
  --device cuda:0
```

主要输出：

```text
work_dirs/two_stage_eval/lane_706_20260518_stage2_test_1024x544_topcrop8_epoch25_s0.35_top8_nms50/
├── predictions/          # 每张图片的预测 JSON
├── error_vis/            # 错误样例可视化
├── eval_report.json      # 端到端分类指标与逐图明细
└── confusion_matrix.csv
```

### 5.5 图片或目录推理

对一个图片目录运行两阶段推理并保存渲染结果：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input dataset/lane_706_20260518 \
  --det-config configs/clrernet/lane_706_20260518/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_20260518_stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/lane_706_20260518 \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-thres 50 \
  --nms-topk 8 \
  --device cuda:0
```

推理输出目录包含 `predictions/`、`visualizations/` 和 `summary.json`；加入 `--save-crops` 可额外保存送入分类器的 lane strip crop。

## 6. 当前保留的有效产物

为避免历史试验混淆，当前只保留本轮需要复现或查看的关键结果目录：

```text
work_dirs/
├── clrernet_lane_706_20260518_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/
│   └── epoch_25.pth
├── lane_classifier/lane_706_20260518_stage2_strip_288x128_w128/
│   ├── best.pth
│   └── metrics.json
└── two_stage_eval/lane_706_20260518_stage2_test_1024x544_topcrop8_epoch25_s0.35_top8_nms50/
    ├── eval_report.json
    ├── test_pred_vis/
    └── test_gt_pred_vis/
```

`dataset/`、`checkpoints/` 与 `work_dirs/` 均是本机数据或实验产物，不应加入代码版本控制。

## 7. 上游项目来源

本项目以 [CLRerNet: Improving Confidence of Lane Detection with LaneIoU (WACV 2024)](https://github.com/hirotomusiker/CLRerNet) 官方实现为基础。上游提供 CULane 基准实现与预训练权重；本仓库的高速公路 flat image/JSON 数据加载、top crop、Stage 2 类型分类及端到端评测属于当前项目扩展。