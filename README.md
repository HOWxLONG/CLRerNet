# CLRerNet Highway 两阶段车道线检测

作者：Haoxiang Long

Email：haoxianglong@std.uestc.edu.com

本仓库基于 CLRerNet 实现高速公路场景两阶段车道线检测：

- Stage 1：class-agnostic lane locator，只学习哪里有车道线，不区分线型。
- Stage 2：lane instance classifier，对每条 lane instance 分类为 `solid`、`dashed`、`joint`。

当前有效状态（核验日期：2026-05-31）：本轮训练、验证、测试使用 `dataset/lane_706_20260518` 与 `dataset/lane_812_20260531`。`dataset/culane_highway` 只作为历史数据保留，不参与当前 split、训练或评测。

## 最新已验证结果

### 数据与 split

| 数据源 | 图片/JSON | Split | 说明 |
| --- | ---: | --- | --- |
| `lane_706_20260518` | 706/706 | 沿用既有 split：493/105/107 | 与历史结果可比；合并时去掉 1 张重复训练图 |
| `lane_812_20260531` | 812/812 | seed 0：568/121/123 | LabelMe `shapes` schema，标签合法 |
| merged | 1517 image-level samples | 1061/226/230 | 不复制图片，split 行格式为 `dataset_key<TAB>relative_image_path` |

重复图片：`outside_20250910112526_000206.jpg` 在两批数据中图片相同但 JSON 不同；合并 split 按规则保留 `lane_812_20260531` 标注，丢弃 `lane_706_20260518` 中该训练样本。

合并训练集 lane 标签计数：train `solid=4091,dashed=1800,joint=763`，val `solid=862,dashed=396,joint=193`，test `solid=899,dashed=429,joint=198`。

### Stage 1：车道线定位器

配置：

```text
configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py
```

权重：

```text
work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth
```

| Split | pred_lanes | gt_lanes | P@0.3 | R@0.3 | F1@0.3 | P@0.5 | R@0.5 | F1@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 1824 | 1524 | 0.6272 | 0.7507 | 0.6834 | 0.5510 | 0.6594 | 0.6004 |
| lane_706 test | 848 | 745 | 0.6120 | 0.6966 | 0.6516 | 0.5271 | 0.6000 | 0.5612 |
| lane_812 test | 976 | 779 | 0.6404 | 0.8023 | 0.7123 | 0.5717 | 0.7163 | 0.6359 |

说明：Stage 1 是类别无关检测，上表不评估 `solid/dashed/joint` 分类。

### Stage 2：GT lane instance 分类器

权重：

```text
work_dirs/lane_classifier/lane_706_812_20260531_stage2_strip_288x128_w128/best.pth
```

最佳 epoch：`25`

| Split | Samples | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| val | 1451 | 0.9235 | 0.8769 |

Stage 2 训练样本：train `6654`，val `1451`；类别顺序为 `solid,dashed,joint`。

### 两阶段端到端结果

参数：`score_thr=0.35`、`det_conf_thr=0.35`、`nms_topk=8`、`nms_thres=50`、`IoU=0.3`、`match_width=20`。

| Split | Images | Matched GT | Unmatched GT | Unmatched Pred | Matched-lane Acc | Matched-lane Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged test | 230 | 1111 | 415 | 128 | 0.7732 | 0.7228 |
| lane_706 test | 107 | 507 | 238 | 63 | 0.7515 | 0.7134 |
| lane_812 test | 123 | 604 | 177 | 65 | 0.7914 | 0.7291 |

两阶段分类指标只统计已与 GT 匹配的预测 lane，不等价于 Stage 1 检测 F1。

输出目录：

```text
work_dirs/two_stage_eval/lane_706_812_20260531_test_s0.35_top8_nms50
work_dirs/two_stage_eval/lane_706_812_20260531_test_lane_706_s0.35_top8_nms50
work_dirs/two_stage_eval/lane_706_812_20260531_test_lane_812_s0.35_top8_nms50
```

## 1. 项目环境与安装

### 1.1 当前服务器环境

```bash
ssh 190server
cd /datadisk2/longhaoxiang/CLRerNet
source /home/longhaoxiang/anaconda3/etc/profile.d/conda.sh
conda activate clrernet
```

2026-05-31 实际核验版本：

| Package | Version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.1.0+cu121 |
| TorchVision | 0.16.0+cu121 |
| CUDA runtime reported by PyTorch | 12.1 |
| cuDNN reported by PyTorch | 8902 |
| OpenCV | 4.9.0 |
| MMEngine | 0.10.5 |
| MMCV | 2.1.0 |
| MMDetection | 3.3.0 |
| scikit-learn | 1.8.0 |

所有命令默认从仓库根目录运行，并设置 `PYTHONPATH=.`。

### 1.2 新机器安装参考

当前服务器已有可用 `clrernet` 环境。新机器需要先安装匹配 CUDA 的 PyTorch，然后安装 MMDetection 3.x 相关依赖和本项目 CUDA NMS 扩展：

```bash
conda create -n clrernet python=3.11 -y
conda activate clrernet

pip install -r requirements.txt
pip install mmengine==0.10.5 mmcv==2.1.0 mmdet==3.3.0 opencv-python==4.9.0.80 scikit-learn==1.8.0

cd libs/models/layers/nms
python setup.py install
cd ../../../..
```

官方 CULane EMA 初始化权重路径：

```text
checkpoints/clrernet_culane_dla34_ema.pth
```

环境检查：

```bash
PYTHONPATH=. python -c "import torch, cv2, mmengine, mmcv, mmdet; print(torch.__version__, torch.cuda.is_available())"
```

## 2. 整体算法架构

### 2.1 Stage 1：Class-Agnostic Lane Locator

`HighwayLaneDataset` 读取 LabelMe 标注中的 `solid/dashed/joint`，但训练 CLRerNet 时统一映射为一个 `lane` 类。原始线型只保留在 metadata 中，供调试和 Stage 2 使用。

当前定位器流程：

1. 图像执行 `top_crop_ratio=0.08`，同步裁剪 polyline 坐标。
2. resize 到 `1024x544`，过滤越界或不足 2 个点的 lane。
3. DLA34 CLRerNet 从 `checkpoints/clrernet_culane_dla34_ema.pth` warm start。
4. 每 5 epoch 在 val split 上评估，当前最终 checkpoint 为 `epoch_25.pth`。
5. 推理使用 CLRerNet NMS，端到端评测采用 `conf=0.35,nms_thres=50,topk=8`。

关键设置：

| Setting | Value |
| --- | --- |
| Input resolution | `1024 x 544` |
| Top crop | `0.08` |
| Batch size | `8` |
| Epochs | `25` |
| Optimizer | `AdamW(lr=5e-5, weight_decay=0.01)` |
| Warm start | `checkpoints/clrernet_culane_dla34_ema.pth` |

### 2.2 Stage 2：Lane Instance Type Classifier

Stage 2 对单条 lane 做线型分类，不重新定位车道线。训练时使用 GT polyline 裁剪局部条带；端到端推理时使用 Stage 1 预测 polyline 裁剪同样格式的条带。

关键设置：

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

## 3. 项目目录结构与文件作用

```text
CLRerNet/
├── README.md
├── checkpoints/                              # 本地 checkpoint，不纳入 Git
├── configs/clrernet/
│   ├── base_clrernet.py                      # CLRerNet 基础模型配置
│   ├── lane_706_20260518/                    # 旧单数据源 lane_706 流程
│   └── lane_706_812_20260531/                # 当前多数据源合并训练配置
├── dataset/                                  # 本地数据，不纳入 Git
│   ├── lane_706_20260518/
│   ├── lane_812_20260531/
│   └── lane_706_812_20260531/splits/         # 虚拟合并 split，不复制图片
├── libs/
│   ├── datasets/
│   │   ├── highway_lane_dataset.py           # Stage 1 数据集，支持单 root / 多 root
│   │   ├── metrics/highway_lane_metric.py    # 类别无关 lane P/R/F1
│   │   └── pipelines/                        # top crop、resize、formatting 等 pipeline
│   ├── lane_classifier/
│   │   ├── crop.py                           # lane 条带裁剪、IoU、绘图工具
│   │   ├── dataset.py                        # Stage 2 GT lane crop 数据集，支持多 root
│   │   ├── model.py                          # 轻量 CNN 分类器
│   │   ├── train.py                          # Stage 2 训练入口
│   │   ├── infer.py                          # 两阶段推理入口
│   │   └── eval.py                           # 两阶段评测入口
│   └── utils/highway_data.py                 # 多 root split 解析与路径解析工具
├── tools/
│   ├── prepare_culane_highway.py             # 单数据源检查与 deterministic split
│   ├── build_highway_multiroot_splits.py     # 多数据源虚拟合并 split
│   ├── visualize_culane_highway_dataset.py   # pipeline 后 GT 可视化
│   ├── train.py                              # MMEngine Stage 1 训练入口
│   └── test.py                               # MMEngine Stage 1 测试入口
├── docs/
│   └── culane_highway_two_stage_report.md    # 历史实验与当前追加记录
└── work_dirs/                                # 训练/评测产物，不纳入 Git
```

## 4. 数据集结构

### 4.1 单数据源目录

每个标注批次使用平铺图片和同名 JSON：

```text
dataset/lane_812_20260531/
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

JSON 为 LabelMe `shapes` schema。有效 lane 标签只允许：

```text
solid
dashed
joint
```

### 4.2 多数据源 split 格式

合并 split 不复制图片，不生成合并图片目录。每行两列：

```text
dataset_key<TAB>relative_image_path
```

示例：

```text
lane_706_20260518	outside_20250910112526_000330.jpg
lane_812_20260531	outside_20250910112526_008611.PNG
```

当前合并 split 文件：

```text
dataset/lane_706_812_20260531/splits/train.txt
dataset/lane_706_812_20260531/splits/val.txt
dataset/lane_706_812_20260531/splits/test.txt
dataset/lane_706_812_20260531/splits/test_lane_706_20260518.txt
dataset/lane_706_812_20260531/splits/test_lane_812_20260531.txt
dataset/lane_706_812_20260531/splits/merge_report.json
```

旧一列 split 仍兼容：只传 `data_root` 时，一行就是相对图片路径。

## 5. 训练、评测与推理命令

### 5.1 生成新数据 split

```bash
PYTHONPATH=. python tools/prepare_culane_highway.py \
  --data-root dataset/lane_812_20260531 \
  --seed 0 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

### 5.2 生成合并 split

```bash
PYTHONPATH=. python tools/build_highway_multiroot_splits.py \
  --source lane_706_20260518=dataset/lane_706_20260518 \
  --source lane_812_20260531=dataset/lane_812_20260531 \
  --out-dir dataset/lane_706_812_20260531/splits \
  --prefer-on-duplicate lane_812_20260531
```

### 5.3 Stage 1 训练

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py \
  configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py \
  --work-dir work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain
```

如训练中断，从最近 checkpoint 恢复：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/train.py \
  configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py \
  --work-dir work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain \
  --resume auto
```

### 5.4 Stage 1 检测评测

合并 test：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py \
  configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py \
  work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth
```

706 / 812 子集：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py \
  configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py \
  work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_20260531/splits/test_lane_706_20260518.txt

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py \
  configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py \
  work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth \
  --cfg-options test_dataloader.dataset.data_list=dataset/lane_706_812_20260531/splits/test_lane_812_20260531.txt
```

### 5.5 Stage 2 训练

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.train \
  --data-roots lane_706_20260518=dataset/lane_706_20260518 lane_812_20260531=dataset/lane_812_20260531 \
  --split-root dataset/lane_706_812_20260531/splits \
  --work-dir work_dirs/lane_classifier/lane_706_812_20260531_stage2_strip_288x128_w128 \
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

### 5.6 Two-stage 评测

合并 test：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.eval \
  --data-roots lane_706_20260518=dataset/lane_706_20260518 lane_812_20260531=dataset/lane_812_20260531 \
  --split-file dataset/lane_706_812_20260531/splits/test.txt \
  --det-config configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_812_20260531_stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_eval/lane_706_812_20260531_test_s0.35_top8_nms50 \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-thres 50 \
  --nms-topk 8 \
  --iou-thr 0.3 \
  --match-width 20 \
  --device cuda:0
```

706 / 812 子集只需要替换 `--split-file` 和 `--out-dir`：

```text
--split-file dataset/lane_706_812_20260531/splits/test_lane_706_20260518.txt
--out-dir work_dirs/two_stage_eval/lane_706_812_20260531_test_lane_706_s0.35_top8_nms50

--split-file dataset/lane_706_812_20260531/splits/test_lane_812_20260531.txt
--out-dir work_dirs/two_stage_eval/lane_706_812_20260531_test_lane_812_s0.35_top8_nms50
```

### 5.7 单图或目录推理

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input dataset/lane_812_20260531 \
  --det-config configs/clrernet/lane_706_812_20260531/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_lane_706_812_20260531_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_25.pth \
  --cls-checkpoint work_dirs/lane_classifier/lane_706_812_20260531_stage2_strip_288x128_w128/best.pth \
  --out-dir work_dirs/two_stage_infer/lane_706_812_20260531 \
  --score-thr 0.35 \
  --det-conf-thr 0.35 \
  --nms-thres 50 \
  --nms-topk 8 \
  --device cuda:0
```

输出包含预测 JSON 和可视化图片。目录推理默认只读取当前层图片；需要递归时增加 `--recursive`。