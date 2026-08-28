# Stage 2 车道线分类优化

本文记录针对 `solid/dashed/joint` 分类的可复现实验流程。Stage 1 在本轮保持冻结，Stage 2 的模型、裁条方式和训练分布按 validation 选择。

## 实现内容

- 使用冻结 Stage 1 为 train/val split 生成预测车道线 JSON。
- 预测车道与 GT 按 lane IoU 贪心匹配，继承 GT 类别；训练默认按 `70%` 预测裁条、`30%` GT/扰动 GT 裁条采样。
- GT 裁条增强包含法向偏移、端点截断、strip 宽度扰动、遮挡、模糊和亮度变化。
- `crop_mode` 支持：原图固定宽度、按原图宽度缩放、先归一化到 `1024x544 topcrop8`。
- 新分类器沿 strip 宽度池化，保留纵向序列；使用 dilation `1/2/4` 的 depthwise `Conv1d`、注意力/均值/最大值池化。
- 辅助形态分支使用平滑的白色、黄色和局部对比度响应，在 16 个纵向区段中编码占比、连续、间隔和状态转换，再与 CNN 特征联合训练。它不会用固定阈值覆盖分类结果。
- 最优 epoch 在 validation 上进行温度标定。推理 JSON 保留 `label`、`label_score`，并新增三类完整 `label_probs`。
- 评测同时输出 matched-lane 分类、IoU 0.3/0.5 定位指标、包含漏检/多检/错分类的端到端逐类 P/R/F1，以及按原始分辨率分组的分类指标。

## 训练数据准备

先分别为训练集和验证集生成预测车道。必须使用部署时同一 Stage 1 checkpoint 和后处理参数，并使用空的新目录，避免混入旧 JSON。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/build_stage2_predicted_lanes.py \
  --data-roots <key1>=<data_root1> <key2>=<data_root2> \
  --split-file <split_root>/train.txt \
  --det-config <stage1_config.py> \
  --det-checkpoint <stage1_checkpoint.pth> \
  --out-dir work_dirs/stage2_predicted_lanes/train \
  --score-thr 0.35 --det-conf-thr 0.35 --nms-topk 8 --nms-thres 50 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/build_stage2_predicted_lanes.py \
  --data-roots <key1>=<data_root1> <key2>=<data_root2> \
  --split-file <split_root>/val.txt \
  --det-config <stage1_config.py> \
  --det-checkpoint <stage1_checkpoint.pth> \
  --out-dir work_dirs/stage2_predicted_lanes/val \
  --score-thr 0.35 --det-conf-thr 0.35 --nms-topk 8 --nms-thres 50 \
  --device cuda:0
```

## 消融训练

所有选择只看 validation `Macro-F1`。295 张独立测试集已经用于错误诊断，不能再用它选择模型、尺度或阈值。

1. 旧基线：`--model-type legacy --crop-mode fixed`，且不传预测目录。
2. 分布对齐：传 `--train-pred-dir/--val-pred-dir`，仍使用 `legacy/fixed`。
3. 尺度：比较 `fixed`、`scaled`、`normalized`。归一化模式分别测试 `--strip-width 48` 和 `64`。
4. 纵向序列与形态融合：`--model-type sequence_fusion`。

推荐起始命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.train \
  --data-roots <key1>=<data_root1> <key2>=<data_root2> \
  --split-root <split_root> \
  --train-pred-dir work_dirs/stage2_predicted_lanes/train \
  --val-pred-dir work_dirs/stage2_predicted_lanes/val \
  --work-dir work_dirs/lane_classifier/stage2_sequence_fusion_scaled \
  --model-type sequence_fusion \
  --predicted-crop-prob 0.7 \
  --prediction-iou-thr 0.3 --match-width 20 \
  --crop-mode scaled \
  --strip-width 128 --strip-reference-width 2560 \
  --strip-min-width 64 --strip-max-width 192 \
  --crop-height 288 --crop-width 128 \
  --normal-offset-ratio 0.12 --endpoint-truncate-ratio 0.15 \
  --strip-width-jitter 0.8 1.2 \
  --epochs 30 --batch-size 64 --num-workers 4 \
  --lr 1e-3 --weight-decay 1e-4 --dropout 0.25 \
  --class-balance loss --device cuda:0 --seed 0
```

## 推理与评测

新 checkpoint 会自动恢复训练时的模型类型、尺度策略和温度，不需要在部署端重新填写。同一张图的所有 lane crop 会合并成一个 batch 做分类，避免逐 lane GPU 调用开销。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.infer \
  --input <image_or_directory> \
  --det-config <stage1_config.py> \
  --det-checkpoint <stage1_checkpoint.pth> \
  --cls-checkpoint <stage2_best.pth> \
  --out-dir work_dirs/two_stage_infer/stage2_optimized \
  --device cuda:0

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.eval \
  --data-roots <key1>=<data_root1> <key2>=<data_root2> \
  --split-file <frozen_test.txt> \
  --det-config <stage1_config.py> \
  --det-checkpoint <stage1_checkpoint.pth> \
  --cls-checkpoint <stage2_best.pth> \
  --out-dir work_dirs/two_stage_eval/stage2_optimized \
  --iou-thr 0.3 --iou-thresholds 0.3 0.5 --match-width 20 \
  --device cuda:0
```

## 验收规则

- `matched-lane Macro-F1 >= 0.75`
- `joint F1 >= 0.55`
- `Accuracy >= 0.80`
- `solid/dashed` 任一类别 F1 相对基线下降不超过 `0.01`
- Stage 1 指标不变
- 两阶段单图推理耗时增幅不超过 `10%`

2026-08-27 时甲方提供资源中没有 5496 张训练数据的图片、标注、split 和数据源映射，因此当时只能完成代码实现与 smoke test。2026-08-28 收到的四批新增训练数据已完成下述正式实验；结果只代表服务器当前十数据源训练集和冻结六数据源 test，不等同于缺少复现材料的甲方 5496 图版本。

## 2026-08-27 Smoke Test

- 使用服务器现有六数据源 validation 构建数据集：3878 条 lane，类别计数 `2362/934/582`。
- 冻结 Stage 1 对 8 张图生成 34 条预测 lane，其中 30 条与 GT 在 IoU 0.3 下成功匹配。
- 一次真实优化步：batch `(8,3,288,128)`，预测/GT crop 混合正常，CUDA 峰值约 468 MiB，logits 和 loss 均为有限值。
- 8 张图、每图 5 次、各 lane 数量预热后的延迟测量：旧两阶段平均 `42.06 ms/image`，新结构 `43.98 ms/image`，增幅 `4.55%`。该结果通过实现层面的 10% 延迟门槛，但精度仍必须在完整 5496 validation 和新冻结验收集上验证。

## 2026-08-28 四批新增数据正式实验

### 数据集成

四个压缩包均先通过 `7z t` 完整性检查和路径越界检查，再解压为独立数据根。每批均有 450 对可读图片/LabelMe JSON，标签只包含 `solid/dashed/joint`：

| 数据源 | 图片/JSON 对 | solid | dashed | joint | prepare errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lane_260727_260802` | 450 | 1832 | 701 | 519 | 0 |
| `lane_260803_260809` | 450 | 1815 | 696 | 531 | 0 |
| `lane_260810_260816` | 450 | 1836 | 725 | 514 | 0 |
| `lane_260817_260823` | 450 | 1867 | 686 | 563 | 0 |

新四批只进入 train。合并后的十数据源 split 位于：

```text
dataset/lane_706_812_349_696_350_1149_260727_260823/splits/
```

最终计数为 `train=4624`、`val=606`、`test=615`。train 标签计数为 `solid=18525`、`dashed=7190`、`joint=4767`；固定 val/test 标签计数分别为 `2362/934/582` 和 `2524/1053/578`。新 split 的 val/test 与原六数据源文件逐字节一致。合并报告记录 17 组重复图；其中九个新训练副本因为对应图片已在旧 val/test 中而被排除，避免测试泄漏。旧六源整体优先，以保留原 split；新四批彼此重复时按日期较新的来源优先。

实际单源 split 命令对四个新数据根各运行一次：

```bash
PYTHONPATH=. python tools/prepare_culane_highway.py \
  --data-root dataset/<lane_2607xx_2608xx> \
  --seed 0 --train-ratio 1.0 --val-ratio 0.0 --test-ratio 0.0
```

十数据源合并命令：

```bash
PYTHONPATH=. python tools/build_highway_multiroot_splits.py \
  --source lane_706_20260518=dataset/lane_706_20260518 \
  --source lane_812_20260531=dataset/lane_812_20260531 \
  --source lane_349_20260609=dataset/lane_349_20260609 \
  --source lane_696_20260624=dataset/lane_696_20260624 \
  --source lane_350_20260629=dataset/lane_350_20260629 \
  --source lane_1149_20260722=dataset/lane_1149_20260722 \
  --source lane_260727_260802=dataset/lane_260727_260802 \
  --source lane_260803_260809=dataset/lane_260803_260809 \
  --source lane_260810_260816=dataset/lane_260810_260816 \
  --source lane_260817_260823=dataset/lane_260817_260823 \
  --out-dir dataset/lane_706_812_349_696_350_1149_260727_260823/splits \
  --source-priority lane_1149_20260722,lane_350_20260629,lane_696_20260624,lane_349_20260609,lane_812_20260531,lane_706_20260518,lane_260817_260823,lane_260810_260816,lane_260803_260809,lane_260727_260802
```

### 冻结 Stage 1 与部署一致裁条

Stage 1 未训练或调参，固定使用：

```text
checkpoint = work_dirs/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_20.pth
score_thr = det_conf_thr = 0.40
nms_topk = 12
nms_thres = 40
```

冻结定位器为 4624 张 train 图生成 25758 条预测 lane，为 606 张 val 图生成 3357 条。预测 lane 与 GT 按 IoU 0.3、宽度 20 匹配；Stage 2 train 最终有 30482 个候选实例，类别计数与合并 train 标签一致。每个 epoch 默认以 `70%` 概率使用预测 lane crop，其余使用 GT 或扰动 GT crop。

预测 lane 生成命令对 `train.txt` 和 `val.txt` 各运行一次，仅替换 split 文件和输出子目录：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/build_stage2_predicted_lanes.py \
  --data-roots <上述十个 key=root 条目> \
  --split-file dataset/lane_706_812_349_696_350_1149_260727_260823/splits/<train_or_val>.txt \
  --det-config configs/clrernet/lane_706_812_349_696_350_1149_20260722/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8.py \
  --det-checkpoint work_dirs/clrernet_lane_706_812_349_696_350_1149_20260722_dla34_ema_locator_1024x544_topcrop8_culane_pretrain/epoch_20.pth \
  --out-dir work_dirs/stage2_predicted_lanes/lane_706_812_349_696_350_1149_260727_260823_epoch20_s0.40_top12_nms40/<train_or_val> \
  --score-thr 0.40 --det-conf-thr 0.40 --nms-topk 12 --nms-thres 40 \
  --device cuda:0
```

归一化裁条扫描发现 96 条 train GT lane 完全位于 `topcrop8` 上方。它们在 normalized 模式中被明确排除，而不是生成空白 crop；穿过裁切边界的 polyline 会插入精确边界交点。固定和宽度缩放模式仍使用全部实例。四种模式的固定 validation matched support 均为 3156（`solid=1821,dashed=819,joint=516`）。

### 裁条方案选择

四个 seed 0 模型均从零训练 30 epochs，只按固定 validation 的 matched-lane Macro-F1 选择 epoch：

| 模式 | 最佳 epoch | Accuracy | Macro-F1 | solid F1 | dashed F1 | joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed 128 px | 22 | 0.910646 | 0.866631 | 0.966869 | 0.898604 | 0.734420 |
| scaled 64-192 px | 16 | 0.913498 | 0.868821 | 0.966428 | 0.905301 | 0.734735 |
| normalized 1024x544, 48 px | 26 | 0.909062 | 0.865345 | 0.962180 | 0.906935 | 0.726919 |
| normalized 1024x544, 64 px | 17 | 0.912864 | 0.866071 | 0.967688 | 0.901573 | 0.728953 |

scaled 与 fixed 的 Macro-F1 差为 `0.002190 < 0.003`。32 张 validation 图、每张 7 次交替测量显示 fixed Stage 2 为 `13.434 ms/image`，scaled 为 `13.471 ms/image`，所以按预先规定的平局规则选择 fixed 128 px。

### 随机种子稳定性与最终权重

| Seed | 最佳 epoch | Accuracy | Macro-F1 | joint F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 22 | 0.910646 | 0.866631 | 0.734420 |
| 1 | 23 | 0.910646 | 0.869454 | 0.735831 |
| 2 | 27 | 0.907795 | 0.867553 | 0.733826 |

Macro-F1 均值为 `0.867879`，总体标准差为 `0.001176`。最终按 validation 选择 seed 1、epoch 23：

```text
work_dirs/lane_classifier/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_p70_seed1/best.pth
```

最终 seed 1 的训练命令如下；seed 0/2 只替换 `--seed` 和 work dir：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m libs.lane_classifier.train \
  --data-roots <上述十个 key=root 条目> \
  --split-root dataset/lane_706_812_349_696_350_1149_260727_260823/splits \
  --train-pred-dir work_dirs/stage2_predicted_lanes/lane_706_812_349_696_350_1149_260727_260823_epoch20_s0.40_top12_nms40/train \
  --val-pred-dir work_dirs/stage2_predicted_lanes/lane_706_812_349_696_350_1149_260727_260823_epoch20_s0.40_top12_nms40/val \
  --work-dir work_dirs/lane_classifier/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_p70_seed1 \
  --model-type sequence_fusion --predicted-crop-prob 0.7 \
  --prediction-iou-thr 0.3 --match-width 20 \
  --crop-mode fixed --crop-height 288 --crop-width 128 --strip-width 128 \
  --normal-offset-ratio 0.12 --endpoint-truncate-ratio 0.15 \
  --strip-width-jitter 0.8 1.2 \
  --epochs 30 --batch-size 64 --num-workers 4 \
  --lr 1e-3 --weight-decay 1e-4 --dropout 0.25 \
  --class-balance loss --debug-crops 0 --device cuda:0 --seed 1
```

### 固定 test 一次性评测

旧模型与新模型使用完全相同的 615 张 test、Stage 1 checkpoint、预测 lane 和匹配参数：

| Stage 2 | Accuracy | Macro-F1 | solid F1 | dashed F1 | joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 六数据源 legacy | 0.766903 | 0.706844 | 0.832919 | 0.842205 | 0.445407 |
| 十数据源 sequence_fusion | **0.907070** | **0.859555** | **0.959844** | **0.913411** | **0.705411** |

Stage 1 保持 `P=0.939385`、`R=0.779543`、`F1@0.3=0.852032`，`F1@0.5=0.792056`。新模型的端到端 Macro-F1 为 `0.750499`（IoU 0.3）和 `0.701552`（IoU 0.5）。各源 matched-lane 结果如下：

| Split | Accuracy | Macro-F1 | solid F1 | dashed F1 | joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lane_706 | 0.875000 | 0.836973 | 0.931408 | 0.906433 | 0.673077 |
| lane_812 | 0.935385 | 0.900378 | 0.962963 | 0.951654 | 0.786517 |
| lane_349 | 0.869416 | 0.796836 | 0.945783 | 0.876543 | 0.568182 |
| lane_696 | 0.900000 | 0.848831 | 0.965964 | 0.891641 | 0.688889 |
| lane_350 | 0.896296 | 0.851144 | 0.946667 | 0.909091 | 0.697674 |
| lane_1149 | 0.925764 | 0.876349 | 0.977230 | 0.915385 | 0.736434 |

最终真实权重延迟测量使用 32 张固定 test 图，每张 7 次。第一次运行的首张 Stage 1 出现 `781.946 ms` 一次性异常点，因此保留原报告但不采用其总耗时作为验收数字。独立空闲 GPU 复测无该异常点：旧 Stage 2 为 `12.300 ms/image`，新 Stage 2 为 `13.975 ms/image`；两阶段总耗时从 `44.763` 增至 `46.438 ms/image`，增幅 `3.740%`。

### 验收结论

- Accuracy `0.9071 >= 0.80`：通过。
- Macro-F1 `0.8596 >= 0.75`：通过。
- joint F1 `0.7054 >= 0.55`：通过。
- solid/dashed F1 均高于旧模型，未发生超过 `0.01` 的回退：通过。
- Stage 1 checkpoint、后处理和 test 预测数量未改变：通过。
- 两阶段延迟增加 `3.74% <= 10%`：通过。
- 295 张甲方诊断集未参与模型、裁条、阈值或 checkpoint 选择。

主要机器可读产物：

```text
dataset/lane_706_812_349_696_350_1149_260727_260823/splits/merge_report.json
work_dirs/lane_classifier/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_selection_summary.json
work_dirs/lane_classifier/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_crop_sweep_summary.json
work_dirs/lane_classifier/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_latency/old_vs_final_fixed128_seed1_repeat_gpu3.json
work_dirs/two_stage_eval/lane_706_812_349_696_350_1149_260727_260823_stage2_sequence_fusion_fixed128_seed1/merged/eval_report.json
```
