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

目前甲方提供资源中没有 5496 张训练数据的图片、标注、split 和数据源映射，因此本仓库只能完成代码实现与现有服务器数据 smoke test，不能据此声称已完成 5496 数据重训或达到上述精度门槛。

## 2026-08-27 Smoke Test

- 使用服务器现有六数据源 validation 构建数据集：3878 条 lane，类别计数 `2362/934/582`。
- 冻结 Stage 1 对 8 张图生成 34 条预测 lane，其中 30 条与 GT 在 IoU 0.3 下成功匹配。
- 一次真实优化步：batch `(8,3,288,128)`，预测/GT crop 混合正常，CUDA 峰值约 468 MiB，logits 和 loss 均为有限值。
- 8 张图、每图 5 次、各 lane 数量预热后的延迟测量：旧两阶段平均 `42.06 ms/image`，新结构 `43.98 ms/image`，增幅 `4.55%`。该结果通过实现层面的 10% 延迟门槛，但精度仍必须在完整 5496 validation 和新冻结验收集上验证。
