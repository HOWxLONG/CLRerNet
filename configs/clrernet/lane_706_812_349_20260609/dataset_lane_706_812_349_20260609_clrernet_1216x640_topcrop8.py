
dataset_type = 'HighwayLaneDataset'
data_roots = dict(
    lane_706_20260518='dataset/lane_706_20260518',
    lane_812_20260531='dataset/lane_812_20260531',
    lane_349_20260609='dataset/lane_349_20260609',
)
split_root = 'dataset/lane_706_812_349_20260609/splits'
img_scale = (1216, 640)
top_crop_ratio = 0.08
max_lanes = 24
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
compose_cfg = dict(bboxes=False, keypoints=True, masks=False)

resize_al_pipeline = [
    dict(type='Compose', params=compose_cfg),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

train_al_pipeline = [
    dict(type='Compose', params=compose_cfg),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='ChannelShuffle', p=0.1),
    dict(type='RandomBrightnessContrast', brightness_limit=0.04, contrast_limit=0.15, p=0.6),
    dict(type='HueSaturationValue', hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10), val_shift_limit=(-10, 10), p=0.7),
    dict(
        type='OneOf',
        transforms=[
            dict(type='MotionBlur', blur_limit=5, p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
        ],
        p=0.2,
    ),
]

train_pipeline = [
    dict(type='TopCrop', crop_ratio=top_crop_ratio, min_points=2),
    dict(type='albumentation', pipelines=train_al_pipeline),
    dict(type='LanePointFilter', min_points=2),
    dict(
        type='PackCLRNetInputs',
        max_lanes=max_lanes,
        img_w=img_scale[0],
        img_h=img_scale[1],
        meta_keys=[
            'filename',
            'sub_img_name',
            'data_root_key',
            'lane_json_path',
            'ori_shape',
            'img_shape',
            'top_crop_px',
            'top_crop_ratio',
            'top_crop_original_shape',
            'top_crop_removed_lanes',
            'lane_point_filter_removed_lanes',
            'gt_points',
            'gt_lane_labels',
            'gt_masks',
            'lanes',
        ],
    ),
]

val_pipeline = [
    dict(type='TopCrop', crop_ratio=top_crop_ratio, min_points=2),
    dict(type='albumentation', pipelines=resize_al_pipeline),
    dict(type='LanePointFilter', min_points=2),
    dict(
        type='PackCLRNetInputs',
        max_lanes=max_lanes,
        img_w=img_scale[0],
        img_h=img_scale[1],
        meta_keys=[
            'filename',
            'sub_img_name',
            'data_root_key',
            'lane_json_path',
            'ori_shape',
            'img_shape',
            'top_crop_px',
            'top_crop_ratio',
            'top_crop_original_shape',
            'top_crop_removed_lanes',
            'lane_point_filter_removed_lanes',
            'gt_points',
            'gt_lane_labels',
        ],
    ),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_roots=data_roots,
        data_list=split_root + '/train.txt',
        pipeline=train_pipeline,
        test_mode=False,
        max_samples=None,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_roots=data_roots,
        data_list=split_root + '/val.txt',
        pipeline=val_pipeline,
        test_mode=True,
        max_samples=None,
    ),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_roots=data_roots,
        data_list=split_root + '/test.txt',
        pipeline=val_pipeline,
        test_mode=True,
        max_samples=None,
    ),
)

val_evaluator = dict(type='HighwayLaneMetric', img_w=img_scale[0], img_h=img_scale[1], iou_thresholds=[0.3, 0.5])
test_evaluator = dict(type='HighwayLaneMetric', img_w=img_scale[0], img_h=img_scale[1], iou_thresholds=[0.3, 0.5])
