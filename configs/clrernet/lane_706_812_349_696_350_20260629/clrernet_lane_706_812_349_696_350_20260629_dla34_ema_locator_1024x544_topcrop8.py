
_base_ = [
    '../base_clrernet.py',
    'dataset_lane_706_812_349_696_350_20260629_clrernet_1024x544_topcrop8.py',
    '../../_base_/default_runtime.py',
]

default_scope = 'mmdet'

custom_imports = dict(
    imports=[
        'libs.models',
        'libs.datasets',
        'libs.core.bbox',
        'libs.core.anchor',
        'libs.core.hook',
    ],
    allow_failed_imports=False,
)

cfg_name = 'clrernet_lane_706_812_349_696_350_20260629_dla34_ema_locator_1024x544_topcrop8.py'
load_from = 'checkpoints/clrernet_culane_dla34_ema.pth'
work_dir = './work_dirs/clrernet_lane_706_812_349_696_350_20260629_dla34_ema_locator_1024x544_topcrop8_culane_pretrain'

model = dict(
    backbone=dict(pretrained=False),
    bbox_head=dict(
        img_w=1024,
        img_h=544,
        loss_iou=dict(lane_width=7.5 / 1024, img_w=1024, img_h=544),
        loss_seg=dict(loss_weight=0.0),
    ),
    train_cfg=dict(
        assigner=dict(
            iou_dynamick=dict(lane_width=7.5 / 1024, img_w=1024, img_h=544),
            iou_cost=dict(lane_width=30 / 1024, img_w=1024, img_h=544),
        ),
    ),
    test_cfg=dict(
        conf_threshold=0.4,
        use_nms=True,
        as_lanes=True,
        extend_bottom=True,
        nms_thres=40,
        nms_topk=10,
        ori_img_w=1024,
        ori_img_h=544,
        cut_height=0,
    ),
)

custom_hooks = []

total_epochs = 25
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=4)
test_dataloader = dict(batch_size=4)

randomness = dict(seed=0, deterministic=False)

optim_wrapper = dict(
    type='OptimWrapper',
    accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=5e-7,
        begin=0,
        T_max=total_epochs,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=6),
    logger=dict(type='LoggerHook', interval=20),
)
