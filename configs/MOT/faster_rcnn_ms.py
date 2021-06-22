_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4_reid.py',
    '../_base_/datasets/coco_mot17.py',
    '../_base_/schedules/schedule_2x_mot17.py', '../_base_/default_runtime.py'
]
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(544, 304), (1306, 730)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
query_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['proposals']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline, query_test_pipeline=None),
    val=dict(pipeline=test_pipeline, query_test_pipeline=None),
    test=dict(pipeline=test_pipeline,
        query_test_pipeline=query_test_pipeline,
    ))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
load_from = ''
