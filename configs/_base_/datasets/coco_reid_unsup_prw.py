dataset_type = 'CUHK_SYSU_UNSUPDataset'
data_root = '/home/jl2/data/person_search/prw/PRW-v16.04.20/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_pid.json',
        img_prefix=data_root + 'frames',
        pipeline=train_pipeline),
    train_cluster=dict(
        type=dataset_type,
        ann_file=data_root + 'train_pid.json',
        img_prefix=data_root + 'frames',
        pipeline=test_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_pid.json',
        img_prefix=data_root + 'frames',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_pid.json',
        img_prefix=data_root + 'frames',
        proposal_file=data_root+'annotation/test/train_test/TestG50.mat',
        pipeline=test_pipeline))
evaluation = dict(interval=30, metric='bbox')
