_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4_reid_norm_unsu.py',
    '../_base_/datasets/coco_reid_unsup_prw.py',
    '../_base_/schedules/schedule_1x_reid_norm_base.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='CGPSHead',
            id_num=18048,
            top_percent=0.6,
            )
    )
)
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
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
        #img_scale=(1333, 800),
        img_scale=(1000, 600),
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
        #img_scale=(1333, 800),
        img_scale=(1000, 600),
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

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1333, 800),
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline, query_test_pipeline=None),
    train_cluster=dict(pipeline=val_pipeline, query_test_pipeline=None),
    val=dict(pipeline=test_pipeline, query_test_pipeline=None),
    test=dict(pipeline=test_pipeline,
        query_test_pipeline=query_test_pipeline,
    ))

SPCL=True
PSEUDO_LABELS = dict(
    freq=1, # epochs
    use_outliers=True,
    norm_feat=True,
    norm_center=True,
    cluster='dbscan_context',
    eps=[0.68, 0.7, 0.72],
    min_samples=4, # for dbscan
    dist_metric='jaccard',
    k1=30, # for jaccard distance
    k2=6, # for jaccard distance
    search_type=0, # 0,1,2 for GPU, 3 for CPU (work for faiss)
    cluster_num=None
)
# fp16 = dict(loss_scale=512.)