model = dict(                                                                                                                                                                                               
    type='CascadeRCNN',                                                                                                                                                                                                                                                                                                                                                                          
    pretrained='open-mmlab://msra/hrnetv2_w32',                                                                                                                                                             
    backbone=dict(                                                                                                                                                                                          
        type='HRNet',                                                                                                                                                                                       
        extra=dict(                                                                                                                                                                                         
            stage1=dict(                                                                                                                                                                                    
                num_modules=1,                                                                                                                                                                              
                num_branches=1,                                                                                                                                                                             
                block='BOTTLENECK',                                                                                                                                                                         
                num_blocks=(4, ),                                                                                                                                                                           
                num_channels=(64, )),                                                                                                                                                                       
            stage2=dict(                                                                                                                                                                                    
                num_modules=1,                                                                                                                                                                              
                num_branches=2,                                                                                                                                                                             
                block='BASIC',                                                                                                                                                                              
                num_blocks=(4, 4),                                                                                                                                                                          
                num_channels=(32, 64)),                                                                                                                                                                     
            stage3=dict(                                                                                                                                                                                    
                num_modules=4,                                                                                                                                                                              
                num_branches=3,                                                                                                                                                                             
                block='BASIC',                                                                                                                                                                              
                num_blocks=(4, 4, 4),                                                                                                                                                                       
                num_channels=(32, 64, 128)),                                                                                                                                                                
            stage4=dict(                                                                                                                                                                                    
                num_modules=3,                                                                                                                                                                              
                num_branches=4,                                                                                                                                                                             
                block='BASIC',                                                                                                                                                                              
                num_blocks=(4, 4, 4, 4),                                                                                                                                                                    
                num_channels=(32, 64, 128, 256))                                                                                                                                                            
        ),                                                                                                                                                                                                  
        #frozen_stages=-1,                                                                                                                                                                                  
        norm_eval=False,                                                                                                                                                                                    
    ),                                                                                                                                                                                                      
    neck=dict(                                                                                                                                                                                              
        type='HRFPN',                                                                                                                                                                                       
        in_channels=[32, 64, 128, 256],                                                                                                                                                                     
        out_channels=256),                                                                                                                                                                                  
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            strides=[4, 8, 16, 32, 64],
            ratios=[0.5, 1.0, 2.0],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),

    roi_head=dict(
        type='CascadeRoIReidHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxIdHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxIdHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxIdHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]))
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=0.7),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=0.7),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=0.7),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                match_low_quality=False,
                ignore_iof_thr=0.7),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=12000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=1000))
dataset_type = 'CocoDataset'
data_root = '/raid/ljp/data/MOT/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + 'MOT17/mot17half.json',
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
		        img_scale=[(1600, 400), (1600, 1400)],
                multiscale_mode='range',
                keep_ratio=True),
	   dict(
        	type='PhotoMetricDistortion',
        	brightness_delta=180,
        	contrast_range=(0.5, 1.5),
        	saturation_range=(0.5, 1.5),
        	hue_delta=18),
	   dict(
        	type='MinIoURandomCrop',
        	min_ious=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        	min_crop_size=0.1),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + 'MOT17/mot17val.json',
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
		        img_scale=(2000, 1000),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'MOT17/mot17val.json',
        img_prefix=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
		        img_scale=(2000, 1000),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,)
total_epochs = 240
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/cascade_small_lr'
gpu_ids = range(0, 1)
