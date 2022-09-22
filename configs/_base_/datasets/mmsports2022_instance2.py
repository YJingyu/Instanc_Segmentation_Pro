dataset_type = 'MMSports2022Dataset'
data_root = '/lengyu.yb/datasets/segmentation/MMSports22/basketball-instants-dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1440, 1920)   # (h,w)
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='GeoAugment'),
    dict(type='Resize', img_scale=[(3680, 820), (3680, 3080)], keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=image_size),  #(h, w)
]

train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='GridMask', use_w=True, use_h=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # single scale
        img_scale=(1920, 1440),
        flip=False,
        # mul scale
        # scale_factor=[1., 1.5, 2.0, 2.5, 2.75],
        # scale_factor=[1., 1.5, 2.0, 2.5, 3.0],
        # flip=True,
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
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            # ann_file=data_root + 'annotations/train.json',
            # ann_file=data_root + 'annotations/trainval.json',
            ann_file=data_root + 'annotations/trainvaltest.json',
            img_prefix=data_root,
            pipeline=load_pipeline),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/val.json',
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/test.json',
        ann_file=data_root + 'annotations/challenge.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])