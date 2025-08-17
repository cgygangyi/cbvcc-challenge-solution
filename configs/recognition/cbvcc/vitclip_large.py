model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_CLIP',
        input_resolution=224,
        patch_size=14,
        num_frames=21,
        width=1024,
        layers=24,
        heads=16,
        drop_path_rate=0.25,
        adapter_scale=0.5),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg=dict(average_clips='prob', max_testing_views=4))
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=7,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmaction2', entity='uw-cbvcc', name='vit_large'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
dataset_type = 'VideoDataset'
data_root = 'data/videos'
data_root_val = 'data/videos'
ann_file_train = 'data/train_list_videos.txt'
ann_file_val = 'data/phase1_list_videos.txt'
ann_file_test = 'data/phase1_list_videos.txt'
img_norm_cfg = dict(
    mean=[122.769, 116.74, 104.04], std=[68.493, 66.63, 70.321], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=21,
        frame_interval=1,
        num_clips=1,
        frame_uniform=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Imgaug', transforms=[dict(type='RandAugment', n=4, m=7)]),
    dict(
        type='Normalize',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=21,
        frame_interval=1,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=21,
        frame_interval=1,
        num_clips=1,
        frame_uniform=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=1,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type='VideoDataset',
        ann_file='data/train_list_videos.txt',
        data_prefix='data/videos',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=21,
                frame_interval=1,
                num_clips=1,
                frame_uniform=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Imgaug', transforms=[dict(type='RandAugment', n=4,
                                                m=7)]),
            dict(
                type='Normalize',
                mean=[122.769, 116.74, 104.04],
                std=[68.493, 66.63, 70.321],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        start_index=0),
    val=dict(
        type='VideoDataset',
        ann_file='data/phase1_list_videos.txt',
        data_prefix='data/videos',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=21,
                frame_interval=1,
                num_clips=1,
                frame_uniform=True,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[122.769, 116.74, 104.04],
                std=[68.493, 66.63, 70.321],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ],
        start_index=0),
    test=dict(
        type='VideoDataset',
        ann_file='data/phase1_list_videos.txt',
        data_prefix='data/videos',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=21,
                frame_interval=1,
                num_clips=1,
                frame_uniform=True,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[122.769, 116.74, 104.04],
                std=[68.493, 66.63, 70.321],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ],
        start_index=0))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.08,
    paramwise_cfg=dict(
        custom_keys=dict(
            class_embedding=dict(decay_mult=0.0),
            positional_embedding=dict(decay_mult=0.0),
            ln_1=dict(decay_mult=0.0),
            ln_2=dict(decay_mult=0.0),
            ln_pre=dict(decay_mult=0.0),
            ln_post=dict(decay_mult=0.0))))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-06,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=3)
total_epochs = 80
checkpoint = dict(
    type='CheckpointHook',
    interval=1,
    save_best='auto',
    rule='greater',
    max_keep_ckpts=3)
work_dir = './work_dirs/vitclip_large'
find_unused_parameters = False
fp16 = None
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)
gpu_ids = range(0, 2)
omnisource = False
module_hooks = []
