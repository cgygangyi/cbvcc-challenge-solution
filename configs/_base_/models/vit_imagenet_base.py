# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_ImageNet',
        img_size=224,
        patch_size=16,
        num_frames=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg = dict(average_clips='prob'))