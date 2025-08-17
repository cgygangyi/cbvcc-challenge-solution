# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer2D',
        patch_size=4,
        num_frames=32,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        frozen_stages=-1,),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg = dict(average_clips='prob'))