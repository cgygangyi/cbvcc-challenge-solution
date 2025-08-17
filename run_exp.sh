#!/usr/bin/env bash

# Training
bash tools/dist_train.sh configs/recognition/vit/vitclip_base_diving48.py 8 --test-last --validate \
--cfg-options model.backbone.pretrained=openaiclip work_dir=work_dirs_vit/diving48/debug

# Evaluation only
# bash tools/dist_test.sh configs/recognition/vit/vitclip_large_k400.py checkpoints/vit_b_clip_8frame_k400.pth 8 --eval top_k_accuracy