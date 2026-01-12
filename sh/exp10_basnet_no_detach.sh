#!/bin/bash
export PYTHONPATH=.

# ============================
# EXP10: BASNet (NO boundary detach)
# ============================
MODEL=unetpp
ENCODER=resnext50_32x4d
PRETRAIN=imagenet
FOLD=0

LOSS=bce_dice
INPUT_SIZE=1024
BATCH=4
EPOCHS=80
LR=1e-4

mkdir -p logs

echo "======================================"
echo " EXP10: BASNET (NO BOUNDARY DETACH) "
echo " img=1024 | batch=4 | e=80 "
echo "======================================"

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode basnet \
  --use_refinement \
  --no_boundary_detach \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size $INPUT_SIZE \
  --batch_size $BATCH \
  --num_epochs $EPOCHS \
  --lr $LR \
  --val_every 1 \
  --use_wandb \
  --exp_tag exp10_basnet_nodetach_b4_img1024 \
> logs/exp10_basnet_nodetach_b4_img1024.log 2>&1

echo "===== EXP10 FINISHED ====="