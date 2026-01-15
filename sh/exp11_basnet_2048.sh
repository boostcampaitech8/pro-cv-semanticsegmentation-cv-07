#!/bin/bash
export PYTHONPATH=.

# ============================
# EXP11: BASNet (NO boundary detach)
# ============================
MODEL=unetpp
ENCODER=resnext50_32x4d
PRETRAIN=imagenet
FOLD=0

LOSS=bce_dice
INPUT_SIZE=2048
BATCH=2
EPOCHS=80
LR=1e-4

mkdir -p logs

echo "======================================"
echo " EXP11: BASNET "
echo " img=2048 | batch=2 | e=80 "
echo "======================================"

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode basnet \
  --use_refinement \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size $INPUT_SIZE \
  --batch_size $BATCH \
  --num_epochs $EPOCHS \
  --lr $LR \
  --val_every 1 \
  --use_wandb \
  --exp_tag exp11_basnet_nodetach_b4_img1024 \
> logs/exp11_basnet_b2_img2048_e80.log 2>&1

echo "===== EXP11 FINISHED ====="