#!/bin/bash

export PYTHONPATH=.

# ============================
# 공통 설정
# ============================
MODEL=unetpp
ENCODER=resnext50_32x4d
BOUNDARY=basnet
PRETRAIN=imagenet
FOLD=0
EPOCHS=60

mkdir -p logs

echo "============================"
echo " EXP 1: BCE + Dice"
echo "============================"

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode $BOUNDARY \
  --use_refinement \
  --pretrained $PRETRAIN \
  --fold $FOLD \
  --num_epochs $EPOCHS \
  --loss_mode bce_dice \
  --no_boundary_detach \
  --use_wandb \
> logs/bce_dice.log 2>&1

echo "============================"
echo " EXP 2: BCE + Dice + Jaccard"
echo "============================"

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode $BOUNDARY \
  --use_refinement \
  --pretrained $PRETRAIN \
  --fold $FOLD \
  --num_epochs $EPOCHS \
  --loss_mode bce_dice_jaccard \
  --no_boundary_detach \
  --use_wandb \
> logs/bce_dice_jaccard.log 2>&1

echo "✅ All loss experiments finished"