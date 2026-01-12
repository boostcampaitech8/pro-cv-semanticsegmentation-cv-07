#!/bin/bash
export PYTHONPATH=.

# ============================
# Fixed experimental settings
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
echo " FINAL BOUNDARY STRUCTURE COMPARISON "
echo " batch=2 | aug=ON | sched=ON | e=80 "
echo "======================================"

# =====================================================
# EXP6: Baseline (NO boundary)
# =====================================================
echo "===== EXP6: BASELINE | no boundary ====="

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode none \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size $INPUT_SIZE \
  --batch_size $BATCH \
  --num_epochs $EPOCHS \
  --lr $LR \
  --val_every 1 \
  --use_wandb \
  --exp_tag exp6_baseline_b2_aug_sched \
> logs/exp6_baseline_b2_aug_sched.log 2>&1


# =====================================================
# EXP7: Dual-head boundary
# =====================================================
echo "===== EXP7: DUAL-HEAD boundary ====="

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode dual \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size $INPUT_SIZE \
  --batch_size $BATCH \
  --num_epochs $EPOCHS \
  --lr $LR \
  --val_every 1 \
  --use_wandb \
  --exp_tag exp7_dual_b2_aug_sched \
> logs/exp7_dual_b2_aug_sched.log 2>&1


# =====================================================
# EXP8: BASNet-like (boundary + refinement)
# =====================================================
echo "===== EXP8: BASNET boundary ====="

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
  --exp_tag exp8_basnet_b2_aug_sched \
> logs/exp8_basnet_b2_aug_sched.log 2>&1


# =====================================================
# EXP9: BASNet + Transformer mediator
# =====================================================
echo "===== EXP9: BASNET + TRANSFORMER ====="

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode basnet \
  --use_refinement \
  --use_transformer \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size $INPUT_SIZE \
  --batch_size $BATCH \
  --num_epochs $EPOCHS \
  --lr $LR \
  --val_every 1 \
  --use_wandb \
  --exp_tag exp9_basnet_trans_b2_aug_sched \
> logs/exp9_basnet_trans_b2_aug_sched.log 2>&1


echo "===== FINAL BOUNDARY EXPERIMENTS FINISHED ====="