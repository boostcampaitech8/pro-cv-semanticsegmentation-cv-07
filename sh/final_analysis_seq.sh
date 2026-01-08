#!/bin/bash
export PYTHONPATH=.

# ============================
# Common fixed settings
# ============================
MODEL=unetpp
ENCODER=resnext50_32x4d
BOUNDARY=basnet
PRETRAIN=imagenet
FOLD=0
LOSS=bce_dice
EPOCHS=40
LR=1e-4

mkdir -p logs

# =====================================================
# EXP 1: 2048 / batch=2 / NO aug / NO scheduler
# =====================================================
echo "===== EXP1: 2048 | B2 | no aug | no sched ====="

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode $BOUNDARY \
  --use_refinement \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size 2048 \
  --batch_size 2 \
  --num_epochs $EPOCHS \
  --no_sched \
  --val_every 2 \
  --no_aug \
  --use_wandb \
  --exp_tag exp1_b2_noaug_nosched \
> logs/exp1_2048_b2_noaug_nosched.log 2>&1


# =====================================================
# EXP 2: 2048 / batch=1 / NO aug / NO scheduler
# =====================================================
echo "===== EXP2: 2048 | B1 | no aug | no sched ====="

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode $BOUNDARY \
  --use_refinement \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size 2048 \
  --batch_size 1 \
  --num_epochs $EPOCHS \
  --no_sched \
  --val_every 2 \
  --no_aug \
  --use_wandb \
  --exp_tag exp2_b1_noaug_nosched \
> logs/exp2_2048_b1_noaug_nosched.log 2>&1


# =====================================================
# EXP 3: 2048 / batch=2 / AUG ON / NO scheduler
# =====================================================
echo "===== EXP3: 2048 | B2 | aug ON | no sched ====="

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode $BOUNDARY \
  --use_refinement \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size 2048 \
  --batch_size 2 \
  --num_epochs $EPOCHS \
  --no_sched \
  --val_every 2 \
  --use_wandb \
  --exp_tag exp3_b2_aug_nosched \
> logs/exp3_2048_b2_aug_nosched.log 2>&1


# =====================================================
# EXP 4: 2048 / batch=2 / NO aug / scheduler ON
# =====================================================
echo "===== EXP4: 2048 | B2 | no aug | sched ON ====="

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode $BOUNDARY \
  --use_refinement \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size 2048 \
  --batch_size 2 \
  --num_epochs $EPOCHS \
  --val_every 1 \
  --no_aug \
  --use_wandb \
  --exp_tag exp4_b2_noaug_sched \
> logs/exp4_2048_b2_noaug_sched.log 2>&1


# =====================================================
# EXP 5: 2048 / batch=2 / AUG ON / scheduler ON (BROKEN CASE)
# =====================================================
echo "===== EXP5: 2048 | B2 | aug ON | sched ON ====="

python scripts/train.py \
  --model $MODEL \
  --encoder $ENCODER \
  --boundary_mode $BOUNDARY \
  --use_refinement \
  --pretrained $PRETRAIN \
  --loss_mode $LOSS \
  --fold $FOLD \
  --input_size 2048 \
  --batch_size 2 \
  --num_epochs $EPOCHS \
  --val_every 1 \
  --use_wandb \
  --exp_tag exp5_b2_aug_sched \
> logs/exp5_2048_b2_aug_sched.log 2>&1

echo "===== ALL EXPERIMENTS FINISHED ====="