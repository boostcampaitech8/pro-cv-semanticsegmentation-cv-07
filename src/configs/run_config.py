import argparse
import os
from src.configs.defaults import *
from src.configs.train_config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser()

    # 경로 설정
    parser.add_argument("--data", type=str, default=TRIAN_ROOT)
    parser.add_argument("--saved_dir", type=str, default=SAVED_DIR)
    parser.add_argument("--saved_name", type=str, default="none")
    parser.add_argument("--split_file", type=str, default=SPLIT_FILE_ROOT)
    
    # 데이터 설정
    parser.add_argument("--total", action="store_true")   # no valid
    parser.add_argument("--tta", action="store_true")
    
    parser.add_argument("-img_size", type=int, default=2048)
    parser.add_argument("--use_scale", action="store_true")
    parser.add_argument("--use_rotate", action="store_true")
    parser.add_argument("--use_flip", action="store_true")
    parser.add_argument("--use_contrast", action="store_true")
    
    # 모델 설정
    parser.add_argument("--model", type=str, default='upernet')
    parser.add_argument("--encoder", type=str, default='resnext50_32x4d')
    parser.add_argument("--loss", type=str, default='BDJ')
    parser.add_argument("--scheduler", type=str, default="warmup")
    parser.add_argument("--optim", type=str, default="adam")
    
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--val_every", type=int, default=VAL_EVERY)
    parser.add_argument("--num_patience", type=int, default=NUM_PATIENCE)
    
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")

    return parser.parse_args()


def build_config(args):
    return TrainConfig(
        data_root = args.data,
        saved_root = args.saved_dir,
        saved_name=f"{args.model}_{args.encoder}_{args.loss}" if args.saved_name == "none" else args.saved_name,
        split_file_root = args.split_file,
        
        total=args.total,
        tta=args.tta,
        
        img_size=args.img_size,
        use_scale=args.use_scale,
        use_rotate=args.use_rotate,
        use_flip=args.use_flip,
        use_contrast=args.use_contrast,
        
        model_name=args.model,
        encoder_name=args.encoder,
        loss_type=args.loss,
        scheduler=args.scheduler,
        optimizer=args.optim,

        batch_size=args.batch_size,
        num_workers_train=NUM_WORKERS_TRAIN,
        num_workers_val=NUM_WORKERS_VAL,

        lr=args.lr,
        num_epochs=args.num_epochs,
        val_every=args.val_every,
        num_patience=args.num_patience,

        seed=RANDOM_SEED,
        
        use_wandb=args.use_wandb,
    )