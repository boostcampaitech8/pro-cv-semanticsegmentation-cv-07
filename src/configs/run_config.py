import argparse
import os
from src.configs.defaults import *
from src.configs.train_config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--val_every", type=int, default=VAL_EVERY)
    parser.add_argument("--num_patience", type=int, default=NUM_PATIENCE)
    
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")

    return parser.parse_args()

def build_config(args):
    return TrainConfig(
        model_name=args.model,
        save_name=f"{args.model}_best.pt",

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