import argparse
import os
from src.configs.defaults import *
from src.configs.train_config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='unetpp')
    parser.add_argument("--encoder", type=str, default='resnet50')

    # ✅ 추가
    parser.add_argument("--fold", type=int, default=0)

    # 🔹 boundary switches
    parser.add_argument("--boundary_mode", type=str, default="none",
                        choices=["none", "dual", "basnet"])
    parser.add_argument("--use_refinement", action="store_true")
    parser.add_argument("--use_transformer", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--val_every", type=int, default=VAL_EVERY)
    parser.add_argument("--num_patience", type=int, default=NUM_PATIENCE)
    
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")

    parser.add_argument("--pretrained", type=str, default="imagenet", choices=["scratch", "imagenet", "radimagenet"], help="encoder pretrained weights")

    parser.add_argument("--no_boundary_detach", action="store_false", dest="boundary_detach", help="turn OFF boundary loss detaching (default: ON)")
    parser.set_defaults(boundary_detach=True)

    parser.add_argument("--loss_mode", type=str, default="bce_dice", choices=["bce", "bce_dice", "bce_dice_jaccard"],)

    return parser.parse_args()

def _flag_name(args):
    flags = []
    if args.use_refinement:
        flags.append("ref")
    if args.use_transformer:
        flags.append("trans")
    return "+".join(flags) if flags else "base"

def build_config(args):
    flag = _flag_name(args)
    return TrainConfig(
        model_name=args.model,
        encoder_name=args.encoder,
        save_name = (
            f"{args.model}_"
            f"{args.encoder}_"
            f"{args.boundary_mode}_{flag}_"
            f"{args.loss_mode}_"
            f"img{INPUT_SIZE}_"
            f"fold{args.fold}_best.pt"
        )
        ,

        pretrained=args.pretrained,

        boundary_mode=args.boundary_mode,
        use_refinement=args.use_refinement,
        use_transformer=args.use_transformer,
        boundary_detach=args.boundary_detach,
        loss_mode=args.loss_mode,

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