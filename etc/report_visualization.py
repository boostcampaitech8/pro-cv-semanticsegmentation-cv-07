import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.configs.defaults import CLASSES
from src.metrics.dice import dice_coef


# =========================
# 기본 유틸
# =========================

def load_model(ckpt_path):
    model = torch.load(ckpt_path, weights_only=False)
    model.eval()
    model.cuda()
    return model


@torch.no_grad()
def compute_classwise_dice(model, dataloader, thr=0.5, max_batches=5):
    dices = []

    for i, (images, masks) in enumerate(dataloader):
        if i >= max_batches:
            break

        images = images.cuda()
        masks = masks.cuda()

        outputs = model(images)
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

        outputs = torch.sigmoid(outputs) > thr
        dice = dice_coef(masks, outputs.float())
        dices.append(dice)

    dices = torch.cat(dices, dim=0)
    return dices.mean(dim=0)  # (C,)


def select_spike_classes(dice_prev, dice_spike, dice_next, topk=3):
    score = (
        torch.abs(dice_spike - dice_prev) +
        torch.abs(dice_spike - dice_next)
    )
    topk = torch.topk(score, k=topk)
    return topk.indices.tolist(), score


# =========================
# Figure 생성
# =========================

@torch.no_grad()
def save_comparison_figure(
    model_prev,
    model_spike,
    model_next,
    dataloader,
    class_idx,
    out_dir,
    num_samples=2,
    thr=0.5
):
    os.makedirs(out_dir, exist_ok=True)

    images, masks = next(iter(dataloader))
    images = images[:num_samples].cuda()
    masks = masks[:num_samples]

    outs = []
    for m in [model_prev, model_spike, model_next]:
        o = m(images)
        if o.shape[-2:] != masks.shape[-2:]:
            o = torch.nn.functional.interpolate(
                o, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
        outs.append(torch.sigmoid(o) > thr)

    outs = [o.cpu() for o in outs]

    for i in range(num_samples):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(masks[i, class_idx], cmap="gray")
        axes[0].set_title("GT")

        titles = ["Prev", "Spike", "Next"]
        for j in range(3):
            axes[j + 1].imshow(outs[j][i, class_idx], cmap="gray")
            axes[j + 1].set_title(titles[j])

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(
            out_dir, f"{CLASSES[class_idx]}_sample{i}.png"
        )
        plt.savefig(save_path, dpi=200)
        plt.close()


# =========================
# 메인 리포트 파이프라인
# =========================

def generate_spike_report(
    ckpt_prev,
    ckpt_spike,
    ckpt_next,
    val_dataset,
    out_root,
    batch_size=2,
    topk=3
):
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model_prev = load_model(ckpt_prev)
    model_spike = load_model(ckpt_spike)
    model_next = load_model(ckpt_next)

    dice_prev = compute_classwise_dice(model_prev, loader)
    dice_spike = compute_classwise_dice(model_spike, loader)
    dice_next = compute_classwise_dice(model_next, loader)

    cls_indices, score = select_spike_classes(
        dice_prev, dice_spike, dice_next, topk=topk
    )

    # =========================
    # 표 저장 (CSV)
    # =========================
    records = []
    for idx in cls_indices:
        records.append({
            "class": CLASSES[idx],
            "dice_prev": dice_prev[idx].item(),
            "dice_spike": dice_spike[idx].item(),
            "dice_next": dice_next[idx].item(),
            "delta_prev_spike": (dice_spike[idx] - dice_prev[idx]).item(),
            "delta_spike_next": (dice_next[idx] - dice_spike[idx]).item(),
            "score": score[idx].item()
        })

    df = pd.DataFrame(records)
    os.makedirs(out_root, exist_ok=True)
    df.to_csv(os.path.join(out_root, "spike_summary.csv"), index=False)

    # =========================
    # 이미지 저장
    # =========================
    for idx in cls_indices:
        save_comparison_figure(
            model_prev,
            model_spike,
            model_next,
            loader,
            idx,
            out_dir=os.path.join(out_root, CLASSES[idx])
        )

    del model_prev, model_spike, model_next
    torch.cuda.empty_cache()
