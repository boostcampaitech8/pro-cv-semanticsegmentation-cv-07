# etc/visualization.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.configs.defaults import CLASSES, PALETTE
from src.metrics.dice import dice_coef


# ======================================================
# Common utils
# ======================================================

def label2rgb(label):
    """
    label: (C, H, W) binary mask
    """
    h, w = label.shape[1:]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for i, cls_mask in enumerate(label):
        canvas[cls_mask > 0] = PALETTE[i]

    return canvas


def load_model(ckpt_path):
    model = torch.load(ckpt_path, weights_only=False)
    model.eval()
    model.cuda()
    return model


@torch.no_grad()
def compute_classwise_dice(model, dataloader, thr=0.5, max_batches=5):
    """
    return: (num_classes,)
    """
    dices = []

    for i, (images, masks) in enumerate(dataloader):
        if i >= max_batches:
            break

        images = images.cuda()
        masks = masks.cuda()

        outputs = model(images)

        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = torch.nn.functional.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        outputs = torch.sigmoid(outputs) > thr
        dice = dice_coef(masks, outputs.float())
        dices.append(dice)

    dices = torch.cat(dices, dim=0)
    return dices.mean(dim=0)


def select_topk_changed_classes(dice_prev, dice_spike, dice_next, k=3):
    """
    spike 중심 변화량 기준으로 class 선택
    """
    score = (
        torch.abs(dice_spike - dice_prev) +
        torch.abs(dice_spike - dice_next)
    )
    topk = torch.topk(score, k=k)
    return topk.indices.tolist(), score


# ======================================================
# Visualization core
# ======================================================

@torch.no_grad()
def visualize_model_comparison(
    model_a,
    model_b,
    dataloader,
    class_indices,
    title_a="Model A",
    title_b="Model B",
    num_samples=2,
    thr=0.5,
    save_path=None
):
    images, masks = next(iter(dataloader))
    images = images[:num_samples].cuda()
    masks = masks[:num_samples]

    out_a = model_a(images)
    out_b = model_b(images)

    if out_a.shape[-2:] != masks.shape[-2:]:
        out_a = torch.nn.functional.interpolate(out_a, masks.shape[-2:])
        out_b = torch.nn.functional.interpolate(out_b, masks.shape[-2:])

    out_a = (torch.sigmoid(out_a) > thr).cpu()
    out_b = (torch.sigmoid(out_b) > thr).cpu()

    fig, axes = plt.subplots(
        len(class_indices),
        3,
        figsize=(12, 4 * len(class_indices))
    )

    if len(class_indices) == 1:
        axes = np.expand_dims(axes, 0)

    for r, cls_idx in enumerate(class_indices):
        cls_name = CLASSES[cls_idx]

        axes[r, 0].imshow(masks[0, cls_idx], cmap="gray")
        axes[r, 0].set_title(f"GT – {cls_name}")

        axes[r, 1].imshow(out_a[0, cls_idx], cmap="gray")
        axes[r, 1].set_title(title_a)

        axes[r, 2].imshow(out_b[0, cls_idx], cmap="gray")
        axes[r, 2].set_title(title_b)

        for c in range(3):
            axes[r, c].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


# ======================================================
# Analysis entry (experiment / spike)
# ======================================================

def analyze_spike(
    ckpt_prev,
    ckpt_spike,
    ckpt_next,
    val_dataset,
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

    cls_indices, score = select_topk_changed_classes(
        dice_prev, dice_spike, dice_next, k=topk
    )

    print("📈 Spike-dominant classes")
    for idx in cls_indices:
        print(
            f"{CLASSES[idx]:<12} | "
            f"{dice_prev[idx]:.4f} → {dice_spike[idx]:.4f} "
            f"(score={score[idx]:.4f})"
        )

    visualize_model_comparison(
        model_prev,
        model_spike,
        loader,
        cls_indices,
        title_a="Prev",
        title_b="Spike"
    )

    visualize_model_comparison(
        model_spike,
        model_next,
        loader,
        cls_indices,
        title_a="Spike",
        title_b="Next"
    )


# ======================================================
# Report entry (paper / presentation)
# ======================================================

@torch.no_grad()
def make_report_figures(
    ckpt,
    val_dataset,
    class_indices,
    save_dir,
    num_samples=3,
    thr=0.5
):
    os.makedirs(save_dir, exist_ok=True)

    loader = DataLoader(
        val_dataset,
        batch_size=num_samples,
        shuffle=False,
        num_workers=0
    )

    model = load_model(ckpt)
    images, masks = next(iter(loader))
    images = images.cuda()

    outputs = model(images)
    outputs = torch.sigmoid(outputs) > thr
    outputs = outputs.cpu()

    for cls_idx in class_indices:
        cls_name = CLASSES[cls_idx]

        fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

        for i in range(num_samples):
            axes[i, 0].imshow(masks[i, cls_idx], cmap="gray")
            axes[i, 0].set_title("GT")

            axes[i, 1].imshow(outputs[i, cls_idx], cmap="gray")
            axes[i, 1].set_title("Prediction")

            overlay = label2rgb(outputs[i])
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Overlay")

            for j in range(3):
                axes[i, j].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{cls_name}.png"), dpi=200)
        plt.close()