import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.configs.defaults import CLASSES, PALETTE
from src.metrics.dice import dice_coef


# =========================
# 기본 유틸
# =========================

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


# =========================
# Dice 계산 (class-wise)
# =========================

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

        # 🔧 [ADD] prediction / GT 해상도 맞추기
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = torch.nn.functional.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > thr).float()

        dice = dice_coef(masks, outputs)  # (B, C)
        dices.append(dice)

    dices = torch.cat(dices, dim=0)      # (N, C)
    return dices.mean(dim=0)              # (C,)


# =========================
# Dice 변화 큰 class 자동 선택
# =========================

def select_topk_changed_classes(dice_prev, dice_spike, dice_next, k=3):
    score = (
        torch.abs(dice_spike - dice_prev) +
        torch.abs(dice_spike - dice_next)
    )
    topk = torch.topk(score, k=k)
    return topk.indices.tolist(), score


# =========================
# 시각화
# =========================

@torch.no_grad()
def visualize_model_comparison(
    model_a,
    model_b,
    dataloader,
    class_indices,
    title_a="Model A",
    title_b="Model B",
    num_samples=2,
    thr=0.5
):
    """
    두 모델을 같은 이미지에서 비교
    """
    images, masks = next(iter(dataloader))
    images = images[:num_samples].cuda()
    masks = masks[:num_samples]

    out_a = model_a(images)
    out_b = model_b(images)

    # 🔧 해상도 맞추기
    if out_a.shape[-2:] != masks.shape[-2:]:
        out_a = torch.nn.functional.interpolate(
            out_a, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )
        out_b = torch.nn.functional.interpolate(
            out_b, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

    out_a = torch.sigmoid(out_a) > thr
    out_b = torch.sigmoid(out_b) > thr


    out_a = out_a.cpu()
    out_b = out_b.cpu()

    for i in range(num_samples):
        fig, axes = plt.subplots(
            len(class_indices), 3,
            figsize=(12, 4 * len(class_indices))
        )

        if len(class_indices) == 1:
            axes = np.expand_dims(axes, 0)

        for r, cls_idx in enumerate(class_indices):
            cls_name = CLASSES[cls_idx]

            gt = masks[i, cls_idx].numpy()
            pa = out_a[i, cls_idx].numpy()
            pb = out_b[i, cls_idx].numpy()

            axes[r, 0].imshow(gt, cmap="gray")
            axes[r, 0].set_title(f"GT – {cls_name}")

            axes[r, 1].imshow(pa, cmap="gray")
            axes[r, 1].set_title(title_a)

            axes[r, 2].imshow(pb, cmap="gray")
            axes[r, 2].set_title(title_b)

            for c in range(3):
                axes[r, c].axis("off")

        plt.tight_layout()
        plt.show()


# =========================
# 메인 파이프라인
# =========================

def analyze_spike(
    ckpt_prev,
    ckpt_spike,
    ckpt_next,
    val_dataset,
    batch_size=2,
    topk=3
):
    """
    prev → spike → next 비교
    """
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

    delta = dice_spike - dice_prev

    print("📈 Spike-dominant classes")
    for idx in cls_indices:
        print(
            f"{CLASSES[idx]:<12} | "
            f"prev={dice_prev[idx]:.4f} → spike={dice_spike[idx]:.4f} "
            f"(Δ={delta[idx]:+.4f}, score={score[idx]:.4f})"
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

    del model_prev, model_spike, model_next
    torch.cuda.empty_cache()