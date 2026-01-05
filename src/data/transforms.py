# src/data/transforms.py

import albumentations as A
import cv2
from src.configs.defaults import INPUT_SIZE


def get_train_transform(size=INPUT_SIZE):
    return A.Compose([
        A.Resize(size, size),

        # 🔹 Scale & slight affine (전체 구조 유지)
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=None,
            rotate=0,
            shear=0,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            # cval=0,
            # cval_mask=0,
            value=0,
            mask_value=0,
            p=0.8,
        ),

        # 🔹 Rotation + Y-translation (손가락 끝 / carpal에 중요)
        A.Affine(
            rotate=(-30, 30),
            translate_percent={"x": 0.0, "y": 0.15},
            scale=1.0,
            shear=0,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            # cval=0,
            # cval_mask=0,
            value=0,
            mask_value=0,
            p=0.8,
        ),

        # 🔹 좌우 반전
        A.HorizontalFlip(p=0.5),

        # 🔹 X-ray 대비 강화
        A.RandomBrightnessContrast(
            brightness_limit=0.0,
            contrast_limit=(0.1, 0.3),
            p=0.5
        ),
    ])


def get_valid_transform(size=INPUT_SIZE):
    return A.Compose([
        A.Resize(size, size),
    ])


def get_test_transform(size=INPUT_SIZE):
    return A.Compose([
        A.Resize(size, size),
    ])