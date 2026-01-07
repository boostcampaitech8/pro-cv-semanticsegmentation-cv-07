import albumentations as A
import cv2

def get_train_transform(cfg):
    t = [A.Resize(cfg.img_size, cfg.img_size)]
    
    if cfg.use_scale:
        t.append(
            A.Affine(
                scale=(1 - 0.05, 1 + 0.05),
                translate_percent=None,
                rotate=0,
                shear=0,
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                cval_mask=0,
                p=0.8,
            )
        )

    if cfg.use_rotate:
        t.append(
            A.Affine(
                rotate=(-30, 30),
                translate_percent={"x": 0.0, "y": 0.15}, 
                scale=1.0,
                shear=0,
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                cval_mask=0,
                p=0.8,
            )
        )
        
    if cfg.use_flip:
        t.append(
            A.HorizontalFlip(p=0.5)
        )
    
    if cfg.use_contrast:
        t.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.0,
                contrast_limit=(0.1, 0.3),
                p=0.5
            )
        )
        
    return A.Compose(t)

def get_valid_transform(cfg):
    return A.Compose([
        A.Resize(cfg.img_size, cfg.img_size)
    ])

def get_test_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size)
    ])

def get_tta_transform():
    return A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(
            brightness_limit=0.0,
            contrast_limit=(0.1, 0.3),
            p=1.0
        )
    ])