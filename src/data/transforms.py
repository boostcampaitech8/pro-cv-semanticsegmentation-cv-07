import albumentations as A
import cv2

def get_train_transform(
    img_size=512,
    use_rotate=False,
    rotate_limit=30,
    rotate_p=0.8,

    use_scale=False,
    scale_limit=0.05,
    scale_p=0.8,

    use_hflip=False,
    hflip_p=0.4,  

    fixed_shift_y=0.15,
):
    t = [A.Resize(img_size, img_size)]

    # Scale (rotate 여부와 무관)
    if use_scale:
        t.append(
            A.Affine(
                scale=(1 - scale_limit, 1 + scale_limit),
                translate_percent=None,
                rotate=0,
                shear=0,
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                cval_mask=0,
                p=scale_p,
            )
        )

    # Rotate + 고정 shift (rotate 발생 시에만)
    if use_rotate:
        t.append(
            A.Affine(
                rotate=(-rotate_limit, rotate_limit),   # ±30 랜덤
                translate_percent={"x": 0.0, "y": fixed_shift_y}, 
                scale=1.0,
                shear=0,
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                cval_mask=0,
                p=rotate_p,
            )
        )

    if use_hflip:
        t.append(
            A.HorizontalFlip(p=hflip_p)
        )

    return A.Compose(t)


def get_valid_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size)
    ])

def get_test_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size)
    ])