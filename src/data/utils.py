import os
import numpy as np
from sklearn.model_selection import GroupKFold
from pathlib import Path


def load_HandBonesDataset(image_root, label_root):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=image_root)
        for root, _dirs, files in os.walk(image_root)
        # if not is_excluded(root, image_root, EXCLUDE_IMAGE_DIRS)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_root)
        for root, _dirs, files in os.walk(label_root)
        # if not is_excluded(root, label_root, EXCLUDE_LABEL_DIRS)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
        
    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
        
    pngs = sorted(pngs)
    jsons = sorted(jsons)
    
    return pngs, jsons


def remove_noise(pngs, jsons):
    exclude_pngs = [
        "ID363/image1664935962797.png",
        "ID487/image1666661955150.png"
    ]
    exclude_jsons = [
        "ID363/image1664935962797.json",
        "ID487/image1666661955150.json"
    ]

    filenames = np.array(pngs)
    labelnames = np.array(jsons)
    
    mask_png = np.array([f not in exclude_pngs for f in filenames])
    mask_json = np.array([f not in exclude_jsons for f in labelnames])
    
    pngs = list(filenames[mask_png])
    jsons = list(labelnames[mask_json])
    
    return pngs, jsons


def split_train_val(cfg, pngs, jsons, is_train=True):
    
    # 파일 명이 random일 시, 5-fold random split 적용 
    if cfg.split_file_root == "random":
        pngs, jsons = remove_noise(pngs, jsons)
        
        filenames = np.array(pngs)
        labelnames = np.array(jsons)
        
        groups = [os.path.dirname(f) for f in filenames]
        ys = [0] * len(filenames)
    
        gkf = GroupKFold(n_splits=5)
        
        files, labels = [], []
        for i, (x, y) in enumerate(gkf.split(filenames, ys, groups)):
            if is_train:
                if i == 0:
                    continue
                    
                files += list(filenames[y])
                labels += list(labelnames[y])
            
            else:
                files = list(filenames[y])
                labels = list(labelnames[y])
                break
    
    # 파일 명이 실제 spilt file인 경우, 해당 파일 활용
    else:
        base = Path(cfg.split_file_root)

        if is_train:
            split_file = base.with_name(base.name + "_train.txt")
        else:
            split_file = base.with_name(base.name + "_val.txt")       
        
        with open(split_file, "r") as f:
            allowed_ids = set(line.strip() for line in f if line.strip())

        pngs = [p for p in pngs if os.path.basename(p) in allowed_ids]
        jsons = [
            j for j in jsons
            if os.path.basename(j).replace(".json", ".png") in allowed_ids
        ]
        
        files = np.array(pngs)
        labels = np.array(jsons)

    return files, labels


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)