import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.configs.defaults import IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND
from src.data.transforms import get_train_transform, get_valid_transform

class XRayDataset(Dataset):
    def __init__(self, fold, is_train=True):
        self.is_train = is_train
        self.transforms = get_train_transform() if is_train else get_valid_transform()

        # 1) split json 로드
        split_path = "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/src/datasets/splits/splits_5fold_subject.json"
        with open(split_path) as f:
            splits = json.load(f)

        id_list = splits[f"fold_{fold}"]["train" if is_train else "val"]

        # 2) ID → PNG 2장씩 펼치기
        self.filenames = []
        self.labelnames = []

        for id_ in id_list:
            img_dir = os.path.join(IMAGE_ROOT, id_)
            lbl_dir = os.path.join(LABEL_ROOT, id_)

            pngs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            for png in pngs:
                self.filenames.append(os.path.join(id_, png))
                self.labelnames.append(os.path.join(id_, png.replace(".png", ".json")))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(IMAGE_ROOT, self.filenames[idx])
        label_path = os.path.join(LABEL_ROOT, self.labelnames[idx])

        image = cv2.imread(image_path)
        assert image is not None, f"Failed to read image: {image_path}"
        image = image / 255.0

        label = np.zeros((*image.shape[:2], len(CLASSES)), dtype=np.uint8)
        with open(label_path) as f:
            anns = json.load(f)["annotations"]

        for ann in anns:
            cls = CLASS2IND[ann["label"]]
            pts = np.array(ann["points"])
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            label[..., cls] = mask

        if self.transforms:
            out = self.transforms(image=image, mask=label)
            image, label = out["image"], out["mask"]

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        # to tenser will be done later
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label