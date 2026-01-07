import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.configs.defaults import CLASSES, CLASS2IND
from src.data.transforms import get_train_transform, get_valid_transform, get_tta_transform
from src.data.utils import load_HandBonesDataset, split_train_val

class XRayDataset(Dataset):
    def __init__(self, cfg, is_train=True):
        self.image_root = os.path.join(cfg.data_root, "DCM")
        self.label_root = os.path.join(cfg.data_root, "outputs_json")
        
        pngs, jsons = load_HandBonesDataset(self.image_root, self.label_root)
        
        if cfg.total:
            _filenames = np.array(pngs)
            _labelnames = np.array(jsons)
            
            filenames, labelnames = _filenames, _labelnames
        else:
            filenames, labelnames = split_train_val(cfg, pngs, jsons, is_train=is_train)
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = get_train_transform(cfg) if self.is_train else get_valid_transform(cfg)
        self.tta_transforms = get_tta_transform() if (not self.is_train) and cfg.tta else None
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
        
        if (not self.is_train) and self.tta_transforms is not None:
            tta_result = self.tta_transforms(image=image)
            tta_image = tta_result["image"]

            image = np.stack([image, tta_image], axis=0)
            label = np.stack([label, label], axis=0)
        
        if self.tta_transforms is None:
            image = image.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)
        else:
            image = image.transpose(0, 3, 1, 2)
            label = label.transpose(0, 3, 1, 2)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label