import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.configs.config import IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND
from src.data.transforms import get_train_transform, get_valid_transform

class XRayDataset(Dataset):
    def __init__(self, is_train=True, split_file=None):
        
        # 이미지 및 라벨 불러오기
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
            for root, _dirs, files in os.walk(IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
            for root, _dirs, files in os.walk(LABEL_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        
        # 모든 .png 파일에 대해 .json 파일이 존재하는지 확인
        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
        
        pngs = sorted(pngs)
        jsons = sorted(jsons)
        
        if split_file is not None:
            with open(split_file, "r") as f:
                allowed_ids = set(line.strip() for line in f if line.strip())

            pngs = [p for p in pngs if os.path.basename(p) in allowed_ids]
            jsons = [
                j for j in jsons
                if os.path.basename(j).replace(".json", ".png") in allowed_ids
            ]
        
        self.filenames = np.array(pngs)
        self.labelnames = np.array(jsons)
        self.is_train = is_train
        self.transforms = get_train_transform() if self.is_train else get_valid_transform()
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
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

        # to tenser will be done later
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label