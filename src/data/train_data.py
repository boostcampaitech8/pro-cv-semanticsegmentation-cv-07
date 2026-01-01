# import os
# import json
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from src.configs.config import IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND
# from src.data.transforms import get_train_transform, get_valid_transform
# from src.data.utils import split_train_val
# class XRayDataset(Dataset):
#     def __init__(self, fold, is_train=True):
#         self.is_train = is_train
#         self.transforms = get_train_transform() if is_train else get_valid_transform()

#         # 🔹 split txt 그대로 사용
#         split_dir = os.path.join("src/datasets/splits")
#         split_file = f"fold_{fold}_train.txt" if is_train else f"fold_{fold}_val.txt"
#         split_path = os.path.join(split_dir, split_file)

#         with open(split_path, "r") as f:
#             self.filenames = [line.strip() for line in f.readlines()]
#         self.labelnames = [f.replace(".png", ".json") for f in self.filenames]

        
#         self.image_index = {}
#         for root, _, files in os.walk(IMAGE_ROOT):
#             for fname in files:
#                 if fname.endswith(".png"):
#                     self.image_index[fname] = os.path.join(root, fname)

#         self.label_index = {}
#         for root, _, files in os.walk(LABEL_ROOT):
#             for fname in files:
#                 if fname.endswith(".json"):
#                     self.label_index[fname] = os.path.join(root, fname)

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         fname = self.filenames[idx]
#         lname = self.labelnames[idx]

#         image_path = self.image_index.get(fname)
#         label_path = self.label_index.get(lname)

#         if image_path is None:
#             raise FileNotFoundError(f"Image not found: {fname}")
#         if label_path is None:
#             raise FileNotFoundError(f"Label not found: {lname}")

#         image = cv2.imread(image_path)
#         if image is None:
#             raise RuntimeError(f"Failed to read image: {image_path}")
#         image = image / 255.0

#         label = np.zeros((*image.shape[:2], len(CLASSES)), dtype=np.uint8)
#         with open(label_path) as f:
#             anns = json.load(f)["annotations"]

#         for ann in anns:
#             cls = CLASS2IND[ann["label"]]
#             pts = np.array(ann["points"])
#             mask = np.zeros(image.shape[:2], np.uint8)
#             cv2.fillPoly(mask, [pts], 1)
#             label[..., cls] = mask

#         if self.transforms:
#             out = self.transforms(image=image, mask=label)
#             image, label = out["image"], out["mask"]

#         return (
#             torch.from_numpy(image.transpose(2, 0, 1)).float(),
#             torch.from_numpy(label.transpose(2, 0, 1)).float()
#         )
import os
import re
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

from src.configs.config import IMAGE_ROOT, LABEL_ROOT, CLASSES, CLASS2IND
from src.data.transforms import get_train_transform, get_valid_transform

def _extract_id_from_relpath(relpath: str):
    """
    Extract patient ID from relative path.
    Example: 'ID001/image....png' -> '001'
    """
    m = re.search(r'ID(\d{3})', relpath)
    return m.group(1) if m else None

class XRayDataset(Dataset):
    def __init__(
        self, 
        image_root=IMAGE_ROOT, 
        label_root=LABEL_ROOT, 
        is_train=True, 
        folds_json=None, 
        fold=0,
        img_size=512,use_rotate=False,use_scale=False,use_hflip=False,use_contrast=False
    ):
        self.image_root = image_root
        self.label_root = label_root
        self.is_train = is_train

        # EXCLUDE_IDS = ["487", "363"]
        
       
        self.transforms = get_train_transform(img_size,use_rotate=use_rotate,use_scale=use_scale,use_hflip=use_hflip,use_contrast=use_contrast) if is_train else get_valid_transform(img_size)
        

        pngs = sorted([
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        ])

        json_set = {
            os.path.relpath(os.path.join(root, fname), start=self.label_root)
            for root, _dirs, files in os.walk(self.label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }

        pairs = []
        for img_rel in pngs:
            lbl_rel = os.path.splitext(img_rel)[0] + ".json"
            if lbl_rel in json_set:
                pid = _extract_id_from_relpath(img_rel)
                # if pid not in EXCLUDE_IDS:
                pairs.append((img_rel, lbl_rel))

        if len(pairs) == 0:
            raise ValueError(f"No matched (png,json) pairs found. image_root={self.image_root}")

        if folds_json is not None:
            print(f"Loading folds from {folds_json} (Fold: {fold})")
            with open(folds_json, "r", encoding="utf-8") as f:
                folds = json.load(f)

            fold_key = f"fold_{fold}"
            if fold_key not in folds:
                raise ValueError(f"{fold_key} not found in {folds_json}")

            fold_info = folds[fold_key]

            # your json keys: "train", "val"
            train_ids = set(fold_info["train"])   # e.g. {"ID001", "ID002", ...}
            val_ids   = set(fold_info["val"])     # e.g. {"ID123", ...}
            target_ids = train_ids if is_train else val_ids

            filtered = []
            for img_rel, lbl_rel in pairs:
                pid = _extract_id_from_relpath(img_rel)  # "001"
                if pid is None:
                    continue

                pid_key = f"ID{pid}"  # "ID001"로 맞추기
                if pid_key in target_ids:
                    filtered.append((img_rel, lbl_rel))

            pairs = filtered

            if len(pairs) == 0:
                raise ValueError(
                    f"After fold filtering, no samples left. "
                    f"is_train={is_train}, fold_idx={fold}"
                )

            self.filenames = [p[0] for p in pairs]
            self.labelnames = [p[1] for p in pairs]


        else:
            print(f"No Fold, Random Split")
            _filenames = np.array([p[0] for p in pairs])
            _labelnames = np.array([p[1] for p in pairs])

            groups = [os.path.dirname(fname) for fname in _filenames]
            ys = [0 for _ in _filenames]
            gkf = GroupKFold(n_splits=5)

            filenames = []
            labelnames = []

            for i, (_train_idx, valid_idx) in enumerate(gkf.split(_filenames, ys, groups)):
                if is_train:
                    if i == 0: # Skip the first fold for training (it's validation)
                        continue
                    filenames += list(_filenames[valid_idx])
                    labelnames += list(_labelnames[valid_idx])
                else:
                    # Use the first fold for validation
                    filenames = list(_filenames[valid_idx])
                    labelnames = list(_labelnames[valid_idx])
                    break
            
            self.filenames = filenames
            self.labelnames = labelnames
        print("len(filenames), len(labelnames)", len(self.filenames), len(self.labelnames))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"]

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label