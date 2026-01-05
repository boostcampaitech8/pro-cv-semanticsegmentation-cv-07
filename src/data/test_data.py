import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.configs.defaults import TEST_IMAGE_ROOT
from .transforms import  get_test_transform, get_tta_transform


class XRayInferenceDataset(Dataset):
    def __init__(self, tta=False):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=TEST_IMAGE_ROOT)
            for root, _dirs, files in os.walk(TEST_IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = get_test_transform()
        self.tta_transforms = get_tta_transform() if tta else None
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(TEST_IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
        
        if self.tta_transforms is not None:
            tta_result = self.tta_transforms(image=image)
            tta_image = tta_result["image"]

            image = np.stack([image, tta_image], axis=0)

        if self.tta_transforms is None:
            image = image.transpose(2, 0, 1)
        else:
            image = image.transpose(0, 3, 1, 2)
        
        image = torch.from_numpy(image).float()
            
        return image, image_name