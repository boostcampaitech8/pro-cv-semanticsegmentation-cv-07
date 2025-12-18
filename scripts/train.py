from src.data.train_data import XRayDataset
from src.configs.config import BATCH_SIZE, CLASSES, LR, SAVED_DIR, RANDOM_SEED
from src.utils.set_seed import set_seed
from src.engine.trainer import train
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models


def main():
    set_seed(RANDOM_SEED)
    
    if not os.path.exists(SAVED_DIR):                                                           
        os.makedirs(SAVED_DIR)
    
    train_dataset = XRayDataset(is_train=True)
    valid_dataset = XRayDataset(is_train=False)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

    train(model, train_loader, valid_loader, criterion, optimizer, save_file_name="fcn_resnet50_best_model.pt")


if __name__ == '__main__':
    main()