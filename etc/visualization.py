import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.configs.config import PALETTE, TEST_IMAGE_ROOT, CLASSES
from src.data.utils import decode_rle_to_mask


def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image


def check_data_sample(dataset, index):
    image, label = dataset[index]
    
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image[0])    # color map 적용을 위해 channel 차원을 생략합니다.
    ax[1].imshow(label2rgb(label))

    plt.show()


def result_visualization(rles, filename_and_class, index):
    image = cv2.imread(os.path.join(TEST_IMAGE_ROOT, filename_and_class[index].split("_")[1]))
    
    preds = []
    for rle in rles[:len(CLASSES)]:
        pred = decode_rle_to_mask(rle, height=2048, width=2048)
        preds.append(pred)

    preds = np.stack(preds, 0)
    
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image)    # remove channel dimension
    ax[1].imshow(label2rgb(preds))

    plt.show()