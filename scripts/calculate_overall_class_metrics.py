
# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# visualization
import matplotlib.pyplot as plt

def calculate_overall_class_metrics(model, dataset, thr=0.5, class_names=None):
    """전체 validation set에 대한 클래스별 semantic segmentation 메트릭 계산"""
    model.eval()
    
    # 첫 번째 이미지로 클래스 수 확인
    _, gt_sample = dataset[0]
    n_classes = gt_sample.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # 클래스별 메트릭 누적을 위한 딕셔너리
    class_stats = {name: {
        'total_intersection': 0,
        'total_union': 0,
        'total_gt_pixels': 0,
        'total_pred_pixels': 0,
        'total_pixels': 0,
        'total_correct_pixels': 0,
        'ious': [],
        'dices': [],
        'pixel_accs': [],
    } for name in class_names}
    
    print(f"Processing {len(dataset)} images...")
    
    for idx in range(len(dataset)):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} images...")
        
        img, gt = dataset[idx]
        img_input = img.unsqueeze(0).cuda()
        
        with torch.no_grad():
            outputs = model(img_input)['out']
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
        
        pred = outputs.squeeze(0)
        gt_np = gt.numpy()
        
        for c in range(n_classes):
            # Intersection, Union 누적
            intersection = np.logical_and(gt_np[c], pred[c]).sum()
            union = np.logical_or(gt_np[c], pred[c]).sum()
            
            class_stats[class_names[c]]['total_intersection'] += int(intersection)
            class_stats[class_names[c]]['total_union'] += int(union)
            
            # 픽셀 수 누적
            gt_pixels = gt_np[c].sum()
            pred_pixels = pred[c].sum()
            total_pixels = gt_np[c].size
            correct_pixels = (gt_np[c] == pred[c]).sum()
            
            class_stats[class_names[c]]['total_gt_pixels'] += int(gt_pixels)
            class_stats[class_names[c]]['total_pred_pixels'] += int(pred_pixels)
            class_stats[class_names[c]]['total_pixels'] += int(total_pixels)
            class_stats[class_names[c]]['total_correct_pixels'] += int(correct_pixels)
            
            # 이미지별 IoU, Dice, Pixel Accuracy 저장
            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / (gt_pixels + pred_pixels + 1e-8)
            pixel_acc = correct_pixels / total_pixels
            
            class_stats[class_names[c]]['ious'].append(iou)
            class_stats[class_names[c]]['dices'].append(dice)
            class_stats[class_names[c]]['pixel_accs'].append(pixel_acc)
    
    print(f"Completed processing {len(dataset)} images!")
    
    # 최종 메트릭 계산
    final_metrics = {}
    
    for class_name, stats in class_stats.items():
        # 전체 데이터에 대한 IoU, Dice (누적된 값으로 계산)
        global_iou = stats['total_intersection'] / (stats['total_union'] + 1e-8)
        global_dice = 2 * stats['total_intersection'] / (stats['total_gt_pixels'] + stats['total_pred_pixels'] + 1e-8)
        global_pixel_acc = stats['total_correct_pixels'] / stats['total_pixels']
        
        
        pixel_diff = stats['total_pred_pixels'] - stats['total_gt_pixels']
        pixel_diff_ratio = pixel_diff / (stats['total_gt_pixels'] + 1e-8)
        
        
        mean_iou_per_image = np.mean(stats['ious'])
        std_iou_per_image = np.std(stats['ious'])
        mean_dice_per_image = np.mean(stats['dices'])
        std_dice_per_image = np.std(stats['dices'])
        mean_pixel_acc_per_image = np.mean(stats['pixel_accs'])
        
        final_metrics[class_name] = {
            # 전체 누적 메트릭 (주요)
            'global_iou': global_iou,
            'global_dice': global_dice,
            'global_pixel_acc': global_pixel_acc,
            
            # 픽셀 통계
            'total_gt_pixels': stats['total_gt_pixels'],
            'total_pred_pixels': stats['total_pred_pixels'],
            'pixel_diff': pixel_diff,
            'pixel_diff_ratio': pixel_diff_ratio,
            
            # 이미지별 평균 (참고용)
            'mean_iou_per_image': mean_iou_per_image,
            'std_iou_per_image': std_iou_per_image,
            'mean_dice_per_image': mean_dice_per_image,
            'std_dice_per_image': std_dice_per_image,
            'mean_pixel_acc_per_image': mean_pixel_acc_per_image,
            
            # 추가 통계
            'num_images': len(dataset),
        }
    
    return final_metrics