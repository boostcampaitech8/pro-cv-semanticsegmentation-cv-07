"""
Semantic Segmentation Ensemble Script (Optimized for Multi-Scale & Multi-Loader TTA)
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.transforms import get_test_transform
from src.data.test_data import XRayInferenceDataset
from src.data.utils import encode_mask_to_rle
from src.configs.config import IND2CLASS, RANDOM_SEED, CLASSES
from src.utils.set_seed import set_seed

# Swin-Unet Imports
import copy
try:
    from external.Swin_Unet.networks.vision_transformer import SwinUnet
    from external.Swin_Unet.config import _C as swin_config
    SWIN_UNET_AVAILABLE = True
except ImportError:
    SWIN_UNET_AVAILABLE = False
    print("[WARNING] Swin-Unet not available.")


def normalize_weights(scores, min_val=0.5, max_val=1.0):
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    else:
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        adjusted = [min_val + (max_val - min_val) * w for w in normalized]
        return adjusted


def get_model_type(model_path):
    filename = os.path.basename(model_path).lower()
    if 'hrnet' in filename:
        return 'hrnet'
    elif 'swin_unet' in filename or 'swinunet' in filename:
        return 'swin_unet'
    elif 'unetpp' in filename or 'unet++' in filename:
        return 'unetpp'
    else:
        return 'default'


def load_swin_unet_model(checkpoint_path, img_size=1024):
    from external.Swin_Unet.networks.vision_transformer import SwinUnet
    from external.Swin_Unet.config import _C as swin_config
    import copy

    # 1. Config 설정
    config = copy.deepcopy(swin_config)
    # 클론한 경로 내의 yaml 파일 로드
    config.merge_from_file("external/Swin_Unet/configs/swin_tiny_patch4_window7_224_lite.yaml")
    
    # 2. 파라미터 덮어쓰기 (팀원 가이드 반영)
    config.DATA.NUM_CLASSES = len(CLASSES)
    config.DATA.IMG_SIZE = img_size
    config.MODEL.SWIN.WINDOW_SIZE = 16 
    
    # 3. 모델 초기화
    model = SwinUnet(config, num_classes=len(CLASSES))
    
    # 4. 가중치 로드
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # 가중치 저장 방식에 따른 분기 처리 (안전장치)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    
    return model

def load_models(model_configs):
    models = []
    model_types = []
    img_sizes = []
    
    print(f"\n{'='*20} 1. Model Loading {'='*20}")
    for path, size in model_configs:
        model_type = get_model_type(path)
        print(f"[*] Loading: {os.path.basename(path)}")
        print(f"    - Type: {model_type} | Target Size: {size}")
        
        if model_type == 'swin_unet':
            model = load_swin_unet_model(path, img_size=size)
        else:
            model = torch.load(path, map_location='cuda', weights_only=False)
        
        model = model.cuda()
        model.eval()
        
        # 모델 파라미터 수 디버깅
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    - Status: OK | Params: {total_params:,}")
        
        models.append(model)
        model_types.append(model_type)
        img_sizes.append(size)
        
    print(f"[+] Total {len(models)} models loaded.\n")
    return models, model_types, img_sizes


def get_raw_output(model, images, model_type='default'):
    if model_type == 'hrnet':
        outputs = model(images, mode='tensor')
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]
    else:  
        outputs = model(images)

        if isinstance(outputs, dict):
                outputs = outputs["seg"]

    return outputs


def ensemble_soft_voting(models, model_types, model_img_sizes, dataloaders_map, weights, thr_list=None, use_flip_tta=True):
    rles = []
    filename_and_class = []
    
    weights_tensor = torch.tensor(weights, device='cuda', dtype=torch.float32)
    weights_tensor = weights_tensor / weights_tensor.sum()

    print(thr_list)
    if thr_list is None:
        thr_tensor = 0.5
    elif len(thr_list) == 1:
        thr_tensor = thr_list[0]
    else:
        # 리스트 -> 텐서 -> GPU 이동 -> 차원 변경 (Broadcasting용)
        thr_tensor = torch.tensor(thr_list, device='cuda', dtype=torch.float32).view(1, -1, 1, 1)
        print(f"[Info] Class-specific thresholds applied. Shape: {thr_tensor.shape}")
    
    loader_iters_map = {size: [iter(l) for l in loaders] for size, loaders in dataloaders_map.items()}
    first_size = list(dataloaders_map.keys())[0]
    total_steps = len(dataloaders_map[first_size][0])
    
    print(f"\n{'='*20} 3. Starting Soft Voting Ensemble {'='*20}")
    
    with torch.no_grad():
        with autocast(enabled=False):
            for step in tqdm(range(total_steps), desc="Soft Voting"):
                
                # 배치를 미리 다 가져옴
                current_batches = {}
                image_names = None
                for size, iters in loader_iters_map.items():
                    current_batches[size] = []
                    for it in iters:
                        images, names = next(it)
                        current_batches[size].append(images.cuda())
                        if image_names is None: image_names = names

                # --- [DEBUG] Step 0에서 구조 검증 ---
                if step == 0:
                    print(f"\n\n[DEBUG Step 0] Data Structure Verification")
                    for size, batch_list in current_batches.items():
                        print(f"    - Size {size}: {len(batch_list)} TTA views fetched.")
                    print(f"    - Processing Images: {image_names}\n")

                all_probs = []
                for i, model in enumerate(models):
                    m_type = model_types[i]
                    m_size = model_img_sizes[i]
                    input_images_list = current_batches[m_size]
                    
                    model_logit_sum = None
                    count = 0
                    
                    # --- [DEBUG] 모델별 입력 검증 ---
                    if step == 0:
                        print(f"    [Model {i}] Input Size: {m_size} | Type: {m_type}")

                    for v_idx, images in enumerate(input_images_list):
                        # Mean값 출력을 통한 실제 데이터 변이 확인
                        if step == 0:
                            print(f"      - View {v_idx} Mean: {images.mean().item():.4f}", end="")

                        output = get_raw_output(model, images, m_type)
                        if model_logit_sum is None: model_logit_sum = output
                        else: model_logit_sum += output
                        count += 1
                        
                        if use_flip_tta:
                            images_f = torch.flip(images, dims=[-1])
                            output_f = get_raw_output(model, images_f, m_type)
                            output_f = torch.flip(output_f, dims=[-1])
                            model_logit_sum += output_f
                            count += 1
                        
                        if step == 0: print(f" (Sub-total count: {count})")

                    model_avg_logit = model_logit_sum / count
                    model_avg_logit = F.interpolate(model_avg_logit, size=(2048, 2048), mode="bilinear", align_corners=False)
                    probs = torch.sigmoid(model_avg_logit)
                    all_probs.append(probs)
                
                if step == 0:
                    print(f"\n    [DEBUG] All {len(models)} models prediction completed.")
                    print(f"    [DEBUG] Proceeding to Weighted Average...\n")

                all_probs = torch.stack(all_probs, dim=0)
                weights_expanded = weights_tensor.view(-1, 1, 1, 1, 1)
                weighted_probs = (all_probs * weights_expanded).sum(dim=0)
                final_preds = (weighted_probs > thr_tensor).cpu().numpy() 
                
                for output, image_name in zip(final_preds, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
    
    return rles, filename_and_class


def ensemble_hard_voting(models, model_types, model_img_sizes, dataloaders_map, thr=0.5, use_flip_tta=True):
    """
    Hard voting implementation with Enhanced Debugging.
    """
    rles = []
    filename_and_class = []
    
    # 사이즈별 로더 Iterator 생성
    loader_iters_map = {size: [iter(l) for l in loaders] for size, loaders in dataloaders_map.items()}

    first_size = list(dataloaders_map.keys())[0]
    total_steps = len(dataloaders_map[first_size][0])
    
    print(f"\n{'='*20} 3. Starting Hard Voting Ensemble {'='*20}")
    
    with torch.no_grad():
        with autocast():
            for step in tqdm(range(total_steps), desc="Hard Voting"):
                
                # 1. 모든 해상도의 TTA 배치를 미리 가져옴
                current_batches = {}
                image_names = None
                for size, iters in loader_iters_map.items():
                    current_batches[size] = []
                    for it in iters:
                        images, names = next(it)
                        if image_names is not None and names != image_names:
                            raise ValueError("DataLoader sync error: Image names do not match!")
                        current_batches[size].append(images.cuda())
                        if image_names is None: image_names = names

                # --- [DEBUG] Step 0 구조 검증 ---
                if step == 0:
                    print(f"\n\n[DEBUG Step 0] Data Structure Verification")
                    for size, batch_list in current_batches.items():
                        print(f"    - Size {size}: {len(batch_list)} TTA views fetched.")
                    print(f"    - Processing Images: {image_names}\n")

                all_preds = []
                for i, model in enumerate(models):
                    m_type = model_types[i]
                    m_size = model_img_sizes[i]
                    input_images_list = current_batches[m_size]

                    model_logit_sum = None
                    count = 0
                    
                    # --- [DEBUG] 모델별 입력 검증 ---
                    if step == 0:
                        print(f"    [Model {i}] Input Size: {m_size} | Type: {m_type}")

                    for v_idx, images in enumerate(input_images_list):
                        # Contrast 적용 여부 확인을 위한 Mean 출력
                        if step == 0:
                            print(f"      - View {v_idx} Mean: {images.mean().item():.4f}", end="")

                        output = get_raw_output(model, images, m_type)
                        if model_logit_sum is None: model_logit_sum = output
                        else: model_logit_sum += output
                        count += 1
                        
                        if use_flip_tta:
                            images_f = torch.flip(images, dims=[-1])
                            output_f = get_raw_output(model, images_f, m_type)
                            output_f = torch.flip(output_f, dims=[-1])
                            model_logit_sum += output_f
                            count += 1
                            
                        if step == 0: print(f" (Sub-total count: {count})")
                    
                    # TTA 평균 계산 후 Threshold 적용하여 Binary Mask(투표권) 생성
                    model_avg_logit = model_logit_sum / count
                    model_avg_logit = F.interpolate(model_avg_logit, size=(2048, 2048), mode="bilinear", align_corners=False)
                    
                    # Hard Voting을 위한 개별 모델의 최종 예측 (0 or 1)
                    preds = (torch.sigmoid(model_avg_logit) > thr).float()
                    
                    # --- [DEBUG] 모델별 투표 성향 확인 ---
                    if step == 0:
                        pos_pixels = preds.sum().item()
                        total_pixels = preds.numel()
                        print(f"      -> Individual Model Positive Pixel Ratio: {(pos_pixels/total_pixels)*100:.4f}%")
                    
                    all_preds.append(preds)
                
                # 2. 투표 집계 (Stacking & Sum)
                all_preds = torch.stack(all_preds, dim=0) # [num_models, B, C, H, W]
                vote_sum = all_preds.sum(dim=0)           # 픽셀당 득표 수
                majority_threshold = len(models) / 2.0    # 과반수 기준
                
                # 과반수 이상 득표한 픽셀만 1로 결정 ########################### 과반수 기준이므로 모델이 짝수일때 조절 필요
                final_preds = (vote_sum > majority_threshold).cpu().numpy()
                
                if step == 0:
                    print(f"\n    [DEBUG] Voting Summary (Step 0)")
                    print(f"    - Majority Threshold: {majority_threshold}")
                    print(f"    - Final Positive Pixel Ratio: {(final_preds.sum()/final_preds.size)*100:.4f}%")
                    print(f"{'='*40}\n")
                
                # 3. RLE Encoding & 결과 저장
                for output, image_name in zip(final_preds, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
    
    return rles, filename_and_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_configs', type=str, nargs='+', default=None, 
                        help='List of path:size (e.g., modelA.pt:1024 modelB.pt:512)')
    # ... (기타 인자들은 이전과 동일)
    parser.add_argument('--model_scores', type=float, nargs='+', default=None)
    parser.add_argument('--voting', type=str, choices=['hard', 'soft'], default='soft')
    parser.add_argument('--thr', type=float, nargs='+', default=[0.5], 
                        help='Single float or List of floats for each class')
    parser.add_argument('--weight_min', type=float, default=0.5)
    parser.add_argument('--weight_max', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--use_flip_tta', action='store_true')
    parser.add_argument('--contrast_settings', type=float, nargs='+', default=[0.0, 0.15, 0.3])
    parser.add_argument('--output_csv', type=str, default='ensemble_output.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    
    # 1. Parse Configs
    model_configs = []
    if args.model_configs:
        for config_str in args.model_configs:
            path, size_str = config_str.rsplit(':', 1)
            model_configs.append((path, int(size_str)))
    else:
        raise ValueError("Please provide --model_configs (e.g., path:1024)")

    # 2. Load Models
    models, model_types, img_sizes = load_models(model_configs)
    
    # 3. Create DataLoaders
    print(f"{'='*20} 2. DataLoader Creation {'='*20}")
    unique_sizes = list(set(img_sizes))
    dataloaders_map = {}
    
    for size in unique_sizes:
        print(f"[*] Target Size {size}: Generating {len(args.contrast_settings)} contrast views...")
        loaders = []
        for c in args.contrast_settings:
            tf = get_test_transform(img_size=size, contrast=c)
            ds = XRayInferenceDataset(transforms=tf)
            ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            loaders.append(ld)
        dataloaders_map[size] = loaders
    print(f"[+] All loaders ready.\n")

    # 4. Run Ensemble
    if args.voting == 'soft':
        if args.model_scores:
            weights = normalize_weights(args.model_scores, args.weight_min, args.weight_max)
        else:
            weights = [1.0] * len(models)
        print('[Debug] Weights', weights)
            
        rles, filename_and_class = ensemble_soft_voting(
            models, model_types, img_sizes, dataloaders_map, weights, 
            thr_list=args.thr, use_flip_tta=args.use_flip_tta
        )
    else:
        rles, filename_and_class = ensemble_hard_voting(
            models, model_types, img_sizes, dataloaders_map, 
            thr=args.thr, use_flip_tta=args.use_flip_tta
        )

    # 5. Save
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({"image_name": image_name, "class": classes, "rle": rles})
    df.to_csv(args.output_csv, index=False)
    print("Done!")

if __name__ == '__main__':
    main()