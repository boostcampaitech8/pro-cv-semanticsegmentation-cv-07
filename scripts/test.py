from src.data.test_data import XRayInferenceDataset
from src.configs.config import SAVED_DIR, RANDOM_SEED
from src.utils.set_seed import set_seed
from src.engine.inference import test
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.configs.config import BATCH_SIZE, CLASSES, LR, SAVED_DIR, RANDOM_SEED, NUM_EPOCHS, VAL_EVERY

def main():
    set_seed(RANDOM_SEED)
    
    save_file_name = "SWIN-UNET_best_model.pt"
    #model = torch.load(os.path.join(SAVED_DIR, save_file_name), weights_only=False)
    try:
        from external.Swin_Unet.networks.vision_transformer import SwinUnet 
        from external.Swin_Unet.config import get_config
        print("✅ Swin-Unet 모듈 로드 성공.")
        
    except ImportError as e:
        print(f"❌ SWIN-UNET 모듈 로드 실패: {e}")
        
    
    from external.Swin_Unet.config import _C
    import copy

    config = copy.deepcopy(_C)
    config.merge_from_file(
        "external/Swin_Unet/configs/swin_tiny_patch4_window7_224_lite.yaml"
    )

    

    config.MODEL.NUM_CLASSES = len(CLASSES)
    
    #512에 맞게끔 윈도우 사이즈?
    config.DATA.NUM_CLASSES = len(CLASSES)
    config.MODEL.SWIN.WINDOW_SIZE = 8

    config.DATA.IMG_SIZE = 512
    
    
    

    model = SwinUnet(config,num_classes=len(CLASSES))
    
    checkpoint = torch.load(
    os.path.join(SAVED_DIR, save_file_name),
    map_location="cuda"
)

    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    
    
    test_dataset = XRayInferenceDataset()

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    rles, filename_and_class = test(model, test_loader,thr=0.4)
    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    #여기 이름 바꿔야해
    df.to_csv("/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/outputs/output_4.csv", index=False)


if __name__ == '__main__':
    main()