from src.data.test_data import XRayInferenceDataset
from src.configs.defaults import SAVED_DIR, RANDOM_SEED, TEST_ROOT, BATCH_SIZE
from src.utils.set_seed import set_seed
from src.engine.inference import test
import os
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    # 경로 설정
    parser.add_argument("--data", type=str, default=TEST_ROOT)
    parser.add_argument("--saved_dir", type=str, default=SAVED_DIR)
    parser.add_argument("--saved_name", type=str, default="none")
    
    # 데이터 설정
    parser.add_argument("--tta", action="store_true")
    
    parser.add_argument("--img_size", type=int, default=2048)
    
    # 모델 설정    
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    return parser.parse_args()



def main():
    args = parse_args()
    
    set_seed(RANDOM_SEED)
    
    model = torch.load(os.path.join(args.saved_dir, args.saved_name), weights_only=False)

    test_dataset = XRayInferenceDataset(args)

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    rles, filename_and_class = test(model, test_loader, tta=args.tta)
    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv("/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/outputs/output.csv", index=False)


if __name__ == '__main__':
    main()