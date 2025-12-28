from src.data.test_data import XRayInferenceDataset
from src.configs.defaults import SAVED_DIR, RANDOM_SEED
from src.utils.set_seed import set_seed
from src.engine.inference import test
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    model = torch.load(os.path.join(SAVED_DIR, args.ckpt), weights_only=False)

    model = model.cuda()
    model.eval()

    test_dataset = XRayInferenceDataset()

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    rles, filename_and_class = test(model, test_loader)
    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    out_name = args.ckpt.replace(".pt", ".csv")
    out_path = os.path.join(
        "/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/outputs",
        out_name
    )
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()