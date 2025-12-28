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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    
    save_file_name = f"{args.model}_{args.encoder}_e{args.num_epochs}_best.pt"
    model = torch.load(os.path.join(SAVED_DIR, save_file_name), weights_only=False)

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
    df.to_csv("/data/ephemeral/home/pro-cv-semanticsegmentation-cv-07/outputs/output.csv", index=False)


if __name__ == '__main__':
    main()