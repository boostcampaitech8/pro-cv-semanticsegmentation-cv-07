from src.data.test_data import XRayInferenceDataset
from src.configs.defaults import SAVED_DIR, RANDOM_SEED
from src.utils.set_seed import set_seed
from src.engine.inference import test
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader


def main():
    set_seed(RANDOM_SEED)
    
    tta = True
    save_file_name = 'upernet_2048_tta_best.pt'
    model = torch.load(os.path.join(SAVED_DIR, save_file_name), weights_only=False)

    test_dataset = XRayInferenceDataset(tta)

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    rles, filename_and_class = test(model, test_loader, tta=tta)
    
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