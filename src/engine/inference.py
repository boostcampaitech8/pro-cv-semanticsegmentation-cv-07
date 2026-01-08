from src.configs.defaults import CLASSES, IND2CLASS
from src.data.utils import encode_mask_to_rle
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F


def test(model, data_loader, thr=0.5, tta=False):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            
            if not tta: 
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["seg"]
            else:
                if images.dim() == 4 and images.size(0) % 2 == 0:  # TTA 2개 가정
                    batch_size = images.size(0) // 2
                    images = images.view(batch_size, 2, images.size(1), images.size(2), images.size(3))

                batch_size, n_tta, C, H, W = images.shape            
                outputs_tta = []
                
                for t in range(n_tta):
                    img = images[:, t]
                    output = model(img)
                    outputs_tta.append(output)

                outputs = torch.stack(outputs_tta, dim=1)
                outputs[:, 1] = torch.flip(outputs[:, 1], dims=[-1])
                outputs = torch.mean(outputs, dim=1)
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class