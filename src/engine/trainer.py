from src.configs.defaults import CLASSES, SAVED_DIR
from src.metrics.dice import dice_coef
import os
import wandb
import torch
import datetime
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


def validation(epoch, model, data_loader, criterion, thr=0.5, tta=False):
    print(f'Start validation #{epoch:2d}')
    model.eval()
    model = model.cuda()

    dices = []
    total_loss = 0
    cnt = 0
    
    with torch.no_grad():
        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            
            if not tta:
                if images.shape[-2:] != (2048, 2048):
                    outputs = model(images)
            
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)
            
                    # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
                    loss = criterion(outputs, masks)
                else:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
            
                total_loss += loss.item()
                cnt += 1
            
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).float()
            
                dice = dice_coef(masks, outputs)
                dices.append(dice)
            else:
                if images.dim() == 4 and images.size(0) % 2 == 0:  # TTA 2개 가정
                    batch_size = images.size(0) // 2
                    images = images.view(batch_size, 2, images.size(1), images.size(2), images.size(3))
                    masks = masks.view(batch_size, 2, masks.size(1), masks.size(2), masks.size(3))
                
                batch_size, n_tta, C, H, W = images.shape            
                outputs_tta = []
                
                for t in range(n_tta):
                    img = images[:, t]
                    with autocast():
                        output = model(img)
                    outputs_tta.append(output)
                
                outputs = torch.stack(outputs_tta, dim=1)
                outputs[:, 1] = torch.flip(outputs[:, 1], dims=[-1])
                outputs = torch.mean(outputs, dim=1)
                
                loss = criterion(outputs, masks[:, 0])
                total_loss += loss.item()
                cnt += 1
                
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).float()
            
                dice = dice_coef(masks[:, 0], outputs)
                dices.append(dice)
                
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = dices_per_class.mean().item()
    
    return total_loss / len(data_loader), avg_dice


def train(model, data_loader, val_loader, criterion, optimizer, scheduler, cfg):
    print(f'Start training..')
    
    model = model.cuda()
    scaler = GradScaler()

    best_dice = 0.
    patience = 0
    
    if cfg.total:
        save_epochs = [cfg.num_epochs - 10, cfg.num_epochs - 5, cfg.num_epochs]
        val_every = None
    else:
        save_epochs = []
        val_every = cfg.val_every
    
    for epoch in range(cfg.num_epochs):
        train_loss = 0
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            
            optimizer.zero_grad()
            
            if images.shape[-2:] != (2048, 2048):
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            else:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            train_loss += loss.item()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{cfg.num_epochs}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )

        if cfg.total and (epoch + 1) in save_epochs:
            output_path = os.path.join(SAVED_DIR, f"{epoch+1}epoch_{cfg.save_name}")
            print(f"Save checkpoint at epoch {epoch+1} -> {output_path}")
            torch.save(model.state_dict(), output_path)
        
        if not cfg.total and (epoch + 1) % val_every == 0:
            val_loss, dice = validation(epoch + 1, model, val_loader, criterion, tta=cfg.tta)
            
            if scheduler is not None:
                if cfg.scheduler == "reduce":
                    scheduler.step(dice)
                else:
                    scheduler.step()
            
            if best_dice < dice:
                output_path = os.path.join(SAVED_DIR, cfg.save_name)
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {output_path}")
                best_dice = dice
                torch.save(model, output_path)
                patience = 0
            else:
                patience += 1
            
            if cfg.use_wandb:
                wandb.log({
                    "train/loss": train_loss / len(data_loader),
                    "val/loss": val_loss,
                    "val/DICE": dice,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                })
        else:
            if scheduler is not None and cfg.scheduler != "reduce":
                scheduler.step()
            
            if cfg.use_wandb:
                wandb.log({
                    "train/loss": train_loss / len(data_loader),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                })
        
        if patience == cfg.num_patience:
            print(f"early stopping at {epoch + 1}epoch")
            break