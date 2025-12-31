from src.configs.config import CLASSES, NUM_EPOCHS, VAL_EVERY, SAVED_DIR
from src.metrics.dice import dice_coef
import os
import wandb
import torch
import datetime
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler



def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()
    #model = model.cuda()
    #이미 cuda에 있으니까?
    torch.cuda.empty_cache()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            
            #outputs = model(images)#['out'] 이거는 unet이 반환하는게 이렇다는데
            
            #이것도 autocast 안에서 AMP를 위해서 
            with autocast():
                outputs = model(images)
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach()
            masks = masks.detach()
            
            dice = dice_coef(outputs, masks).cpu()
            dices.append(dice)
            del images, masks, outputs, dice
            torch.cuda.empty_cache()
            
            #cpu랑
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return total_loss / len(data_loader), avg_dice


def train(model, data_loader, val_loader, criterion, optimizer, save_file_name,start_epoch,early_stopping=None):
    print(f'Start training..')
    scaler = GradScaler()
    n_class = len(CLASSES)
    best_dice = 0.
    
    acum=2
    optimizer.zero_grad()
    
    for epoch in range(start_epoch,NUM_EPOCHS):
        train_loss = 0
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            images, masks = images.cuda(), masks.cuda()
           
            #autocast 안에서 부르려고
            #outputs = model(images)#['out']
            
            # loss 계산 , gradient 업데이트
            
            
            # loss = criterion(outputs, masks)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # train_loss += loss.item()
            
            #AMP 사용
            # optimizer.zero_grad()

            # with autocast():
            #     outputs = model(images)
            #     loss = criterion(outputs, masks)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # train_loss += loss.item()
            #gradient accumulation와 AMP 적용.
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / acum         

            scaler.scale(loss).backward()   

            if (step + 1) % acum == 0:
                scaler.step(optimizer)      #
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
                    
            
            
            
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        if (epoch + 1) % VAL_EVERY == 0:
            val_loss, dice = validation(epoch + 1, model, val_loader, criterion)
            
        
            if early_stopping is not None:
                early_stopping(dice, model, optimizer, epoch)
                #코딩 잘못해서 dice 버전으로 early_stopping을 걸었는데 값을 val_loss 주고 있었음. dice로 수정
            
                if early_stopping.early_stop:
                    print("🛑 Early stopping triggered!")
                    break
            
            if best_dice < dice:
                output_path = os.path.join(SAVED_DIR, save_file_name)
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {output_path}")
                best_dice = dice
                #torch.save(model, output_path)
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },output_path)
                
            
            
            wandb.log({
                "train/loss": train_loss / len(data_loader),
                "val/loss": val_loss,
                "val/DICE": dice,
                "epoch": epoch + 1,
            })
        else:
            wandb.log({
                "train/loss": train_loss / len(data_loader),
                "epoch": epoch + 1,
            })