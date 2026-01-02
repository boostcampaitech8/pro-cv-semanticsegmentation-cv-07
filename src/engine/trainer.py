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

    dices = []
    total_loss = 0
    cnt = 0

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images, masks = images.cuda(), masks.cuda()

            with autocast():
                outputs = model(images)

                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")

                loss = criterion(outputs, masks)

            total_loss += loss.item()
            cnt += 1

            probs = torch.sigmoid(outputs)
            preds = (probs > thr)

            dice = dice_coef(preds, masks).cpu()
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = dices.mean(0)

    avg_dice = dices_per_class.mean().item()
    avg_loss = total_loss / cnt

    return avg_loss, avg_dice


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

            train_loss += loss.item()*acum
            #train loss는 봐야하니까 로그용
        if (step + 1) % acum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        #그리고 이거 step 다 돌렸을 떄 남는게 있을 수도 있잖아  
            
            
            
            
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
                best_epoch=epoch
                #torch.save(model, output_path)
            #     torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # },output_path)
                
            
            
            wandb.log({
                "train/loss": train_loss / len(data_loader),
                "val/loss": val_loss,
                "val/DICE": dice,
                "epoch": epoch + 1,
                "val/best_DICE": best_dice,  
                "best epoch": best_epoch
            })
        else:
            wandb.log({
                "train/loss": train_loss / len(data_loader),
                "epoch": epoch + 1
            })