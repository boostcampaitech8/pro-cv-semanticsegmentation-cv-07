from src.configs.defaults import CLASSES, SAVED_DIR
from src.metrics.dice import dice_coef
import os
import wandb
import torch
import datetime
from tqdm.auto import tqdm
import torch.nn.functional as F

# ===============================
# Boundary GT 생성 (GPU friendly)
# ===============================
def generate_boundary_label(mask, kernel_size=3):
    """
    mask: (B, C, H, W)  (multi-class mask)
    return: (B, 1, H, W) boundary map
    """
    # 클래스별 경계를 OR로 합침
    mask = (mask > 0).float()
    mask = mask.max(dim=1, keepdim=True)[0]

    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)

    boundary = (dilated - eroded).clamp(0, 1)
    return boundary


def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()
    model = model.cuda()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            
            outputs = model(images)

            if isinstance(outputs, dict):
                outputs = outputs["seg"]
                
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(masks, outputs)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()

    # 🔹 class-wise dice dict 생성
    dice_class_dict = {
        f"val_class/{c}": d.item()
        for c, d in zip(CLASSES, dices_per_class)
    }

    return total_loss / len(data_loader), avg_dice, dice_class_dict, dices_per_class


def train(model, data_loader, val_loader, criterion, optimizer, cfg):
    print(f'Start training..')
    
    SPIKE_META_PATH = os.path.join(SAVED_DIR, "spike_meta.txt")

    # 🔹 새 실험 시작 시 spike meta 초기화
    if os.path.exists(SPIKE_META_PATH):
        os.remove(SPIKE_META_PATH)

    model = model.cuda()
    n_class = len(CLASSES)
    best_dice = 0.
    patience = 0

    prev_dice = None
    prev_class_dice = None
    SPIKE_SAVE_TH = 0.05
    SPIKE_OBS_TH = 0.03

    for epoch in range(cfg.num_epochs):
        train_loss = 0
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            
            outputs = model(images)
            
            # loss를 계산합니다.
            # ===============================
            # Boundary-aware training
            # ===============================
            if isinstance(outputs, dict):
                pred_seg = outputs["seg"]
                pred_boundary = outputs["boundary"]

                # 1️⃣ Segmentation loss
                loss_seg = criterion(pred_seg, masks)

                # 2️⃣ Boundary GT 생성
                boundary_gt = generate_boundary_label(masks)

                # 3️⃣ Boundary loss
                loss_boundary = criterion(pred_boundary, boundary_gt)

                # 4️⃣ Total loss
                if cfg.boundary_detach:
                    loss = loss_seg + 0.05 * loss_boundary.detach()
                else:
                    loss = loss_seg + 0.05 * loss_boundary


            else:
                # Baseline (U-Net++)
                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{cfg.num_epochs}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        if (epoch + 1) % cfg.val_every == 0:
            val_loss, dice, dice_class_dict, class_dice = validation(epoch + 1, model, val_loader, criterion)
            # ===== [ADD] dice 변화량 계산 (validation 직후) =====
            delta = None
            abs_delta = None
            if prev_dice is not None:
                delta = dice - prev_dice
                abs_delta = abs(delta)

            # 🔹 spike 감지
            if abs_delta is not None:
                # 1️⃣ 관찰용 spike
                if abs_delta >= SPIKE_OBS_TH:
                    print(f"[OBS] ΔDice={delta:+.4f} at epoch {epoch+1}")
                    if cfg.use_wandb:
                        wandb.log({
                            "debug/dice_spike": delta,
                            "epoch": epoch + 1
                        })

                # 2️⃣ 저장용 spike
                if abs_delta >= SPIKE_SAVE_TH:
                    direction = "up" if delta > 0 else "down"
                    spike_name = f"spike_{direction}_e{epoch+1}_d{dice:.4f}.pt"
                    torch.save(model, os.path.join(SAVED_DIR, spike_name))
                    print(f"[SPIKE-SAVE] {direction.upper()} ΔDice={delta:+.4f}")
                    
                    with open(SPIKE_META_PATH, "a") as f:
                        f.write(
                            f"epoch={epoch+1}, "
                            f"direction={direction}, "
                            f"dice={dice:.4f}, "
                            f"delta={delta:+.4f}\n"
                        )

            # 🔹 class-wise spike 감지
            if prev_class_dice is not None:
                class_delta = class_dice - prev_class_dice
                for c, d in zip(CLASSES, class_delta):
                    if abs(d.item()) >= 0.05:
                        print(f"[CLASS-SPIKE] {c}: ΔDice={d.item():+.4f} at epoch {epoch+1}")
                        if cfg.use_wandb:
                            wandb.log({
                                f"debug/class_spike/{c}": d.item(),
                                "epoch": epoch + 1
                            })
            
            if best_dice < dice:
                output_path = os.path.join(SAVED_DIR, cfg.save_name)
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {output_path}")
                best_dice = dice
                torch.save(model, output_path)
                patience = 0
                
                with open(SPIKE_META_PATH, "a") as f:
                    f.write(
                        f"[BEST] epoch={epoch+1}, dice={dice:.4f}\n"
                    )
            else:
                if abs_delta is not None and abs_delta < SPIKE_OBS_TH:
                    patience += 1

            prev_dice = dice
            prev_class_dice = class_dice
            
            if cfg.use_wandb:
                log_dict = {
                    "train/loss": train_loss / len(data_loader),
                    "val/loss": val_loss,
                    "val/DICE": dice,
                    "epoch": epoch + 1,
                }
                # 🔹 class-wise dice 추가
                log_dict.update(dice_class_dict)

                wandb.log(log_dict)
        else:
            if cfg.use_wandb:
                wandb.log({
                    "train/loss": train_loss / len(data_loader),
                    "epoch": epoch + 1,
                })
        
        # 🔹 Baseline에서만 early stopping 적용
        if cfg.boundary_mode == "none":
            if patience == cfg.num_patience:
                print(f"early stopping at {epoch + 1}epoch")
                break
        elif cfg.boundary_mode == "basnet":
            # basnet: 60epoch 이후부터 early stopping 활성화
            if (epoch + 1) >= 60 and patience == cfg.num_patience:
                print(f"[BASNet] early stopping at {epoch + 1}epoch (post-60)")
                break