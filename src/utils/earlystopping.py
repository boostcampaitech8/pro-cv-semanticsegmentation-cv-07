import torch
import wandb 
#원래 loss 기준이었음 . 기존 trainer처럼 val_dice 기준으로 가자
class EarlyStopping:
    """Early stops the training if validation Dice doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=True, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선되지 않아도 기다리는 epoch 수
            verbose (bool): 메시지 출력 여부
            delta (float): 개선으로 간주할 최소 변화량
            path (str): checkpoint 저장 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_Dice_max = -float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_Dice, model, optimizer, epoch):
        score = val_Dice

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_Dice, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'⏰ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_Dice, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_Dice, model, optimizer, epoch):
        '''Saves model when validation Dice increase'''
        if self.verbose:
            print(f'💾 Validation Dice increased ({self.val_Dice_max:.6f} --> {val_Dice:.6f}). Saving model...')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_Dice': val_Dice,
        }, self.path)
        
        wandb.run.summary["best_val_dice"] = val_Dice
        wandb.run.summary["best_epoch"] = epoch

        
        self.val_Dice_max = val_Dice