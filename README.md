# Hand Bone Segmentation

## 주의사항
- `src/configs/config.py` 파일에서 데이터 경로 확인
- 데이터는 `src/datasets` 내부에 넣는 것을 추천
- `wandb` 설정을 위해 `.env` 파일 작성:

```text
WANDB_API_KEY=[본인 키] 
WANDB_ENTITY=CV-07
WANDB_PROJECT=Hand-Bone-Segmentation
```

## 학습
```bash
python -m scripts.train --model unet --use_wandb
```

## 추론
- 아직 parser 추가 X, 하드 코딩해야 됨
```bash
python -m scripts.test
```


