from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import os

print("[MODEL LOADER] 모델 로드 중...")

model_path = os.path.join(os.path.dirname(__file__), "Classifier_Model")
Device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(Device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

if model:
    print(f"모델 로드 완료")
else:
    print(f"모델 로드 실패")