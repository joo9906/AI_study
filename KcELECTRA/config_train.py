from pathlib import Path

class ModelConfig: 
    TRAIN_BATCH_SIZE = 16     # 데이터 크기 보고 정해야함. 일단은 16으로 하고 데이터가 클수록 사이즈도 증가
    EVAL_BATCH_SIZE = 32      
    EPOCHS = 50      
    LEARNING_RATE = 2e-5
    DROPOUT_RATE = 0.3      
    
    MODEL_NAME = "beomi/KcELECTRA-base"
    TOKENIZER = "beomi/KcELECTRA-base"
    
    SEQUENCE_LEN = 64
    SEQUENCE_STRIDE_LEN = 16