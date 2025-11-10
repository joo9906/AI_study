from pathlib import Path

class ModelConfig: 
    BATCH_SIZE = 8      # 2070 Super 8GB GPU 최적화
    EPOCHS = 100         # 2070 Super 8GB GPU 최적화, 상황 보고 줄일 수 있음.
    LEARNING_RATE = 5e-4 # 2070 Super 8GB GPU 최적화
    DROPOUT_RATE = 0.3   # 2070 Super 8GB GPU 최적화
    MHA_HEADS = 8        # 2070 Super 8GB GPU 최적화
    MHA_KEY_DIM = 64     # 2070 Super 8GB GPU 최적화