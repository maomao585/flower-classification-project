import os

from dotenv import load_dotenv


def load_config():
    """
    加载和验证环境配置
    """
    load_dotenv()

    config = {
        # 数据路径配置
        "DATA_DIR": os.getenv("DATA_DIR", "./data/flowers17"),
        "TRAIN_DIR": os.getenv("TRAIN_DIR", "./data/train"),
        "TEST_DIR": os.getenv("TEST_DIR", "./data/test"),
        # MLflow配置
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
        "MLFLOW_EXPERIMENT_NAME": os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "flower-classification"
        ),
        # 模型训练配置
        "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 8)),
        "LEARNING_RATE": float(os.getenv("LEARNING_RATE", 0.001)),
        "EPOCHS": int(os.getenv("EPOCHS", 20)),
        "NUM_CLASSES": int(os.getenv("NUM_CLASSES", 17)),
        # 模型保存配置
        "MODEL_SAVE_PATH": os.getenv("MODEL_SAVE_PATH", "./models"),
    }

    return config