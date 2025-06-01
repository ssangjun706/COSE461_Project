import os
import sys

# ADA: Automatic Data Augmentation via Large Language Model
root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.dataset import TitanicDataset
from src.inference import InferenceModel
from src.trainer import PPOTrainer, PPOConfig
from config import TRAIN_CONFIG


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    config = PPOConfig()

    inference_model = InferenceModel(host="localhost", port=23456)
    inference_model.check_status()

    train_dataset = TitanicDataset(path="../dataset/titanic-dataset.csv")
    val_dataset = TitanicDataset(path="../dataset/titanic-dataset.csv", train=False)

    trainer = PPOTrainer(
        config=config,
        inference_model=inference_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.train()