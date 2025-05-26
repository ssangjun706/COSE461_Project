import os
import sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.dataset import TitanicDataset
from src.inference import InferenceModel
from src.trainer import PPOTrainer, PPOConfig


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    config = PPOConfig(
        batch_size=4,
        learning_rate=5e-7,
        max_epochs=10,
        max_new_tokens=512,
        use_wandb=False,
        # resume_from_checkpoint="../checkpoints/final_model",
        project_name="ape-finetune",
    )

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