import os
import sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.dataset import TitanicDataset
from src.inference import InferenceModel
from src.trainer_ape import APETrainer, APEConfig


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    config = APEConfig(
        batch_size=16,
        learning_rate=3e-7,
        use_wandb=True,
        project_name="ape-finetune",
    )

    inference_model = InferenceModel(host="localhost", port=23456)
    inference_model.check_status()

    train_dataset = TitanicDataset(path="../dataset/titanic-dataset.csv")
    val_dataset = TitanicDataset(path="../dataset/titanic-dataset.csv", train=False)

    trainer = APETrainer(
        config=config,
        inference_model=inference_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.train()