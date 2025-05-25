import os
import sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.dataset import TitanicDataset
from src.inference import InferenceModel
from src.trainer import APETrainer, APEConfig
from src.utils import evaluate_multiple_runs


def final_evaluation():
    config = APEConfig(
        use_lora=False,
        batch_size=16,
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

    num_epochs = 20
    evaluation_results = []
    for _ in range(num_epochs):
        y_pred, y_true = trainer.evaluate(use_tqdm=True)
        evaluation_results.append((y_pred, y_true))

    evaluate_multiple_runs(evaluation_results, "evaluation", save_json=True, output_dir="../results/")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    config = APEConfig(
        batch_size=16,
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

    num_epochs = 20
    evaluation_results = []
    for epoch in range(num_epochs):
        y_pred, y_true = trainer.evaluate(use_tqdm=True)
        evaluation_results.append((y_pred, y_true))

    evaluate_multiple_runs(evaluation_results, "evaluation", save_json=True, output_dir="../results/")