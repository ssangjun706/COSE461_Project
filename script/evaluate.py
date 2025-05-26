import os
import sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.dataset import TitanicDataset
from src.inference import InferenceModel
from src.trainer import PPOTrainer, PPOConfig
from src.utils import evaluate_multiple_runs, calculate_metrics


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    config = PPOConfig(
        batch_size=16,
        learning_rate=3e-7,
        max_epochs=1,
        use_lora=True,
        max_new_tokens=512,
        use_wandb=False,
        resume_from_checkpoint="../checkpoints/final_model",
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

    # Evaluate
    num_epochs = 20
    evaluation_results = []
    for _ in range(num_epochs):
        y_pred, y_true = trainer.evaluate(use_tqdm=True, no_metrics=True)
        evaluation_results.append((y_pred, y_true))
        print(calculate_metrics(y_pred, y_true))

    evaluate_multiple_runs(evaluation_results, "evaluation", save_json=True, output_dir="../results/")

    
    # Prompt Check
    # X, y = random.choice(val_dataset)
    # input_ids = trainer.tokenizer(
    #     X,
    #     padding=True,
    #     return_tensors="pt",
    # ).to(trainer.device)

    # generated_tokens, _, _ = trainer.generate(input_ids, output_scores=False)
    # generated_texts = trainer.tokenizer.batch_decode(
    #     generated_tokens, skip_special_tokens=True
    # )

    # print(generated_texts[0])
