import os
import sys
import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.utils import process, parse
from src.model import PromptModel, RewardModel, APIServer
from src.dataset import TitanicDataset


DATASET_PATH = "../dataset/titanic-dataset.csv"
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1

PROMPT_SAMPLING_PARAMS = {
    "max_new_tokens": 2048,
    "do_sample": True,
}

INFERENCE_SAMPLING_PARAMS = {
    "n": 1,
    "max_new_tokens": 4,
}

MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def train_epoch(
    model: PromptModel,
    dataset: Dataset,
    inference_server: APIServer,
):
    model.train()
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for X, y in tqdm(data_loader):
        meta_prompts = process(
            data=X,
            target=dataset.target,
            target_values=dataset.target_values,
        )

        prompts, logits = model.generate(meta_prompts, PROMPT_SAMPLING_PARAMS)
        y_pred = inference_server.request(prompts, INFERENCE_SAMPLING_PARAMS)

        batch_rewards = []
        for preds, y_i in zip(y_pred, y):
            sample_rewards = [
                1.0 if parse(p) == ("ALIVE" if y_i == 1 else "DEAD") else 0.0
                for p in preds
            ]
            batch_rewards.append(sample_rewards)

        batch_rewards = torch.tensor(batch_rewards, device=model.device)
        # TODO: Implement Training Steps


if __name__ == "__main__":
    server = APIServer("localhost", 23456)
    prompt_model = PromptModel()

    train_data = TitanicDataset(DATASET_PATH, train=True)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        avg_loss = train_epoch(prompt_model, train_data, server)
        print(f"Average Loss: {avg_loss:.4f}")

        # checkpoint_path = f"checkpoints/epoch_{epoch+1}.pt"
        # os.makedirs("checkpoints", exist_ok=True)
        # torch.save(
        #     {
        #         "epoch": epoch,
        #         "model_state_dict": prompt_model.state_dict(),
        #         "loss": avg_loss,
        #     },
        #     checkpoint_path,
        # )

    # Evaluation
    # dataset = TitanicDataset(DATASET_PATH, train=False)
    # data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # total_predictions = 0
    # total_parse_error = 0

    # for X, y in tqdm(data_loader):
    #     meta_prompts = process(
    #         data=X,
    #         target="Survived",
    #         target_values=CHOICE,
    #     )

    #     answer = ["ALIVE" if y_i == 1 else "DEAD" for y_i in y]
    #     prompts = loader.generate(meta_prompts, PROMPT_SAMPLING_PARAMS)

    #     if isinstance(prompts, str):
    #         prompts = [prompts]

    #     y_pred = server.request(prompts, INFERENCE_SAMPLING_PARAMS)
    #     flatten_y_pred = [p for sample in y_pred for p in sample]
    #     parsed_y_pred = [parse(p) for p in flatten_y_pred]

    #     parsing_errors = sum(1 for pred in parsed_y_pred if pred == "ERROR")
    #     correct = sum(1 for a, b in zip(answer, parsed_y_pred) if a == b)
    #     total_predictions += correct
    #     total_parse_error += parsing_errors

    # print(f"Accuracy: {total_predictions / len(data_loader.dataset)}")
    # print(f"Parsing Error: {total_parse_error}/{len(data_loader.dataset)}")
