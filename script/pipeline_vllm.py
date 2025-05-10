import os
import sys
import argparse
import requests

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.utils import process

from src.vllm_module import LLMTrainer
from src.dataset import TitanicDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    # default="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    default="meta-llama/Llama-3.3-70B-Instruct",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="../dataset/titanic-dataset.csv",
)
parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=2,
    help="Number of GPUs to use for tensor parallelism",
)
parser.add_argument(
    "--num_sequences",
    type=int,
    default=1,
    help="Number of sequences to generate per sample",
)
parser.add_argument("--max_tokens", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.7)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

API_URL = "http://localhost:23456"


def request(prompts: list[str]):
    response = requests.post(
        f"{API_URL}/generate",
        json={"prompts": prompts},
    )
    response.raise_for_status()
    return response.json()["text"]


def check_server_status():
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200 and response.json():
            print(f"Inference server is ready")
        else:
            print("Error: Inference server is not ready or model is not loaded")
            exit(1)
    except Exception as e:
        print(f"Error connecting to inference server: {e}")


if __name__ == "__main__":
    check_server_status()
    args = parser.parse_args()

    dataset = TitanicDataset(args.dataset, train=True)
    target, target_values = dataset.target, dataset.target_values
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LLMTrainer(
        args.model_name,
        num_sequences=args.num_sequences,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
    )

    for X, y in tqdm(data_loader):
        prompts = process(
            X, target, target_values
        )  # processing input(json) into prompt
        outputs = model.generate(prompts)

        # Fine-tuning the model
        # ...

        # For debugging purposes
        print("=" * 50)
        print(f"Input Data: {X}")
        print(f"Output Data: {y}")
        print()

        for x in outputs:
            print(x[0])
            t = request(x[0])
            print(t)
        print("=" * 50)
