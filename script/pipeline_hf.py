import os
import sys
import argparse

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.hf_module import LLMLoader
from src.dataset import IncomeDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen3-32B",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="scikit-learn/adult-census-income",
)
parser.add_argument("--num_sequences", type=int, default=2)
parser.add_argument("--max_token", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=4)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    args = parser.parse_args()

    _data = load_dataset(args.dataset, split="train")
    _dataset = IncomeDataset(_data)
    label, label_values = _dataset.target, _dataset.label_values
    data_loader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=True)

    model = LLMLoader(
        args.model_name,
        num_sequences=args.num_sequences,
        max_tokens=args.max_token,
    )

    for X, _ in tqdm(data_loader):
        X = model.collate_fn(X, label, label_values)
        outputs = model.generate(X)
        decoded_outputs = model.decode(outputs)

        for i, output_text in enumerate(decoded_outputs):
            print("=" * 50)
            print(output_text.strip())
            print("=" * 50)
            print("\nPress Enter to continue or Ctrl+C to exit.", end="")
            input()

        break
