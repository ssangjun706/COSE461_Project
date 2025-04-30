import os
import sys
import time
import json
import torch

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

import src.utils as utils
from src.dataset_classes import IncomeDataset

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def build_meta_prompt(data, label, label_values=None):
    label_values = ", ".join(label_values)
    return f"""You are an AI assistant specialized in crafting **diverse and creative prediction prompts** that will be given to another LLM (a predictor).

The predictor LLM's ultimate goal is to analyze input features and predict a target variable.

**Your main objective now is to generate *one prediction prompt*, focusing on exploring different ways to structure the request.** Be creative! The prompt you generate should contain the necessary X data, Y-label to predict '{label}', and the requirement for the predictor's final output format {label_values} only.

**Crucially, *how* you structure this prompt is up to you.** You could simply list the data and ask for the prediction, include explanations of what the features mean or any other structures entirely!

Input Information for you to use:

X data: {data}
Y-label to predict: {label}
Required final output from the predictor: {label_values} (strictly one of these)

Now, generate **one prediction prompt** based on these instructions. Focus on creating a potentially effective but possibly unconventional structure.
/no_think
"""


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

    data = load_dataset("scikit-learn/adult-census-income", split="train")
    dataset = IncomeDataset(data)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model_name = "Qwen/Qwen3-32B"
    model_path = utils.find_model_path(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model: {model_name}...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    print("Model loaded successfully in {:.2f}s".format(time.time() - start))

    generation_params = {
        "num_return_sequences": 2,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "max_new_tokens": 256,
        "do_sample": True,
    }

    for X, _ in tqdm(data_loader):
        preprompts = [
            build_meta_prompt(x, dataset.target, dataset.label_values) for x in X
        ]
        inputs = tokenizer(
            preprompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(model.device)

        input_length = inputs.input_ids.shape[1]
        start = time.time()

        print("\n--- Generating Outputs ---")
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)

        decoded_outputs = tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )

        # =========================
        results = []

        print("\n--- Processing Generated Outputs ---")
        for i, output_text in enumerate(decoded_outputs):
            result = {
                "batch_index": len(results) // 4,
                "sample_index": i % 4,
                "output_index": i // 4,
                "generated_text": output_text.strip().replace("</think>\n\n", ""),
            }
            results.append(result)

        output_file = "generated_prompts.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Total time (incl. decoding): {time.time() - start:.2f}s")
        break
