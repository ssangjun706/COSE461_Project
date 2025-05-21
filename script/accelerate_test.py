import os
import sys
import torch

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

import src.hf_utils
from transformers import AutoModelForImageTextToText, AutoTokenizer


if __name__ == "__main__":
    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    prompts = ["Hello, world!"]
    tokenized_inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = tokenized_inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **tokenized_inputs,
        pad_token_id=tokenizer.eos_token_id,
    )

    logits = outputs[:, input_length:]
    prompts = tokenizer.batch_decode(
        logits,
        skip_special_tokens=True,
    )

    print(prompts)
