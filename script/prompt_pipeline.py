import os
import sys

import torch
import time

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

import setup
from transformers import AutoTokenizer, MistralForCausalLM, BitsAndBytesConfig


from vllm import SamplingParams
from vllm_module import LLMProcessor
from dataset_classes import IncomeDataset

from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv
from dataset_classes import load_dataset


def build_meta_prompt(data, label, label_values=None):
    label_values = ", ".join(label_values)
    return f"""You are an AI assistant specialized in crafting **diverse and creative prompts** for other LLM models. Your current task is to generate a **potential prediction prompt** that will be given to another LLM (a predictor).

The predictor LLM's ultimate goal is to analyze input features and predict a target variable, outputting **only** one of two specific values: {label_values}.

**Your main objective now is to generate *one example* of a prediction prompt, focusing on exploring different ways to structure the request.** Be creative! The prompt you generate should contain the necessary X data, mention the Y-label to predict ('income'), and somehow convey the requirement for the predictor's final output format {label_values} only).

**Crucially, *how* you structure this prompt is up to you.** You could:
*   Simply list the data and ask for the prediction.
*   Include explanations of what the feature labels mean (inferring from names).
*   Phrase it as a question about the person's profile.
*   Use a specific persona (e.g., "As a data analyst, predict...").
*   Describe the profile in prose.
*   **Or try other unique structures entirely!**

The key is **variety and experimentation** in the prompt format itself, while still providing the necessary information and output constraints *to the predictor*.

Input Information for you to use:

X data: {data}
Y-label to predict: {label}
Required final output from the predictor: {label_values} (strictly one of these)

Now, generate **one unique prediction prompt** based on these instructions. Focus on creating a potentially effective but possibly unconventional structure.
"""


def find_path(model_name: str):
    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            ignore_patterns=["*.safetensors", "*.bin"],
        )

        return snapshot_path
    except Exception:
        assert (
            False
        ), f"Could not get snapshot path for {model_name} from cache. Try download again"


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable must be set for Hugging Face login."
        )
    login(token=hf_token)

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    uci_adult = load_dataset("scikit-learn/adult-census-income", split="train")
    dataset = IncomeDataset(uci_adult)
    label = "income"
    label_values = ["<=50K", ">50K"]

    sample_X, sample_y = dataset[0]

    simple_prompt = build_meta_prompt(
        data=sample_X, label=label, label_values=label_values
    )

    model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    model_name = find_path(model_name)
    use_quantization = False

    # if use_quantization:
    #     print("Using 4-bit quantization.")
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #     )
    # else:
    #     print("Loading model in full precision (requires substantial VRAM).")
    #     quantization_config = None
    #     warnings.warn(
    #         f"Loading {model_name} without quantization requires significant GPU memory (>50GB VRAM likely)."
    #     )

    # --- Model and Tokenizer Loading ---
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = MistralForCausalLM.from_pretrained(
        model_name,
        quantization_config=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("Model loaded successfully.")
    print(f"Model loaded on device(s): {model.hf_device_map}")

    generation_params = {
        "num_return_sequences": 5,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # print("\nGeneration parameters:")
    # for key, value in generation_params.items():
    #     print(f"- {key}: {value}")
    # print("\n--- Ready for Inference ---")

    # --- Example Prompt (Instruction Format is Crucial!) ---
    # Use the chat template provided by the tokenizer for instruct models
    simple_prompt_content = (
        "What are the main challenges faced by renewable energy sources?"
    )
    # messages = [{"role": "user", "content": simple_prompt_content}]
    # # This formats the input correctly with [INST] ... [/INST] tokens
    # formatted_prompt = tokenizer.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # print(f"\nFormatted Prompt Example:\n{formatted_prompt}")
    formatted_prompt = simple_prompt

    # --- Inference Loop ---
    while True:
        print(
            "\nPress Enter to generate based on the example prompt or type 'exit' to quit: ",
            end="",
        )
        x = input()
        if x.lower() == "exit":
            break

        print("Tokenizing prompt...")
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        input_length = inputs.input_ids.shape[1]  # Get length of input tokens

        print(f"Generating {generation_params['num_return_sequences']} responses...")
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)

        generation_time = time.time() - start
        print(f"Raw generation finished in {generation_time:.2f}s")

        decoded_outputs = tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )

        print("\n--- Generated Outputs ---")
        for i, output_text in enumerate(decoded_outputs):
            print(f"--- Output {i+1} ---")
            print(output_text.strip())
            print("-" * 20)

        print(f"\nTotal time (incl. decoding): {time.time() - start:.2f}s")

    # instance = LLMProcessor(
    #     model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    #     mode="generation",
    #     tensor_parallel_size=2,
    #     max_model_len=4096,
    #     gpu_memory_utilization=0.9,
    #     sampling_params_dict=SamplingParams(
    #         n=5,
    #         temperature=0.8,
    #         top_p=0.95,
    #         max_tokens=512,
    #     ),
    # )

    # while True:
    #     print("Press Enter to generate a new prompt or type 'exit' to quit: ", end="")
    #     x = input()
    #     if x.lower() == "exit":
    #         break

    #     start = time.time()
    #     parsed_outputs = instance.generate(simple_prompt)
    #     print(parsed_outputs)
    #     print(f"Generation time: {time.time() - start:.2f}s")
