import time
import torch
import logging
import src.utils

from src.hf_utils import find_model_path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class HFLLMLoader:
    def __init__(self, model_name: str):
        self.model = None
        self.tokenizer = None
        model_path = find_model_path(model_name)

        logging.info(f"Loading model: {model_name}...")
        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        self.model = torch.compile(self.model)
        self.device = self.model.device
        logging.info(f"Model loaded successfully in {(time.time() - start_time):.2f}s")

    def generate(
        self,
        prompts: list[str],
        sampling_params: dict,
        decode: bool = True,
    ) -> torch.Tensor:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
            for prompt in prompts
        ]

        start_time = time.time()

        tokenized_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_length = tokenized_inputs["input_ids"].shape[-1]
        outputs = self.model.generate(**tokenized_inputs, **sampling_params)
        logging.info(f"Generation completed in {(time.time() - start_time):.2f}s")

        if decode:
            outputs = self.decode(outputs, input_length)

        return outputs

    def decode(self, outputs: torch.Tensor, input_length: int):
        decoded_outputs = self.tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
        )
        return decoded_outputs[0]
