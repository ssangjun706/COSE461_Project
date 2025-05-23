import time
import logging
import torch

from src.hf_utils import find_model_path

from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import LoraConfig, LoraRuntimeConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class PromptModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.adapter = None

        self.model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        self.model_path = find_model_path(self.model_name)

        self.peft_config = {
            "use_dora": True,
            "target_modules": ["q_proj", "v_proj"],
            "runtime_config": LoraRuntimeConfig(ephemeral_gpu_offload=True),
        }

        logging.info(f"Loading model: {self.model_name}...")
        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self.device = self.base_model.device
        self.lora_config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            use_dora=True,
            runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True),
        )
        self.model = get_peft_model(self.base_model, self.lora_config)
        logging.info(f"Model loaded successfully in {(time.time() - start_time):.2f}s")

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict)

    def generate(
        self,
        prompts: list[str],
        sampling_params: dict,
    ):
        start_time = time.time()

        tokenized_inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        input_length = tokenized_inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **tokenized_inputs,
            **sampling_params,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        logging.info(f"Generation completed in {(time.time() - start_time):.2f}s")

        logits = outputs[:, input_length:]
        prompts = self.tokenizer.batch_decode(
            logits,
            skip_special_tokens=True,
        )
        return prompts, logits
