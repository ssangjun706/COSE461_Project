import time
import torch
import logging
import src.utils

from src.hf_utils import find_model_path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import LoraConfig, LoraRuntimeConfig, get_peft_model
from vllm import LLM, SamplingParams

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

    def find_model_path(self, model_name: str):
        try:
            snapshot_path = snapshot_download(
                repo_id=model_name,
                local_files_only=True,
            )
            return snapshot_path
        except Exception:
            logging.error(f"Failed to find model path for '{model_name}'.")
            exit(1)

    def generate(
        self,
        prompts: list[str],
        sampling_params: dict,
    ) -> tuple[torch.Tensor, list[str]]:
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


class InferenceModel:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        max_model_len: int = 2048,
    ):

        model = find_model_path(model_name)
        logging.info(f"Initializing vLLM server for model: {model_name}...")
        start_time = time.time()
        try:
            self.instance = LLM(
                model=model,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
            )
            logging.info(
                f"Model loaded successfully in {time.time() - start_time:.2f} seconds."
            )
        except Exception as e:
            logging.error(f"Failed to load model {model_name} with error: {e}")
            exit(1)

    def generate(
        self,
        prompts: list[str],
        sampling_params: dict,
    ) -> list[str]:
        start_time = time.time()

        self.sampling_params = SamplingParams(n=sampling_params["n"])

        for param, value in sampling_params.items():
            if hasattr(self.sampling_params, param):
                setattr(self.sampling_params, param, value)

        try:
            batch_outputs = self.instance.generate(
                prompts,
                use_tqdm=False,
                sampling_params=self.sampling_params,
            )
        except Exception:
            logging.error("Failed to generate output")
            return None

        flattened_outputs = [
            gen.text.strip() for sample in batch_outputs for gen in sample.outputs
        ]

        logging.info(f"Processed in {time.time() - start_time:.2f} seconds.")
        return flattened_outputs
