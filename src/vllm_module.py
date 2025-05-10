import os
import sys
import time
import logging

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.hf_utils import find_model_path

from typing import Sequence
from vllm import LLM, SamplingParams, PromptType, RequestOutput
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class LLMTrainer:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        mode: str = "generation",
        max_model_len: int = 512,
        max_tokens: int = 512,
        num_sequences: int = 1,
        dtype: str = "auto",
        temperature: float = 0.15,
        gpu_memory_utilization: float = 0.90,
    ):
        """
        Initialize LLMProcessor

        Args:
            model_name (str): Hugging Face model name.
            tensor_parallel_size (int): Tensor parallel processing size.
            dtype (str): Model data type ('auto', 'bfloat16', 'float16').
            num_sequences (int): Number of sequences to generate per sample.
            max_model_len (int): Maximum model length.
            temperature (float): Sampling temperature.
        """

        self.mode = mode
        self.quantization = None
        self.gpu_memory_utilization = gpu_memory_utilization
        self.sampling_params_dict = SamplingParams(
            n=num_sequences,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self.model_path = find_model_path(model_name)
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len

        logging.info(f"Initializing vLLM server for model: {model_name}...")
        start_time = time.time()
        try:
            self.instance = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                quantization=self.quantization,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            logging.info(
                f"Model loaded successfully in {time.time() - start_time:.2f} seconds."
            )
        except Exception as e:
            logging.error(f"Failed to load model {model_name} with error: {e}")
            exit(1)

    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        prompt_adapter_request: PromptAdapterRequest | None = None,
    ) -> list[list[str]]:
        """Generates the completions for the input prompts.

        This automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.
        Returns:
            A list of ``RequestOutput`` objects containing the
            generated completions in the same order as the input prompts.

        """
        if not prompts:
            return []

        start_time = time.time()

        outputs = self.instance.generate(
            prompts,
            sampling_params=self.sampling_params_dict,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            use_tqdm=False,
        )

        generated_texts = []
        for output in outputs:
            prompt_outputs = [
                output.outputs[i].text.strip() for i in range(len(output.outputs))
            ]
            generated_texts.append(prompt_outputs)

        logging.info(f"Processed in {time.time() - start_time:.2f} seconds.")
        return generated_texts
