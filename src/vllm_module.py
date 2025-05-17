import os
import sys
import time
import logging

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.hf_utils import find_model_path

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
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
        prompts: str | list[str],
        sampling_params: dict,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        prompt_adapter_request: PromptAdapterRequest | None = None,
    ) -> tuple[list[str]]:
        start_time = time.time()

        self.sampling_params = SamplingParams(n=sampling_params["n"])

        for param, value in sampling_params.items():
            if hasattr(self.sampling_params, param):
                setattr(self.sampling_params, param, value)

        # if "choice" in sampling_params:
        #     self.sampling_params.guided_decoding = GuidedDecodingParams(
        #         choice=sampling_params["choice"]
        #     )

        try:
            outputs = self.instance.generate(
                prompts,
                use_tqdm=False,
                sampling_params=self.sampling_params,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
            )
        except Exception:
            logging.error("Failed to generate output")
            return None

        generated_texts = [
            [output.text.strip() for output in sample.outputs] for sample in outputs
        ]

        logging.info(f"Processed in {time.time() - start_time:.2f} seconds.")
        return generated_texts
