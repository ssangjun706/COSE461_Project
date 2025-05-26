import time
import logging
import requests

from .hf_utils import find_model_path
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class InferenceModelWrapper:
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
        self.sampling_params = SamplingParams(**sampling_params)

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


class InferenceModel:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}"

    def generate(
        self,
        prompts: list[str],
        sampling_params: dict,
    ):
        response = requests.post(
            f"{self.url}/generate",
            json={
                "prompts": prompts,
                "sampling_params": sampling_params,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["text"]

    def check_status(self):
        try:
            response = requests.get(f"{self.url}/health")
            if response.status_code != 200 or not response.json():
                print("Error: Server is not ready")
                exit(1)
        except Exception as e:
            print(f"Error connecting to inference server: {e}")
            exit(1)
