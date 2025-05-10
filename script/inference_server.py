import os
import sys
import uvicorn
import time
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.vllm_module import LLMTrainer
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
from io import StringIO


class InferenceRequest(BaseModel):
    prompts: str | list[str]


class InferenceResponse(BaseModel):
    text: list[list[str]]
    time_taken: float


os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

model = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Llama-3.3-70B-Instruct",
)
parser.add_argument("--tensor_parallel_size", type=int, default=4)
parser.add_argument("--num_sequences", type=int, default=1)
parser.add_argument("--max_tokens", type=int, default=32)
parser.add_argument("--temperature", type=float, default=0.15)
args = parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Initializing vLLM server for model: {args.model_name}...")

    global model
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        model = LLMTrainer(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_tokens=args.max_tokens,
            num_sequences=args.num_sequences,
            temperature=args.temperature,
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    logging.info(f"Model {args.model_name} loaded successfully")
    yield
    del model


app = FastAPI(title="Inference LLM Server", lifespan=lifespan)


@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    global model

    if not model:
        return {"error": "Model not loaded yet", "time_taken": 0}

    start_time = time.time()
    outputs = model.generate(request.prompts)
    time_taken = time.time() - start_time

    return {"text": outputs, "time_taken": time_taken}


@app.get("/health", response_model=bool)
async def health_check():
    global model

    return model is not None


if __name__ == "__main__":
    uvicorn.run("inference_server:app", host="localhost", port=23456)
