import os
import sys
import uvicorn
import time
import argparse

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.vllm_module import LLMTrainer
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys


class InferenceRequest(BaseModel):
    prompts: str | list[str]


class InferenceResponse(BaseModel):
    text: list[list[str]]
    time_taken: float


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

model = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Llama-3.3-70B-Instruct",
)
parser.add_argument("--tensor_parallel_size", type=int, default=4)
parser.add_argument("--num_sequences", type=int, default=1)
parser.add_argument("--max_model_len", type=int, default=1024)
parser.add_argument("--max_tokens", type=int, default=16)
parser.add_argument("--temperature", type=float, default=0.15)
args = parser.parse_args()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = LLMTrainer(
        model_name=args.model_name,
        mode="classify",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        num_sequences=args.num_sequences,
        temperature=args.temperature,
    )

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
