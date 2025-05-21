import os
import sys
import uvicorn
import time
import argparse

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.model import InferenceModel

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys


class InferenceRequest(BaseModel):
    prompts: list[str]
    sampling_params: dict


class InferenceResponse(BaseModel):
    text: list[str]
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
args = parser.parse_args()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model

    model = InferenceModel(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
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
    text = model.generate(
        prompts=request.prompts,
        sampling_params=request.sampling_params,
    )
    time_taken = time.time() - start_time

    return {"text": text, "time_taken": time_taken}


@app.get("/health", response_model=bool)
async def health_check():
    global model

    return model is not None


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="localhost", port=23456)
