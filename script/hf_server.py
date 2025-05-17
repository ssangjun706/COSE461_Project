import os
import sys
import uvicorn
import time
import argparse

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.hf_module import HFLLMLoader

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys


class InferenceRequest(BaseModel):
    prompts: str | list[str]
    sampling_params: dict
    decode: bool | None = None


class InferenceResponse(BaseModel):
    text: list[str] | list[list[str]]
    time_taken: float


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
)
args = parser.parse_args()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model

    model = HFLLMLoader(model_name=args.model_name)

    yield
    del model


app = FastAPI(title="Prompt LLM Server", lifespan=lifespan)


@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    global model

    if not model:
        return {"error": "Model not loaded yet", "time_taken": 0}

    start_time = time.time()
    outputs = model.generate(
        prompts=request.prompts,
        sampling_params=request.sampling_params,
        decode=request.decode,
    )
    time_taken = time.time() - start_time

    return {"text": outputs, "time_taken": time_taken}


@app.get("/health", response_model=bool)
async def health_check():
    global model

    return model is not None


if __name__ == "__main__":
    uvicorn.run("hf_server:app", host="localhost", port=23458)
