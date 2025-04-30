import os
import sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

import src.setup
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def download_hf_model(model_id: str):
    assert os.environ[
        "HF_HOME"
    ], "Please ensure that 'src.setup' is properly imported and executed before running this script."

    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

    if access_token is None:
        raise ValueError(
            "Hugging Face access token is required. Please set the HUGGINGFACE_ACCESS_TOKEN environment variable."
        )

    snapshot_download(model_id, token=access_token)


if __name__ == "__main__":
    model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    download_hf_model(model_id)
