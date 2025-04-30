import os
from dotenv import load_dotenv

HF_HOME = "/DATA2/nara/.cache/huggingface"

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_METRICS_CACHE"] = os.path.join(HF_HOME, "metrics")

from huggingface_hub import login, snapshot_download


load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
if not HF_TOKEN:
    raise ValueError("Environment variable must be set for Huggingface login.")

login(token=HF_TOKEN)


def find_model_path(model_name: str):
    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            ignore_patterns=["*.safetensors", "*.bin"],
        )

        return snapshot_path
    except Exception:
        assert (
            False
        ), f"Could not get snapshot path for {model_name} from cache. Try download again"
