import os
import sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

import src.utils
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    model_id = "Qwen/Qwen3-32B"
    snapshot_download(model_id)
