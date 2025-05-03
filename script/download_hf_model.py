import os
import sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

import src.utils
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_hf_model.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]
    snapshot_download(model_id)
