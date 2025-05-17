import os
import sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.hf_utils import list_models


if __name__ == "__main__":
    print("Downloaded models:")
    for model in list_models():
        print(f"- {model}")
