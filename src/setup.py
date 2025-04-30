import os

HF_HOME = "/DATA2/nara/.cache/huggingface"

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_METRICS_CACHE"] = os.path.join(HF_HOME, "metrics")
