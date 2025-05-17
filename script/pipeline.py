import os
import sys
import re

from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Environment variable must be set for Gemini API key.")


from torch.utils.data import DataLoader
from tqdm import tqdm

# from google import genai


root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.utils import process, APIServer

from src.dataset import TitanicDataset


DATASET_PATH = "../dataset/titanic-dataset.csv"
CHOICE = ["DEAD", "ALIVE"]

BATCH_SIZE = 16

PROMPT_NUM_SEQUENCES = 1
PROMPT_MAX_TOKENS = 2048
# PROMPT_TEMPERATURE = 0.15

INFERENCE_NUM_SEQUENCES = 1
INFERENCE_MAX_TOKENS = 32
# INFERENCE_TEMPERATURE = 0.15

PROMPT_SAMPLING_PARAMS = {
    "n": PROMPT_NUM_SEQUENCES,
    "max_tokens": PROMPT_MAX_TOKENS,
    # "temperature": PROMPT_TEMPERATURE,
}

INFERENCE_SAMPLING_PARAMS = {
    "n": INFERENCE_NUM_SEQUENCES,
    "max_tokens": INFERENCE_MAX_TOKENS,
    # "temperature": INFERENCE_TEMPERATURE,
}


def parse(text):
    match = re.search(r"\\boxed\{(DEAD|ALIVE)\}", text)
    if match and len(match.groups()) == 1:
        return match.group(1)
    else:
        return "ERROR"


if __name__ == "__main__":
    inference_server = APIServer("localhost", 23456)
    prompt_server = APIServer("localhost", 23457)
    # prompt_server = APIServer("localhost", 23457)
    # inference_server = genai.Client(api_key=GEMINI_API_KEY)

    inference_server.check_status()
    prompt_server.check_status()

    dataset = TitanicDataset(DATASET_PATH, train=False, target_values=CHOICE)
    target, target_values = dataset.target, dataset.target_values
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_predictions = 0
    total_parse_error = 0

    for X, y in tqdm(data_loader):
        # processing input(json) into prompt
        meta_prompts = process(
            data=X,
            target=target,
            target_values=target_values,
        )

        # Fine-tuning the model
        # ...

        # For debugging purposes
        answer = ["ALIVE" if y_i == 1 else "DEAD" for y_i in y]

        prompts = prompt_server.request(meta_prompts, PROMPT_SAMPLING_PARAMS)
        flatten_prompts = [p for sample in prompts for p in sample]

        # fn = lambda x: inference_server.models.generate_content(
        #     model="gemini-2.5-flash-preview-04-17",
        #     contents=x,
        #     # config={
        #     #     "response_mime_type": "text/x.enum",
        #     #     "response_schema": Survival,
        #     # },
        # )

        y_pred = inference_server.request(flatten_prompts, INFERENCE_SAMPLING_PARAMS)
        y_pred = [parse(r) for y in y_pred for r in y]

        # Calculate parsing error rate
        parsing_errors = sum(1 for pred in y_pred if pred == "ERROR")
        correct = sum(1 for a, b in zip(answer, y_pred) if a == b)
        print(f"Parsing Error: {parsing_errors}/{BATCH_SIZE}")
        print(f"Correct: {correct}/{BATCH_SIZE}")
        total_predictions += correct
        total_parse_error += parsing_errors

        # print("Press Enter to continue...")
        # input()

    # Accuracy: 0.720 (local attempt)
    # Parsing Error: 10/179 (local attempt)

    # Accuracy: 0.792 (api attempt)
    print(f"Accuracy: {total_predictions / len(data_loader.dataset)}")
    print(f"Parsing Error: {total_parse_error}/{len(data_loader.dataset)}")
