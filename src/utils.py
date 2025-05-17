import requests

from typing import Any


def process(
    data: str,
    target: str,
    target_values: list[Any],
) -> list[str]:
    target_values = " or ".join(map(lambda x: f"\\boxed{{{x}}}", target_values))

    def build_meta_prompt(X_data: Any) -> str:
        return f"""You are an expert prompt engineer who creates sophisticated natural language prompts for LLMs (Predictors), based on structured input data.

Your task is to convert the given input data into an engaging, contextually rich natural language prompt that enables a language model to accurately predict the value of the target column '{target}'.

The generated prompt must follow these guidelines:

1. **Must** use the input data values as the foundation, but feel free to rephrase or contextualize them naturally.
2. Select the most relevant fields from the input that would help with prediction accuracy.
3. Leverage common knowledge to create contextually rich prompts that frame the prediction task in a meaningful way.
4. Direct the language model to respond with one of these specific values: {target_values}.
5. Craft varied, sophisticated, and natural language using diverse sentence structures, appropriate domain terminology, and engaging phrasing.
6. Ensure the prompt flows naturally while clearly communicating all relevant input information.
7. Consider the semantic relationships between fields to create a coherent narrative.
8. **Critical**: Your prompt MUST explicitly instruct the model to respond with ONLY the prediction value, with no explanations, no step-by-step reasoning, and no additional text.

Below is the input data in a structured format:

{X_data}

Now, generate a high-quality natural language prompt for predicting '{target}'.
The model should respond with one of: {target_values}.

**IMPORTANT: The final part of your prompt must contain these exact instructions:**
"Based on the information provided, respond with ONLY {target_values}. Do not include any explanations, reasoning, or additional text. Do not use phrases like 'I think' or 'The answer is'. Your entire response must be only the prediction value."
"""

    return list(map(build_meta_prompt, data))


class APIServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}"

    def request(
        self,
        prompts: list[str],
        sampling_params: dict,
        decode: bool | None = True,
    ):
        response = requests.post(
            f"{self.url}/generate",
            json={
                "prompts": prompts,
                "sampling_params": sampling_params,
                "decode": decode,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["text"]

    def check_status(self):
        try:
            response = requests.get(f"{self.url}/health")
            if response.status_code != 200 or not response.json():
                print("Error: Server is not ready")
                exit(1)
        except Exception as e:
            print(f"Error connecting to inference server: {e}")
            exit(1)
