from typing import Any


def process(
    data: str,
    target: str,
    target_values: list[Any],
) -> list[str]:
    target_values = "or ".join(map(str, target_values))

    def build_meta_prompt(X_data: Any) -> str:
        return f"""You are a system that generates natural language prompts for LLM (Predictor), based on structured input data.

Your task is to convert the given input data into a natural language prompt that allows a LLM to predict the value of the target column '{target}'.

The generated prompt must follow these rules:

1. Do not modify the input data. Use the values exactly as provided.
2. Include *some* (or *all*) fields and their values from the input.
3. The prompt must be self-contained and must not assume external knowledge.
4. Encourage the language model to respond *only* with one of the following values: {target_values}.
5. Express the prompt in varied and natural English, such as a question or instruction.
6. The prompt should be in complete, fluent sentences and must mention all input fields clearly.

Below is the input data in a structured format:

{X_data}

Now, generate a natural language prompt suitable for predicting '{target}'.
The model's response should be one of: {target_values}.
Only output the prompt, without explanation.
"""

    return list(map(build_meta_prompt, data))
