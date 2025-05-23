import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Any

# class IncomeDataset(Dataset):
#     def __init__(
#         self,
#         data,
#         target: str = "income",
#         serialize: bool = False,
#     ):
#         super().__init__()
#         self.data = data
#         self.serialize = serialize
#         self.target = target
#         self.label_values = label_values
#         assert (
#             self.data[0].get(target) is not None
#         ), f"Target column '{target}' not found in data."

#     def batch_parse(self, batch: List[str]) -> List[int | None]:
#         if isinstance(batch, str):
#             batch = [batch]

#         pattern = re.compile(r"<=50k|>50k", re.IGNORECASE)

#         def process_item(x: str) -> int:
#             match = pattern.match(x.strip())
#             if match:
#                 category = match.group(0).lower()
#                 if category == "<=50k":
#                     return 0
#                 elif category == ">50k":
#                     return 1
#             return 2

#         vectorized_process = np.vectorize(process_item)
#         result = vectorized_process(batch)

#         return result

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         X = {key: value for key, value in self.data[idx].items() if key != self.target}
#         y = self.data[idx][self.target]
#         return X, y


class TitanicDataset(Dataset):
    def __init__(
        self,
        path: str,
        train: bool = True,
        shuffle: bool = True,
        train_size: float = 0.8,
    ):
        super().__init__()
        self.target = "Survived"
        self.target_values = ["DEAD", "ALIVE"]

        df = pd.read_csv(path)

        assert (
            self.target in df.columns
        ), f"Target column '{self.target}' not found in the dataframe."

        self.feature_labels = df.drop(columns=[self.target]).columns.tolist()

        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=42,
            shuffle=shuffle,
            stratify=y,
        )

        self.X = X_train if train else X_test
        self.y = y_train if train else y_test

    def build_meta_prompt(self, X: Any) -> str:
        return f"""You are an expert prompt engineer who creates sophisticated natural language prompts for LLMs (Predictors), based on structured input data.

    Your task is to convert the given input data into an engaging, contextually rich natural language prompt that enables a language model to accurately predict the value of the target column '{self.target}'.

    The generated prompt must follow these guidelines:

    1. **Must** use the input data values as the foundation, but feel free to rephrase or contextualize them naturally.
    2. Select the most relevant fields from the input that would help with prediction accuracy.
    3. Leverage common knowledge to create contextually rich prompts that frame the prediction task in a meaningful way.
    4. Direct the language model to respond with one of these specific values: {self.target_values}.
    5. Craft varied, sophisticated, and natural language using diverse sentence structures, appropriate domain terminology, and engaging phrasing.
    6. Ensure the prompt flows naturally while clearly communicating all relevant input information.
    7. Consider the semantic relationships between fields to create a coherent narrative.
    8. **Critical**: Your prompt MUST explicitly instruct the model to respond with ONLY the prediction value, with no explanations, no step-by-step reasoning, and no additional text.

    Below is the input data in a structured format:

    {X}

    Now, generate a high-quality natural language prompt for predicting '{self.target}'.
    The model should respond with one of: {self.target_values}.

    **IMPORTANT: The final part of your prompt must contain these exact instructions:**
    "Based on the information provided, respond with ONLY {self.target_values}. Do not include any explanations, reasoning, or additional text. Do not use phrases like 'I think' or 'The answer is'. Your entire response must be only the prediction value."
    """

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_dict = self.build_meta_prompt(self.X.iloc[idx].to_dict())
        y_value = self.target_values[self.y.iloc[idx].item()]
        return X_dict, y_value
