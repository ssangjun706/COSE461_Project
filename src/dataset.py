import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Any


def build_meta_prompt(X: Any, target: str, target_values: list[str]) -> str:
    """
    Original complex meta-prompt (kept for comparison)
    """
    return f"""You are an expert prompt engineer who creates sophisticated natural language prompts for LLMs (Predictors), based on structured input data.

    Your task is to convert the given input data into a natural language prompt that enables a language model to accurately predict the value of the target column '{target}'.

    The generated prompt must follow these guidelines:

    1. **Must** use the input data values as the foundation, but feel free to rephrase or contextualize them naturally.
    2. Select the most relevant fields from the input that would help with prediction accuracy.
    3. Leverage common knowledge to create contextually rich prompts that frame the prediction task in a meaningful way.
    4. Direct the language model to respond with one of these specific values: {target_values}.
    5. Craft varied, sophisticated, and natural language using diverse sentence structures, appropriate domain terminology, and engaging phrasing.
    6. Consider the semantic relationships between fields to create a coherent narrative.

    Below is the input data in a structured format:

    {X}

    Now, generate a high-quality natural language prompt for predicting '{target}'.

    **IMPORTANT: The final part of your prompt must contain these exact instructions:**
    "Respond with ONLY {target_values}. Do not include any explanations, reasoning, or additional text. Your entire response must be only the prediction value."
    """


class BaseDataset(Dataset):
    """Base interface for datasets used in the fine-tuning pipeline"""

    def __init__(
        self,
        path: str,
        target: str,
        target_values: list[str],
        train: bool = True,
        shuffle: bool = True,
        train_size: float | None = 0.8,
    ):
        super().__init__()
        self.target = target
        self.target_values = target_values
        self.feature_labels = []

        self.X = None
        self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        raise NotImplementedError("Subclasses must implement this method")


class TitanicDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        train: bool = True,
        shuffle: bool = True,
        train_size: float = 0.8,
    ):
        super().__init__(
            path=path,
            target="Survived",
            target_values=["DEAD", "ALIVE"],
            train=train,
            shuffle=shuffle,
            train_size=train_size,
        )

        df = pd.read_csv(path)

        assert (
            self.target in df.columns
        ), f"Target column '{self.target}' not found in the dataframe."

        self.feature_labels = df.drop(columns=[self.target]).columns.tolist()

        X = df.drop(columns=[self.target])
        y = df[self.target]

        if train_size:
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
        else:
            self.X = X
            self.y = y

    def __getitem__(self, idx: int) -> tuple[str, str]:
        data_dict = self.X.iloc[idx].to_dict()
        X = build_meta_prompt(data_dict, self.target, self.target_values)
        y = self.target_values[self.y.iloc[idx].item()]
        return X, y
