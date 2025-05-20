import re
import numpy as np

from typing import List
from torch.utils.data import Dataset

import pandas as pd
from sklearn.model_selection import train_test_split


class IncomeDataset(Dataset):
    def __init__(
        self,
        data,
        target: str = "income",
        serialize: bool = False,
        label_values: List[str] = ["<=50K", ">50K"],
    ):
        super().__init__()
        self.data = data
        self.serialize = serialize
        self.target = target
        self.label_values = label_values
        assert (
            self.data[0].get(target) is not None
        ), f"Target column '{target}' not found in data."

    def batch_parse(self, batch: List[str]) -> List[int | None]:
        if isinstance(batch, str):
            batch = [batch]

        pattern = re.compile(r"<=50k|>50k", re.IGNORECASE)

        def process_item(x: str) -> int:
            match = pattern.match(x.strip())
            if match:
                category = match.group(0).lower()
                if category == "<=50k":
                    return 0
                elif category == ">50k":
                    return 1
            return 2

        vectorized_process = np.vectorize(process_item)
        result = vectorized_process(batch)

        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = {key: value for key, value in self.data[idx].items() if key != self.target}
        y = self.data[idx][self.target]
        # label = 0 if self.data[idx]["income"] == "<=50K" else 1
        # x = self._serialize(self.data[idx]) if self.serialize else self.data[idx]
        # return x, label
        return X, y

    # def _serialize(self, data):
    #     text = "Read a given information and questions. Think step by step, and then choose the most important feature to predict whether a person's income is '<=50K' or '>50K'. You must choose one from [age, workclass, education, education.num, marital.status, occupation, relationship, race, sex, capital.gain, capital.loss, hours.per.week, native.country]."
    #     text += "\nThe dataset consists of 13 input variables: age, workclass, education, education.num, marital.status, occupation, relationship, race, sex, capital.gain, capital.loss, hours.per.week, and native.country. The output variable is: income, where '<=50K' indicates the person earns 50K or less annually, and '>50K' indicates the person earns more than 50K annually.\n"
    #     text += "Question: If the "
    #     text += f"age is {data['age']}, "
    #     text += f"workclass is '{data['workclass']}', "
    #     text += f"education is '{data['education']}', "
    #     text += f"education.num is {data['education.num']}, "
    #     text += f"marital.status is '{data['marital.status']}', "
    #     text += f"occupation is '{data['occupation']}', "
    #     text += f"relationship is '{data['relationship']}', "
    #     text += f"race is '{data['race']}', "
    #     text += f"sex is '{data['sex']}', "
    #     text += f"capital.gain is {data['capital.gain']}, "
    #     text += f"capital.loss is {data['capital.loss']}, "
    #     text += f"hours.per.week is {data['hours.per.week']}, "
    #     text += f"native.country is '{data['native.country']}', "
    #     text += f"then what is the person's income? Choose between ['<=50K', '>50K'].\n"
    #     text += "Answer: "
    #     return text


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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X_dict = {k: v for k, v in self.X.iloc[idx].to_dict().items()}
        X_dict = str(self.X.iloc[idx].to_dict())
        y_value = self.y.iloc[idx].item()
        return X_dict, y_value
