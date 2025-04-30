import re
import numpy as np

from typing import List
from torch.utils.data import Dataset


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
