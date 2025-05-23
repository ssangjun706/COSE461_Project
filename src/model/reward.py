import re
import torch

from src.model import APIServer


class RewardModel:
    def __init__(
        self,
        api_server: APIServer,
        target_values: list[str],
        reward: float = 1.0,
        penalty: float = 0.0,
        error_value: str = "ERROR",
        error_penalty: float = 0.0,
    ):
        self.api_server = api_server
        self.target_values = target_values
        self.reward = reward
        self.penalty = penalty
        self.error_value = error_value
        self.error_penalty = error_penalty
        self.sampling_params = {
            "n": 1,
            "max_new_tokens": 4,
        }

    def parse(self, text: str) -> str:
        match = re.search(rf'({"|".join(self.target_values)})', text)
        if match and len(match.groups()) == 1:
            return match.group(1)
        else:
            return self.error_value

    def evaluate(self, prompts: list[str], y: list[str]) -> torch.Tensor:
        y_pred = self.api_server.request(prompts, self.sampling_params)

        rewards = [
            self.reward if self.parse(a) == b else self.penalty
            for a, b in zip(y_pred, y)
        ]

        rewards = torch.tensor(rewards)
        return rewards
