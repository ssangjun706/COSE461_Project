import re
import torch
import numpy as np

from .inference import InferenceModel


class RewardShaping:
    def __init__(
        self,
        base_reward: float = 1.0,
        penalty: float = -1.0,
        length_bonus: bool = True,
        diversity_bonus: bool = True,
        confidence_threshold: float = 0.8,
    ):
        self.base_reward = base_reward
        self.penalty = penalty
        self.length_bonus = length_bonus
        self.diversity_bonus = diversity_bonus
        self.confidence_threshold = confidence_threshold
        self.prompt_history = []

    def compute_reward(
        self,
        prediction: str,
        true_label: str,
        generated_prompt: str,
        confidence: float = None,
    ) -> float:
        accuracy_reward = self.base_reward if prediction == true_label else self.penalty

        length_reward = 0.0
        if self.length_bonus:
            prompt_length = len(generated_prompt.split())
            if 150 <= prompt_length <= 400:  # Task-specific prompt 적절한 길이
                length_reward = 0.1
            elif prompt_length < 100 or prompt_length > 600:  # 너무 짧거나 김
                length_reward = -0.2

        diversity_reward = 0.0
        if self.diversity_bonus and self.prompt_history:
            current_words = set(generated_prompt.lower().split())
            avg_overlap = np.mean(
                [
                    len(current_words & set(prev.lower().split()))
                    / len(current_words | set(prev.lower().split()))
                    for prev in self.prompt_history[-5:]  # 최근 5개만 비교
                ]
            )
            diversity_reward = 0.1 * (1 - avg_overlap)

        # 신뢰도 보너스
        confidence_reward = 0.0
        if confidence and confidence > self.confidence_threshold:
            confidence_reward = 0.1 * (confidence - self.confidence_threshold)

        # 프롬프트 히스토리 업데이트
        self.prompt_history.append(generated_prompt)
        if len(self.prompt_history) > 50:  # 최대 50개까지만 유지
            self.prompt_history.pop(0)

        total_reward = (
            accuracy_reward + length_reward + diversity_reward + confidence_reward
        )
        return total_reward


class RewardModel:
    def __init__(
        self,
        inference_model: InferenceModel,
        target_values: list[str],
        error_value: str = "ERROR",
        reward_correct: float = 1.0,
        reward_wrong: float = -0.3,
        reward_error: float = -1.0,
        reward_noise_scale: float = 1e-3,
        use_reward_normalization: bool = False,
    ):
        self.inference_model = inference_model
        self.target_values = target_values
        self.error_value = error_value
        
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.reward_error = reward_error
        self.reward_noise_scale = reward_noise_scale
        self.use_reward_normalization = use_reward_normalization


    def parse(self, text: str) -> str:
        match = re.search(rf'({"|".join(self.target_values)})', text)
        if match and len(match.groups()) == 1:
            return match.group(1)
        else:
            return self.error_value

    def evaluate(self, prompts: list[str], y: list[str]) -> tuple[torch.Tensor, tuple[str]]:
        y_pred = self.inference_model.generate(prompts, max_tokens=4)
        y_pred = tuple(map(self.parse, y_pred))
        
        raw_rewards = []
        for pred, true_label in zip(y_pred, y):
            if pred == self.error_value:
                raw_rewards.append(self.reward_error)
            elif pred == true_label:
                raw_rewards.append(self.reward_correct)
            else:
                raw_rewards.append(self.reward_wrong)
        
        rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32, requires_grad=False)
        if self.reward_noise_scale > 0:
            noise = torch.randn_like(rewards_tensor) * self.reward_noise_scale
            rewards_tensor = rewards_tensor + noise
        
        if self.use_reward_normalization and len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        return rewards_tensor, y_pred