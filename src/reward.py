import re
import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import logging

from .inference import InferenceModel


class RewardModel:
    def __init__(
        self,
        inference_model: InferenceModel,
        target_values: List[str],
        error_value: str = "ERROR",
        # Base rewards
        reward_correct: float = 1.0,
        reward_wrong: float = -0.3,
        reward_error: float = -1.0,
        # Enhanced reward components
        use_prompt_quality_scoring: bool = True,
        use_instruction_clarity: bool = True,
        use_context_utilization: bool = True,
        use_format_compliance: bool = True,
        # Weights for different components
        accuracy_weight: float = 0.4,
        quality_weight: float = 0.3,
        clarity_weight: float = 0.15,
        context_weight: float = 0.1,
        format_weight: float = 0.05,
    ):
        self.inference_model = inference_model
        self.target_values = target_values
        self.error_value = error_value
        
        # Base rewards
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.reward_error = reward_error
        
        # Feature flags
        self.use_prompt_quality_scoring = use_prompt_quality_scoring
        self.use_instruction_clarity = use_instruction_clarity
        self.use_context_utilization = use_context_utilization
        self.use_format_compliance = use_format_compliance
        
        # Component weights (should sum to 1.0)
        self.accuracy_weight = accuracy_weight
        self.quality_weight = quality_weight
        self.clarity_weight = clarity_weight
        self.context_weight = context_weight
        self.format_weight = format_weight
        
        # Normalize weights
        total_weight = (accuracy_weight + quality_weight + clarity_weight + 
                       context_weight + format_weight)
        if abs(total_weight - 1.0) > 1e-6:
            logging.warning(f"Reward weights sum to {total_weight}, normalizing...")
            self.accuracy_weight /= total_weight
            self.quality_weight /= total_weight
            self.clarity_weight /= total_weight
            self.context_weight /= total_weight
            self.format_weight /= total_weight

    def parse_prediction(self, text: str) -> str:
        """Parse prediction from generated text"""
        text_upper = text.upper()
        for value in self.target_values:
            if value.upper() in text_upper:
                return value
        
        # Try regex pattern matching
        pattern = rf'({"|".join(re.escape(val) for val in self.target_values)})'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matched_text = match.group(1).upper()
            for value in self.target_values:
                if value.upper() == matched_text:
                    return value
        
        return self.error_value

    def score_prompt_quality(self, prompt: str) -> float:
        """
        Score prompt quality based on various linguistic and structural features
        Returns score between 0 and 1
        """
        score = 0.0
        
        # Length appropriateness (optimal range: 150-400 words)
        word_count = len(prompt.split())
        if 150 <= word_count <= 400:
            score += 0.25
        elif 100 <= word_count < 150 or 400 < word_count <= 500:
            score += 0.15
        elif 80 <= word_count < 100 or 500 < word_count <= 600:
            score += 0.05
        
        # Vocabulary richness (unique words / total words)
        words = prompt.lower().split()
        if len(words) > 0:
            vocab_richness = len(set(words)) / len(words)
            score += 0.2 * vocab_richness
        
        # Sentence structure variety
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            if len(sentence_lengths) > 1:
                length_variance = np.var(sentence_lengths) / (np.mean(sentence_lengths) + 1e-8)
                score += 0.15 * min(length_variance / 10, 1.0)  # Normalize
        
        # Domain-specific terminology
        domain_terms = [
            'analyze', 'examine', 'evaluate', 'assess', 'determine', 'predict',
            'context', 'information', 'data', 'evidence', 'factors', 'patterns',
            'expert', 'analysis', 'consideration', 'based on', 'given'
        ]
        domain_score = sum(1 for term in domain_terms if term in prompt.lower())
        score += 0.2 * min(domain_score / 5, 1.0)  # Normalize to max 0.2
        
        # Coherence (avoid repetitive patterns)
        words_lower = [w.lower() for w in words]
        word_freq = Counter(words_lower)
        max_freq = max(word_freq.values()) if word_freq else 1
        repetition_penalty = max(0, (max_freq - 3) * 0.05)  # Penalize words appearing >3 times
        score -= repetition_penalty
        
        return max(0.0, min(1.0, score))

    def score_instruction_clarity(self, prompt: str) -> float:
        """
        Score how clear and direct the instructions are
        Returns score between 0 and 1
        """
        score = 0.0
        prompt_lower = prompt.lower()
        
        # Clear instruction phrases
        instruction_phrases = [
            "respond with only", "answer with only", "reply with only",
            "provide only", "give only", "output only"
        ]
        if any(phrase in prompt_lower for phrase in instruction_phrases):
            score += 0.4
        
        # Prohibition of explanations
        prohibition_phrases = [
            "do not include", "no explanations", "no reasoning", 
            "no additional text", "without explanation", "don't explain"
        ]
        if any(phrase in prompt_lower for phrase in prohibition_phrases):
            score += 0.3
        
        # Clear format specification
        if any(val.lower() in prompt_lower for val in self.target_values):
            score += 0.2
        
        # Imperative mood (commands)
        imperative_indicators = [
            "predict", "determine", "classify", "identify", "select",
            "choose", "decide", "analyze", "evaluate"
        ]
        if any(indicator in prompt_lower for indicator in imperative_indicators):
            score += 0.1
        
        return min(1.0, score)

    def score_context_utilization(self, prompt: str) -> float:
        """
        Score how well the prompt utilizes provided context/data
        Returns score between 0 and 1
        """
        score = 0.0
        prompt_lower = prompt.lower()
        
        # References to provided data
        data_references = [
            "provided data", "given information", "input data", "the data",
            "information provided", "based on", "using the", "from the data"
        ]
        data_ref_score = sum(1 for ref in data_references if ref in prompt_lower)
        score += 0.4 * min(data_ref_score / 3, 1.0)
        
        # Contextual framing
        context_phrases = [
            "given the context", "in this context", "considering the",
            "taking into account", "based on the information"
        ]
        context_score = sum(1 for phrase in context_phrases if phrase in prompt_lower)
        score += 0.3 * min(context_score / 2, 1.0)
        
        # Specific field references (would need to be customized per dataset)
        # For Titanic dataset example:
        field_indicators = [
            "age", "sex", "class", "fare", "embarked", "passenger",
            "survival", "demographic", "socioeconomic"
        ]
        field_score = sum(1 for field in field_indicators if field in prompt_lower)
        score += 0.3 * min(field_score / 3, 1.0)
        
        return min(1.0, score)

    def score_format_compliance(self, prompt: str) -> float:
        """
        Score adherence to expected format and structure
        Returns score between 0 and 1
        """
        score = 0.0
        
        # Has clear beginning and end
        if len(prompt.strip()) > 50:  # Minimum length
            score += 0.2
        
        # Proper sentence structure
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        if len(sentences) >= 2:  # At least 2 sentences
            score += 0.3
        
        # Ends with clear instruction
        last_part = prompt.lower()[-200:]  # Last 200 characters
        if any(phrase in last_part for phrase in ["respond with", "answer with", "provide"]):
            score += 0.3
        
        # Proper capitalization and punctuation
        if prompt[0].isupper() and prompt.endswith('.'):
            score += 0.2
        
        return min(1.0, score)

    def compute_comprehensive_reward(self, prompt: str, prediction: str, true_label: str) -> Dict[str, float]:
        """
        Compute comprehensive reward with multiple components
        Returns dictionary with component scores and total reward
        """
        components = {}
        
        # 1. Accuracy component
        if prediction == self.error_value:
            accuracy_score = self.reward_error
        elif prediction == true_label:
            accuracy_score = self.reward_correct
        else:
            accuracy_score = self.reward_wrong
        components['accuracy'] = accuracy_score
        
        # 2. Prompt quality component
        if self.use_prompt_quality_scoring:
            quality_score = self.score_prompt_quality(prompt)
            components['quality'] = quality_score
        else:
            components['quality'] = 0.5  # Neutral
        
        # 3. Instruction clarity component
        if self.use_instruction_clarity:
            clarity_score = self.score_instruction_clarity(prompt)
            components['clarity'] = clarity_score
        else:
            components['clarity'] = 0.5  # Neutral
        
        # 4. Context utilization component
        if self.use_context_utilization:
            context_score = self.score_context_utilization(prompt)
            components['context'] = context_score
        else:
            components['context'] = 0.5  # Neutral
        
        # 5. Format compliance component
        if self.use_format_compliance:
            format_score = self.score_format_compliance(prompt)
            components['format'] = format_score
        else:
            components['format'] = 0.5  # Neutral
        
        # Compute weighted total reward
        total_reward = (
            self.accuracy_weight * accuracy_score +
            self.quality_weight * components['quality'] +
            self.clarity_weight * components['clarity'] +
            self.context_weight * components['context'] +
            self.format_weight * components['format']
        )
        
        components['total'] = total_reward
        return components

    def evaluate(self, prompts: List[str], y: List[str]) -> Tuple[torch.Tensor, Tuple[str]]:
        """
        Evaluate prompts and return comprehensive rewards
        """
        # Get predictions from inference model
        y_pred = self.inference_model.generate(prompts, {"max_tokens": 8})
        y_pred = tuple(map(self.parse_prediction, y_pred))
        
        # Compute comprehensive rewards
        all_rewards = []
        detailed_scores = []
        
        for prompt, pred, true_label in zip(prompts, y_pred, y):
            reward_components = self.compute_comprehensive_reward(prompt, pred, true_label)
            all_rewards.append(reward_components['total'])
            detailed_scores.append(reward_components)
        
        # Convert to tensor
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, requires_grad=False)
        
        # Log some statistics for debugging
        if len(detailed_scores) > 0:
            avg_components = {}
            for key in detailed_scores[0].keys():
                avg_components[key] = np.mean([score[key] for score in detailed_scores])
            
            logging.debug(f"Average reward components: {avg_components}")
        
        return rewards_tensor, y_pred

    def get_reward_statistics(self, prompts: List[str], y: List[str]) -> Dict[str, float]:
        """
        Get detailed statistics about reward components
        """
        y_pred = self.inference_model.generate(prompts, {"max_tokens": 8})
        y_pred = list(map(self.parse_prediction, y_pred))
        
        all_components = []
        for prompt, pred, true_label in zip(prompts, y_pred, y):
            components = self.compute_comprehensive_reward(prompt, pred, true_label)
            all_components.append(components)
        
        # Compute statistics
        stats = {}
        if all_components:
            for key in all_components[0].keys():
                values = [comp[key] for comp in all_components]
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_min"] = np.min(values)
                stats[f"{key}_max"] = np.max(values)
        
        return stats 