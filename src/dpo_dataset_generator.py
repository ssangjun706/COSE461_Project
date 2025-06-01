import json
import os
import random
import logging
from typing import List, Dict, Tuple, Any
from collections import Counter
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from .inference import InferenceModel
from .dataset import build_meta_prompt


class DPODatasetGenerator:
    """
    DPO 학습을 위한 데이터셋 생성기 (개선된 버전)
    
    파이프라인:
    1. Meta Prompt + X data → Generator(LLM) → Generated Prompt
    2. Generated Prompt → Predictor(LLM) → y_pred (n번 수행하여 일관성 체크)
    3. 일관성과 정확성에 따라 accepted/rejected pair를 만듭니다.
    """
    
    def __init__(
        self,
        generator_model: InferenceModel,
        predictor_model: InferenceModel,
        target_values: List[str],
        target_column: str,
        error_value: str = "ERROR",
        num_predictions_per_prompt: int = 10,
        consistency_threshold: float = 0.75,  # 75% 이상 같은 결과
        error_to_wrong_ratio: float = 0.4,    # ERROR:틀린케이스 = 4:6 비율
        output_dir: str = "dataset"
    ):
        self.generator_model = generator_model
        self.predictor_model = predictor_model
        self.target_values = target_values
        self.target_column = target_column
        self.error_value = error_value
        self.num_predictions_per_prompt = num_predictions_per_prompt
        self.consistency_threshold = consistency_threshold
        self.error_to_wrong_ratio = error_to_wrong_ratio
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )
    
    def parse_prediction(self, prediction: str) -> str:
        """
        예측 결과를 파싱하여 타겟 값 또는 ERROR 반환
        """
        prediction = prediction.strip().upper()
        
        # 정확한 타겟 값이 포함되어 있는지 확인
        for target_value in self.target_values:
            if target_value.upper() in prediction:
                return target_value
        
        # 타겟 값이 없으면 ERROR
        return self.error_value
    
    def evaluate_prompt_consistency(
        self, 
        generated_prompt: str, 
        true_label: str
    ) -> Tuple[str, float, List[str], str]:
        """
        생성된 프롬프트의 일관성을 평가
        
        Args:
            generated_prompt: 생성된 프롬프트
            true_label: 실제 정답
            
        Returns:
            (final_prediction, consistency_score, predictions, category)
            category: "chosen", "rejected_error", "rejected_wrong"
        """
        try:
            # "n" 파라미터를 사용하여 한 번에 여러 번 예측
            raw_predictions = self.predictor_model.generate(
                [generated_prompt],  # 단일 프롬프트
                {
                    "max_tokens": 8,
                    "n": self.num_predictions_per_prompt  # 한 번에 여러 번 예측
                }
            )
            
            # 예측 결과 파싱
            parsed_predictions = [
                self.parse_prediction(pred) 
                for pred in raw_predictions
            ]
            
            # 일관성 체크: 가장 많이 나온 결과와 그 비율
            prediction_counts = Counter(parsed_predictions)
            most_common_prediction, most_common_count = prediction_counts.most_common(1)[0]
            consistency_score = most_common_count / len(parsed_predictions)
            
            # 일관성이 threshold를 넘지 못하면 rejected
            if consistency_score < self.consistency_threshold:
                return most_common_prediction, consistency_score, parsed_predictions, "rejected_error"
            
            # ERROR가 가장 많이 나왔으면 rejected
            if most_common_prediction == self.error_value:
                return most_common_prediction, consistency_score, parsed_predictions, "rejected_error"
            
            # 정답과 다르면 rejected
            if most_common_prediction != true_label:
                return most_common_prediction, consistency_score, parsed_predictions, "rejected_wrong"
            
            # 정답이고 일관성도 높으면 chosen
            return most_common_prediction, consistency_score, parsed_predictions, "chosen"
            
        except Exception as e:
            logging.error(f"Error evaluating prompt: {e}")
            return self.error_value, 0.0, [self.error_value] * self.num_predictions_per_prompt, "rejected_error"
    
    def generate_prompts_from_data(
        self, 
        x_data: List[Dict[str, Any]], 
        num_prompts_per_data: int = 5
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        X data로부터 여러 개의 generated prompt를 생성
        """
        results = []
        
        for x_dict in tqdm(x_data, desc="Generating prompts"):
            # Meta prompt 생성
            meta_prompt = build_meta_prompt(x_dict, self.target_column, self.target_values)
            
            try:
                # "n" 파라미터를 사용하여 한 번에 여러 개 생성
                generated_prompts = self.generator_model.generate(
                    [meta_prompt],  # 단일 프롬프트
                    {
                        "max_tokens": 512,
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "n": num_prompts_per_data  # 한 번에 여러 개 생성
                    }
                )
                
                for generated_prompt in generated_prompts:
                    results.append((meta_prompt, x_dict, generated_prompt.strip()))
                    
            except Exception as e:
                logging.error(f"Error generating prompts: {e}")
                continue
        
        return results
    
    def generate_prompts_from_data_batch(
        self, 
        x_data: List[Dict[str, Any]], 
        num_prompts_per_data: int = 5,
        batch_size: int = 8
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        X data로부터 배치 단위로 여러 개의 generated prompt를 생성 (더 빠름)
        """
        results = []
        
        # 데이터를 배치로 나누기
        for i in tqdm(range(0, len(x_data), batch_size), desc="Generating prompts (batch)"):
            batch_x_data = x_data[i:i+batch_size]
            batch_meta_prompts = []
            batch_x_dicts = []
            
            # 배치용 meta prompt 생성
            for x_dict in batch_x_data:
                meta_prompt = build_meta_prompt(x_dict, self.target_column, self.target_values)
                batch_meta_prompts.append(meta_prompt)
                batch_x_dicts.append(x_dict)
            
            try:
                # 배치로 한 번에 여러 개 생성
                all_generated_prompts = self.generator_model.generate(
                    batch_meta_prompts,
                    {
                        "max_tokens": 512,
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "n": num_prompts_per_data  # 각 프롬프트당 n개씩 생성
                    }
                )
                
                # 결과를 정리 (각 입력당 num_prompts_per_data개씩 나옴)
                for j, (meta_prompt, x_dict) in enumerate(zip(batch_meta_prompts, batch_x_dicts)):
                    start_idx = j * num_prompts_per_data
                    end_idx = start_idx + num_prompts_per_data
                    generated_prompts = all_generated_prompts[start_idx:end_idx]
                    
                    for generated_prompt in generated_prompts:
                        results.append((meta_prompt, x_dict, generated_prompt.strip()))
                        
            except Exception as e:
                logging.error(f"Error generating prompts for batch {i//batch_size}: {e}")
                # 에러 시 개별 처리로 폴백
                for x_dict in batch_x_data:
                    meta_prompt = build_meta_prompt(x_dict, self.target_column, self.target_values)
                    try:
                        generated_prompts = self.generator_model.generate(
                            [meta_prompt],
                            {
                                "max_tokens": 512,
                                "temperature": 0.8,
                                "top_p": 0.9,
                                "n": num_prompts_per_data
                            }
                        )
                        for generated_prompt in generated_prompts:
                            results.append((meta_prompt, x_dict, generated_prompt.strip()))
                    except Exception as e2:
                        logging.error(f"Error generating prompt for individual item: {e2}")
                        continue
        
        return results
    
    def evaluate_prompts_consistency_batch(
        self, 
        prompt_data: List[Tuple[str, Dict[str, Any], str]], 
        y_true: List[str],
        batch_size: int = 16
    ) -> List[Tuple[str, Dict[str, Any], str, str, float, List[str], str]]:
        """
        배치 단위로 프롬프트 일관성을 평가 (더 빠름)
        
        Returns:
            (meta_prompt, x_dict, generated_prompt, final_prediction, consistency_score, predictions, category) 리스트
        """
        results = []
        
        # x_data와 y_true 매핑
        data_to_label = {}
        for i, (_, x_dict, _) in enumerate(prompt_data):
            x_key = str(sorted(x_dict.items()))
            if x_key not in data_to_label:
                # 해당 x_dict에 맞는 y_true 찾기
                for j, y in enumerate(y_true):
                    if j < len(prompt_data):
                        other_x_dict = prompt_data[j][1]
                        if str(sorted(other_x_dict.items())) == x_key:
                            data_to_label[x_key] = y
                            break
                if x_key not in data_to_label:
                    data_to_label[x_key] = y_true[0] if y_true else "UNKNOWN"
        
        # 배치로 처리
        for i in tqdm(range(0, len(prompt_data), batch_size), desc="Evaluating prompts (batch)"):
            batch_prompt_data = prompt_data[i:i+batch_size]
            batch_prompts = []
            batch_true_labels = []
            
            for meta_prompt, x_dict, generated_prompt in batch_prompt_data:
                batch_prompts.append(generated_prompt)
                x_key = str(sorted(x_dict.items()))
                true_label = data_to_label.get(x_key, y_true[0] if y_true else "UNKNOWN")
                batch_true_labels.append(true_label)
            
            try:
                # 배치로 한 번에 여러 번 예측
                all_raw_predictions = self.predictor_model.generate(
                    batch_prompts,
                    {
                        "max_tokens": 8,
                        "n": self.num_predictions_per_prompt
                    }
                )
                
                # 결과를 정리
                for j, (meta_prompt, x_dict, generated_prompt) in enumerate(batch_prompt_data):
                    start_idx = j * self.num_predictions_per_prompt
                    end_idx = start_idx + self.num_predictions_per_prompt
                    raw_predictions = all_raw_predictions[start_idx:end_idx]
                    
                    # 예측 결과 파싱 및 일관성 체크
                    parsed_predictions = [
                        self.parse_prediction(pred) 
                        for pred in raw_predictions
                    ]
                    
                    true_label = batch_true_labels[j]
                    
                    # 일관성 체크
                    prediction_counts = Counter(parsed_predictions)
                    most_common_prediction, most_common_count = prediction_counts.most_common(1)[0]
                    consistency_score = most_common_count / len(parsed_predictions)
                    
                    # 카테고리 결정
                    if consistency_score < self.consistency_threshold:
                        category = "rejected_error"
                    elif most_common_prediction == self.error_value:
                        category = "rejected_error"
                    elif most_common_prediction != true_label:
                        category = "rejected_wrong"
                    else:
                        category = "chosen"
                    
                    results.append((meta_prompt, x_dict, generated_prompt, most_common_prediction, 
                                  consistency_score, parsed_predictions, category))
                    
            except Exception as e:
                logging.error(f"Error evaluating batch {i//batch_size}: {e}")
                # 에러 시 개별 처리로 폴백
                for meta_prompt, x_dict, generated_prompt in batch_prompt_data:
                    x_key = str(sorted(x_dict.items()))
                    true_label = data_to_label.get(x_key, y_true[0] if y_true else "UNKNOWN")
                    final_prediction, consistency_score, predictions, category = self.evaluate_prompt_consistency(
                        generated_prompt, true_label
                    )
                    results.append((meta_prompt, x_dict, generated_prompt, final_prediction, 
                                  consistency_score, predictions, category))
        
        return results
    
    def create_dpo_pairs_balanced(
        self,
        prompt_results: List[Tuple[str, Dict[str, Any], str, str, float, List[str], str]],
        true_labels: List[str],
        fallback_strategy: str = "adaptive"  # "adaptive", "relative", "strict"
    ) -> List[Dict[str, Any]]:
        """
        ERROR와 틀린 케이스의 배분을 고려한 DPO 페어 생성
        
        Args:
            prompt_results: (meta_prompt, x_dict, generated_prompt, final_prediction, consistency_score, predictions, category) 리스트
            true_labels: 실제 정답 리스트
            fallback_strategy: chosen이 부족할 때의 전략
                - "adaptive": threshold를 낮춰서 chosen 확보
                - "relative": 상대적으로 가장 좋은 것을 chosen으로 선택
                - "strict": chosen이 없으면 빈 리스트 반환
            
        Returns:
            DPO 학습용 데이터셋
        """
        dpo_pairs = []
        
        # 카테고리별로 분류
        chosen_prompts = []
        rejected_error_prompts = []
        rejected_wrong_prompts = []
        
        for result in prompt_results:
            meta_prompt, x_dict, generated_prompt, final_prediction, consistency_score, predictions, category = result
            
            prompt_info = {
                'meta_prompt': meta_prompt,
                'x_dict': x_dict,
                'generated_prompt': generated_prompt,
                'final_prediction': final_prediction,
                'consistency_score': consistency_score,
                'predictions': predictions
            }
            
            if category == "chosen":
                chosen_prompts.append(prompt_info)
            elif category == "rejected_error":
                rejected_error_prompts.append(prompt_info)
            elif category == "rejected_wrong":
                rejected_wrong_prompts.append(prompt_info)
        
        logging.info(f"Chosen prompts: {len(chosen_prompts)}")
        logging.info(f"Rejected ERROR prompts: {len(rejected_error_prompts)}")
        logging.info(f"Rejected WRONG prompts: {len(rejected_wrong_prompts)}")
        
        # chosen이 없는 경우 fallback 전략 적용
        if len(chosen_prompts) == 0:
            logging.warning("No chosen prompts found! Applying fallback strategy...")
            
            if fallback_strategy == "strict":
                logging.warning("Using strict strategy - returning empty dataset")
                return []
            
            elif fallback_strategy == "adaptive":
                logging.info("Using adaptive strategy - lowering consistency threshold")
                # 일관성 점수로 재분류 (임계값을 점진적으로 낮춤)
                all_prompts = rejected_error_prompts + rejected_wrong_prompts
                
                # 정답과 일치하는 프롬프트들 중에서 일관성이 가장 높은 것들을 선택
                correct_prompts = []
                for prompt_info in all_prompts:
                    # true_label 찾기
                    x_key = str(sorted(prompt_info['x_dict'].items()))
                    true_label = "UNKNOWN"
                    for i, y in enumerate(true_labels):
                        if i < len(prompt_results):
                            other_x_dict = prompt_results[i][1]
                            if str(sorted(other_x_dict.items())) == x_key:
                                true_label = y
                                break
                    
                    if prompt_info['final_prediction'] == true_label:
                        correct_prompts.append(prompt_info)
                
                if correct_prompts:
                    # 일관성 점수로 정렬하여 상위 몇 개를 chosen으로 승격
                    correct_prompts.sort(key=lambda x: x['consistency_score'], reverse=True)
                    num_to_promote = min(len(correct_prompts) // 2, 10)  # 최대 10개 또는 절반
                    chosen_prompts = correct_prompts[:num_to_promote]
                    
                    # 나머지는 rejected로 유지
                    remaining_prompts = correct_prompts[num_to_promote:]
                    for prompt in remaining_prompts:
                        if prompt['final_prediction'] == self.error_value:
                            rejected_error_prompts.append(prompt)
                        else:
                            rejected_wrong_prompts.append(prompt)
                    
                    logging.info(f"Promoted {len(chosen_prompts)} prompts to chosen using adaptive strategy")
                
            elif fallback_strategy == "relative":
                logging.info("Using relative strategy - selecting best among available")
                # 모든 프롬프트를 일관성 점수로 정렬
                all_prompts = rejected_error_prompts + rejected_wrong_prompts
                
                if all_prompts:
                    # 정답과 일치하는 것들 중 상위 것들을 chosen으로 선택
                    correct_prompts = []
                    for prompt_info in all_prompts:
                        # true_label 찾기 (위와 동일한 로직)
                        x_key = str(sorted(prompt_info['x_dict'].items()))
                        true_label = "UNKNOWN"
                        for i, y in enumerate(true_labels):
                            if i < len(prompt_results):
                                other_x_dict = prompt_results[i][1]
                                if str(sorted(other_x_dict.items())) == x_key:
                                    true_label = y
                                    break
                        
                        if prompt_info['final_prediction'] == true_label:
                            correct_prompts.append(prompt_info)
                    
                    if correct_prompts:
                        # 상위 30%를 chosen으로 선택
                        correct_prompts.sort(key=lambda x: x['consistency_score'], reverse=True)
                        num_chosen = max(1, len(correct_prompts) // 3)
                        chosen_prompts = correct_prompts[:num_chosen]
                        logging.info(f"Selected {len(chosen_prompts)} prompts as chosen using relative strategy")
        
        # 여전히 chosen이 없으면 빈 리스트 반환
        if len(chosen_prompts) == 0:
            logging.error("Still no chosen prompts after fallback strategy!")
            return []
        
        logging.info(f"Final counts - Chosen: {len(chosen_prompts)}, ERROR: {len(rejected_error_prompts)}, WRONG: {len(rejected_wrong_prompts)}")
        
        # 각 chosen prompt에 대해 rejected pair 생성
        # 전체 DPO 페어에서 ERROR:WRONG 비율을 맞추기 위한 계산
        total_pairs_needed = len(chosen_prompts)
        target_error_count = int(total_pairs_needed * self.error_to_wrong_ratio)
        target_wrong_count = total_pairs_needed - target_error_count
        
        # 실제 사용 가능한 수량 확인
        available_error_count = len(rejected_error_prompts)
        available_wrong_count = len(rejected_wrong_prompts)
        
        # 사용 가능한 수량에 맞춰 조정
        actual_error_count = min(target_error_count, available_error_count)
        actual_wrong_count = min(target_wrong_count, available_wrong_count)
        
        # 부족한 경우 다른 타입으로 보충
        remaining_pairs = total_pairs_needed - actual_error_count - actual_wrong_count
        if remaining_pairs > 0:
            if available_error_count > actual_error_count:
                additional_error = min(remaining_pairs, available_error_count - actual_error_count)
                actual_error_count += additional_error
                remaining_pairs -= additional_error
            
            if remaining_pairs > 0 and available_wrong_count > actual_wrong_count:
                additional_wrong = min(remaining_pairs, available_wrong_count - actual_wrong_count)
                actual_wrong_count += additional_wrong
        
        logging.info(f"Target ratio - ERROR: {target_error_count}, WRONG: {target_wrong_count}")
        logging.info(f"Actual ratio - ERROR: {actual_error_count}, WRONG: {actual_wrong_count}")
        
        # rejected 프롬프트들을 미리 섞어서 선택
        shuffled_error_prompts = random.sample(rejected_error_prompts, min(actual_error_count, len(rejected_error_prompts)))
        shuffled_wrong_prompts = random.sample(rejected_wrong_prompts, min(actual_wrong_count, len(rejected_wrong_prompts)))
        
        # 전체 rejected 리스트 생성 (비율에 맞춰)
        all_rejected = shuffled_error_prompts + shuffled_wrong_prompts
        random.shuffle(all_rejected)  # 순서 섞기
        
        # chosen과 rejected 페어링
        for i, chosen in enumerate(chosen_prompts):
            if i < len(all_rejected):
                rejected = all_rejected[i]
                
                # true_label 찾기
                x_key = str(sorted(chosen['x_dict'].items()))
                true_label = "UNKNOWN"
                for j, y in enumerate(true_labels):
                    if j < len(prompt_results):
                        other_x_dict = prompt_results[j][1]
                        if str(sorted(other_x_dict.items())) == x_key:
                            true_label = y
                            break
                
                dpo_pairs.append({
                    'prompt': chosen['meta_prompt'],
                    'chosen': chosen['generated_prompt'],
                    'rejected': rejected['generated_prompt'],
                    'chosen_prediction': chosen['final_prediction'],
                    'rejected_prediction': rejected['final_prediction'],
                    'chosen_consistency': chosen['consistency_score'],
                    'rejected_consistency': rejected['consistency_score'],
                    'x_data': chosen['x_dict'],
                    'true_label': true_label,
                    'rejected_type': 'error' if rejected in shuffled_error_prompts else 'wrong'
                })
        
        return dpo_pairs
    
    def generate_dataset_from_dataset_class(
        self,
        dataset_class,
        dataset_path: str,
        num_prompts_per_data: int = 5,
        use_batch_processing: bool = True,
        generation_batch_size: int = 8,
        evaluation_batch_size: int = 16,
        fallback_strategy: str = "adaptive",
        patience: int = 3,
        min_success_rate: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Dataset 클래스로부터 train/test DPO 데이터셋을 생성
        
        Args:
            dataset_class: Dataset 클래스 (예: TitanicDataset)
            dataset_path: 데이터셋 파일 경로
            num_prompts_per_data: 각 데이터에 대해 생성할 프롬프트 수
            use_batch_processing: 배치 처리 사용 여부
            generation_batch_size: 프롬프트 생성용 배치 크기
            evaluation_batch_size: 프롬프트 평가용 배치 크기
            fallback_strategy: chosen이 부족할 때의 전략
            patience: 재시도 횟수
            min_success_rate: 최소 성공률
            
        Returns:
            (train_dpo_dataset, test_dpo_dataset) 튜플
        """
        logging.info("Starting DPO dataset generation from dataset class...")
        
        # Train 데이터셋 생성
        logging.info("Processing train dataset...")
        train_dataset = dataset_class(dataset_path, train=True)
        train_x_data, train_y_true = self._extract_data_from_dataset(train_dataset)
        train_dpo_dataset = self._generate_dataset_internal_with_patience(
            train_x_data, train_y_true, num_prompts_per_data,
            use_batch_processing, generation_batch_size, evaluation_batch_size, 
            fallback_strategy, patience, min_success_rate
        )
        
        # Test 데이터셋 생성  
        logging.info("Processing test dataset...")
        test_dataset = dataset_class(dataset_path, train=False)
        test_x_data, test_y_true = self._extract_data_from_dataset(test_dataset)
        test_dpo_dataset = self._generate_dataset_internal_with_patience(
            test_x_data, test_y_true, num_prompts_per_data,
            use_batch_processing, generation_batch_size, evaluation_batch_size, 
            fallback_strategy, patience, min_success_rate
        )
        
        return train_dpo_dataset, test_dpo_dataset
    
    def _extract_data_from_dataset(self, dataset) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Dataset 객체에서 x_data와 y_true 추출"""
        x_data = []
        y_true = []
        
        for i in range(len(dataset)):
            _, y = dataset[i]  # meta_prompt는 무시하고 y만 사용
            x_dict = dataset.X.iloc[i].to_dict()
            x_data.append(x_dict)
            y_true.append(y)
        
        return x_data, y_true
    
    def _generate_dataset_internal_with_patience(
        self,
        x_data: List[Dict[str, Any]],
        y_true: List[str],
        num_prompts_per_data: int,
        use_batch_processing: bool,
        generation_batch_size: int,
        evaluation_batch_size: int,
        fallback_strategy: str = "adaptive",
        patience: int = 3,
        min_success_rate: float = 0.3,
        retry_strategies: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Patience 기반 DPO 데이터셋 생성 (품질 우선)
        
        Args:
            patience: 재시도 횟수
            retry_strategies: 재시도 시 사용할 전략들
        """
        if retry_strategies is None:
            retry_strategies = [
                {"consistency_threshold": self.consistency_threshold * 0.8, "num_prompts": num_prompts_per_data * 2},
                {"consistency_threshold": self.consistency_threshold * 0.6, "num_prompts": num_prompts_per_data * 3, "temperature": 0.9},
                {"consistency_threshold": self.consistency_threshold * 0.4, "num_prompts": num_prompts_per_data * 4, "temperature": 1.0}
            ]
        
        original_threshold = self.consistency_threshold
        
        for attempt in range(patience + 1):
            logging.info(f"Attempt {attempt + 1}/{patience + 1} for dataset generation...")
            
            if attempt == 0:
                # 첫 번째 시도: 기본 파라미터
                logging.info("Using original parameters")
                current_num_prompts = num_prompts_per_data
                generation_params = {
                    "max_tokens": 512,
                    "temperature": 0.8,
                    "top_p": 0.9
                }
            elif attempt <= len(retry_strategies):
                # 재시도: 점진적으로 완화된 파라미터
                strategy = retry_strategies[attempt - 1]
                self.consistency_threshold = strategy["consistency_threshold"]
                current_num_prompts = strategy["num_prompts"]
                
                generation_params = {
                    "max_tokens": 512,
                    "temperature": strategy.get("temperature", 0.8),
                    "top_p": 0.9
                }
                
                logging.info(f"Retry {attempt}: threshold={self.consistency_threshold:.2f}, "
                           f"num_prompts={current_num_prompts}, temp={generation_params['temperature']}")
            else:
                # 마지막 시도: fallback 전략 사용
                logging.info(f"Final attempt: using fallback strategy '{fallback_strategy}'")
                current_num_prompts = num_prompts_per_data
                generation_params = {
                    "max_tokens": 512,
                    "temperature": 0.8,
                    "top_p": 0.9
                }
            
            # 프롬프트 생성
            prompt_data = self._generate_prompts_with_params(
                x_data, current_num_prompts, use_batch_processing, 
                generation_batch_size, generation_params
            )
            
            if not prompt_data:
                logging.warning(f"No prompts generated in attempt {attempt + 1}")
                continue
            
            # 일관성 평가
            prompt_results = self._evaluate_prompts_with_consistency(
                prompt_data, y_true, use_batch_processing, evaluation_batch_size
            )
            
            # DPO 페어 생성
            if attempt < patience:
                # 재시도 가능: strict 모드로 품질 확인
                dpo_dataset = self.create_dpo_pairs_balanced(prompt_results, y_true, "strict")
            else:
                # 마지막 시도: fallback 전략 사용
                dpo_dataset = self.create_dpo_pairs_balanced(prompt_results, y_true, fallback_strategy)
            
            # 성공 조건 확인
            if len(dpo_dataset) > 0:
                success_rate = len(dpo_dataset) / len(x_data) if x_data else 0
                logging.info(f"Success! Generated {len(dpo_dataset)} DPO pairs (success rate: {success_rate:.2f})")
                
                # 품질 기준 확인
                if attempt == 0 or success_rate >= min_success_rate:  # 30% 이상 성공률이면 수용
                    self.consistency_threshold = original_threshold  # 원래 threshold 복구
                    return dpo_dataset
                elif attempt < patience:
                    logging.info(f"Success rate {success_rate:.2f} is low, trying next strategy...")
                    continue
                else:
                    logging.info(f"Final attempt succeeded with {success_rate:.2f} success rate")
                    self.consistency_threshold = original_threshold  # 원래 threshold 복구
                    return dpo_dataset
            else:
                logging.warning(f"Attempt {attempt + 1} failed: no DPO pairs generated")
        
        # 모든 시도 실패
        self.consistency_threshold = original_threshold  # 원래 threshold 복구
        logging.error("All attempts failed! Returning empty dataset.")
        return []
    
    def _generate_prompts_with_params(
        self,
        x_data: List[Dict[str, Any]],
        num_prompts_per_data: int,
        use_batch_processing: bool,
        batch_size: int,
        generation_params: Dict
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """파라미터를 받아서 프롬프트 생성"""
        results = []
        
        if use_batch_processing:
            # 배치 처리
            for i in tqdm(range(0, len(x_data), batch_size), desc="Generating prompts (batch)"):
                batch_x_data = x_data[i:i+batch_size]
                batch_meta_prompts = []
                batch_x_dicts = []
                
                for x_dict in batch_x_data:
                    meta_prompt = build_meta_prompt(x_dict, self.target_column, self.target_values)
                    batch_meta_prompts.append(meta_prompt)
                    batch_x_dicts.append(x_dict)
                
                try:
                    generation_params["n"] = num_prompts_per_data
                    all_generated_prompts = self.generator_model.generate(
                        batch_meta_prompts, generation_params
                    )
                    
                    for j, (meta_prompt, x_dict) in enumerate(zip(batch_meta_prompts, batch_x_dicts)):
                        start_idx = j * num_prompts_per_data
                        end_idx = start_idx + num_prompts_per_data
                        generated_prompts = all_generated_prompts[start_idx:end_idx]
                        
                        for generated_prompt in generated_prompts:
                            results.append((meta_prompt, x_dict, generated_prompt.strip()))
                            
                except Exception as e:
                    logging.error(f"Error in batch generation: {e}")
                    # 개별 처리로 폴백
                    for x_dict in batch_x_data:
                        meta_prompt = build_meta_prompt(x_dict, self.target_column, self.target_values)
                        try:
                            generation_params["n"] = num_prompts_per_data
                            generated_prompts = self.generator_model.generate([meta_prompt], generation_params)
                            for generated_prompt in generated_prompts:
                                results.append((meta_prompt, x_dict, generated_prompt.strip()))
                        except Exception as e2:
                            logging.error(f"Error in individual generation: {e2}")
                            continue
        else:
            # 개별 처리
            for x_dict in tqdm(x_data, desc="Generating prompts"):
                meta_prompt = build_meta_prompt(x_dict, self.target_column, self.target_values)
                try:
                    generation_params["n"] = num_prompts_per_data
                    generated_prompts = self.generator_model.generate([meta_prompt], generation_params)
                    for generated_prompt in generated_prompts:
                        results.append((meta_prompt, x_dict, generated_prompt.strip()))
                except Exception as e:
                    logging.error(f"Error generating prompt: {e}")
                    continue
        
        return results
    
    def _evaluate_prompts_with_consistency(
        self,
        prompt_data: List[Tuple[str, Dict[str, Any], str]],
        y_true: List[str],
        use_batch_processing: bool,
        batch_size: int
    ) -> List[Tuple[str, Dict[str, Any], str, str, float, List[str], str]]:
        """프롬프트 일관성 평가"""
        if use_batch_processing:
            return self.evaluate_prompts_consistency_batch(prompt_data, y_true, batch_size)
        else:
            results = []
            data_to_label = {str(sorted(x.items())): y for x, y in zip([item[1] for item in prompt_data], y_true)}
            
            for meta_prompt, x_dict, generated_prompt in tqdm(prompt_data, desc="Evaluating prompts"):
                x_key = str(sorted(x_dict.items()))
                true_label = data_to_label.get(x_key, y_true[0] if y_true else "UNKNOWN")
                
                final_prediction, consistency_score, predictions, category = self.evaluate_prompt_consistency(
                    generated_prompt, true_label
                )
                results.append((meta_prompt, x_dict, generated_prompt, final_prediction, 
                               consistency_score, predictions, category))
            return results
    
    def save_datasets(
        self,
        train_dataset: List[Dict[str, Any]],
        test_dataset: List[Dict[str, Any]],
        train_filename: str = "train_dpo_dataset.json",
        test_filename: str = "test_dpo_dataset.json"
    ) -> Tuple[str, str]:
        """
        훈련/테스트 데이터셋을 각각 저장
        
        Returns:
            (train_path, test_path) 튜플
        """
        # 훈련 데이터셋 저장
        train_path = os.path.join(self.output_dir, train_filename)
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_dataset, f, indent=2, ensure_ascii=False)
        
        # 테스트 데이터셋 저장
        test_path = os.path.join(self.output_dir, test_filename)
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_dataset, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved train dataset to: {train_path}")
        logging.info(f"Saved test dataset to: {test_path}")
        
        return train_path, test_path

    def analyze_and_fix_ratio(
        self,
        dpo_dataset: List[Dict[str, Any]],
        target_error_ratio: float = None
    ) -> List[Dict[str, Any]]:
        """
        기존 DPO 데이터셋의 ERROR/WRONG 비율을 분석하고 수정
        
        Args:
            dpo_dataset: 기존 DPO 데이터셋
            target_error_ratio: 목표 ERROR 비율 (None이면 self.error_to_wrong_ratio 사용)
            
        Returns:
            비율이 수정된 DPO 데이터셋
        """
        if not dpo_dataset:
            return dpo_dataset
        
        if target_error_ratio is None:
            target_error_ratio = self.error_to_wrong_ratio
        
        # 현재 비율 분석
        error_pairs = [pair for pair in dpo_dataset if pair['rejected_type'] == 'error']
        wrong_pairs = [pair for pair in dpo_dataset if pair['rejected_type'] == 'wrong']
        
        total_pairs = len(dpo_dataset)
        current_error_ratio = len(error_pairs) / total_pairs if total_pairs > 0 else 0
        current_wrong_ratio = len(wrong_pairs) / total_pairs if total_pairs > 0 else 0
        
        logging.info(f"Current ratio - ERROR: {len(error_pairs)} ({current_error_ratio:.1%}), WRONG: {len(wrong_pairs)} ({current_wrong_ratio:.1%})")
        logging.info(f"Target ratio - ERROR: {target_error_ratio:.1%}, WRONG: {1-target_error_ratio:.1%}")
        
        # 목표 수량 계산
        target_error_count = int(total_pairs * target_error_ratio)
        target_wrong_count = total_pairs - target_error_count
        
        # 비율 조정
        if len(error_pairs) >= target_error_count and len(wrong_pairs) >= target_wrong_count:
            # 둘 다 충분한 경우: 목표 수량만큼 샘플링
            selected_error = random.sample(error_pairs, target_error_count)
            selected_wrong = random.sample(wrong_pairs, target_wrong_count)
            fixed_dataset = selected_error + selected_wrong
        elif len(error_pairs) < target_error_count:
            # ERROR가 부족한 경우: 모든 ERROR + 나머지는 WRONG으로
            remaining_count = total_pairs - len(error_pairs)
            selected_wrong = random.sample(wrong_pairs, min(remaining_count, len(wrong_pairs)))
            fixed_dataset = error_pairs + selected_wrong
        else:
            # WRONG이 부족한 경우: 모든 WRONG + 나머지는 ERROR로
            remaining_count = total_pairs - len(wrong_pairs)
            selected_error = random.sample(error_pairs, min(remaining_count, len(error_pairs)))
            fixed_dataset = selected_error + wrong_pairs
        
        # 순서 섞기
        random.shuffle(fixed_dataset)
        
        # 최종 비율 로깅
        final_error_count = sum(1 for pair in fixed_dataset if pair['rejected_type'] == 'error')
        final_wrong_count = len(fixed_dataset) - final_error_count
        final_error_ratio = final_error_count / len(fixed_dataset) if fixed_dataset else 0
        
        logging.info(f"Fixed ratio - ERROR: {final_error_count} ({final_error_ratio:.1%}), WRONG: {final_wrong_count} ({1-final_error_ratio:.1%})")
        
        return fixed_dataset


class DPODataset(Dataset):
    """
    DPO 학습을 위한 PyTorch Dataset (새로운 구조)
    """
    
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        result = {
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected']
        }
        
        # 추가 메타데이터 포함 (있는 경우)
        if 'chosen_consistency' in item:
            result['chosen_consistency'] = item['chosen_consistency']
        if 'rejected_consistency' in item:
            result['rejected_consistency'] = item['rejected_consistency']
        if 'chosen_prediction' in item:
            result['chosen_prediction'] = item['chosen_prediction']
        if 'rejected_prediction' in item:
            result['rejected_prediction'] = item['rejected_prediction']
        if 'true_label' in item:
            result['true_label'] = item['true_label']
        if 'rejected_type' in item:
            result['rejected_type'] = item['rejected_type']
        if 'x_data' in item:
            result['x_data'] = item['x_data']
        
        return result


def load_tabular_data_from_csv(csv_path: str, target_column: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    CSV 파일에서 tabular data 로드
    
    Args:
        csv_path: CSV 파일 경로
        target_column: 타겟 컬럼명
        
    Returns:
        (x_data, y_true) 튜플
    """
    df = pd.read_csv(csv_path)
    
    # 타겟 컬럼 분리
    y_true = df[target_column].tolist()
    x_data = df.drop(columns=[target_column]).to_dict('records')
    
    return x_data, y_true


def load_tabular_data_from_dataset(dataset: Dataset) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    기존 Dataset 객체에서 데이터 로드
    
    Args:
        dataset: BaseDataset을 상속한 데이터셋
        
    Returns:
        (x_data, y_true) 튜플
    """
    x_data = []
    y_true = []
    
    for i in range(len(dataset)):
        _, y = dataset[i]  # meta_prompt는 무시하고 y만 사용
        x_dict = dataset.X.iloc[i].to_dict()
        x_data.append(x_dict)
        y_true.append(y)
    
    return x_data, y_true 