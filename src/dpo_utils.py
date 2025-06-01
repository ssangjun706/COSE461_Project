import json
from typing import Dict, Any, Optional
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


class DPODatasetForTRL(Dataset):
    def __init__(
        self, 
        data_path: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 1024,
        prompt_field: str = "prompt",
        chosen_field: str = "chosen", 
        rejected_field: str = "rejected"
    ):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_field = prompt_field
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        result = {
            self.prompt_field: item['prompt'],
            self.chosen_field: item['chosen'], 
            self.rejected_field: item['rejected']
        }
        
        if 'chosen_accuracy' in item:
            result['chosen_accuracy'] = item['chosen_accuracy']
        if 'rejected_accuracy' in item:
            result['rejected_accuracy'] = item['rejected_accuracy']
        if 'x_data' in item:
            result['x_data'] = item['x_data']
        if 'true_label' in item:
            result['true_label'] = item['true_label']
        
        return result


def convert_to_hf_dataset(data_path: str) -> HFDataset:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # TRL DPOTrainer가 기대하는 형식으로 변환
    hf_data = {
        'prompt': [item['prompt'] for item in data],
        'chosen': [item['chosen'] for item in data],
        'rejected': [item['rejected'] for item in data]
    }
    
    # 추가 메타데이터 포함
    if data and 'chosen_accuracy' in data[0]:
        hf_data['chosen_accuracy'] = [item.get('chosen_accuracy', 1.0) for item in data]
    if data and 'rejected_accuracy' in data[0]:
        hf_data['rejected_accuracy'] = [item.get('rejected_accuracy', 0.0) for item in data]
    
    return HFDataset.from_dict(hf_data)


def load_dpo_datasets_for_training(
    train_path: str, 
    test_path: str,
    convert_to_hf: bool = True
) -> tuple:
    """
    훈련용 DPO 데이터셋을 로드
    
    Args:
        train_path: 훈련 데이터셋 경로
        test_path: 테스트 데이터셋 경로
        convert_to_hf: HuggingFace Dataset으로 변환할지 여부
        
    Returns:
        (train_dataset, test_dataset) 튜플
    """
    if convert_to_hf:
        train_dataset = convert_to_hf_dataset(train_path)
        test_dataset = convert_to_hf_dataset(test_path)
    else:
        train_dataset = DPODatasetForTRL(train_path)
        test_dataset = DPODatasetForTRL(test_path)
    
    return train_dataset, test_dataset


def analyze_dpo_dataset(data_path: str) -> Dict[str, Any]:
    """
    DPO 데이터셋 분석 및 통계 정보 제공
    
    Args:
        data_path: DPO 데이터셋 JSON 파일 경로
        
    Returns:
        분석 결과 딕셔너리
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        return {"error": "Empty dataset"}
    
    analysis = {
        "total_pairs": len(data),
        "avg_prompt_length": 0,
        "avg_chosen_length": 0, 
        "avg_rejected_length": 0,
        "consistency_stats": {},
        "rejected_type_stats": {},
        "unique_prompts": 0,
        "sample_data": data[0] if data else None
    }
    
    # 길이 통계
    prompt_lengths = [len(item['prompt']) for item in data]
    chosen_lengths = [len(item['chosen']) for item in data]
    rejected_lengths = [len(item['rejected']) for item in data]
    
    analysis["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
    analysis["avg_chosen_length"] = sum(chosen_lengths) / len(chosen_lengths)
    analysis["avg_rejected_length"] = sum(rejected_lengths) / len(rejected_lengths)
    
    analysis["min_prompt_length"] = min(prompt_lengths)
    analysis["max_prompt_length"] = max(prompt_lengths)
    analysis["min_chosen_length"] = min(chosen_lengths)
    analysis["max_chosen_length"] = max(chosen_lengths)
    analysis["min_rejected_length"] = min(rejected_lengths)
    analysis["max_rejected_length"] = max(rejected_lengths)
    
    # 일관성 통계 (있는 경우)
    if 'chosen_consistency' in data[0]:
        chosen_consistencies = [item['chosen_consistency'] for item in data]
        rejected_consistencies = [item['rejected_consistency'] for item in data]
        
        analysis["consistency_stats"] = {
            "avg_chosen_consistency": sum(chosen_consistencies) / len(chosen_consistencies),
            "avg_rejected_consistency": sum(rejected_consistencies) / len(rejected_consistencies),
            "min_chosen_consistency": min(chosen_consistencies),
            "max_chosen_consistency": max(chosen_consistencies),
            "min_rejected_consistency": min(rejected_consistencies),
            "max_rejected_consistency": max(rejected_consistencies),
            "consistency_gap": sum(chosen_consistencies) / len(chosen_consistencies) - sum(rejected_consistencies) / len(rejected_consistencies)
        }
    
    # rejected type 통계
    if 'rejected_type' in data[0]:
        rejected_types = [item['rejected_type'] for item in data]
        from collections import Counter
        type_counts = Counter(rejected_types)
        analysis["rejected_type_stats"] = dict(type_counts)
        analysis["error_to_wrong_ratio"] = type_counts.get('error', 0) / len(data) if data else 0
    
    # 고유 프롬프트 수
    unique_prompts = set(item['prompt'] for item in data)
    analysis["unique_prompts"] = len(unique_prompts)
    analysis["prompt_reuse_ratio"] = len(data) / len(unique_prompts) if unique_prompts else 0
    
    return analysis


def print_dataset_analysis(data_path: str):
    """
    데이터셋 분석 결과를 출력
    
    Args:
        data_path: DPO 데이터셋 JSON 파일 경로
    """
    analysis = analyze_dpo_dataset(data_path)
    
    print(f"\n{'='*50}")
    print(f"DPO Dataset Analysis: {data_path}")
    print(f"{'='*50}")
    
    print(f"Total DPO pairs: {analysis['total_pairs']}")
    print(f"Unique prompts: {analysis['unique_prompts']}")
    print(f"Prompt reuse ratio: {analysis['prompt_reuse_ratio']:.2f}")
    
    print(f"\nText Length Statistics:")
    print(f"  Prompt length: {analysis['avg_prompt_length']:.0f} chars (min: {analysis['min_prompt_length']}, max: {analysis['max_prompt_length']})")
    print(f"  Chosen length: {analysis['avg_chosen_length']:.0f} chars (min: {analysis['min_chosen_length']}, max: {analysis['max_chosen_length']})")
    print(f"  Rejected length: {analysis['avg_rejected_length']:.0f} chars (min: {analysis['min_rejected_length']}, max: {analysis['max_rejected_length']})")
    
    if analysis['consistency_stats']:
        stats = analysis['consistency_stats']
        print(f"\nConsistency Statistics:")
        print(f"  Chosen consistency: {stats['avg_chosen_consistency']:.3f} (min: {stats['min_chosen_consistency']:.3f}, max: {stats['max_chosen_consistency']:.3f})")
        print(f"  Rejected consistency: {stats['avg_rejected_consistency']:.3f} (min: {stats['min_rejected_consistency']:.3f}, max: {stats['max_rejected_consistency']:.3f})")
        print(f"  Consistency gap: {stats['consistency_gap']:.3f}")
    
    if analysis['rejected_type_stats']:
        print(f"\nRejected Type Statistics:")
        for reject_type, count in analysis['rejected_type_stats'].items():
            print(f"  {reject_type.upper()}: {count} ({count/analysis['total_pairs']*100:.1f}%)")
        print(f"  ERROR to WRONG ratio: {analysis['error_to_wrong_ratio']:.3f}")
    
    if analysis['sample_data']:
        print(f"\nSample DPO Pair:")
        sample = analysis['sample_data']
        print(f"  Prompt (first 150 chars): {sample['prompt'][:150]}...")
        print(f"  Chosen (first 100 chars): {sample['chosen'][:100]}...")
        print(f"  Rejected (first 100 chars): {sample['rejected'][:100]}...")
        if 'chosen_consistency' in sample:
            print(f"  Chosen consistency: {sample['chosen_consistency']:.3f}")
            print(f"  Rejected consistency: {sample['rejected_consistency']:.3f}")
        if 'rejected_type' in sample:
            print(f"  Rejected type: {sample['rejected_type']}")
        if 'chosen_prediction' in sample:
            print(f"  Chosen prediction: {sample['chosen_prediction']}")
            print(f"  Rejected prediction: {sample['rejected_prediction']}")
            print(f"  True label: {sample['true_label']}")


def validate_dpo_dataset(data_path: str) -> Dict[str, Any]:
    """
    DPO 데이터셋의 유효성 검증
    
    Args:
        data_path: DPO 데이터셋 JSON 파일 경로
        
    Returns:
        검증 결과 딕셔너리
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {"valid": False, "error": f"Failed to load file: {e}"}
    
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "total_items": len(data)
    }
    
    required_fields = ['prompt', 'chosen', 'rejected']
    optional_fields = ['chosen_prediction', 'rejected_prediction', 'chosen_consistency', 
                      'rejected_consistency', 'x_data', 'true_label', 'rejected_type']
    
    for i, item in enumerate(data):
        # 필수 필드 확인
        for field in required_fields:
            if field not in item:
                validation["errors"].append(f"Item {i}: Missing required field '{field}'")
                validation["valid"] = False
            elif not isinstance(item[field], str):
                validation["errors"].append(f"Item {i}: Field '{field}' must be string")
                validation["valid"] = False
            elif len(item[field].strip()) == 0:
                validation["warnings"].append(f"Item {i}: Field '{field}' is empty")
        
        # 일관성 필드 확인 (있는 경우)
        for consistency_field in ['chosen_consistency', 'rejected_consistency']:
            if consistency_field in item:
                if not isinstance(item[consistency_field], (int, float)):
                    validation["errors"].append(f"Item {i}: Field '{consistency_field}' must be numeric")
                    validation["valid"] = False
                elif not (0.0 <= item[consistency_field] <= 1.0):
                    validation["warnings"].append(f"Item {i}: Field '{consistency_field}' should be between 0 and 1")
        
        # rejected_type 필드 확인 (있는 경우)
        if 'rejected_type' in item:
            if item['rejected_type'] not in ['error', 'wrong']:
                validation["warnings"].append(f"Item {i}: rejected_type should be 'error' or 'wrong', got '{item['rejected_type']}'")
        
        # prediction 필드와 true_label 일관성 확인
        if all(field in item for field in ['chosen_prediction', 'rejected_prediction', 'true_label']):
            if item['chosen_prediction'] != item['true_label']:
                validation["warnings"].append(f"Item {i}: chosen_prediction should match true_label for a chosen response")
    
    return validation


def create_sample_dpo_data(output_path: str, num_samples: int = 10):
    """
    테스트용 샘플 DPO 데이터 생성 (새로운 구조)
    
    Args:
        output_path: 출력 파일 경로
        num_samples: 생성할 샘플 수
    """
    sample_data = []
    
    for i in range(num_samples):
        rejected_type = "error" if i % 3 == 0 else "wrong"  # 에러와 틀린 케이스 섞어서
        
        sample_data.append({
            "prompt": f"You are an expert data analyst. Analyze the following passenger data and predict survival: Age: {20+i*5}, Sex: {'male' if i%2==0 else 'female'}, Class: {(i%3)+1}, Fare: ${10+i*5}. Respond with ONLY ['DEAD', 'ALIVE']. Do not include any explanations.",
            "chosen": f"Based on the passenger profile - Age: {20+i*5}, Sex: {'male' if i%2==0 else 'female'}, Class: {(i%3)+1}, Fare: ${10+i*5} - I analyze the survival probability. Considering the demographic factors and historical Titanic data, this passenger would likely be {'ALIVE' if i%2==1 else 'DEAD'}.",
            "rejected": f"Looking at this passenger data with Age: {20+i*5}, the prediction is unclear. The survival depends on many factors. {'ERROR' if rejected_type == 'error' else ('DEAD' if i%2==1 else 'ALIVE')}.",  # 의도적으로 틀린 답 또는 ERROR
            "chosen_prediction": "ALIVE" if i%2==1 else "DEAD",
            "rejected_prediction": "ERROR" if rejected_type == "error" else ("DEAD" if i%2==1 else "ALIVE"),
            "chosen_consistency": 0.8 + (i % 3) * 0.1,  # 0.8~1.0
            "rejected_consistency": 0.2 + (i % 3) * 0.1 if rejected_type == "error" else 0.7 + (i % 3) * 0.1,  # ERROR면 낮은 일관성, WRONG이면 높은 일관성
            "x_data": {
                "Age": 20+i*5,
                "Sex": "male" if i%2==0 else "female", 
                "Pclass": (i%3)+1,
                "Fare": 10+i*5
            },
            "true_label": "ALIVE" if i%2==1 else "DEAD",
            "rejected_type": rejected_type
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample DPO dataset with {num_samples} pairs: {output_path}")


if __name__ == "__main__":
    # 테스트용 샘플 데이터 생성
    create_sample_dpo_data("data/sample_dpo_dataset.json", 10)
    
    # 생성된 샘플 데이터 분석
    print_dataset_analysis("data/sample_dpo_dataset.json")
    
    # 검증
    validation = validate_dpo_dataset("data/sample_dpo_dataset.json")
    print(f"\nValidation result: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
    if validation['errors']:
        print("Errors:", validation['errors'])
    if validation['warnings']:
        print("Warnings:", validation['warnings']) 