#!/usr/bin/env python3
"""
기존 DPO 데이터셋의 ERROR/WRONG 비율을 수정하는 스크립트
"""

import argparse
import json
import os
import sys
import logging
import random

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dpo_dataset_generator import DPODatasetGenerator
from src.inference import InferenceModel


def fix_dataset_ratio(
    input_path: str,
    output_path: str,
    target_error_ratio: float = 0.4,
    backup: bool = True
):
    """
    DPO 데이터셋의 ERROR/WRONG 비율을 수정
    
    Args:
        input_path: 입력 데이터셋 경로
        output_path: 출력 데이터셋 경로
        target_error_ratio: 목표 ERROR 비율 (0.4 = 40% ERROR, 60% WRONG)
        backup: 원본 파일 백업 여부
    """
    # 원본 데이터 로드
    logging.info(f"Loading dataset from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        original_dataset = json.load(f)
    
    logging.info(f"Original dataset size: {len(original_dataset)}")
    
    # 현재 비율 분석
    error_count = sum(1 for item in original_dataset if item.get('rejected_type') == 'error')
    wrong_count = sum(1 for item in original_dataset if item.get('rejected_type') == 'wrong')
    total_count = len(original_dataset)
    
    current_error_ratio = error_count / total_count if total_count > 0 else 0
    current_wrong_ratio = wrong_count / total_count if total_count > 0 else 0
    
    logging.info(f"Current ratio - ERROR: {error_count} ({current_error_ratio:.1%}), WRONG: {wrong_count} ({current_wrong_ratio:.1%})")
    logging.info(f"Target ratio - ERROR: {target_error_ratio:.1%}, WRONG: {1-target_error_ratio:.1%}")
    
    # 비율이 이미 적절하면 스킵
    if abs(current_error_ratio - target_error_ratio) < 0.05:  # 5% 이내 차이면 OK
        logging.info("Ratio is already close to target. No changes needed.")
        return
    
    # 백업 생성
    if backup and input_path == output_path:
        backup_path = input_path + '.backup'
        logging.info(f"Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(original_dataset, f, indent=2, ensure_ascii=False)
    
    # 더미 DPODatasetGenerator 생성 (비율 수정 함수만 사용)
    dummy_generator = DPODatasetGenerator(
        generator_model=None,
        predictor_model=None,
        target_values=['DEAD', 'ALIVE'],
        target_column='Survived',
        error_to_wrong_ratio=target_error_ratio
    )
    
    # 비율 수정
    logging.info("Fixing ratio...")
    fixed_dataset = dummy_generator.analyze_and_fix_ratio(original_dataset, target_error_ratio)
    
    # 수정된 데이터셋 저장
    logging.info(f"Saving fixed dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_dataset, f, indent=2, ensure_ascii=False)
    
    # 최종 결과 확인
    final_error_count = sum(1 for item in fixed_dataset if item.get('rejected_type') == 'error')
    final_wrong_count = len(fixed_dataset) - final_error_count
    final_error_ratio = final_error_count / len(fixed_dataset) if fixed_dataset else 0
    
    logging.info("=" * 60)
    logging.info("Ratio Fix Complete!")
    logging.info(f"Original: ERROR {error_count} ({current_error_ratio:.1%}), WRONG {wrong_count} ({current_wrong_ratio:.1%})")
    logging.info(f"Fixed:    ERROR {final_error_count} ({final_error_ratio:.1%}), WRONG {final_wrong_count} ({1-final_error_ratio:.1%})")
    logging.info(f"Dataset size: {len(original_dataset)} → {len(fixed_dataset)}")


def main():
    parser = argparse.ArgumentParser(description='Fix ERROR/WRONG ratio in existing DPO datasets')
    
    parser.add_argument('input_path', type=str,
                        help='Path to input DPO dataset JSON file')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to output fixed dataset (default: overwrite input)')
    parser.add_argument('--target_error_ratio', type=float, default=0.4,
                        help='Target ERROR ratio (0.4 = 40% ERROR, 60% WRONG)')
    parser.add_argument('--no_backup', action='store_true',
                        help='Do not create backup when overwriting')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    
    # 시드 설정
    random.seed(args.seed)
    
    # 출력 경로 설정
    output_path = args.output_path if args.output_path else args.input_path
    
    # 입력 파일 존재 확인
    if not os.path.exists(args.input_path):
        logging.error(f"Input file not found: {args.input_path}")
        return
    
    # 비율 유효성 확인
    if not 0 <= args.target_error_ratio <= 1:
        logging.error(f"Invalid target_error_ratio: {args.target_error_ratio}. Must be between 0 and 1.")
        return
    
    # 비율 수정 실행
    fix_dataset_ratio(
        input_path=args.input_path,
        output_path=output_path,
        target_error_ratio=args.target_error_ratio,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main() 