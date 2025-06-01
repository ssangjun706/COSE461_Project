import argparse
import sys
import os
import logging

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import InferenceModel
from src.dpo_dataset_generator import DPODatasetGenerator
from src.dataset import TitanicDataset


def main():
    parser = argparse.ArgumentParser(description='Generate DPO dataset for prompt optimization with consistency check')
    
    # 데이터 설정
    parser.add_argument('--dataset_path', type=str, default="../dataset/titanic-dataset.csv",
                        help='Path to dataset file')
    parser.add_argument('--target_values', type=str, nargs='+', default=['DEAD', 'ALIVE'],
                        help='Possible target values for classification')
    parser.add_argument('--target_column', type=str, default='Survived',
                        help='Target column name')
    
    # 모델 서버 설정
    parser.add_argument('--generator_host', type=str, default='localhost',
                        help='Host for generator model server')
    parser.add_argument('--generator_port', type=int, default=23456,
                        help='Port for generator model server')
    parser.add_argument('--predictor_host', type=str, default='localhost',
                        help='Host for predictor model server')
    parser.add_argument('--predictor_port', type=int, default=23456,
                        help='Port for predictor model server')
    
    # 데이터셋 생성 파라미터
    parser.add_argument('--num_prompts_per_data', type=int, default=5,
                        help='Number of prompts to generate per data point')
    parser.add_argument('--num_predictions', type=int, default=10,
                        help='Number of predictions per generated prompt for consistency check')
    parser.add_argument('--consistency_threshold', type=float, default=0.75,
                        help='Minimum consistency threshold (75% same predictions)')
    parser.add_argument('--error_to_wrong_ratio', type=float, default=0.4,
                        help='Ratio of ERROR to WRONG cases in rejected pairs (0.4 = 40% ERROR, 60% WRONG)')
    
    # Fallback 전략 추가
    parser.add_argument('--fallback_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'relative', 'strict'],
                        help='Strategy when no chosen prompts meet threshold: adaptive (lower threshold), relative (best available), strict (empty dataset)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of retry attempts with different parameters before using fallback strategy')
    parser.add_argument('--min_success_rate', type=float, default=0.3,
                        help='Minimum success rate to accept a retry attempt (0.3 = 30%)')
    
    # 성능 최적화 파라미터
    parser.add_argument('--use_batch_processing', action='store_true', default=True,
                        help='Use batch processing for faster generation (default: True)')
    parser.add_argument('--no_batch_processing', dest='use_batch_processing', action='store_false',
                        help='Disable batch processing')
    parser.add_argument('--generation_batch_size', type=int, default=8,
                        help='Batch size for prompt generation')
    parser.add_argument('--evaluation_batch_size', type=int, default=16,
                        help='Batch size for prompt evaluation')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='../dataset/titanic-dpo',
                        help='Directory to save generated datasets')
    parser.add_argument('--train_filename', type=str, default='titanic-train-dpo-dataset.json',
                        help='Train dataset filename')
    parser.add_argument('--test_filename', type=str, default='titanic-test-dpo-dataset.json',
                        help='Test dataset filename')
    
    # 데이터셋 크기 제한 (테스트용)
    parser.add_argument('--max_data_points', type=int, default=None,
                        help='Maximum number of data points to process per split (for testing)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    
    logging.info("Starting DPO dataset generation with consistency check...")
    
    # 데이터셋 파일 존재 확인
    if not os.path.exists(args.dataset_path):
        logging.error(f"Dataset file not found: {args.dataset_path}")
        logging.info("Please provide correct --dataset_path or ensure the file exists")
        return
    
    # 모델 서버 초기화
    logging.info("Connecting to inference servers...")
    generator_model = InferenceModel(
        host=args.generator_host,
        port=args.generator_port
    )
    
    predictor_model = InferenceModel(
        host=args.predictor_host,
        port=args.predictor_port
    )
    
    # 서버 연결 확인
    try:
        generator_model.check_status()
        logging.info(f"✓ Generator server connected: {args.generator_host}:{args.generator_port}")
    except Exception as e:
        logging.error(f"✗ Failed to connect to generator server: {e}")
        return
    
    try:
        predictor_model.check_status()
        logging.info(f"✓ Predictor server connected: {args.predictor_host}:{args.predictor_port}")
    except Exception as e:
        logging.error(f"✗ Failed to connect to predictor server: {e}")
        return
    
    # DPO 데이터셋 생성기 초기화
    dataset_generator = DPODatasetGenerator(
        generator_model=generator_model,
        predictor_model=predictor_model,
        target_values=args.target_values,
        target_column=args.target_column,
        error_value="ERROR",
        num_predictions_per_prompt=args.num_predictions,
        consistency_threshold=args.consistency_threshold,
        error_to_wrong_ratio=args.error_to_wrong_ratio,
        output_dir=args.output_dir
    )
    
    # TitanicDataset을 사용하여 train/test DPO 데이터셋 생성
    logging.info("Generating DPO datasets from TitanicDataset (preserving original train/test split)...")
    if args.use_batch_processing:
        logging.info(f"Batch processing enabled - Generation: {args.generation_batch_size}, Evaluation: {args.evaluation_batch_size}")
    else:
        logging.info("Batch processing disabled - using sequential processing")
    
    train_dpo_dataset, test_dpo_dataset = dataset_generator.generate_dataset_from_dataset_class(
        dataset_class=TitanicDataset,
        dataset_path=args.dataset_path,
        num_prompts_per_data=args.num_prompts_per_data,
        use_batch_processing=args.use_batch_processing,
        generation_batch_size=args.generation_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        fallback_strategy=args.fallback_strategy,
        patience=args.patience,
        min_success_rate=args.min_success_rate
    )
    
    # 데이터셋 크기 제한 (테스트용)
    if args.max_data_points:
        if len(train_dpo_dataset) > args.max_data_points:
            train_dpo_dataset = train_dpo_dataset[:args.max_data_points]
            logging.info(f"Limited train dataset to {args.max_data_points} points for testing")
        
        if len(test_dpo_dataset) > args.max_data_points:
            test_dpo_dataset = test_dpo_dataset[:args.max_data_points]
            logging.info(f"Limited test dataset to {args.max_data_points} points for testing")
    
    if len(train_dpo_dataset) == 0 and len(test_dpo_dataset) == 0:
        logging.error("No DPO pairs were generated. Check your parameters and model outputs.")
        return
    
    # 데이터셋 저장
    logging.info("Saving datasets...")
    train_path, test_path = dataset_generator.save_datasets(
        train_dataset=train_dpo_dataset,
        test_dataset=test_dpo_dataset,
        train_filename=args.train_filename,
        test_filename=args.test_filename
    )
    
    # 결과 요약
    logging.info("=" * 60)
    logging.info("DPO Dataset Generation Complete!")
    logging.info(f"Train DPO pairs: {len(train_dpo_dataset)}")
    logging.info(f"Test DPO pairs: {len(test_dpo_dataset)}")
    logging.info(f"Total DPO pairs: {len(train_dpo_dataset) + len(test_dpo_dataset)}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # 샘플 출력 (Train)
    if len(train_dpo_dataset) > 0:
        logging.info("\nSample Train DPO pair:")
        sample = train_dpo_dataset[0]
        logging.info(f"Meta prompt (first 100 chars): {sample['prompt'][:100]}...")
        logging.info(f"Chosen consistency: {sample['chosen_consistency']:.3f}")
        logging.info(f"Rejected consistency: {sample['rejected_consistency']:.3f}")
        logging.info(f"Rejected type: {sample['rejected_type']}")
        logging.info(f"Chosen prediction: {sample['chosen_prediction']}")
        logging.info(f"Rejected prediction: {sample['rejected_prediction']}")
        logging.info(f"True label: {sample['true_label']}")
        logging.info(f"Chosen prompt (first 100 chars): {sample['chosen'][:100]}...")
        logging.info(f"Rejected prompt (first 100 chars): {sample['rejected'][:100]}...")
    
    # 샘플 출력 (Test)
    if len(test_dpo_dataset) > 0:
        logging.info("\nSample Test DPO pair:")
        sample = test_dpo_dataset[0]
        logging.info(f"Meta prompt (first 100 chars): {sample['prompt'][:100]}...")
        logging.info(f"Chosen consistency: {sample['chosen_consistency']:.3f}")
        logging.info(f"Rejected consistency: {sample['rejected_consistency']:.3f}")
        logging.info(f"Rejected type: {sample['rejected_type']}")
        logging.info(f"Chosen prediction: {sample['chosen_prediction']}")
        logging.info(f"Rejected prediction: {sample['rejected_prediction']}")
        logging.info(f"True label: {sample['true_label']}")
    
    # 통계 정보
    if len(train_dpo_dataset) > 0:
        error_count = sum(1 for item in train_dpo_dataset if item['rejected_type'] == 'error')
        wrong_count = sum(1 for item in train_dpo_dataset if item['rejected_type'] == 'wrong')
        logging.info(f"\nTrain dataset rejected types - ERROR: {error_count}, WRONG: {wrong_count}")
    
    if len(test_dpo_dataset) > 0:
        error_count = sum(1 for item in test_dpo_dataset if item['rejected_type'] == 'error')
        wrong_count = sum(1 for item in test_dpo_dataset if item['rejected_type'] == 'wrong')
        logging.info(f"Test dataset rejected types - ERROR: {error_count}, WRONG: {wrong_count}")


if __name__ == "__main__":
    main() 