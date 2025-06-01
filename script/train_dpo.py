import os
import sys
import json
import torch
import logging
import wandb

from dataclasses import dataclass, field

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

from src.hf_utils import find_model_path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    # Mistral3ForConditionalGeneration
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig


logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
)


@dataclass
class TrainingConfig:
    # model_name: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    
    # DPO 설정
    beta: float = 0.1
    loss_type: str = "sigmoid"
    
    # 학습 설정
    learning_rate: float = 5e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    bf16: bool = True
    
    # 데이터 설정
    max_length: int = 1024
    max_prompt_length: int = 512
    
    # 저장 및 로깅
    output_dir: str = "../checkpoints/dpo_mistral"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    use_wandb: bool = True
    wandb_project: str = "dpo-mistral-training"


def load_dpo_dataset(dataset_path) -> Dataset:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required_keys = ["prompt", "chosen", "rejected"]
    if not all(key in data[0] for key in required_keys):
        raise ValueError(f"Dataset must contain keys: {required_keys}")
    
    # 데이터에 빈 이미지 필드 추가
    processed_data = []
    for item in data:
        processed_item = {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }
        processed_data.append(processed_item)
    
    dataset = Dataset.from_list(processed_data)
    return dataset



if __name__ == "__main__":
    # 분산 학습 환경 변수 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    
    config = TrainingConfig()
    model_path = find_model_path(config.model_name)
    processor = AutoTokenizer.from_pretrained(model_path)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )

    # if config.use_wandb:
    #     wandb.init(project=config.wandb_project)
    
    train_dataset = load_dpo_dataset("../dataset/dpo_train_dataset.json")
    eval_dataset = load_dpo_dataset("../dataset/dpo_eval_dataset.json")
    
    logging.info(f"Training examples: {len(train_dataset)}")
    logging.info(f"Evaluation examples: {len(eval_dataset)}")
    
    args = DPOConfig(
        output_dir=config.output_dir,
        logging_steps=config.logging_steps,
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=processor,
    )
    
    logging.info("Starting DPO training...")
    try:
        trainer.train()
        final_output_dir = os.path.join(config.output_dir, "final_model")
        logging.info(f"Saving final model to: {final_output_dir}")
        trainer.model.save_pretrained(final_output_dir)
        config_path = os.path.join(final_output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.__dict__, f, indent=2, ensure_ascii=False)
        
        logging.info("Final model saved successfully!")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
