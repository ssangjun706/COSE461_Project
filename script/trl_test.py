import os
import sys
import torch

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.append(root)

import src.hf_utils
from src.dataset import TitanicDataset
from src.model import RewardModel, APIServer, PromptModel

from peft import LoraConfig

from trl.trainer import PPOConfig, PPOTrainer
from trl.core import LengthSampler
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
dataset = TitanicDataset("../dataset/titanic-dataset.csv", train=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    use_dora=True,
)

# PPO 설정
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    num_ppo_epochs=4,
    seed=42,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

api_server = APIServer(host="localhost", port=23456)
reward_model = RewardModel(
    api_server=api_server,
    target_values=dataset.target_values,
)


prompt_model = PromptModel()
tokenizer = prompt_model.tokenizer

ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=prompt_model.model,
    ref_model=None,
    train_dataset=dataset,
)

generation_kwargs = {
    "max_new_tokens": 2048,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

num_ppo_epochs = 1

for epoch in range(num_ppo_epochs):
    print(f"PPO Epoch: {epoch + 1}/{num_ppo_epochs}")
    for batch in tqdm(ppo_trainer.dataloader):
        query_texts = batch["query"]
        query_tensors = tokenizer(
            query_texts, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            length_sampler=LengthSampler(
                min_length=10, max_length=generation_kwargs["max_new_tokens"]
            ),
            **generation_kwargs,
        )

        batch["response"] = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        breakpoint()
        rewards = reward_model.evaluate(batch["response"]).to(device)

        try:
            stats = ppo_trainer.step(
                query_tensors.squeeze(dim=0), response_tensors.squeeze(dim=0), rewards
            )  # batch_size=1일때 squeeze 필요할 수 있음
            # stats = ppo_trainer.step(query_tensors, response_tensors, rewards) # 일반적 batch
            ppo_trainer.log_stats(stats, batch, rewards)
        except RuntimeError as e:
            if "Trying to backward through the graph a second time" in str(e):
                print(
                    "Error during PPO step (backward pass). Ensure reward calculation is detached."
                )
                # 추가 디버깅 정보
                print(f"Query tensors shape: {query_tensors.shape}")
                print(f"Response tensors shape: {response_tensors.shape}")
                print(f"Rewards: {rewards}")
                raise e
            else:
                raise e

        torch.cuda.empty_cache()

    print(f"Epoch {epoch+1} finished. Example generation:")
    test_prompt = "The future of AI is "
    test_input = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)

    # ppo_trainer.model은 PEFT 모델이므로, get_ mehreren 모델을 사용해야 합니다.
    # AutoModelForCausalLMWithValueHead는 내부에 `pretrained_model` 속성으로 원래 모델(PEFT 적용된)을 가짐
    # generate 메서드는 AutoModelForCausalLMWithValueHead에 정의되어 있음.
    generated_tokens = ppo_trainer.model.generate(test_input, **generation_kwargs)
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")


# # --- 6. 어댑터 저장 ---
# output_dir = "../result/Mistral-Small-3.1-24B-PPO"
# ppo_trainer.model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)
# print(f"LoRA adapter and tokenizer saved to {output_dir}")
