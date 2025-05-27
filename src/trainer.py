import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import glob
import logging
import wandb

from tqdm import tqdm
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass

from .hf_utils import find_model_path
from .reward import RewardModel
from .inference import InferenceModel
from .dataset import BaseDataset
from .utils import calculate_metrics

from torch.utils.data import DataLoader
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model


@dataclass
class PPOConfig:
    learning_rate: float
    batch_size: int
    max_new_tokens: int
    max_epochs: int

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    
    save_freq: int
    eval_freq: int
    
    reward_lambda: float
    entropy_weight: float
    value_loss_weight: float

    ppo_epochs: int 
    mini_batch_size: int
    cliprange: float
    cliprange_value: float
    
    use_wandb: bool
    project_name: str
    resume_from_checkpoint: str


class PPOBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.queries = []
        self.responses = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.advantages = []
        self.returns = []
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def store(
        self,
        queries,
        responses,
        log_probs,
        values,
        rewards,
        advantages,
        returns,
    ):
        self.queries.extend(queries)
        self.responses.extend(responses)
        self.log_probs.extend(log_probs)
        self.values.extend(values)
        self.rewards.extend(rewards)
        self.advantages.extend(advantages)
        self.returns.extend(returns)

    def get_batch(self, batch_size):
        if len(self.queries) == 0:
            return None

        indices = torch.randperm(len(self.queries))[:batch_size]

        batch = {
            "queries": [self.queries[i] for i in indices],
            "responses": [self.responses[i] for i in indices],
            "log_probs": torch.stack([self.log_probs[i] for i in indices]),
            "values": torch.stack([self.values[i] for i in indices]),
            "rewards": torch.stack([self.rewards[i] for i in indices]),
            "advantages": torch.stack([self.advantages[i] for i in indices]),
            "returns": torch.stack([self.returns[i] for i in indices]),
        }

        return batch

    def __len__(self):
        return len(self.queries)


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        inference_model: InferenceModel,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )

        self.model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        self.config = config
        self.inference_model = inference_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dtype = torch.bfloat16
        self.buffer = PPOBuffer()

        self._setup_models()
        self._setup_optimizer()

        if config.use_wandb:
            wandb.init(project=config.project_name)

    def _setup_models(self):
        model_path = find_model_path(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "temperature": 0.15,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            device_map="auto",
            torch_dtype=self.dtype,
        )
        
        self.lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            use_dora=True,
        )
        self.model = get_peft_model(base_model, self.lora_config)
        self.model.gradient_checkpointing_enable()
        self.device = self.model.device

        
        self.hidden_size = 5120
        self.value_head = nn.Linear(self.hidden_size, 1, dtype=self.dtype)
        self.value_head.to(self.model.device)
        self.reward_model = RewardModel(
            inference_model=self.inference_model,
            target_values=self.train_dataset.target_values,
        )

    def _setup_optimizer(self):
        params = list(self.model.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

    def generate(
        self,
        tokenized_inputs: Any,
        output_scores: bool = True,
        output_hidden_states: bool = False,
    ):
        input_length = tokenized_inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **tokenized_inputs,
            **self.generation_config,
            return_dict_in_generate=True,
            output_scores=output_scores,
            output_hidden_states=output_hidden_states,
        )

        generated_tokens = outputs.sequences[:, input_length:]
        log_probs = None
        values = None

        if output_scores:
            log_probs = []

            for i, scores in enumerate(outputs.scores):
                if i < generated_tokens.shape[1]:
                    scores = torch.clamp(scores, min=-5.0, max=5.0)
                    token_log_probs = F.log_softmax(scores, dim=-1)
                    token_log_probs = torch.clamp(token_log_probs, min=-5.0, max=0.0)
                    selected_log_probs = token_log_probs.gather(
                        1, generated_tokens[:, i : i + 1]
                    )
                    log_probs.append(selected_log_probs)

            if log_probs:
                log_probs = torch.cat(log_probs, dim=1)
                log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.tensor(-50.0, device=log_probs.device))
            else:
                logging.warning("No log_probs collected, using zeros!")
                log_probs = torch.zeros_like(
                    generated_tokens, dtype=torch.float, device=self.device
                )

        if output_hidden_states:
            last_hidden_state = outputs.hidden_states[-1]
            last_hidden_state = last_hidden_state[-1].squeeze(1)
            values = self.value_head(last_hidden_state).squeeze(-1)

        return generated_tokens, values, log_probs

    def compute_rewards(
        self,
        generated_texts: list[str],
        true_labels: list[str],
    ):
        rewards, y_pred = self.reward_model.evaluate(generated_texts, true_labels)
        rewards = rewards.to(dtype=self.dtype)
        return rewards, y_pred

    def compute_ppo_loss(self, batch):
        queries = batch["queries"]
        responses = batch["responses"]
        
        input_ids_list = []
        attention_mask_list = []
        
        for query, response in zip(queries, responses):
            full_sequence = torch.cat([query, response], dim=0)
            input_ids_list.append(full_sequence)
            attention_mask_list.append(torch.ones_like(full_sequence))
        
        max_len = max(seq.shape[0] for seq in input_ids_list)
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            padding_length = max_len - input_ids.shape[0]
            if padding_length > 0:
                padded_input_ids.append(
                    F.pad(input_ids, (0, padding_length), value=self.tokenizer.pad_token_id)
                )
                padded_attention_masks.append(
                    F.pad(attention_mask, (0, padding_length), value=0)
                )
            else:
                padded_input_ids.append(input_ids)
                padded_attention_masks.append(attention_mask)
        
        input_ids_tensor = torch.stack(padded_input_ids).to(self.device)
        attention_mask_tensor = torch.stack(padded_attention_masks).to(self.device)
        
        outputs = self.model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            output_hidden_states=True,
        )
        
        batch_size = len(queries)
        current_log_probs_list = []
        current_values_list = []
        
        for i in range(batch_size):
            query_len = queries[i].shape[0]
            response_len = responses[i].shape[0]
            
            response_logits = outputs.logits[i, query_len-1:query_len-1+response_len, :]
            response_logits = torch.clamp(response_logits, min=-5.0, max=5.0)
            response_log_probs = F.log_softmax(response_logits, dim=-1)
            response_log_probs = torch.clamp(response_log_probs, min=-5.0, max=0.0)
            
            response_token_log_probs = response_log_probs.gather(
                1, responses[i].unsqueeze(-1)
            ).squeeze(-1)
            response_token_log_probs = torch.where(
                torch.isfinite(response_token_log_probs), 
                response_token_log_probs, 
                torch.tensor(-5.0, device=response_token_log_probs.device)
            )
            current_log_probs_list.append(response_token_log_probs)
            
            sequence_len = query_len + response_len
            if sequence_len <= outputs.hidden_states[-1].shape[1]:
                last_hidden = outputs.hidden_states[-1][i, sequence_len-1, :]
            else:
                last_hidden = outputs.hidden_states[-1][i, -1, :]
            current_value = self.value_head(last_hidden)
            current_values_list.append(current_value)
        
        max_response_len = max(lp.shape[0] for lp in current_log_probs_list)
        current_log_probs = torch.zeros(batch_size, max_response_len, device=self.device)
        
        for i, log_prob in enumerate(current_log_probs_list):
            current_log_probs[i, :log_prob.shape[0]] = log_prob
        
        current_values = torch.stack(current_values_list).squeeze(-1)
        
        old_log_probs = batch["log_probs"]
        min_len = min(current_log_probs.shape[1], old_log_probs.shape[1])
        current_log_probs = current_log_probs[:, :min_len]
        old_log_probs = old_log_probs[:, :min_len]
        
        mask = (old_log_probs != 0).float()
        
        valid_tokens = mask.sum(dim=1)
        log_ratio = ((current_log_probs - old_log_probs) * mask).sum(dim=1) / (valid_tokens + 1e-8)
        log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
        log_ratio = torch.where(torch.isfinite(log_ratio), log_ratio, torch.zeros_like(log_ratio))
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, min=0.1, max=10.0)
        
        advantages = batch["advantages"]
        returns = batch["returns"]
        
        advantages = torch.where(torch.isfinite(advantages), advantages, torch.zeros_like(advantages))
        returns = torch.where(torch.isfinite(returns), returns, torch.zeros_like(returns))
        
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, min=-5.0, max=5.0)
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange
        )
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        if current_values.dim() > 1:
            current_values = current_values.squeeze(-1)
        
        old_values = batch["values"]
        value_pred_clipped = old_values + torch.clamp(
            current_values - old_values,
            -self.config.cliprange_value,
            self.config.cliprange_value,
        )
        
        value_losses1 = F.mse_loss(current_values, returns, reduction="none")
        value_losses2 = F.mse_loss(value_pred_clipped, returns, reduction="none")
        value_loss = torch.max(value_losses1, value_losses2).mean()
        
        batch_size = len(queries)
        total_entropy = 0.0
        valid_entropy_count = 0
        
        for i in range(batch_size):
            query_len = queries[i].shape[0]
            response_len = responses[i].shape[0]
            
            # Get logits for response tokens only
            response_logits = outputs.logits[i, query_len-1:query_len-1+response_len, :]
            response_logits = torch.clamp(response_logits, min=-5.0, max=5.0)
            
            response_probs = F.softmax(response_logits, dim=-1)
            response_probs = torch.clamp(response_probs, min=1e-8, max=1.0)
            token_entropy = -(response_probs * torch.log(response_probs)).sum(dim=-1)
            
            # Average entropy over valid tokens in this response
            if response_len > 0:
                sequence_entropy = token_entropy.mean()
                if torch.isfinite(sequence_entropy):
                    total_entropy += sequence_entropy
                    valid_entropy_count += 1
        
        # Average entropy across batch
        if valid_entropy_count > 0:
            entropy = total_entropy / valid_entropy_count
        else:
            entropy = torch.tensor(0.0, device=self.device)
            
        entropy_loss = -self.config.entropy_weight * entropy
        
        total_loss = (
            policy_loss + self.config.value_loss_weight * value_loss + entropy_loss
        )
        
        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > self.config.cliprange).float().mean()
            approxkl = (ratio - 1.0 - log_ratio).mean()
            approxkl = torch.clamp(approxkl, min=0.0, max=5.0)
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "clipfrac": clipfrac,
            "approxkl": approxkl,
            "ratio": ratio.mean(),
        }

    def ppo_step(self, data: tuple[str], labels: tuple[str]) -> dict[str, float]:
        self.model.train()
        self.value_head.train()

        tokenized_inputs = self.tokenizer(
            data,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_tokens, values, log_probs = self.generate(
                tokenized_inputs,
                output_hidden_states=True,
            )

            generated_texts = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            rewards, _ = self.compute_rewards(generated_texts, labels)
            rewards = rewards.to(self.device)

            if values.dim() > 1:
                values = values.squeeze(-1)

            advantages = rewards - values
            advantages = torch.where(torch.isfinite(advantages), advantages, torch.zeros_like(advantages))
            
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, min=-5.0, max=5.0)
            
            returns = rewards.clone()
            returns = torch.where(torch.isfinite(returns), returns, torch.zeros_like(returns))

        queries = [tokenized_inputs["input_ids"][i] for i in range(len(data))]
        responses = [generated_tokens[i] for i in range(generated_tokens.shape[0])]

        self.buffer.store(
            queries=queries,
            responses=responses,
            log_probs=[log_probs[i] for i in range(log_probs.shape[0])],
            values=[values[i] for i in range(values.shape[0])],
            rewards=[rewards[i] for i in range(rewards.shape[0])],
            advantages=[advantages[i] for i in range(advantages.shape[0])],
            returns=[returns[i] for i in range(returns.shape[0])],
        )

        total_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "entropy": 0.0,
            "clipfrac": 0.0,
            "approxkl": 0.0,
            "ratio": 0.0,
        }

        num_updates = 0
        
        for _ in range(self.config.ppo_epochs):
            buffer_size = len(self.buffer)
            if buffer_size < self.config.mini_batch_size:
                logging.info("Buffer size is less than mini batch size, skipping PPO update")
                continue

            for _ in range(0, buffer_size, self.config.mini_batch_size):
                batch = self.buffer.get_batch(self.config.mini_batch_size)
                if batch is None:
                    continue

                loss_stats = self.compute_ppo_loss(batch)
                
                self.optimizer.zero_grad()
                loss_stats["total_loss"].backward()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    max_norm=1,
                )
                
                if torch.isfinite(total_norm):
                    self.optimizer.step()
                else:
                    logging.warning(f"Skipping optimizer step due to nan/inf gradients. Norm: {total_norm}")
                    continue

                for key in total_stats:
                    total_stats[key] += loss_stats[key].item()
                num_updates += 1

        if num_updates > 0:
            for key in total_stats:
                total_stats[key] /= num_updates

        total_stats.update(
            {
                "mean_reward": rewards.mean().item(),
                "mean_advantage": advantages.mean().item(),
                "buffer_size": len(self.buffer),
            }
        )

        self.buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return total_stats

    def train(self, use_tqdm: bool = True):
        start_epoch = 0
        total_steps = 0

        if self.config.resume_from_checkpoint:
            if self.config.resume_from_checkpoint == "auto":
                checkpoint_path = self.find_latest_checkpoint()
                logging.info(f"Auto-found checkpoint: {checkpoint_path}")
                start_epoch, total_steps = self.load_checkpoint(checkpoint_path)
            else:
                start_epoch, total_steps = self.load_checkpoint(
                    self.config.resume_from_checkpoint
                )

        logging.info(
            f"Starting training from epoch {start_epoch + 1}, step {total_steps}"
        )

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        for epoch in range(start_epoch, self.config.max_epochs):
            logging.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")

            epoch_stats = {
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "mean_reward": 0.0,
            }

            accumulated_loss = 0.0
            for data, labels in tqdm(dataloader, disable=not use_tqdm):
                stats = self.ppo_step(data, labels)
                accumulated_loss += stats["total_loss"]
                total_steps += 1

                for key in epoch_stats:
                    if key in stats:
                        epoch_stats[key] += stats[key]

                if self.config.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in stats.items()},
                        step=total_steps,
                    )

                if total_steps % self.config.eval_freq == 0:
                    val_stats = self.evaluate()

                    if self.config.use_wandb:
                        wandb.log(
                            {f"val/{k}": v for k, v in val_stats.items()},
                            step=total_steps,
                        )

                if total_steps % self.config.save_freq == 0:
                    self.save_checkpoint(
                        f"checkpoint-{total_steps}", epoch, total_steps
                    )

            num_batches = len(dataloader)
            avg_stats = {k: v / num_batches for k, v in epoch_stats.items()}
            logging.info(f"Epoch {epoch + 1} completed: {avg_stats}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_checkpoint(f"final-model-{timestamp}", epoch, total_steps)
        logging.info("Training completed!")

    def evaluate(self, use_tqdm: bool = False, use_metrics: bool = True):
        self.model.eval()
        self.value_head.eval()

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        total_reward = 0
        num_batches = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for data, labels in tqdm(dataloader, disable=not use_tqdm):
                X = self.tokenizer(
                    data,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                generated_tokens, _, _ = self.generate(X, output_scores=False)
                generated_texts = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                rewards, preds = self.compute_rewards(generated_texts, labels)
                total_reward += rewards.mean().item()
                num_batches += 1

                all_predictions.extend(preds)
                all_labels.extend(labels)

        self.model.train()
        self.value_head.train()

        if not use_metrics:
            return all_predictions, all_labels

        metrics = calculate_metrics(all_predictions, all_labels)
        metrics["mean_reward"] = total_reward / num_batches if num_batches > 0 else 0.0

        return metrics

    def save_checkpoint(self, checkpoint_name: str, epoch: int, total_steps: int):
        save_path = f"../checkpoints/{checkpoint_name}"
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)

        torch.save(
            self.value_head.state_dict(),
            os.path.join(save_path, "value_head.pt"),
        )

        torch.save(
            self.optimizer.state_dict(),
            os.path.join(save_path, "optimizer.pt"),
        )

        training_state = {
            "epoch": epoch,
            "total_steps": total_steps,
            "config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "max_new_tokens": self.config.max_new_tokens,
                "max_epochs": self.config.max_epochs,
                "use_wandb": self.config.use_wandb,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "save_freq": self.config.save_freq,
                "eval_freq": self.config.eval_freq,
                "reward_lambda": self.config.reward_lambda,
                "entropy_weight": self.config.entropy_weight,
                "value_loss_weight": self.config.value_loss_weight,
                "ppo_epochs": self.config.ppo_epochs,
                "mini_batch_size": self.config.mini_batch_size,
                "cliprange": self.config.cliprange,
                "cliprange_value": self.config.cliprange_value,
                "project_name": self.config.project_name,
            },
        }

        with open(os.path.join(save_path, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)

        logging.info(f"Checkpoint saved to {save_path}")

    def save_model(self, checkpoint_name: str):
        save_path = f"../checkpoints/{checkpoint_name}"
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)

        torch.save(
            self.value_head.state_dict(),
            os.path.join(save_path, "value_head.pt"),
        )

        logging.info(f"Model saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> tuple[int, int]:
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        training_state_path = os.path.join(checkpoint_path, "training_state.json")
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(
                f"Training state file not found: {training_state_path}"
            )

        with open(training_state_path, "r") as f:
            training_state = json.load(f)

        self.model.load_adapter(checkpoint_path, adapter_name="default")
        value_head_path = os.path.join(checkpoint_path, "value_head.pt")
        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(
                torch.load(value_head_path, map_location=self.device)
            )
            logging.info("Value head loaded successfully")

        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device)
            )
            logging.info("Optimizer state loaded successfully")

        epoch = training_state["epoch"]
        total_steps = training_state["total_steps"]

        logging.info(
            f"Checkpoint loaded successfully - Epoch: {epoch}, Steps: {total_steps}"
        )
        return epoch, total_steps

    def find_latest_checkpoint(
        self, checkpoint_dir: str = "../checkpoints"
    ) -> Optional[str]:
        if not os.path.exists(checkpoint_dir):
            return None

        checkpoint_pattern = os.path.join(
            checkpoint_dir, "checkpoint-*", "training_state.json"
        )
        checkpoint_files = glob.glob(checkpoint_pattern)

        if not checkpoint_files:
            return None

        checkpoint_steps = []
        for file_path in checkpoint_files:
            dir_path = os.path.dirname(file_path)
            checkpoint_name = os.path.basename(dir_path)
            try:
                step = int(checkpoint_name.split("-")[1])
                checkpoint_steps.append((step, dir_path))
            except (IndexError, ValueError):
                continue

        if not checkpoint_steps:
            return None

        latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])[1]
        return latest_checkpoint
