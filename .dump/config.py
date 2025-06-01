DEBUG_CONFIG = {
    "learning_rate": 5e-5,
    "batch_size": 4,
    "max_new_tokens": 256,
    "max_epochs": 1,

    "lora_r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    
    "save_freq": 50,
    "eval_freq": 25,
    
    "reward_lambda": 1.0,
    "entropy_weight": 0.02,
    "value_loss_weight": 1.0,

    "ppo_epochs": 2,
    "mini_batch_size": 2,
    "cliprange": 0.1,
    "cliprange_value": 0.1,
    
    "use_wandb": False,
    "project_name": None,
    "resume_from_checkpoint": None,
}

TRAIN_CONFIG = {
    "learning_rate": 3e-5,      # Increased back for better learning
    "batch_size": 24,           # Increased batch size
    "max_new_tokens": 512,
    "max_epochs": 10,

    "lora_r": 32,               # Increased rank back
    "lora_alpha": 64,           # Matched to r * 2
    "lora_dropout": 0.1,        # Standard dropout
    
    "save_freq": 25,
    "eval_freq": 50,
    
    "reward_lambda": 1.0,
    "entropy_weight": 0.01,     # Standard entropy weight
    "value_loss_weight": 0.5,   # Standard value loss weight

    "ppo_epochs": 3,            # Standard PPO epochs
    "mini_batch_size": 4,       # Standard mini-batch size
    "cliprange": 0.2,           # Standard PPO clipping
    "cliprange_value": 0.2,     # Standard value clipping
    
    "use_wandb": True,
    "project_name": "ape-finetune",
    "resume_from_checkpoint": None,
}

EVAL_CONFIG = {
    "learning_rate": 1e-6,
    "batch_size": 16,
    "max_new_tokens": 512,
    "max_epochs": 1,

    "lora_r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    
    "save_freq": 50,
    "eval_freq": 25,
    
    "reward_lambda": 1.0,
    "entropy_weight": 0.02,
    "value_loss_weight": 1.0,

    "ppo_epochs": 2,
    "mini_batch_size": 2,
    "cliprange": 0.1,
    "cliprange_value": 0.1,
    
    "use_wandb": False,
    "project_name": None,
    "resume_from_checkpoint": None,
}
