import argparse
import os
from dataclasses import asdict

import pandas as pd
import torch.optim as optim
from llama_recipes.configs import lora_config as LORA_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.utils.train_utils import train
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim.lr_scheduler import StepLR

from llama_helper import get_data, load_llama, reformat_to_llama


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_dir", type=str, help="Directory of base LLM")
    parser.add_argument(
        "--context_dict_path",
        type=str,
        help="Path to pickle file with context sentences "
             "(e.g. /resources/context_data/context_fineTune/ft_context_dict.pkl)",
    )
    parser.add_argument(
        "--target_symptoms_path",
        type=str,
        help="Path to target symptoms file (e.g. /resources/data/UTI/target_symptoms.csv).",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save fine-tuned model")
    return parser.parse_args()


args = parse_args()

# set system prompt
SYSTEM_PROMPT = (
    "You are a medical expert deciding whether a patient has a certain symptom. Answer only with 'Yes' or 'No'."
)

# get model and tokenizer
tokenizer, model = load_llama(args.llama_dir)

# create output directory
os.makedirs(args.output_dir, exist_ok=True)

# set training configurations
train_config = TRAIN_CONFIG()
train_config.num_epochs = 1
train_config.run_validation = False
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 512
train_config.batching_strategy = "packing"
train_config.output_dir = args.output_dir

# set lora configurations
lora_config = LORA_CONFIG()
lora_config.r = 8
lora_config.lora_alpha = 32
lora_dropout: float = 0.01

# get training data
target_symptoms = pd.read_csv(args.target_symptoms_path)["target_codes"].tolist()
prompt_list, response_list = get_data(args.context_dict_path, target_symptoms)
train_dataloader, eval_dataloader = reformat_to_llama(prompt_list, response_list, SYSTEM_PROMPT, tokenizer)

# prepare model
peft_config = LoraConfig(**asdict(lora_config))
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.train()

# set optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=train_config.lr,
    weight_decay=train_config.weight_decay,
)
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

# Start the training process
results = train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    scheduler,
    train_config.gradient_accumulation_steps,
    train_config,
    None,
    None,
    None,
)
