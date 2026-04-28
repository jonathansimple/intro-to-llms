import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

# Define model names
model_name = "google/gemma-2-2b-it"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Load model and tokenizer with full fine-tuning
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager',)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# You must disable the cache to prevent issues during training
model.config.use_cache = False

# Load the Gemma tokenizer
tokenizer.padding_side = "right" 

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:80%]")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

block_size = 512  # Adjust based on available VRAM

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // block_size) * block_size
    result = {
        "input_ids": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)],
        "labels": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
    }
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./gemma2-2b-it-full-finetuned",   # Save location
    overwrite_output_dir=True,
    num_train_epochs=3,              # Adjust based on the dataset size and model performance
    per_device_train_batch_size=2,   # Small batch size to prevent VRAM overload
    save_steps=500,                  # Save model after every 500 steps
    save_total_limit=2,              # Keep only the last 2 checkpoints
    logging_dir="./logs",            # Log file location
    logging_steps=50,                # Log every 50 steps
    eval_steps=500,                  # Evaluate after every 500 steps
    warmup_steps=100,                # Warm-up steps for learning rate scheduler
    weight_decay=0.01,               # Weight decay (for regularization)
    learning_rate=5e-5,              # Adjust learning rate based on the model size
    fp16=True,                       # Mixed-precision training (use if your GPU supports it)
    report_to="none"                 # Disable reporting (e.g., to MLFlow)
)


from transformers import Trainer, DataCollatorForLanguageModeling

# Data collator for language modeling (dynamic padding, etc.)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
)

# Start the training
trainer.train()

model.save_pretrained("./gemma2-2b-it-finetuned")
tokenizer.save_pretrained("./gemma2-2b-it-finetuned")
