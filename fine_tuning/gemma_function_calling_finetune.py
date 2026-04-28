import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
import pandas as pd
from trl import SFTTrainer, SFTConfig

# Define model names
model_name = "google/gemma-2-2b-it"
new_model = "gemma-func-ft"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Load the Gemma pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation='eager',
    #img_context_token_id=128212, #breeze
)

# You must disable the cache to prevent issues during training
model.config.use_cache = False

# Load the Gemma tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right" 

# Get the current working directory and ensure it's cross-platform friendly
dataset_dir = os.path.join(os.getcwd(), "dataset")
dataset = load_dataset(dataset_dir, split="train[:80%]")

chat_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['from'] == 'system' %}"\
        "{{'user\n' + messages[0]['value'] | trim + ' ' + messages[1]['value'] | trim + '\n'}}"\
        "{% set messages = messages[2:] %}"\
    "{% endif %}"\
    "{% for message in messages %}"\
        "{% if message['from'] == 'human' %}"\
            "{{'user\n' + message['value'] | trim + '\n'}}"\
        "{% elif message['from'] == 'gpt' %}"\
            "{{'model\n' + message['value'] | trim + '\n' }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ 'model\n' }}"\
    "{% endif %}"

tokenizer.chat_template = chat_template

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False,
                      add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)

df_train = pd.DataFrame(dataset)
df_train["text"] = df_train["text"].apply(
    lambda x: x.replace("<|endoftext|>", ""))

pd.set_option('display.max_colwidth', None)
print(df_train.head(1))

dataset = Dataset.from_pandas(df_train[['text']])

print(dataset)

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=32,       # Alpha parameter for LoRA scaling
    lora_dropout=0,    # Dropout probability for LoRA layers
    r=16,                # LoRA attention dimension
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",]
)


# Set training parameters
training_arguments = SFTConfig(
    output_dir="./results",
    overwrite_output_dir=True,
    save_strategy="no",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="adafactor",
    dataloader_drop_last=True,
    learning_rate=0.0002,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    max_seq_length=1024,
    dataset_text_field="text",
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
    packing=True,
    logging_steps=1,
    report_to="none",
    seed=42
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments
)

trainer.train()

# Save the LoRA adapter
trainer.model.to('cpu').save_pretrained(new_model)

model = PeftModel.from_pretrained(model, "./" + new_model)
model = model.merge_and_unload()
print(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./gemma-ft")
tokenizer.save_pretrained("./gemma-ft")

