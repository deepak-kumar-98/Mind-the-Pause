from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
import os

# Set cache location
os.environ["HF_DATASETS_CACHE"] = "/home/paritosh/.cache/huggingface/datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATHS = {
    "train": "/home/paritosh/zero-shot-disfluency-detection/disfluency_signal_llama/alpaca_disfluency_dataset_train.jsonl",
    "validation": "/home/paritosh/zero-shot-disfluency-detection/disfluency_signal_llama/alpaca_disfluency_dataset_val.jsonl",
    # "test": "/home/paritosh/zero-shot-disfluency-detection/detection_hi_bn_mr/llama_training/alpaca_disfluency_dataset_test.jsonl"
}
OUTPUT_DIR = "/home/paritosh/zero-shot-disfluency-detection/disfluency_signal_llama/output1"
MAX_LENGTH = 512

# 1. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# 2. Dataset preparation using given format
def format_dataset(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    ) + tokenizer.eos_token
    return {"text": text}

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

# Load and process dataset
dataset = load_dataset("json", data_files=DATA_PATHS, cache_dir=os.environ["HF_DATASETS_CACHE"])
dataset = dataset.map(format_dataset)
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    batch_size=4
)

# 3. Training configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1.5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=5,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    optim="adamw_torch_fused",
    remove_unused_columns=False,
    torch_compile=True,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    lr_scheduler_type="cosine_with_restarts",
    max_grad_norm=1.0,
    )

# 4. Trainer setup
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator
)

# 5. Memory optimizations
torch.cuda.empty_cache()
torch.backends.cuda.enable_mem_efficient_sdp(True)

# 6. Training
print("Starting training...")
try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {str(e)}")
    print("Trying fallback configuration...")
    model.gradient_checkpointing_disable()
    trainer.train()

# print("\nEvaluating on test set...")
# test_results = trainer.evaluate(tokenized_dataset["test"])
# print(f"Test loss: {test_results['eval_loss']:.4f}")

# Save final model
trainer.save_model(f"{OUTPUT_DIR}/final_model")
print("Training completed!")
