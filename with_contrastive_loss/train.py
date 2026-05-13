#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

# ----------------- Config -----------------
os.environ["HF_DATASETS_CACHE"] = "/home/paritosh/.cache/huggingface/datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATHS = {
    "train": "/home/paritosh/zero-shot-disfluency-detection/contrastive_p2/contrastive_p3/disfluency_dataset_train.jsonl",
    "validation": "/home/paritosh/zero-shot-disfluency-detection/contrastive_p2/contrastive_p3/disfluency_dataset_val.jsonl",
}
OUTPUT_DIR = "//home/paritosh/zero-shot-disfluency-detection/contrastive_p2/contrastive_p3/output1"
MAX_LENGTH = 512

# # Contrastive penalty hyperparameters
# LAMBDA_BASE = 0.2          # start modestly; tune 0.2–0.5
# LAMBDA_WARMUP_FRAC = 0.1   # first 10% steps ramp λ from 0 -> LAMBDA_BASE
# LABEL_SMOOTH = 0.05        # CE label smoothing

LAMBDA_BASE = 0.3
LAMBDA_WARMUP_FRAC = 0.3
LABEL_SMOOTH = 0.01

# Perf knobs
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ===================== Model & Tokenizer =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# ===================== Dataset processing =====================
def format_dataset(example):
    """
    Expected JSON per sample:
      - instruction: str
      - input: str   (contains: Disfluent sentence, Tokens: [...], Labels: [...])
      - output: str  (gold fluent sentence)
      - disfluent_tokens: list[str]  (word-level; can be empty)
    """
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n"
    )
    full_text = prompt + example["output"] + tokenizer.eos_token

    # response_start = tokenized length of the prompt (so penalty applies only to response tokens)
    rs_ids = tokenizer(
        prompt, max_length=MAX_LENGTH, truncation=True, add_special_tokens=True
    ).input_ids
    response_start = len(rs_ids)

    return {
        "text": full_text,
        "disfluent_tokens": example.get("disfluent_tokens", []),  # list[str]
        "response_start": response_start,
    }

def tokenize_function(examples):
    tok = tokenizer(
        examples["text"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    # labels mirror input_ids; collator will set pad labels to -100
    tok["labels"] = tok["input_ids"][:]
    tok["disfluent_tokens"] = examples.get("disfluent_tokens", [])
    tok["response_start"] = examples.get("response_start", 0)
    return tok

# Load → format → tokenize
raw = load_dataset("json", data_files=DATA_PATHS, cache_dir=os.environ["HF_DATASETS_CACHE"])
orig_cols = raw["train"].column_names

formatted = raw.map(
    format_dataset,
    remove_columns=orig_cols,
    desc="Formatting dataset"
)
fmt_cols = formatted["train"].column_names  # ['text','disfluent_tokens','response_start']

tokenized_dataset = formatted.map(
    tokenize_function,
    batched=True,
    batch_size=64,
    remove_columns=fmt_cols,
    desc="Tokenizing dataset"
)
# tokenized columns: ['input_ids','attention_mask','labels','disfluent_tokens','response_start']

# ===================== Collator =====================
class DataCollatorForContrastiveDisfluency(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Pop custom fields first
        disfluent_tokens = [f.pop("disfluent_tokens", []) for f in features]
        response_start = torch.tensor(
            [f.pop("response_start", 0) for f in features],
            dtype=torch.long
        )
        # Keep only expected keys; tensorize lists
        allowed = {"input_ids", "attention_mask", "labels"}
        for f in features:
            for k in list(f.keys()):
                if k not in allowed:
                    f.pop(k, None)
        for f in features:
            for k, v in list(f.items()):
                if isinstance(v, list):
                    f[k] = torch.tensor(v, dtype=torch.long)

        batch = super().__call__(features)  # pads & sets pad labels to -100
        batch["disfluent_tokens"] = disfluent_tokens
        batch["response_start"] = response_start
        return batch

data_collator = DataCollatorForContrastiveDisfluency(tokenizer=tokenizer, mlm=False)

# ===================== Trainer with Weighted Unlikelihood =====================
class ContrastiveDisfluencyTrainer(Trainer):
    def __init__(self, *args, lambda_base=LAMBDA_BASE, lambda_warmup_frac=LAMBDA_WARMUP_FRAC, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_base = lambda_base
        self.lambda_warmup_frac = lambda_warmup_frac

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Custom fields
        raw_disfluent_words = inputs.pop("disfluent_tokens", None)  # list[list[str]]
        response_start = inputs.pop("response_start", None)         # LongTensor[B]
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs["logits"]     # [B, T, V]
        B, T, V = logits.shape

        # ----- CE with label smoothing -----
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTH)
        ce = ce_loss_fct(
            logits[..., :-1, :].contiguous().view(-1, V),
            labels[..., 1:].contiguous().view(-1)
        )

        # ----- Build weighted negative vocab mask over subwords -----
        forbid = {
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "unk_token_id", None),
        }
        forbid = {x for x in forbid if x is not None}

        if raw_disfluent_words is None:
            raw_disfluent_words = [[] for _ in range(B)]

        neg_mask = torch.zeros(B, V, device=logits.device)
        for i, words in enumerate(raw_disfluent_words):
            ids_weight = {}
            for w in words:
                w = (w or "").strip()
                if not w:
                    continue
                sp = self.tokenizer.encode(w, add_special_tokens=False)
                # geometric decay: 1.0, 0.5, 0.25, ...
                for j, tid in enumerate(sp):
                    if tid in forbid:
                        continue
                    weight = 0.5 ** j
                    if tid not in ids_weight or weight > ids_weight[tid]:
                        ids_weight[tid] = weight
            if ids_weight:
                idx = torch.tensor(list(ids_weight.keys()), device=logits.device)
                val = torch.tensor(list(ids_weight.values()), device=logits.device)
                neg_mask[i, idx] = val

        # ----- response-only mask -----
        if response_start is None:
            resp_mask = torch.ones(B, T, device=logits.device)
        else:
            if not torch.is_tensor(response_start):
                response_start = torch.tensor(response_start, device=logits.device, dtype=torch.long)
            arange_t = torch.arange(T, device=logits.device).unsqueeze(0).expand(B, -1)
            resp_mask = (arange_t >= response_start.unsqueeze(1)).float()

        ce_attn = (labels != -100).float()
        final_mask = resp_mask * ce_attn
        denom = final_mask.sum().clamp_min(1.0)

        # ----- Unlikelihood penalty -----
        probs = torch.softmax(logits, dim=-1)
        p_neg = (probs * neg_mask.unsqueeze(1)).sum(dim=-1)    # [B, T]
        ul = -torch.log(torch.clamp(1.0 - p_neg, min=1e-6))    # [B, T]
        penalty = (ul * final_mask).sum() / denom

        # ----- λ warmup -----
        if self.state.max_steps and self.lambda_warmup_frac > 0:
            frac = min(1.0, (self.state.global_step + 1) / (self.state.max_steps * self.lambda_warmup_frac))
        else:
            frac = 1.0
        lam = self.lambda_base * frac

        total = ce + lam * penalty
        return (total, outputs) if return_outputs else total

# ===================== Training args =====================
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=4,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=8,
#     gradient_accumulation_steps=1,
#     learning_rate=1.5e-5,
#     weight_decay=0.01,
#     warmup_ratio=0.1,
#     logging_steps=500,
#     eval_strategy="steps",
#     save_strategy="steps",
#     eval_steps=1000,
#     save_steps=1000,
#     save_total_limit=5,
#     bf16=True,
#     gradient_checkpointing=True,
#     gradient_checkpointing_kwargs={"use_reentrant": True},
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     report_to="tensorboard",
#     optim="adamw_torch_fused",
#     remove_unused_columns=True,
#     torch_compile=False,
#     dataloader_num_workers=16,
#     dataloader_pin_memory=True,
#     dataloader_persistent_workers=True,
#     lr_scheduler_type="cosine_with_restarts",
#     max_grad_norm=1.0,
#     group_by_length=True,
#     # If you run multi-GPU via torchrun, DDP will be used automatically (NCCL by default)
#     ddp_find_unused_parameters=False,
# )

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,        # effective batch 32
    learning_rate=1e-5,                   # lower LR
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=200,
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
    lr_scheduler_type="cosine",           # no restarts
    max_grad_norm=1.0,
    group_by_length=True,
    ddp_find_unused_parameters=False,
)

trainer = ContrastiveDisfluencyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,  # avoids deprecation warning
    data_collator=data_collator,
    lambda_base=LAMBDA_BASE,
    lambda_warmup_frac=LAMBDA_WARMUP_FRAC,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ===================== Train & Eval =====================
torch.cuda.empty_cache()
torch.backends.cuda.enable_mem_efficient_sdp(True)

print("Starting training...")
try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")
    print("Retrying without gradient checkpointing and smaller batch...")
    model.gradient_checkpointing_disable()
    trainer.args.per_device_train_batch_size = 4
    trainer.train()

# print("\nEvaluating on test set...")
# test_results = trainer.evaluate(tokenized_dataset["test"])
# print(f"Test loss: {test_results['eval_loss']:.4f}")

trainer.save_model(f"{OUTPUT_DIR}/final_model")
print("Training completed!")
