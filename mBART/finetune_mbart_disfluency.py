# finetune_mbart_disfluency.py
import argparse, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import EarlyStoppingCallback

import numpy as np
from datasets import load_dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import torch

@dataclass
class Config:
    model_name: str = "facebook/mbart-large-50"
    max_source_length: int = 128
    max_target_length: int = 128

def preprocess_function(examples, tokenizer: MBart50TokenizerFast, cfg: Config):
    src_langs = examples["mbart_lang"]
    sources = examples["source"]
    targets = examples["target"]

    # Tokenize source per example with its language code
    model_inputs = {"input_ids": [], "attention_mask": []}
    labels = []

    for s, t, lcode in zip(sources, targets, src_langs):
        tokenizer.src_lang = lcode
        tokenizer.tgt_lang = lcode
        enc = tokenizer(
            s,
            max_length=cfg.max_source_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(
                t,
                max_length=cfg.max_target_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
        model_inputs["input_ids"].append(enc["input_ids"])
        model_inputs["attention_mask"].append(enc["attention_mask"])
        labels.append(dec["input_ids"])

    model_inputs["labels"] = labels
    return model_inputs

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with train.jsonl and valid.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default="facebook/mbart-large-50")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--num_epochs", type=float, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_source_length", type=int, default=128)
    ap.add_argument("--max_target_length", type=int, default=128)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--push_to_hub", action="store_true")
    args = ap.parse_args()

    cfg = Config(model_name=args.model_name,
                 max_source_length=args.max_source_length,
                 max_target_length=args.max_target_length)

    # Load data
    data_files = {
        "train": f"{args.data_dir}/train.jsonl",
        "validation": f"{args.data_dir}/valid.jsonl",
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    # Model & tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(cfg.model_name)
    model = MBartForConditionalGeneration.from_pretrained(cfg.model_name)
    model.gradient_checkpointing_enable()

    # Ensure pad token id
    model.config.pad_token_id = tokenizer.pad_token_id
    # We set decoder_start_token_id per-example implicitly through tokenizer.tgt_lang during generation

    # Tokenize
    tokenized = raw_datasets.map(
        lambda batch: preprocess_function(batch, tokenizer, cfg),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding="longest"
    )

    # Metrics: chrF++
    chrf = evaluate.load("chrf")

    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    #     # Replace -100 in labels
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    #     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    #     result = chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)  # chrF++
    #     return {"chrf++": result}
    # 1) replace compute_metrics with this:
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # chrF++ (word_order=2). Sacrebleu returns a dict with "score"
        res = chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        return {"chrfpp": float(res["score"])}  # <-- a plain float


    # Training args
    torch.backends.cuda.matmul.allow_tf32 = True
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=5,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        load_best_model_at_end=True,
        # metric_for_best_model="chrf++",
        metric_for_best_model="chrfpp",
        greater_is_better=True,
        label_smoothing_factor=0.1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train(resume_from_checkpoint="/home/paritosh/zero-shot-disfluency-detection/mBART/runs/mbart_m50_disfluency/checkpoint-18000")

    # Save final
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub()

    print("Done. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()
