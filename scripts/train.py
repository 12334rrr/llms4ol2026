#!/usr/bin/env python
"""
LLMs4OL 2026 — 训练入口

Usage:
    python scripts/train.py --gpu auto
    python scripts/train.py --gpu a100 --epochs 5
    python scripts/train.py --config configs/a100_40gb.json
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import TrainingArguments, Trainer

from src import (
    Config, load_config, detect_gpu, MODEL_RECOMMENDATIONS,
    load_and_prepare_data, OntologyDataset,
    setup_model_and_tokenizer,
    ReportCallback, generate_report,
)


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            p = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * p)
            attn.append([1] * len(b["input_ids"]) + [0] * p)
            labels.append(b["labels"] + [-100] * p)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def main():
    parser = argparse.ArgumentParser(description="LLMs4OL 2026 Training")
    parser.add_argument("--config", type=str, default=None, help="JSON config preset")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU type: auto/a100/v100/t4/rtx3090")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    config = load_config(config_path=args.config, gpu_hint=None if args.config else args.gpu)
    if args.model:      config.MODEL_NAME = args.model
    if args.epochs:     config.NUM_EPOCHS = args.epochs
    if args.batch_size: config.BATCH_SIZE = args.batch_size
    if args.lr:         config.LEARNING_RATE = args.lr
    if args.lora_r:     config.LORA_R = args.lora_r
    if args.output_dir: config.OUTPUT_DIR = args.output_dir
    if args.no_4bit:    config.USE_4BIT = False
    if args.multi_gpu:  config.GPU_COUNT = detect_gpu()["count"]

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # ── GPU Info ──
    print("=" * 60)
    print("GPU / Model Selection")
    print("=" * 60)
    print(f"  GPU:     {config.GPU_NAME} ({config.GPU_MEMORY_GB:.0f} GB × {config.GPU_COUNT})")
    print(f"  Model:   {config.MODEL_NAME}")
    print(f"  Preset:  {config.PRESET}")
    print(f"  LoRA:    r={config.LORA_R}, α={config.LORA_ALPHA}")
    print(f"  Batch:   {config.BATCH_SIZE} × {config.GRADIENT_ACCUMULATION} = {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION}")
    print("=" * 60)

    # ── Data ──
    print("\nLoading data...")
    train_data, val_data = load_and_prepare_data(
        config.TRAIN_PATH, config.VAL_SPLIT, config.SEED,
        config.OVERSAMPLE_THRESHOLD, config.OVERSAMPLE_MULTIPLIER,
    )

    # ── Model ──
    print("\nLoading model...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # ── Training ──
    train_ds = OntologyDataset(train_data, tokenizer, config.MAX_LENGTH)
    val_ds = OntologyDataset(val_data, tokenizer, config.MAX_LENGTH)
    collator = DataCollator(tokenizer.pad_token_id)
    report_cb = ReportCallback(config, config.OUTPUT_DIR)

    eff_batch = config.BATCH_SIZE * config.GRADIENT_ACCUMULATION * max(1, config.GPU_COUNT)
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR, num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        warmup_ratio=config.WARMUP_RATIO, learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY, max_grad_norm=config.MAX_GRAD_NORM,
        logging_steps=config.LOGGING_STEPS, save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        evaluation_strategy="steps", save_strategy="steps",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, fp16=config.FP16,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        report_to="none", dataloader_num_workers=0,
        remove_unused_columns=False, save_total_limit=config.SAVE_TOTAL_LIMIT,
    )

    print(f"\nTraining: epochs={config.NUM_EPOCHS}, eff_batch={eff_batch}, lr={config.LEARNING_RATE}\n")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator, callbacks=[report_cb])
    trainer.train()

    # ── Save ──
    final_path = os.path.join(config.OUTPUT_DIR, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Model saved to: {final_path}")

    # ── Report ──
    generate_report(config, report_cb, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
