"""Train a DeBERTa-v3-base misconception classifier.

Usage:
    python src/train_classifier.py [--epochs 10] [--lr 2e-5] [--batch_size 16] [--output_dir models/deberta]

Produces a fine-tuned model and evaluation metrics on the validation set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset"
KG_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_labels() -> list[str]:
    """Load label list from the knowledge graph (sorted misconception IDs + 'correct')."""
    with open(KG_PATH) as f:
        kg = json.load(f)
    ids = []
    for concept in kg["concepts"]:
        for m in concept["misconceptions"]:
            ids.append(m["id"])
    return sorted(ids) + ["correct"]


def load_split(split: str, label2id: dict[str, int]) -> list[dict]:
    path = DATA_DIR / f"{split}.json"
    with open(path) as f:
        data = json.load(f)
    examples = []
    for item in data:
        text = f"Question: {item['question']}\nStudent answer: {item['student_response']}"
        label = item["misconception_id"]
        if label not in label2id:
            continue
        examples.append({"text": text, "label": label2id[label]})
    return examples


class MisconceptionDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    # Macro F1
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1}


def main():
    parser = argparse.ArgumentParser(description="Train misconception classifier")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="models/deberta")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(SEED)

    # Labels
    labels = load_labels()
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels)
    print(f"Labels ({num_labels}): {labels}")

    # Data
    train_examples = load_split("train", label2id)
    val_examples = load_split("val", label2id)
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    train_dataset = MisconceptionDataset(train_examples, tokenizer, args.max_length)
    val_dataset = MisconceptionDataset(val_examples, tokenizer, args.max_length)

    # Determine device
    if torch.cuda.is_available():
        fp16 = True
        use_cpu = False
        device_note = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        fp16 = False
        use_cpu = False
        device_note = "Apple MPS"
    else:
        fp16 = False
        use_cpu = False
        device_note = "CPU"
    print(f"Device: {device_note}")

    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=fp16,
        use_cpu=use_cpu,
        logging_steps=10,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("\n=== Training ===")
    trainer.train()

    # Evaluate
    print("\n=== Validation Results ===")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Detailed classification report
    preds_out = trainer.predict(val_dataset)
    preds = np.argmax(preds_out.predictions, axis=-1)
    true_labels = [ex["label"] for ex in val_examples]
    target_names = [id2label[i] for i in sorted(set(true_labels) | set(preds))]
    print("\n=== Per-Class Report ===")
    print(classification_report(true_labels, preds, target_names=target_names, zero_division=0))

    # Save
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    # Save metrics
    results = {
        "model": args.model_name,
        "epochs": args.epochs,
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "val_accuracy": float(metrics.get("eval_accuracy", 0)),
        "val_f1_macro": float(metrics.get("eval_f1_macro", 0)),
        "labels": labels,
    }
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nModel saved to {output_dir / 'best'}")
    print(f"Results saved to {output_dir / 'training_results.json'}")


if __name__ == "__main__":
    main()
