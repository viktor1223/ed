"""TF-IDF + Logistic Regression baseline for misconception classification.

Usage:
    python src/baseline_tfidf.py

Trains and evaluates a simple baseline for comparison with DeBERTa.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset"
KG_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"


def load_labels() -> list[str]:
    with open(KG_PATH) as f:
        kg = json.load(f)
    ids = []
    for concept in kg["concepts"]:
        for m in concept["misconceptions"]:
            ids.append(m["id"])
    return sorted(ids) + ["correct"]


def load_split(split: str) -> tuple[list[str], list[str]]:
    with open(DATA_DIR / f"{split}.json") as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data:
        label = item["misconception_id"]
        if label is None:
            continue  # Skip MaE examples without mapped misconception IDs
        text = f"Question: {item['question']} Student answer: {item['student_response']}"
        texts.append(text)
        labels.append(label)
    return texts, labels


def main():
    label_names = load_labels()

    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)

    # Logistic Regression
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_labels)

    # Validation
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

    print(f"\n=== Validation ===")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  F1 Macro: {val_f1:.4f}")
    print("\n" + classification_report(val_labels, val_preds, zero_division=0))

    # Test
    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)

    print(f"=== Test ===")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Macro: {test_f1:.4f}")
    print("\n" + classification_report(test_labels, test_preds, zero_division=0))

    # Save results
    results = {
        "model": "TF-IDF + LogisticRegression",
        "train_size": len(train_texts),
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
    }
    out_path = Path(__file__).resolve().parent.parent / "results" / "baseline_tfidf.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
