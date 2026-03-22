"""Validate Phase 3 dataset against PLAYBOOK.md checklist."""
import json
from collections import Counter

def main():
    splits = {}
    for name in ["train", "val", "test"]:
        with open(f"data/dataset/{name}.json") as f:
            splits[name] = json.load(f)
        print(f"{name}: {len(splits[name])} examples")

    total = sum(len(v) for v in splits.values())
    print(f"\nTotal: {total}")
    print(f"Split ratios: train={100*len(splits['train'])/total:.1f}%  val={100*len(splits['val'])/total:.1f}%  test={100*len(splits['test'])/total:.1f}%")
    print("Target: 70/15/15")

    # Per-concept counts across all splits
    all_data = splits["train"] + splits["val"] + splits["test"]
    concept_counts = Counter(d["concept_id"] for d in all_data)
    print("\nPer-concept totals:")
    for c, n in sorted(concept_counts.items()):
        status = "PASS" if n >= 100 else "FAIL"
        print(f"  {c}: {n}  [{status}]")

    # Stratification check
    print("\nStratification (concept % per split):")
    for name in ["train", "val", "test"]:
        dist = Counter(d["concept_id"] for d in splits[name])
        n = len(splits[name])
        pcts = {c: 100*dist[c]/n for c in sorted(concept_counts)}
        print(f"  {name}: " + "  ".join(f"{c}={pcts[c]:.1f}%" for c in sorted(pcts)))

    # Data leakage check
    def fingerprint(d):
        return d["question"] + "|||" + d["incorrect_answer"]

    train_fp = set(fingerprint(d) for d in splits["train"])
    val_fp = set(fingerprint(d) for d in splits["val"])
    test_fp = set(fingerprint(d) for d in splits["test"])

    tv = train_fp & val_fp
    tt = train_fp & test_fp
    vt = val_fp & test_fp
    leakage = len(tv) + len(tt) + len(vt)
    print(f"\nData leakage: train-val={len(tv)}  train-test={len(tt)}  val-test={len(vt)}  total={leakage}  [{'PASS' if leakage == 0 else 'FAIL'}]")

    # Dataset card
    with open("data/dataset/dataset_card.json") as f:
        card = json.load(f)
    print(f"\nDataset card keys: {list(card.keys())}  [PASS]")

    # Source distribution
    sources = Counter(d["source"] for d in all_data)
    print(f"\nSource breakdown: {dict(sources)}")

    # Summary
    print("\n=== Phase 3 Checklist ===")
    checks = [
        (">=100 examples per concept", all(n >= 100 for n in concept_counts.values())),
        ("Stratified splits ~70/15/15", abs(len(splits["train"])/total - 0.70) < 0.05),
        ("No data leakage", leakage == 0),
        ("Structured JSON format", True),
        ("Dataset card exists", True),
    ]
    for label, ok in checks:
        print(f"  [{'x' if ok else ' '}] {label}")

    print("\n  [ ] 20% synthetic manually reviewed (requires human action)")

if __name__ == "__main__":
    main()
