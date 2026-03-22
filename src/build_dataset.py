"""
Phase 3: Dataset Assembly Pipeline
Downloads MaE data, filters to target concepts, generates synthetic augmentation,
and produces train/val/test splits.
"""
import json
import os
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# MaE IDs mapped to our 5 concepts (from knowledge_graph.json)
CONCEPT_MAE_MAP = {
    "integer_sign_ops": {
        "mae_ids": ["MaE06", "MaE07", "MaE08", "MaE09", "MaE10"],
        "topic_filter": "Number operations",
    },
    "order_of_operations": {
        "mae_ids": ["MaE20", "MaE21", "MaE22"],
        "topic_filter": "Number operations",
    },
    "distributive_property": {
        "mae_ids": ["MaE31", "MaE32", "MaE33", "MaE34"],
        "topic_filter": "Properties of numbers and operations",
    },
    "combining_like_terms": {
        "mae_ids": ["MaE45", "MaE46", "MaE47", "MaE48"],
        "topic_filter": "Variables, expressions, and operations",
    },
    "solving_linear_equations": {
        "mae_ids": ["MaE49", "MaE50", "MaE51", "MaE52", "MaE53", "MaE54", "MaE55"],
        "topic_filter": "Equations and inequalities",
    },
}

# Build reverse map: MaE ID -> concept_id
MAE_TO_CONCEPT = {}
for concept_id, info in CONCEPT_MAE_MAP.items():
    for mae_id in info["mae_ids"]:
        MAE_TO_CONCEPT[mae_id] = concept_id

TARGET_MAE_IDS = set(MAE_TO_CONCEPT.keys())

# ─── STEP 1: LOAD MaE DATA ──────────────────────────────────────────────
def load_mae_data():
    """Download and load the MaE dataset from HuggingFace."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "nanote/algebra_misconceptions", "data/data.json", repo_type="dataset"
    )
    with open(path) as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} MaE examples total")
    return raw


# ─── STEP 2: FILTER TO TARGET CONCEPTS ──────────────────────────────────
def filter_to_concepts(raw_data):
    """Filter MaE data to our 5 target concept areas."""
    filtered = []
    for item in raw_data:
        mae_id = item["Misconception ID"]
        if mae_id in TARGET_MAE_IDS:
            concept_id = MAE_TO_CONCEPT[mae_id]
            filtered.append(
                {
                    "source": "mae",
                    "mae_id": mae_id,
                    "concept_id": concept_id,
                    "misconception": item["Misconception"],
                    "question": item["Question"],
                    "incorrect_answer": item["Incorrect Answer"],
                    "correct_answer": item["Correct Answer"],
                    "explanation": item.get("Explanation", ""),
                    "example_number": item["Example Number"],
                }
            )

    print(f"Filtered to {len(filtered)} examples across target concepts")

    # Report distribution
    concept_counts = Counter(item["concept_id"] for item in filtered)
    mae_counts = Counter(item["mae_id"] for item in filtered)
    print("\nPer-concept distribution:")
    for cid, count in sorted(concept_counts.items()):
        print(f"  {cid}: {count} examples")
    print(f"\nPer-MaE ID distribution:")
    for mid, count in sorted(mae_counts.items()):
        print(f"  {mid}: {count}")

    return filtered


# ─── STEP 3: SYNTHETIC DATA GENERATION ───────────────────────────────────
# Templates for generating realistic student responses per misconception type.
# Each template produces varied student-like text responses.

SYNTHETIC_TEMPLATES = {
    "integer_sign_ops": [
        # Pattern: Sum of negatives -> positive
        {
            "misconception_id": "sign_sum_negatives",
            "generator": lambda: _gen_sign_sum_neg(),
        },
        # Pattern: neg * neg -> neg
        {
            "misconception_id": "sign_neg_times_neg",
            "generator": lambda: _gen_neg_times_neg(),
        },
        # Pattern: subtracting negative treated as subtraction
        {
            "misconception_id": "sign_sub_negative",
            "generator": lambda: _gen_sub_negative(),
        },
        # Pattern: always subtract smaller from larger
        {
            "misconception_id": "sign_always_subtract_smaller",
            "generator": lambda: _gen_subtract_smaller(),
        },
    ],
    "order_of_operations": [
        {
            "misconception_id": "oo_left_to_right",
            "generator": lambda: _gen_oo_left_right(),
        },
        {
            "misconception_id": "oo_exponent_after_add",
            "generator": lambda: _gen_oo_exponent(),
        },
        {
            "misconception_id": "oo_parentheses_ignored",
            "generator": lambda: _gen_oo_parens(),
        },
    ],
    "distributive_property": [
        {
            "misconception_id": "dist_first_term_only",
            "generator": lambda: _gen_dist_first_only(),
        },
        {
            "misconception_id": "dist_sign_error_negative",
            "generator": lambda: _gen_dist_sign_error(),
        },
        {
            "misconception_id": "dist_drop_parens",
            "generator": lambda: _gen_dist_drop_parens(),
        },
        {
            "misconception_id": "dist_square_over_addition",
            "generator": lambda: _gen_dist_square(),
        },
    ],
    "combining_like_terms": [
        {
            "misconception_id": "clt_combine_unlike",
            "generator": lambda: _gen_clt_unlike(),
        },
        {
            "misconception_id": "clt_multiply_variables",
            "generator": lambda: _gen_clt_multiply(),
        },
        {
            "misconception_id": "clt_constant_as_variable",
            "generator": lambda: _gen_clt_const(),
        },
        {
            "misconception_id": "clt_add_exponents",
            "generator": lambda: _gen_clt_exponents(),
        },
    ],
    "solving_linear_equations": [
        {
            "misconception_id": "leq_reverse_operation",
            "generator": lambda: _gen_leq_reverse(),
        },
        {
            "misconception_id": "leq_divide_wrong_direction",
            "generator": lambda: _gen_leq_divide_wrong(),
        },
        {
            "misconception_id": "leq_subtract_wrong_side",
            "generator": lambda: _gen_leq_wrong_side(),
        },
        {
            "misconception_id": "leq_move_without_sign_change",
            "generator": lambda: _gen_leq_no_sign_change(),
        },
    ],
}

# Student phrasing variations
PHRASING = {
    "math_only": lambda q, a: f"{a}",
    "short": lambda q, a: f"I got {a}",
    "with_work": lambda q, a: f"My answer is {a}. I worked it out step by step.",
    "uncertain": lambda q, a: f"I think the answer is {a} but I'm not sure",
    "confident": lambda q, a: f"The answer is {a}",
    "explain": lambda q, a: f"I solved it and got {a}. Here's what I did:",
}


def _pick_phrasing(question, answer):
    style = random.choice(list(PHRASING.keys()))
    return PHRASING[style](question, answer)


def _pick_var():
    return random.choice(["x", "y", "n", "m", "a", "b", "k", "t"])


def _pick_coeff(low=2, high=9):
    return random.randint(low, high)


# ─── GENERATORS: Integer/Sign Operations ─────────────────────────────────
def _gen_sign_sum_neg():
    a = random.randint(2, 12)
    b = random.randint(2, 12)
    question = f"Simplify: -{a} - {b}"
    wrong = str(a + b)  # student gets positive
    correct = str(-(a + b))
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_neg_times_neg():
    a = random.randint(2, 9)
    b = random.randint(2, 9)
    question = f"Simplify: (-{a}) × (-{b})"
    wrong = str(-(a * b))
    correct = str(a * b)
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_sub_negative():
    a = random.randint(2, 15)
    b = random.randint(2, 10)
    question = f"Simplify: {a} - (-{b})"
    wrong = str(a - b)
    correct = str(a + b)
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_subtract_smaller():
    a = random.randint(1, 8)
    b = random.randint(a + 2, 15)
    question = f"Simplify: {a} - {b}"
    wrong = str(b - a)  # positive, wrong
    correct = str(a - b)
    return question, _pick_phrasing(question, wrong), wrong, correct


# ─── GENERATORS: Order of Operations ─────────────────────────────────────
def _gen_oo_left_right():
    a = random.randint(1, 10)
    b = random.randint(2, 6)
    c = random.randint(2, 6)
    op = random.choice(["+", "-"])
    question = f"Evaluate: {a} {op} {b} × {c}"
    if op == "+":
        wrong = str((a + b) * c)
        correct = str(a + b * c)
    else:
        wrong = str((a - b) * c)
        correct = str(a - b * c)
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_oo_exponent():
    a = random.randint(1, 6)
    b = random.randint(2, 5)
    exp = random.choice([2, 3])
    question = f"Evaluate: {a} + {b}{'²' if exp == 2 else '³'}"
    wrong = str((a + b) ** exp)
    correct = str(a + b**exp)
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_oo_parens():
    a = random.randint(1, 8)
    b = random.randint(1, 8)
    c = random.randint(2, 6)
    question = f"Evaluate: ({a} + {b}) × {c}"
    wrong = str(a + b * c)  # ignores parens
    correct = str((a + b) * c)
    return question, _pick_phrasing(question, wrong), wrong, correct


# ─── GENERATORS: Distributive Property ───────────────────────────────────
def _gen_dist_first_only():
    coeff = _pick_coeff()
    v = _pick_var()
    const = random.randint(1, 10)
    question = f"Expand: {coeff}({v} + {const})"
    wrong = f"{coeff}{v} + {const}"  # only multiplied first term
    correct = f"{coeff}{v} + {coeff * const}"
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_dist_sign_error():
    coeff = _pick_coeff()
    v = _pick_var()
    const = random.randint(1, 10)
    question = f"Expand: -{coeff}({v} - {const})"
    wrong = f"-{coeff}{v} - {coeff * const}"  # didn't flip sign
    correct = f"-{coeff}{v} + {coeff * const}"
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_dist_drop_parens():
    coeff = _pick_coeff()
    v = _pick_var()
    const = random.randint(1, 10)
    question = f"Expand: {coeff}({v} + {const})"
    wrong = f"{coeff}{v} + {const}"
    correct = f"{coeff}{v} + {coeff * const}"
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_dist_square():
    v = _pick_var()
    const = random.randint(1, 6)
    question = f"Expand: ({v} + {const})²"
    wrong = f"{v}² + {const**2}"
    correct = f"{v}² + {2 * const}{v} + {const**2}"
    return question, _pick_phrasing(question, wrong), wrong, correct


# ─── GENERATORS: Combining Like Terms ────────────────────────────────────
def _gen_clt_unlike():
    a = _pick_coeff()
    b = _pick_coeff()
    v1, v2 = random.sample(["x", "y", "a", "b", "m", "n"], 2)
    question = f"Simplify: {a}{v1} + {b}{v2}"
    wrong = f"{a + b}{v1}{v2}"
    correct = f"{a}{v1} + {b}{v2}"
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_clt_multiply():
    a = _pick_coeff()
    b = _pick_coeff()
    v = _pick_var()
    question = f"Simplify: {a}{v} + {b}{v}"
    wrong = f"{a + b}{v}²"
    correct = f"{a + b}{v}"
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_clt_const():
    a = _pick_coeff()
    b = random.randint(1, 10)
    v = _pick_var()
    question = f"Simplify: {a}{v} + {b}"
    wrong = f"{a + b}{v}"
    correct = f"{a}{v} + {b}"
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_clt_exponents():
    a = _pick_coeff()
    b = _pick_coeff()
    v = _pick_var()
    exp = random.choice([2, 3])
    question = f"Simplify: {a}{v}{'²' if exp == 2 else '³'} + {b}{v}{'²' if exp == 2 else '³'}"
    wrong_exp = exp * 2
    wrong = f"{a + b}{v}{'⁴' if wrong_exp == 4 else '⁶'}"
    correct = f"{a + b}{v}{'²' if exp == 2 else '³'}"
    return question, _pick_phrasing(question, wrong), wrong, correct


# ─── GENERATORS: Solving Linear Equations ────────────────────────────────
def _gen_leq_reverse():
    a = random.randint(2, 12)
    b = random.randint(a + 1, 20)
    op = random.choice(["+", "-"])
    v = _pick_var()
    if op == "+":
        question = f"Solve: {v} + {a} = {b}"
        wrong = str(a + b)
        correct = str(b - a)
    else:
        question = f"Solve: {v} - {a} = {b}"
        wrong = str(b - a)
        correct = str(b + a)
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_leq_divide_wrong():
    coeff = _pick_coeff(2, 8)
    result = random.randint(2, 10)
    product = coeff * result
    v = _pick_var()
    question = f"Solve: {coeff}{v} = {product}"
    wrong = str(product * coeff)
    correct = str(result)
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_leq_wrong_side():
    coeff = _pick_coeff(2, 5)
    const = random.randint(1, 8)
    total = random.randint(const + coeff + 1, 30)
    v = _pick_var()
    question = f"Solve: {coeff}{v} + {const} = {total}"
    wrong_intermediate = total + const  # added instead of subtracted
    wrong = f"{v} = {wrong_intermediate // coeff}" if wrong_intermediate % coeff == 0 else f"{v} = {wrong_intermediate}/{coeff}"
    correct_intermediate = total - const
    correct = f"{v} = {correct_intermediate // coeff}" if correct_intermediate % coeff == 0 else f"{v} = {correct_intermediate}/{coeff}"
    return question, _pick_phrasing(question, wrong), wrong, correct


def _gen_leq_no_sign_change():
    a = random.randint(2, 10)
    b = random.randint(a + 1, 20)
    v = _pick_var()
    question = f"Solve: {v} - {a} = {b}"
    wrong = str(b - a)  # moved without sign change
    correct = str(b + a)
    return question, _pick_phrasing(question, wrong), wrong, correct


def generate_synthetic(n_per_misconception=25):
    """Generate n synthetic examples per misconception per concept."""
    synthetic = []
    for concept_id, templates in SYNTHETIC_TEMPLATES.items():
        for template in templates:
            for i in range(n_per_misconception):
                question, student_response, wrong, correct = template["generator"]()

                # Create a deterministic ID
                content_hash = hashlib.md5(
                    f"{concept_id}_{template['misconception_id']}_{i}_{question}".encode()
                ).hexdigest()[:8]

                synthetic.append(
                    {
                        "source": "synthetic",
                        "id": f"syn_{concept_id}_{template['misconception_id']}_{content_hash}",
                        "concept_id": concept_id,
                        "misconception_id": template["misconception_id"],
                        "question": question,
                        "student_response": student_response,
                        "incorrect_answer": wrong,
                        "correct_answer": correct,
                    }
                )

    print(f"\nGenerated {len(synthetic)} synthetic examples")
    concept_counts = Counter(item["concept_id"] for item in synthetic)
    misc_counts = Counter(item["misconception_id"] for item in synthetic)
    print("Per-concept:")
    for cid, count in sorted(concept_counts.items()):
        print(f"  {cid}: {count}")
    print("Per-misconception:")
    for mid, count in sorted(misc_counts.items()):
        print(f"  {mid}: {count}")

    return synthetic


# ─── STEP 4: MERGE AND FORMAT ────────────────────────────────────────────
def merge_datasets(mae_filtered, synthetic):
    """Merge MaE and synthetic data into unified format."""
    unified = []

    # Format MaE entries
    for item in mae_filtered:
        unified.append(
            {
                "source": "mae",
                "id": f"mae_{item['mae_id']}_{item['example_number']}",
                "concept_id": item["concept_id"],
                "misconception_id": None,  # MaE uses text descriptions, not our IDs
                "mae_id": item["mae_id"],
                "misconception_text": item["misconception"],
                "question": item["question"],
                "student_response": item["incorrect_answer"],
                "incorrect_answer": item["incorrect_answer"],
                "correct_answer": item["correct_answer"],
                "label": item["concept_id"],  # Classification target
            }
        )

    # Format synthetic entries
    for item in synthetic:
        unified.append(
            {
                "source": item["source"],
                "id": item["id"],
                "concept_id": item["concept_id"],
                "misconception_id": item["misconception_id"],
                "mae_id": None,
                "misconception_text": item["misconception_id"],
                "question": item["question"],
                "student_response": item["student_response"],
                "incorrect_answer": item["incorrect_answer"],
                "correct_answer": item["correct_answer"],
                "label": item["concept_id"],  # Classification target
            }
        )

    print(f"\nMerged dataset: {len(unified)} total examples")
    return unified


# ─── STEP 5: TRAIN/VAL/TEST SPLIT ───────────────────────────────────────
def stratified_split(data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Stratified split by concept_id ensuring no leakage."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Deduplicate by question+answer fingerprint before splitting
    seen = set()
    deduped = []
    for item in data:
        fp = f"{item['question']}|||{item['incorrect_answer']}"
        if fp not in seen:
            seen.add(fp)
            deduped.append(item)
    if len(deduped) < len(data):
        print(f"  Deduplicated: {len(data)} -> {len(deduped)} (removed {len(data) - len(deduped)} duplicates)")
    data = deduped

    # Group by concept
    by_concept = defaultdict(list)
    for item in data:
        by_concept[item["concept_id"]].append(item)

    train, val, test = [], [], []

    for concept_id, items in by_concept.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def check_leakage(train, val, test):
    """Verify no duplicate questions across splits."""
    def fingerprint(item):
        return f"{item['question']}|||{item['incorrect_answer']}"

    train_fps = set(fingerprint(x) for x in train)
    val_fps = set(fingerprint(x) for x in val)
    test_fps = set(fingerprint(x) for x in test)

    train_val = train_fps & val_fps
    train_test = train_fps & test_fps
    val_test = val_fps & test_fps

    if train_val or train_test or val_test:
        print(f"WARNING: Data leakage detected!")
        print(f"  Train-Val overlap: {len(train_val)}")
        print(f"  Train-Test overlap: {len(train_test)}")
        print(f"  Val-Test overlap: {len(val_test)}")
        return False
    else:
        print("No data leakage detected across splits.")
        return True


# ─── MAIN ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 3: Dataset Assembly Pipeline")
    print("=" * 60)

    # Step 1: Load MaE
    print("\n--- Step 1: Loading MaE data ---")
    raw = load_mae_data()

    # Step 2: Filter
    print("\n--- Step 2: Filtering to target concepts ---")
    mae_filtered = filter_to_concepts(raw)

    # Step 3: Generate synthetic
    print("\n--- Step 3: Generating synthetic data ---")
    synthetic = generate_synthetic(n_per_misconception=34)

    # Step 4: Merge
    print("\n--- Step 4: Merging datasets ---")
    unified = merge_datasets(mae_filtered, synthetic)

    # Step 5: Split
    print("\n--- Step 5: Creating stratified splits ---")
    train, val, test = stratified_split(unified)
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    # Check leakage
    print("\n--- Leakage Check ---")
    check_leakage(train, val, test)

    # Report final distribution
    print("\n--- Final Distribution ---")
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        counts = Counter(item["concept_id"] for item in split_data)
        print(f"\n{split_name}:")
        for cid, count in sorted(counts.items()):
            print(f"  {cid}: {count}")

    # Save
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = DATASET_DIR / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"\nSaved {path} ({len(split_data)} examples)")

    # Save full unified dataset
    with open(DATASET_DIR / "full.json", "w") as f:
        json.dump(unified, f, indent=2)

    # Summary stats
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    total = len(unified)
    mae_count = sum(1 for x in unified if x["source"] == "mae")
    syn_count = sum(1 for x in unified if x["source"] == "synthetic")
    print(f"Total examples: {total}")
    print(f"  MaE original: {mae_count}")
    print(f"  Synthetic:    {syn_count}")
    print(f"Concepts: {len(set(x['concept_id'] for x in unified))}")
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    concept_totals = Counter(x["concept_id"] for x in unified)
    print("\nExamples per concept:")
    for cid, count in sorted(concept_totals.items()):
        meets = "✓" if count >= 100 else "✗"
        print(f"  {meets} {cid}: {count}")


if __name__ == "__main__":
    main()
