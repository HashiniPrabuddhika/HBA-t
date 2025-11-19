import json
import re
import time
from tabulate import tabulate
from collections import Counter
from zai import ZAILLM

# -------------------------
# Test dataset (ground truth)
# -------------------------
dataset = [
    {
        "query": "Book Hall A on Sept 5th from 10 am to 2 pm",
        "intent": "add_booking",
        "params": {"room_name": "Hall A", "date": "2025-09-05", "start_time": "10:00", "end_time": "14:00"}
    },
    {
        "query": "Cancel my booking Hall A on Sept 5th from 10 am to 2 pm",
        "intent": "cancel_booking",
        "params": {"room_name": "Hall A", "date": "2025-09-05", "start_time": "10:00", "end_time": "14:00"}
    },
    # {
    #     "query": "Show availability of the conference room next Monday morning",
    #     "intent": "check_availability",
    #     "params": {"room_name": "conference", "date": "2025-08-18", "start_time": "08:00", "end_time": "12:00"}
    # },
]

# -------------------------
# Model call functions
# -------------------------
def call_zai(question: str):
    print(f"\n[DEBUG] Sending question to ZAILLM: {question}")
    llm = ZAILLM()
    prompt = f"""
You are an intelligent assistant that helps manage room bookings.

From the following user request:
\"{question}\"

Extract the **action** and its corresponding **parameters** in **strict JSON format**.

Supported actions:
- "check_availability"
- "add_booking"
- "cancel_booking"
- "alternatives"

Required JSON structure:
{{
  "action": "check_availability" | "add_booking" | "cancel_booking" | "alternatives",
  "parameters": {{"room_name": "...", "date": "yyyy-mm-dd", "start_time": "HH:MM", "end_time": "HH:MM"}}
}}
Respond in **only JSON format**, without explanations.
"""
    raw = llm._call(prompt)
    print(f"[DEBUG] Raw response:\n{raw}")

    cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    print(f"[DEBUG] Cleaned response:\n{cleaned}")

    try:
        parsed = json.loads(cleaned)
        print(f"[DEBUG] Parsed JSON:\n{parsed}")
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        parsed = {"action": "unsupported", "parameters": {}}

    return parsed

# -------------------------
# Evaluation function
# -------------------------
def evaluate(model_func, dataset):
    intent_correct = 0
    slot_precision, slot_recall, slot_f1 = [], [], []
    exact_matches = 0
    per_sample = []

    for item in dataset:
        print(f"\n[DEBUG] Evaluating query: {item['query']}")
        gold_action = item["intent"]
        gold_params = item["params"]

        pred = model_func(item["query"])
        pred_action = pred.get("action", "")
        pred_params = pred.get("parameters", {})

        print(f"[DEBUG] Predicted action: {pred_action}")
        print(f"[DEBUG] Predicted params: {pred_params}")

        # Intent accuracy
        if pred_action == gold_action:
            intent_correct += 1

        # Slot evaluation
        gold_keys = set(gold_params.keys())
        pred_keys = set(pred_params.keys())

        tp = len(gold_keys & pred_keys)
        fp = len(pred_keys - gold_keys)
        fn = len(gold_keys - pred_keys)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        slot_precision.append(precision)
        slot_recall.append(recall)
        slot_f1.append(f1)

        # Exact match
        if pred_params == gold_params:
            exact_matches += 1

        per_sample.append({
            "query": item["query"],
            "gold_intent": gold_action,
            "pred_intent": pred_action,
            "intent_correct": pred_action == gold_action,
            "gold_params": gold_params,
            "pred_params": pred_params,
            "params_exact": pred_params == gold_params
        })

    metrics = {
        "Intent Accuracy": intent_correct / len(dataset),
        "Slot Precision": sum(slot_precision) / len(dataset),
        "Slot Recall": sum(slot_recall) / len(dataset),
        "Slot F1": sum(slot_f1) / len(dataset),
        "Exact Match": exact_matches / len(dataset),
    }
    print(f"\n[DEBUG] Metrics: {metrics}")
    return metrics, per_sample

# -------------------------
# Analysis helpers
# -------------------------
def intent_confusion(per_sample):
    pairs = [(s["gold_intent"], s["pred_intent"]) for s in per_sample]
    return Counter(pairs)

def analyze_slot_errors(per_sample):
    issues = []
    for s in per_sample:
        if not s["params_exact"]:
            gold_keys = set(s["gold_params"].keys())
            pred_keys = set(s["pred_params"].keys())
            missing = gold_keys - pred_keys
            extra = pred_keys - gold_keys
            issues.append({
                "query": s["query"],
                "missing": list(missing),
                "extra": list(extra),
                "gold_params": s["gold_params"],
                "pred_params": s["pred_params"]
            })
    return issues

# -------------------------
# Run and show results
# -------------------------
if __name__ == "__main__":
    results = []
    all_per_sample = {}

    models = {
        "DeepSeek": call_zai,
    }

    for model_name, fn in models.items():
        print(f"\n[DEBUG] Running evaluation for model: {model_name}")
        start = time.time()
        metrics, per_sample = evaluate(fn, dataset)
        latency = (time.time() - start) * 1000

        results.append([
            model_name,
            f"{metrics['Intent Accuracy']*100:.1f}%",
            f"{metrics['Slot F1']*100:.1f}%",
            f"{metrics['Exact Match']*100:.1f}%",
            f"{latency:.2f} ms"
        ])
        all_per_sample[model_name] = per_sample

    # 📊 Summary table
    print("\n=== Summary Metrics ===")
    print(tabulate(results, headers=["Model", "Intent Acc", "Slot F1", "Exact Match", "Eval Time"]))

    # 🕵️ Detailed per-sample results
    for model_name, per_sample in all_per_sample.items():
        print(f"\n=== Detailed Results for {model_name} ===")
        for s in per_sample:
            print(f"Q: {s['query']}")
            print(f"  Gold Intent: {s['gold_intent']} | Pred: {s['pred_intent']} | Correct: {s['intent_correct']}")
            print(f"  Gold Params: {s['gold_params']}")
            print(f"  Pred Params: {s['pred_params']} | Exact Match: {s['params_exact']}")
            print("-" * 60)

        # 🔎 Confusion matrix
        confusion = intent_confusion(per_sample)
        print("\nIntent Confusion (gold → pred):")
        for (gold, pred), count in confusion.items():
            print(f"  {gold} → {pred}: {count}")

        # ⚡ Slot error analysis
        slot_issues = analyze_slot_errors(per_sample)
        if slot_issues:
            print("\nSlot-level Issues:")
            for issue in slot_issues:
                print(f"Q: {issue['query']}")
                print(f"  Missing: {issue['missing']}")
                print(f"  Extra: {issue['extra']}")
                print(f"  Gold Params: {issue['gold_params']}")
                print(f"  Pred Params: {issue['pred_params']}")
                print("-" * 60)
