import json
import re
import time
from tabulate import tabulate
from collections import Counter
from qwen import QWEN

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
def call_qwen(question: str):
    print(f"\n[DEBUG] Sending question to QWEN: {question}")
    llm = QWEN()
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
    print("Prompt sent to model:")
    print(prompt)

    

    try:
        raw = llm._call(prompt)
        print("Raw model output:")
        print(raw)

        cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        print("Cleaned model output (after stripping markdown):")
        print(cleaned)
        parsed = json.loads(cleaned)
        print("Parsed JSON:")
        print(parsed)
    except Exception as e:
        print("JSON parsing failed:", e)
        parsed = {"action": "unsupported", "parameters": {}}
    print("--- call_qwen END ---\n")
    return parsed

# -------------------------
# Evaluation function
# -------------------------
def evaluate(model_func, dataset):
    print("\n=== Starting evaluation ===")
    intent_correct = 0
    slot_precision, slot_recall, slot_f1 = [], [], []
    exact_matches = 0
    per_sample = []

    for idx, item in enumerate(dataset):
        print(f"\n--- Evaluating sample {idx+1} ---")
        gold_action = item["intent"]
        gold_params = item["params"]
        print(f"Gold action: {gold_action}")
        print(f"Gold params: {gold_params}")

        pred = model_func(item["query"])
        pred_action = pred.get("action", "")
        pred_params = pred.get("parameters", {})
        print(f"Predicted action: {pred_action}")
        print(f"Predicted params: {pred_params}")

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

        print(f"Slot metrics - TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision}, Recall: {recall}, F1: {f1}")

        slot_precision.append(precision)
        slot_recall.append(recall)
        slot_f1.append(f1)

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
    print("\n=== Evaluation metrics computed ===")
    print(metrics)
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
    models = {"QWEN": call_qwen}

    for model_name, fn in models.items():
        print(f"\n=== Running evaluation for {model_name} ===")
        start = time.time()
        metrics, per_sample = evaluate(fn, dataset)
        latency = (time.time() - start) * 1000
        print(f"Evaluation completed in {latency:.2f} ms")

        results.append([
            model_name,
            f"{metrics['Intent Accuracy']*100:.1f}%",
            f"{metrics['Slot F1']*100:.1f}%",
            f"{metrics['Exact Match']*100:.1f}%",
            f"{latency:.2f} ms"
        ])
        all_per_sample[model_name] = per_sample

    print("\n=== Summary Metrics ===")
    print(tabulate(results, headers=["Model", "Intent Acc", "Slot F1", "Exact Match", "Eval Time"]))

    for model_name, per_sample in all_per_sample.items():
        print(f"\n=== Detailed Results for {model_name} ===")
        for s in per_sample:
            print(f"Q: {s['query']}")
            print(f"  Gold Intent: {s['gold_intent']} | Pred: {s['pred_intent']} | Correct: {s['intent_correct']}")
            print(f"  Gold Params: {s['gold_params']}")
            print(f"  Pred Params: {s['pred_params']} | Exact Match: {s['params_exact']}")
            print("-" * 60)

        confusion = intent_confusion(per_sample)
        print("\nIntent Confusion (gold → pred):")
        for (gold, pred), count in confusion.items():
            print(f"  {gold} → {pred}: {count}")

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
