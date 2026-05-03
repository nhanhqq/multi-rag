import json
import os
import pandas as pd
import argparse
from sklearn.metrics import classification_report, accuracy_score

def load_ground_truth():
    gt_path = "scifact_data/claims_dev.jsonl"
    gt_map = {}
    try:
        with open(gt_path, "r") as f:
            for line in f:
                c = json.loads(line)
                label = "NOT_ENOUGH_INFO"
                if c.get("evidence"):
                    for doc_id in c["evidence"]:
                        for ev in c["evidence"][doc_id]:
                            label = ev["label"]
                            break
                        break
                gt_map[c["id"]] = label
    except Exception as e:
        print(f"Error loading Ground Truth: {e}")
    return gt_map

def analyze_file(filepath, gt_map):
    try:
        with open(filepath, "r") as f:
            results = json.load(f)
    except Exception as e:
        return None

    y_true, y_pred = [], []
    for r in results:
        gold = gt_map.get(r["id"], "UNKNOWN")
        y_true.append(gold)
        y_pred.append(r["predicted"])
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {"acc": acc, "report": report, "total": len(results)}

def main():
    gt_map = load_ground_truth()
    output_dir = "scifact_output"
    
    if not os.path.exists(output_dir):
        print(f"No results found in {output_dir}")
        return

    files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    if not files:
        print("No .json result files found.")
        return

    comparison_data = []
    print("\n" + "="*80)
    print(f"{'Filename':<30} | {'Total':<6} | {'Accuracy':<10}")
    print("-" * 80)
    
    for f in sorted(files):
        path = os.path.join(output_dir, f)
        metrics = analyze_file(path, gt_map)
        if metrics:
            print(f"{f:<30} | {metrics['total']:<6} | {metrics['acc']:.2%}")
            comparison_data.append({"file": f, "acc": metrics["acc"], "metrics": metrics})

    if comparison_data:
        best_run = max(comparison_data, key=lambda x: x["acc"])
        print("\n" + "="*80)
        print(f"DETAILED REPORT FOR BEST RUN: {best_run['file']}")
        print("="*80)
        df_report = pd.DataFrame(best_run["metrics"]["report"]).transpose()
        print(df_report)
        print("="*80)

if __name__ == "__main__":
    main()
