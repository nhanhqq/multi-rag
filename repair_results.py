import json

def repair_and_show():
    try:
        with open("scifact_results.json", "r") as f:
            results = json.load(f)
        with open("scifact_data/claims_dev.jsonl", "r") as f:
            claims_gt = [json.loads(l) for l in f]
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    gt_map = {}
    for c in claims_gt:
        label = "NOT_ENOUGH_INFO"
        if c.get("evidence"):
            for doc_id in c["evidence"]:
                for ev in c["evidence"][doc_id]:
                    label = ev["label"] 
                    break
                break
        gt_map[c["id"]] = label

    correct = 0
    total = len(results)
    
    print("\n" + "="*80)
    print(f"{'ID':<6} | {'PREDICTED':<15} | {'GROUND TRUTH':<15} | {'RESULT'}")
    print("-" * 80)
    
    for r in results[:20]:
        gold = gt_map.get(r["id"], "UNKNOWN")
        r["gold"] = gold
        status = "✅ PASS" if r["predicted"] == gold else "❌ FAIL"
        if r["predicted"] == gold: correct += 1
        print(f"{r['id']:<6} | {r['predicted']:<15} | {gold:<15} | {status}")

    for r in results[20:]:
        gold = gt_map.get(r["id"], "UNKNOWN")
        r["gold"] = gold
        if r["predicted"] == gold: correct += 1

    print("-" * 80)
    print(f"OVERALL ACCURACY: {correct/total:.2%} ({correct}/{total})")
    print("="*80)
    print("\nFile 'scifact_results.json' đã được cập nhật nhãn chuẩn. Bạn có thể mở file này để xem chi tiết.")

    with open("scifact_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    repair_and_show()
