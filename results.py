import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def main():
    res_path = "eval_results/result/end2end_quick_match_metric_result.json"
    if not os.path.exists(res_path):
        print("Results file not found. Please run eval first.")
        return

    with open(res_path, "r") as f:
        data = json.load(f)

    summary = []
    
    if "text_block" in data:
        s = data["text_block"]["all"]
        summary.append({"Category": "Text (Edit Dist)", "Score": 1 - s["Edit_dist"]["ALL_page_avg"]})
        summary.append({"Category": "Text (BLEU)", "Score": s["BLEU"]["all"]})
        
    if "table" in data:
        s = data["table"]["all"]
        summary.append({"Category": "Table (TEDS)", "Score": s["TEDS"]["all"]})
        summary.append({"Category": "Table (TEDS Struct)", "Score": s["TEDS_structure_only"]["all"]})
        
    if "display_formula" in data:
        s = data["display_formula"]["all"]
        summary.append({"Category": "Formula (Edit Dist)", "Score": 1 - s["Edit_dist"]["ALL_page_avg"]})
        
    if "reading_order" in data:
        s = data["reading_order"]["all"]
        summary.append({"Category": "Reading Order", "Score": 1 - s["Edit_dist"]["ALL_page_avg"]})

    df = pd.DataFrame(summary)
    print("\n=== OmniDocBench Evaluation Summary ===")
    print(df.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Category"], df["Score"], color='skyblue')
    plt.ylim(0, 1.1)
    plt.ylabel("Score (Higher is better)")
    plt.title("OmniDocBench Overall Performance")
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 3), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("eval_results/summary_plot.png")
    print(f"\nPlot saved to eval_results/summary_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
