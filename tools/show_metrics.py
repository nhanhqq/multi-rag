import json
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

def main():
    try:
        with open("scifact_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: scifact_results.json not found.")
        return

    df = pd.DataFrame(results)
    
    if "gold_label" in df.columns:
        y_true = df["gold_label"]
        y_pred = df["predicted_label"]
    else:
        y_true = df["gold"]
        y_pred = df["predicted"]

    print("\n" + "="*50)
    print("      SCIFACT EVALUATION RESULTS")
    print("="*50)
    print(f"Total Claims: {len(df)}")
    print(f"Accuracy Score: {accuracy_score(y_true, y_pred):.4f}")
    print("-" * 50)
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred))
    print("="*50)

    print("\nSample Logic Chain (First Claim):")
    sample = results[0]
    print(f"Claim: {sample.get('claim', 'N/A')}")
    print(f"Predicted: {sample.get('predicted_label', sample.get('predicted', 'N/A'))} | Gold: {sample.get('gold_label', sample.get('gold', 'N/A'))}")
    if "final_response" in sample:
        print("-" * 30)
        print(f"Final Agent Response:\n{sample['final_response'][:500]}...")

if __name__ == "__main__":
    main()
