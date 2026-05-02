import os
from datetime import datetime
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Cập nhật thư viện theo chuẩn mới để tránh DeprecationWarning
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    start_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
    output_dir = f"eval_run_{start_time}"
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = "eval_results_sailor_inference.json"
    print(f"Loading data from {input_file}...")
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"File {input_file} not found. Please check again.")
        return

    df = pd.DataFrame(results)
    
    # Đảm bảo có cột response cho Ragas
    if "answer" in df.columns:
        df["response"] = df["answer"]
        
    ragas_ds = Dataset.from_pandas(df)
    
    print("Initializing Model for Ragas...")
    # Bạn có thể đổi model ở đây. 
    # Đặt temperature=0 và định dạng json (nếu hỗ trợ) để tránh lỗi parse.
    # Lựa chọn khác: model="deepseek-v2:16b"
    eval_llm = ChatOllama(
        model="llama3:8b", 
        base_url="http://localhost:11434", 
        temperature=0,
        format="json"
    )
    
    eval_embeddings = HuggingFaceEmbeddings(model_name="allenai/specter2_base")
    
    print("Starting Ragas Evaluation...")
    try:
        result = evaluate(
            ragas_ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False # Tiếp tục chạy nếu có 1 vài câu bị lỗi parse
        )
        
        print("\nEvaluation Results:")
        print(result)
        
        output_csv = os.path.join(output_dir, "eval_results_sailor_final.csv")
        result.to_pandas().to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Final results saved to {output_csv}")
        
    except Exception as e:
        print(f"\nEvaluation encountered a critical error: {e}")

if __name__ == "__main__":
    main()
