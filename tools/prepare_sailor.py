import os
import hashlib
from datasets import load_dataset
from tqdm import tqdm

def prepare_sailor_data(output_dir="data/sailor_contexts"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Loading dataset sailor2/Vietnamese_RAG (BKAI_RAG)...")
    ds = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG", split="train")
    
    unique_contexts = set()
    for item in tqdm(ds, desc="Extracting contexts"):
        if isinstance(item['context'], list):
            for ctx in item['context']:
                unique_contexts.add(ctx)
        else:
            unique_contexts.add(item['context'])
    
    print(f"Found {len(unique_contexts)} unique contexts. Saving as .md files...")
    
    for ctx in tqdm(unique_contexts, desc="Saving files"):
        ctx_hash = hashlib.md5(ctx.encode()).hexdigest()
        file_path = os.path.join(output_dir, f"{ctx_hash}.md")
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(ctx)

if __name__ == "__main__":
    prepare_sailor_data()
