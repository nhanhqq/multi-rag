from datasets import load_dataset
configs = ['expert', 'viQuAD', 'LegalRAG', 'BKAI_RAG']
for config in configs:
    try:
        ds = load_dataset("sailor2/Vietnamese_RAG", config, split="train")
        print(f"Dataset {config} size: {len(ds)}")
    except Exception as e:
        print(f"Error loading {config}: {e}")
