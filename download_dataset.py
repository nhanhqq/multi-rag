import os
from huggingface_hub import snapshot_download

def download():
    repo_id = "opendatalab/OmniDocBench"
    local_dir = "OmniDocBench_data"
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded to {local_dir}")

if __name__ == "__main__":
    download()
