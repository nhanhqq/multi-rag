import os
import sys

def download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    repo_id = "opendatalab/OmniDocBench"
    local_dir = "./OmniDocBench_data"
    
    print(f"Downloading dataset {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete.")

if __name__ == "__main__":
    download()
