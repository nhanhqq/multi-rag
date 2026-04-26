import argparse
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import requests

MINERU_BASE = "http://localhost:8000"
OMNIDOCBENCH_DIR = Path(__file__).parent / "OmniDocBench"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_json", default=str(OMNIDOCBENCH_DIR / "OmniDocBench.json"))
    p.add_argument("--pred_dir", default="./eval_predictions")
    p.add_argument("--pdf_dir", default="./OmniDocBench_data/pdfs")
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip_parse", action="store_true")
    p.add_argument("--config", default=None)
    return p.parse_args()

def mineru_parse_pdf(pdf_path: str) -> str | None:
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(f"{MINERU_BASE}/file_parse", files={"files": f}, timeout=3600)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("markdown")
        if not content:
            results = data.get("results", {})
            if isinstance(results, dict):
                for val in results.values():
                    if isinstance(val, dict) and val.get("md_content"):
                        content = val["md_content"]
                        break
        if not content and data.get("result_url"):
            r = requests.get(data["result_url"], timeout=30)
            if r.status_code == 200:
                content = r.json().get("markdown") or r.text
        return content
    except:
        return None

def build_pred_dir(pdf_dir: str, pred_dir: str):
    os.makedirs(pred_dir, exist_ok=True)
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        dest = os.path.join(pred_dir, pdf_path.stem + ".md")
        if os.path.exists(dest):
            continue
        content = mineru_parse_pdf(str(pdf_path))
        if content:
            with open(dest, "w", encoding="utf-8") as f:
                f.write(content)

def run_omnidocbench_eval(gt_json: str, pred_dir: str, output_dir: str, config_path: str | None):
    os.makedirs(output_dir, exist_ok=True)
    if config_path is None:
        config_path = str(OMNIDOCBENCH_DIR / "configs/end2end.yaml")
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = os.path.abspath(gt_json)
    cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] = os.path.abspath(pred_dir)
    workers = max(1, (os.cpu_count() or 4) // 3)
    cfg["end2end_eval"]["dataset"]["match_workers"] = workers
    if "display_formula" in cfg["end2end_eval"]["metrics"]:
        cfg["end2end_eval"]["metrics"]["display_formula"]["cdm_workers"] = workers
    if "table" in cfg["end2end_eval"]["metrics"]:
        cfg["end2end_eval"]["metrics"]["table"]["teds_workers"] = workers
    tmp_config = os.path.abspath(os.path.join(output_dir, "eval_config.yaml"))
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)
    result = subprocess.run(
        [sys.executable, str(OMNIDOCBENCH_DIR / "pdf_validation.py"), "--config", tmp_config],
        cwd=str(OMNIDOCBENCH_DIR),
        capture_output=True,
        text=True,
    )
    log_path = os.path.join(output_dir, "eval_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr: f.write(result.stderr)
    res_src = OMNIDOCBENCH_DIR / "result"
    res_dst = Path(output_dir) / "result"
    if res_src.exists():
        if res_dst.exists(): shutil.rmtree(res_dst)
        shutil.copytree(res_src, res_dst)

def main():
    args = parse_args()
    if not args.skip_parse:
        build_pred_dir(args.pdf_dir, args.pred_dir)
    run_omnidocbench_eval(args.gt_json, args.pred_dir, args.output_dir, args.config)

if __name__ == "__main__":
    main()
