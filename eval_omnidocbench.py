"""
eval_omnidocbench.py

Run end-to-end evaluation of the MinerU PDF parser against the OmniDocBench dataset.

Usage:
    python eval_omnidocbench.py \
        --gt_json ./OmniDocBench/demo_data/omnidocbench_demo/OmniDocBench_demo.json \
        --pred_dir ./eval_predictions \
        --pdf_dir ./eval_pdfs \
        --output_dir ./eval_results \
        [--workers 4] \
        [--skip_parse]

Steps:
  1. For each page in gt_json, call MinerU to parse its PDF and save the markdown to pred_dir.
  2. Call the OmniDocBench evaluation pipeline on (gt_json, pred_dir).
  3. Print and save metrics: Edit Distance, TEDS, BLEU, METEOR, CDM (if available).
"""

import argparse
import json
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

MINERU_BASE = "http://localhost:8000"
OMNIDOCBENCH_DIR = Path(__file__).parent / "OmniDocBench"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_json", default=str(OMNIDOCBENCH_DIR / "demo_data/omnidocbench_demo/OmniDocBench_demo.json"))
    p.add_argument("--pred_dir", default="./eval_predictions")
    p.add_argument("--pdf_dir", default="./eval_pdfs", help="Folder containing PDF files to parse")
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip_parse", action="store_true", help="Skip parsing step, use existing pred_dir")
    p.add_argument("--config", default=None, help="Custom OmniDocBench config YAML path")
    return p.parse_args()


def mineru_parse_pdf(pdf_path: str, output_dir: str) -> str | None:
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{MINERU_BASE}/file_parse",
                files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
                data={"parse_method": "auto", "is_json_md_dump": "true", "output_dir": output_dir},
                timeout=300,
            )
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"  [ERROR] MinerU request failed for {pdf_path}: {e}")
        return None

    task_id = result.get("task_id") or result.get("id")
    if task_id:
        return poll_task(task_id, pdf_path, output_dir)

    return find_markdown(pdf_path, output_dir, result)


def poll_task(task_id: str, pdf_path: str, output_dir: str, timeout: int = 300) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{MINERU_BASE}/task/{task_id}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            time.sleep(3)
            continue
        status = data.get("status", "")
        if status in ("done", "success", "completed"):
            return find_markdown(pdf_path, output_dir, data)
        if status in ("failed", "error"):
            print(f"  [ERROR] MinerU task {task_id} failed")
            return None
        time.sleep(3)
    print(f"  [TIMEOUT] MinerU task {task_id}")
    return None


def find_markdown(pdf_path: str, output_dir: str, response_data: dict) -> str | None:
    md_path = response_data.get("markdown_path") or response_data.get("md_path")
    if md_path and os.path.exists(md_path):
        return md_path
    stem = Path(pdf_path).stem
    for root, _, files in os.walk(output_dir):
        for fname in files:
            if fname.endswith(".md") and stem in fname:
                return os.path.join(root, fname)
    return None


def build_pred_dir(gt_json_path: str, pdf_dir: str, pred_dir: str, workers: int):
    os.makedirs(pred_dir, exist_ok=True)

    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    pages = gt_data if isinstance(gt_data, list) else gt_data.get("pages", [])

    image_names = []
    for page in pages:
        info = page.get("page_info", {})
        img_path = info.get("image_path", "")
        if img_path:
            image_names.append(Path(img_path).stem)

    image_names = list(dict.fromkeys(image_names))
    print(f"[Eval] Found {len(image_names)} unique pages in GT.")

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    print(f"[Eval] Found {len(pdf_files)} PDFs in {pdf_dir}")

    mineru_tmp = "./mineru_output_tmp"
    os.makedirs(mineru_tmp, exist_ok=True)

    def process_pdf(pdf_path):
        md_path = mineru_parse_pdf(str(pdf_path), mineru_tmp)
        if md_path is None:
            print(f"  [SKIP] No markdown output for {pdf_path.name}")
            return
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        dest = os.path.join(pred_dir, pdf_path.stem + ".md")
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [OK] {pdf_path.name} -> {dest}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_pdf, p): p for p in pdf_files}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"  [ERROR] {futures[fut].name}: {exc}")


def run_omnidocbench_eval(gt_json: str, pred_dir: str, output_dir: str, config_path: str | None):
    os.makedirs(output_dir, exist_ok=True)

    if config_path is None:
        config_path = str(OMNIDOCBENCH_DIR / "configs/end2end.yaml")

    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["end2end_eval"]["dataset"]["ground_truth"]["data_path"] = os.path.abspath(gt_json)
    cfg["end2end_eval"]["dataset"]["prediction"]["data_path"] = os.path.abspath(pred_dir)

    cpu_count = os.cpu_count() or 4
    workers = max(1, cpu_count // 3)
    cfg["end2end_eval"]["dataset"]["match_workers"] = workers
    if "display_formula" in cfg["end2end_eval"]["metrics"]:
        cfg["end2end_eval"]["metrics"]["display_formula"]["cdm_workers"] = workers
    if "table" in cfg["end2end_eval"]["metrics"]:
        cfg["end2end_eval"]["metrics"]["table"]["teds_workers"] = workers

    tmp_config = os.path.join(output_dir, "eval_config.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)

    print(f"\n[Eval] Running OmniDocBench evaluation...")
    print(f"  GT   : {gt_json}")
    print(f"  PRED : {pred_dir}")
    print(f"  CFG  : {tmp_config}")

    result = subprocess.run(
        [sys.executable, str(OMNIDOCBENCH_DIR / "pdf_validation.py"), "--config", tmp_config],
        cwd=str(OMNIDOCBENCH_DIR),
        capture_output=True,
        text=True,
    )

    print("\n--- STDOUT ---")
    print(result.stdout)
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)

    log_path = os.path.join(output_dir, "eval_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    result_src = OMNIDOCBENCH_DIR / "result"
    result_dst = Path(output_dir) / "result"
    if result_src.exists():
        if result_dst.exists():
            shutil.rmtree(result_dst)
        shutil.copytree(result_src, result_dst)
        print(f"[Eval] Results copied to {result_dst}")

    print(f"[Eval] Log saved to {log_path}")


def main():
    args = parse_args()

    if not args.skip_parse:
        print(f"[Eval] Step 1: Parsing PDFs with MinerU...")
        build_pred_dir(args.gt_json, args.pdf_dir, args.pred_dir, args.workers)
    else:
        print(f"[Eval] Skipping parse step, using existing predictions in {args.pred_dir}")

    print(f"\n[Eval] Step 2: Running OmniDocBench metrics...")
    run_omnidocbench_eval(args.gt_json, args.pred_dir, args.output_dir, args.config)


if __name__ == "__main__":
    main()
