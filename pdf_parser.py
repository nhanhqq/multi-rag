import os
import requests
import time
import json
from pathlib import Path

MINERU_BASE = "http://localhost:8000"


def parse_pdf(pdf_path: str, output_dir: str = "./output") -> str:
    pdf_path = os.path.abspath(pdf_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{MINERU_BASE}/file_parse",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
            data={"parse_method": "auto", "is_json_md_dump": "true", "output_dir": output_dir},
            timeout=300,
        )
    resp.raise_for_status()
    result = resp.json()

    task_id = result.get("task_id") or result.get("id")
    if task_id:
        return _poll_task(task_id, pdf_path, output_dir)

    return _find_markdown(pdf_path, output_dir, result)


def _poll_task(task_id: str, pdf_path: str, output_dir: str, timeout: int = 300) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{MINERU_BASE}/task/{task_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status in ("done", "success", "completed"):
            return _find_markdown(pdf_path, output_dir, data)
        if status in ("failed", "error"):
            raise RuntimeError(f"MinerU task {task_id} failed: {data}")
        time.sleep(3)
    raise TimeoutError(f"MinerU task {task_id} timed out after {timeout}s")


def _find_markdown(pdf_path: str, output_dir: str, response_data: dict) -> str:
    md_path = response_data.get("markdown_path") or response_data.get("md_path")
    if md_path and os.path.exists(md_path):
        return md_path

    stem = Path(pdf_path).stem
    for root, _, files in os.walk(output_dir):
        for fname in files:
            if fname.endswith(".md") and stem in fname:
                return os.path.join(root, fname)

    raise FileNotFoundError(f"Cannot find markdown output for {pdf_path}")


def parse_pdf_to_text(pdf_path: str, output_dir: str = "./output") -> str:
    md_path = parse_pdf(pdf_path, output_dir)
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()
