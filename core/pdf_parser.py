import os
import time
import json
import tempfile
from pathlib import Path
import requests
from PIL import Image

MINERU_BASE = "http://localhost:8000"

def parse_pdf(pdf_path: str, output_dir: str = "./output") -> str:
    pdf_path = os.path.abspath(pdf_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    temp_pdf = None
    try:
        ext = Path(pdf_path).suffix.lower()
        if ext in (".png", ".jpg", ".jpeg"):
            temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            image = Image.open(pdf_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(temp_pdf.name, "PDF")
            temp_pdf.close()
            actual_pdf_path = temp_pdf.name
        else:
            actual_pdf_path = pdf_path
        with open(actual_pdf_path, "rb") as f:
            resp = requests.post(
                f"{MINERU_BASE}/file_parse",
                files={"files": f},
                timeout=3600,
            )
        resp.raise_for_status()
        result = resp.json()
    finally:
        if temp_pdf and os.path.exists(temp_pdf.name):
            try:
                os.remove(temp_pdf.name)
            except:
                pass
    return _find_markdown(pdf_path, output_dir, result)

def _find_markdown(pdf_path: str, output_dir: str, response_data: dict) -> str:
    content = response_data.get("markdown")
    if not content:
        results = response_data.get("results", {})
        if isinstance(results, dict):
            for val in results.values():
                if isinstance(val, dict) and val.get("md_content"):
                    content = val["md_content"]
                    break
    if not content and response_data.get("result_url"):
        try:
            r = requests.get(response_data["result_url"], timeout=30)
            if r.status_code == 200:
                content = r.json().get("markdown") or r.text
        except: pass
        
    if content:
        tmp_md = os.path.join(output_dir, Path(pdf_path).stem + ".md")
        with open(tmp_md, "w", encoding="utf-8") as f:
            f.write(content)
        return tmp_md
        
    md_path = response_data.get("markdown_path") or response_data.get("md_path")
    if md_path and os.path.exists(md_path):
        return md_path
    stem = Path(pdf_path).stem
    for root, _, files in os.walk(output_dir):
        for fname in files:
            if fname.endswith(".md") and (stem in fname or "input" in fname):
                return os.path.join(root, fname)
    raise FileNotFoundError(f"Cannot find markdown output for {pdf_path}")

def parse_pdf_to_text(pdf_path: str, output_dir: str = "./output") -> str:
    md_path = parse_pdf(pdf_path, output_dir)
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()
