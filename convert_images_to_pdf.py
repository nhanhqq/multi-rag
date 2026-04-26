import os
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def convert_image_to_pdf(img_path, pdf_path):
    try:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(pdf_path, "PDF")
        return True
    except Exception as e:
        print(f"Error {img_path}: {e}")
        return False

def main():
    src_dir = "OmniDocBench_data"
    dst_dir = "OmniDocBench_data/pdfs"
    os.makedirs(dst_dir, exist_ok=True)
    extensions = ("*.jpg", "*.jpeg", "*.png")
    img_files = []
    for ext in extensions:
        img_files.extend(list(Path(src_dir).glob(ext)))
    def task(img_path):
        pdf_name = img_path.stem + ".pdf"
        pdf_path = os.path.join(dst_dir, pdf_name)
        if os.path.exists(pdf_path): return
        convert_image_to_pdf(img_path, pdf_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(task, img_files)

if __name__ == "__main__":
    main()
