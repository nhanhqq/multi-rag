import os
import requests
import glob
from fastapi import FastAPI
import uvicorn

app = FastAPI()

PDF_FOLDER = "./pdf"
REMOTE_API_URL = " https://c93e-109-237-69-254.ngrok-free.app"

@app.get("/process")
def process_pdfs():
    if not os.path.exists(PDF_FOLDER):
        return {"error": "Folder not found"}
    
    output_folder = "./output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    import glob
    import subprocess
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    
    results = []
    for pdf_path in pdf_files:
        try:
            cmd = ["mineru", "-p", pdf_path, "-o", output_folder, "--api-url", REMOTE_API_URL]
            process = subprocess.run(cmd, capture_output=True, text=True)
            results.append({"file": os.path.basename(pdf_path), "stdout": process.stdout, "stderr": process.stderr})
        except Exception as e:
            results.append({"file": os.path.basename(pdf_path), "error": str(e)})

    return {"status": "done", "results": results}

@app.get("/")
def health():
    return {"status": "online"}

if __name__ == "__main__":
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
    uvicorn.run(app, host="0.0.0.0", port=9999)
