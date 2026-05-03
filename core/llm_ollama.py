import requests
import json
import re

class LLM:
    def __init__(self, model="llama3:8b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt, system_prompt=""):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        output = response.json().get("response", "")
        
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
        # Bỏ thay thế \n để giữ lại markdown và cấu trúc đoạn văn
        # output = output.replace('\n', ' ')
        output = "".join(c for c in output if c.isprintable() or c.isspace())
        return re.sub(r' +', ' ', output).strip()
