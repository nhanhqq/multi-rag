import requests
import json

class OllamaLLM:
    def __init__(self, model="gemma4:e4b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def embed(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "prompt": text
        }
        resp = requests.post(f"{self.base_url}/api/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
