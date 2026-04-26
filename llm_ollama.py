import requests

OLLAMA_BASE = "http://localhost:46479"
DEFAULT_MODEL = "llama3"


class OllamaLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.base = OLLAMA_BASE

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def embed(self, text: str) -> list[float]:
        resp = requests.post(
            f"{self.base}/api/embed",
            json={"model": self.model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings") or data.get("embedding")
        if isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings
