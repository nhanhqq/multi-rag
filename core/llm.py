from .llm_ollama import OllamaLLM

class LLM:
    def __init__(self, key=None):
        self.engine = OllamaLLM()
    def generate(self, prompt):
        return self.engine.generate(prompt)
