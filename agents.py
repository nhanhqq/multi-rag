import os
from llm_ollama import LLM

def read_prompt(filename):
    path = os.path.join("system_promt", filename)
    if not os.path.exists(path):
        # Fallback if in different folder structure
        path = os.path.join(os.path.dirname(__file__), "system_promt", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class RagAgent:
    def __init__(self, model="llama3:8b"):
        self.llm = LLM(model=model)
        # Check if rag_vn.txt exists, otherwise use rag.txt
        if os.path.exists("system_promt/rag_vn.txt"):
            self.tpl = read_prompt("rag_vn.txt")
        else:
            self.tpl = read_prompt("rag.txt")
        
    def draft(self, q, s, c):
        return self.llm.generate(self.tpl.format(q, s, c))

class Agent1:
    def __init__(self, model="llama3:8b"):
        self.llm = LLM(model=model)
        self.tpl = read_prompt("agent1.txt")

    def evaluate(self, q, c, draft):
        return self.llm.generate(self.tpl.format(q, c, draft))

class Agent2:
    def __init__(self, model="llama3:8b"):
        self.llm = LLM(model=model)
        self.tpl = read_prompt("agent2.txt")

    def evaluate(self, q, s, draft):
        return self.llm.generate(self.tpl.format(q, s, draft))

class FusionAgent:
    def __init__(self, model="llama3:8b"):
        self.llm = LLM(model=model)
        self.tpl = read_prompt("agent3.txt")

    def fuse(self, q, d, e1, e2, s, c):
        return self.llm.generate(self.tpl.format(q, d, e1, e2, s, c))
