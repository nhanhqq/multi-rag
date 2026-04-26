import os
from llm import LLM
from dotenv import load_dotenv
load_dotenv()
def read_prompt(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "system_promt", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
class RagAgent:
    def __init__(self):
        self.llm = LLM(os.getenv("GROQ_API_KEY1"))
        self.tpl = read_prompt("rag.txt")
        
    def draft(self, q, s, c):
        return self.llm.generate(self.tpl.format(q, s, c))
class Agent1:
    def __init__(self):
        self.llm = LLM(os.getenv("GROQ_API_KEY2"))
        self.tpl = read_prompt("agent1.txt")
    def evaluate(self, q, c, draft):
        return self.llm.generate(self.tpl.format(q, c, draft))
class Agent2:
    def __init__(self):
        self.llm = LLM(os.getenv("GROQ_API_KEY3"))
        self.tpl = read_prompt("agent2.txt")
    def evaluate(self, q, s, draft):
        return self.llm.generate(self.tpl.format(q, s, draft))
class FusionAgent:
    def __init__(self):
        self.llm = LLM(os.getenv("GROQ_API_KEY4"))
        self.tpl = read_prompt("agent3.txt")
    def fuse(self, q, d, e1, e2, s, c):
        return self.llm.generate(self.tpl.format(q, d, e1, e2, s, c))
