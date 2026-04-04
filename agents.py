import os
from llm import LLM
from dotenv import load_dotenv
load_dotenv()
class RagAgent:
    def __init__(self):
        self.llm = LLM(os.getenv("GROQ_API_KEY1"))
        
    def draft(self, q, s, c):
        p = f"Answer query based on summary and chunks.\nQuery: {q}\nSummary: {s}\nContext: {c}\nAnswer:"
        return self.llm.generate(p)

class Agent1:
    def __init__(self):
        self.llm = LLM(os.getenv("GROQ_API_KEY2"))

    def evaluate(self, q, draft):
        p = f"Verify if draft matches the query facts.\nQuery: {q}\nDraft: {draft}\nVerdict:"
        return self.llm.generate(p)

class Agent2:
    def __init__(self):
        self.llm = LLM(os.getenv("GROQ_API_KEY3"))

    def evaluate(self, q, draft):
        p = f"Check if the draft tone is professional.\nQuery: {q}\nDraft: {draft}\nVerdict:"
        return self.llm.generate(p)

class FusionAgent:
    def __init__(self):
        self.llm = LLM(os.getenv("GROQ_API_KEY4"))

    def fuse(self, q, e1, e2):
        p = f"Refine answer using fact and tone check.\nQuery: {q}\nFact: {e1}\nTone: {e2}\nFinal:"
        return self.llm.generate(p)
