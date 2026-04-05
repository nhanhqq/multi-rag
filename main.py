from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import os
import re
from agents import RagAgent, Agent1, Agent2, FusionAgent
from loader import load_domain_data
from RAG import Retriever

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = Retriever()
rag_agent = RagAgent()
agent1 = Agent1()
agent2 = Agent2()
fusion_agent = FusionAgent()

HISTORY_FILE = "chat_history.json"

@app.on_event("startup")
def startup_event():
    chunks = load_domain_data("./pdf")
    if not rag.load():
        rag.sync(chunks)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

@app.post("/api/clear")
def clear_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
    return {"status": "ok"}

@app.get("/api/health")
def health_check():
    return {"status": 1}

class ChatRequest(BaseModel):
    query: str

def get_recent_history_context(max_tokens=512):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except:
        history = []
    
    context = ""
    current_tokens = 0
    
    for entry in reversed(history):
        user_text = entry.get("user", "")
        bot_text = entry.get("bot", "")
        turn_text = f"User: {user_text}\nBot: {bot_text}\n"
        turn_tokens = len(turn_text.split()) * 1.3
        if current_tokens + turn_tokens > max_tokens:
            break
        context = turn_text + context
        current_tokens += turn_tokens
        
    return context

def save_history(user_q, bot_ans):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except:
        history = []
        
    history.append({"user": user_q, "bot": bot_ans})
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    user_q = req.query
    history_context = get_recent_history_context()
    augmented_q = f"Previous chat context:\n{history_context}\nCurrent Query: {user_q}" if history_context else user_q
    
    res, summary = rag.retrieve(user_q)
    
    if not res:
        return {
            "draft": "",
            "eval1": "",
            "eval2": "",
            "final": "No context found.",
            "confidence": 0,
            "sources": []
        }
        
    chunks_text = " ".join([f"{r['source']} {r['text']}" for r in res])
    
    unique_sources = {}
    for r in res:
        src = r['source']
        if src not in unique_sources:
            unique_sources[src] = r['text']
            
    sources = [{"source": k, "text": v} for k, v in unique_sources.items()]
    
    draft_ans = rag_agent.draft(augmented_q, summary, chunks_text)
    eval1 = agent1.evaluate(augmented_q, chunks_text, draft_ans)
    eval2 = agent2.evaluate(augmented_q, summary, draft_ans)
    final_ans = fusion_agent.fuse(augmented_q, draft_ans, eval1, eval2, summary, chunks_text)
    
    save_history(user_q, final_ans)
    
    confidence = 0
    for i in range(len(final_ans)-1, 0, -1):
        if final_ans[i].isdigit() and final_ans[i-1].isdigit():
            confidence = int(final_ans[i-1] + final_ans[i])
            break
    if confidence > 100:
        confidence = 100
    print(confidence)
    return {
        "draft": draft_ans,
        "eval1": eval1,
        "eval2": eval2,
        "final": final_ans,
        "confidence": confidence,
        "sources": sources
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
