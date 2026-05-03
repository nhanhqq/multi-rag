from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import os
import re
import time
import threading
import asyncio
from datetime import datetime
from core.agents import RagAgent, Agent1, Agent2, FusionAgent
from loader import load_domain_data
from core.RAG import Retriever

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
STATS_FILE = "stats.json"

def init_stats():
    if not os.path.exists(STATS_FILE):
        stats = {
            "total_queries": 0,
            "unique_queries": set(),
            "total_tokens": 0,
            "cache_hits": 0,
            "queries": []
        }
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump({"total_queries": 0, "unique_queries": [], "total_tokens": 0, "cache_hits": 0, "queries": []}, f)

IS_READY = False
IGNORED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.mp4', '.zip', '.rar', '.exe', '.pkl', '.index', '.db', '.sqlite', '.pyc', '.DS_Store')

def sync_knowledge_loop():
    global IS_READY
    last_state = {}
    pdf_dir = "./pdf"
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        
    initial_chunks = load_domain_data(pdf_dir)
    rag.sync(initial_chunks)
    IS_READY = True
    
    while True:
        try:
            current_files = [f for f in os.listdir(pdf_dir) if not f.lower().endswith(IGNORED_EXTENSIONS)]
            current_state = {f: os.path.getmtime(os.path.join(pdf_dir, f)) for f in current_files}
            if current_state != last_state:
                chunks = load_domain_data(pdf_dir)
                rag.sync(chunks)
                last_state = current_state
        except Exception:
            pass
        time.sleep(5)

@app.on_event("startup")
def startup_event():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    init_stats()
    thread = threading.Thread(target=sync_knowledge_loop, daemon=True)
    thread.start()

@app.post("/api/clear")
def clear_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
    return {"status": "ok"}

@app.post("/api/clear_stats")
def clear_stats():
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump({"total_queries": 0, "unique_queries": [], "total_tokens": 0, "cache_hits": 0, "queries": []}, f)
    return {"status": "ok"}

import numpy as np

@app.get("/api/graph")
def get_knowledge_graph():
    if not rag.chunks or not rag.embeddings_cache:
        return {"nodes": [], "links": []}
    
    nodes = []
    sources_map = {}
    current_group = 0
    for i, chunk in enumerate(rag.chunks):
        src = chunk['source']
        if src not in sources_map:
            sources_map[src] = current_group
            current_group += 1
        nodes.append({"id": i, "label": src, "group": sources_map[src], "text": chunk['text']})
        
    links = []
    embeddings = np.array(rag.embeddings_cache).astype('float32')
    num_nodes = len(embeddings)
    
    for i in range(num_nodes):
        query_vec = embeddings[i].reshape(1, -1)
        D, I = rag.index.search(query_vec, 8)
        for dist, idx in zip(D[0], I[0]):
            if idx != -1 and idx != i and dist > 0.6:
                links.append({"source": i, "target": int(idx), "weight": float(dist)})
                
    return {"nodes": nodes, "links": links}

def update_stats(query, tokens, cache_hit, trace=None):
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except:
        stats = {"total_queries": 0, "unique_queries": [], "total_tokens": 0, "cache_hits": 0, "queries": []}
    
    stats["total_queries"] += 1
    if query not in stats["unique_queries"]:
        stats["unique_queries"].append(query)
    stats["total_tokens"] += tokens
    if cache_hit:
        stats["cache_hits"] += 1
    
    stats["queries"].append({
        "time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "query": query,
        "tokens": tokens,
        "cache": cache_hit,
        "trace": trace
    })
    
    if len(stats["queries"]) > 500:
        stats["queries"] = stats["queries"][-500:]
        
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

@app.get("/api/health")
def health_check():
    return {"status": 1 if IS_READY else 0}

@app.get("/api/stats")
def get_stats():
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"total_queries": 0, "unique_queries": [], "total_tokens": 0, "cache_hits": 0, "queries": []}

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

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

from fastapi.responses import StreamingResponse

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    async def generate_events():
        user_q = req.query
        history_context = get_recent_history_context()
        augmented_q = f"Previous context:\n{history_context}\nQuery: {user_q}" if history_context else user_q
        
        start_time = time.time()
        res, summary = rag.retrieve(user_q, top_k=req.top_k)
        retrieve_time = time.time() - start_time
        
        if not res:
            yield json.dumps({"type": "final", "data": "No context found.", "confidence": 0}) + "\n"
            return
            
        indices = [r.get('id', -1) for r in res]
        unique_sources = {}
        for r in res:
            if r['source'] not in unique_sources:
                unique_sources[r['source']] = r['text']
        sources = [{"source": k, "text": v} for k, v in unique_sources.items()]
        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        
        trace_data = {
            "retrieve_time": retrieve_time,
            "chunks_found": len(res),
            "sources": [r['source'] for r in res],
            "scores": [float(r.get('final_score', 0)) for r in res],
            "indices": indices
        }
        
        chunks_text = " ".join([f"{r['source']} {r['text']}" for r in res])
        draft_ans = rag_agent.draft(augmented_q, summary, chunks_text)
        yield json.dumps({"type": "draft", "data": draft_ans}) + "\n"
        
        eval1 = agent1.evaluate(augmented_q, chunks_text, draft_ans)
        yield json.dumps({"type": "eval1", "data": eval1}) + "\n"
        
        eval2 = agent2.evaluate(augmented_q, summary, draft_ans)
        yield json.dumps({"type": "eval2", "data": eval2}) + "\n"
        
        final_ans = fusion_agent.fuse(augmented_q, draft_ans, eval1, eval2, summary, chunks_text)
        save_history(user_q, final_ans)
        
        match = re.search(r"Confidence Score:\s*(\d+)", final_ans, re.IGNORECASE)
        confidence = int(match.group(1)) if match else 0
        if not confidence:
            nums = re.findall(r"\d+", final_ans)
            confidence = int(nums[-1]) if nums else 0
        if confidence > 100: confidence = 100
        
        tokens = len(final_ans.split()) + len(user_q.split())
        update_stats(user_q, tokens, confidence > 90, trace_data)
        
        yield json.dumps({"type": "final", "data": final_ans, "confidence": confidence}) + "\n"

    return StreamingResponse(generate_events(), media_type="application/x-ndjson")

@app.get("/api/files")
def get_files():
    pdf_dir = "./pdf"
    if not os.path.exists(pdf_dir):
        return []
    return [f for f in os.listdir(pdf_dir) if not f.lower().endswith(IGNORED_EXTENSIONS)]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

