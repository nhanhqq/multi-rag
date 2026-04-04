from init import DataInitializer
from agents import RagAgent, Agent1, Agent2, FusionAgent

def main():
    initializer = DataInitializer()
    rag = initializer.init_data()
    
    rag_agent = RagAgent()
    agent1 = Agent1()
    agent2 = Agent2()
    fusion_agent = FusionAgent()
    
    while True:
        try:
            user_q = input("\nSearch: ").strip()
            if not user_q or user_q.lower() in ['exit', 'quit']:
                break
                
            res, summary = rag.retrieve(user_q)
            if not res:
                print("No result found in RAG.")
                continue
                
            chunks_text = " ".join([f"{r['source']} {r['text']}" for r in res])
            
            print("--- [Step 1] RagAgent: Generating Draft Answer...")
            draft_ans = rag_agent.draft(user_q, summary, chunks_text)
            
            print("--- [Step 2] Agent 1: Evaluating Accuracy...")
            eval1 = agent1.evaluate(user_q, chunks_text, draft_ans)
                
            print("--- [Step 3] Agent 2: Evaluating Tone...")
            eval2 = agent2.evaluate(user_q, summary, draft_ans)
                
            print("--- [Step 4] FusionAgent: Fusing results into final answer...")
            final_ans = fusion_agent.fuse(user_q, draft_ans, eval1, eval2, summary, chunks_text)
            
            print("\nFinal Result:")
            print(final_ans)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
