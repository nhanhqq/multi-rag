import os
from groq import Groq
from core.RAG import Retriever
from dotenv import load_dotenv

load_dotenv()

def main():
    rag = Retriever()
    if not rag.load():
        rag.sync("./pdf")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Enter your GROQ_API_KEY: ").strip()
        
    client = Groq(api_key=api_key)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, "system_promt", "rag.txt")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    while True:
        user_q = input("\nSearch: ").strip()
        if not user_q or user_q.lower() in ['exit', 'quit']:
            break
            
        res, summary = rag.retrieve(user_q)
        if not res:
            print("No relevant information found.")
            continue
            
        chunks_text = "\n".join([f"Source: {r['source']}\nContent: {r['text']}" for r in res])
        
        full_prompt = prompt_template.format(user_q, summary, chunks_text)
        
        completion = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0,
        )
        
        print("\nAnswer:")
        print(completion.choices[0].message.content)

if __name__ == "__main__":
    main()
