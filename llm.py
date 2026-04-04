import os
import re
from groq import Groq
from dotenv import load_dotenv

class LLM:
    def __init__(self, key_env_name, model="groq/compound-mini"):
        load_dotenv()
        self.model = model
        self.key_env_name = key_env_name
        self.client = None

    def get_key(self):
        api_key = os.environ.get(self.key_env_name)
        if not api_key:
            api_key = input(f"Enter {self.key_env_name}: ").strip()
            os.environ[self.key_env_name] = api_key
        self.client = Groq(api_key=api_key)

    def generate(self, prompt):
        if not self.client:
            self.get_key()
            
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        output = completion.choices[0].message.content
        output = output.replace('\n', ' ')
        output = re.sub(r'[^a-zA-Z0-9\s]', '', output)
        return re.sub(r'\s+', ' ', output).strip()
