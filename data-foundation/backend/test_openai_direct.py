from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Get the API key from environment
config = os.getenv('LLM_MODEL_CONFIG_openai_gpt_4o')
if config:
    model_name, api_key = config.split(',')
    print(f"Testing with API key ending in: ...{api_key[-10:]}")
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print("Success! Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Config not found")
