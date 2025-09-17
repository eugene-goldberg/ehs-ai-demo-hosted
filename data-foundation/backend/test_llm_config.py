import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check the environment variable
config = os.getenv('LLM_MODEL_CONFIG_openai_gpt_4o')
print(f"LLM_MODEL_CONFIG_openai_gpt_4o = {config}")

if config:
    parts = config.split(',')
    print(f"Model: {parts[0]}")
    print(f"API Key (first 20 chars): {parts[1][:20] if len(parts) > 1 else 'N/A'}...")
    print(f"API Key (last 10 chars): ...{parts[1][-10:] if len(parts) > 1 else 'N/A'}")
