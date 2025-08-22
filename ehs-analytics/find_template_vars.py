#!/usr/bin/env python3
import sys
import re
sys.path.insert(0, '/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src')

from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever

config = {'neo4j_uri': 'bolt://test:7687', 'neo4j_user': 'test', 'neo4j_password': 'test', 'openai_api_key': 'fake-key'}
retriever = Text2CypherRetriever(config)
prompt = retriever._build_ehs_cypher_prompt()

vars = re.findall(r'\{(\w+)\}', prompt)
print('Template variables found:', set(vars))

# Print lines containing name or source
lines = prompt.split('\n')
for i, line in enumerate(lines, 1):
    if '{name}' in line or '{source}' in line:
        print(f'Line {i}: {line}')

# Print the full prompt to examine
print("\n" + "="*50)
print("FULL PROMPT:")
print("="*50)
print(prompt)