#!/usr/bin/env python3
"""
Fix script to add industry_citations field to context_retriever.py
"""

import re

# Read the current file
with open('/home/azureuser/dev/ehs-ai-demo/data-foundation/backend/src/services/context_retriever.py', 'r') as f:
    content = f.read()

# Find the processed_rec dictionary and add the industry_citations field
pattern = r"(\s+'recommendation_id': rec\.get\('id', f"{site_id}_{len\(all_recommendations\)}"\))(\s+}")"

replacement = r"\1,\n                                'industry_citations': rec.get('industry_citation', '')\2"

# Apply the fix
fixed_content = re.sub(pattern, replacement, content)

# Write back to file
with open('/home/azureuser/dev/ehs-ai-demo/data-foundation/backend/src/services/context_retriever.py', 'w') as f:
    f.write(fixed_content)

print("Fixed context_retriever.py to include industry_citations field")
