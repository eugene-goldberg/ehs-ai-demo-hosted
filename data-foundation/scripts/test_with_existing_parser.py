#!/usr/bin/env python3
"""
Test using the existing PDF parsing capabilities in the project.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.src.document_sources.local_file import get_documents_from_file_by_path
from backend.src.create_chunks import CreateChunksofDocument
from backend.src.llm import get_llm, get_graph_from_llm
import asyncio

# Test with the existing infrastructure
file_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/electric_bill.pdf"
file_name = "electric_bill.pdf"

print(f"Testing existing parser with: {file_name}")
print("-" * 50)

try:
    # Use existing document loader
    file_name, pages, file_extension = get_documents_from_file_by_path(file_path, file_name)
    
    print(f"Loaded {len(pages)} pages from {file_name}")
    
    # Print first page content
    if pages:
        first_page = pages[0]
        print(f"\nFirst page metadata: {first_page.metadata}")
        print(f"\nFirst page content (first 500 chars):")
        print(first_page.page_content[:500])
        
        # Save full content for inspection
        with open("parsed_electric_bill_existing.txt", "w") as f:
            for i, page in enumerate(pages):
                f.write(f"\n\n{'='*50}\nPage {i+1}\n{'='*50}\n")
                f.write(page.page_content)
        
        print(f"\nFull content saved to: parsed_electric_bill_existing.txt")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()