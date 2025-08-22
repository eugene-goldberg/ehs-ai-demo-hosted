#!/usr/bin/env python3
"""
Simple test script to verify LlamaParse is working with the electric bill.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set the API key
os.environ["LLAMA_PARSE_API_KEY"] = "llx-R34kCCUyTeeZKwsQm6LeAyAfe7YGXOJlnBdmYEfb9PMduibA"

try:
    from llama_parse import LlamaParse
    
    # Initialize parser
    parser = LlamaParse(
        api_key=os.environ["LLAMA_PARSE_API_KEY"],
        result_type="markdown",
        parsing_instruction="""
        Extract the following information from this utility bill:
        - Billing period (start and end dates)
        - Total energy consumption (kWh)
        - Total cost
        - Account number
        - Service address
        Preserve all tabular data in markdown format.
        """,
        verbose=True
    )
    
    # Parse the electric bill
    file_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/electric_bill.pdf"
    print(f"Parsing document: {file_path}")
    
    documents = parser.load_data(file_path)
    
    print(f"\nSuccessfully parsed {len(documents)} pages")
    
    # Print the first page content
    if documents:
        print("\nFirst page content:")
        print("-" * 50)
        print(documents[0].text[:1000])  # First 1000 characters
        print("-" * 50)
        
        # Save the full output for inspection
        with open("parsed_electric_bill.md", "w") as f:
            for i, doc in enumerate(documents):
                f.write(f"\n\n# Page {i+1}\n\n")
                f.write(doc.text)
        print("\nFull parsed content saved to: parsed_electric_bill.md")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()