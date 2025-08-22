#!/usr/bin/env python3
"""
Simple test of PDF parsing using PyMuPDF which is already installed.
"""

import os
import sys
from pathlib import Path

# Test with PyMuPDF directly
try:
    import fitz  # PyMuPDF
    
    file_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/electric_bill.pdf"
    
    print(f"Opening PDF: {file_path}")
    print("-" * 50)
    
    # Open PDF
    pdf_document = fitz.open(file_path)
    
    print(f"Number of pages: {pdf_document.page_count}")
    
    # Extract text from first page
    first_page = pdf_document[0]
    text = first_page.get_text()
    
    print(f"\nFirst page text (first 1000 chars):")
    print("-" * 50)
    print(text[:1000])
    print("-" * 50)
    
    # Look for key information
    lines = text.split('\n')
    for line in lines:
        # Look for account number, kWh, amount due, etc.
        if any(keyword in line.lower() for keyword in ['account', 'kwh', 'amount due', 'total', 'billing period']):
            print(f"Found: {line.strip()}")
    
    # Extract tables if any
    print("\nLooking for tables...")
    tables = first_page.find_tables()
    if tables:
        print(f"Found {len(tables)} tables")
        for i, table in enumerate(tables):
            print(f"\nTable {i+1}:")
            extracted = table.extract()
            for row in extracted[:5]:  # First 5 rows
                print(row)
    
    # Save full text
    with open("electric_bill_text.txt", "w") as f:
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            f.write(f"\n\n{'='*50}\nPage {page_num + 1}\n{'='*50}\n")
            f.write(page.get_text())
    
    print(f"\nFull text saved to: electric_bill_text.txt")
    
    pdf_document.close()
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()