#!/usr/bin/env python3
"""
Test script to verify transcript forwarding works when extractors are called.
"""

import sys
import os
sys.path.insert(0, 'src')

# Set proper PYTHONPATH
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

from extractors.ehs_extractors import UtilityBillExtractor

def test_extractor_transcript():
    """Test if extractor properly forwards transcript entries."""
    print("Testing UtilityBillExtractor with transcript forwarding...")
    
    # Sample utility bill content
    test_content = """
    ELECTRIC BILL
    Account Number: 12345
    Bill Date: November 15, 2024
    Due Date: December 5, 2024
    
    Service Period: October 1 - October 31, 2024
    
    Current Reading: 5432 kWh
    Previous Reading: 5232 kWh
    Usage This Period: 200 kWh
    
    Total Amount Due: $45.00
    """
    
    # Create extractor
    extractor = UtilityBillExtractor(llm_model="gpt-4")
    
    # Extract data (this should trigger transcript forwarding)
    print("\nCalling extractor.extract()...")
    result = extractor.extract(
        content=test_content,
        metadata={"filename": "test_bill.pdf", "source": "test_script"}
    )
    
    print("\nExtraction completed!")
    print(f"Result keys: {list(result.keys())}")
    
    # Check transcript file
    transcript_file = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/web-app/backend/transcript_data.json"
    if os.path.exists(transcript_file):
        with open(transcript_file, 'r') as f:
            import json
            transcript = json.load(f)
            print(f"\nTranscript entries: {len(transcript)}")
            if transcript:
                print("Latest entry:")
                print(f"  Role: {transcript[-1]['role']}")
                print(f"  Content length: {len(transcript[-1]['content'])}")
    else:
        print("\nTranscript file not found!")

if __name__ == "__main__":
    test_extractor_transcript()