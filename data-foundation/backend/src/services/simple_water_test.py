#!/usr/bin/env python3
"""
Simple test for water consumption context retrieval.
"""

import sys
import os
import json

# Import the context retriever
from context_retriever import ContextRetriever, get_context_for_intent

def test_water_context():
    """Test the water context retrieval"""
    print("Testing Water Consumption Context Retrieval")
    print("=" * 50)
    
    try:
        # Test 1: Direct method call
        print("\nTest 1: Direct get_water_context() call for Algonquin")
        retriever = ContextRetriever()
        context = retriever.get_water_context('algonquin_il')
        print(f"Result: {json.dumps(context, indent=2)}")
        retriever.close()
        
        # Test 2: Via intent function
        print("\nTest 2: Via get_context_for_intent() for water_consumption")
        context_json = get_context_for_intent('water_consumption', 'algonquin_il')
        print(f"Result: {context_json}")
        
        # Test 3: Test with Houston site
        print("\nTest 3: Water context for Houston site")
        context_json = get_context_for_intent('water_consumption', 'houston_texas')
        print(f"Result: {context_json}")
        
        # Test 4: Test with date range
        print("\nTest 4: Water context with date range")
        context_json = get_context_for_intent('water_consumption', 'algonquin_il', '2025-06-01', '2025-06-30')
        print(f"Result: {context_json}")
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_water_context()
