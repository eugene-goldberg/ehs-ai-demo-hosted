#!/usr/bin/env python3
"""
Test script for water consumption context retrieval.
Tests the new get_water_context() method in ContextRetriever class.
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from services.context_retriever import ContextRetriever, get_context_for_intent

def test_water_context_direct():
    """Test the get_water_context method directly"""
    print("=== Testing get_water_context() method directly ===")
    
    retriever = ContextRetriever()
    
    try:
        # Test 1: Get water context for Algonquin site
        print("\nTest 1: Water context for Algonquin site")
        context = retriever.get_water_context('algonquin_il')
        print(f"Result: {json.dumps(context, indent=2)}")
        
        # Test 2: Get water context for Houston site
        print("\nTest 2: Water context for Houston site")
        context = retriever.get_water_context('houston_texas')
        print(f"Result: {json.dumps(context, indent=2)}")
        
        # Test 3: Get water context for Algonquin with date range
        print("\nTest 3: Water context for Algonquin with date range")
        context = retriever.get_water_context('algonquin_il', '2025-06-01', '2025-06-30')
        print(f"Result: {json.dumps(context, indent=2)}")
        
        # Test 4: Test with mapped site name
        print("\nTest 4: Water context using mapped site name 'algonquin'")
        context = retriever.get_water_context('algonquin')
        print(f"Result: {json.dumps(context, indent=2)}")
        
    except Exception as e:
        print(f"Error in direct test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        retriever.close()

def test_water_context_via_intent():
    """Test the water context via get_context_for_intent function"""
    print("\n=== Testing water context via get_context_for_intent() ===")
    
    try:
        # Test 1: Water consumption intent for Algonquin
        print("\nTest 1: Water consumption intent for Algonquin")
        context_json = get_context_for_intent('water_consumption', 'algonquin_il')
        print(f"Result: {context_json}")
        
        # Test 2: Water consumption intent for Houston
        print("\nTest 2: Water consumption intent for Houston")
        context_json = get_context_for_intent('water_consumption', 'houston_texas')
        print(f"Result: {context_json}")
        
        # Test 3: Water consumption intent with date range
        print("\nTest 3: Water consumption intent with date range")
        context_json = get_context_for_intent('water_consumption', 'algonquin_il', '2025-06-01', '2025-06-30')
        print(f"Result: {context_json}")
        
    except Exception as e:
        print(f"Error in intent test: {e}")
        import traceback
        traceback.print_exc()

def test_comparison_with_electricity():
    """Compare water and electricity context structures"""
    print("\n=== Comparing water and electricity context structures ===")
    
    try:
        # Get both water and electricity contexts for comparison
        print("\nElectricity context for Algonquin:")
        elec_context = get_context_for_intent('electricity_consumption', 'algonquin_il')
        print(elec_context)
        
        print("\nWater context for Algonquin:")
        water_context = get_context_for_intent('water_consumption', 'algonquin_il')
        print(water_context)
        
    except Exception as e:
        print(f"Error in comparison test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting Water Consumption Context Retrieval Tests")
    print("=" * 60)
    
    test_water_context_direct()
    test_water_context_via_intent()
    test_comparison_with_electricity()
    
    print("\n" + "=" * 60)
    print("Water Consumption Context Tests Completed")
