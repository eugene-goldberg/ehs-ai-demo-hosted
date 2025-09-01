#!/usr/bin/env python3
"""
Verification script to test that the water consumption fix works.
This script tests the fixed environmental assessment service.
"""

import os
import sys
import json
from datetime import datetime

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from database.neo4j_client import create_neo4j_client
from services.environmental_assessment_service import EnvironmentalAssessmentService

def test_water_consumption_fix():
    """Test that the water consumption query fix works."""
    print("="*60)
    print("TESTING WATER CONSUMPTION FIX")
    print("="*60)
    
    # Create Neo4j client
    client = create_neo4j_client()
    if not client.connect():
        print("✗ Failed to connect to Neo4j")
        return False
    
    try:
        # Create environmental assessment service
        service = EnvironmentalAssessmentService(client)
        
        # Test 1: Get comprehensive assessment
        print("\nTest 1: Comprehensive Environmental Assessment")
        print("-" * 40)
        
        assessment = service.get_comprehensive_assessment(
            location_filter=None,  # Get all locations
            start_date="2025-07-01",
            end_date="2025-08-31"
        )
        
        if 'water' in assessment and 'error' not in assessment['water']:
            water_data = assessment['water']
            print(f"✓ Water data retrieved successfully")
            print(f"  Data points: {water_data['data_points_count']}")
            print(f"  Total consumption: {water_data['facts'].get('total_consumption', 0):,.0f} gallons")
            print(f"  Average consumption: {water_data['facts'].get('average_consumption', 0):,.0f} gallons")
            print(f"  Total cost: ${water_data['facts'].get('total_cost', 0):,.2f}")
            
            if water_data.get('raw_data'):
                print(f"  Sample data point: {water_data['raw_data'][0]}")
                
        else:
            print(f"✗ Water data retrieval failed: {assessment.get('water', {}).get('error', 'Unknown error')}")
            return False
        
        # Test 2: Test with specific location filter
        print("\nTest 2: Location-Filtered Water Assessment")
        print("-" * 40)
        
        filtered_assessment = service.assess_water_consumption(
            location_filter="DEMO_FACILITY_001",
            start_date="2025-08-01",
            end_date="2025-08-31"
        )
        
        if 'error' not in filtered_assessment:
            print(f"✓ Location-filtered water data retrieved successfully")
            print(f"  Data points: {filtered_assessment['data_points_count']}")
            print(f"  Total consumption: {filtered_assessment['facts'].get('total_consumption', 0):,.0f} gallons")
            
        else:
            print(f"✗ Location-filtered query failed: {filtered_assessment.get('error', 'Unknown error')}")
            return False
        
        # Test 3: Test LLM context generation
        print("\nTest 3: LLM Context Data Generation")
        print("-" * 40)
        
        context_data = service.get_llm_context_data(
            location_filter="DEMO_FACILITY_001",
            start_date="2025-08-01", 
            end_date="2025-08-31"
        )
        
        if "WATER CONSUMPTION" in context_data:
            print("✓ LLM context generation includes water data")
            # Show first 200 characters of context
            print(f"  Context preview: {context_data[:200]}...")
        else:
            print("✗ LLM context generation missing water data")
            print(f"  Full context: {context_data}")
            return False
        
        # Test 4: Compare with electricity (should both work)
        print("\nTest 4: Comparison with Electricity Data")
        print("-" * 40)
        
        electricity_data = service.assess_electricity_consumption(
            location_filter="DEMO_FACILITY_001",
            start_date="2025-08-01",
            end_date="2025-08-31"
        )
        
        if 'error' not in electricity_data:
            print(f"✓ Electricity data: {electricity_data['data_points_count']} points")
        
        if 'error' not in filtered_assessment:
            print(f"✓ Water data: {filtered_assessment['data_points_count']} points")
        
        print("\n✓ ALL TESTS PASSED - Water consumption fix successful!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        client.close()

def main():
    """Main test function."""
    print("Starting Water Consumption Fix Verification")
    print("="*60)
    
    success = test_water_consumption_fix()
    
    if success:
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("Water consumption endpoints should now return data.")
    else:
        print("\n❌ VERIFICATION FAILED!")
        print("Additional troubleshooting may be needed.")

if __name__ == "__main__":
    main()