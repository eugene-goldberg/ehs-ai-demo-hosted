#!/usr/bin/env python3
"""
Test Electricity Endpoint Directly

This script tests the electricity endpoint behavior to confirm the issue
and validate the fix.

Created: 2025-08-31
"""

import os
import sys
import asyncio
from datetime import datetime

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.api.environmental_assessment_api import get_service, get_electricity_facts


async def test_electricity_endpoint_direct():
    """Test the electricity endpoint directly"""
    print("=== TESTING ELECTRICITY ENDPOINT DIRECTLY ===")
    
    try:
        # Get service instance
        service = await get_service()
        
        if service is None:
            print("❌ Service is None - connection failed")
            return
        
        print("✓ Service connected successfully")
        
        # Test with no filters (should return data)
        print("\n--- Test 1: No filters ---")
        facts = await get_electricity_facts(
            location_path=None,
            start_date=None,
            end_date=None,
            service=service
        )
        
        print(f"Facts returned: {len(facts)}")
        if facts:
            print(f"First fact: {facts[0].model_dump()}")
        
        # Test with location filter that doesn't match (should return empty)
        print("\n--- Test 2: With location filter ---")
        facts_filtered = await get_electricity_facts(
            location_path="nonexistent",
            start_date=None,
            end_date=None,
            service=service
        )
        
        print(f"Facts with filter: {len(facts_filtered)}")
        
        # Test with location filter that might match
        print("\n--- Test 3: With partial location filter ---")
        facts_partial = await get_electricity_facts(
            location_path="FAC",  # Partial match
            start_date=None,
            end_date=None,
            service=service
        )
        
        print(f"Facts with partial filter: {len(facts_partial)}")
        
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        import traceback
        traceback.print_exc()


def test_service_with_filters():
    """Test the service directly with various filters"""
    print("\n=== TESTING SERVICE WITH FILTERS ===")
    
    from src.database.neo4j_client import Neo4jClient, ConnectionConfig
    from src.services.environmental_assessment_service import EnvironmentalAssessmentService
    
    try:
        config = ConnectionConfig.from_env()
        client = Neo4jClient(config=config, enable_logging=False)
        client.connect()
        
        service = EnvironmentalAssessmentService(client)
        
        # Test 1: No filters
        print("\n--- Service Test 1: No filters ---")
        result1 = service.assess_electricity_consumption()
        print(f"Records: {len(result1.get('raw_data', []))}")
        print(f"Has error: {'error' in result1}")
        
        # Test 2: With location filter that won't match NULL values
        print("\n--- Service Test 2: Location filter 'FAC' ---") 
        result2 = service.assess_electricity_consumption(location_filter="FAC")
        print(f"Records: {len(result2.get('raw_data', []))}")
        print(f"Has error: {'error' in result2}")
        if 'error' in result2:
            print(f"Error: {result2['error']}")
        
        # Test 3: Check if any records actually have location data
        print("\n--- Raw data analysis ---")
        raw_data = result1.get('raw_data', [])
        if raw_data:
            locations = [r.get('location') for r in raw_data[:5]]
            print(f"Sample locations: {locations}")
            non_null_locations = [loc for loc in locations if loc is not None]
            print(f"Non-null locations: {non_null_locations}")
        
    except Exception as e:
        print(f"❌ Error testing service: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("ELECTRICITY ENDPOINT DEBUG TEST")
    print("="*50)
    
    # Test service behavior first
    test_service_with_filters()
    
    # Test endpoint behavior
    asyncio.run(test_electricity_endpoint_direct())
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("""
Based on the tests above, the electricity endpoint issue is likely caused by:

1. All ElectricityConsumption records have NULL location values
2. When any location filter is applied, it tries to match against NULL values
3. The CONTAINS operator on NULL always returns NULL (not true/false)
4. This results in all records being filtered out

IMMEDIATE FIX REQUIRED:
Update the query in environmental_assessment_service.py to use 'facility_id' 
instead of 'location' property.
    """)


if __name__ == '__main__':
    main()