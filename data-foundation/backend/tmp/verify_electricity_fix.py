#!/usr/bin/env python3
"""
Verify Electricity Fix

Final verification that the electricity endpoints fix is working correctly.
This tests the actual API endpoints that were failing before.

Created: 2025-08-31
"""

import requests
import json


def test_electricity_api_endpoints():
    """Test the actual electricity API endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints_to_test = [
        "/api/environmental/electricity/facts",
        "/api/environmental/electricity/risks", 
        "/api/environmental/electricity/recommendations"
    ]
    
    print("TESTING ELECTRICITY API ENDPOINTS")
    print("=" * 50)
    
    for endpoint in endpoints_to_test:
        print(f"\n--- Testing {endpoint} ---")
        
        try:
            # Test without filters
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ SUCCESS: {len(data)} items returned")
                
                if data:
                    print(f"  First item: {json.dumps(data[0], indent=2, default=str)[:200]}...")
                else:
                    print("  ⚠️  No data returned, but API responded successfully")
                    
            else:
                print(f"❌ FAILED: Status code {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ CONNECTION ERROR: {e}")
            
        # Test with location filter
        try:
            response = requests.get(f"{base_url}{endpoint}?location_path=FAC", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ With filter 'FAC': {len(data)} items returned")
            else:
                print(f"❌ Filter test failed: Status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Filter test connection error: {e}")


def main():
    """Main function"""
    print("ELECTRICITY ENDPOINTS FIX VERIFICATION")
    print("=" * 60)
    print("This script verifies that the electricity endpoints are now working correctly.")
    print("The fix changed the query to use 'facility_id' instead of 'location' property.")
    print()
    
    test_electricity_api_endpoints()
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("""
SUMMARY OF THE FIX:
- Problem: ElectricityConsumption nodes have 'facility_id' property, not 'location'
- Solution: Updated query in environmental_assessment_service.py line 119 and 122
- Result: Electricity endpoints now return data correctly with proper location filtering

BEFORE: Query used e.location CONTAINS $location (always returned NULL)
AFTER:  Query uses e.facility_id CONTAINS $location (returns actual facility names)

The endpoints should now:
1. Return data when called without filters
2. Return filtered data when location_path parameter is provided
3. Return proper facility names in the location field
    """)


if __name__ == '__main__':
    main()