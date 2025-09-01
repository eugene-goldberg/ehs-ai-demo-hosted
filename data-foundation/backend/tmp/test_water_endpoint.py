#!/usr/bin/env python3
"""
Test the water consumption API endpoint directly.
"""

import requests
import json
import time

def test_water_endpoint():
    """Test the water consumption API endpoint."""
    base_url = "http://localhost:8000"  # Assuming API runs on port 8000
    
    print("="*60)
    print("TESTING WATER CONSUMPTION API ENDPOINT")
    print("="*60)
    
    # Test 1: Basic water consumption endpoint
    print("\nTest 1: Basic Water Consumption Query")
    print("-" * 40)
    
    endpoint = f"{base_url}/environmental/water"
    params = {
        "start_date": "2025-08-01",
        "end_date": "2025-08-31",
        "location": "DEMO_FACILITY_001"
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API call successful")
            print(f"  Status: {response.status_code}")
            print(f"  Data points: {data.get('data_points_count', 'N/A')}")
            
            if 'facts' in data:
                facts = data['facts']
                print(f"  Total consumption: {facts.get('total_consumption', 0):,.0f} gallons")
                print(f"  Total cost: ${facts.get('total_cost', 0):,.2f}")
                print(f"  Average consumption: {facts.get('average_consumption', 0):,.0f} gallons")
            
            if 'error' in data:
                print(f"  Error in response: {data['error']}")
                return False
                
        else:
            print(f"✗ API call failed")
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed - API server may not be running")
        print("  Try starting the API server first")
        return False
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False
    
    # Test 2: Comprehensive environmental assessment
    print("\nTest 2: Comprehensive Environmental Assessment")
    print("-" * 40)
    
    endpoint = f"{base_url}/environmental/assessment"
    params = {
        "start_date": "2025-08-01", 
        "end_date": "2025-08-31",
        "location": "DEMO_FACILITY_001"
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Comprehensive assessment successful")
            
            if 'water' in data and 'error' not in data['water']:
                water_data = data['water']
                print(f"  Water data points: {water_data.get('data_points_count', 'N/A')}")
                if 'facts' in water_data:
                    facts = water_data['facts']
                    print(f"  Water total: {facts.get('total_consumption', 0):,.0f} gallons")
            else:
                print(f"  Water data error: {data.get('water', {}).get('error', 'Not found')}")
                return False
        else:
            print(f"✗ Comprehensive assessment failed")
            print(f"  Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Comprehensive assessment request failed: {e}")
        return False
    
    print("\n✓ ALL ENDPOINT TESTS PASSED!")
    return True

def main():
    """Main test function."""
    print("Testing Water Consumption API Endpoints")
    print("="*60)
    
    success = test_water_endpoint()
    
    if success:
        print("\n🎉 WATER ENDPOINT TESTS SUCCESSFUL!")
        print("The water consumption API endpoints are working correctly.")
    else:
        print("\n❌ WATER ENDPOINT TESTS FAILED!")
        print("Check that the API server is running and accessible.")

if __name__ == "__main__":
    main()