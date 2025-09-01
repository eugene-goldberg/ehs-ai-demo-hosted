#!/usr/bin/env python3
"""
Test the water consumption API endpoint with correct paths.
"""

import requests
import json
import time

def test_water_endpoints():
    """Test the water consumption API endpoints."""
    base_url = "http://localhost:8000"
    
    print("="*60)
    print("TESTING WATER CONSUMPTION API ENDPOINTS")
    print("="*60)
    
    # Test 1: Water Facts endpoint
    print("\nTest 1: Water Facts")
    print("-" * 40)
    
    endpoint = f"{base_url}/api/environmental/water/facts"
    params = {
        "start_date": "2025-08-01T00:00:00",
        "end_date": "2025-08-31T23:59:59",
        "location_path": "DEMO_FACILITY_001"
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Water facts endpoint successful")
            print(f"  Status: {response.status_code}")
            print(f"  Number of facts: {len(data)}")
            
            if data:
                print(f"  Sample fact: {data[0]['title']}")
                print(f"  Description: {data[0]['description']}")
            else:
                print("  No facts returned")
                
        else:
            print(f"✗ Water facts endpoint failed")
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed - API server may not be running")
        return False
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False
    
    # Test 2: Water Risks endpoint
    print("\nTest 2: Water Risks")
    print("-" * 40)
    
    endpoint = f"{base_url}/api/environmental/water/risks"
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Water risks endpoint successful")
            print(f"  Number of risks: {len(data)}")
            
            if data:
                print(f"  Sample risk: {data[0]['title']}")
                print(f"  Severity: {data[0]['severity']}")
        else:
            print(f"✗ Water risks endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Water risks request failed: {e}")
        return False
    
    # Test 3: Water Recommendations endpoint
    print("\nTest 3: Water Recommendations")
    print("-" * 40)
    
    endpoint = f"{base_url}/api/environmental/water/recommendations"
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Water recommendations endpoint successful")
            print(f"  Number of recommendations: {len(data)}")
            
            if data:
                print(f"  Sample recommendation: {data[0]['title']}")
                print(f"  Priority: {data[0]['priority']}")
        else:
            print(f"✗ Water recommendations endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Water recommendations request failed: {e}")
        return False
    
    # Test 4: Generic category endpoint
    print("\nTest 4: Generic Water Category Endpoint")
    print("-" * 40)
    
    endpoint = f"{base_url}/api/environmental/water/facts"
    params_no_location = {
        "start_date": "2025-08-01T00:00:00",
        "end_date": "2025-08-31T23:59:59"
        # No location filter - should return all water data
    }
    
    try:
        response = requests.get(endpoint, params=params_no_location, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Generic water facts endpoint successful")
            print(f"  Number of facts (all locations): {len(data)}")
            
            if data:
                # Show facts from different locations
                locations = set(fact.get('location_path', 'unknown') for fact in data)
                print(f"  Locations found: {list(locations)}")
        else:
            print(f"✗ Generic water facts endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Generic water facts request failed: {e}")
        return False
    
    print("\n✓ ALL WATER ENDPOINT TESTS PASSED!")
    return True

def main():
    """Main test function."""
    print("Testing Water Consumption API Endpoints (Corrected)")
    print("="*60)
    
    success = test_water_endpoints()
    
    if success:
        print("\n🎉 WATER ENDPOINT TESTS SUCCESSFUL!")
        print("The water consumption API endpoints are working correctly.")
        print("This confirms that the water consumption fix is working in the API.")
    else:
        print("\n❌ WATER ENDPOINT TESTS FAILED!")
        print("The fix may not be working in the API layer.")

if __name__ == "__main__":
    main()