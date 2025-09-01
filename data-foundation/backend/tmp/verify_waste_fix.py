#!/usr/bin/env python3
"""
Verification script to test waste endpoints after schema fixes.
Tests waste facts, risks, and recommendations endpoints.
"""

import requests
import json
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
ENDPOINTS = {
    "waste_facts": "/api/environmental/waste/facts",
    "waste_risks": "/api/environmental/waste/risks", 
    "waste_recommendations": "/api/environmental/waste/recommendations"
}

def test_endpoint(endpoint_name, url):
    """Test a single endpoint and return results."""
    print(f"\n{'='*50}")
    print(f"Testing {endpoint_name}: {url}")
    print('='*50)
    
    try:
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if data is returned
            if isinstance(data, list):
                count = len(data)
                print(f"Data Count: {count} items")
                
                if count > 0:
                    print(f"Sample Item (first item):")
                    print(json.dumps(data[0], indent=2))
                    return True, count, "SUCCESS"
                else:
                    return False, 0, "EMPTY_LIST"
            elif isinstance(data, dict):
                print(f"Response Data:")
                print(json.dumps(data, indent=2))
                return True, 1, "SUCCESS"
            else:
                print(f"Unexpected data type: {type(data)}")
                return False, 0, "UNEXPECTED_TYPE"
        else:
            print(f"Error Response: {response.text}")
            return False, 0, f"HTTP_{response.status_code}"
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is the API server running?")
        return False, 0, "CONNECTION_ERROR"
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
        return False, 0, "TIMEOUT"
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False, 0, f"EXCEPTION_{type(e).__name__}"

def main():
    """Main verification function."""
    print(f"Waste Endpoints Verification Test")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {BASE_URL}")
    
    results = {}
    total_success = 0
    total_tests = len(ENDPOINTS)
    
    # Test each endpoint
    for endpoint_name, endpoint_path in ENDPOINTS.items():
        full_url = f"{BASE_URL}{endpoint_path}"
        success, count, status = test_endpoint(endpoint_name, full_url)
        
        results[endpoint_name] = {
            "success": success,
            "count": count,
            "status": status,
            "url": full_url
        }
        
        if success:
            total_success += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY")
    print('='*70)
    
    for endpoint_name, result in results.items():
        status_symbol = "✅" if result["success"] else "❌"
        print(f"{status_symbol} {endpoint_name:25} | Count: {result['count']:3} | Status: {result['status']}")
    
    print(f"\nOverall Result: {total_success}/{total_tests} endpoints working")
    
    if total_success == total_tests:
        print("\n🎉 ALL WASTE ENDPOINTS ARE WORKING CORRECTLY!")
        print("The schema fixes have been successful.")
        return True
    else:
        print(f"\n⚠️  {total_tests - total_success} endpoint(s) still have issues.")
        print("Schema fixes may need additional work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
