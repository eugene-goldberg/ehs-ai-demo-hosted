#!/usr/bin/env python3
"""
Test script to verify prorating endpoint availability and functionality.
This helps diagnose the current state and guides Task 1 implementation.
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Test results storage
results = []

def test_endpoint(method, path, data=None, expected_status=200, description=""):
    """Test an endpoint and record results."""
    url = f"{BASE_URL}{path}"
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Method: {method}")
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=HEADERS)
        elif method == "POST":
            print(f"Data: {json.dumps(data, indent=2) if data else 'None'}")
            response = requests.post(url, headers=HEADERS, json=data)
        else:
            response = None
            
        if response:
            print(f"Status: {response.status_code}")
            print(f"Expected: {expected_status}")
            
            # Try to parse JSON response
            try:
                response_data = response.json()
                print(f"Response: {json.dumps(response_data, indent=2)}")
            except:
                print(f"Response: {response.text}")
            
            success = response.status_code == expected_status
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
            
            results.append({
                "endpoint": path,
                "method": method,
                "status": response.status_code,
                "expected": expected_status,
                "success": success,
                "description": description
            })
        else:
            print("Error: No response object")
            results.append({
                "endpoint": path,
                "method": method,
                "status": "ERROR",
                "expected": expected_status,
                "success": False,
                "description": description
            })
            
    except Exception as e:
        print(f"Error: {str(e)}")
        results.append({
            "endpoint": path,
            "method": method,
            "status": "EXCEPTION",
            "expected": expected_status,
            "success": False,
            "description": description,
            "error": str(e)
        })

def main():
    """Run all prorating endpoint tests."""
    print("="*60)
    print("PRORATING ENDPOINT TESTING")
    print(f"Started at: {datetime.now().isoformat()}")
    print("="*60)
    
    # Test 1: Check main API health
    test_endpoint(
        "GET", 
        "/health",
        expected_status=200,
        description="Main API Health Check"
    )
    
    # Test 2: Try different possible prorating endpoint paths
    possible_paths = [
        "/api/v1/prorating/health",
        "/prorating/health",
        "/api/v1/prorating/api/v1/prorating/health",  # Double prefix from test script
        "/api/prorating/health",
    ]
    
    for path in possible_paths:
        test_endpoint(
            "GET",
            path,
            expected_status=200,
            description=f"Prorating Health Check - Path: {path}"
        )
    
    # Test 3: Check if prorating router is registered
    test_endpoint(
        "GET",
        "/openapi.json",
        expected_status=200,
        description="OpenAPI Schema (to check registered routes)"
    )
    
    # Test 4: Try to access a prorating process endpoint
    test_data = {
        "billing_period": {
            "start_date": "2025-08-01",
            "end_date": "2025-08-31"
        },
        "facility_id": "test_facility",
        "allocation_method": "equal_distribution"
    }
    
    test_endpoint(
        "POST",
        "/api/v1/prorating/process/test_doc_123",
        data=test_data,
        expected_status=200,
        description="Process Document Prorating"
    )
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nFailed Tests:")
    for result in results:
        if not result["success"]:
            print(f"  - {result['description']}: {result['method']} {result['endpoint']} "
                  f"(Got {result['status']}, Expected {result['expected']})")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Check if any prorating endpoints work
    prorating_works = any(r["success"] and "prorating" in r["endpoint"].lower() for r in results)
    
    if not prorating_works:
        print("❌ No prorating endpoints are accessible")
        print("\nNext Steps:")
        print("1. Check if prorating router is imported in ehs_extraction_api.py")
        print("2. Verify phase1_integration.setup_phase1_features() is called")
        print("3. Check logs for prorating service initialization errors")
        print("4. Ensure correct URL prefix configuration")
    else:
        print("✅ Some prorating endpoints are accessible")
        print("\nWorking endpoint pattern detected - update test scripts accordingly")
    
    # Save detailed results
    with open("prorating_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: prorating_test_results.json")

if __name__ == "__main__":
    main()