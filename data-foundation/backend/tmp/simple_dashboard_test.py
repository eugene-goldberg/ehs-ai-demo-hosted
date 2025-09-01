#!/usr/bin/env python3
"""
Simple test script to check the dashboard endpoint
Tests basic HTTP request and response structure
"""

import requests
import json
import sys
from datetime import datetime

def test_dashboard_endpoint():
    """Test the dashboard endpoint for basic functionality"""
    
    # Dashboard endpoint URL
    url = "http://localhost:8000/dashboard"
    
    print(f"[{datetime.now()}] Testing dashboard endpoint: {url}")
    print("-" * 60)
    
    try:
        # Make the HTTP request
        print("Making HTTP GET request...")
        response = requests.get(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print("-" * 60)
        
        # Check if request was successful
        if response.status_code == 200:
            print("✓ Request successful (200 OK)")
            
            # Try to parse JSON response
            try:
                data = response.json()
                print("✓ Response is valid JSON")
                print(f"Response type: {type(data)}")
                
                # Print the full response for inspection
                print("\nFull Response:")
                print(json.dumps(data, indent=2))
                print("-" * 60)
                
                # Check basic structure
                print("\nStructure Analysis:")
                if isinstance(data, dict):
                    print(f"✓ Response is a dictionary with {len(data)} keys")
                    print(f"Keys: {list(data.keys())}")
                    
                    # Check for goals specifically
                    if 'goals' in data:
                        goals = data['goals']
                        print(f"✓ Goals found in response")
                        print(f"Goals type: {type(goals)}")
                        if isinstance(goals, list):
                            print(f"Goals count: {len(goals)}")
                            if goals:
                                print("Sample goal structure:")
                                print(json.dumps(goals[0], indent=2))
                        elif isinstance(goals, dict):
                            print(f"Goals dictionary keys: {list(goals.keys())}")
                    else:
                        print("⚠ No 'goals' key found in response")
                        
                elif isinstance(data, list):
                    print(f"✓ Response is a list with {len(data)} items")
                    if data:
                        print("First item structure:")
                        print(json.dumps(data[0], indent=2))
                else:
                    print(f"⚠ Unexpected response type: {type(data)}")
                    
            except json.JSONDecodeError as e:
                print(f"✗ Failed to parse JSON: {e}")
                print("Raw response content:")
                print(response.text[:500])  # First 500 chars
                
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print("Response content:")
            print(response.text[:500])
            
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed - is the server running on localhost:8000?")
        return False
        
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    print(f"\n[{datetime.now()}] Test completed")
    return response.status_code == 200 if 'response' in locals() else False

if __name__ == "__main__":
    success = test_dashboard_endpoint()
    sys.exit(0 if success else 1)