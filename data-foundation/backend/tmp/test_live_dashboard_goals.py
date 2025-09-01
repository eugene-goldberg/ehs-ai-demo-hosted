#!/usr/bin/env python3
"""
Live Dashboard Goals Test Script

This script tests the actual running dashboard API to verify that annual goals
are properly integrated and displayed correctly.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any


def test_live_dashboard_api():
    """Test the live dashboard API for goals integration"""
    
    print("🔍 Testing Live Dashboard API for Goals Integration")
    print("=" * 60)
    
    api_url = "http://localhost:8000/api/v2/executive-dashboard"
    
    try:
        # Make request to dashboard API
        print(f"Making request to: {api_url}")
        response = requests.get(api_url, timeout=10)
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for goals-related sections
            goals_sections_found = []
            
            # Check for environmental_goals section
            if 'environmental_goals' in data:
                print("✅ Found 'environmental_goals' section")
                goals_sections_found.append('environmental_goals')
                
                env_goals = data['environmental_goals']
                print(f"   - Section title: {env_goals.get('section_title', 'N/A')}")
                print(f"   - Section priority: {env_goals.get('section_priority', 'N/A')}")
                print(f"   - Display position: {env_goals.get('display_position', 'N/A')}")
                
                # Check goals data
                if 'goals_data' in env_goals:
                    goals_data = env_goals['goals_data']
                    if 'error' in goals_data:
                        print(f"   ⚠️  Error in goals data: {goals_data['error']}")
                    else:
                        print("   ✅ Goals data appears to be valid")
            
            # Check for annual_goals section
            if 'annual_goals' in data:
                print("✅ Found 'annual_goals' section")
                goals_sections_found.append('annual_goals')
                
            # Check for goals in other sections
            for key, value in data.items():
                if isinstance(value, dict):
                    if any(goal_keyword in str(value).lower() for goal_keyword in ['goal', 'target', 'reduction']):
                        if key not in goals_sections_found:
                            print(f"✅ Found goals-related content in '{key}' section")
                            goals_sections_found.append(key)
            
            # Check for site-specific data
            algonquin_found = False
            houston_found = False
            
            def check_for_sites(obj, path=""):
                nonlocal algonquin_found, houston_found
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if any(site in str(v).lower() for site in ['algonquin', 'illinois']):
                            algonquin_found = True
                        if any(site in str(v).lower() for site in ['houston', 'texas']):
                            houston_found = True
                        check_for_sites(v, f"{path}.{k}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_for_sites(item, f"{path}[{i}]")
            
            check_for_sites(data)
            
            # Summary
            print("\n📊 ANALYSIS RESULTS:")
            print("-" * 40)
            print(f"Goals sections found: {len(goals_sections_found)}")
            for section in goals_sections_found:
                print(f"  • {section}")
            
            print(f"\nSite coverage:")
            print(f"  • Algonquin/Illinois: {'✅ Found' if algonquin_found else '❌ Not found'}")
            print(f"  • Houston/Texas: {'✅ Found' if houston_found else '❌ Not found'}")
            
            # Check response structure
            print(f"\nResponse structure:")
            print(f"  • Total sections: {len(data)}")
            print(f"  • Response size: {len(str(data))} characters")
            
            # Key sections
            key_sections = ['dashboard_info', 'environmental_goals', 'annual_goals', 'summary', 'kpis']
            for section in key_sections:
                status = "✅" if section in data else "❌"
                print(f"  • {section}: {status}")
            
            return True
            
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to API: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_goals_api():
    """Test the goals API endpoint"""
    
    print("\n🎯 Testing Goals API Endpoint")
    print("=" * 40)
    
    goals_api_url = "http://localhost:8000/api/goals/"
    
    try:
        response = requests.get(goals_api_url, timeout=10)
        
        print(f"Goals API Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Goals API Response:")
            print(f"   - Total goals: {data.get('total_goals', 'N/A')}")
            print(f"   - Sites: {data.get('sites', [])}")
            print(f"   - Categories: {data.get('categories', [])}")
            
            if 'goals' in data:
                print(f"   - Goals data: {len(data['goals'])} items")
                
                # Show sample goals
                for i, goal in enumerate(data['goals'][:3]):  # Show first 3
                    print(f"      Goal {i+1}: {goal.get('site', 'N/A')} - {goal.get('category', 'N/A')} ({goal.get('reduction_percentage', 'N/A')}%)")
            
            return True
        else:
            print(f"❌ Goals API failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to Goals API: {e}")
        return False


def main():
    """Main test execution"""
    print("🧪 Live Dashboard Goals Integration Test")
    print("Generated:", datetime.now())
    print("=" * 60)
    
    # Test dashboard API
    dashboard_success = test_live_dashboard_api()
    
    # Test goals API
    goals_success = test_goals_api()
    
    # Summary
    print(f"\n📋 TEST SUMMARY:")
    print("=" * 30)
    print(f"Dashboard API: {'✅ PASS' if dashboard_success else '❌ FAIL'}")
    print(f"Goals API: {'✅ PASS' if goals_success else '❌ FAIL'}")
    
    overall_success = dashboard_success and goals_success
    print(f"Overall: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    if overall_success:
        print("\n🎉 All tests passed! Dashboard appears to have goals integration.")
    else:
        print("\n⚠️  Some tests failed. Check the API implementations.")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
