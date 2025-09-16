#!/usr/bin/env python3
"""
Test script for CO2 goals context retrieval functionality
Tests the get_co2_goals_context method for Algonquin IL and Houston TX sites
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to Python path
sys.path.append('src')

from services.context_retriever import ContextRetriever, get_context_for_intent
from database.neo4j_client import Neo4jClient

def test_co2_goals_retrieval():
    """Test CO2 goals retrieval for both supported sites"""
    
    print("=" * 60)
    print("TESTING CO2 GOALS CONTEXT RETRIEVAL")
    print("=" * 60)
    
    # Test sites
    test_sites = ['algonquin_il', 'houston_tx']
    
    # Test cases
    test_cases = [
        # Test with exact site ID
        {'site': 'algonquin_il', 'description': 'Algonquin IL with exact site ID'},
        {'site': 'houston_tx', 'description': 'Houston TX with exact site ID'},
        
        # Test with alternate site names
        {'site': 'algonquin', 'description': 'Algonquin with short name'},
        {'site': 'houston', 'description': 'Houston with short name'},
        {'site': 'algonquin_illinois', 'description': 'Algonquin with full name'},
        {'site': 'houston_texas', 'description': 'Houston with full name'},
        
        # Test with unsupported site (should return error)
        {'site': 'unsupported_site', 'description': 'Unsupported site (should fail)'},
        
        # Test with date range
        {'site': 'algonquin_il', 'start_date': '2025-01-01', 'end_date': '2025-12-31', 'description': 'Algonquin IL with date range'},
        {'site': 'houston_tx', 'start_date': '2025-01-01', 'end_date': '2025-12-31', 'description': 'Houston TX with date range'},
        
        # Test without site (should fail gracefully)
        {'site': None, 'description': 'No site specified'},
    ]
    
    # Connect to Neo4j
    print("Connecting to Neo4j...")
    client = Neo4jClient()
    try:
        client.connect()
        print("✓ Connected to Neo4j successfully")
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        return False
    
    retriever = ContextRetriever(client)
    all_tests_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['description']} ---")
        
        try:
            # Get parameters
            site = test_case.get('site')
            start_date = test_case.get('start_date')
            end_date = test_case.get('end_date')
            
            print(f"Parameters: site='{site}', start_date='{start_date}', end_date='{end_date}'")
            
            # Call the method
            result = retriever.get_co2_goals_context(site, start_date, end_date)
            
            # Analyze result
            if 'error' in result:
                if site in ['unsupported_site', None]:
                    print(f"✓ Expected error for unsupported site: {result['error']}")
                else:
                    print(f"✗ Unexpected error: {result['error']}")
                    all_tests_passed = False
            else:
                print(f"✓ Success! Record count: {result.get('record_count', 0)}")
                
                # Print summary if available
                if 'summary' in result:
                    summary = result['summary']
                    print(f"  - CO2 goals: {summary.get('co2_goals_count', 0)}")
                    print(f"  - Electricity goals: {summary.get('electricity_goals_count', 0)}")
                    print(f"  - Environmental targets: {summary.get('environmental_targets_count', 0)}")
                    print(f"  - Total CO2 reduction target: {summary.get('total_co2_reduction_target', 0)}")
                
                # Print some sample data
                if result.get('co2_goals'):
                    print(f"  - Sample CO2 goal: {result['co2_goals'][0].get('description', 'N/A')[:60]}...")
                
                if result.get('environmental_targets'):
                    print(f"  - Sample env target: {result['environmental_targets'][0].get('description', 'N/A')[:60]}...")
            
        except Exception as e:
            print(f"✗ Exception occurred: {e}")
            all_tests_passed = False
    
    print(f"\n--- Testing get_context_for_intent function ---")
    
    # Test the convenience function
    for site in ['algonquin_il', 'houston_tx']:
        try:
            print(f"\nTesting get_context_for_intent for {site}...")
            context_json = get_context_for_intent('co2_goals', site, neo4j_client=client)
            context = json.loads(context_json)
            
            if 'error' in context:
                print(f"✗ Error in get_context_for_intent: {context['error']}")
                all_tests_passed = False
            else:
                print(f"✓ get_context_for_intent success! Record count: {context.get('record_count', 0)}")
                
        except Exception as e:
            print(f"✗ Exception in get_context_for_intent: {e}")
            all_tests_passed = False
    
    # Close connections
    retriever.close()
    client.close()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! CO2 goals retrieval is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please review the errors above.")
    print("=" * 60)
    
    return all_tests_passed

def test_specific_queries():
    """Test specific queries to understand the data structure better"""
    
    print("\n" + "=" * 60)
    print("DETAILED DATA STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Connect to Neo4j
    client = Neo4jClient()
    try:
        client.connect()
        print("✓ Connected to Neo4j for detailed analysis")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    # Query all CO2-related data for supported sites
    queries = [
        {
            'name': 'All Goals for supported sites',
            'query': '''
            MATCH (g:Goal) 
            WHERE g.site_id IN ["algonquin_il", "houston_tx"]
            RETURN g.site_id, g.category, g.description, g.unit, g.target_value
            ORDER BY g.site_id, g.category
            '''
        },
        {
            'name': 'All Environmental Targets for supported sites',
            'query': '''
            MATCH (et:EnvironmentalTarget)
            WHERE et.site_id IN ["algonquin_il", "houston_tx"]
            RETURN et.site_id, et.target_type, et.description, et.target_unit, et.target_value, et.status
            ORDER BY et.site_id, et.target_type
            '''
        },
        {
            'name': 'CO2-specific Goals',
            'query': '''
            MATCH (g:Goal)
            WHERE g.site_id IN ["algonquin_il", "houston_tx"]
            AND (toLower(g.description) CONTAINS 'co2' OR toLower(g.description) CONTAINS 'carbon' 
                 OR toLower(g.description) CONTAINS 'emission' OR g.unit CONTAINS 'CO2')
            RETURN g.site_id, g.category, g.description, g.unit, g.target_value
            '''
        }
    ]
    
    for query_info in queries:
        print(f"\n--- {query_info['name']} ---")
        try:
            result = client.execute_read_query(query_info['query'])
            if result:
                for record in result:
                    record_dict = dict(record)
                    print(f"  {json.dumps(record_dict, default=str, indent=2)}")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Error: {e}")
    
    client.close()

if __name__ == "__main__":
    print(f"Starting CO2 goals context retrieval tests at {datetime.now()}")
    
    # Test the main functionality
    success = test_co2_goals_retrieval()
    
    # Test specific queries for better understanding
    test_specific_queries()
    
    print(f"\nTest completed at {datetime.now()}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
