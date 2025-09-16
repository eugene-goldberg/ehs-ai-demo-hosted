#!/usr/bin/env python3
"""
Test script for Context Retriever Service

This script tests the context retriever with real Neo4j data.
"""

import os
import sys
import logging
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.services.context_retriever import ContextRetriever, ContextRequest, IntentType, get_context_for_intent
    print("Successfully imported context retriever components")
except ImportError as e:
    print(f"Error importing context retriever: {e}")
    sys.exit(1)

def test_electricity_context():
    """Test electricity consumption context retrieval"""
    print("\n" + "="*60)
    print("TESTING ELECTRICITY CONSUMPTION CONTEXT RETRIEVAL")
    print("="*60)
    
    try:
        with ContextRetriever() as retriever:
            # Test with Algonquin site filter
            request = ContextRequest(
                intent_type=IntentType.ELECTRICITY_CONSUMPTION,
                site_filter="algonquin",
                start_date="2024-01-01",
                end_date="2024-12-31"
            )
            
            print(f"Retrieving context for: {request.intent_type.value}")
            print(f"Site filter: {request.site_filter}")
            print(f"Date range: {request.start_date} to {request.end_date}")
            print("-" * 40)
            
            context = retriever.retrieve_context(request)
            
            print("CONTEXT DATA RETRIEVED:")
            print(f"Intent Type: {context.intent_type}")
            print(f"Summary: {context.summary}")
            print(f"Raw Data Points: {len(context.raw_data)}")
            print(f"Aggregated Metrics Keys: {list(context.aggregated_metrics.keys())}")
            print(f"Trends Keys: {list(context.trends.keys())}")
            print(f"Metadata: {context.metadata}")
            
            print("\n" + "-" * 40)
            print("FORMATTED LLM CONTEXT:")
            print("-" * 40)
            print(context.to_llm_context())
            
            return True
            
    except Exception as e:
        print(f"Error in electricity context test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_function():
    """Test the convenience function"""
    print("\n" + "="*60)
    print("TESTING CONVENIENCE FUNCTION")
    print("="*60)
    
    try:
        context_str = get_context_for_intent(
            intent_type="electricity_consumption",
            site_filter="algonquin",
            start_date="2024-06-01",
            end_date="2024-08-31"
        )
        
        print("CONVENIENCE FUNCTION RESULT:")
        print("-" * 40)
        print(context_str)
        
        return True
        
    except Exception as e:
        print(f"Error in convenience function test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stub_implementations():
    """Test stub implementations for other intent types"""
    print("\n" + "="*60)
    print("TESTING STUB IMPLEMENTATIONS")
    print("="*60)
    
    stub_intents = [
        IntentType.WATER_CONSUMPTION,
        IntentType.WASTE_GENERATION,
        IntentType.ENVIRONMENTAL_COMPLIANCE,
        IntentType.SAFETY_INCIDENTS,
        IntentType.OPERATIONAL_METRICS
    ]
    
    try:
        with ContextRetriever() as retriever:
            for intent in stub_intents:
                print(f"\nTesting {intent.value}:")
                request = ContextRequest(intent_type=intent, site_filter="algonquin")
                context = retriever.retrieve_context(request)
                print(f"  Summary: {context.summary}")
                print(f"  Status: {context.metadata.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"Error in stub implementations test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neo4j_connection():
    """Test Neo4j connection"""
    print("\n" + "="*60)
    print("TESTING NEO4J CONNECTION")
    print("="*60)
    
    try:
        retriever = ContextRetriever()
        if retriever.neo4j_client.test_connection():
            print("✓ Neo4j connection successful")
            
            # Test basic query
            result = retriever.neo4j_client.execute_read_query("RETURN 1 as test")
            if result and result[0].get('test') == 1:
                print("✓ Basic query test successful")
            else:
                print("✗ Basic query test failed")
                return False
                
            # Test site query
            sites_query = "MATCH (s:Site) RETURN s.id as site_id, s.name as site_name LIMIT 5"
            sites = retriever.neo4j_client.execute_read_query(sites_query)
            print(f"✓ Found {len(sites)} sites in database")
            for site in sites:
                print(f"  - {site['site_id']}: {site['site_name']}")
            
            retriever.close()
            return True
        else:
            print("✗ Neo4j connection failed")
            return False
            
    except Exception as e:
        print(f"Error in Neo4j connection test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("CONTEXT RETRIEVER SERVICE TESTS")
    print("=" * 60)
    
    tests = [
        ("Neo4j Connection", test_neo4j_connection),
        ("Electricity Context", test_electricity_context),
        ("Convenience Function", test_convenience_function),
        ("Stub Implementations", test_stub_implementations),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n\nRunning {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Context Retriever is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
