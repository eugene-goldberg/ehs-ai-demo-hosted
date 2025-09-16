#!/usr/bin/env python3
"""
Test script for risk assessment context retrieval functionality
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add the services directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from context_retriever import ContextRetriever, get_context_for_intent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_risk_assessment_retrieval():
    """Test risk assessment data retrieval for both supported sites"""
    
    logger.info("Starting risk assessment retrieval tests...")
    
    # Test cases
    test_cases = [
        {"site": "algonquin_il", "name": "Algonquin IL"},
        {"site": "houston_tx", "name": "Houston TX"},
        {"site": "algonquin", "name": "Algonquin (mapped)"},
        {"site": "houston", "name": "Houston (mapped)"},
        {"site": "invalid_site", "name": "Invalid Site"}
    ]
    
    results = {}
    
    for test_case in test_cases:
        site = test_case["site"]
        name = test_case["name"]
        
        logger.info(f"Testing risk assessment retrieval for {name} ({site})...")
        
        try:
            # Test using get_context_for_intent function
            context_json = get_context_for_intent("risk_assessment", site)
            context = json.loads(context_json)
            
            results[site] = {
                "status": "success",
                "site": context.get("site"),
                "record_count": context.get("record_count", 0),
                "has_error": "error" in context,
                "error_message": context.get("error"),
                "risk_summary": context.get("risk_summary"),
                "assessments_count": len(context.get("assessments", []))
            }
            
            logger.info(f"✓ {name}: {context.get('record_count', 0)} records found")
            
            if "error" in context:
                logger.warning(f"  Error: {context['error']}")
            elif context.get("record_count", 0) > 0:
                risk_summary = context.get("risk_summary", {})
                logger.info(f"  Risk levels: {risk_summary.get('risk_level_breakdown', {})}")
                logger.info(f"  Categories: {list(risk_summary.get('category_breakdown', {}).keys())}")
                
        except Exception as e:
            results[site] = {
                "status": "error",
                "error": str(e)
            }
            logger.error(f"✗ {name}: Error - {e}")
    
    return results

def test_direct_context_retriever():
    """Test using ContextRetriever directly"""
    
    logger.info("Testing direct ContextRetriever usage...")
    
    retriever = ContextRetriever()
    try:
        # Test Algonquin IL
        logger.info("Testing Algonquin IL directly...")
        algonquin_context = retriever.get_risk_assessment_context("algonquin_il")
        logger.info(f"Algonquin direct: {algonquin_context.get('record_count', 0)} records")
        
        # Test Houston TX
        logger.info("Testing Houston TX directly...")
        houston_context = retriever.get_risk_assessment_context("houston_tx")
        logger.info(f"Houston direct: {houston_context.get('record_count', 0)} records")
        
        return {
            "algonquin_direct": algonquin_context,
            "houston_direct": houston_context
        }
        
    except Exception as e:
        logger.error(f"Direct test error: {e}")
        return {"error": str(e)}
    finally:
        retriever.close()

def main():
    """Run all tests"""
    
    print("="*80)
    print("RISK ASSESSMENT CONTEXT RETRIEVAL TEST")
    print("="*80)
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Test 1: Using get_context_for_intent function
    print("Test 1: Using get_context_for_intent function")
    print("-"*50)
    intent_results = test_risk_assessment_retrieval()
    
    print()
    print("Test 2: Using ContextRetriever directly")
    print("-"*50)
    direct_results = test_direct_context_retriever()
    
    print()
    print("TEST SUMMARY")
    print("="*50)
    
    # Summary of intent results
    for site, result in intent_results.items():
        status = result.get("status", "unknown")
        record_count = result.get("record_count", 0)
        has_error = result.get("has_error", False)
        
        print(f"Site {site}: {status} - {record_count} records" + (" (with error)" if has_error else ""))
    
    # Check if supported sites have data
    algonquin_success = intent_results.get("algonquin_il", {}).get("record_count", 0) > 0
    houston_success = intent_results.get("houston_tx", {}).get("record_count", 0) > 0
    invalid_blocked = intent_results.get("invalid_site", {}).get("has_error", False)
    
    print()
    if algonquin_success and houston_success and invalid_blocked:
        print("✓ ALL TESTS PASSED!")
        print("  - Algonquin IL: Risk assessment data retrieved successfully")
        print("  - Houston TX: Risk assessment data retrieved successfully")
        print("  - Invalid sites: Properly blocked")
    else:
        print("✗ SOME TESTS FAILED:")
        if not algonquin_success:
            print("  - Algonquin IL: No risk assessment data found")
        if not houston_success:
            print("  - Houston TX: No risk assessment data found")
        if not invalid_blocked:
            print("  - Invalid sites: Not properly blocked")
    
    print(f"\nTest completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
