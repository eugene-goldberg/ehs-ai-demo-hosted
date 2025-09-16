#!/usr/bin/env python3
"""
Comprehensive test for recommendations retrieval functionality
Tests the get_recommendations_context() method for Algonquin IL and Houston TX only
"""

import sys
import os
sys.path.append('/home/azureuser/dev/ehs-ai-demo/data-foundation/backend/src')

from services.context_retriever import ContextRetriever, get_context_for_intent
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/recommendations_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_recommendations_algonquin_il():
    """Test recommendations retrieval for Algonquin IL"""
    logger.info("=== Testing Recommendations for Algonquin IL ===")
    
    retriever = ContextRetriever()
    try:
        # Test basic retrieval
        result = retriever.get_recommendations_context("algonquin_il")
        logger.info(f"Algonquin IL basic test - Record count: {result.get('record_count', 0)}")
        
        if result.get('error'):
            logger.error(f"Error in Algonquin IL basic test: {result['error']}")
            return False
        
        # Test with category filter
        result_electricity = retriever.get_recommendations_context("algonquin_il", category="electricity")
        logger.info(f"Algonquin IL electricity filter - Record count: {result_electricity.get('record_count', 0)}")
        
        # Test with priority filter
        result_high_priority = retriever.get_recommendations_context("algonquin_il", priority="medium")
        logger.info(f"Algonquin IL medium priority filter - Record count: {result_high_priority.get('record_count', 0)}")
        
        # Validate structure
        if result.get('recommendations'):
            first_rec = result['recommendations'][0]
            required_fields = ['recommendation_id', 'title', 'category', 'priority', 'action_description']
            missing_fields = [field for field in required_fields if field not in first_rec or first_rec[field] is None]
            if missing_fields:
                logger.warning(f"Missing fields in Algonquin IL recommendation: {missing_fields}")
            else:
                logger.info("Algonquin IL recommendation structure validation: PASSED")
        
        return True
        
    except Exception as e:
        logger.error(f"Exception in Algonquin IL test: {e}")
        return False
    finally:
        retriever.close()

def test_recommendations_houston_tx():
    """Test recommendations retrieval for Houston TX"""
    logger.info("=== Testing Recommendations for Houston TX ===")
    
    retriever = ContextRetriever()
    try:
        # Test basic retrieval
        result = retriever.get_recommendations_context("houston_tx")
        logger.info(f"Houston TX basic test - Record count: {result.get('record_count', 0)}")
        
        if result.get('error'):
            logger.error(f"Error in Houston TX basic test: {result['error']}")
            return False
        
        # Test with category filter
        result_water = retriever.get_recommendations_context("houston_tx", category="water")
        logger.info(f"Houston TX water filter - Record count: {result_water.get('record_count', 0)}")
        
        # Test with priority filter
        result_high_priority = retriever.get_recommendations_context("houston_tx", priority="high")
        logger.info(f"Houston TX high priority filter - Record count: {result_high_priority.get('record_count', 0)}")
        
        # Validate structure
        if result.get('recommendations'):
            first_rec = result['recommendations'][0]
            required_fields = ['recommendation_id', 'title', 'category', 'priority', 'action_description']
            missing_fields = [field for field in required_fields if field not in first_rec or first_rec[field] is None]
            if missing_fields:
                logger.warning(f"Missing fields in Houston TX recommendation: {missing_fields}")
            else:
                logger.info("Houston TX recommendation structure validation: PASSED")
        
        return True
        
    except Exception as e:
        logger.error(f"Exception in Houston TX test: {e}")
        return False
    finally:
        retriever.close()

def test_unsupported_sites():
    """Test error handling for unsupported sites"""
    logger.info("=== Testing Error Handling for Unsupported Sites ===")
    
    retriever = ContextRetriever()
    try:
        unsupported_sites = ["california", "new_york", "texas", "facility_xyz"]
        
        for site in unsupported_sites:
            result = retriever.get_recommendations_context(site)
            if result.get('error') and 'only available for Algonquin IL and Houston TX' in result['error']:
                logger.info(f"Unsupported site '{site}' correctly rejected")
            else:
                logger.error(f"Unsupported site '{site}' was not properly rejected")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Exception in unsupported sites test: {e}")
        return False
    finally:
        retriever.close()

def test_get_context_for_intent():
    """Test the convenience function get_context_for_intent"""
    logger.info("=== Testing get_context_for_intent Function ===")
    
    try:
        # Test Algonquin IL
        result_json = get_context_for_intent("recommendations", "algonquin_il")
        result = json.loads(result_json)
        logger.info(f"get_context_for_intent Algonquin IL - Record count: {result.get('record_count', 0)}")
        
        # Test Houston TX
        result_json = get_context_for_intent("recommendations", "houston_tx")
        result = json.loads(result_json)
        logger.info(f"get_context_for_intent Houston TX - Record count: {result.get('record_count', 0)}")
        
        # Test unsupported site
        result_json = get_context_for_intent("recommendations", "unsupported_site")
        result = json.loads(result_json)
        if result.get('error'):
            logger.info("get_context_for_intent properly handles unsupported sites")
        else:
            logger.error("get_context_for_intent did not properly reject unsupported site")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Exception in get_context_for_intent test: {e}")
        return False

def test_site_name_variations():
    """Test various site name variations"""
    logger.info("=== Testing Site Name Variations ===")
    
    retriever = ContextRetriever()
    try:
        # Test Algonquin variations
        algonquin_variations = ["algonquin_il", "algonquin_illinois", "algonquin", "Algonquin_IL", "ALGONQUIN"]
        for variation in algonquin_variations:
            result = retriever.get_recommendations_context(variation)
            if result.get('error') and 'only available for' in result['error']:
                logger.error(f"Algonquin variation '{variation}' was incorrectly rejected")
                return False
            elif result.get('record_count', 0) > 0:
                logger.info(f"Algonquin variation '{variation}' correctly mapped to algonquin_il")
        
        # Test Houston variations  
        houston_variations = ["houston_tx", "houston_texas", "houston", "Houston_TX", "HOUSTON"]
        for variation in houston_variations:
            result = retriever.get_recommendations_context(variation)
            if result.get('error') and 'only available for' in result['error']:
                logger.error(f"Houston variation '{variation}' was incorrectly rejected")
                return False
            elif result.get('record_count', 0) > 0:
                logger.info(f"Houston variation '{variation}' correctly mapped to houston_tx")
        
        return True
        
    except Exception as e:
        logger.error(f"Exception in site name variations test: {e}")
        return False
    finally:
        retriever.close()

def main():
    """Run all tests"""
    logger.info("Starting comprehensive recommendations functionality test")
    
    tests = [
        ("Algonquin IL Recommendations", test_recommendations_algonquin_il),
        ("Houston TX Recommendations", test_recommendations_houston_tx),
        ("Unsupported Sites Error Handling", test_unsupported_sites),
        ("get_context_for_intent Function", test_get_context_for_intent),
        ("Site Name Variations", test_site_name_variations)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"Test '{test_name}': {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Recommendations functionality is working correctly.")
    else:
        logger.error("❌ SOME TESTS FAILED! Please check the logs and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
