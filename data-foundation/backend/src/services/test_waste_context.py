#!/usr/bin/env python3
"""
Test script for waste generation context retrieval
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.context_retriever import ContextRetriever, get_context_for_intent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_algonquin_waste_context():
    """Test waste context retrieval for Algonquin IL"""
    logger.info("Testing Algonquin IL waste context retrieval...")
    
    retriever = ContextRetriever()
    
    try:
        # Test basic retrieval
        context = retriever.get_waste_context('algonquin_il')
        
        assert context['record_count'] > 0, f"No records found for Algonquin IL: {context}"
        assert context['site_id'] == 'algonquin_il', f"Unexpected site ID: {context['site_id']}"
        assert 'aggregates' in context, "Missing aggregates in context"
        assert 'recent_data' in context, "Missing recent_data in context"
        
        # Check aggregate data
        aggregates = context['aggregates']
        assert aggregates['total_pounds'] > 0, "Total pounds should be greater than 0"
        assert 'waste_types_breakdown' in aggregates, "Missing waste types breakdown"
        
        # Check recent data
        recent_data = context['recent_data']
        assert len(recent_data) > 0, "No recent data found"
        
        # Verify data structure
        first_record = recent_data[0]
        required_fields = ['date', 'amount_pounds', 'waste_type', 'disposal_method']
        for field in required_fields:
            assert field in first_record, f"Missing field {field} in recent data"
        
        logger.info(f"Algonquin IL test PASSED - Found {context['record_count']} records")
        logger.info(f"Total waste: {aggregates['total_pounds']} pounds")
        logger.info(f"Total cost: USD {aggregates['total_cost']:.2f}")
        logger.info(f"Waste types: {list(aggregates['waste_types_breakdown'].keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Algonquin IL test FAILED: {e}")
        return False
    finally:
        retriever.close()

def test_houston_waste_context():
    """Test waste context retrieval for Houston TX"""
    logger.info("Testing Houston TX waste context retrieval...")
    
    retriever = ContextRetriever()
    
    try:
        # Test basic retrieval  
        context = retriever.get_waste_context('houston_tx')
        
        assert context['record_count'] > 0, f"No records found for Houston TX: {context}"
        assert context['site_id'] == 'houston_tx', f"Unexpected site ID: {context['site_id']}"
        assert 'aggregates' in context, "Missing aggregates in context"
        assert 'recent_data' in context, "Missing recent_data in context"
        
        # Check aggregate data
        aggregates = context['aggregates']
        assert aggregates['total_pounds'] > 0, "Total pounds should be greater than 0"
        assert 'waste_types_breakdown' in aggregates, "Missing waste types breakdown"
        
        # Check recent data
        recent_data = context['recent_data']
        assert len(recent_data) > 0, "No recent data found"
        
        logger.info(f"Houston TX test PASSED - Found {context['record_count']} records")
        logger.info(f"Total waste: {aggregates['total_pounds']} pounds")
        logger.info(f"Total cost: USD {aggregates['total_cost']:.2f}")
        logger.info(f"Waste types: {list(aggregates['waste_types_breakdown'].keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Houston TX test FAILED: {e}")
        return False
    finally:
        retriever.close()

def test_site_name_mapping():
    """Test site name mapping for various formats"""
    logger.info("Testing site name mapping...")
    
    retriever = ContextRetriever()
    
    try:
        # Test various site name formats
        test_cases = [
            ('houston', 'houston_tx'),
            ('houston_tx', 'houston_tx'),
            ('houston_texas', 'houston_tx'),
            ('algonquin', 'algonquin_il'),
            ('algonquin_il', 'algonquin_il'), 
            ('algonquin_illinois', 'algonquin_il')
        ]
        
        for input_site, expected_site in test_cases:
            context = retriever.get_waste_context(input_site)
            assert context['site_id'] == expected_site, f"Site mapping failed: {input_site} -> {context['site_id']} (expected {expected_site})"
            assert context['record_count'] > 0, f"No records for site {input_site}"
        
        logger.info("Site name mapping test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Site name mapping test FAILED: {e}")
        return False
    finally:
        retriever.close()

def test_unsupported_site():
    """Test that unsupported sites are properly rejected"""
    logger.info("Testing unsupported site rejection...")
    
    retriever = ContextRetriever()
    
    try:
        # Test unsupported site
        context = retriever.get_waste_context('some_other_site')
        
        assert 'error' in context, "Should return error for unsupported site"
        assert context['record_count'] == 0, "Should have 0 records for unsupported site"
        assert "only available for Algonquin IL and Houston TX" in context['error'], "Error message should mention supported sites"
        
        logger.info("Unsupported site rejection test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Unsupported site rejection test FAILED: {e}")
        return False
    finally:
        retriever.close()

def test_get_context_for_intent():
    """Test the convenience function for waste_generation intent"""
    logger.info("Testing get_context_for_intent function...")
    
    try:
        # Test Algonquin IL
        context_json = get_context_for_intent('waste_generation', 'algonquin_il')
        context = json.loads(context_json)
        
        assert context['record_count'] > 0, "No records found via intent function"
        assert context['site_id'] == 'algonquin_il', "Unexpected site ID via intent function"
        
        # Test Houston TX
        context_json = get_context_for_intent('waste_generation', 'houston_tx')
        context = json.loads(context_json)
        
        assert context['record_count'] > 0, "No records found for Houston TX via intent function"
        assert context['site_id'] == 'houston_tx', "Unexpected site ID for Houston TX via intent function"
        
        logger.info("get_context_for_intent test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"get_context_for_intent test FAILED: {e}")
        return False

def main():
    """Run all waste context tests"""
    logger.info("Starting waste generation context tests...")
    
    tests = [
        test_algonquin_waste_context,
        test_houston_waste_context,
        test_site_name_mapping,
        test_unsupported_site,
        test_get_context_for_intent
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("All waste generation tests PASSED!")
        return 0
    else:
        logger.error(f"{failed} tests FAILED!")
        return 1

if __name__ == '__main__':
    exit(main())
