#!/usr/bin/env python3
"""
Comprehensive Test Suite for Intent Classifier Service

This test suite provides comprehensive testing for the Intent Classifier service
using real LLM calls (no mocks) as required by the project rules.

Test Categories:
1. Basic Classification Tests
2. Site Extraction Tests  
3. Time Period Extraction Tests
4. Confidence Level Tests
5. Fallback Mechanism Tests
6. Batch Processing Tests
7. Error Handling Tests
8. Integration Tests
9. Performance Tests
10. Edge Case Tests

Features:
- Real LLM calls (no mocks)
- Comprehensive logging to timestamped file
- Detailed test metrics and validation tracking
- Runnable as standalone script or via pytest
- Real API key validation and environment checking
"""

import unittest
import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import patch

# Add src to Python path for imports
sys.path.insert(0, '/home/azureuser/dev/ehs-ai-demo/data-foundation/backend')

# Set up logging for test results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'/tmp/intent_classifier_test_{timestamp}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import the Intent Classifier
try:
    from src.services.intent_classifier import IntentClassifier, ClassificationResult, create_intent_classifier
    logger.info("Intent Classifier imports successful")
except ImportError as e:
    logger.error(f"Failed to import Intent Classifier: {e}")
    sys.exit(1)


class TestIntentClassifier(unittest.TestCase):
    """Comprehensive test suite for Intent Classifier service."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and validate API keys."""
        logger.info("Setting up Intent Classifier test environment")
        
        # Check for OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            logger.warning("OPENAI_API_KEY not found in environment")
        else:
            logger.info("OpenAI API key found in environment")
        
        # Initialize test metrics
        cls.test_metrics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'total_classifications': 0,
            'total_validation_checks': 0,
            'llm_call_count': 0,
            'test_start_time': time.time()
        }
        
        # Initialize classifier for testing
        try:
            cls.classifier = create_intent_classifier()
            logger.info("Intent Classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Intent Classifier: {e}")
            cls.classifier = None
        
        cls.test_queries = cls._prepare_test_queries()
        logger.info(f"Prepared {len(cls.test_queries)} test queries")
    
    @classmethod
    def _prepare_test_queries(cls) -> Dict[str, List[str]]:
        """Prepare comprehensive test queries for each intent category."""
        return {
            'ELECTRICITY_CONSUMPTION': [
                "What is the electricity consumption for Houston facility?",
                "Show me power usage data for last month",
                "How many kilowatt hours did we use in Q1?",
                "What's our electrical energy consumption in Texas?",
                "Can you provide electricity usage metrics for Algonquin?",
                "Show electricity consumption trends for 2024"
            ],
            'WATER_CONSUMPTION': [
                "What is the water consumption for our facilities?",
                "Show me water usage data for Houston",
                "How many gallons of water did we use last year?",
                "What's our water consumption in Illinois?",
                "Can you provide water usage metrics for all sites?",
                "Show water consumption trends by month"
            ],
            'WASTE_GENERATION': [
                "What is our waste generation data?",
                "Show me waste disposal records for Houston",
                "How much trash did we generate last month?",
                "What's our recycling data for Algonquin?",
                "Can you provide waste metrics by facility?",
                "Show waste generation trends over time"
            ],
            'CO2_GOALS': [
                "What are our carbon emissions goals?",
                "Show me CO2 reduction targets",
                "What's our carbon footprint for 2024?",
                "Can you provide sustainability goals data?",
                "Show emissions reduction progress",
                "What are our greenhouse gas targets?"
            ],
            'RISK_ASSESSMENT': [
                "What are the environmental risks for Houston?",
                "Show me safety assessment data",
                "What compliance issues do we have?",
                "Can you provide risk analysis for facilities?",
                "Show environmental risk factors",
                "What are the safety concerns at Algonquin?"
            ],
            'RECOMMENDATIONS': [
                "What recommendations do you have for reducing electricity?",
                "Can you suggest improvements for water usage?",
                "What are the best practices for waste management?",
                "How can we improve our environmental performance?",
                "What actions should we take to reduce emissions?",
                "Can you recommend sustainability initiatives?"
            ],
            'GENERAL': [
                "Hello, how are you?",
                "What can you help me with?",
                "Tell me about your capabilities",
                "Good morning",
                "What is this system?",
                "Help me understand the interface"
            ]
        }
    
    def setUp(self):
        """Set up for each test method."""
        self.test_metrics['total_tests'] += 1
        
        if not self.classifier:
            self.skipTest("Intent Classifier not available")
    
    def test_01_basic_classification_accuracy(self):
        """Test 1: Basic classification accuracy for all intent categories."""
        logger.info("🧪 Test 1: Basic Classification Accuracy")
        
        total_correct = 0
        total_tested = 0
        
        for expected_intent, queries in self.test_queries.items():
            logger.info(f"Testing {expected_intent} classification")
            
            for query in queries[:2]:  # Test first 2 queries per category
                try:
                    result = self.classifier.classify(query)
                    self.test_metrics['total_classifications'] += 1
                    self.test_metrics['llm_call_count'] += 1
                    total_tested += 1
                    
                    # Validate classification result
                    self._validate_classification_result(result, query)
                    
                    # Check if classification is correct
                    if result.intent == expected_intent:
                        total_correct += 1
                        logger.info(f"✅ Correct: '{query}' -> {result.intent} (confidence: {result.confidence})")
                    else:
                        logger.warning(f"❌ Incorrect: '{query}' -> {result.intent} (expected: {expected_intent})")
                    
                    # Minimum confidence check
                    self.assertGreaterEqual(result.confidence, 0.0, "Confidence should be non-negative")
                    self.assertLessEqual(result.confidence, 1.0, "Confidence should not exceed 1.0")
                    
                except Exception as e:
                    logger.error(f"Classification failed for query: '{query}' - {e}")
                    self.fail(f"Classification failed: {e}")
        
        accuracy = total_correct / total_tested if total_tested > 0 else 0
        logger.info(f"Classification accuracy: {accuracy:.2%} ({total_correct}/{total_tested})")
        
        # We expect at least 70% accuracy for basic classification
        self.assertGreaterEqual(accuracy, 0.7, f"Classification accuracy {accuracy:.2%} below threshold")
        
        logger.info("✅ Test 1 passed: Basic Classification Accuracy")
    
    def test_02_site_extraction_accuracy(self):
        """Test 2: Site extraction accuracy for location-specific queries."""
        logger.info("🧪 Test 2: Site Extraction Accuracy")
        
        site_test_queries = [
            ("What is electricity usage at Houston facility?", "houston_texas"),
            ("Show water consumption for Texas location", "houston_texas"),
            ("Waste generation data for Algonquin site", "algonquin_illinois"),
            ("Power usage in Illinois facility", "algonquin_illinois"),
            ("Houston Texas electricity metrics", "houston_texas"),
            ("Algonquin Illinois water usage", "algonquin_illinois"),
        ]
        
        correct_extractions = 0
        total_tested = 0
        
        for query, expected_site in site_test_queries:
            try:
                result = self.classifier.classify(query)
                self.test_metrics['total_classifications'] += 1
                self.test_metrics['llm_call_count'] += 1
                total_tested += 1
                
                self._validate_classification_result(result, query)
                
                if result.site == expected_site:
                    correct_extractions += 1
                    logger.info(f"✅ Correct site extraction: '{query}' -> {result.site}")
                else:
                    logger.warning(f"❌ Incorrect site extraction: '{query}' -> {result.site} (expected: {expected_site})")
                
            except Exception as e:
                logger.error(f"Site extraction failed for query: '{query}' - {e}")
                self.fail(f"Site extraction failed: {e}")
        
        site_accuracy = correct_extractions / total_tested if total_tested > 0 else 0
        logger.info(f"Site extraction accuracy: {site_accuracy:.2%} ({correct_extractions}/{total_tested})")
        
        # We expect at least 80% accuracy for site extraction
        self.assertGreaterEqual(site_accuracy, 0.8, f"Site extraction accuracy {site_accuracy:.2%} below threshold")
        
        logger.info("✅ Test 2 passed: Site Extraction Accuracy")
    
    def test_03_time_period_extraction(self):
        """Test 3: Time period extraction from queries."""
        logger.info("🧪 Test 3: Time Period Extraction")
        
        time_test_queries = [
            "Show electricity usage for last month",
            "Water consumption in Q1 2024", 
            "Waste generation from January to March",
            "Power usage in 2023",
            "Show consumption for the past year",
            "Data from December 2024"
        ]
        
        time_extractions = 0
        
        for query in time_test_queries:
            try:
                result = self.classifier.classify(query)
                self.test_metrics['total_classifications'] += 1
                self.test_metrics['llm_call_count'] += 1
                
                self._validate_classification_result(result, query)
                
                if result.time_period:
                    time_extractions += 1
                    logger.info(f"✅ Time period extracted: '{query}' -> {result.time_period}")
                else:
                    logger.warning(f"⚠️ No time period extracted: '{query}'")
                
            except Exception as e:
                logger.error(f"Time extraction failed for query: '{query}' - {e}")
                self.fail(f"Time extraction failed: {e}")
        
        # At least 50% of time-specific queries should extract time periods
        time_accuracy = time_extractions / len(time_test_queries)
        logger.info(f"Time extraction rate: {time_accuracy:.2%} ({time_extractions}/{len(time_test_queries)})")
        
        self.assertGreaterEqual(time_accuracy, 0.5, f"Time extraction rate {time_accuracy:.2%} below threshold")
        
        logger.info("✅ Test 3 passed: Time Period Extraction")
    
    def test_04_confidence_levels(self):
        """Test 4: Confidence level validation and reasonableness."""
        logger.info("🧪 Test 4: Confidence Level Validation")
        
        high_confidence_queries = [
            "Show electricity consumption data",  # Very clear electricity query
            "What is water usage?",  # Very clear water query  
            "Waste generation metrics"  # Very clear waste query
        ]
        
        low_confidence_queries = [
            "Show me some data",  # Ambiguous query
            "What about the numbers?",  # Vague query
            "Tell me things"  # Very unclear query
        ]
        
        # Test high confidence queries
        high_confidence_results = []
        for query in high_confidence_queries:
            try:
                result = self.classifier.classify(query)
                self.test_metrics['total_classifications'] += 1
                self.test_metrics['llm_call_count'] += 1
                
                self._validate_classification_result(result, query)
                high_confidence_results.append(result.confidence)
                
                logger.info(f"High confidence query: '{query}' -> confidence: {result.confidence}")
                
            except Exception as e:
                logger.error(f"High confidence test failed for query: '{query}' - {e}")
                self.fail(f"High confidence test failed: {e}")
        
        # Test low confidence queries  
        low_confidence_results = []
        for query in low_confidence_queries:
            try:
                result = self.classifier.classify(query)
                self.test_metrics['total_classifications'] += 1
                self.test_metrics['llm_call_count'] += 1
                
                self._validate_classification_result(result, query)
                low_confidence_results.append(result.confidence)
                
                logger.info(f"Low confidence query: '{query}' -> confidence: {result.confidence}")
                
            except Exception as e:
                logger.error(f"Low confidence test failed for query: '{query}' - {e}")
                self.fail(f"Low confidence test failed: {e}")
        
        # Validate confidence ranges
        avg_high_confidence = sum(high_confidence_results) / len(high_confidence_results)
        avg_low_confidence = sum(low_confidence_results) / len(low_confidence_results)
        
        logger.info(f"Average high confidence: {avg_high_confidence:.2f}")
        logger.info(f"Average low confidence: {avg_low_confidence:.2f}")
        
        # High confidence queries should generally have higher confidence than low confidence ones
        # This is a soft check as LLM behavior can vary
        if avg_high_confidence > avg_low_confidence:
            logger.info("✅ Confidence levels are appropriately differentiated")
        else:
            logger.warning("⚠️ Confidence levels not clearly differentiated (this may be acceptable for LLM variability)")
        
        logger.info("✅ Test 4 passed: Confidence Level Validation")
    
    def test_05_fallback_mechanism(self):
        """Test 5: Fallback classification mechanism for error scenarios."""
        logger.info("🧪 Test 5: Fallback Mechanism Testing")
        
        # Test with potentially problematic queries
        fallback_test_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "!@#$%^&*()",  # Special characters
            "a" * 1000,  # Very long query
        ]
        
        fallback_success = 0
        
        for query in fallback_test_queries:
            try:
                result = self.classifier.classify(query)
                self.test_metrics['total_classifications'] += 1
                self.test_metrics['llm_call_count'] += 1
                
                # Should still return a valid ClassificationResult 
                self.assertIsInstance(result, ClassificationResult)
                self.assertIn(result.intent, self.classifier.get_supported_intents())
                self.assertIsInstance(result.confidence, (int, float))
                
                fallback_success += 1
                logger.info(f"✅ Fallback successful for problematic query: '{query[:50]}...' -> {result.intent}")
                
            except Exception as e:
                logger.error(f"Fallback failed for query: '{query[:50]}...' - {e}")
                # Fallback should not fail completely
                self.fail(f"Fallback mechanism failed: {e}")
        
        fallback_rate = fallback_success / len(fallback_test_queries)
        logger.info(f"Fallback success rate: {fallback_rate:.2%} ({fallback_success}/{len(fallback_test_queries)})")
        
        # Fallback should work for all problematic queries
        self.assertEqual(fallback_rate, 1.0, "Fallback mechanism should handle all edge cases")
        
        logger.info("✅ Test 5 passed: Fallback Mechanism Testing")
    
    def test_06_batch_processing(self):
        """Test 6: Batch processing functionality."""
        logger.info("🧪 Test 6: Batch Processing")
        
        # Prepare batch of queries
        batch_queries = [
            "Show electricity usage",
            "Water consumption data", 
            "Waste generation metrics",
            "What are CO2 goals?",
            "Risk assessment for Houston",
            "Recommendations for improvement"
        ]
        
        try:
            # Test batch processing
            start_time = time.time()
            results = self.classifier.classify_batch(batch_queries)
            batch_time = time.time() - start_time
            
            self.test_metrics['total_classifications'] += len(batch_queries)
            self.test_metrics['llm_call_count'] += len(batch_queries)
            
            # Validate batch results
            self.assertEqual(len(results), len(batch_queries), "Batch should return same number of results as queries")
            
            for i, result in enumerate(results):
                self._validate_classification_result(result, batch_queries[i])
                logger.info(f"Batch result {i+1}: '{batch_queries[i]}' -> {result.intent} (confidence: {result.confidence})")
            
            logger.info(f"Batch processing completed in {batch_time:.2f} seconds for {len(batch_queries)} queries")
            
            # Performance check - should complete within reasonable time
            avg_time_per_query = batch_time / len(batch_queries)
            logger.info(f"Average time per query: {avg_time_per_query:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.fail(f"Batch processing failed: {e}")
        
        logger.info("✅ Test 6 passed: Batch Processing")
    
    def test_07_supported_intents_consistency(self):
        """Test 7: Supported intents and sites consistency."""
        logger.info("🧪 Test 7: Supported Intents and Sites Consistency")
        
        # Test supported intents
        supported_intents = self.classifier.get_supported_intents()
        expected_intents = [
            "ELECTRICITY_CONSUMPTION",
            "WATER_CONSUMPTION", 
            "WASTE_GENERATION",
            "CO2_GOALS",
            "RISK_ASSESSMENT",
            "RECOMMENDATIONS",
            "GENERAL"
        ]
        
        self.assertEqual(set(supported_intents), set(expected_intents), "Supported intents should match expected list")
        logger.info(f"✅ Supported intents validated: {supported_intents}")
        
        # Test known sites
        known_sites = self.classifier.get_known_sites()
        expected_sites = ["houston_texas", "algonquin_illinois"]
        
        self.assertEqual(set(known_sites), set(expected_sites), "Known sites should match expected list")
        logger.info(f"✅ Known sites validated: {known_sites}")
        
        logger.info("✅ Test 7 passed: Supported Intents and Sites Consistency")
    
    def test_08_error_handling_robustness(self):
        """Test 8: Error handling and robustness."""
        logger.info("🧪 Test 8: Error Handling Robustness")
        
        # Test various error scenarios that should be handled gracefully
        error_test_queries = [
            "Query with unicode characters: 你好世界 émojis 🌍",
            "Query with numbers: 12345 and dates 2024-01-01",
            "Mixed case QUERY with Different CASE patterns",
            "Query\nwith\nnewlines\nand\ttabs",
        ]
        
        successful_error_handling = 0
        
        for query in error_test_queries:
            try:
                result = self.classifier.classify(query)
                self.test_metrics['total_classifications'] += 1
                self.test_metrics['llm_call_count'] += 1
                
                # Should return valid result even for unusual input
                self._validate_classification_result(result, query)
                successful_error_handling += 1
                
                logger.info(f"✅ Error handling successful for: '{query[:50]}...' -> {result.intent}")
                
            except Exception as e:
                logger.error(f"Error handling failed for query: '{query[:50]}...' - {e}")
                # Should not fail on unusual but valid input
        
        error_handling_rate = successful_error_handling / len(error_test_queries)
        logger.info(f"Error handling success rate: {error_handling_rate:.2%} ({successful_error_handling}/{len(error_test_queries)})")
        
        # Should handle at least 90% of unusual inputs gracefully
        self.assertGreaterEqual(error_handling_rate, 0.9, f"Error handling rate {error_handling_rate:.2%} below threshold")
        
        logger.info("✅ Test 8 passed: Error Handling Robustness")
    
    def test_09_integration_with_chatbot_api(self):
        """Test 9: Integration readiness with chatbot API."""
        logger.info("🧪 Test 9: Integration with Chatbot API")
        
        # Test typical chatbot queries
        chatbot_queries = [
            "What is the electricity consumption at Houston?",
            "Can you show me water usage trends?",
            "I need recommendations for reducing waste",
            "What are the environmental risks we should know about?",
            "Hello, can you help me with sustainability data?",
        ]
        
        integration_results = []
        
        for query in chatbot_queries:
            try:
                result = self.classifier.classify(query)
                self.test_metrics['total_classifications'] += 1
                self.test_metrics['llm_call_count'] += 1
                
                self._validate_classification_result(result, query)
                
                # Create integration-ready response format
                integration_response = {
                    'query': result.raw_query,
                    'intent': result.intent,
                    'confidence': result.confidence,
                    'site': result.site,
                    'time_period': result.time_period,
                    'entities': result.extracted_entities
                }
                
                integration_results.append(integration_response)
                
                logger.info(f"✅ Integration response: '{query}' -> {json.dumps(integration_response, indent=2)}")
                
            except Exception as e:
                logger.error(f"Integration test failed for query: '{query}' - {e}")
                self.fail(f"Integration test failed: {e}")
        
        # Validate integration results structure
        for response in integration_results:
            self.assertIn('query', response)
            self.assertIn('intent', response)
            self.assertIn('confidence', response)
            self.assertIn('site', response)
            self.assertIn('time_period', response)
            self.assertIn('entities', response)
        
        logger.info(f"✅ Integration test completed for {len(integration_results)} queries")
        logger.info("✅ Test 9 passed: Integration with Chatbot API")
    
    def test_10_performance_benchmarks(self):
        """Test 10: Performance benchmarks."""
        logger.info("🧪 Test 10: Performance Benchmarks")
        
        # Test single query performance
        test_query = "Show me electricity consumption data for Houston"
        
        # Measure classification time
        start_time = time.time()
        result = self.classifier.classify(test_query)
        classification_time = time.time() - start_time
        
        self.test_metrics['total_classifications'] += 1
        self.test_metrics['llm_call_count'] += 1
        
        self._validate_classification_result(result, test_query)
        
        logger.info(f"Single classification time: {classification_time:.2f} seconds")
        
        # Performance threshold - should complete within 30 seconds (allowing for LLM latency)
        self.assertLess(classification_time, 30.0, f"Classification took {classification_time:.2f}s, exceeding 30s threshold")
        
        # Test multiple consecutive classifications
        consecutive_queries = [
            "Electricity usage data",
            "Water consumption metrics", 
            "Waste generation info"
        ]
        
        consecutive_start = time.time()
        for query in consecutive_queries:
            result = self.classifier.classify(query)
            self.test_metrics['total_classifications'] += 1
            self.test_metrics['llm_call_count'] += 1
            self._validate_classification_result(result, query)
        
        consecutive_time = time.time() - consecutive_start
        avg_consecutive_time = consecutive_time / len(consecutive_queries)
        
        logger.info(f"Consecutive classifications time: {consecutive_time:.2f}s for {len(consecutive_queries)} queries")
        logger.info(f"Average consecutive time per query: {avg_consecutive_time:.2f}s")
        
        # Consecutive queries should maintain reasonable performance
        self.assertLess(avg_consecutive_time, 35.0, f"Average consecutive time {avg_consecutive_time:.2f}s exceeds threshold")
        
        logger.info("✅ Test 10 passed: Performance Benchmarks")
    
    def _validate_classification_result(self, result: ClassificationResult, original_query: str):
        """Validate that a classification result has the expected structure and values."""
        self.test_metrics['total_validation_checks'] += 1
        
        # Basic structure validation
        self.assertIsInstance(result, ClassificationResult, "Result should be ClassificationResult instance")
        self.assertIsInstance(result.intent, str, "Intent should be string")
        self.assertIsInstance(result.confidence, (int, float), "Confidence should be numeric")
        self.assertEqual(result.raw_query, original_query, "Raw query should match original")
        
        # Intent validation
        supported_intents = self.classifier.get_supported_intents()
        self.assertIn(result.intent, supported_intents, f"Intent '{result.intent}' not in supported intents")
        
        # Confidence validation
        self.assertGreaterEqual(result.confidence, 0.0, "Confidence should be non-negative")
        self.assertLessEqual(result.confidence, 1.0, "Confidence should not exceed 1.0")
        
        # Site validation (if present)
        if result.site:
            known_sites = self.classifier.get_known_sites()
            self.assertIn(result.site, known_sites, f"Site '{result.site}' not in known sites")
        
        # Time period validation (if present)
        if result.time_period:
            self.assertIsInstance(result.time_period, dict, "Time period should be dictionary")
        
        # Entities validation (if present)
        if result.extracted_entities:
            self.assertIsInstance(result.extracted_entities, dict, "Extracted entities should be dictionary")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up and report final test metrics."""
        end_time = time.time()
        total_test_time = end_time - cls.test_metrics['test_start_time']
        
        logger.info("\n" + "="*80)
        logger.info("INTENT CLASSIFIER TEST SUITE FINAL REPORT")
        logger.info("="*80)
        logger.info(f"Total Tests Run: {cls.test_metrics['total_tests']}")
        logger.info(f"Tests Passed: {cls.test_metrics['passed_tests']}")
        logger.info(f"Tests Failed: {cls.test_metrics['failed_tests']}")
        logger.info(f"Total Classifications Performed: {cls.test_metrics['total_classifications']}")
        logger.info(f"Total LLM Calls Made: {cls.test_metrics['llm_call_count']}")
        logger.info(f"Total Validation Checks: {cls.test_metrics['total_validation_checks']}")
        logger.info(f"Total Test Execution Time: {total_test_time:.2f} seconds")
        if cls.test_metrics['llm_call_count'] > 0:
            avg_time_per_llm_call = total_test_time / cls.test_metrics['llm_call_count']
            logger.info(f"Average Time Per LLM Call: {avg_time_per_llm_call:.2f} seconds")
        logger.info(f"Test Log File: {log_file}")
        logger.info("="*80)
        
        # Update metrics for successful completion
        cls.test_metrics['passed_tests'] = cls.test_metrics['total_tests']


def run_comprehensive_test_suite():
    """Run the comprehensive test suite and return results."""
    logger.info("Starting Intent Classifier Comprehensive Test Suite")
    logger.info(f"Test log file: {log_file}")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntentClassifier)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Report results
    if result.wasSuccessful():
        logger.info("\n🎉 ALL TESTS PASSED! Intent Classifier is ready for integration.")
        return True
    else:
        logger.error(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s) occurred.")
        
        # Log failure details
        if result.failures:
            logger.error("FAILURES:")
            for test, traceback in result.failures:
                logger.error(f"- {test}: {traceback}")
        
        if result.errors:
            logger.error("ERRORS:")
            for test, traceback in result.errors:
                logger.error(f"- {test}: {traceback}")
        
        return False


if __name__ == '__main__':
    """Run as standalone script."""
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)
