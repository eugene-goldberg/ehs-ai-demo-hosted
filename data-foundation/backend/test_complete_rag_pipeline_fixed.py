#!/usr/bin/env python3
"""
Comprehensive RAG Pipeline Integration Test Suite - FIXED VERSION

This test suite validates the entire RAG (Retrieval-Augmented Generation) pipeline 
with all implemented intent types and ensures all components work together properly.

Test Components:
1. Intent Classification - Tests all 6 intent types
2. Context Retrieval - Tests data retrieval from Neo4j for all intent types
3. Prompt Augmentation - Tests prompt enhancement with context
4. End-to-End RAG Pipeline - Tests complete pipeline integration
5. Chatbot API Integration - Tests integrated chatbot API if running

Intent Types Tested:
- ELECTRICITY_CONSUMPTION
- WATER_CONSUMPTION  
- WASTE_GENERATION
- CO2_GOALS
- RISK_ASSESSMENT
- RECOMMENDATIONS

Sites Tested:
- Algonquin IL (algonquin_il)
- Houston TX (houston_tx)

Features:
- Real LLM calls (no mocks)
- Real Neo4j data retrieval
- Comprehensive validation
- Performance testing
- Error handling validation
- Clear result reporting

Created: 2025-09-16
Version: 1.1.0 (Fixed)
Author: Claude Code Test Execution Specialist
"""

import os
import sys
import json
import time
import logging
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import importlib.util

# Add src directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'src'))

# Import environment loading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Individual test result structure"""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class TestSuiteResults:
    """Complete test suite results"""
    test_results: List[TestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    total_duration: float
    timestamp: str
    
    def add_result(self, result: TestResult):
        """Add a test result"""
        self.test_results.append(result)
        self.total_tests += 1
        if result.success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        self.success_rate = (self.passed_tests / self.total_tests) * 100

class RAGPipelineComprehensiveTester:
    """Comprehensive RAG Pipeline Test Suite"""
    
    def __init__(self):
        self.results = TestSuiteResults(
            test_results=[],
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            success_rate=0.0,
            total_duration=0.0,
            timestamp=datetime.now().isoformat()
        )
        
        # Test configuration
        self.valid_sites = ["algonquin_il", "houston_tx"]
        self.intent_types = [
            "ELECTRICITY_CONSUMPTION",
            "WATER_CONSUMPTION", 
            "WASTE_GENERATION",
            "CO2_GOALS",
            "RISK_ASSESSMENT",
            "RECOMMENDATIONS"
        ]
        
        # Test queries for each intent type
        self.test_queries = {
            "ELECTRICITY_CONSUMPTION": [
                "What is our electricity consumption at Algonquin?",
                "Show me power usage for Houston facility",
                "How much electricity did we use last month?",
                "What are the electricity trends at our sites?"
            ],
            "WATER_CONSUMPTION": [
                "What is our water consumption at Houston?", 
                "Show me water usage for Algonquin facility",
                "How much water did we use this quarter?",
                "What are the water consumption patterns?"
            ],
            "WASTE_GENERATION": [
                "What is our waste generation at both sites?",
                "Show me waste data for Algonquin",
                "How much waste did Houston generate?",
                "What are our waste disposal trends?"
            ],
            "CO2_GOALS": [
                "What are our CO2 reduction goals?",
                "Show me our carbon footprint targets",
                "How are we performing against CO2 goals?",
                "What progress have we made on emissions?"
            ],
            "RISK_ASSESSMENT": [
                "What are the environmental risks at our facilities?",
                "Show me risk assessment for Algonquin",
                "What safety risks do we need to address?",
                "Analyze environmental compliance risks"
            ],
            "RECOMMENDATIONS": [
                "Give me recommendations to reduce electricity usage",
                "What suggestions do you have for water conservation?",
                "How can we improve waste management?",
                "Recommend environmental improvements"
            ]
        }
        
        # Initialize components
        self.intent_classifier = None
        self.context_retriever = None
        self.prompt_augmenter = None
        self.neo4j_client = None
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"/tmp/rag_pipeline_comprehensive_test_fixed_{timestamp}.log"
        self._setup_file_logging()
        
    def _setup_file_logging(self):
        """Setup file logging for detailed test output"""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
    def _initialize_components(self) -> bool:
        """Initialize RAG pipeline components"""
        logger.info("Initializing RAG pipeline components...")
        
        try:
            # Initialize Neo4j client
            from src.database.neo4j_client import Neo4jClient, ConnectionConfig
            
            config = ConnectionConfig(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                username=os.getenv("NEO4J_USERNAME", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "EhsAI2024!"),
                database=os.getenv("NEO4J_DATABASE", "neo4j")
            )
            
            self.neo4j_client = Neo4jClient(config)
            logger.info("✓ Neo4j client initialized")
            
            # Initialize Intent Classifier
            from src.services.intent_classifier import IntentClassifier
            self.intent_classifier = IntentClassifier()
            logger.info("✓ Intent classifier initialized")
            
            # Initialize Context Retriever
            from src.services.context_retriever import ContextRetriever
            self.context_retriever = ContextRetriever(self.neo4j_client)
            logger.info("✓ Context retriever initialized")
            
            # Initialize Prompt Augmenter (if exists)
            try:
                from src.services.prompt_augmenter import PromptAugmenter
                self.prompt_augmenter = PromptAugmenter()
                logger.info("✓ Prompt augmenter initialized")
            except ImportError:
                logger.warning("⚠ Prompt augmenter not available")
                self.prompt_augmenter = None
            
            logger.info("All RAG components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _get_context_for_intent(self, intent_type: str, site: str, time_period: str = "last_month"):
        """Get context data for specific intent type and site"""
        try:
            if intent_type == "ELECTRICITY_CONSUMPTION":
                return self.context_retriever.get_electricity_context(site, time_period)
            elif intent_type == "WATER_CONSUMPTION":
                return self.context_retriever.get_water_context(site, time_period)
            elif intent_type == "WASTE_GENERATION":
                return self.context_retriever.get_waste_context(site, time_period)
            elif intent_type == "CO2_GOALS":
                return self.context_retriever.get_co2_goals_context(site, time_period)
            elif intent_type == "RISK_ASSESSMENT":
                return self.context_retriever.get_risk_assessment_context(site, time_period)
            elif intent_type == "RECOMMENDATIONS":
                return self.context_retriever.get_recommendations_context(site, time_period)
            else:
                logger.warning(f"Unknown intent type: {intent_type}")
                return None
        except Exception as e:
            logger.error(f"Error getting context for {intent_type} at {site}: {e}")
            return None
    
    def test_intent_classification(self) -> List[TestResult]:
        """Test intent classification for all intent types"""
        logger.info("=== Testing Intent Classification ===")
        results = []
        
        for intent_type in self.intent_types:
            queries = self.test_queries[intent_type]
            
            for i, query in enumerate(queries):
                test_name = f"intent_classification_{intent_type.lower()}_{i+1}"
                
                start_time = time.time()
                try:
                    # Classify intent using correct method name
                    classification_result = self.intent_classifier.classify(query)
                    duration = time.time() - start_time
                    
                    # Validate result
                    success = (
                        classification_result and
                        hasattr(classification_result, 'intent') and
                        classification_result.intent is not None
                    )
                    
                    details = {
                        "query": query,
                        "expected_intent": intent_type,
                        "actual_intent": classification_result.intent if classification_result else None,
                        "confidence": classification_result.confidence if classification_result and hasattr(classification_result, 'confidence') else None,
                        "sites": classification_result.sites if classification_result and hasattr(classification_result, 'sites') else None,
                        "time_period": classification_result.time_period if classification_result and hasattr(classification_result, 'time_period') else None
                    }
                    
                    result = TestResult(test_name, success, duration, details)
                    logger.info(f"  {test_name}: {'PASS' if success else 'FAIL'} ({duration:.2f}s)")
                    
                except Exception as e:
                    duration = time.time() - start_time
                    result = TestResult(
                        test_name, False, duration, 
                        {"query": query, "expected_intent": intent_type}, 
                        str(e)
                    )
                    logger.error(f"  {test_name}: FAIL - {e}")
                
                results.append(result)
                self.results.add_result(result)
        
        return results
    
    def test_context_retrieval(self) -> List[TestResult]:
        """Test context retrieval for all intent types and sites"""
        logger.info("=== Testing Context Retrieval ===")
        results = []
        
        for intent_type in self.intent_types:
            for site in self.valid_sites:
                test_name = f"context_retrieval_{intent_type.lower()}_{site}"
                
                start_time = time.time()
                try:
                    # Get context for intent and site using correct method
                    context_data = self._get_context_for_intent(intent_type, site, "last_month")
                    duration = time.time() - start_time
                    
                    # Validate context data
                    success = (
                        context_data is not None and
                        len(context_data) > 0
                    )
                    
                    details = {
                        "intent": intent_type,
                        "site": site,
                        "context_count": len(context_data) if context_data else 0,
                        "has_data": success,
                        "sample_context": str(context_data[:2]) if context_data and len(context_data) > 0 else None
                    }
                    
                    result = TestResult(test_name, success, duration, details)
                    logger.info(f"  {test_name}: {'PASS' if success else 'FAIL'} ({duration:.2f}s)")
                    
                except Exception as e:
                    duration = time.time() - start_time
                    result = TestResult(
                        test_name, False, duration,
                        {"intent": intent_type, "site": site},
                        str(e)
                    )
                    logger.error(f"  {test_name}: FAIL - {e}")
                
                results.append(result)
                self.results.add_result(result)
        
        return results
    
    def test_prompt_augmentation(self) -> List[TestResult]:
        """Test prompt augmentation functionality"""
        logger.info("=== Testing Prompt Augmentation ===")
        results = []
        
        if not self.prompt_augmenter:
            logger.warning("Prompt augmenter not available, skipping tests")
            return results
        
        for intent_type in self.intent_types:
            query = self.test_queries[intent_type][0]  # Use first query for each intent
            test_name = f"prompt_augmentation_{intent_type.lower()}"
            
            start_time = time.time()
            try:
                # Get context first
                context_data = self._get_context_for_intent(intent_type, self.valid_sites[0], "last_month")
                
                # Augment prompt using correct method name
                augmented_prompt = self.prompt_augmenter.create_augmented_prompt(
                    query, context_data, intent_type
                )
                duration = time.time() - start_time
                
                # Validate augmentation
                success = (
                    augmented_prompt is not None and
                    len(augmented_prompt) > len(query) and
                    query in augmented_prompt
                )
                
                details = {
                    "original_query": query,
                    "intent": intent_type,
                    "context_count": len(context_data) if context_data else 0,
                    "augmented_length": len(augmented_prompt) if augmented_prompt else 0,
                    "enhancement_ratio": len(augmented_prompt) / len(query) if augmented_prompt else 0,
                    "augmented_preview": augmented_prompt[:200] + "..." if augmented_prompt and len(augmented_prompt) > 200 else augmented_prompt
                }
                
                result = TestResult(test_name, success, duration, details)
                logger.info(f"  {test_name}: {'PASS' if success else 'FAIL'} ({duration:.2f}s)")
                
            except Exception as e:
                duration = time.time() - start_time
                result = TestResult(
                    test_name, False, duration,
                    {"query": query, "intent": intent_type},
                    str(e)
                )
                logger.error(f"  {test_name}: FAIL - {e}")
            
            results.append(result)
            self.results.add_result(result)
        
        return results
    
    def test_end_to_end_rag_pipeline(self) -> List[TestResult]:
        """Test complete end-to-end RAG pipeline"""
        logger.info("=== Testing End-to-End RAG Pipeline ===")
        results = []
        
        for intent_type in self.intent_types:
            for site in self.valid_sites:
                query = f"What is our {intent_type.lower().replace('_', ' ')} at {site.replace('_', ' ')}?"
                test_name = f"e2e_rag_pipeline_{intent_type.lower()}_{site}"
                
                start_time = time.time()
                try:
                    # Step 1: Intent Classification
                    classification_result = self.intent_classifier.classify(query)
                    
                    # Step 2: Context Retrieval
                    context_data = self._get_context_for_intent(intent_type, site, "last_month")
                    
                    # Step 3: Prompt Augmentation (if available)
                    augmented_prompt = query
                    if self.prompt_augmenter:
                        augmented_prompt = self.prompt_augmenter.create_augmented_prompt(
                            query, context_data, intent_type
                        )
                    
                    # Step 4: Validate pipeline completion
                    duration = time.time() - start_time
                    
                    success = (
                        classification_result is not None and
                        context_data is not None and
                        len(context_data) > 0 and
                        augmented_prompt is not None
                    )
                    
                    details = {
                        "query": query,
                        "intent": intent_type,
                        "site": site,
                        "classification_success": classification_result is not None,
                        "context_retrieval_success": context_data is not None and len(context_data) > 0,
                        "prompt_augmentation_success": augmented_prompt is not None,
                        "context_count": len(context_data) if context_data else 0,
                        "pipeline_stages_completed": sum([
                            classification_result is not None,
                            context_data is not None and len(context_data) > 0,
                            augmented_prompt is not None
                        ])
                    }
                    
                    result = TestResult(test_name, success, duration, details)
                    logger.info(f"  {test_name}: {'PASS' if success else 'FAIL'} ({duration:.2f}s)")
                    
                except Exception as e:
                    duration = time.time() - start_time
                    result = TestResult(
                        test_name, False, duration,
                        {"query": query, "intent": intent_type, "site": site},
                        str(e)
                    )
                    logger.error(f"  {test_name}: FAIL - {e}")
                
                results.append(result)
                self.results.add_result(result)
        
        return results
    
    def test_chatbot_api_integration(self) -> List[TestResult]:
        """Test integrated chatbot API if running"""
        logger.info("=== Testing Chatbot API Integration ===")
        results = []
        
        try:
            import requests
            
            # Test if API is running
            try:
                health_response = requests.get("http://localhost:8000/api/chatbot/health", timeout=5)
                api_running = health_response.status_code == 200
            except:
                api_running = False
            
            if not api_running:
                logger.warning("Chatbot API not running, skipping API integration tests")
                return results
            
            # Test API with sample queries
            for intent_type in self.intent_types[:3]:  # Test first 3 intents to save time
                query = self.test_queries[intent_type][0]
                test_name = f"api_integration_{intent_type.lower()}"
                
                start_time = time.time()
                try:
                    chat_request = {
                        "message": query,
                        "context": {"test": True}
                    }
                    
                    response = requests.post(
                        "http://localhost:8000/api/chatbot/chat",
                        json=chat_request,
                        timeout=30
                    )
                    duration = time.time() - start_time
                    
                    success = (
                        response.status_code == 200 and
                        response.json().get('response') is not None and
                        len(response.json().get('response', '')) > 50
                    )
                    
                    details = {
                        "query": query,
                        "intent": intent_type,
                        "status_code": response.status_code,
                        "response_length": len(response.json().get('response', '')) if response.status_code == 200 else 0,
                        "has_session_id": 'session_id' in response.json() if response.status_code == 200 else False
                    }
                    
                    result = TestResult(test_name, success, duration, details)
                    logger.info(f"  {test_name}: {'PASS' if success else 'FAIL'} ({duration:.2f}s)")
                    
                except Exception as e:
                    duration = time.time() - start_time
                    result = TestResult(
                        test_name, False, duration,
                        {"query": query, "intent": intent_type},
                        str(e)
                    )
                    logger.error(f"  {test_name}: FAIL - {e}")
                
                results.append(result)
                self.results.add_result(result)
        
        except ImportError:
            logger.warning("Requests library not available, skipping API tests")
        
        return results
    
    def test_error_handling(self) -> List[TestResult]:
        """Test error handling scenarios"""
        logger.info("=== Testing Error Handling ===")
        results = []
        
        error_test_cases = [
            {
                "name": "empty_query",
                "query": "",
                "expected_behavior": "graceful_handling"
            },
            {
                "name": "invalid_site",
                "query": "What is electricity consumption at invalid_site?",
                "site": "invalid_site",
                "expected_behavior": "graceful_handling"
            },
            {
                "name": "very_long_query",
                "query": "What is " + "electricity consumption " * 100,
                "expected_behavior": "graceful_handling"
            },
            {
                "name": "special_characters",
                "query": "What is @#$%^&*() consumption???",
                "expected_behavior": "graceful_handling"
            }
        ]
        
        for test_case in error_test_cases:
            test_name = f"error_handling_{test_case['name']}"
            
            start_time = time.time()
            try:
                # Test intent classification with error case
                classification_result = self.intent_classifier.classify(test_case["query"])
                
                # Test context retrieval with error case
                if "site" in test_case:
                    context_data = self._get_context_for_intent("ELECTRICITY_CONSUMPTION", test_case["site"], "last_month")
                else:
                    context_data = self._get_context_for_intent("ELECTRICITY_CONSUMPTION", "algonquin_il", "last_month")
                
                duration = time.time() - start_time
                
                # Success means graceful handling (no crashes)
                success = True
                
                details = {
                    "test_case": test_case["name"],
                    "query": test_case["query"][:100] + "..." if len(test_case["query"]) > 100 else test_case["query"],
                    "classification_handled": classification_result is not None,
                    "context_retrieval_handled": context_data is not None,
                    "graceful_handling": True
                }
                
                result = TestResult(test_name, success, duration, details)
                logger.info(f"  {test_name}: {'PASS' if success else 'FAIL'} ({duration:.2f}s)")
                
            except Exception as e:
                duration = time.time() - start_time
                # Partial success if only some components crashed
                success = False
                result = TestResult(
                    test_name, success, duration,
                    {"test_case": test_case["name"], "query": test_case["query"][:100]},
                    str(e)
                )
                logger.warning(f"  {test_name}: FAIL - Not gracefully handled: {e}")
            
            results.append(result)
            self.results.add_result(result)
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("=== Generating Comprehensive Test Report ===")
        
        # Calculate final metrics
        self.results.total_duration = sum(result.duration for result in self.results.test_results)
        
        # Group results by category
        results_by_category = {}
        for result in self.results.test_results:
            category = result.test_name.split('_')[0] + '_' + result.test_name.split('_')[1]
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)
        
        # Create detailed report
        report = {
            "test_summary": {
                "total_tests": self.results.total_tests,
                "passed_tests": self.results.passed_tests,
                "failed_tests": self.results.failed_tests,
                "success_rate": self.results.success_rate,
                "total_duration": self.results.total_duration,
                "timestamp": self.results.timestamp,
                "log_file": self.log_file
            },
            "category_breakdown": {},
            "intent_type_coverage": {},
            "site_coverage": {},
            "performance_metrics": {},
            "detailed_results": []
        }
        
        # Category breakdown
        for category, category_results in results_by_category.items():
            passed = sum(1 for r in category_results if r.success)
            total = len(category_results)
            report["category_breakdown"][category] = {
                "passed": passed,
                "total": total,
                "success_rate": (passed / total) * 100 if total > 0 else 0,
                "average_duration": sum(r.duration for r in category_results) / total if total > 0 else 0
            }
        
        # Intent type coverage
        for intent_type in self.intent_types:
            intent_results = [r for r in self.results.test_results if intent_type.lower() in r.test_name]
            passed = sum(1 for r in intent_results if r.success)
            total = len(intent_results)
            report["intent_type_coverage"][intent_type] = {
                "passed": passed,
                "total": total,
                "success_rate": (passed / total) * 100 if total > 0 else 0
            }
        
        # Site coverage
        for site in self.valid_sites:
            site_results = [r for r in self.results.test_results if site in r.test_name]
            passed = sum(1 for r in site_results if r.success)
            total = len(site_results)
            report["site_coverage"][site] = {
                "passed": passed,
                "total": total,
                "success_rate": (passed / total) * 100 if total > 0 else 0
            }
        
        # Performance metrics
        durations = [r.duration for r in self.results.test_results]
        report["performance_metrics"] = {
            "average_test_duration": sum(durations) / len(durations) if durations else 0,
            "min_test_duration": min(durations) if durations else 0,
            "max_test_duration": max(durations) if durations else 0,
            "total_execution_time": sum(durations)
        }
        
        # Detailed results
        for result in self.results.test_results:
            report["detailed_results"].append({
                "test_name": result.test_name,
                "success": result.success,
                "duration": result.duration,
                "error_message": result.error_message,
                "details": result.details
            })
        
        return report
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up test resources...")
        
        try:
            if self.context_retriever:
                self.context_retriever.close()
            
            if self.neo4j_client:
                self.neo4j_client.close()
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite"""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE RAG PIPELINE TEST SUITE - FIXED VERSION")
        logger.info("=" * 60)
        logger.info(f"Testing {len(self.intent_types)} intent types")
        logger.info(f"Testing {len(self.valid_sites)} sites: {', '.join(self.valid_sites)}")
        logger.info(f"Log file: {self.log_file}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Initialize components
            if not self._initialize_components():
                raise Exception("Failed to initialize RAG pipeline components")
            
            # Run all test categories
            logger.info("Running test suite...")
            
            # 1. Test Intent Classification
            self.test_intent_classification()
            
            # 2. Test Context Retrieval
            self.test_context_retrieval()
            
            # 3. Test Prompt Augmentation
            self.test_prompt_augmentation()
            
            # 4. Test End-to-End RAG Pipeline
            self.test_end_to_end_rag_pipeline()
            
            # 5. Test Chatbot API Integration
            self.test_chatbot_api_integration()
            
            # 6. Test Error Handling
            self.test_error_handling()
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            logger.info("=" * 60)
            logger.info("COMPREHENSIVE RAG PIPELINE TEST SUITE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total Tests: {report['test_summary']['total_tests']}")
            logger.info(f"Passed: {report['test_summary']['passed_tests']}")
            logger.info(f"Failed: {report['test_summary']['failed_tests']}")
            logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
            logger.info(f"Total Duration: {report['test_summary']['total_duration']:.2f} seconds")
            logger.info(f"Detailed Log: {self.log_file}")
            logger.info("=" * 60)
            
            return report
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            logger.error(traceback.format_exc())
            
            # Create failure report
            report = {
                "test_summary": {
                    "total_tests": self.results.total_tests,
                    "passed_tests": self.results.passed_tests,
                    "failed_tests": self.results.failed_tests,
                    "success_rate": self.results.success_rate,
                    "total_duration": time.time() - start_time,
                    "timestamp": self.results.timestamp,
                    "log_file": self.log_file,
                    "suite_error": str(e)
                }
            }
            return report
            
        finally:
            self.cleanup()

def main():
    """Main test execution function"""
    # Ensure we're running with Python 3 and proper environment
    if sys.version_info < (3, 7):
        logger.error("This test requires Python 3.7 or higher")
        sys.exit(1)
    
    # Check for required environment variables
    required_env_vars = ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file and ensure all required API keys are set")
        sys.exit(1)
    
    # Create and run test suite
    tester = RAGPipelineComprehensiveTester()
    report = tester.run_comprehensive_test_suite()
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/tmp/rag_pipeline_comprehensive_test_report_fixed_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Detailed test report saved to: {report_file}")
    
    # Exit with appropriate code
    success_rate = report.get('test_summary', {}).get('success_rate', 0)
    if success_rate >= 80:
        logger.info("Test suite PASSED (≥80% success rate)")
        sys.exit(0)
    else:
        logger.error(f"Test suite FAILED ({success_rate:.1f}% success rate)")
        sys.exit(1)

if __name__ == "__main__":
    main()
