#!/usr/bin/env python3
"""
Comprehensive End-to-End RAG Integration Test Suite for Chatbot API

This test suite provides comprehensive testing of the RAG-integrated chatbot API,
verifying the complete pipeline from user query to grounded response.

Features Tested:
- Complete RAG pipeline: Intent Classification -> Context Retrieval -> Prompt Augmentation -> LLM Response
- Different intent types (electricity, water, waste, goals, recommendations)
- Context retrieval from Neo4j database
- Prompt augmentation functionality
- LLM response grounding in retrieved data
- Error handling scenarios
- FastAPI server integration testing
- Real API calls with curl and requests library (no mocks)

Test Strategy:
- Start FastAPI server in background
- Send real queries to /api/chatbot/chat endpoint
- Verify responses contain data-grounded information
- Test error handling scenarios
- Clean shutdown of server

Created: 2025-09-15
Version: 1.0.0
Author: Claude Code Agent
"""

import os
import sys
import json
import time
import signal
import subprocess
import requests
import logging
import asyncio
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import queue

# Add src directory to Python path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir / 'src'))

# Import test components
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Test Configuration
@dataclass
class RAGTestConfig:
    """RAG Integration Test configuration"""
    BASE_URL = "http://localhost:8000"
    CHAT_ENDPOINT = "/api/chatbot/chat"
    HEALTH_ENDPOINT = "/api/chatbot/health"
    CLEAR_SESSION_ENDPOINT = "/api/chatbot/clear-session"
    
    # Test timeouts
    SERVER_START_TIMEOUT = 60  # seconds to wait for server start
    REQUEST_TIMEOUT = 30       # seconds for individual requests
    TEST_TIMEOUT = 300         # seconds for entire test suite
    
    # Performance thresholds
    MAX_RESPONSE_TIME = 10.0   # seconds for chat responses
    MIN_RESPONSE_LENGTH = 50   # minimum characters in response
    
    # Test data
    ALGONQUIN_SITE = "algonquin_il"
    HOUSTON_SITE = "houston_tx"

class RAGTestResults:
    """Track test results and metrics"""
    def __init__(self):
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        self.end_time = None
        self.test_details = []
        self.server_process = None
        self.log_file = None
    
    def add_test_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Add test result"""
        self.test_count += 1
        if success:
            self.passed_count += 1
        else:
            self.failed_count += 1
        
        self.test_details.append({
            "test_name": test_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
    
    def get_duration(self) -> float:
        """Get test duration"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        return {
            "total_tests": self.test_count,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "success_rate": (self.passed_count / self.test_count * 100) if self.test_count > 0 else 0,
            "duration_seconds": self.get_duration(),
            "timestamp": datetime.now().isoformat()
        }

class FastAPIServerManager:
    """Manage FastAPI server for testing"""
    
    def __init__(self, log_file: str):
        self.process = None
        self.log_file = log_file
        self.server_ready = False
        
    def start_server(self) -> bool:
        """Start FastAPI server in background"""
        try:
            logger.info("Starting FastAPI server...")
            
            # Change to backend directory
            backend_dir = Path(__file__).parent.parent.parent
            
            # Start server with logging
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "src.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ]
            
            with open(self.log_file, 'w') as f:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(backend_dir),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            # Wait for server to start
            if self._wait_for_server():
                logger.info("FastAPI server started successfully")
                return True
            else:
                logger.error("FastAPI server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting FastAPI server: {e}")
            return False
    
    def _wait_for_server(self) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < RAGTestConfig.SERVER_START_TIMEOUT:
            try:
                response = requests.get(
                    f"{RAGTestConfig.BASE_URL}{RAGTestConfig.HEALTH_ENDPOINT}",
                    timeout=5
                )
                if response.status_code == 200:
                    self.server_ready = True
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        return False
    
    def stop_server(self):
        """Stop FastAPI server"""
        if self.process:
            try:
                logger.info("Stopping FastAPI server...")
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Server didn't stop gracefully, killing...")
                    self.process.kill()
                    self.process.wait()
                
                logger.info("FastAPI server stopped")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")

class RAGIntegrationTester:
    """Main RAG integration test class"""
    
    def __init__(self):
        self.config = RAGTestConfig()
        self.results = RAGTestResults()
        self.session = requests.Session()
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"/tmp/rag_integration_test_{timestamp}.log"
        self.results.log_file = self.log_file
        
        # Configure session
        self.session.timeout = self.config.REQUEST_TIMEOUT
        
        # Setup detailed logging
        self._setup_logging()
        
        self.server_manager = FastAPIServerManager(self.log_file.replace('.log', '_server.log'))
    
    def _setup_logging(self):
        """Setup detailed logging"""
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete RAG integration test suite"""
        logger.info("=== Starting RAG Integration Test Suite ===")
        logger.info(f"Log file: {self.log_file}")
        
        try:
            # Start server
            if not self.server_manager.start_server():
                raise Exception("Failed to start FastAPI server")
            
            # Run test categories
            self._test_basic_functionality()
            self._test_rag_pipeline_components()
            self._test_intent_classification()
            self._test_context_retrieval()
            self._test_prompt_augmentation()
            self._test_llm_grounding()
            self._test_error_handling()
            self._test_performance()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.results.add_test_result(
                "test_suite_execution",
                False,
                {"error": str(e)}
            )
        finally:
            # Clean shutdown
            self.server_manager.stop_server()
            self.results.end_time = time.time()
        
        # Generate final report
        summary = self.results.get_summary()
        logger.info("=== RAG Integration Test Suite Complete ===")
        logger.info(f"Results: {summary['passed']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1f}%)")
        logger.info(f"Duration: {summary['duration_seconds']:.1f} seconds")
        logger.info(f"Detailed log: {self.log_file}")
        
        return summary
    
    def _test_basic_functionality(self):
        """Test basic API functionality"""
        logger.info("Testing basic API functionality...")
        
        # Test health endpoint
        try:
            response = self.session.get(f"{self.config.BASE_URL}{self.config.HEALTH_ENDPOINT}")
            success = response.status_code == 200
            details = {
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "response_data": response.json() if success else response.text
            }
            self.results.add_test_result("health_endpoint", success, details)
            logger.info(f"Health endpoint test: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            self.results.add_test_result("health_endpoint", False, {"error": str(e)})
            logger.error(f"Health endpoint test failed: {e}")
    
    def _test_rag_pipeline_components(self):
        """Test RAG pipeline components individually"""
        logger.info("Testing RAG pipeline components...")
        
        # Test basic chat functionality
        test_queries = [
            "Hello, can you help me with electricity data?",
            "What is our current electricity consumption?",
            "How much water did we use last month?",
            "Show me waste generation data",
            "What are our EHS goals?"
        ]
        
        for query in test_queries:
            try:
                start_time = time.time()
                
                chat_request = {
                    "message": query,
                    "context": {"test": True}
                }
                
                response = self.session.post(
                    f"{self.config.BASE_URL}{self.config.CHAT_ENDPOINT}",
                    json=chat_request
                )
                
                response_time = time.time() - start_time
                success = response.status_code == 200 and response_time < self.config.MAX_RESPONSE_TIME
                
                if success:
                    response_data = response.json()
                    # Check response structure
                    required_fields = ['response', 'session_id', 'timestamp']
                    success = all(field in response_data for field in required_fields)
                    
                    # Check response content quality
                    if success:
                        response_text = response_data.get('response', '')
                        success = len(response_text) >= self.config.MIN_RESPONSE_LENGTH
                
                details = {
                    "query": query,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "response_data": response.json() if response.status_code == 200 else response.text[:500]
                }
                
                test_name = f"basic_chat_query_{len(query)//10}"
                self.results.add_test_result(test_name, success, details)
                logger.info(f"Basic chat query test '{query[:30]}...': {'PASS' if success else 'FAIL'}")
                
            except Exception as e:
                self.results.add_test_result(f"basic_chat_query_error", False, {"query": query, "error": str(e)})
                logger.error(f"Basic chat query test failed for '{query}': {e}")
    
    def _test_intent_classification(self):
        """Test intent classification functionality"""
        logger.info("Testing intent classification...")
        
        # Test different intent types
        intent_test_queries = [
            ("What's our electricity consumption in Algonquin?", "electricity"),
            ("How much water did Houston use last month?", "water"),
            ("Show me waste generation data for both sites", "waste"),
            ("What are our CO2 reduction goals?", "goals"),
            ("Give me recommendations to reduce energy usage", "recommendations"),
            ("Compare electricity usage between sites", "comparison"),
            ("What are the trends in water consumption?", "trends")
        ]
        
        for query, expected_intent in intent_test_queries:
            try:
                chat_request = {
                    "message": query,
                    "context": {"test_intent": expected_intent}
                }
                
                response = self.session.post(
                    f"{self.config.BASE_URL}{self.config.CHAT_ENDPOINT}",
                    json=chat_request
                )
                
                success = response.status_code == 200
                response_data = None
                
                if success:
                    response_data = response.json()
                    response_text = response_data.get('response', '').lower()
                    
                    # Check if response contains relevant keywords for the intent
                    intent_keywords = {
                        "electricity": ["electricity", "electric", "power", "kwh", "energy"],
                        "water": ["water", "h2o", "gallons", "liters", "consumption"],
                        "waste": ["waste", "garbage", "disposal", "recycling", "generation"],
                        "goals": ["goal", "target", "reduction", "achievement"],
                        "recommendations": ["recommend", "suggest", "improve", "optimize"],
                        "comparison": ["compare", "comparison", "between", "versus"],
                        "trends": ["trend", "pattern", "over time", "historical"]
                    }
                    
                    keywords = intent_keywords.get(expected_intent, [])
                    has_relevant_content = any(keyword in response_text for keyword in keywords)
                    success = success and has_relevant_content
                
                details = {
                    "query": query,
                    "expected_intent": expected_intent,
                    "status_code": response.status_code,
                    "response_data": response_data
                }
                
                test_name = f"intent_classification_{expected_intent}"
                self.results.add_test_result(test_name, success, details)
                logger.info(f"Intent classification test '{expected_intent}': {'PASS' if success else 'FAIL'}")
                
            except Exception as e:
                self.results.add_test_result(f"intent_classification_error_{expected_intent}", False, {"error": str(e)})
                logger.error(f"Intent classification test failed for '{expected_intent}': {e}")
    
    def _test_context_retrieval(self):
        """Test context retrieval from Neo4j"""
        logger.info("Testing context retrieval from Neo4j...")
        
        # Test queries that should retrieve specific data types
        context_test_queries = [
            {
                "query": "What's the electricity consumption at Algonquin?",
                "site": self.config.ALGONQUIN_SITE,
                "expected_data_indicators": ["kwh", "consumption", "electricity", "algonquin"]
            },
            {
                "query": "How much water did Houston use?",
                "site": self.config.HOUSTON_SITE,
                "expected_data_indicators": ["gallons", "water", "consumption", "houston"]
            },
            {
                "query": "Show me waste data for both facilities",
                "site": None,
                "expected_data_indicators": ["waste", "generation", "tons", "disposal"]
            }
        ]
        
        for test_case in context_test_queries:
            try:
                chat_request = {
                    "message": test_case["query"],
                    "site_filter": test_case["site"]
                }
                
                response = self.session.post(
                    f"{self.config.BASE_URL}{self.config.CHAT_ENDPOINT}",
                    json=chat_request
                )
                
                success = response.status_code == 200
                response_data = None
                
                if success:
                    response_data = response.json()
                    response_text = response_data.get('response', '').lower()
                    
                    # Check for data-grounded indicators
                    expected_indicators = test_case["expected_data_indicators"]
                    has_data_indicators = any(indicator in response_text for indicator in expected_indicators)
                    
                    # Check for numerical data (indicating real data retrieval)
                    import re
                    has_numbers = bool(re.search(r'\d+\.?\d*', response_text))
                    
                    # Check for data sources
                    data_sources = response_data.get('data_sources', [])
                    has_data_sources = len(data_sources) > 0
                    
                    success = success and has_data_indicators and (has_numbers or has_data_sources)
                
                details = {
                    "query": test_case["query"],
                    "site": test_case["site"],
                    "status_code": response.status_code,
                    "has_data_indicators": has_data_indicators if success else False,
                    "has_numbers": has_numbers if success else False,
                    "has_data_sources": has_data_sources if success else False,
                    "response_data": response_data
                }
                
                test_name = f"context_retrieval_{test_case['site'] or 'multi_site'}"
                self.results.add_test_result(test_name, success, details)
                logger.info(f"Context retrieval test '{test_case['site'] or 'multi-site'}': {'PASS' if success else 'FAIL'}")
                
            except Exception as e:
                self.results.add_test_result(f"context_retrieval_error", False, {"error": str(e)})
                logger.error(f"Context retrieval test failed: {e}")
    
    def _test_prompt_augmentation(self):
        """Test prompt augmentation functionality"""
        logger.info("Testing prompt augmentation...")
        
        # Test queries with different complexities to verify prompt augmentation
        augmentation_test_queries = [
            "Give me a detailed analysis of our electricity usage patterns",
            "What recommendations do you have for reducing water consumption?",
            "Compare our current performance against EHS goals",
            "Analyze waste generation trends and suggest improvements"
        ]
        
        for query in augmentation_test_queries:
            try:
                chat_request = {
                    "message": query,
                    "context": {"detailed_analysis": True}
                }
                
                response = self.session.post(
                    f"{self.config.BASE_URL}{self.config.CHAT_ENDPOINT}",
                    json=chat_request
                )
                
                success = response.status_code == 200
                response_data = None
                
                if success:
                    response_data = response.json()
                    response_text = response_data.get('response', '')
                    
                    # Check for detailed, structured response (indicating prompt augmentation)
                    quality_indicators = [
                        len(response_text) > 200,  # Substantial response
                        '.' in response_text,      # Complete sentences
                        any(word in response_text.lower() for word in ['analysis', 'data', 'recommendation', 'trend']),  # Analytical content
                    ]
                    
                    success = success and sum(quality_indicators) >= 2
                
                details = {
                    "query": query,
                    "status_code": response.status_code,
                    "response_length": len(response_text) if response_data else 0,
                    "quality_score": sum(quality_indicators) if success else 0,
                    "response_data": response_data
                }
                
                test_name = f"prompt_augmentation_{len(query)//20}"
                self.results.add_test_result(test_name, success, details)
                logger.info(f"Prompt augmentation test: {'PASS' if success else 'FAIL'}")
                
            except Exception as e:
                self.results.add_test_result("prompt_augmentation_error", False, {"error": str(e)})
                logger.error(f"Prompt augmentation test failed: {e}")
    
    def _test_llm_grounding(self):
        """Test LLM response grounding in retrieved data"""
        logger.info("Testing LLM response grounding...")
        
        # Test queries that require data grounding
        grounding_test_queries = [
            "What specific electricity consumption numbers do we have for last month?",
            "Give me exact water usage figures for Houston",
            "What are the precise waste generation amounts?",
            "Show me actual data from our EHS monitoring"
        ]
        
        for query in grounding_test_queries:
            try:
                chat_request = {
                    "message": query,
                    "context": {"require_data_grounding": True}
                }
                
                response = self.session.post(
                    f"{self.config.BASE_URL}{self.config.CHAT_ENDPOINT}",
                    json=chat_request
                )
                
                success = response.status_code == 200
                response_data = None
                
                if success:
                    response_data = response.json()
                    response_text = response_data.get('response', '')
                    
                    # Check for data grounding indicators
                    grounding_indicators = [
                        # Numerical data presence
                        bool(__import__('re').search(r'\d+\.?\d*\s*(kwh|gallons?|tons?)', response_text.lower())),
                        # Specific site references
                        any(site in response_text.lower() for site in ['algonquin', 'houston']),
                        # Time-specific references
                        any(time_ref in response_text.lower() for time_ref in ['month', 'year', 'quarter', 'period']),
                        # Data source attribution
                        len(response_data.get('data_sources', [])) > 0
                    ]
                    
                    grounding_score = sum(grounding_indicators)
                    success = success and grounding_score >= 2
                
                details = {
                    "query": query,
                    "status_code": response.status_code,
                    "grounding_score": grounding_score if success else 0,
                    "grounding_indicators": grounding_indicators if success else [],
                    "response_data": response_data
                }
                
                test_name = f"llm_grounding_{grounding_score if success else 0}"
                self.results.add_test_result(test_name, success, details)
                logger.info(f"LLM grounding test (score {grounding_score if success else 0}/4): {'PASS' if success else 'FAIL'}")
                
            except Exception as e:
                self.results.add_test_result("llm_grounding_error", False, {"error": str(e)})
                logger.error(f"LLM grounding test failed: {e}")
    
    def _test_error_handling(self):
        """Test error handling scenarios"""
        logger.info("Testing error handling scenarios...")
        
        # Test various error scenarios
        error_test_cases = [
            {
                "name": "empty_message",
                "request": {"message": ""},
                "expected_status": 422  # Validation error
            },
            {
                "name": "invalid_site_filter",
                "request": {"message": "Hello", "site_filter": "invalid_site"},
                "expected_status": 200  # Should handle gracefully
            },
            {
                "name": "malformed_json",
                "request": "invalid json",
                "expected_status": 422
            },
            {
                "name": "very_long_message",
                "request": {"message": "x" * 10000},
                "expected_status": 200  # Should handle gracefully
            }
        ]
        
        for test_case in error_test_cases:
            try:
                if test_case["name"] == "malformed_json":
                    # Test malformed JSON
                    response = requests.post(
                        f"{self.config.BASE_URL}{self.config.CHAT_ENDPOINT}",
                        data=test_case["request"],
                        headers={"Content-Type": "application/json"}
                    )
                else:
                    response = self.session.post(
                        f"{self.config.BASE_URL}{self.config.CHAT_ENDPOINT}",
                        json=test_case["request"]
                    )
                
                expected_status = test_case["expected_status"]
                success = response.status_code == expected_status
                
                details = {
                    "test_case": test_case["name"],
                    "expected_status": expected_status,
                    "actual_status": response.status_code,
                    "response_data": response.text[:500]
                }
                
                self.results.add_test_result(f"error_handling_{test_case['name']}", success, details)
                logger.info(f"Error handling test '{test_case['name']}': {'PASS' if success else 'FAIL'}")
                
            except Exception as e:
                self.results.add_test_result(f"error_handling_{test_case['name']}_error", False, {"error": str(e)})
                logger.error(f"Error handling test '{test_case['name']}' failed: {e}")
    
    def _test_performance(self):
        """Test performance characteristics"""
        logger.info("Testing performance characteristics...")
        
        # Test response times for different query types
        performance_queries = [
            "Quick question: electricity usage?",
            "Give me a detailed analysis of electricity, water, and waste consumption patterns across both facilities with recommendations for improvement",
            "Hello",
            "What are our EHS goals and how are we performing against them?",
        ]
        
        response_times = []
        
        for query in performance_queries:
            try:
                start_time = time.time()
                
                chat_request = {"message": query}
                response = self.session.post(
                    f"{self.config.BASE_URL}{self.config.CHAT_ENDPOINT}",
                    json=chat_request
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                success = (response.status_code == 200 and 
                          response_time < self.config.MAX_RESPONSE_TIME)
                
                details = {
                    "query": query,
                    "response_time": response_time,
                    "status_code": response.status_code
                }
                
                test_name = f"performance_{len(query)//50}"
                self.results.add_test_result(test_name, success, details)
                logger.info(f"Performance test ({response_time:.2f}s): {'PASS' if success else 'FAIL'}")
                
            except Exception as e:
                self.results.add_test_result("performance_error", False, {"error": str(e)})
                logger.error(f"Performance test failed: {e}")
        
        # Overall performance summary
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            performance_summary = {
                "average_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "total_queries": len(response_times),
                "within_threshold": sum(1 for t in response_times if t < self.config.MAX_RESPONSE_TIME)
            }
            
            overall_success = avg_response_time < self.config.MAX_RESPONSE_TIME
            self.results.add_test_result("performance_summary", overall_success, performance_summary)
            logger.info(f"Performance summary - Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s")

def main():
    """Main test execution function"""
    
    # Check environment setup
    required_env_vars = ['OPENAI_API_KEY', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please check your .env file")
        return 1
    
    print("🚀 Starting RAG Integration Test Suite")
    print(f"📝 OpenAI API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    print(f"🗄️  Neo4j URI: {os.getenv('NEO4J_URI', 'Not set')}")
    print()
    
    # Run tests
    tester = RAGIntegrationTester()
    
    try:
        summary = tester.run_all_tests()
        
        # Print final results
        print("\n" + "="*60)
        print("📊 RAG INTEGRATION TEST RESULTS")
        print("="*60)
        print(f"📈 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"📄 Log File: {tester.log_file}")
        print(f"🖥️  Server Log: {tester.log_file.replace('.log', '_server.log')}")
        
        # Return appropriate exit code
        return 0 if summary['failed'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
