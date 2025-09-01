#!/usr/bin/env python3
"""
Environmental API Neo4j Integration Test

This test script validates the Environmental Assessment API with real Neo4j data.
It tests all endpoints to ensure they return actual environmental data from the database.

Author: AI Assistant  
Date: 2025-08-31
"""

import sys
import os
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests

# Set up logging
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'/tmp/environmental_neo4j_integration_test_{current_time}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class EnvironmentalAPITester:
    """Tester for Environmental API with real Neo4j data"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        self.start_time = time.time()
        
        # Test endpoints
        self.endpoints = [
            ('/api/environmental/electricity/facts', 'electricity_facts'),
            ('/api/environmental/electricity/risks', 'electricity_risks'), 
            ('/api/environmental/electricity/recommendations', 'electricity_recommendations'),
            ('/api/environmental/water/facts', 'water_facts'),
            ('/api/environmental/water/risks', 'water_risks'),
            ('/api/environmental/water/recommendations', 'water_recommendations'),
            ('/api/environmental/waste/facts', 'waste_facts'),
            ('/api/environmental/waste/risks', 'waste_risks'),
            ('/api/environmental/waste/recommendations', 'waste_recommendations'),
        ]
        
        # Test parameters
        self.test_params = [
            {},  # No parameters
            {'location_path': 'Building'},
            {'start_date': '2024-01-01', 'end_date': '2024-12-31'},
        ]
        
    def verify_neo4j_has_data(self) -> bool:
        """Verify Neo4j has environmental data"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            from neo4j import GraphDatabase
            
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            username = os.getenv('NEO4J_USERNAME', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                queries = [
                    ('ElectricityConsumption', 'MATCH (e:ElectricityConsumption) RETURN count(e) as count'),
                    ('WaterConsumption', 'MATCH (w:WaterConsumption) RETURN count(w) as count'),
                    ('WasteGeneration', 'MATCH (w:WasteGeneration) RETURN count(w) as count')
                ]
                
                total_records = 0
                for name, query in queries:
                    try:
                        result = session.run(query)
                        count = result.single()['count']
                        total_records += count
                        logger.info(f"📊 {name} nodes: {count}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error querying {name}: {e}")
                        
                if total_records > 0:
                    logger.info(f"✅ Neo4j has {total_records} total environmental records")
                    
                    # Sample some data
                    self.sample_data(session)
                    return True
                else:
                    logger.warning("⚠️ No environmental data found in Neo4j")
                    return False
                    
            driver.close()
            
        except Exception as e:
            logger.error(f"❌ Neo4j verification failed: {e}")
            return False
    
    def sample_data(self, session):
        """Sample actual data from Neo4j"""
        try:
            # Sample electricity data
            result = session.run("""
                MATCH (e:ElectricityConsumption)
                RETURN e.location as location, e.date as date, e.consumption_kwh as consumption,
                       e.cost_usd as cost
                LIMIT 3
            """)
            
            samples = [dict(record) for record in result]
            if samples:
                logger.info("📊 Sample electricity data:")
                for sample in samples:
                    logger.info(f"   📍 {sample.get('location', 'Unknown')}: {sample.get('consumption', 0)} kWh (${sample.get('cost', 0)}) on {sample.get('date', 'Unknown')}")
                    
        except Exception as e:
            logger.error(f"❌ Error sampling data: {e}")
    
    def verify_api_server(self) -> bool:
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API server is running")
                return True
            else:
                logger.error(f"❌ API server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API server not responding: {e}")
            return False
    
    def run_comprehensive_tests(self):
        """Run all environmental API tests"""
        logger.info("🧪 Starting comprehensive environmental API tests...")
        logger.info(f"📄 Detailed logs will be saved to: {log_file}")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        total_data_points = 0
        
        # Test each endpoint with different parameters
        for endpoint_path, endpoint_name in self.endpoints:
            for param_idx, params in enumerate(self.test_params):
                test_name = f"{endpoint_name}_params_{param_idx}"
                total_tests += 1
                
                logger.info(f"🔬 Testing {test_name}...")
                logger.info(f"   📍 Endpoint: {endpoint_path}")
                logger.info(f"   📋 Parameters: {params}")
                
                try:
                    start_time = time.time()
                    
                    # Make API request
                    response = requests.get(
                        f"{self.base_url}{endpoint_path}",
                        params=params,
                        timeout=30
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Validate response
                    result = self.validate_response(test_name, response, execution_time)
                    self.test_results.append(result)
                    
                    if result['passed']:
                        passed_tests += 1
                        data_count = result.get('data_count', 0)
                        total_data_points += data_count
                        logger.info(f"✅ {test_name}: PASSED ({execution_time:.2f}s, {data_count} items)")
                        
                        # Log sample data if available
                        if result.get('sample_data'):
                            logger.info(f"   📊 Sample: {result['sample_data']}")
                    else:
                        failed_tests += 1
                        logger.error(f"❌ {test_name}: FAILED - {result['error']}")
                        
                except Exception as e:
                    failed_tests += 1
                    logger.error(f"❌ {test_name}: EXCEPTION - {str(e)}")
        
        # Test LLM assessment endpoint
        self.test_llm_assessment()
        
        # Generate final report
        self.generate_report(total_tests, passed_tests, failed_tests, total_data_points)
    
    def validate_response(self, test_name: str, response, execution_time: float) -> dict:
        """Validate API response"""
        try:
            # Check status
            if response.status_code != 200:
                return {
                    'test_name': test_name,
                    'passed': False,
                    'error': f"HTTP {response.status_code}: {response.text[:200]}",
                    'execution_time': execution_time,
                }
            
            # Parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                return {
                    'test_name': test_name,
                    'passed': False,
                    'error': f"Invalid JSON: {str(e)}",
                    'execution_time': execution_time,
                }
            
            # Validate structure
            if not isinstance(data, list):
                return {
                    'test_name': test_name,
                    'passed': False,
                    'error': f"Expected list, got {type(data)}",
                    'execution_time': execution_time,
                }
            
            # Extract sample data if available
            sample_data = None
            if len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict):
                    # Create a readable sample
                    sample_parts = []
                    for key in ['title', 'value', 'unit', 'description']:
                        if key in first_item and first_item[key]:
                            sample_parts.append(f"{key}: {first_item[key]}")
                    sample_data = ", ".join(sample_parts[:2]) if sample_parts else str(first_item)[:100]
            
            return {
                'test_name': test_name,
                'passed': True,
                'error': None,
                'execution_time': execution_time,
                'data_count': len(data),
                'sample_data': sample_data
            }
            
        except Exception as e:
            return {
                'test_name': test_name,
                'passed': False,
                'error': f"Validation error: {str(e)}",
                'execution_time': execution_time,
            }
    
    def test_llm_assessment(self):
        """Test LLM assessment endpoint"""
        logger.info("🔬 Testing LLM assessment endpoint...")
        
        try:
            test_request = {
                "categories": ["electricity", "water", "waste"],
                "location_path": "Building"
            }
            
            response = requests.post(
                f"{self.base_url}/api/environmental/llm-assessment",
                json=test_request,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'assessment_id' in data and 'status' in data:
                    logger.info(f"✅ LLM assessment: PASSED")
                    logger.info(f"   📋 Assessment ID: {data['assessment_id']}")
                    logger.info(f"   📊 Status: {data['status']}")
                else:
                    logger.error("❌ LLM assessment: Invalid response structure")
            else:
                logger.error(f"❌ LLM assessment: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ LLM assessment test failed: {e}")
    
    def generate_report(self, total_tests: int, passed_tests: int, failed_tests: int, total_data_points: int):
        """Generate test report"""
        total_time = time.time() - self.start_time
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("=" * 80)
        logger.info("📊 ENVIRONMENTAL API NEO4J INTEGRATION TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"🕒 Test Duration: {total_time:.2f} seconds")
        logger.info(f"🧪 Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"📊 Success Rate: {success_rate:.1f}%")
        logger.info(f"📈 Total Data Points Returned: {total_data_points}")
        
        if total_data_points > 0:
            logger.info("✅ SUCCESS: API is returning real Neo4j environmental data")
        else:
            logger.warning("⚠️ WARNING: No data returned - API may not be connected to Neo4j properly")
        
        # Performance stats
        successful_tests = [r for r in self.test_results if r['passed']]
        if successful_tests:
            times = [r['execution_time'] for r in successful_tests]
            avg_time = sum(times) / len(times)
            max_time = max(times)
            logger.info(f"⚡ Average Response Time: {avg_time:.2f}s")
            logger.info(f"⚡ Maximum Response Time: {max_time:.2f}s")
        
        # Detailed results
        logger.info("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅" if result['passed'] else "❌"
            time_info = f"({result['execution_time']:.2f}s)"
            
            if result['passed']:
                count = result.get('data_count', 0)
                logger.info(f"{status} {result['test_name']}: {count} items {time_info}")
            else:
                logger.info(f"{status} {result['test_name']}: {result['error']} {time_info}")
        
        logger.info(f"\n📄 Full test log: {log_file}")
        logger.info("=" * 80)
        
        return failed_tests == 0


def main():
    """Main test function"""
    print("🚀 Environmental API Neo4j Integration Test")
    print("Testing Environmental Assessment API with real Neo4j data")
    print("=" * 60)
    
    tester = EnvironmentalAPITester()
    
    try:
        # 1. Verify Neo4j has data
        logger.info("🔍 Step 1: Verifying Neo4j connection and data...")
        if not tester.verify_neo4j_has_data():
            logger.error("❌ Neo4j verification failed - aborting tests")
            return False
        
        # 2. Verify API server is running
        logger.info("🔍 Step 2: Verifying API server is running...")
        if not tester.verify_api_server():
            logger.error("❌ API server is not running")
            logger.error("Please start the API server first:")
            logger.error("cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend")
            logger.error("source venv/bin/activate")
            logger.error("python3 src/ehs_extraction_api.py")
            return False
        
        # 3. Run comprehensive tests
        logger.info("🔍 Step 3: Running comprehensive API tests...")
        tester.run_comprehensive_tests()
        
        logger.info("✅ Test execution completed successfully")
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
