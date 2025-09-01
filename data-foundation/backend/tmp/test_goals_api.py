#!/usr/bin/env python3
"""
EHS Goals API Test Script

This script comprehensively tests all EHS goals API endpoints to verify:
1. All goals endpoints are working correctly
2. Annual goals are returned correctly
3. Site-specific goals for Algonquin and Houston
4. Progress endpoint functionality (with simulated data)
5. Clear output showing which endpoints work

Usage:
    python3 test_goals_api.py

Requirements:
    - requests
    - json
    - tabulate (for pretty output)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import requests
    from tabulate import tabulate
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install requests tabulate")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/test_goals_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Data class for test results"""
    endpoint: str
    method: str
    status_code: int
    success: bool
    response_time: float
    error_message: Optional[str] = None
    data_summary: Optional[str] = None

class EHSGoalsAPITester:
    """Comprehensive tester for EHS Goals API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.test_results: List[TestResult] = []
        
        # Test configuration
        self.timeout = 30  # seconds
        self.expected_sites = ['algonquin_illinois', 'houston_texas']
        self.expected_categories = ['co2_emissions', 'water_consumption', 'waste_generation']
        
    def run_all_tests(self) -> bool:
        """
        Run all EHS Goals API tests
        
        Returns:
            bool: True if all tests pass, False otherwise
        """
        logger.info("=" * 80)
        logger.info("Starting EHS Goals API Comprehensive Test Suite")
        logger.info("=" * 80)
        
        test_methods = [
            self.test_health_endpoint,
            self.test_annual_goals_endpoint,
            self.test_site_specific_goals_algonquin,
            self.test_site_specific_goals_houston,
            self.test_invalid_site_goals,
            self.test_goals_summary_endpoint,
            self.test_progress_endpoint_algonquin,
            self.test_progress_endpoint_houston,
            self.test_progress_invalid_site,
            self.test_api_response_structure,
            self.test_data_consistency
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"\nRunning {test_method.__name__}...")
                test_method()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with exception: {e}")
                self.test_results.append(TestResult(
                    endpoint=test_method.__name__,
                    method="UNKNOWN",
                    status_code=0,
                    success=False,
                    response_time=0.0,
                    error_message=str(e)
                ))
        
        # Generate and display summary
        self.generate_test_summary()
        
        # Return overall success
        return all(result.success for result in self.test_results)
    
    def test_health_endpoint(self):
        """Test the goals health check endpoint"""
        endpoint = "/api/goals/health"
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            success = (
                response.status_code == 200 and
                "status" in response.json()
            )
            
            data_summary = None
            if success:
                data = response.json()
                data_summary = f"Status: {data.get('status', 'unknown')}"
                if 'total_goals' in data:
                    data_summary += f", Total Goals: {data['total_goals']}"
            
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                data_summary=data_summary
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=0,
                success=False,
                response_time=response_time,
                error_message=str(e)
            ))
    
    def test_annual_goals_endpoint(self):
        """Test the annual goals endpoint"""
        endpoint = "/api/goals/annual"
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ['goals', 'total_goals', 'sites', 'categories']
                has_required_fields = all(field in data for field in required_fields)
                
                # Validate goals data
                goals_valid = (
                    isinstance(data.get('goals', []), list) and
                    len(data.get('goals', [])) > 0
                )
                
                # Check if we have expected sites and categories
                sites_valid = all(site in data.get('sites', []) for site in self.expected_sites)
                categories_valid = all(cat in data.get('categories', []) for cat in self.expected_categories)
                
                success = has_required_fields and goals_valid and sites_valid and categories_valid
                
                data_summary = f"Goals: {len(data.get('goals', []))}, Sites: {len(data.get('sites', []))}, Categories: {len(data.get('categories', []))}"
                
            else:
                success = False
                data_summary = f"HTTP {response.status_code}"
            
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                data_summary=data_summary
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=0,
                success=False,
                response_time=response_time,
                error_message=str(e)
            ))
    
    def test_site_specific_goals_algonquin(self):
        """Test site-specific goals for Algonquin"""
        self._test_site_specific_goals("algonquin_illinois")
    
    def test_site_specific_goals_houston(self):
        """Test site-specific goals for Houston"""
        self._test_site_specific_goals("houston_texas")
    
    def _test_site_specific_goals(self, site_id: str):
        """Helper method to test site-specific goals"""
        endpoint = f"/api/goals/annual/{site_id}"
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ['goals', 'total_goals', 'sites', 'categories']
                has_required_fields = all(field in data for field in required_fields)
                
                # Validate that all goals are for the requested site
                goals_for_site = all(
                    goal.get('site') == site_id 
                    for goal in data.get('goals', [])
                )
                
                # Check we have goals for all expected categories
                goal_categories = {goal.get('category') for goal in data.get('goals', [])}
                has_all_categories = all(cat in goal_categories for cat in self.expected_categories)
                
                success = has_required_fields and goals_for_site and has_all_categories
                
                data_summary = f"Site: {site_id}, Goals: {len(data.get('goals', []))}, Categories: {len(goal_categories)}"
                
            else:
                success = False
                data_summary = f"HTTP {response.status_code}"
            
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                data_summary=data_summary
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=0,
                success=False,
                response_time=response_time,
                error_message=str(e)
            ))
    
    def test_invalid_site_goals(self):
        """Test goals endpoint with invalid site"""
        endpoint = "/api/goals/annual/invalid_site"
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            # Should return 404 for invalid site
            success = response.status_code == 404
            data_summary = f"Correctly returned 404 for invalid site"
            
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                data_summary=data_summary
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=0,
                success=False,
                response_time=response_time,
                error_message=str(e)
            ))
    
    def test_goals_summary_endpoint(self):
        """Test the goals summary endpoint"""
        endpoint = "/api/goals/summary"
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ['site_summary', 'category_summary', 'total_sites', 'total_categories']
                has_required_fields = all(field in data for field in required_fields)
                
                # Check site summary structure
                site_summary_valid = (
                    isinstance(data.get('site_summary', {}), dict) and
                    len(data.get('site_summary', {})) > 0
                )
                
                # Check category summary structure
                category_summary_valid = (
                    isinstance(data.get('category_summary', {}), dict) and
                    len(data.get('category_summary', {})) > 0
                )
                
                success = has_required_fields and site_summary_valid and category_summary_valid
                
                data_summary = f"Sites: {data.get('total_sites', 0)}, Categories: {data.get('total_categories', 0)}"
                
            else:
                success = False
                data_summary = f"HTTP {response.status_code}"
            
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                data_summary=data_summary
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=0,
                success=False,
                response_time=response_time,
                error_message=str(e)
            ))
    
    def test_progress_endpoint_algonquin(self):
        """Test progress endpoint for Algonquin"""
        self._test_progress_endpoint("algonquin_illinois")
    
    def test_progress_endpoint_houston(self):
        """Test progress endpoint for Houston"""
        self._test_progress_endpoint("houston_texas")
    
    def _test_progress_endpoint(self, site_id: str):
        """Helper method to test progress endpoint"""
        endpoint = f"/api/goals/progress/{site_id}"
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ['site_progress', 'overall_status', 'summary', 'calculation_date']
                has_required_fields = all(field in data for field in required_fields)
                
                # Check site progress structure
                site_progress_valid = (
                    isinstance(data.get('site_progress', []), list) and
                    len(data.get('site_progress', [])) > 0
                )
                
                # Check that progress entries have required fields
                progress_entries_valid = True
                for progress in data.get('site_progress', []):
                    required_progress_fields = ['site', 'category', 'goal', 'progress']
                    if not all(field in progress for field in required_progress_fields):
                        progress_entries_valid = False
                        break
                
                success = has_required_fields and site_progress_valid and progress_entries_valid
                
                progress_count = len(data.get('site_progress', []))
                overall_status = data.get('overall_status', 'unknown')
                data_summary = f"Site: {site_id}, Progress entries: {progress_count}, Status: {overall_status}"
                
            else:
                success = False
                data_summary = f"HTTP {response.status_code}"
            
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                data_summary=data_summary
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=0,
                success=False,
                response_time=response_time,
                error_message=str(e)
            ))
    
    def test_progress_invalid_site(self):
        """Test progress endpoint with invalid site"""
        endpoint = "/api/goals/progress/invalid_site"
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            # Should return 404 for invalid site
            success = response.status_code == 404
            data_summary = f"Correctly returned 404 for invalid site"
            
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                data_summary=data_summary
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append(TestResult(
                endpoint=endpoint,
                method="GET",
                status_code=0,
                success=False,
                response_time=response_time,
                error_message=str(e)
            ))
    
    def test_api_response_structure(self):
        """Test API response structure consistency"""
        endpoint = "/api/goals/annual"
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            success = False
            data_summary = ""
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response headers
                content_type_valid = 'application/json' in response.headers.get('content-type', '')
                
                # Check goal structure
                goal_structure_valid = True
                if data.get('goals'):
                    for goal in data['goals']:
                        required_goal_fields = ['site', 'category', 'reduction_percentage', 'unit', 'description']
                        if not all(field in goal for field in required_goal_fields):
                            goal_structure_valid = False
                            break
                
                success = content_type_valid and goal_structure_valid
                data_summary = f"Content-type valid: {content_type_valid}, Goal structure valid: {goal_structure_valid}"
            else:
                data_summary = f"HTTP {response.status_code}"
            
            self.test_results.append(TestResult(
                endpoint=f"{endpoint} (structure test)",
                method="GET",
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                data_summary=data_summary
            ))
            
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append(TestResult(
                endpoint=f"{endpoint} (structure test)",
                method="GET",
                status_code=0,
                success=False,
                response_time=response_time,
                error_message=str(e)
            ))
    
    def test_data_consistency(self):
        """Test data consistency across endpoints"""
        # This test validates that the same data is returned consistently
        # across different endpoints (annual vs site-specific)
        
        success = True
        error_messages = []
        total_response_time = 0.0
        
        try:
            # Get annual goals
            start_time = time.time()
            annual_response = self.session.get(f"{self.base_url}/api/goals/annual", timeout=self.timeout)
            total_response_time += time.time() - start_time
            
            if annual_response.status_code != 200:
                success = False
                error_messages.append("Annual goals endpoint failed")
            
            annual_data = annual_response.json() if annual_response.status_code == 200 else {}
            
            # Get site-specific goals and compare
            for site in self.expected_sites:
                start_time = time.time()
                site_response = self.session.get(f"{self.base_url}/api/goals/annual/{site}", timeout=self.timeout)
                total_response_time += time.time() - start_time
                
                if site_response.status_code != 200:
                    success = False
                    error_messages.append(f"Site {site} goals endpoint failed")
                    continue
                
                site_data = site_response.json()
                
                # Check that site-specific goals are subset of annual goals
                annual_goals_for_site = [g for g in annual_data.get('goals', []) if g.get('site') == site]
                site_goals = site_data.get('goals', [])
                
                if len(annual_goals_for_site) != len(site_goals):
                    success = False
                    error_messages.append(f"Goal count mismatch for {site}")
            
            data_summary = "Data consistent across endpoints" if success else f"Inconsistencies: {', '.join(error_messages)}"
            
        except Exception as e:
            success = False
            error_messages.append(str(e))
            data_summary = f"Exception: {str(e)}"
        
        self.test_results.append(TestResult(
            endpoint="/api/goals/* (consistency test)",
            method="GET",
            status_code=200 if success else 500,
            success=success,
            response_time=total_response_time,
            data_summary=data_summary,
            error_message='; '.join(error_messages) if error_messages else None
        ))
    
    def generate_test_summary(self):
        """Generate and display comprehensive test summary"""
        logger.info("\n" + "=" * 80)
        logger.info("EHS GOALS API TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        # Overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Detailed results table
        table_data = []
        for result in self.test_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            endpoint = result.endpoint[:50] + "..." if len(result.endpoint) > 50 else result.endpoint
            response_time = f"{result.response_time:.3f}s"
            
            table_data.append([
                status,
                result.method,
                endpoint,
                result.status_code,
                response_time,
                result.data_summary or result.error_message or "N/A"
            ])
        
        headers = ["Status", "Method", "Endpoint", "Code", "Time", "Details"]
        logger.info(f"\n{tabulate(table_data, headers=headers, tablefmt='grid')}")
        
        # API Coverage Summary
        logger.info("\n" + "=" * 80)
        logger.info("API ENDPOINT COVERAGE")
        logger.info("=" * 80)
        
        endpoint_coverage = {
            "Health Check": any("health" in r.endpoint for r in self.test_results if r.success),
            "Annual Goals": any("annual" in r.endpoint and "{site_id}" not in r.endpoint for r in self.test_results if r.success),
            "Site Goals - Algonquin": any("algonquin_illinois" in r.endpoint for r in self.test_results if r.success),
            "Site Goals - Houston": any("houston_texas" in r.endpoint for r in self.test_results if r.success),
            "Goals Summary": any("summary" in r.endpoint for r in self.test_results if r.success),
            "Progress - Algonquin": any("progress/algonquin_illinois" in r.endpoint for r in self.test_results if r.success),
            "Progress - Houston": any("progress/houston_texas" in r.endpoint for r in self.test_results if r.success),
        }
        
        for feature, working in endpoint_coverage.items():
            status = "✅ Working" if working else "❌ Not Working"
            logger.info(f"{feature:25} {status}")
        
        # Performance summary
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        
        successful_results = [r for r in self.test_results if r.success and r.response_time > 0]
        if successful_results:
            avg_response_time = sum(r.response_time for r in successful_results) / len(successful_results)
            max_response_time = max(r.response_time for r in successful_results)
            min_response_time = min(r.response_time for r in successful_results)
            
            logger.info(f"Average Response Time: {avg_response_time:.3f}s")
            logger.info(f"Maximum Response Time: {max_response_time:.3f}s")
            logger.info(f"Minimum Response Time: {min_response_time:.3f}s")
        
        # Recommendations
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 80)
        
        if failed_tests == 0:
            logger.info("✅ All tests passed! The EHS Goals API is working correctly.")
        else:
            logger.info("❌ Some tests failed. Please check the following:")
            for result in self.test_results:
                if not result.success:
                    logger.info(f"   - {result.endpoint}: {result.error_message or 'Check endpoint functionality'}")
        
        # Save detailed results to file
        self.save_results_to_file()
    
    def save_results_to_file(self):
        """Save detailed test results to JSON file"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "total_tests": len(self.test_results),
            "passed_tests": sum(1 for r in self.test_results if r.success),
            "failed_tests": sum(1 for r in self.test_results if not r.success),
            "results": [
                {
                    "endpoint": result.endpoint,
                    "method": result.method,
                    "status_code": result.status_code,
                    "success": result.success,
                    "response_time": result.response_time,
                    "error_message": result.error_message,
                    "data_summary": result.data_summary
                }
                for result in self.test_results
            ]
        }
        
        results_file = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/test_goals_api_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"\nDetailed results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results file: {e}")

def main():
    """Main function to run the test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EHS Goals API endpoints")
    parser.add_argument(
        "--base-url", 
        default="http://localhost:8000",
        help="Base URL for the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create tester instance
    tester = EHSGoalsAPITester(base_url=args.base_url)
    
    # Check if server is running
    logger.info(f"Testing EHS Goals API at: {args.base_url}")
    try:
        response = tester.session.get(f"{args.base_url}/api/goals/health", timeout=5)
        logger.info("✅ Server is responding")
    except Exception as e:
        logger.error(f"❌ Cannot connect to server: {e}")
        logger.error("Please ensure the API server is running")
        sys.exit(1)
    
    # Run all tests
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
