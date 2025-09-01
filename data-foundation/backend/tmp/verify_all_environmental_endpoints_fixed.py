#!/usr/bin/env python3
"""
Comprehensive Environmental Endpoints Verification Script

This script tests all 9 environmental endpoints to verify they work correctly
after schema mapping fixes:
- electricity/{facts, risks, recommendations}
- water/{facts, risks, recommendations} 
- waste/{facts, risks, recommendations}

Also tests:
- LLM assessment endpoint with correct request format
- Location filtering functionality
- Response structure validation
"""

import requests
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration
BASE_URL = "http://localhost:8000"
LOG_FILE = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/endpoint_verification_results.log"
SUMMARY_FILE = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verification_summary.json"

class EndpointVerifier:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": BASE_URL,
            "endpoints_tested": 0,
            "endpoints_passed": 0,
            "endpoints_failed": 0,
            "detailed_results": []
        }
        self.log_entries = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_entries.append(log_entry)
        print(log_entry)
        
    def test_endpoint(self, endpoint: str, params: Dict = None, expected_fields: List[str] = None) -> Dict[str, Any]:
        """Test a single endpoint and return results"""
        self.log(f"Testing endpoint: {endpoint}")
        
        result = {
            "endpoint": endpoint,
            "params": params or {},
            "status": "UNKNOWN",
            "response_time_ms": 0,
            "status_code": 0,
            "data_count": 0,
            "expected_fields_present": [],
            "missing_fields": [],
            "error": None,
            "sample_data": None
        }
        
        try:
            start_time = time.time()
            
            # Make request
            if params:
                response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=30)
            else:
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
                
            end_time = time.time()
            result["response_time_ms"] = round((end_time - start_time) * 1000, 2)
            result["status_code"] = response.status_code
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if it's a list or dict with data
                if isinstance(data, list):
                    result["data_count"] = len(data)
                    sample_data = data[0] if data else None
                elif isinstance(data, dict):
                    # Handle different response structures
                    if "data" in data:
                        result["data_count"] = len(data["data"]) if isinstance(data["data"], list) else 1
                        sample_data = data["data"][0] if isinstance(data["data"], list) and data["data"] else data["data"]
                    elif "results" in data:
                        result["data_count"] = len(data["results"]) if isinstance(data["results"], list) else 1
                        sample_data = data["results"][0] if isinstance(data["results"], list) and data["results"] else data["results"]
                    else:
                        result["data_count"] = 1
                        sample_data = data
                else:
                    result["data_count"] = 1 if data else 0
                    sample_data = data
                
                # Store sample for analysis
                if sample_data:
                    result["sample_data"] = sample_data
                    
                    # Check expected fields if provided
                    if expected_fields and isinstance(sample_data, dict):
                        for field in expected_fields:
                            if field in sample_data:
                                result["expected_fields_present"].append(field)
                            else:
                                result["missing_fields"].append(field)
                
                # Determine status
                if result["data_count"] > 0:
                    result["status"] = "PASSED"
                    self.log(f"✓ {endpoint} - SUCCESS ({result['data_count']} items, {result['response_time_ms']}ms)")
                else:
                    result["status"] = "FAILED"
                    result["error"] = "No data returned"
                    self.log(f"✗ {endpoint} - FAILED: No data returned")
                    
            else:
                result["status"] = "FAILED"
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                self.log(f"✗ {endpoint} - FAILED: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            result["status"] = "FAILED"
            result["error"] = f"Request failed: {str(e)}"
            self.log(f"✗ {endpoint} - FAILED: {str(e)}")
        except Exception as e:
            result["status"] = "FAILED"
            result["error"] = f"Unexpected error: {str(e)}"
            self.log(f"✗ {endpoint} - FAILED: Unexpected error: {str(e)}")
            
        return result
    
    def test_environmental_endpoints(self):
        """Test all 9 environmental endpoints"""
        self.log("=== Testing Environmental Data Endpoints ===")
        
        categories = ["electricity", "water", "waste"]
        types = ["facts", "risks", "recommendations"]
        
        # Expected fields for environmental data
        expected_fields = ["id", "category", "title", "description"]
        
        for category in categories:
            for data_type in types:
                endpoint = f"/api/environmental/{category}/{data_type}"
                
                # Test without location filter
                result = self.test_endpoint(endpoint, expected_fields=expected_fields)
                self.results["detailed_results"].append(result)
                self.results["endpoints_tested"] += 1
                
                if result["status"] == "PASSED":
                    self.results["endpoints_passed"] += 1
                else:
                    self.results["endpoints_failed"] += 1
                
                # Test with location filter (if endpoint supports it)
                if result["status"] == "PASSED":
                    location_result = self.test_endpoint(
                        endpoint, 
                        params={"location_path": "california"}, 
                        expected_fields=expected_fields
                    )
                    location_result["endpoint"] = f"{endpoint}?location_path=california"
                    self.results["detailed_results"].append(location_result)
                    self.results["endpoints_tested"] += 1
                    
                    if location_result["status"] == "PASSED":
                        self.results["endpoints_passed"] += 1
                    else:
                        self.results["endpoints_failed"] += 1
    
    def test_llm_assessment(self):
        """Test LLM assessment endpoint"""
        self.log("=== Testing LLM Assessment Endpoint ===")
        
        endpoint = "/api/environmental/llm-assessment"
        test_cases = [
            {
                "payload": {
                    "categories": ["electricity"],
                    "custom_prompt": "What are the main electricity consumption risks for data centers?"
                },
                "name": "Electricity assessment"
            },
            {
                "payload": {
                    "categories": ["water"], 
                    "location_path": "california",
                    "custom_prompt": "How can we reduce water waste in manufacturing?"
                },
                "name": "Water assessment with location"
            },
            {
                "payload": {
                    "categories": ["waste"],
                    "custom_prompt": "What are best practices for waste management?"
                },
                "name": "Waste assessment"
            }
        ]
        
        for test_case in test_cases:
            self.log(f"Testing: {test_case['name']}")
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{BASE_URL}{endpoint}",
                    json=test_case["payload"],
                    timeout=60  # LLM endpoints may take longer
                )
                end_time = time.time()
                
                result = {
                    "endpoint": f"{endpoint} ({test_case['name']})",
                    "params": test_case["payload"],
                    "status": "UNKNOWN",
                    "response_time_ms": round((end_time - start_time) * 1000, 2),
                    "status_code": response.status_code,
                    "error": None,
                    "response_length": 0
                }
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for assessment response
                    if isinstance(data, dict) and ("assessment_id" in data or "summary" in data):
                        summary_text = data.get("summary", "")
                        result["response_length"] = len(summary_text)
                        
                        if len(summary_text) > 50:  # Reasonable response length
                            result["status"] = "PASSED"
                            self.log(f"✓ {test_case['name']} - SUCCESS ({result['response_length']} chars, {result['response_time_ms']}ms)")
                        else:
                            result["status"] = "FAILED"
                            result["error"] = "Response summary too short"
                            self.log(f"✗ {test_case['name']} - FAILED: Response summary too short")
                    else:
                        result["status"] = "FAILED"
                        result["error"] = "Invalid response format"
                        self.log(f"✗ {test_case['name']} - FAILED: Invalid response format")
                else:
                    result["status"] = "FAILED"
                    result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                    self.log(f"✗ {test_case['name']} - FAILED: HTTP {response.status_code}")
                
            except Exception as e:
                result = {
                    "endpoint": f"{endpoint} ({test_case['name']})",
                    "params": test_case["payload"],
                    "status": "FAILED",
                    "response_time_ms": 0,
                    "status_code": 0,
                    "error": f"Request failed: {str(e)}",
                    "response_length": 0
                }
                self.log(f"✗ {test_case['name']} - FAILED: {str(e)}")
            
            self.results["detailed_results"].append(result)
            self.results["endpoints_tested"] += 1
            
            if result["status"] == "PASSED":
                self.results["endpoints_passed"] += 1
            else:
                self.results["endpoints_failed"] += 1
    
    def test_server_health(self):
        """Test if server is running"""
        self.log("=== Testing Server Health ===")
        
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                self.log("✓ Server is running")
                return True
            else:
                self.log(f"✗ Server health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log(f"✗ Cannot connect to server: {str(e)}")
            return False
    
    def generate_summary(self):
        """Generate human-readable summary"""
        self.log("=== VERIFICATION SUMMARY ===")
        
        total = self.results["endpoints_tested"]
        passed = self.results["endpoints_passed"]
        failed = self.results["endpoints_failed"]
        
        self.log(f"Total endpoints tested: {total}")
        self.log(f"Passed: {passed}")
        self.log(f"Failed: {failed}")
        self.log(f"Success rate: {(passed/total*100):.1f}%" if total > 0 else "No tests run")
        
        # Group results by category
        categories = {}
        for result in self.results["detailed_results"]:
            endpoint = result["endpoint"]
            
            if "/api/environmental/" in endpoint:
                if "electricity" in endpoint:
                    cat = "Electricity"
                elif "water" in endpoint:
                    cat = "Water"
                elif "waste" in endpoint:
                    cat = "Waste"
                elif "llm-assessment" in endpoint:
                    cat = "LLM Assessment"
                else:
                    cat = "Other"
            else:
                cat = "Other"
            
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0, "total": 0}
            
            categories[cat]["total"] += 1
            if result["status"] == "PASSED":
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
        
        self.log("\n=== Results by Category ===")
        for cat, stats in categories.items():
            status = "✓" if stats["failed"] == 0 else "✗"
            self.log(f"{status} {cat}: {stats['passed']}/{stats['total']} passed")
        
        # List failed endpoints
        failed_endpoints = [r for r in self.results["detailed_results"] if r["status"] == "FAILED"]
        if failed_endpoints:
            self.log("\n=== Failed Endpoints ===")
            for result in failed_endpoints:
                self.log(f"✗ {result['endpoint']}: {result['error']}")
        
        self.log(f"\nDetailed results saved to: {SUMMARY_FILE}")
        self.log(f"Full logs saved to: {LOG_FILE}")
    
    def save_results(self):
        """Save results to files"""
        # Save JSON summary
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save log file
        with open(LOG_FILE, 'w') as f:
            f.write('\n'.join(self.log_entries))
    
    def run_verification(self):
        """Run complete verification suite"""
        self.log("Starting comprehensive environmental endpoints verification")
        self.log(f"Base URL: {BASE_URL}")
        
        # Check server health first
        if not self.test_server_health():
            self.log("Server is not accessible. Please ensure the FastAPI server is running on port 8000.")
            return False
        
        # Test all endpoints
        self.test_environmental_endpoints()
        self.test_llm_assessment()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.results["endpoints_failed"] == 0

def main():
    """Main execution function"""
    print("Environmental Endpoints Verification Script")
    print("=" * 50)
    
    verifier = EndpointVerifier()
    success = verifier.run_verification()
    
    if success:
        print("\n🎉 All endpoints are working correctly!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some endpoints failed. Check {LOG_FILE} for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
