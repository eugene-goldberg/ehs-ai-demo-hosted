#!/usr/bin/env python3
"""
Test script for LangSmith integration with batch ingestion endpoint.

This script:
1. Makes a request to the batch ingestion endpoint
2. Waits for it to complete
3. Extracts and displays the LangSmith traces from the response
4. Verifies the traces are in the correct format for the frontend

Requirements:
- Backend API server running on localhost:8000
- Valid LangSmith API key in environment
- Neo4j database accessible
"""

import os
import sys
import json
import time
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add the backend src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
src_dir = os.path.join(backend_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for the test script."""
    api_base_url: str = "http://localhost:8000"
    batch_endpoint: str = "/api/v1/ingest/batch"
    health_endpoint: str = "/health"
    timeout_seconds: int = 300  # 5 minutes
    poll_interval_seconds: int = 5
    clear_database: bool = True
    

class LangSmithTraceAnalyzer:
    """Analyzer for LangSmith trace data."""
    
    def __init__(self):
        self.expected_trace_fields = [
            'trace_id',
            'run_id', 
            'session_id',
            'project_name',
            'operation_type',
            'status',
            'start_time',
            'end_time',
            'duration',
            'metadata'
        ]
    
    def analyze_traces(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze LangSmith traces from the batch ingestion response.
        
        Args:
            response_data: Response data from batch ingestion endpoint
            
        Returns:
            Dictionary with trace analysis results
        """
        analysis = {
            'langsmith_available': False,
            'traces_found': False,
            'trace_count': 0,
            'valid_traces': 0,
            'invalid_traces': 0,
            'trace_details': [],
            'errors': [],
            'session_info': {}
        }
        
        try:
            # Check if LangSmith data is present in response
            metadata = response_data.get('metadata', {})
            
            # Look for LangSmith session information
            if 'langsmith_session' in metadata:
                analysis['langsmith_available'] = True
                analysis['session_info'] = metadata['langsmith_session']
                logger.info(f"Found LangSmith session: {analysis['session_info']}")
            
            # Look for trace data in various locations
            traces = []
            
            # Check metadata for traces
            if 'traces' in metadata:
                traces.extend(metadata['traces'])
            
            # Check data section for traces
            data = response_data.get('data', {})
            if 'traces' in data:
                traces.extend(data['traces'])
            
            # Check for langsmith_traces specifically
            if 'langsmith_traces' in data:
                traces.extend(data['langsmith_traces'])
            
            if traces:
                analysis['traces_found'] = True
                analysis['trace_count'] = len(traces)
                
                # Analyze each trace
                for trace in traces:
                    trace_analysis = self._analyze_single_trace(trace)
                    analysis['trace_details'].append(trace_analysis)
                    
                    if trace_analysis['valid']:
                        analysis['valid_traces'] += 1
                    else:
                        analysis['invalid_traces'] += 1
            else:
                logger.warning("No traces found in response data")
                analysis['errors'].append("No traces found in response")
                
        except Exception as e:
            error_msg = f"Error analyzing traces: {str(e)}"
            logger.error(error_msg)
            analysis['errors'].append(error_msg)
        
        return analysis
    
    def _analyze_single_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single trace for validity and completeness.
        
        Args:
            trace: Individual trace data
            
        Returns:
            Dictionary with trace analysis
        """
        trace_analysis = {
            'valid': True,
            'missing_fields': [],
            'present_fields': [],
            'trace_id': trace.get('trace_id', 'unknown'),
            'run_id': trace.get('run_id', 'unknown'),
            'operation_type': trace.get('operation_type', 'unknown'),
            'status': trace.get('status', 'unknown'),
            'duration': trace.get('duration'),
            'errors': []
        }
        
        # Check for required fields
        for field in self.expected_trace_fields:
            if field in trace:
                trace_analysis['present_fields'].append(field)
            else:
                trace_analysis['missing_fields'].append(field)
                if field in ['trace_id', 'run_id', 'status']:  # Critical fields
                    trace_analysis['valid'] = False
        
        # Validate data types and formats
        if 'start_time' in trace:
            try:
                datetime.fromisoformat(trace['start_time'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                trace_analysis['errors'].append("Invalid start_time format")
                trace_analysis['valid'] = False
        
        if 'end_time' in trace:
            try:
                datetime.fromisoformat(trace['end_time'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                trace_analysis['errors'].append("Invalid end_time format")
                trace_analysis['valid'] = False
        
        return trace_analysis


class BatchIngestionTester:
    """Test harness for batch ingestion with LangSmith tracing."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.trace_analyzer = LangSmithTraceAnalyzer()
        self.session = requests.Session()
        
    def run_complete_test(self) -> Dict[str, Any]:
        """
        Run the complete test workflow.
        
        Returns:
            Dictionary with complete test results
        """
        test_results = {
            'test_id': f"langsmith_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.utcnow().isoformat(),
            'config': {
                'api_base_url': self.config.api_base_url,
                'clear_database': self.config.clear_database,
                'timeout_seconds': self.config.timeout_seconds
            },
            'steps': {
                'health_check': {'completed': False, 'success': False, 'details': {}},
                'batch_ingestion': {'completed': False, 'success': False, 'details': {}},
                'trace_analysis': {'completed': False, 'success': False, 'details': {}},
                'frontend_verification': {'completed': False, 'success': False, 'details': {}}
            },
            'overall_success': False,
            'errors': []
        }
        
        try:
            # Step 1: Health check
            logger.info("Step 1: Performing health check...")
            test_results['steps']['health_check'] = self._health_check()
            
            if not test_results['steps']['health_check']['success']:
                raise Exception("Health check failed - cannot proceed with test")
            
            # Step 2: Run batch ingestion
            logger.info("Step 2: Starting batch ingestion...")
            test_results['steps']['batch_ingestion'] = self._run_batch_ingestion()
            
            if not test_results['steps']['batch_ingestion']['success']:
                raise Exception("Batch ingestion failed")
            
            # Step 3: Analyze LangSmith traces
            logger.info("Step 3: Analyzing LangSmith traces...")
            test_results['steps']['trace_analysis'] = self._analyze_traces(
                test_results['steps']['batch_ingestion']['details']
            )
            
            # Step 4: Verify frontend compatibility
            logger.info("Step 4: Verifying frontend compatibility...")
            test_results['steps']['frontend_verification'] = self._verify_frontend_format(
                test_results['steps']['trace_analysis']['details']
            )
            
            # Overall success determination
            test_results['overall_success'] = all(
                step['success'] for step in test_results['steps'].values()
            )
            
        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}"
            logger.error(error_msg)
            test_results['errors'].append(error_msg)
            test_results['overall_success'] = False
        
        test_results['end_time'] = datetime.utcnow().isoformat()
        return test_results
    
    def _health_check(self) -> Dict[str, Any]:
        """Perform health check on the API."""
        result = {'completed': False, 'success': False, 'details': {}}
        
        try:
            url = f"{self.config.api_base_url}{self.config.health_endpoint}"
            response = self.session.get(url, timeout=10)
            
            result['details'] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None
            }
            
            if response.status_code == 200:
                health_data = response.json()
                result["success"] = health_data.get("components", {}).get("database", {}).get("healthy", False)
                if not result['success']:
                    result['details']['error'] = "Database connection failed"
            else:
                result['details']['error'] = f"Health check returned {response.status_code}"
            
            result['completed'] = True
            
        except Exception as e:
            result['details']['error'] = str(e)
            result['completed'] = True
        
        return result
    
    def _run_batch_ingestion(self) -> Dict[str, Any]:
        """Run batch ingestion and wait for completion."""
        result = {'completed': False, 'success': False, 'details': {}}
        
        try:
            url = f"{self.config.api_base_url}{self.config.batch_endpoint}"
            payload = {
                'clear_database': self.config.clear_database
            }
            
            logger.info(f"Sending batch ingestion request to {url}")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            
            start_time = time.time()
            response = self.session.post(
                url, 
                json=payload,
                timeout=self.config.timeout_seconds
            )
            end_time = time.time()
            
            result['details'] = {
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'request_payload': payload,
                'response_headers': dict(response.headers),
                'response_data': response.json() if response.status_code == 200 else response.text
            }
            
            if response.status_code == 200:
                response_data = response.json()
                result['success'] = response_data.get('status') == 'success'
                if not result['success']:
                    result['details']['ingestion_errors'] = response_data.get('errors', [])
            else:
                result['details']['error'] = f"Request failed with status {response.status_code}"
            
            result['completed'] = True
            
        except Exception as e:
            result['details']['error'] = str(e)
            result['completed'] = True
        
        return result
    
    def _analyze_traces(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze LangSmith traces from batch ingestion response."""
        result = {'completed': False, 'success': False, 'details': {}}
        
        try:
            if 'response_data' not in batch_result:
                raise Exception("No response data available for trace analysis")
            
            response_data = batch_result['response_data']
            trace_analysis = self.trace_analyzer.analyze_traces(response_data)
            
            result['details'] = trace_analysis
            result['success'] = (
                trace_analysis['langsmith_available'] and 
                trace_analysis['traces_found'] and
                trace_analysis['valid_traces'] > 0
            )
            result['completed'] = True
            
        except Exception as e:
            result['details']['error'] = str(e)
            result['completed'] = True
        
        return result
    
    def _verify_frontend_format(self, trace_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Verify traces are in correct format for frontend consumption."""
        result = {'completed': False, 'success': False, 'details': {}}
        
        try:
            frontend_checks = {
                'has_session_info': 'session_info' in trace_analysis and trace_analysis['session_info'],
                'has_valid_traces': trace_analysis.get('valid_traces', 0) > 0,
                'trace_fields_complete': True,
                'json_serializable': True
            }
            
            # Check trace field completeness
            if 'trace_details' in trace_analysis:
                for trace in trace_analysis['trace_details']:
                    if len(trace.get('missing_fields', [])) > 2:  # Allow some missing optional fields
                        frontend_checks['trace_fields_complete'] = False
                        break
            
            # Test JSON serialization
            try:
                json.dumps(trace_analysis)
            except Exception:
                frontend_checks['json_serializable'] = False
            
            result['details'] = {
                'frontend_checks': frontend_checks,
                'recommendations': self._generate_frontend_recommendations(trace_analysis)
            }
            
            result['success'] = all(frontend_checks.values())
            result['completed'] = True
            
        except Exception as e:
            result['details']['error'] = str(e)
            result['completed'] = True
        
        return result
    
    def _generate_frontend_recommendations(self, trace_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for frontend integration."""
        recommendations = []
        
        if not trace_analysis.get('langsmith_available'):
            recommendations.append("Enable LangSmith tracing by setting LANGSMITH_API_KEY")
        
        if not trace_analysis.get('traces_found'):
            recommendations.append("Ensure trace data is included in API response")
        
        if trace_analysis.get('invalid_traces', 0) > 0:
            recommendations.append("Fix invalid trace data format issues")
        
        if not trace_analysis.get('session_info'):
            recommendations.append("Include session information in response metadata")
        
        return recommendations


def print_test_summary(test_results: Dict[str, Any]):
    """Print a formatted summary of test results."""
    print("\n" + "="*80)
    print("LANGSMITH INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Test ID: {test_results['test_id']}")
    print(f"Overall Success: {'✅ PASS' if test_results['overall_success'] else '❌ FAIL'}")
    print(f"Start Time: {test_results['start_time']}")
    print(f"End Time: {test_results.get('end_time', 'N/A')}")
    
    print("\nSTEP RESULTS:")
    print("-" * 40)
    for step_name, step_result in test_results['steps'].items():
        status = "✅ PASS" if step_result['success'] else "❌ FAIL"
        completed = "✓" if step_result['completed'] else "✗"
        print(f"{step_name:20} [{completed}] {status}")
    
    print("\nTRACE ANALYSIS DETAILS:")
    print("-" * 40)
    if 'trace_analysis' in test_results['steps']:
        trace_details = test_results['steps']['trace_analysis']['details']
        print(f"LangSmith Available: {trace_details.get('langsmith_available', 'N/A')}")
        print(f"Traces Found: {trace_details.get('traces_found', 'N/A')}")
        print(f"Total Traces: {trace_details.get('trace_count', 0)}")
        print(f"Valid Traces: {trace_details.get('valid_traces', 0)}")
        print(f"Invalid Traces: {trace_details.get('invalid_traces', 0)}")
        
        if trace_details.get('session_info'):
            print(f"Session Project: {trace_details['session_info']}")
    
    print("\nFRONTEND COMPATIBILITY:")
    print("-" * 40)
    if 'frontend_verification' in test_results['steps']:
        frontend_details = test_results['steps']['frontend_verification']['details']
        if 'frontend_checks' in frontend_details:
            for check_name, passed in frontend_details['frontend_checks'].items():
                status = "✅" if passed else "❌"
                print(f"{check_name:25} {status}")
        
        if 'recommendations' in frontend_details and frontend_details['recommendations']:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(frontend_details['recommendations'], 1):
                print(f"  {i}. {rec}")
    
    if test_results.get('errors'):
        print("\nERRORS:")
        print("-" * 40)
        for error in test_results['errors']:
            print(f"  • {error}")
    
    print("\n" + "="*80)


def main():
    """Main test execution function."""
    print("Starting LangSmith Integration Test")
    print("=" * 50)
    
    # Check environment
    if not os.getenv('LANGSMITH_API_KEY'):
        print("⚠️  Warning: LANGSMITH_API_KEY not found in environment")
        print("   This test will still run but LangSmith features may not work")
    else:
        print("✓ LANGSMITH_API_KEY found in environment")
    
    # Initialize test configuration
    config = TestConfig()
    
    # Allow command line override of base URL
    if len(sys.argv) > 1:
        config.api_base_url = sys.argv[1]
        print(f"Using API base URL: {config.api_base_url}")
    
    # Run the test
    tester = BatchIngestionTester(config)
    test_results = tester.run_complete_test()
    
    # Print summary
    print_test_summary(test_results)
    
    # Save detailed results to file
    results_file = f"langsmith_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if test_results['overall_success'] else 1)


if __name__ == "__main__":
    main()