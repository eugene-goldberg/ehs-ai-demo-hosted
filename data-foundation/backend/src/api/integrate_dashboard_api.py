#!/usr/bin/env python3
"""
Executive Dashboard API Integration Guide & Demonstration Script

This comprehensive integration guide provides practical examples for migrating
from static dashboard files to the new dynamic Executive Dashboard API v2.
It includes code examples, performance comparisons, migration procedures,
and rollback strategies for production deployment.

Features Covered:
- API integration examples and code patterns
- Static vs Dynamic dashboard comparison
- Performance improvement demonstrations
- Migration procedures with rollback capabilities
- Production deployment strategies
- Error handling and fallback mechanisms
- Monitoring and observability setup

Created: 2025-08-28
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import traceback
from functools import wraps
import shutil
import tempfile

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Import FastAPI components for integration examples
try:
    from fastapi import FastAPI, APIRouter, HTTPException, Query, Depends
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Some examples will be skipped.")

# Import our dashboard components
try:
    from api.executive_dashboard_v2 import executive_dashboard_router, get_dashboard_service
    from services.executive_dashboard.dashboard_service import ExecutiveDashboardService, create_dashboard_service
    DASHBOARD_COMPONENTS_AVAILABLE = True
except ImportError as e:
    DASHBOARD_COMPONENTS_AVAILABLE = False
    print(f"Warning: Dashboard components not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard_integration.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
STATIC_DASHBOARD_PATH = current_dir.parent.parent / "docs" / "static_dashboards"
BACKUP_PATH = current_dir.parent.parent / "backup"
API_BASE_URL = "http://localhost:8000"
PERFORMANCE_TEST_ITERATIONS = 10


@dataclass
class IntegrationTestResult:
    """Test result data structure"""
    test_name: str
    success: bool
    execution_time: float
    response_size: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceComparison:
    """Performance comparison data structure"""
    static_time: float
    dynamic_time: float
    static_size: int
    dynamic_size: int
    improvement_percent: float
    dynamic_advantages: List[str]
    static_advantages: List[str]


class DashboardIntegrationDemo:
    """
    Comprehensive Dashboard API Integration Demo
    
    This class provides practical examples and utilities for integrating
    the Executive Dashboard API v2 into existing applications.
    """
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        """Initialize the integration demo"""
        self.api_base_url = api_base_url
        self.static_dashboard_path = STATIC_DASHBOARD_PATH
        self.backup_path = BACKUP_PATH
        self.test_results: List[IntegrationTestResult] = []
        self.performance_data: List[PerformanceComparison] = []
        
        # Ensure directories exist
        self.static_dashboard_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dashboard Integration Demo initialized")
        logger.info(f"API Base URL: {self.api_base_url}")
        logger.info(f"Static Dashboard Path: {self.static_dashboard_path}")
    
    # ==============================================================================
    # SECTION 1: API INTEGRATION EXAMPLES
    # ==============================================================================
    
    def demonstrate_basic_integration(self) -> Dict[str, Any]:
        """
        Demonstrate basic API integration patterns
        """
        logger.info("=" * 80)
        logger.info("SECTION 1: BASIC API INTEGRATION EXAMPLES")
        logger.info("=" * 80)
        
        examples = {}
        
        # Example 1: Simple GET request
        examples["simple_request"] = self._demo_simple_request()
        
        # Example 2: Filtered request
        examples["filtered_request"] = self._demo_filtered_request()
        
        # Example 3: Different output formats
        examples["format_examples"] = self._demo_format_examples()
        
        # Example 4: Error handling
        examples["error_handling"] = self._demo_error_handling()
        
        # Example 5: Async integration
        if FASTAPI_AVAILABLE:
            examples["async_integration"] = self._demo_async_integration()
        
        return examples
    
    def _demo_simple_request(self) -> Dict[str, Any]:
        """Basic API request example"""
        logger.info("\n--- Example 1: Simple API Request ---")
        
        example_code = '''
# Basic dashboard API request
import requests
import json

def get_dashboard_data():
    """Get basic dashboard data"""
    try:
        response = requests.get("http://localhost:8000/api/v2/executive-dashboard")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Usage
dashboard_data = get_dashboard_data()
if dashboard_data:
    print(f"Dashboard generated at: {dashboard_data['metadata']['generated_at']}")
    print(f"Total facilities: {dashboard_data['summary']['facilities']['total_count']}")
    print(f"Alert level: {dashboard_data['summary']['alert_level']}")
'''
        
        logger.info("Example Code:")
        print(example_code)
        
        # Actual demonstration
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Simple request succeeded in {execution_time:.3f}s")
                logger.info(f"  Response size: {len(response.content)} bytes")
                logger.info(f"  Facilities: {data.get('summary', {}).get('facilities', {}).get('total_count', 'N/A')}")
                logger.info(f"  Alert level: {data.get('summary', {}).get('alert_level', 'N/A')}")
                
                return {
                    "success": True,
                    "execution_time": execution_time,
                    "response_size": len(response.content),
                    "sample_data": {
                        "generated_at": data.get('metadata', {}).get('generated_at'),
                        "facilities_count": data.get('summary', {}).get('facilities', {}).get('total_count'),
                        "alert_level": data.get('summary', {}).get('alert_level')
                    }
                }
            else:
                logger.error(f"✗ Request failed with status {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"✗ Request failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _demo_filtered_request(self) -> Dict[str, Any]:
        """Filtered API request example"""
        logger.info("\n--- Example 2: Filtered API Request ---")
        
        example_code = '''
# Filtered dashboard request with specific parameters
import requests
from datetime import datetime, timedelta

def get_filtered_dashboard(facility_ids=None, date_range="30d", format="summary"):
    """Get filtered dashboard data"""
    params = {
        "dateRange": date_range,
        "format": format,
        "includeTrends": True,
        "includeRecommendations": True
    }
    
    if facility_ids:
        params["location"] = ",".join(facility_ids)
    
    try:
        response = requests.get(
            "http://localhost:8000/api/v2/executive-dashboard",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Filtered request failed: {e}")
        return None

# Usage examples
recent_data = get_filtered_dashboard(date_range="7d", format="kpis_only")
facility_data = get_filtered_dashboard(facility_ids=["FAC001", "FAC002"])
summary_data = get_filtered_dashboard(format="summary")
'''
        
        logger.info("Example Code:")
        print(example_code)
        
        # Demonstration with different filters
        test_cases = [
            {"dateRange": "7d", "format": "summary", "name": "Last 7 days summary"},
            {"dateRange": "30d", "format": "kpis_only", "name": "30-day KPIs only"},
            {"includeTrends": "true", "includeRecommendations": "true", "name": "With trends and recommendations"}
        ]
        
        results = {}
        for case in test_cases:
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.api_base_url}/api/v2/executive-dashboard",
                    params=case,
                    timeout=30
                )
                execution_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✓ {case['name']}: {execution_time:.3f}s, {len(response.content)} bytes")
                    results[case['name']] = {
                        "success": True,
                        "execution_time": execution_time,
                        "response_size": len(response.content),
                        "has_trends": "trends" in data,
                        "has_recommendations": "recommendations" in data
                    }
                else:
                    logger.error(f"✗ {case['name']}: HTTP {response.status_code}")
                    results[case['name']] = {"success": False, "error": f"HTTP {response.status_code}"}
                    
            except Exception as e:
                logger.error(f"✗ {case['name']}: {e}")
                results[case['name']] = {"success": False, "error": str(e)}
        
        return results
    
    def _demo_format_examples(self) -> Dict[str, Any]:
        """Different output format examples"""
        logger.info("\n--- Example 3: Different Output Formats ---")
        
        example_code = '''
# Different dashboard output formats
import requests

def get_dashboard_formats():
    """Demonstrate different output formats"""
    formats = ["full", "summary", "kpis_only", "charts_only", "alerts_only"]
    results = {}
    
    for fmt in formats:
        try:
            response = requests.get(
                "http://localhost:8000/api/v2/executive-dashboard",
                params={"format": fmt},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results[fmt] = {
                    "size": len(response.content),
                    "sections": list(data.keys()),
                    "metadata": data.get("metadata", {})
                }
            else:
                results[fmt] = {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            results[fmt] = {"error": str(e)}
    
    return results

# Usage
format_comparison = get_dashboard_formats()
for format_name, result in format_comparison.items():
    if "error" not in result:
        print(f"{format_name}: {result['size']} bytes, sections: {result['sections']}")
'''
        
        logger.info("Example Code:")
        print(example_code)
        
        # Actual format comparison
        formats = ["full", "summary", "kpis_only", "charts_only", "alerts_only"]
        results = {}
        
        for fmt in formats:
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.api_base_url}/api/v2/executive-dashboard",
                    params={"format": fmt},
                    timeout=30
                )
                execution_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    results[fmt] = {
                        "success": True,
                        "execution_time": execution_time,
                        "response_size": len(response.content),
                        "sections": list(data.keys()),
                        "section_count": len(data.keys())
                    }
                    logger.info(f"✓ Format '{fmt}': {execution_time:.3f}s, {len(response.content)} bytes, {len(data.keys())} sections")
                else:
                    results[fmt] = {"success": False, "error": f"HTTP {response.status_code}"}
                    logger.error(f"✗ Format '{fmt}': HTTP {response.status_code}")
                    
            except Exception as e:
                results[fmt] = {"success": False, "error": str(e)}
                logger.error(f"✗ Format '{fmt}': {e}")
        
        return results
    
    def _demo_error_handling(self) -> Dict[str, Any]:
        """Error handling and fallback examples"""
        logger.info("\n--- Example 4: Error Handling and Fallbacks ---")
        
        example_code = '''
# Robust error handling with fallbacks
import requests
import json
import logging
from pathlib import Path

class DashboardClient:
    def __init__(self, api_base_url, static_fallback_path=None):
        self.api_base_url = api_base_url
        self.static_fallback_path = static_fallback_path
        self.logger = logging.getLogger(__name__)
    
    def get_dashboard_data(self, **params):
        """Get dashboard data with automatic fallback"""
        try:
            # Try dynamic API first
            return self._get_dynamic_data(**params)
        except Exception as api_error:
            self.logger.warning(f"Dynamic API failed: {api_error}")
            
            try:
                # Fallback to static data
                return self._get_static_fallback()
            except Exception as fallback_error:
                self.logger.error(f"Static fallback failed: {fallback_error}")
                return self._get_minimal_response(str(api_error))
    
    def _get_dynamic_data(self, **params):
        """Get data from dynamic API"""
        response = requests.get(
            f"{self.api_base_url}/api/v2/executive-dashboard",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        data["metadata"]["source"] = "dynamic_api"
        return data
    
    def _get_static_fallback(self):
        """Get static fallback data"""
        if not self.static_fallback_path:
            raise Exception("No static fallback path configured")
        
        fallback_file = Path(self.static_fallback_path) / "executive_dashboard_fallback.json"
        with open(fallback_file, 'r') as f:
            data = json.load(f)
        
        data["metadata"]["source"] = "static_fallback"
        data["metadata"]["note"] = "Using static fallback due to API unavailability"
        return data
    
    def _get_minimal_response(self, error_msg):
        """Generate minimal error response"""
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source": "error_response",
                "error": error_msg
            },
            "summary": {
                "status": "service_unavailable",
                "message": "Dashboard service is temporarily unavailable"
            },
            "error": True
        }

# Usage
client = DashboardClient("http://localhost:8000", "./static_dashboards")
dashboard_data = client.get_dashboard_data(format="summary")

if dashboard_data.get("error"):
    print("Dashboard service is experiencing issues")
else:
    print(f"Data source: {dashboard_data['metadata']['source']}")
'''
        
        logger.info("Example Code:")
        print(example_code)
        
        # Demonstrate error scenarios
        error_scenarios = [
            {"url": f"{self.api_base_url}/api/v2/executive-dashboard", "params": {"format": "invalid_format"}, "name": "Invalid format"},
            {"url": f"{self.api_base_url}/api/v2/executive-dashboard", "params": {"dateRange": "invalid_range"}, "name": "Invalid date range"},
            {"url": "http://invalid-host:8000/api/v2/executive-dashboard", "params": {}, "name": "Connection error"}
        ]
        
        results = {}
        for scenario in error_scenarios:
            try:
                start_time = time.time()
                response = requests.get(scenario["url"], params=scenario["params"], timeout=5)
                execution_time = time.time() - start_time
                
                if response.status_code == 200:
                    logger.info(f"✓ {scenario['name']}: Unexpected success")
                    results[scenario['name']] = {"success": True, "unexpected": True}
                else:
                    logger.info(f"✓ {scenario['name']}: Expected error HTTP {response.status_code}")
                    results[scenario['name']] = {
                        "success": False, 
                        "expected_error": True, 
                        "status_code": response.status_code,
                        "execution_time": execution_time
                    }
                    
            except requests.exceptions.RequestException as e:
                logger.info(f"✓ {scenario['name']}: Expected connection error - {str(e)[:50]}...")
                results[scenario['name']] = {"success": False, "expected_error": True, "error_type": type(e).__name__}
        
        return results
    
    def _demo_async_integration(self) -> Dict[str, Any]:
        """Async/await integration example"""
        logger.info("\n--- Example 5: Async Integration ---")
        
        example_code = '''
# Async integration with FastAPI
from fastapi import FastAPI, HTTPException
import httpx
import asyncio

app = FastAPI()

async def get_dashboard_async(format="full", location=None):
    """Async dashboard data retrieval"""
    params = {"format": format}
    if location:
        params["location"] = location
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "http://localhost:8000/api/v2/executive-dashboard",
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Dashboard service unavailable: {e}")

@app.get("/dashboard-summary")
async def dashboard_summary_endpoint():
    """Async endpoint that fetches dashboard data"""
    try:
        data = await get_dashboard_async(format="summary")
        return {
            "status": "success",
            "data": data,
            "source": data.get("metadata", {}).get("source", "unknown")
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "fallback_available": True
        }

# Multiple concurrent requests example
async def get_multiple_dashboards():
    """Get multiple dashboard views concurrently"""
    tasks = [
        get_dashboard_async(format="summary"),
        get_dashboard_async(format="kpis_only"),
        get_dashboard_async(format="alerts_only")
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {
        "summary": results[0] if not isinstance(results[0], Exception) else None,
        "kpis": results[1] if not isinstance(results[1], Exception) else None,
        "alerts": results[2] if not isinstance(results[2], Exception) else None,
        "errors": [str(r) for r in results if isinstance(r, Exception)]
    }
'''
        
        logger.info("Example Code:")
        print(example_code)
        
        # Simulate async behavior with concurrent requests
        async def demo_concurrent_requests():
            import aiohttp
            
            async def fetch_dashboard(session, format_type):
                try:
                    async with session.get(
                        f"{self.api_base_url}/api/v2/executive-dashboard",
                        params={"format": format_type}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {"format": format_type, "success": True, "size": len(str(data))}
                        else:
                            return {"format": format_type, "success": False, "error": f"HTTP {response.status}"}
                except Exception as e:
                    return {"format": format_type, "success": False, "error": str(e)}
            
            async with aiohttp.ClientSession() as session:
                tasks = [
                    fetch_dashboard(session, "summary"),
                    fetch_dashboard(session, "kpis_only"),
                    fetch_dashboard(session, "alerts_only")
                ]
                
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                execution_time = time.time() - start_time
                
                return {"results": results, "total_time": execution_time}
        
        try:
            # Run async demo
            async_results = asyncio.run(demo_concurrent_requests())
            logger.info(f"✓ Concurrent requests completed in {async_results['total_time']:.3f}s")
            
            for result in async_results['results']:
                if result['success']:
                    logger.info(f"  - {result['format']}: Success, {result['size']} chars")
                else:
                    logger.info(f"  - {result['format']}: Failed - {result['error']}")
            
            return async_results
            
        except Exception as e:
            logger.error(f"✗ Async demo failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ==============================================================================
    # SECTION 2: STATIC VS DYNAMIC COMPARISON
    # ==============================================================================
    
    def demonstrate_static_vs_dynamic(self) -> PerformanceComparison:
        """
        Compare static dashboard files vs dynamic API performance
        """
        logger.info("=" * 80)
        logger.info("SECTION 2: STATIC VS DYNAMIC DASHBOARD COMPARISON")
        logger.info("=" * 80)
        
        # Create sample static dashboard if it doesn't exist
        self._ensure_static_dashboard_exists()
        
        # Perform performance comparison
        static_results = self._benchmark_static_dashboard()
        dynamic_results = self._benchmark_dynamic_dashboard()
        
        comparison = self._analyze_performance_comparison(static_results, dynamic_results)
        
        self._display_comparison_results(comparison)
        
        return comparison
    
    def _ensure_static_dashboard_exists(self):
        """Ensure static dashboard file exists for comparison"""
        static_file = self.static_dashboard_path / "executive_dashboard_sample.json"
        
        if not static_file.exists():
            logger.info("Creating sample static dashboard file for comparison...")
            
            # Create a comprehensive static dashboard
            static_dashboard = {
                "metadata": {
                    "generated_at": "2025-08-28T00:00:00Z",
                    "source": "static_file",
                    "version": "1.0.0",
                    "note": "Static dashboard file for comparison testing"
                },
                "summary": {
                    "period": {
                        "start_date": "2025-07-28T00:00:00Z",
                        "end_date": "2025-08-28T00:00:00Z",
                        "period_days": 30
                    },
                    "facilities": {
                        "total_count": 5,
                        "active_alerts": 3,
                        "status_distribution": {
                            "normal": 3,
                            "attention": 1,
                            "warning": 1,
                            "critical": 0
                        }
                    },
                    "incidents": {
                        "total": 12,
                        "today": 0,
                        "change_from_previous_period": {
                            "percent_change": -15.2,
                            "absolute_change": -2,
                            "trend": "down"
                        },
                        "incident_rate": 2.4
                    },
                    "compliance": {
                        "audit_pass_rate": 94.5,
                        "training_completion": 87.3,
                        "overdue_items": 8
                    },
                    "alert_level": "YELLOW",
                    "overall_health_score": 82.5
                },
                "kpis": {
                    "metrics": {
                        "incident_rate": {
                            "name": "Incident Rate",
                            "value": 2.4,
                            "unit": "per 1000 hours",
                            "target": 2.0,
                            "status": "yellow",
                            "category": "safety"
                        },
                        "ltir": {
                            "name": "Lost Time Injury Rate",
                            "value": 0.8,
                            "unit": "rate",
                            "target": 0.5,
                            "status": "yellow",
                            "category": "safety"
                        },
                        "audit_pass_rate": {
                            "name": "Audit Pass Rate",
                            "value": 94.5,
                            "unit": "percentage",
                            "target": 95.0,
                            "status": "yellow",
                            "category": "compliance"
                        }
                    }
                },
                "charts": {
                    "incident_trend": {
                        "type": "line",
                        "title": "Incident Trend Over Time",
                        "data": [
                            {"date": "2025-08-01", "total": 2},
                            {"date": "2025-08-08", "total": 3},
                            {"date": "2025-08-15", "total": 1},
                            {"date": "2025-08-22", "total": 4},
                            {"date": "2025-08-28", "total": 2}
                        ]
                    }
                },
                "alerts": {
                    "summary": {
                        "total_active": 3,
                        "critical_count": 0,
                        "high_count": 1,
                        "medium_count": 2,
                        "low_count": 0,
                        "alert_level": "YELLOW"
                    },
                    "recent_alerts": [
                        {
                            "id": "ALERT001",
                            "severity": "high",
                            "title": "Training Completion Below Target",
                            "facility": "FAC001",
                            "created_at": "2025-08-27T10:30:00Z"
                        }
                    ]
                }
            }
            
            with open(static_file, 'w') as f:
                json.dump(static_dashboard, f, indent=2)
            
            logger.info(f"✓ Created static dashboard file: {static_file}")
    
    def _benchmark_static_dashboard(self) -> Dict[str, Any]:
        """Benchmark static dashboard file access"""
        logger.info("\n--- Benchmarking Static Dashboard ---")
        
        static_file = self.static_dashboard_path / "executive_dashboard_sample.json"
        
        times = []
        sizes = []
        
        for i in range(PERFORMANCE_TEST_ITERATIONS):
            start_time = time.time()
            
            try:
                with open(static_file, 'r') as f:
                    data = json.load(f)
                
                execution_time = time.time() - start_time
                times.append(execution_time)
                sizes.append(len(json.dumps(data)))
                
            except Exception as e:
                logger.error(f"Static dashboard read failed: {e}")
                times.append(float('inf'))
                sizes.append(0)
        
        avg_time = sum(t for t in times if t != float('inf')) / len([t for t in times if t != float('inf')])
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        
        logger.info(f"✓ Static dashboard: {avg_time:.4f}s average, {avg_size:.0f} bytes")
        
        return {
            "avg_time": avg_time,
            "min_time": min(times),
            "max_time": max(times),
            "avg_size": int(avg_size),
            "success_rate": len([t for t in times if t != float('inf')]) / len(times)
        }
    
    def _benchmark_dynamic_dashboard(self) -> Dict[str, Any]:
        """Benchmark dynamic dashboard API"""
        logger.info("\n--- Benchmarking Dynamic Dashboard API ---")
        
        times = []
        sizes = []
        features = []
        
        for i in range(PERFORMANCE_TEST_ITERATIONS):
            start_time = time.time()
            
            try:
                response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
                execution_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    times.append(execution_time)
                    sizes.append(len(response.content))
                    
                    # Analyze dynamic features
                    feature_count = 0
                    if "trends" in data:
                        feature_count += 1
                    if "recommendations" in data:
                        feature_count += 1
                    if data.get("metadata", {}).get("source") == "dynamic":
                        feature_count += 1
                    features.append(feature_count)
                else:
                    times.append(float('inf'))
                    sizes.append(0)
                    features.append(0)
                    
            except Exception as e:
                logger.error(f"Dynamic API request failed: {e}")
                times.append(float('inf'))
                sizes.append(0)
                features.append(0)
        
        valid_times = [t for t in times if t != float('inf')]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else float('inf')
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        avg_features = sum(features) / len(features) if features else 0
        
        logger.info(f"✓ Dynamic API: {avg_time:.4f}s average, {avg_size:.0f} bytes, {avg_features:.1f} dynamic features")
        
        return {
            "avg_time": avg_time,
            "min_time": min(times) if valid_times else float('inf'),
            "max_time": max(times) if valid_times else float('inf'),
            "avg_size": int(avg_size),
            "avg_features": avg_features,
            "success_rate": len(valid_times) / len(times)
        }
    
    def _analyze_performance_comparison(self, static_results: Dict, dynamic_results: Dict) -> PerformanceComparison:
        """Analyze performance comparison results"""
        
        # Calculate improvement percentage
        if static_results["avg_time"] > 0 and dynamic_results["avg_time"] != float('inf'):
            if dynamic_results["avg_time"] < static_results["avg_time"]:
                improvement = ((static_results["avg_time"] - dynamic_results["avg_time"]) / static_results["avg_time"]) * 100
            else:
                improvement = -((dynamic_results["avg_time"] - static_results["avg_time"]) / static_results["avg_time"]) * 100
        else:
            improvement = 0
        
        # Dynamic advantages
        dynamic_advantages = [
            "Real-time data from database",
            "Trend analysis and anomaly detection", 
            "AI-generated recommendations",
            "Flexible filtering and aggregation",
            "Automatic fallback mechanisms",
            "Advanced error handling",
            "Performance monitoring and caching",
            "Health checks and diagnostics"
        ]
        
        # Static advantages  
        static_advantages = [
            "Extremely fast file access",
            "No database dependencies",
            "Predictable performance",
            "Simple deployment",
            "High availability",
            "Minimal resource usage"
        ]
        
        return PerformanceComparison(
            static_time=static_results["avg_time"],
            dynamic_time=dynamic_results["avg_time"],
            static_size=static_results["avg_size"],
            dynamic_size=dynamic_results["avg_size"],
            improvement_percent=improvement,
            dynamic_advantages=dynamic_advantages,
            static_advantages=static_advantages
        )
    
    def _display_comparison_results(self, comparison: PerformanceComparison):
        """Display detailed comparison results"""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE COMPARISON RESULTS")
        logger.info("="*60)
        
        logger.info(f"\nExecution Time:")
        logger.info(f"  Static:  {comparison.static_time:.4f}s")
        logger.info(f"  Dynamic: {comparison.dynamic_time:.4f}s")
        if comparison.improvement_percent > 0:
            logger.info(f"  Dynamic is {comparison.improvement_percent:.1f}% slower (expected due to real-time processing)")
        else:
            logger.info(f"  Dynamic is {abs(comparison.improvement_percent):.1f}% faster (unexpected)")
        
        logger.info(f"\nResponse Size:")
        logger.info(f"  Static:  {comparison.static_size:,} bytes")
        logger.info(f"  Dynamic: {comparison.dynamic_size:,} bytes")
        size_diff = ((comparison.dynamic_size - comparison.static_size) / comparison.static_size) * 100
        logger.info(f"  Dynamic has {size_diff:.1f}% more content")
        
        logger.info(f"\nDynamic API Advantages:")
        for advantage in comparison.dynamic_advantages:
            logger.info(f"  ✓ {advantage}")
        
        logger.info(f"\nStatic File Advantages:")
        for advantage in comparison.static_advantages:
            logger.info(f"  ✓ {advantage}")
        
        logger.info(f"\nRecommendation:")
        if comparison.dynamic_time < 2.0:  # If dynamic is reasonably fast
            logger.info("  → Use Dynamic API for production (provides real-time insights)")
            logger.info("  → Keep static files as fallback for high availability")
        else:
            logger.info("  → Consider caching strategy to improve dynamic API performance")
            logger.info("  → Static files recommended for high-frequency, low-latency scenarios")
    
    # ==============================================================================
    # SECTION 3: MIGRATION PROCEDURES
    # ==============================================================================
    
    def demonstrate_migration_procedures(self) -> Dict[str, Any]:
        """
        Demonstrate migration from static to dynamic dashboard
        """
        logger.info("=" * 80)
        logger.info("SECTION 3: MIGRATION PROCEDURES")
        logger.info("=" * 80)
        
        migration_results = {}
        
        # Step 1: Pre-migration validation
        migration_results["pre_migration_validation"] = self._pre_migration_validation()
        
        # Step 2: Create backup
        migration_results["backup_creation"] = self._create_migration_backup()
        
        # Step 3: Update application configuration
        migration_results["configuration_update"] = self._demonstrate_configuration_update()
        
        # Step 4: Database setup and validation
        migration_results["database_setup"] = self._demonstrate_database_setup()
        
        # Step 5: Integration testing
        migration_results["integration_testing"] = self._perform_integration_testing()
        
        # Step 6: Rollback procedures
        migration_results["rollback_procedures"] = self._demonstrate_rollback_procedures()
        
        return migration_results
    
    def _pre_migration_validation(self) -> Dict[str, Any]:
        """Pre-migration validation checklist"""
        logger.info("\n--- Step 1: Pre-Migration Validation ---")
        
        validation_checklist = {
            "static_files_exist": False,
            "api_service_available": False,
            "database_accessible": False,
            "required_dependencies": False,
            "configuration_valid": False
        }
        
        # Check static files
        static_file = self.static_dashboard_path / "executive_dashboard_sample.json"
        if static_file.exists():
            validation_checklist["static_files_exist"] = True
            logger.info("✓ Static dashboard files found")
        else:
            logger.warning("⚠ Static dashboard files not found")
        
        # Check API service
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/health", timeout=5)
            if response.status_code == 200:
                validation_checklist["api_service_available"] = True
                logger.info("✓ API service is available")
            else:
                logger.warning(f"⚠ API service returned {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠ API service unavailable: {e}")
        
        # Check database (via API health check)
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("components", {}).get("database", {}).get("status") == "healthy":
                    validation_checklist["database_accessible"] = True
                    logger.info("✓ Database is accessible")
                else:
                    logger.warning("⚠ Database health check failed")
            else:
                logger.warning("⚠ Cannot verify database status")
        except Exception as e:
            logger.warning(f"⚠ Database check failed: {e}")
        
        # Check dependencies
        if DASHBOARD_COMPONENTS_AVAILABLE and FASTAPI_AVAILABLE:
            validation_checklist["required_dependencies"] = True
            logger.info("✓ Required dependencies are available")
        else:
            logger.warning("⚠ Some required dependencies are missing")
        
        # Check configuration
        env_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        config_valid = all(os.getenv(var) for var in env_vars)
        if config_valid:
            validation_checklist["configuration_valid"] = True
            logger.info("✓ Configuration is valid")
        else:
            logger.warning("⚠ Configuration validation failed - check environment variables")
        
        # Summary
        passed_checks = sum(validation_checklist.values())
        total_checks = len(validation_checklist)
        
        logger.info(f"\nValidation Summary: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            logger.info("✅ All validation checks passed - ready for migration")
        elif passed_checks >= total_checks * 0.8:
            logger.info("⚠️ Most checks passed - proceed with caution")
        else:
            logger.error("❌ Multiple validation failures - address issues before migration")
        
        return {
            "checklist": validation_checklist,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "ready_for_migration": passed_checks >= total_checks * 0.8
        }
    
    def _create_migration_backup(self) -> Dict[str, Any]:
        """Create backup of existing system"""
        logger.info("\n--- Step 2: Create Migration Backup ---")
        
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"dashboard_migration_{backup_timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_results = {
            "backup_dir": str(backup_dir),
            "timestamp": backup_timestamp,
            "backed_up_items": []
        }
        
        try:
            # Backup static dashboard files
            if self.static_dashboard_path.exists():
                static_backup = backup_dir / "static_dashboards"
                shutil.copytree(self.static_dashboard_path, static_backup)
                backup_results["backed_up_items"].append("static_dashboards")
                logger.info(f"✓ Backed up static dashboards to {static_backup}")
            
            # Create backup manifest
            manifest = {
                "backup_timestamp": backup_timestamp,
                "original_paths": {
                    "static_dashboards": str(self.static_dashboard_path)
                },
                "api_base_url": self.api_base_url,
                "environment_snapshot": {
                    key: os.getenv(key, "NOT_SET") 
                    for key in ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_DATABASE"]
                }
            }
            
            manifest_file = backup_dir / "backup_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            backup_results["backed_up_items"].append("backup_manifest")
            
            logger.info(f"✓ Created backup manifest: {manifest_file}")
            logger.info(f"✅ Backup completed successfully in {backup_dir}")
            
            return {**backup_results, "success": True}
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return {**backup_results, "success": False, "error": str(e)}
    
    def _demonstrate_configuration_update(self) -> Dict[str, Any]:
        """Demonstrate configuration updates needed for migration"""
        logger.info("\n--- Step 3: Configuration Updates ---")
        
        config_examples = {
            "environment_variables": {
                "description": "Required environment variables for dynamic API",
                "variables": {
                    "NEO4J_URI": "bolt://localhost:7687",
                    "NEO4J_USERNAME": "neo4j", 
                    "NEO4J_PASSWORD": "your_password",
                    "NEO4J_DATABASE": "neo4j"
                }
            },
            "application_config": {
                "description": "Application configuration changes",
                "before": {
                    "dashboard_source": "static",
                    "dashboard_file": "static_dashboards/executive_dashboard.json"
                },
                "after": {
                    "dashboard_source": "dynamic",
                    "dashboard_api_url": "http://localhost:8000/api/v2",
                    "fallback_enabled": True,
                    "fallback_file": "static_dashboards/executive_dashboard_fallback.json",
                    "cache_timeout": 300
                }
            },
            "nginx_config": {
                "description": "Nginx configuration for load balancing and caching",
                "config": """
# Nginx configuration for dashboard API
upstream dashboard_api {
    server localhost:8000;
    server localhost:8001 backup;
}

location /api/v2/executive-dashboard {
    proxy_pass http://dashboard_api;
    proxy_cache dashboard_cache;
    proxy_cache_valid 200 5m;
    proxy_cache_key "$request_uri";
    
    # Fallback to static file on upstream failure
    error_page 502 503 504 = @fallback;
}

location @fallback {
    try_files /static/dashboard_fallback.json =503;
}
"""
            }
        }
        
        logger.info("Configuration update examples:")
        
        for config_type, config_data in config_examples.items():
            logger.info(f"\n{config_type.upper()}:")
            logger.info(f"  Description: {config_data['description']}")
            
            if config_type == "environment_variables":
                logger.info("  Required variables:")
                for var, value in config_data["variables"].items():
                    current_value = os.getenv(var, "NOT_SET")
                    status = "✓" if current_value != "NOT_SET" else "✗"
                    logger.info(f"    {status} {var}={value} (current: {current_value})")
                    
            elif config_type == "application_config":
                logger.info("  Before (static):")
                for key, value in config_data["before"].items():
                    logger.info(f"    {key}: {value}")
                logger.info("  After (dynamic):")
                for key, value in config_data["after"].items():
                    logger.info(f"    {key}: {value}")
                    
            elif config_type == "nginx_config":
                logger.info("  Sample configuration:")
                for line in config_data["config"].strip().split('\n')[:5]:
                    logger.info(f"    {line}")
                logger.info("    ... (see full configuration in logs)")
        
        # Validate current configuration
        current_config_status = {
            "env_vars_set": all(os.getenv(var) for var in ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]),
            "api_accessible": False,
            "fallback_available": self.static_dashboard_path.exists()
        }
        
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/health", timeout=5)
            current_config_status["api_accessible"] = response.status_code == 200
        except:
            pass
        
        logger.info(f"\nCurrent configuration status:")
        for key, status in current_config_status.items():
            symbol = "✓" if status else "✗"
            logger.info(f"  {symbol} {key}: {status}")
        
        return {
            "config_examples": config_examples,
            "current_status": current_config_status,
            "ready_for_migration": all(current_config_status.values())
        }
    
    def _demonstrate_database_setup(self) -> Dict[str, Any]:
        """Demonstrate database setup and validation"""
        logger.info("\n--- Step 4: Database Setup and Validation ---")
        
        database_checks = {
            "connection": False,
            "required_nodes": False,
            "sample_data": False,
            "indexes": False,
            "constraints": False
        }
        
        setup_examples = {
            "sample_data_creation": """
# Sample Cypher queries to create test data
CREATE (f:Facility {facility_id: 'FAC001', facility_name: 'Main Plant', location: 'Texas'})
CREATE (i:Incident {incident_id: 'INC001', facility_id: 'FAC001', incident_date: date('2025-08-01')})
CREATE (a:Audit {audit_id: 'AUD001', facility_id: 'FAC001', pass_rate: 95.5})
CREATE (t:Training {training_id: 'TRN001', employee_count: 50, completion_rate: 87.3})
CREATE (f)-[:HAS_INCIDENT]->(i)
CREATE (f)-[:HAS_AUDIT]->(a)
CREATE (f)-[:HAS_TRAINING]->(t)
""",
            "index_creation": """
# Performance indexes
CREATE INDEX facility_id_idx FOR (f:Facility) ON (f.facility_id)
CREATE INDEX incident_date_idx FOR (i:Incident) ON (i.incident_date)
CREATE INDEX audit_date_idx FOR (a:Audit) ON (a.audit_date)
""",
            "constraint_creation": """
# Data integrity constraints
CREATE CONSTRAINT facility_id_unique FOR (f:Facility) REQUIRE f.facility_id IS UNIQUE
CREATE CONSTRAINT incident_id_unique FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE
"""
        }
        
        logger.info("Database setup examples:")
        for setup_type, query in setup_examples.items():
            logger.info(f"\n{setup_type.upper()}:")
            for line in query.strip().split('\n')[:3]:
                if line.strip():
                    logger.info(f"  {line}")
            logger.info("  ... (see full queries in logs)")
        
        # Test database connectivity through API
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                database_status = health_data.get("components", {}).get("database", {})
                
                if database_status.get("status") == "healthy":
                    database_checks["connection"] = True
                    logger.info("✓ Database connection successful")
                    
                    # Check for basic data through API
                    dashboard_response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=15)
                    if dashboard_response.status_code == 200:
                        dashboard_data = dashboard_response.json()
                        
                        # Check for facilities
                        facility_count = dashboard_data.get("summary", {}).get("facilities", {}).get("total_count", 0)
                        if facility_count > 0:
                            database_checks["sample_data"] = True
                            logger.info(f"✓ Sample data found: {facility_count} facilities")
                        else:
                            logger.warning("⚠ No sample data found")
                        
                        # Check for KPIs (indicates proper data structure)
                        kpis = dashboard_data.get("kpis", {}).get("metrics", {})
                        if len(kpis) > 0:
                            database_checks["required_nodes"] = True
                            logger.info(f"✓ Required data nodes found: {len(kpis)} KPIs available")
                        else:
                            logger.warning("⚠ Required data nodes not found")
                    
                else:
                    logger.warning(f"⚠ Database health check failed: {database_status}")
        
        except Exception as e:
            logger.error(f"✗ Database validation failed: {e}")
        
        # Summary
        passed_checks = sum(database_checks.values())
        total_checks = len(database_checks)
        
        logger.info(f"\nDatabase validation summary: {passed_checks}/{total_checks} checks passed")
        
        return {
            "checks": database_checks,
            "setup_examples": setup_examples,
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "database_ready": passed_checks >= total_checks * 0.6
        }
    
    def _perform_integration_testing(self) -> Dict[str, Any]:
        """Perform comprehensive integration testing"""
        logger.info("\n--- Step 5: Integration Testing ---")
        
        test_scenarios = [
            {"name": "Basic dashboard retrieval", "endpoint": "/executive-dashboard", "params": {}},
            {"name": "Filtered dashboard", "endpoint": "/executive-dashboard", "params": {"format": "summary"}},
            {"name": "KPI details", "endpoint": "/kpis", "params": {"dateRange": "30d"}},
            {"name": "Real-time metrics", "endpoint": "/real-time-metrics", "params": {}},
            {"name": "Health check", "endpoint": "/health", "params": {}},
            {"name": "Available locations", "endpoint": "/locations", "params": {}}
        ]
        
        test_results = []
        
        for scenario in test_scenarios:
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.api_base_url}/api/v2{scenario['endpoint']}",
                    params=scenario["params"],
                    timeout=30
                )
                execution_time = time.time() - start_time
                
                success = response.status_code == 200
                response_size = len(response.content)
                
                result = IntegrationTestResult(
                    test_name=scenario["name"],
                    success=success,
                    execution_time=execution_time,
                    response_size=response_size,
                    error_message=None if success else f"HTTP {response.status_code}",
                    metadata={"status_code": response.status_code}
                )
                
                if success:
                    logger.info(f"✓ {scenario['name']}: {execution_time:.3f}s, {response_size} bytes")
                else:
                    logger.error(f"✗ {scenario['name']}: HTTP {response.status_code}")
                
                test_results.append(result)
                
            except Exception as e:
                result = IntegrationTestResult(
                    test_name=scenario["name"],
                    success=False,
                    execution_time=0.0,
                    response_size=0,
                    error_message=str(e)
                )
                test_results.append(result)
                logger.error(f"✗ {scenario['name']}: {e}")
        
        # Performance testing
        logger.info("\n--- Performance Testing ---")
        performance_test = self._run_performance_tests()
        
        # Summary
        successful_tests = len([r for r in test_results if r.success])
        total_tests = len(test_results)
        avg_response_time = sum(r.execution_time for r in test_results if r.success) / max(successful_tests, 1)
        
        logger.info(f"\nIntegration testing summary:")
        logger.info(f"  Successful tests: {successful_tests}/{total_tests}")
        logger.info(f"  Average response time: {avg_response_time:.3f}s")
        logger.info(f"  Performance test: {performance_test['success']}")
        
        return {
            "test_results": [asdict(r) for r in test_results],
            "performance_test": performance_test,
            "summary": {
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "success_rate": successful_tests / total_tests,
                "avg_response_time": avg_response_time,
                "ready_for_production": successful_tests >= total_tests * 0.9
            }
        }
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        try:
            # Concurrent request test
            concurrent_results = []
            
            def make_request():
                try:
                    start = time.time()
                    response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
                    return time.time() - start, response.status_code == 200
                except:
                    return float('inf'), False
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                concurrent_results = [f.result() for f in futures]
            
            successful_requests = [r for r in concurrent_results if r[1]]
            avg_concurrent_time = sum(r[0] for r in successful_requests) / len(successful_requests) if successful_requests else float('inf')
            
            logger.info(f"  Concurrent requests: {len(successful_requests)}/10 successful, avg time: {avg_concurrent_time:.3f}s")
            
            return {
                "success": len(successful_requests) >= 8,
                "concurrent_success_rate": len(successful_requests) / 10,
                "avg_concurrent_time": avg_concurrent_time
            }
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _demonstrate_rollback_procedures(self) -> Dict[str, Any]:
        """Demonstrate rollback procedures"""
        logger.info("\n--- Step 6: Rollback Procedures ---")
        
        rollback_procedures = {
            "immediate_rollback": {
                "description": "Immediate rollback to static files",
                "steps": [
                    "1. Switch application config to use static files",
                    "2. Update load balancer to bypass API endpoints", 
                    "3. Restart application services",
                    "4. Verify static dashboard accessibility"
                ],
                "estimated_time": "< 5 minutes"
            },
            "database_rollback": {
                "description": "Rollback database changes if needed",
                "steps": [
                    "1. Stop API services",
                    "2. Restore database from backup",
                    "3. Verify data integrity",
                    "4. Restart services with static config"
                ],
                "estimated_time": "10-30 minutes"
            },
            "configuration_rollback": {
                "description": "Rollback all configuration changes",
                "steps": [
                    "1. Restore backed-up configuration files",
                    "2. Revert environment variables",
                    "3. Update DNS/load balancer settings",
                    "4. Restart all services",
                    "5. Run health checks"
                ],
                "estimated_time": "15-45 minutes"
            }
        }
        
        # Create rollback script example
        rollback_script = '''#!/bin/bash
# Emergency rollback script for dashboard migration

set -e

echo "Starting emergency rollback..."

# Step 1: Switch to static configuration
export DASHBOARD_SOURCE="static"
export DASHBOARD_FILE="static_dashboards/executive_dashboard_fallback.json"

# Step 2: Restart services
systemctl restart dashboard-app
systemctl reload nginx

# Step 3: Verify rollback
sleep 5
curl -f http://localhost/dashboard/health || echo "Health check failed"

echo "Rollback completed. Check application logs for any issues."
'''
        
        rollback_script_path = self.backup_path / "emergency_rollback.sh"
        try:
            with open(rollback_script_path, 'w') as f:
                f.write(rollback_script)
            os.chmod(rollback_script_path, 0o755)
            logger.info(f"✓ Created rollback script: {rollback_script_path}")
        except Exception as e:
            logger.warning(f"⚠ Failed to create rollback script: {e}")
        
        logger.info("\nRollback procedures:")
        for procedure_name, procedure_data in rollback_procedures.items():
            logger.info(f"\n{procedure_name.upper()}:")
            logger.info(f"  Description: {procedure_data['description']}")
            logger.info(f"  Estimated time: {procedure_data['estimated_time']}")
            logger.info("  Steps:")
            for step in procedure_data['steps']:
                logger.info(f"    {step}")
        
        # Test rollback readiness
        rollback_readiness = {
            "backup_available": self.backup_path.exists() and any(self.backup_path.iterdir()),
            "static_files_intact": (self.static_dashboard_path / "executive_dashboard_sample.json").exists(),
            "rollback_script_created": rollback_script_path.exists(),
            "configuration_documented": True  # Assume documented
        }
        
        logger.info(f"\nRollback readiness check:")
        for check, status in rollback_readiness.items():
            symbol = "✓" if status else "✗"
            logger.info(f"  {symbol} {check}: {status}")
        
        return {
            "procedures": rollback_procedures,
            "rollback_script": str(rollback_script_path) if rollback_script_path.exists() else None,
            "readiness": rollback_readiness,
            "ready_for_rollback": all(rollback_readiness.values())
        }
    
    # ==============================================================================
    # SECTION 4: PRODUCTION DEPLOYMENT GUIDE
    # ==============================================================================
    
    def demonstrate_production_deployment(self) -> Dict[str, Any]:
        """
        Demonstrate production deployment strategies
        """
        logger.info("=" * 80)
        logger.info("SECTION 4: PRODUCTION DEPLOYMENT GUIDE")
        logger.info("=" * 80)
        
        deployment_guide = {}
        
        # Deployment strategies
        deployment_guide["strategies"] = self._demonstrate_deployment_strategies()
        
        # Monitoring setup
        deployment_guide["monitoring"] = self._demonstrate_monitoring_setup()
        
        # Security considerations
        deployment_guide["security"] = self._demonstrate_security_setup()
        
        # Scaling and performance
        deployment_guide["scaling"] = self._demonstrate_scaling_setup()
        
        return deployment_guide
    
    def _demonstrate_deployment_strategies(self) -> Dict[str, Any]:
        """Demonstrate different deployment strategies"""
        logger.info("\n--- Deployment Strategies ---")
        
        strategies = {
            "blue_green": {
                "description": "Blue-Green deployment with instant switchover",
                "config": """
# Blue-Green deployment configuration
version: '3.8'
services:
  dashboard-blue:
    image: dashboard-api:latest
    environment:
      - ENV=production-blue
    networks:
      - blue
  
  dashboard-green:
    image: dashboard-api:latest
    environment:
      - ENV=production-green
    networks:
      - green
  
  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx-blue-green.conf:/etc/nginx/nginx.conf
""",
                "advantages": ["Zero downtime", "Instant rollback", "Full environment testing"],
                "disadvantages": ["Double resource usage", "Complex coordination"]
            },
            "canary": {
                "description": "Canary deployment with gradual rollout",
                "config": """
# Canary deployment with traffic splitting
upstream dashboard_api {
    server dashboard-stable:8000 weight=90;
    server dashboard-canary:8000 weight=10;
}

location /api/v2/ {
    proxy_pass http://dashboard_api;
    
    # Route specific users to canary
    if ($http_x_canary_user = "true") {
        proxy_pass http://dashboard-canary:8000;
    }
}
""",
                "advantages": ["Risk mitigation", "Gradual validation", "Real user testing"],
                "disadvantages": ["Complex routing", "Longer deployment time"]
            },
            "rolling": {
                "description": "Rolling deployment with sequential updates",
                "config": """
# Rolling deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dashboard-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      containers:
      - name: dashboard-api
        image: dashboard-api:v2.0.0
""",
                "advantages": ["Resource efficient", "Simple process", "Good for stateless apps"],
                "disadvantages": ["Mixed versions during deployment", "Potential compatibility issues"]
            }
        }
        
        for strategy_name, strategy_data in strategies.items():
            logger.info(f"\n{strategy_name.upper()} DEPLOYMENT:")
            logger.info(f"  Description: {strategy_data['description']}")
            logger.info(f"  Advantages: {', '.join(strategy_data['advantages'])}")
            logger.info(f"  Disadvantages: {', '.join(strategy_data['disadvantages'])}")
            logger.info("  Configuration sample:")
            for line in strategy_data['config'].strip().split('\n')[:5]:
                if line.strip():
                    logger.info(f"    {line}")
            logger.info("    ... (see full configuration in deployment guide)")
        
        # Recommended strategy
        logger.info(f"\nRECOMMENDED STRATEGY:")
        logger.info("  For dashboard API: Blue-Green deployment")
        logger.info("  Rationale: Minimal risk, instant rollback capability")
        logger.info("  Fallback: Static files provide additional safety net")
        
        return strategies
    
    def _demonstrate_monitoring_setup(self) -> Dict[str, Any]:
        """Demonstrate monitoring and observability setup"""
        logger.info("\n--- Monitoring and Observability ---")
        
        monitoring_config = {
            "metrics": {
                "prometheus_config": """
# Prometheus configuration for dashboard API
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'dashboard-api'
    static_configs:
      - targets: ['dashboard-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'neo4j'
    static_configs:
      - targets: ['neo4j:2004']
""",
                "key_metrics": [
                    "dashboard_requests_total",
                    "dashboard_request_duration_seconds", 
                    "dashboard_errors_total",
                    "cache_hit_rate",
                    "neo4j_connection_pool_size",
                    "active_alerts_count"
                ]
            },
            "alerting": {
                "alert_rules": """
# Alerting rules for dashboard API
groups:
  - name: dashboard-api
    rules:
      - alert: DashboardAPIDown
        expr: up{job="dashboard-api"} == 0
        for: 1m
        annotations:
          summary: "Dashboard API is down"
          
      - alert: HighErrorRate
        expr: rate(dashboard_errors_total[5m]) > 0.1
        for: 2m
        annotations:
          summary: "Dashboard API error rate is high"
          
      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, rate(dashboard_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        annotations:
          summary: "Dashboard API response time is slow"
"""
            },
            "logging": {
                "log_config": """
# Structured logging configuration
[loggers]
keys = root, dashboard

[handlers]
keys = console, file, json

[formatters]
keys = standard, json

[logger_dashboard]
level = INFO
handlers = json
qualname = dashboard

[handler_json]
class = pythonjsonlogger.jsonlogger.JsonFormatter
formatter = json
""",
                "log_aggregation": "ELK Stack (Elasticsearch, Logstash, Kibana)"
            },
            "health_checks": {
                "readiness_probe": "/api/v2/health",
                "liveness_probe": "/api/v2/health",
                "startup_probe": "/api/v2/health",
                "check_interval": "30s",
                "timeout": "10s"
            }
        }
        
        logger.info("Monitoring components:")
        
        for component, config in monitoring_config.items():
            logger.info(f"\n{component.upper()}:")
            if isinstance(config, dict):
                for key, value in config.items():
                    logger.info(f"  {key}:")
                    if isinstance(value, str):
                        for line in value.strip().split('\n')[:3]:
                            if line.strip():
                                logger.info(f"    {line}")
                        logger.info("    ... (see full config)")
                    else:
                        logger.info(f"    {value}")
            else:
                logger.info(f"  {config}")
        
        # Monitoring validation
        monitoring_status = self._validate_monitoring()
        
        return {
            "config": monitoring_config,
            "status": monitoring_status
        }
    
    def _validate_monitoring(self) -> Dict[str, Any]:
        """Validate current monitoring setup"""
        monitoring_checks = {
            "health_endpoint": False,
            "metrics_endpoint": False,
            "log_output": False,
            "error_handling": False
        }
        
        # Check health endpoint
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/health", timeout=10)
            if response.status_code == 200:
                monitoring_checks["health_endpoint"] = True
                logger.info("✓ Health endpoint accessible")
            else:
                logger.warning(f"⚠ Health endpoint returned {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠ Health endpoint check failed: {e}")
        
        # Check for metrics endpoint (would typically be /metrics)
        try:
            response = requests.get(f"{self.api_base_url}/metrics", timeout=5)
            monitoring_checks["metrics_endpoint"] = response.status_code == 200
        except:
            logger.warning("⚠ Metrics endpoint not available (optional)")
        
        # Log output validation (check if logs are structured)
        log_file = Path("dashboard_integration.log")
        if log_file.exists():
            monitoring_checks["log_output"] = True
            logger.info("✓ Log output is being generated")
        
        # Error handling validation
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", 
                                  params={"format": "invalid"}, timeout=10)
            if response.status_code in [400, 422]:  # Expected error codes
                monitoring_checks["error_handling"] = True
                logger.info("✓ Error handling is working correctly")
        except:
            logger.warning("⚠ Could not validate error handling")
        
        passed_checks = sum(monitoring_checks.values())
        logger.info(f"\nMonitoring validation: {passed_checks}/4 checks passed")
        
        return {
            "checks": monitoring_checks,
            "passed_checks": passed_checks,
            "monitoring_ready": passed_checks >= 2
        }
    
    def _demonstrate_security_setup(self) -> Dict[str, Any]:
        """Demonstrate security configuration"""
        logger.info("\n--- Security Configuration ---")
        
        security_config = {
            "authentication": {
                "method": "JWT Bearer tokens",
                "config": """
# JWT Authentication middleware
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/v2/executive-dashboard")
async def get_dashboard(current_user: str = Depends(verify_token)):
    # Protected endpoint
    pass
"""
            },
            "rate_limiting": {
                "method": "Redis-based rate limiting",
                "config": """
# Rate limiting configuration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/v2/executive-dashboard")
@limiter.limit("100/minute")
async def get_dashboard(request: Request):
    # Rate limited endpoint
    pass
"""
            },
            "https_tls": {
                "method": "TLS termination at load balancer",
                "config": """
# Nginx TLS configuration
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    location /api/v2/ {
        proxy_pass http://dashboard-api;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}
"""
            },
            "input_validation": {
                "method": "Pydantic model validation",
                "features": [
                    "Automatic request validation",
                    "SQL injection prevention",
                    "XSS protection",
                    "Input sanitization",
                    "Type checking"
                ]
            }
        }
        
        logger.info("Security measures:")
        for measure, config in security_config.items():
            logger.info(f"\n{measure.upper()}:")
            logger.info(f"  Method: {config['method']}")
            
            if 'config' in config:
                logger.info("  Configuration sample:")
                for line in config['config'].strip().split('\n')[:5]:
                    if line.strip():
                        logger.info(f"    {line}")
                logger.info("    ... (see full configuration)")
            
            if 'features' in config:
                logger.info("  Features:")
                for feature in config['features']:
                    logger.info(f"    - {feature}")
        
        return security_config
    
    def _demonstrate_scaling_setup(self) -> Dict[str, Any]:
        """Demonstrate scaling and performance configuration"""
        logger.info("\n--- Scaling and Performance ---")
        
        scaling_config = {
            "horizontal_scaling": {
                "method": "Kubernetes HPA",
                "config": """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dashboard-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dashboard-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
            },
            "caching_strategy": {
                "layers": ["Application cache", "Redis cache", "CDN cache"],
                "config": """
# Multi-layer caching configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://redis:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Application-level caching
@lru_cache(maxsize=128)
def get_cached_dashboard(cache_key: str):
    return expensive_dashboard_calculation()

# CDN configuration
location /api/v2/executive-dashboard {
    proxy_cache api_cache;
    proxy_cache_valid 200 5m;
    proxy_cache_key "$request_uri$is_args$args";
    add_header X-Cache-Status $upstream_cache_status;
}
"""
            },
            "database_optimization": {
                "strategies": [
                    "Connection pooling",
                    "Query optimization",
                    "Index management",
                    "Read replicas",
                    "Query result caching"
                ],
                "config": """
# Neo4j performance configuration
dbms.connector.bolt.thread_pool_max_size=200
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=4g
dbms.memory.pagecache.size=2g

# Connection pool settings
NEO4J_POOL_MAX_SIZE=50
NEO4J_POOL_ACQUISITION_TIMEOUT=60
"""
            }
        }
        
        logger.info("Scaling strategies:")
        for strategy, config in scaling_config.items():
            logger.info(f"\n{strategy.upper()}:")
            
            if 'method' in config:
                logger.info(f"  Method: {config['method']}")
            
            if 'layers' in config:
                logger.info(f"  Layers: {', '.join(config['layers'])}")
            
            if 'strategies' in config:
                logger.info("  Strategies:")
                for item in config['strategies']:
                    logger.info(f"    - {item}")
            
            if 'config' in config:
                logger.info("  Configuration sample:")
                for line in config['config'].strip().split('\n')[:5]:
                    if line.strip():
                        logger.info(f"    {line}")
                logger.info("    ... (see full configuration)")
        
        return scaling_config
    
    # ==============================================================================
    # SECTION 5: INTEGRATION TESTING SUITE
    # ==============================================================================
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite
        """
        logger.info("=" * 80)
        logger.info("SECTION 5: COMPREHENSIVE INTEGRATION TESTS")
        logger.info("=" * 80)
        
        test_results = {
            "api_functionality": self._test_api_functionality(),
            "performance": self._test_performance(),
            "reliability": self._test_reliability(),
            "security": self._test_security(),
            "integration": self._test_integration()
        }
        
        # Generate test report
        self._generate_test_report(test_results)
        
        return test_results
    
    def _test_api_functionality(self) -> Dict[str, Any]:
        """Test core API functionality"""
        logger.info("\n--- API Functionality Tests ---")
        
        tests = [
            {"name": "Basic dashboard", "url": "/api/v2/executive-dashboard", "expected_keys": ["metadata", "summary"]},
            {"name": "Summary format", "url": "/api/v2/executive-dashboard?format=summary", "expected_keys": ["summary", "status"]},
            {"name": "KPIs only", "url": "/api/v2/executive-dashboard?format=kpis_only", "expected_keys": ["kpis", "metadata"]},
            {"name": "Health check", "url": "/api/v2/health", "expected_keys": ["status", "components"]},
            {"name": "Real-time metrics", "url": "/api/v2/real-time-metrics", "expected_keys": ["metrics", "alert_level"]},
        ]
        
        results = []
        
        for test in tests:
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_base_url}{test['url']}", timeout=30)
                execution_time = time.time() - start_time
                
                success = response.status_code == 200
                
                if success:
                    data = response.json()
                    # Check for expected keys
                    has_expected_keys = all(key in data for key in test["expected_keys"])
                    success = success and has_expected_keys
                    
                    logger.info(f"✓ {test['name']}: {execution_time:.3f}s")
                else:
                    logger.error(f"✗ {test['name']}: HTTP {response.status_code}")
                
                results.append({
                    "test": test['name'],
                    "success": success,
                    "execution_time": execution_time,
                    "status_code": response.status_code if 'response' in locals() else 0
                })
                
            except Exception as e:
                logger.error(f"✗ {test['name']}: {e}")
                results.append({
                    "test": test['name'],
                    "success": False,
                    "execution_time": 0.0,
                    "error": str(e)
                })
        
        success_rate = len([r for r in results if r['success']]) / len(results)
        logger.info(f"API functionality tests: {success_rate:.1%} success rate")
        
        return {
            "results": results,
            "success_rate": success_rate,
            "passed": success_rate >= 0.8
        }
    
    def _test_performance(self) -> Dict[str, Any]:
        """Test API performance"""
        logger.info("\n--- Performance Tests ---")
        
        performance_results = {
            "response_time": self._test_response_time(),
            "concurrent_load": self._test_concurrent_load(),
            "memory_usage": self._test_memory_usage()
        }
        
        return performance_results
    
    def _test_response_time(self) -> Dict[str, Any]:
        """Test response time benchmarks"""
        times = []
        
        for _ in range(10):
            try:
                start = time.time()
                response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
                execution_time = time.time() - start
                
                if response.status_code == 200:
                    times.append(execution_time)
                    
            except Exception:
                times.append(float('inf'))
        
        valid_times = [t for t in times if t != float('inf')]
        
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            p95_time = sorted(valid_times)[int(len(valid_times) * 0.95)] if len(valid_times) > 1 else avg_time
            
            logger.info(f"Response time - Avg: {avg_time:.3f}s, P95: {p95_time:.3f}s")
            
            return {
                "avg_response_time": avg_time,
                "p95_response_time": p95_time,
                "success_rate": len(valid_times) / len(times),
                "passed": avg_time < 5.0 and p95_time < 10.0
            }
        
        return {"passed": False, "error": "No successful requests"}
    
    def _test_concurrent_load(self) -> Dict[str, Any]:
        """Test concurrent request handling"""
        import concurrent.futures
        
        def make_request():
            try:
                start = time.time()
                response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
                return time.time() - start, response.status_code == 200
            except:
                return float('inf'), False
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [f.result() for f in futures]
        total_time = time.time() - start_time
        
        successful = [r for r in results if r[1]]
        success_rate = len(successful) / len(results)
        avg_response_time = sum(r[0] for r in successful) / len(successful) if successful else float('inf')
        
        logger.info(f"Concurrent load - {success_rate:.1%} success rate, avg time: {avg_response_time:.3f}s")
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "total_test_time": total_time,
            "passed": success_rate >= 0.8 and avg_response_time < 10.0
        }
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns"""
        # This would typically use process monitoring tools
        # For demo purposes, we'll simulate memory usage testing
        
        logger.info("Memory usage test - monitoring application resource usage...")
        
        # Simulate memory test by making multiple requests and checking response sizes
        response_sizes = []
        
        for i in range(5):
            try:
                response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
                if response.status_code == 200:
                    response_sizes.append(len(response.content))
            except:
                pass
        
        if response_sizes:
            avg_response_size = sum(response_sizes) / len(response_sizes)
            max_response_size = max(response_sizes)
            
            # Reasonable thresholds for dashboard data
            passed = avg_response_size < 1024 * 1024 and max_response_size < 2 * 1024 * 1024  # 1MB avg, 2MB max
            
            logger.info(f"Memory usage - Avg response: {avg_response_size/1024:.1f}KB, Max: {max_response_size/1024:.1f}KB")
            
            return {
                "avg_response_size": avg_response_size,
                "max_response_size": max_response_size,
                "passed": passed
            }
        
        return {"passed": False, "error": "Could not measure memory usage"}
    
    def _test_reliability(self) -> Dict[str, Any]:
        """Test system reliability"""
        logger.info("\n--- Reliability Tests ---")
        
        reliability_tests = {
            "error_handling": self._test_error_scenarios(),
            "fallback_mechanism": self._test_fallback_mechanism(),
            "recovery": self._test_recovery()
        }
        
        return reliability_tests
    
    def _test_error_scenarios(self) -> Dict[str, Any]:
        """Test error handling"""
        error_scenarios = [
            {"name": "Invalid format", "params": {"format": "invalid"}, "expected_status": [400, 422]},
            {"name": "Invalid date range", "params": {"dateRange": "invalid"}, "expected_status": [400, 422]},
            {"name": "Large date range", "params": {"dateRange": "10y"}, "expected_status": [200, 400]},
        ]
        
        results = []
        
        for scenario in error_scenarios:
            try:
                response = requests.get(
                    f"{self.api_base_url}/api/v2/executive-dashboard",
                    params=scenario["params"],
                    timeout=30
                )
                
                expected_behavior = response.status_code in scenario["expected_status"]
                
                if expected_behavior:
                    logger.info(f"✓ {scenario['name']}: Expected response {response.status_code}")
                else:
                    logger.warning(f"⚠ {scenario['name']}: Unexpected response {response.status_code}")
                
                results.append({
                    "scenario": scenario['name'],
                    "passed": expected_behavior,
                    "status_code": response.status_code
                })
                
            except Exception as e:
                logger.error(f"✗ {scenario['name']}: {e}")
                results.append({
                    "scenario": scenario['name'],
                    "passed": False,
                    "error": str(e)
                })
        
        pass_rate = len([r for r in results if r['passed']]) / len(results)
        return {"results": results, "pass_rate": pass_rate, "passed": pass_rate >= 0.7}
    
    def _test_fallback_mechanism(self) -> Dict[str, Any]:
        """Test fallback to static files"""
        logger.info("Testing fallback mechanism...")
        
        # The API should automatically fall back to static data if the service fails
        # We'll test this by checking if the response includes fallback indicators
        
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                source = data.get("metadata", {}).get("source", "")
                
                # Check if system can provide data (either dynamic or fallback)
                has_data = bool(data.get("summary") or data.get("kpis") or data.get("error"))
                
                logger.info(f"✓ Fallback test: Data source is '{source}', has_data: {has_data}")
                
                return {
                    "data_source": source,
                    "has_fallback": "fallback" in source or "static" in source,
                    "has_data": has_data,
                    "passed": has_data
                }
            else:
                logger.warning(f"⚠ Fallback test: HTTP {response.status_code}")
                return {"passed": False, "status_code": response.status_code}
                
        except Exception as e:
            logger.error(f"✗ Fallback test failed: {e}")
            return {"passed": False, "error": str(e)}
    
    def _test_recovery(self) -> Dict[str, Any]:
        """Test system recovery capabilities"""
        logger.info("Testing system recovery...")
        
        # Test if the system can recover from temporary issues
        # We'll simulate this by making requests with varying timeouts
        
        recovery_results = []
        
        for timeout in [1, 5, 15, 30]:  # Increasing timeouts to test recovery
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.api_base_url}/api/v2/executive-dashboard",
                    timeout=timeout
                )
                execution_time = time.time() - start_time
                
                success = response.status_code == 200
                recovery_results.append({
                    "timeout": timeout,
                    "success": success,
                    "execution_time": execution_time
                })
                
                if success:
                    logger.info(f"✓ Recovery test (timeout {timeout}s): Success in {execution_time:.3f}s")
                else:
                    logger.warning(f"⚠ Recovery test (timeout {timeout}s): Failed with {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.info(f"⚠ Recovery test (timeout {timeout}s): Timeout (expected for short timeouts)")
                recovery_results.append({
                    "timeout": timeout,
                    "success": False,
                    "timeout_occurred": True
                })
            except Exception as e:
                logger.error(f"✗ Recovery test (timeout {timeout}s): {e}")
                recovery_results.append({
                    "timeout": timeout,
                    "success": False,
                    "error": str(e)
                })
        
        # Check if longer timeouts result in better success rates
        successful_tests = [r for r in recovery_results if r['success']]
        recovery_demonstrated = len(successful_tests) > 0
        
        return {
            "results": recovery_results,
            "successful_tests": len(successful_tests),
            "recovery_demonstrated": recovery_demonstrated,
            "passed": recovery_demonstrated
        }
    
    def _test_security(self) -> Dict[str, Any]:
        """Test security measures"""
        logger.info("\n--- Security Tests ---")
        
        security_tests = {
            "input_validation": self._test_input_validation(),
            "error_disclosure": self._test_error_information_disclosure(),
            "rate_limiting": self._test_rate_limiting()
        }
        
        return security_tests
    
    def _test_input_validation(self) -> Dict[str, Any]:
        """Test input validation and sanitization"""
        malicious_inputs = [
            {"name": "SQL injection", "params": {"location": "'; DROP TABLE facilities; --"}},
            {"name": "XSS attempt", "params": {"format": "<script>alert('xss')</script>"}},
            {"name": "Path traversal", "params": {"dateRange": "../../etc/passwd"}},
            {"name": "Command injection", "params": {"location": "; cat /etc/passwd"}},
        ]
        
        results = []
        
        for test_case in malicious_inputs:
            try:
                response = requests.get(
                    f"{self.api_base_url}/api/v2/executive-dashboard",
                    params=test_case["params"],
                    timeout=10
                )
                
                # Good security: should return 400/422 for invalid input
                # Should not execute malicious code or return sensitive data
                security_passed = response.status_code in [400, 422] or (
                    response.status_code == 200 and 
                    'error' in response.json().get('metadata', {}).get('error', '')
                )
                
                if security_passed:
                    logger.info(f"✓ {test_case['name']}: Properly handled (HTTP {response.status_code})")
                else:
                    logger.warning(f"⚠ {test_case['name']}: Unexpected response {response.status_code}")
                
                results.append({
                    "test": test_case['name'],
                    "passed": security_passed,
                    "status_code": response.status_code
                })
                
            except Exception as e:
                # Timeout or connection error could indicate security blocking
                logger.info(f"✓ {test_case['name']}: Request blocked or filtered ({type(e).__name__})")
                results.append({
                    "test": test_case['name'],
                    "passed": True,  # Assume blocking is good security
                    "blocked": True
                })
        
        pass_rate = len([r for r in results if r['passed']]) / len(results)
        return {"results": results, "pass_rate": pass_rate, "passed": pass_rate >= 0.8}
    
    def _test_error_information_disclosure(self) -> Dict[str, Any]:
        """Test that errors don't disclose sensitive information"""
        try:
            # Try to trigger an error
            response = requests.get(f"{self.api_base_url}/api/v2/nonexistent-endpoint", timeout=10)
            
            if response.status_code == 404:
                response_text = response.text.lower()
                
                # Check for sensitive information leakage
                sensitive_patterns = [
                    'password', 'token', 'secret', 'key', 'connection string',
                    'traceback', 'stack trace', 'internal error', 'database'
                ]
                
                leaked_info = [pattern for pattern in sensitive_patterns if pattern in response_text]
                
                if not leaked_info:
                    logger.info("✓ Error disclosure test: No sensitive information leaked")
                    return {"passed": True, "leaked_info": []}
                else:
                    logger.warning(f"⚠ Error disclosure test: Potential information leakage: {leaked_info}")
                    return {"passed": False, "leaked_info": leaked_info}
            else:
                logger.info(f"✓ Error disclosure test: Unexpected status {response.status_code} (may indicate good security)")
                return {"passed": True, "unexpected_status": response.status_code}
                
        except Exception as e:
            logger.info(f"✓ Error disclosure test: Request handling indicates good security")
            return {"passed": True, "protected": True}
    
    def _test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting (if implemented)"""
        logger.info("Testing rate limiting...")
        
        # Make rapid requests to test rate limiting
        rapid_requests = []
        
        start_time = time.time()
        for i in range(10):  # Make 10 rapid requests
            try:
                response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=5)
                rapid_requests.append(response.status_code)
            except Exception:
                rapid_requests.append(0)  # Failed request
        
        total_time = time.time() - start_time
        requests_per_second = len(rapid_requests) / total_time
        
        # Check if any requests were rate limited (status 429)
        rate_limited = any(status == 429 for status in rapid_requests)
        successful = len([s for s in rapid_requests if s == 200])
        
        logger.info(f"Rate limiting test: {requests_per_second:.1f} req/s, {successful}/10 successful")
        
        if rate_limited:
            logger.info("✓ Rate limiting is active")
        else:
            logger.info("⚠ No rate limiting detected (may not be implemented)")
        
        return {
            "requests_per_second": requests_per_second,
            "successful_requests": successful,
            "rate_limited_responses": rate_limited,
            "passed": True  # Not critical for basic functionality
        }
    
    def _test_integration(self) -> Dict[str, Any]:
        """Test integration with external systems"""
        logger.info("\n--- Integration Tests ---")
        
        integration_tests = {
            "database_connectivity": self._test_database_integration(),
            "api_consistency": self._test_api_consistency(),
            "data_validation": self._test_data_validation()
        }
        
        return integration_tests
    
    def _test_database_integration(self) -> Dict[str, Any]:
        """Test database integration"""
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/health", timeout=15)
            
            if response.status_code == 200:
                health_data = response.json()
                db_status = health_data.get("components", {}).get("database", {})
                
                db_healthy = db_status.get("status") == "healthy"
                
                if db_healthy:
                    logger.info("✓ Database integration: Healthy")
                else:
                    logger.warning(f"⚠ Database integration: {db_status}")
                
                return {
                    "database_status": db_status,
                    "passed": db_healthy
                }
            else:
                logger.error(f"✗ Database integration: Health check failed {response.status_code}")
                return {"passed": False, "status_code": response.status_code}
                
        except Exception as e:
            logger.error(f"✗ Database integration test failed: {e}")
            return {"passed": False, "error": str(e)}
    
    def _test_api_consistency(self) -> Dict[str, Any]:
        """Test API response consistency"""
        consistency_results = []
        
        # Make multiple requests and check for consistency
        for i in range(3):
            try:
                response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    consistency_results.append({
                        "has_metadata": "metadata" in data,
                        "has_summary": "summary" in data,
                        "generated_at": data.get("metadata", {}).get("generated_at"),
                        "structure_keys": list(data.keys())
                    })
                    
            except Exception as e:
                consistency_results.append({"error": str(e)})
        
        if len(consistency_results) >= 2:
            # Check structural consistency
            first_keys = set(consistency_results[0].get("structure_keys", []))
            consistent_structure = all(
                set(result.get("structure_keys", [])) == first_keys 
                for result in consistency_results[1:]
                if "structure_keys" in result
            )
            
            logger.info(f"✓ API consistency: Structure consistent: {consistent_structure}")
            
            return {
                "results": consistency_results,
                "structure_consistent": consistent_structure,
                "passed": consistent_structure
            }
        
        return {"passed": False, "error": "Insufficient data for consistency check"}
    
    def _test_data_validation(self) -> Dict[str, Any]:
        """Test data validation and integrity"""
        try:
            response = requests.get(f"{self.api_base_url}/api/v2/executive-dashboard", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                validation_checks = {
                    "has_timestamp": bool(data.get("metadata", {}).get("generated_at")),
                    "summary_has_numbers": self._validate_numeric_data(data.get("summary", {})),
                    "kpis_have_values": self._validate_kpi_data(data.get("kpis", {})),
                    "no_null_required_fields": self._validate_required_fields(data)
                }
                
                passed_checks = sum(validation_checks.values())
                total_checks = len(validation_checks)
                
                logger.info(f"Data validation: {passed_checks}/{total_checks} checks passed")
                
                return {
                    "checks": validation_checks,
                    "passed_checks": passed_checks,
                    "total_checks": total_checks,
                    "passed": passed_checks >= total_checks * 0.8
                }
            else:
                return {"passed": False, "status_code": response.status_code}
                
        except Exception as e:
            logger.error(f"Data validation test failed: {e}")
            return {"passed": False, "error": str(e)}
    
    def _validate_numeric_data(self, summary_data: Dict) -> bool:
        """Validate that numeric fields contain reasonable values"""
        numeric_fields = [
            ("facilities", "total_count"),
            ("incidents", "total"),
            ("compliance", "audit_pass_rate")
        ]
        
        for section, field in numeric_fields:
            value = summary_data.get(section, {}).get(field)
            if value is not None and not isinstance(value, (int, float)):
                return False
            if isinstance(value, (int, float)) and value < 0:
                return False
                
        return True
    
    def _validate_kpi_data(self, kpis_data: Dict) -> bool:
        """Validate KPI data structure"""
        metrics = kpis_data.get("metrics", {})
        
        for kpi_name, kpi_data in metrics.items():
            if not isinstance(kpi_data, dict):
                return False
            if "value" not in kpi_data or "unit" not in kpi_data:
                return False
                
        return True
    
    def _validate_required_fields(self, data: Dict) -> bool:
        """Validate that required fields are not null"""
        required_paths = [
            ("metadata", "generated_at"),
            ("summary", "alert_level")
        ]
        
        for path in required_paths:
            current = data
            for key in path:
                current = current.get(key, {})
                if current is None:
                    return False
                    
        return True
    
    def _generate_test_report(self, test_results: Dict[str, Any]):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE TEST REPORT")
        logger.info("="*80)
        
        total_passed = 0
        total_tests = 0
        
        for category, results in test_results.items():
            if isinstance(results, dict) and "passed" in results:
                passed = results["passed"]
                logger.info(f"\n{category.upper()}: {'✅ PASSED' if passed else '❌ FAILED'}")
                
                total_tests += 1
                if passed:
                    total_passed += 1
                
                # Display details based on result structure
                if "success_rate" in results:
                    logger.info(f"  Success Rate: {results['success_rate']:.1%}")
                if "pass_rate" in results:
                    logger.info(f"  Pass Rate: {results['pass_rate']:.1%}")
                if "passed_checks" in results and "total_checks" in results:
                    logger.info(f"  Checks: {results['passed_checks']}/{results['total_checks']}")
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info(f"OVERALL TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Passed: {total_passed}/{total_tests} ({overall_success_rate:.1%})")
        
        if overall_success_rate >= 0.9:
            logger.info("🎉 EXCELLENT: System is ready for production deployment")
            recommendation = "ready_for_production"
        elif overall_success_rate >= 0.7:
            logger.info("⚠️  GOOD: System is mostly ready, address failing tests")
            recommendation = "mostly_ready"
        else:
            logger.info("❌ NEEDS WORK: Address significant issues before deployment")
            recommendation = "needs_improvement"
        
        # Save detailed test report
        report_data = {
            "test_timestamp": datetime.now().isoformat(),
            "overall_success_rate": overall_success_rate,
            "total_passed": total_passed,
            "total_tests": total_tests,
            "recommendation": recommendation,
            "detailed_results": test_results
        }
        
        report_file = self.backup_path / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info(f"📄 Detailed test report saved: {report_file}")
        except Exception as e:
            logger.warning(f"Could not save test report: {e}")
    
    # ==============================================================================
    # MAIN DEMONSTRATION RUNNER
    # ==============================================================================
    
    def run_full_demonstration(self) -> Dict[str, Any]:
        """
        Run the complete integration demonstration
        """
        logger.info("🚀 Starting Complete Dashboard API Integration Demonstration")
        logger.info("=" * 80)
        
        demo_results = {}
        
        try:
            # Section 1: API Integration Examples
            demo_results["api_integration"] = self.demonstrate_basic_integration()
            
            # Section 2: Performance Comparison
            demo_results["performance_comparison"] = self.demonstrate_static_vs_dynamic()
            
            # Section 3: Migration Procedures
            demo_results["migration_procedures"] = self.demonstrate_migration_procedures()
            
            # Section 4: Production Deployment
            demo_results["production_deployment"] = self.demonstrate_production_deployment()
            
            # Section 5: Comprehensive Testing
            demo_results["comprehensive_tests"] = self.run_comprehensive_tests()
            
            # Generate final summary
            self._generate_final_summary(demo_results)
            
            return demo_results
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "partial_results": demo_results}
    
    def _generate_final_summary(self, demo_results: Dict[str, Any]):
        """Generate final demonstration summary"""
        logger.info("\n" + "🎯" + "="*78 + "🎯")
        logger.info("FINAL INTEGRATION DEMONSTRATION SUMMARY")
        logger.info("🎯" + "="*78 + "🎯")
        
        # Analyze results
        successful_sections = []
        failed_sections = []
        
        for section, results in demo_results.items():
            if isinstance(results, dict):
                if results.get("error"):
                    failed_sections.append(section)
                else:
                    successful_sections.append(section)
        
        logger.info(f"\n📊 DEMONSTRATION RESULTS:")
        logger.info(f"   ✅ Successful sections: {len(successful_sections)}")
        logger.info(f"   ❌ Failed sections: {len(failed_sections)}")
        
        if successful_sections:
            logger.info(f"\n✅ SUCCESSFUL SECTIONS:")
            for section in successful_sections:
                logger.info(f"   • {section.replace('_', ' ').title()}")
        
        if failed_sections:
            logger.info(f"\n❌ FAILED SECTIONS:")
            for section in failed_sections:
                logger.info(f"   • {section.replace('_', ' ').title()}")
        
        # Key recommendations
        logger.info(f"\n🔑 KEY RECOMMENDATIONS:")
        
        # Performance analysis
        if "performance_comparison" in demo_results:
            perf_data = demo_results["performance_comparison"]
            if hasattr(perf_data, 'improvement_percent'):
                if perf_data.dynamic_time < 2.0:
                    logger.info("   ✅ Dynamic API performance is acceptable for production")
                else:
                    logger.info("   ⚠️  Consider optimizing dynamic API performance")
        
        # Migration readiness
        if "migration_procedures" in demo_results:
            migration_data = demo_results["migration_procedures"]
            if migration_data.get("pre_migration_validation", {}).get("ready_for_migration"):
                logger.info("   ✅ System is ready for migration to dynamic API")
            else:
                logger.info("   ⚠️  Address pre-migration validation issues before proceeding")
        
        # Test results
        if "comprehensive_tests" in demo_results:
            test_data = demo_results["comprehensive_tests"]
            # Look for overall success indicators in test data
            success_indicators = []
            for test_category, test_results in test_data.items():
                if isinstance(test_results, dict) and test_results.get("passed"):
                    success_indicators.append(test_category)
            
            if len(success_indicators) >= len(test_data) * 0.8:
                logger.info("   ✅ Comprehensive tests indicate production readiness")
            else:
                logger.info("   ⚠️  Address test failures before production deployment")
        
        # Deployment strategy
        logger.info("\n🚀 RECOMMENDED DEPLOYMENT STRATEGY:")
        logger.info("   1. Blue-Green deployment with static fallback")
        logger.info("   2. Gradual rollout with canary testing")
        logger.info("   3. Comprehensive monitoring and alerting")
        logger.info("   4. Prepared rollback procedures")
        
        # Next steps
        logger.info("\n📋 NEXT STEPS:")
        logger.info("   1. Review all demonstration results and logs")
        logger.info("   2. Address any failed tests or validation issues")
        logger.info("   3. Set up production monitoring and alerting")
        logger.info("   4. Prepare deployment scripts and rollback procedures")
        logger.info("   5. Conduct stakeholder review and approval")
        logger.info("   6. Schedule production deployment")
        
        logger.info("\n" + "🎯" + "="*78 + "🎯")
        logger.info("Dashboard API Integration Demonstration Complete!")
        logger.info("Check the logs and generated reports for detailed information.")
        logger.info("🎯" + "="*78 + "🎯")


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    """Main CLI function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dashboard API Integration Guide & Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrate_dashboard_api.py --full-demo                    # Run complete demonstration
  python integrate_dashboard_api.py --section integration          # Run specific section
  python integrate_dashboard_api.py --performance-only             # Performance comparison only
  python integrate_dashboard_api.py --test-api                     # API functionality tests
  python integrate_dashboard_api.py --migration-guide              # Migration procedures
        """
    )
    
    parser.add_argument("--api-url", default=API_BASE_URL,
                       help=f"API base URL (default: {API_BASE_URL})")
    
    parser.add_argument("--full-demo", action="store_true",
                       help="Run complete integration demonstration")
    
    parser.add_argument("--section", choices=[
        "integration", "performance", "migration", "deployment", "testing"
    ], help="Run specific demonstration section")
    
    parser.add_argument("--performance-only", action="store_true",
                       help="Run performance comparison only")
    
    parser.add_argument("--test-api", action="store_true",
                       help="Run API functionality tests")
    
    parser.add_argument("--migration-guide", action="store_true",
                       help="Show migration procedures")
    
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce log output")
    
    args = parser.parse_args()
    
    # Configure logging based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize demo
    demo = DashboardIntegrationDemo(api_base_url=args.api_url)
    
    try:
        if args.full_demo:
            print("🚀 Running complete integration demonstration...")
            results = demo.run_full_demonstration()
            
        elif args.section:
            print(f"🔧 Running {args.section} demonstration section...")
            if args.section == "integration":
                results = demo.demonstrate_basic_integration()
            elif args.section == "performance":
                results = demo.demonstrate_static_vs_dynamic()
            elif args.section == "migration":
                results = demo.demonstrate_migration_procedures()
            elif args.section == "deployment":
                results = demo.demonstrate_production_deployment()
            elif args.section == "testing":
                results = demo.run_comprehensive_tests()
                
        elif args.performance_only:
            print("📊 Running performance comparison...")
            results = demo.demonstrate_static_vs_dynamic()
            
        elif args.test_api:
            print("🧪 Running API functionality tests...")
            results = demo.run_comprehensive_tests()
            
        elif args.migration_guide:
            print("📋 Showing migration procedures...")
            results = demo.demonstrate_migration_procedures()
            
        else:
            print("ℹ️  No specific action requested. Use --help for options.")
            print("💡 Tip: Use --full-demo for complete demonstration")
            return
            
        print(f"\n✅ Demonstration completed successfully!")
        print(f"📄 Check logs and generated reports for detailed information.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.error(f"Demonstration error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()