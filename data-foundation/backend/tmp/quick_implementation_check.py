#!/usr/bin/env python3
"""
Quick Implementation Status Check
Based on the server logs and current state analysis.
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime

def check_implementation_status():
    """Check current implementation status based on available information"""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "server_status": {},
            "api_endpoints": {},
            "codebase_files": {},
            "neo4j_status": {},
            "conclusions": []
        }
    }
    
    base_path = Path("/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend")
    
    print("EHS Implementation Status Check")
    print("=" * 50)
    
    # 1. Check if server is running
    print("\n1. Checking API Server Status...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        results["analysis"]["server_status"] = {
            "running": True,
            "health_status": response.status_code,
            "response": response.text[:200] if response.text else "No response body"
        }
        print(f"   ✓ Server running (status: {response.status_code})")
    except Exception as e:
        results["analysis"]["server_status"] = {
            "running": False,
            "error": str(e)
        }
        print(f"   ✗ Server not running: {e}")
    
    # 2. Check for environmental API endpoints
    print("\n2. Testing Environmental API Endpoints...")
    endpoints_to_test = [
        "/api/environmental/electricity",
        "/api/environmental/water", 
        "/api/environmental/waste",
        "/api/environmental/co2",
        "/api/environmental/summary",
        "/dashboard/executive",
        "/goals/annual"
    ]
    
    working_endpoints = 0
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            status = "Working" if response.status_code == 200 else f"Error {response.status_code}"
            results["analysis"]["api_endpoints"][endpoint] = {
                "status_code": response.status_code,
                "working": response.status_code == 200
            }
            if response.status_code == 200:
                working_endpoints += 1
            print(f"   {endpoint}: {status}")
        except Exception as e:
            results["analysis"]["api_endpoints"][endpoint] = {
                "error": str(e),
                "working": False
            }
            print(f"   {endpoint}: Connection failed")
    
    print(f"   Summary: {working_endpoints}/{len(endpoints_to_test)} endpoints working")
    
    # 3. Check key source files
    print("\n3. Checking Key Implementation Files...")
    key_files = [
        "src/main.py",
        "src/ehs_extraction_api.py", 
        "src/api/environmental_api.py",
        "src/api/dashboard_api.py",
        "src/services/environmental_service.py",
        "src/database/neo4j_client.py"
    ]
    
    existing_files = 0
    for file_path in key_files:
        full_path = base_path / file_path
        if full_path.exists():
            existing_files += 1
            # Check file size and key content
            with open(full_path, 'r') as f:
                content = f.read()
            
            results["analysis"]["codebase_files"][file_path] = {
                "exists": True,
                "size_kb": len(content) // 1024,
                "has_environmental_logic": "environmental" in content.lower(),
                "has_co2_logic": "co2" in content.lower() or "carbon" in content.lower(),
                "has_api_routes": "@app." in content or "@router." in content
            }
            print(f"   ✓ {file_path} ({len(content) // 1024}KB)")
        else:
            results["analysis"]["codebase_files"][file_path] = {"exists": False}
            print(f"   ✗ {file_path} (missing)")
    
    print(f"   Summary: {existing_files}/{len(key_files)} key files exist")
    
    # 4. Analyze server logs for clues
    print("\n4. Analyzing Recent Server Logs...")
    log_files = [
        "tmp/api_server_verify2.log",
        "environmental_assessment_test.log"
    ]
    
    environmental_mentions = 0
    neo4j_errors = 0
    
    for log_file in log_files:
        log_path = base_path / log_file
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            # Count relevant mentions
            env_count = log_content.lower().count("environmental")
            neo4j_error_count = log_content.count("Neo.ClientError")
            
            environmental_mentions += env_count
            neo4j_errors += neo4j_error_count
            
            print(f"   {log_file}: {env_count} environmental mentions, {neo4j_error_count} Neo4j errors")
    
    results["analysis"]["neo4j_status"] = {
        "authentication_errors": neo4j_errors > 0,
        "error_count": neo4j_errors
    }
    
    # 5. Generate conclusions
    print("\n5. Analysis Conclusions:")
    conclusions = []
    
    if results["analysis"]["server_status"]["running"]:
        conclusions.append("✓ API server is running successfully")
    else:
        conclusions.append("✗ API server is not running")
    
    if working_endpoints > 0:
        conclusions.append(f"✓ {working_endpoints} API endpoints are responding")
    else:
        conclusions.append("✗ No environmental API endpoints are working")
    
    if existing_files >= len(key_files) * 0.7:  # 70% of files exist
        conclusions.append("✓ Most key implementation files exist")
    else:
        conclusions.append("✗ Many key implementation files are missing")
    
    if neo4j_errors > 0:
        conclusions.append("✗ Neo4j authentication/connection issues detected")
    else:
        conclusions.append("? Neo4j status unclear from logs")
    
    if environmental_mentions > 10:
        conclusions.append("✓ Environmental functionality is actively being processed")
    else:
        conclusions.append("? Limited environmental functionality detected")
    
    results["analysis"]["conclusions"] = conclusions
    
    for conclusion in conclusions:
        print(f"   {conclusion}")
    
    # 6. Implementation Status Assessment
    print("\n6. Overall Implementation Status:")
    
    # Calculate completion percentage
    completion_factors = {
        "server_running": 1 if results["analysis"]["server_status"]["running"] else 0,
        "files_exist": existing_files / len(key_files),
        "apis_working": working_endpoints / len(endpoints_to_test),
        "no_critical_errors": 0 if neo4j_errors > 5 else 1  # Some errors are okay
    }
    
    overall_completion = sum(completion_factors.values()) / len(completion_factors) * 100
    
    print(f"   Overall Completion: {overall_completion:.1f}%")
    
    if overall_completion >= 80:
        status = "MOSTLY IMPLEMENTED - Minor issues to fix"
    elif overall_completion >= 60:
        status = "PARTIALLY IMPLEMENTED - Significant work needed"
    elif overall_completion >= 40:
        status = "BASIC FRAMEWORK - Major development required"
    else:
        status = "MINIMAL IMPLEMENTATION - Extensive work needed"
    
    print(f"   Status: {status}")
    
    # 7. Next Steps Recommendations
    print(f"\n7. Immediate Next Steps:")
    
    if neo4j_errors > 0:
        print("   1. Fix Neo4j authentication issues")
    
    if working_endpoints == 0:
        print("   2. Debug and fix API endpoint registration")
    
    if existing_files < len(key_files):
        print("   3. Create missing implementation files")
    
    print("   4. Run comprehensive tests to verify functionality")
    
    # Save results
    report_path = base_path / "tmp" / "quick_status_check.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {report_path}")
    
    return results

if __name__ == "__main__":
    check_implementation_status()