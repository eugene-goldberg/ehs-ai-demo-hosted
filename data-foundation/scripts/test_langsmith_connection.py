#!/usr/bin/env python3
"""
LangSmith Connection Test Script

This script performs comprehensive testing of LangSmith API connections,
including detailed debugging, different API endpoints, and troubleshooting
information to diagnose connection issues.

Author: AI Assistant
Date: 2025-08-27
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, Optional, Any
import time
from pathlib import Path

# Add parent directory to Python path to import from project
script_dir = Path(__file__).parent.parent
sys.path.append(str(script_dir))

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")

def print_success(message: str):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_error(message: str):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_info(message: str):
    """Print an info message"""
    print(f"{Colors.CYAN}ℹ {message}{Colors.ENDC}")

def load_environment():
    """Load environment variables from .env file"""
    env_path = script_dir / "web-app" / "backend" / ".env"
    
    print_header("ENVIRONMENT SETUP")
    
    if env_path.exists():
        print_success(f"Found .env file at: {env_path}")
        
        # Load environment variables
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        
        print_success("Environment variables loaded from .env file")
    else:
        print_warning(f".env file not found at: {env_path}")
        print_info("Will check for environment variables in system")

def check_environment_variables():
    """Check for LangSmith API keys in environment"""
    print_header("ENVIRONMENT VARIABLES CHECK")
    
    # Check for different possible API key environment variables
    langsmith_key = os.getenv('LANGSMITH_API_KEY')
    langchain_key = os.getenv('LANGCHAIN_API_KEY')
    
    print(f"LANGSMITH_API_KEY: {'✓ Set' if langsmith_key else '✗ Not set'}")
    if langsmith_key:
        # Show first/last 4 characters of key for verification
        masked_key = f"{langsmith_key[:8]}...{langsmith_key[-8:]}" if len(langsmith_key) > 16 else "***masked***"
        print_info(f"  Value: {masked_key}")
    
    print(f"LANGCHAIN_API_KEY: {'✓ Set' if langchain_key else '✗ Not set'}")
    if langchain_key:
        masked_key = f"{langchain_key[:8]}...{langchain_key[-8:]}" if len(langchain_key) > 16 else "***masked***"
        print_info(f"  Value: {masked_key}")
    
    # Return the primary key to use
    api_key = langsmith_key or langchain_key
    if not api_key:
        print_error("No LangSmith API key found in environment variables!")
        print_info("Expected environment variables: LANGSMITH_API_KEY or LANGCHAIN_API_KEY")
        return None
    
    return api_key

def test_basic_connectivity():
    """Test basic internet connectivity"""
    print_header("BASIC CONNECTIVITY TEST")
    
    try:
        response = requests.get("https://httpbin.org/get", timeout=10)
        if response.status_code == 200:
            print_success("Basic internet connectivity: OK")
            return True
        else:
            print_error(f"Basic connectivity test failed with status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Basic connectivity test failed: {str(e)}")
        return False

def test_langsmith_endpoints(api_key: str):
    """Test different LangSmith API endpoints"""
    print_header("LANGSMITH API ENDPOINTS TEST")
    
    # LangSmith API base URLs to test
    base_urls = [
        "https://api.smith.langchain.com",
        "https://api.langsmith.com"  # Alternative URL
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "langsmith-connection-test/1.0"
    }
    
    successful_url = None
    
    for base_url in base_urls:
        print_info(f"Testing base URL: {base_url}")
        
        # Test different endpoints
        endpoints = [
            "/runs",
            "/projects",
            "/sessions",
            "/feedback",
            "/organizations/current",
            "/workspaces"
        ]
        
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            try:
                print_info(f"  Testing endpoint: {endpoint}")
                response = requests.get(url, headers=headers, timeout=15)
                
                print_info(f"    Status Code: {response.status_code}")
                print_info(f"    Response Headers: {dict(list(response.headers.items())[:3])}")
                
                if response.status_code == 200:
                    print_success(f"    ✓ SUCCESS: {endpoint}")
                    successful_url = base_url
                    
                    # Try to parse response
                    try:
                        data = response.json()
                        if isinstance(data, list):
                            print_info(f"    Response: List with {len(data)} items")
                        elif isinstance(data, dict):
                            print_info(f"    Response keys: {list(data.keys())[:5]}")
                    except:
                        print_info(f"    Response: {response.text[:100]}...")
                        
                elif response.status_code == 401:
                    print_error(f"    ✗ UNAUTHORIZED: Check API key")
                    print_info(f"    Response: {response.text[:200]}")
                elif response.status_code == 403:
                    print_error(f"    ✗ FORBIDDEN: Check permissions")
                    print_info(f"    Response: {response.text[:200]}")
                elif response.status_code == 404:
                    print_warning(f"    ⚠ NOT FOUND: Endpoint may not exist")
                else:
                    print_warning(f"    ⚠ Status {response.status_code}: {response.text[:100]}")
                
            except requests.exceptions.ConnectionError as e:
                print_error(f"    ✗ CONNECTION ERROR: {str(e)}")
            except requests.exceptions.Timeout as e:
                print_error(f"    ✗ TIMEOUT: {str(e)}")
            except requests.exceptions.RequestException as e:
                print_error(f"    ✗ REQUEST ERROR: {str(e)}")
            except Exception as e:
                print_error(f"    ✗ UNEXPECTED ERROR: {str(e)}")
            
            # Small delay between requests
            time.sleep(0.5)
        
        print()  # Empty line between base URLs
    
    return successful_url

def test_langsmith_with_langchain():
    """Test LangSmith using LangChain library if available"""
    print_header("LANGCHAIN LANGSMITH INTEGRATION TEST")
    
    try:
        from langchain.smith import LangChainTracer
        from langsmith import Client
        
        print_success("LangChain and LangSmith libraries available")
        
        # Test LangSmith client
        try:
            client = Client()
            print_success("LangSmith client created successfully")
            
            # Test listing projects
            try:
                projects = list(client.list_projects(limit=5))
                print_success(f"Successfully retrieved {len(projects)} projects")
                for project in projects:
                    print_info(f"  Project: {project.name} (ID: {project.id})")
            except Exception as e:
                print_error(f"Failed to list projects: {str(e)}")
            
            # Test creating a run (minimal test)
            try:
                run = client.create_run(
                    name="connection-test",
                    run_type="chain",
                    inputs={"test": "connection"},
                    start_time=datetime.now()
                )
                print_success(f"Successfully created test run: {run.id}")
                
                # Clean up - end the run
                client.update_run(
                    run.id,
                    outputs={"result": "success"},
                    end_time=datetime.now()
                )
                print_success("Successfully updated test run")
                
            except Exception as e:
                print_error(f"Failed to create/update run: {str(e)}")
            
        except Exception as e:
            print_error(f"Failed to create LangSmith client: {str(e)}")
    
    except ImportError as e:
        print_warning("LangChain or LangSmith library not available")
        print_info(f"Import error: {str(e)}")
        print_info("Install with: pip install langchain langsmith")

def test_api_key_formats(api_key: str):
    """Test API key format and provide debugging info"""
    print_header("API KEY FORMAT ANALYSIS")
    
    print_info(f"API Key length: {len(api_key)}")
    print_info(f"API Key prefix: {api_key[:10]}...")
    print_info(f"API Key suffix: ...{api_key[-10:]}")
    
    # Check expected format patterns
    if api_key.startswith('lsv2_'):
        print_success("API key has expected LangSmith v2 prefix")
    elif api_key.startswith('ls_'):
        print_success("API key has expected LangSmith v1 prefix")
    else:
        print_warning("API key does not have expected prefix (lsv2_ or ls_)")
    
    # Check length (typical LangSmith keys are around 64 characters)
    if 40 <= len(api_key) <= 100:
        print_success("API key length appears reasonable")
    else:
        print_warning(f"API key length ({len(api_key)}) may be unusual")
    
    # Check for common issues
    if api_key.strip() != api_key:
        print_warning("API key has leading/trailing whitespace")
    
    if '\n' in api_key or '\r' in api_key:
        print_warning("API key contains newline characters")

def generate_curl_commands(api_key: str):
    """Generate curl commands for manual testing"""
    print_header("MANUAL CURL COMMANDS FOR TESTING")
    
    base_url = "https://api.smith.langchain.com"
    
    commands = [
        f'curl -H "Authorization: Bearer {api_key}" "{base_url}/projects"',
        f'curl -H "Authorization: Bearer {api_key}" "{base_url}/runs?limit=5"',
        f'curl -H "Authorization: Bearer {api_key}" "{base_url}/organizations/current"'
    ]
    
    print_info("You can test these endpoints manually using curl:")
    for i, cmd in enumerate(commands, 1):
        print(f"\n{Colors.YELLOW}{i}. {cmd[:50]}...{Colors.ENDC}")

def print_troubleshooting_guide():
    """Print comprehensive troubleshooting guide"""
    print_header("TROUBLESHOOTING GUIDE")
    
    issues_and_solutions = [
        {
            "issue": "401 Unauthorized Error",
            "solutions": [
                "Verify your API key is correct and not expired",
                "Check that LANGSMITH_API_KEY or LANGCHAIN_API_KEY is set",
                "Ensure API key has proper permissions",
                "Try regenerating API key from LangSmith dashboard"
            ]
        },
        {
            "issue": "Connection Timeout",
            "solutions": [
                "Check your internet connection",
                "Verify firewall/proxy settings allow HTTPS to api.smith.langchain.com",
                "Try different network (mobile hotspot, etc.)",
                "Check if corporate firewall is blocking the request"
            ]
        },
        {
            "issue": "403 Forbidden",
            "solutions": [
                "Check if your organization has access to LangSmith",
                "Verify your account has proper permissions",
                "Contact your LangSmith administrator"
            ]
        },
        {
            "issue": "Library Import Errors",
            "solutions": [
                "Install required packages: pip install langchain langsmith",
                "Check Python virtual environment is activated",
                "Verify package versions are compatible"
            ]
        }
    ]
    
    for item in issues_and_solutions:
        print(f"\n{Colors.BOLD}{Colors.RED}Issue: {item['issue']}{Colors.ENDC}")
        for solution in item['solutions']:
            print(f"  • {solution}")
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Additional Resources:{Colors.ENDC}")
    print("  • LangSmith Documentation: https://docs.smith.langchain.com/")
    print("  • LangSmith API Reference: https://api.smith.langchain.com/redoc")
    print("  • LangChain Documentation: https://python.langchain.com/docs/")

def main():
    """Main test function"""
    print_header("LANGSMITH CONNECTION DIAGNOSTIC TOOL")
    print_info(f"Test started at: {datetime.now()}")
    print_info(f"Python version: {sys.version}")
    
    # Load environment variables
    load_environment()
    
    # Check environment variables
    api_key = check_environment_variables()
    if not api_key:
        print_error("Cannot proceed without API key")
        print_troubleshooting_guide()
        return 1
    
    # Test basic connectivity
    if not test_basic_connectivity():
        print_error("Basic connectivity failed - check internet connection")
        return 1
    
    # Analyze API key format
    test_api_key_formats(api_key)
    
    # Test LangSmith endpoints
    successful_url = test_langsmith_endpoints(api_key)
    
    if successful_url:
        print_success(f"Successfully connected to LangSmith at: {successful_url}")
    else:
        print_error("Failed to connect to any LangSmith endpoints")
    
    # Test with LangChain integration
    test_langsmith_with_langchain()
    
    # Generate curl commands for manual testing
    generate_curl_commands(api_key)
    
    # Print troubleshooting guide
    print_troubleshooting_guide()
    
    print_header("TEST SUMMARY")
    if successful_url:
        print_success("LangSmith connection test PASSED")
        print_info("Your LangSmith connection is working properly")
        return 0
    else:
        print_error("LangSmith connection test FAILED")
        print_info("Review the troubleshooting guide above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)