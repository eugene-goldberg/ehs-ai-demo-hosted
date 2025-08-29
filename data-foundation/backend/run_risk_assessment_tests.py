#!/usr/bin/env python3
"""
Risk Assessment Integration Test Runner

This script provides an easy way to run the comprehensive risk assessment
workflow integration tests with proper environment setup and logging.

Usage:
    python3 run_risk_assessment_tests.py [--verbose] [--test-name TEST_NAME]
    
Examples:
    python3 run_risk_assessment_tests.py
    python3 run_risk_assessment_tests.py --verbose
    python3 run_risk_assessment_tests.py --test-name test_complete_workflow
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the test environment is properly configured."""
    logger.info("Checking test environment...")
    
    # Check required environment variables
    required_vars = ['OPENAI_API_KEY', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file and ensure all required variables are set.")
        return False
    
    logger.info("✓ Environment variables configured")
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = ['pytest', 'neo4j', 'langchain', 'openai', 'pydantic']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install -r tests/requirements-test.txt")
        return False
    
    logger.info("✓ Dependencies available")
    return True

def check_neo4j_connection():
    """Check if Neo4j is accessible."""
    logger.info("Checking Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USERNAME', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        
        logger.info("✓ Neo4j connection successful")
        return True
        
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        logger.error("Please ensure Neo4j is running and credentials are correct.")
        return False

def run_tests(verbose=False, test_name=None):
    """Run the risk assessment integration tests."""
    logger.info("Running risk assessment integration tests...")
    
    # Build pytest command
    cmd = ['python3', '-m', 'pytest']
    
    # Add test file path
    test_path = 'tests/agents/risk_assessment/test_workflow_integration.py'
    
    if test_name:
        test_path += f'::{test_name}'
    
    cmd.append(test_path)
    
    # Add verbosity flags
    if verbose:
        cmd.extend(['-v', '-s'])
    
    # Add output formatting
    cmd.extend(['--tb=short'])
    
    # Add logging capture
    cmd.extend(['--capture=no'])
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        # Run the tests
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=False)
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        return False

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run Risk Assessment Integration Tests')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--test-name', '-t', type=str,
                       help='Run specific test function (e.g., test_complete_workflow)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip environment and dependency checks')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("RISK ASSESSMENT INTEGRATION TEST RUNNER")
    logger.info("="*60)
    
    # Pre-flight checks
    if not args.skip_checks:
        logger.info("\nPerforming pre-flight checks...")
        
        if not check_environment():
            logger.error("Environment check failed. Aborting tests.")
            sys.exit(1)
        
        if not check_dependencies():
            logger.error("Dependency check failed. Aborting tests.")
            sys.exit(1)
        
        if not check_neo4j_connection():
            logger.error("Neo4j connection check failed. Aborting tests.")
            sys.exit(1)
        
        logger.info("✓ All pre-flight checks passed")
    else:
        logger.info("Skipping pre-flight checks as requested")
    
    # Run tests
    logger.info(f"\nStarting test execution...")
    if args.test_name:
        logger.info(f"Running specific test: {args.test_name}")
    else:
        logger.info("Running all integration tests")
    
    success = run_tests(verbose=args.verbose, test_name=args.test_name)
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("Risk assessment workflow integration tests completed successfully.")
    else:
        logger.error("❌ TESTS FAILED!")
        logger.error("Check the output above for details about test failures.")
    logger.info("="*60)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()