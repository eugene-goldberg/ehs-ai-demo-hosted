#!/usr/bin/env python3
"""
Test Runner for Rejection Tracking Real Database Tests

This script runs the rejection tracking tests with proper environment setup
and provides detailed output for verification of real database operations.

Usage:
    python run_rejection_tests.py
    python run_rejection_tests.py --verbose
    python run_rejection_tests.py --test-name test_reject_document_real_database
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables and Python path for testing."""
    # Add src directory to Python path
    backend_dir = Path(__file__).parent
    src_dir = backend_dir / "src"
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Load environment variables
    from dotenv import load_dotenv
    env_file = backend_dir / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        logger.warning(f"Environment file not found: {env_file}")
    
    # Verify required environment variables
    required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("Environment setup completed successfully")
    return True

def run_tests(test_name=None, verbose=False, capture_output=True):
    """Run rejection tracking tests with pytest."""
    logger.info("Starting rejection tracking real database tests")
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Prepare pytest command
    test_file = Path(__file__).parent / "tests" / "test_rejection_tracking_real.py"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    pytest_args = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v" if verbose else "-q",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ]
    
    # Add specific test if provided
    if test_name:
        pytest_args.extend(["-k", test_name])
    
    # Add output capture control
    if not capture_output:
        pytest_args.append("-s")
    
    logger.info(f"Running pytest with args: {' '.join(pytest_args)}")
    
    # Run tests
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            pytest_args,
            cwd=Path(__file__).parent,
            env=os.environ.copy(),
            capture_output=capture_output,
            text=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Tests completed in {duration:.2f} seconds")
        
        if result.returncode == 0:
            logger.info("✅ All tests passed successfully!")
            if capture_output and result.stdout:
                print("\n" + "="*60)
                print("TEST OUTPUT:")
                print("="*60)
                print(result.stdout)
        else:
            logger.error(f"❌ Tests failed with return code {result.returncode}")
            if capture_output:
                if result.stdout:
                    print("\n" + "="*60)
                    print("TEST OUTPUT:")
                    print("="*60)
                    print(result.stdout)
                if result.stderr:
                    print("\n" + "="*60)
                    print("ERROR OUTPUT:")
                    print("="*60)
                    print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        return False

def run_specific_test_scenarios():
    """Run specific test scenarios to demonstrate functionality."""
    scenarios = [
        {
            "name": "Basic Document Rejection",
            "test": "test_reject_document_real_database",
            "description": "Tests basic document rejection with real database updates"
        },
        {
            "name": "Document Unreejection",
            "test": "test_unreject_document_real_database", 
            "description": "Tests document unreejection with database state restoration"
        },
        {
            "name": "Rejected Documents Retrieval",
            "test": "test_get_rejected_documents_real_data",
            "description": "Tests getting rejected documents from real database"
        },
        {
            "name": "Rejection Statistics",
            "test": "test_rejection_statistics_real_data",
            "description": "Tests rejection statistics generation from real data"
        },
        {
            "name": "Bulk Rejection Operations", 
            "test": "test_bulk_rejection_real_database",
            "description": "Tests bulk rejection with multiple documents"
        },
        {
            "name": "Error Handling",
            "test": "test_error_handling_real_database",
            "description": "Tests error scenarios with database constraints"
        },
        {
            "name": "End-to-End Workflow",
            "test": "test_end_to_end_rejection_workflow",
            "description": "Tests complete rejection workflow"
        }
    ]
    
    print("\n" + "="*70)
    print("REJECTION TRACKING TEST SCENARIOS")
    print("="*70)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        logger.info(f"Running scenario {i}: {scenario['name']}")
        
        success = run_tests(
            test_name=scenario["test"],
            verbose=True,
            capture_output=True
        )
        
        if success:
            print(f"   Result: ✅ PASSED")
        else:
            print(f"   Result: ❌ FAILED")
    
    print("\n" + "="*70)
    print("ALL SCENARIOS COMPLETED")
    print("="*70)

def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run rejection tracking real database tests"
    )
    
    parser.add_argument(
        "--test-name", "-t",
        help="Run specific test by name (e.g., test_reject_document_real_database)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests with verbose output"
    )
    
    parser.add_argument(
        "--scenarios", "-s",
        action="store_true",
        help="Run all test scenarios individually with descriptions"
    )
    
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Don't capture test output (useful for debugging)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("REJECTION TRACKING REAL DATABASE TESTS")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    if args.scenarios:
        run_specific_test_scenarios()
    else:
        success = run_tests(
            test_name=args.test_name,
            verbose=args.verbose,
            capture_output=not args.no_capture
        )
        
        if success:
            print("\n🎉 All tests completed successfully!")
        else:
            print("\n💥 Some tests failed. Check the output above for details.")
            sys.exit(1)

if __name__ == "__main__":
    main()