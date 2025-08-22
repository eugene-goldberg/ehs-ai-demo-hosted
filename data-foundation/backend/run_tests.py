#!/usr/bin/env python3
"""
Test runner script for EHS Data Foundation backend tests.

This script provides various options for running the test suite including
specific test categories, coverage reporting, and performance tests.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and return the result."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="EHS Data Foundation Test Runner")
    parser.add_argument(
        "--test-type", 
        choices=["all", "unit", "integration", "api", "performance"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        help="Run tests in parallel (specify number of workers)"
    )
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    parser.add_argument(
        "--function",
        help="Run specific test function"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running"
    )
    
    args = parser.parse_args()
    
    # Change to the backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"], 
                          "Installing test dependencies"):
            print("Failed to install dependencies")
            return 1
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test type markers
    if args.test_type != "all":
        cmd.extend(["-m", args.test_type])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add specific file or function
    if args.file:
        if args.function:
            cmd.append(f"{args.file}::{args.function}")
        else:
            cmd.append(args.file)
    elif args.function:
        cmd.extend(["-k", args.function])
    
    # Run tests
    success = run_command(cmd, f"Running {args.test_type} tests")
    
    if success:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())