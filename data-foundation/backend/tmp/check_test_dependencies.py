#!/usr/bin/env python3
"""
Dependency Checker for EHS Goals API Tests

This script checks if all required dependencies for running the EHS Goals API tests
are properly installed and accessible.
"""

import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        print(f"✅ Python {current_version[0]}.{current_version[1]} (compatible)")
        return True
    else:
        print(f"❌ Python {current_version[0]}.{current_version[1]} (requires >= {min_version[0]}.{min_version[1]})")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - installed")
        return True
    except ImportError:
        print(f"❌ {package_name} - not installed")
        return False

def check_file_exists(file_path, description):
    """Check if a required file exists"""
    path = Path(file_path)
    if path.exists():
        print(f"✅ {description} - found at {file_path}")
        return True
    else:
        print(f"❌ {description} - not found at {file_path}")
        return False

def main():
    """Main dependency checking function"""
    print("EHS Goals API Test Dependencies Check")
    print("=" * 50)
    
    all_good = True
    
    # Check Python version
    print("\n1. Python Version:")
    if not check_python_version():
        all_good = False
    
    # Check required Python packages
    print("\n2. Required Python Packages:")
    required_packages = [
        ("requests", "requests"),
        ("tabulate", "tabulate"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_good = False
    
    # Check project structure
    print("\n3. Project Files:")
    base_path = Path(__file__).parent.parent
    
    project_files = [
        (base_path / "src" / "api" / "ehs_goals_api.py", "EHS Goals API module"),
        (base_path / "src" / "config" / "ehs_goals_config.py", "EHS Goals configuration"),
        (base_path / "tmp" / "test_goals_api.py", "Test script"),
        (base_path / "tmp" / "start_goals_api_server.py", "Server starter script"),
    ]
    
    for file_path, description in project_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    # Check virtual environment
    print("\n4. Virtual Environment:")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment (recommended but not required)")
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("✅ All dependencies are satisfied!")
        print("\nYou can run the tests with:")
        print("  python3 tmp/test_goals_api.py")
        print("\nOr start the server with:")
        print("  python3 tmp/start_goals_api_server.py")
    else:
        print("❌ Some dependencies are missing.")
        print("\nTo install missing Python packages:")
        print("  pip install requests tabulate fastapi uvicorn")
        print("\nTo create a virtual environment:")
        print("  python3 -m venv test_venv")
        print("  source test_venv/bin/activate")
        print("  pip install requests tabulate fastapi uvicorn")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
