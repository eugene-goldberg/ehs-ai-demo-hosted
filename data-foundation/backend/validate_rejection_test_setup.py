#!/usr/bin/env python3
"""
Validation Script for Rejection Tracking Test Setup

This script validates that all dependencies and configurations are properly
set up for running the rejection tracking real database tests.

Checks:
1. Environment variables
2. Neo4j database connectivity  
3. FastAPI application imports
4. Test database configuration
5. Required Python packages

Usage:
    python validate_rejection_test_setup.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment_variables():
    """Check required environment variables."""
    logger.info("Checking environment variables...")
    
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
    except ImportError:
        logger.warning("python-dotenv not available, relying on system environment")
    
    required_vars = [
        'NEO4J_URI',
        'NEO4J_USERNAME', 
        'NEO4J_PASSWORD',
        'NEO4J_DATABASE'
    ]
    
    missing_vars = []
    present_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            present_vars.append(var)
            # Don't log sensitive values
            if 'PASSWORD' in var:
                logger.info(f"✅ {var}: [REDACTED]")
            else:
                logger.info(f"✅ {var}: {value}")
        else:
            missing_vars.append(var)
            logger.error(f"❌ {var}: Not set")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("✅ All required environment variables are present")
    return True

def check_neo4j_connectivity():
    """Check Neo4j database connectivity."""
    logger.info("Testing Neo4j connectivity...")
    
    try:
        # Add src to path for imports
        src_dir = Path(__file__).parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        from neo4j import GraphDatabase
        
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        if not neo4j_password:
            logger.error("❌ Neo4j password not available")
            return False
        
        # Create driver and test connection
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        
        with driver.session(database=neo4j_database) as session:
            result = session.run("RETURN 'connection test' as test")
            record = result.single()
            
            if record and record["test"] == "connection test":
                logger.info("✅ Neo4j connection successful")
                
                # Get some basic database info
                result = session.run("CALL dbms.components() YIELD name, versions")
                components = list(result)
                
                for component in components:
                    logger.info(f"   {component['name']}: {component['versions']}")
                
                driver.close()
                return True
            else:
                logger.error("❌ Neo4j connection test failed")
                driver.close()
                return False
                
    except Exception as e:
        logger.error(f"❌ Neo4j connectivity check failed: {e}")
        return False

def check_fastapi_imports():
    """Check FastAPI application and related imports."""
    logger.info("Checking FastAPI imports...")
    
    try:
        # Add src to path
        src_dir = Path(__file__).parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        # Test core imports
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        logger.info("✅ FastAPI imports successful")
        
        # Test application import
        from ehs_extraction_api import app
        logger.info("✅ EHS Extraction API import successful")
        
        # Test rejection tracking imports
        from phase1_enhancements.rejection_tracking_api import router
        logger.info("✅ Rejection tracking API import successful")
        
        from phase1_enhancements.rejection_tracking_schema import DocumentStatus
        logger.info("✅ Rejection tracking schema import successful")
        
        from phase1_enhancements.rejection_workflow_service import RejectionReason
        logger.info("✅ Rejection workflow service import successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FastAPI imports failed: {e}")
        return False

def check_test_database_config():
    """Check test database configuration."""
    logger.info("Checking test database configuration...")
    
    try:
        # Add tests to path
        tests_dir = Path(__file__).parent / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        
        from test_database import Neo4jTestClient, TestDataFactory
        logger.info("✅ Test database imports successful")
        
        # Test creating a test client
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        client = Neo4jTestClient(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )
        
        client.connect()
        
        # Test basic operations
        result = client.execute_query("RETURN 'test' as value")
        if result and result[0]["value"] == "test":
            logger.info("✅ Test database client functional")
        else:
            logger.error("❌ Test database client query failed")
            return False
        
        # Test data factory
        factory = TestDataFactory(client)
        test_doc_id = factory.create_test_document(
            doc_type="validation_test",
            content="Test document for validation",
            metadata={"validation": True}
        )
        
        logger.info(f"✅ Test data factory functional (created doc: {test_doc_id})")
        
        # Cleanup test data
        deleted = client.cleanup_test_data()
        logger.info(f"✅ Test cleanup successful ({deleted} items cleaned)")
        
        client.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Test database configuration check failed: {e}")
        return False

def check_required_packages():
    """Check required Python packages."""
    logger.info("Checking required Python packages...")
    
    required_packages = [
        'fastapi',
        'pytest',
        'neo4j',
        'httpx',
        'pydantic',
        'python-dotenv',
        'uvicorn'
    ]
    
    missing_packages = []
    present_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            present_packages.append(package)
            logger.info(f"✅ {package}: Available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}: Not available")
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("✅ All required packages are available")
    return True

def check_test_files():
    """Check that test files exist and are readable."""
    logger.info("Checking test files...")
    
    base_dir = Path(__file__).parent
    
    required_files = [
        "tests/test_database.py",
        "tests/test_rejection_tracking_real.py",
        "src/ehs_extraction_api.py",
        "src/phase1_enhancements/rejection_tracking_api.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists() and full_path.is_file():
            logger.info(f"✅ {file_path}: Found")
        else:
            missing_files.append(file_path)
            logger.error(f"❌ {file_path}: Not found")
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("✅ All required test files are present")
    return True

def run_validation():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("REJECTION TRACKING TEST SETUP VALIDATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print("="*70)
    
    checks = [
        ("Environment Variables", check_environment_variables),
        ("Required Python Packages", check_required_packages),
        ("Test Files", check_test_files),
        ("FastAPI Imports", check_fastapi_imports),
        ("Neo4j Connectivity", check_neo4j_connectivity),
        ("Test Database Configuration", check_test_database_config),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 50)
        
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f"Check '{check_name}' raised exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{check_name:<40} {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 70)
    print(f"Total Checks: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All validation checks passed!")
        print("You can now run the rejection tracking tests with:")
        print("  python run_rejection_tests.py")
        return True
    else:
        print(f"\n💥 {failed} validation checks failed!")
        print("Please fix the issues above before running tests.")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)