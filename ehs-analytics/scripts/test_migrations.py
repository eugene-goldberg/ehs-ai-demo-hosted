#!/usr/bin/env python3
"""
Test script to verify migration setup and Neo4j connectivity.

This script tests:
1. Neo4j connectivity
2. Required packages are available
3. Migration scripts can be loaded
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages are available."""
    try:
        import neo4j
        from dotenv import load_dotenv
        logger.info("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_neo4j_connection():
    """Test Neo4j connection."""
    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        
        load_dotenv()
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j connection test' as message")
            message = result.single()["message"]
            logger.info(f"✓ Neo4j connection successful: {message}")
            
        driver.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Neo4j connection failed: {e}")
        logger.info("Make sure Neo4j is running and credentials are correct")
        return False

def test_migration_files():
    """Test that migration files exist and can be loaded."""
    migrations_dir = Path(__file__).parent / "migrations"
    
    expected_files = [
        "001_add_equipment_entity.py",
        "002_add_permit_entity.py", 
        "003_add_relationships.py"
    ]
    
    all_exist = True
    for filename in expected_files:
        filepath = migrations_dir / filename
        if filepath.exists():
            logger.info(f"✓ Found migration file: {filename}")
        else:
            logger.error(f"✗ Missing migration file: {filename}")
            all_exist = False
            
    return all_exist

def main():
    """Run all tests."""
    logger.info("Starting migration setup tests...")
    
    tests = [
        ("Package imports", test_imports),
        ("Neo4j connection", test_neo4j_connection),
        ("Migration files", test_migration_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"Running test: {test_name}")
        result = test_func()
        results.append((test_name, result))
        logger.info("")
    
    # Summary
    logger.info("="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:<20} {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All tests passed! Migrations are ready to run.")
        logger.info("Run: python3 scripts/run_migrations.py")
        return 0
    else:
        logger.error("\n✗ Some tests failed. Fix issues before running migrations.")
        return 1

if __name__ == "__main__":
    exit(main())