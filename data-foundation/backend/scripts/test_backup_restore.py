#!/usr/bin/env python3
"""
Test Script for Neo4j Backup and Restore System

This script tests the backup and restore functionality by:
1. Creating a small test dataset
2. Running backup
3. Clearing database
4. Running restore
5. Validating the restoration

Created: 2025-09-05
Version: 1.0.0
"""

import os
import sys
import logging
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    from backup_neo4j_full import Neo4jBackupManager
    from restore_neo4j_full import Neo4jRestoreManager
except ImportError as e:
    print(f"Error: required packages not installed: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

class BackupRestoreTest:
    """Test suite for backup and restore functionality."""
    
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'password')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        self.driver = None
        self.test_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'backup_restore_test_{self.test_timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Test results
        self.test_results = {
            'connection_test': False,
            'test_data_creation': False,
            'backup_test': False,
            'database_clear': False,
            'restore_test': False,
            'validation_test': False,
            'cleanup_test': False
        }
    
    def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                database=self.database
            )
            
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    self.logger.info("✅ Neo4j connection successful")
                    self.test_results['connection_test'] = True
                    return True
                    
        except Exception as e:
            self.logger.error(f"❌ Neo4j connection failed: {str(e)}")
            return False
    
    def disconnect(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def create_test_data(self) -> bool:
        """Create test data for backup/restore testing."""
        try:
            with self.driver.session() as session:
                # Clear any existing test data first
                session.run("MATCH (n:TestNode) DETACH DELETE n")
                
                # Create test nodes
                test_data_queries = [
                    # Test sites
                    "CREATE (s1:TestNode:Site {test_id: 'TEST001', name: 'Test Site 1', type: 'test_site'})",
                    "CREATE (s2:TestNode:Site {test_id: 'TEST002', name: 'Test Site 2', type: 'test_site'})",
                    
                    # Test facilities
                    "CREATE (f1:TestNode:Facility {test_id: 'TEST001-F1', name: 'Test Facility 1', site_id: 'TEST001'})",
                    "CREATE (f2:TestNode:Facility {test_id: 'TEST001-F2', name: 'Test Facility 2', site_id: 'TEST001'})",
                    "CREATE (f3:TestNode:Facility {test_id: 'TEST002-F1', name: 'Test Facility 3', site_id: 'TEST002'})",
                    
                    # Test consumption data
                    """CREATE (c1:TestNode:ElectricityConsumption {
                        test_id: 'TEST001-E1', 
                        site_id: 'TEST001', 
                        month: 3, 
                        year: 2025, 
                        consumption_kwh: 1500.5,
                        cost: 234.75
                    })""",
                    """CREATE (c2:TestNode:WaterConsumption {
                        test_id: 'TEST001-W1', 
                        site_id: 'TEST001', 
                        month: 3, 
                        year: 2025, 
                        consumption_gallons: 25000,
                        cost: 87.50
                    })""",
                    
                    # Test goals
                    """CREATE (g1:TestNode:Goal {
                        test_id: 'TEST-G1',
                        goal_type: 'energy_reduction',
                        target_value: 10.0,
                        target_unit: 'percent',
                        target_year: 2025
                    })"""
                ]
                
                # Execute creation queries
                for query in test_data_queries:
                    session.run(query)
                
                # Create test relationships
                relationship_queries = [
                    # Site contains facilities
                    """MATCH (s:TestNode:Site {test_id: 'TEST001'}), (f:TestNode:Facility {site_id: 'TEST001'})
                       CREATE (s)-[r:TEST_CONTAINS {created_date: date()}]->(f)""",
                    
                    """MATCH (s:TestNode:Site {test_id: 'TEST002'}), (f:TestNode:Facility {site_id: 'TEST002'})
                       CREATE (s)-[r:TEST_CONTAINS {created_date: date()}]->(f)""",
                    
                    # Consumption relationships
                    """MATCH (s:TestNode:Site {test_id: 'TEST001'}), (c:TestNode:ElectricityConsumption {site_id: 'TEST001'})
                       CREATE (s)-[r:TEST_HAS_CONSUMPTION {type: 'electricity'}]->(c)""",
                    
                    """MATCH (s:TestNode:Site {test_id: 'TEST001'}), (c:TestNode:WaterConsumption {site_id: 'TEST001'})
                       CREATE (s)-[r:TEST_HAS_CONSUMPTION {type: 'water'}]->(c)""",
                    
                    # Goal relationships
                    """MATCH (s:TestNode:Site), (g:TestNode:Goal)
                       CREATE (s)-[r:TEST_HAS_GOAL {assigned_date: date()}]->(g)"""
                ]
                
                for query in relationship_queries:
                    session.run(query)
                
                # Verify test data creation
                result = session.run("MATCH (n:TestNode) RETURN count(n) as test_nodes")
                test_node_count = result.single()["test_nodes"]
                
                result = session.run("MATCH (:TestNode)-[r:TEST_CONTAINS|TEST_HAS_CONSUMPTION|TEST_HAS_GOAL]-(:TestNode) RETURN count(r) as test_rels")
                test_rel_count = result.single()["test_rels"]
                
                self.logger.info(f"✅ Test data created: {test_node_count} nodes, {test_rel_count} relationships")
                self.test_results['test_data_creation'] = True
                
                return test_node_count > 0 and test_rel_count > 0
                
        except Exception as e:
            self.logger.error(f"❌ Test data creation failed: {str(e)}")
            return False
    
    def test_backup(self) -> tuple[bool, str]:
        """Test the backup functionality."""
        try:
            # Create backup manager
            backup_manager = Neo4jBackupManager()
            
            # Run backup
            success = backup_manager.run_backup()
            
            if success:
                backup_path = str(backup_manager.backup_dir)
                self.logger.info(f"✅ Backup test successful: {backup_path}")
                self.test_results['backup_test'] = True
                return True, backup_path
            else:
                self.logger.error("❌ Backup test failed")
                return False, ""
                
        except Exception as e:
            self.logger.error(f"❌ Backup test exception: {str(e)}")
            return False, ""
    
    def clear_test_database(self) -> bool:
        """Clear the test database."""
        try:
            with self.driver.session() as session:
                # Get count before clearing
                result = session.run("MATCH (n:TestNode) RETURN count(n) as before_count")
                before_count = result.single()["before_count"]
                
                # Clear test data
                session.run("MATCH (n:TestNode) DETACH DELETE n")
                
                # Verify clearing
                result = session.run("MATCH (n:TestNode) RETURN count(n) as after_count")
                after_count = result.single()["after_count"]
                
                if after_count == 0:
                    self.logger.info(f"✅ Database cleared: {before_count} test nodes removed")
                    self.test_results['database_clear'] = True
                    return True
                else:
                    self.logger.error(f"❌ Database clear failed: {after_count} nodes remaining")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Database clear failed: {str(e)}")
            return False
    
    def test_restore(self, backup_path: str) -> bool:
        """Test the restore functionality."""
        try:
            # Create restore manager
            restore_manager = Neo4jRestoreManager(
                backup_path=backup_path,
                clear_database=False,  # We already cleared manually
                restore_constraints=False,
                restore_indexes=False
            )
            
            # Run restore
            success = restore_manager.run_restore()
            
            if success:
                self.logger.info("✅ Restore test successful")
                self.test_results['restore_test'] = True
                return True
            else:
                self.logger.error("❌ Restore test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Restore test exception: {str(e)}")
            return False
    
    def validate_restored_data(self) -> bool:
        """Validate that restored data matches original test data."""
        try:
            with self.driver.session() as session:
                # Check test nodes
                result = session.run("MATCH (n:TestNode) RETURN count(n) as node_count")
                restored_nodes = result.single()["node_count"]
                
                # Check test relationships
                result = session.run("MATCH (:TestNode)-[r:TEST_CONTAINS|TEST_HAS_CONSUMPTION|TEST_HAS_GOAL]-(:TestNode) RETURN count(r) as rel_count")
                restored_rels = result.single()["rel_count"]
                
                # Check specific test data
                test_checks = [
                    ("Test sites", "MATCH (n:TestNode:Site) RETURN count(n) as count", 2),
                    ("Test facilities", "MATCH (n:TestNode:Facility) RETURN count(n) as count", 3),
                    ("Test consumption", "MATCH (n:TestNode:ElectricityConsumption) RETURN count(n) as count", 1),
                    ("Test goals", "MATCH (n:TestNode:Goal) RETURN count(n) as count", 1),
                ]
                
                all_checks_passed = True
                
                for check_name, query, expected_count in test_checks:
                    result = session.run(query)
                    actual_count = result.single()["count"]
                    
                    if actual_count == expected_count:
                        self.logger.info(f"✅ {check_name}: {actual_count}/{expected_count}")
                    else:
                        self.logger.error(f"❌ {check_name}: {actual_count}/{expected_count}")
                        all_checks_passed = False
                
                # Check a specific property
                result = session.run("MATCH (n:TestNode:Site {test_id: 'TEST001'}) RETURN n.name as name")
                record = result.single()
                if record and record["name"] == "Test Site 1":
                    self.logger.info("✅ Property validation: Test Site 1 name correct")
                else:
                    self.logger.error("❌ Property validation: Test Site 1 name incorrect")
                    all_checks_passed = False
                
                if all_checks_passed:
                    self.logger.info(f"✅ Validation successful: {restored_nodes} nodes, {restored_rels} relationships")
                    self.test_results['validation_test'] = True
                    return True
                else:
                    self.logger.error("❌ Validation failed: Data mismatch detected")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Validation exception: {str(e)}")
            return False
    
    def cleanup_test_data(self) -> bool:
        """Clean up test data after testing."""
        try:
            with self.driver.session() as session:
                # Remove all test data
                session.run("MATCH (n:TestNode) DETACH DELETE n")
                
                # Verify cleanup
                result = session.run("MATCH (n:TestNode) RETURN count(n) as remaining")
                remaining = result.single()["remaining"]
                
                if remaining == 0:
                    self.logger.info("✅ Test data cleanup successful")
                    self.test_results['cleanup_test'] = True
                    return True
                else:
                    self.logger.warning(f"⚠️ Test data cleanup incomplete: {remaining} nodes remaining")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Test data cleanup failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run the complete test suite."""
        self.logger.info("=== Starting Backup/Restore Test Suite ===")
        
        backup_path = ""
        overall_success = True
        
        try:
            # 1. Connection Test
            self.logger.info("\n1. Testing Neo4j connection...")
            if not self.connect():
                return False
            
            # 2. Create Test Data
            self.logger.info("\n2. Creating test data...")
            if not self.create_test_data():
                overall_success = False
            
            # 3. Test Backup
            self.logger.info("\n3. Testing backup functionality...")
            backup_success, backup_path = self.test_backup()
            if not backup_success:
                overall_success = False
            
            # 4. Clear Database
            self.logger.info("\n4. Clearing database...")
            if not self.clear_test_database():
                overall_success = False
            
            # 5. Test Restore
            if backup_path:
                self.logger.info("\n5. Testing restore functionality...")
                if not self.test_restore(backup_path):
                    overall_success = False
                
                # 6. Validate Restoration
                self.logger.info("\n6. Validating restored data...")
                if not self.validate_restored_data():
                    overall_success = False
            else:
                self.logger.error("❌ Skipping restore test - no backup path available")
                overall_success = False
            
            # 7. Cleanup
            self.logger.info("\n7. Cleaning up test data...")
            if not self.cleanup_test_data():
                self.logger.warning("⚠️ Test cleanup incomplete - manual cleanup may be needed")
            
            # Cleanup backup files if desired
            if backup_path and Path(backup_path).exists():
                try:
                    shutil.rmtree(backup_path)
                    self.logger.info(f"✅ Test backup files cleaned up: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not cleanup backup files: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Test suite exception: {str(e)}")
            overall_success = False
        
        finally:
            self.disconnect()
        
        # Final Results
        self.logger.info("\n=== Test Results Summary ===")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{test_name}: {status}")
        
        passed_tests = sum(self.test_results.values())
        total_tests = len(self.test_results)
        
        self.logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
        
        if overall_success and passed_tests == total_tests:
            self.logger.info("🎉 ALL TESTS PASSED - Backup/Restore system is working correctly!")
            return True
        else:
            self.logger.error("💥 SOME TESTS FAILED - Check logs for details")
            return False


def main():
    """Main function to run the test suite."""
    print("=== Neo4j Backup/Restore Test Suite ===")
    print("This will test the backup and restore functionality")
    print("by creating test data, backing it up, clearing the database,")
    print("and then restoring from backup.\n")
    
    # Verify environment variables
    required_env_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        return False
    
    # Run tests
    test_suite = BackupRestoreTest()
    success = test_suite.run_all_tests()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)