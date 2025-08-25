#!/usr/bin/env python3
"""
Phase 1 Database Schema Migration Script

This script creates all the necessary constraints, indexes, and node types 
for Phase 1 features including:
- Audit Trail Schema (document provenance tracking)
- Pro-Rating Schema (monthly usage allocations)  
- Rejection Tracking Schema (document rejection workflow)

Usage:
    python3 migrate_phase1_schema.py
"""

import sys
import os
import logging
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from phase1_enhancements.audit_trail_schema import AuditTrailSchema
    from phase1_enhancements.prorating_schema import ProRatingSchema
    from phase1_enhancements.rejection_tracking_schema import RejectionTrackingSchema
    from langchain_neo4j import Neo4jGraph
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required dependencies are installed and the src directory structure is correct.")
    sys.exit(1)

def setup_logging():
    """Configure logging for the migration process"""
    log_filename = f"phase1_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename)
        ]
    )
    
    return log_filename

def load_neo4j_credentials():
    """Load Neo4j connection credentials from .env file"""
    try:
        # Try to load from .env file
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                env_vars = {}
                for line in f:
                    if line.strip() and not line.startswith('#') and '=' in line:
                        key, value = line.strip().split('=', 1)
                        # Remove quotes and extra whitespace
                        value = value.strip().strip('"').strip("'").strip()
                        env_vars[key.strip()] = value
                
                return {
                    'uri': env_vars.get('NEO4J_URI', 'bolt://localhost:7687'),
                    'username': env_vars.get('NEO4J_USERNAME', 'neo4j'), 
                    'password': env_vars.get('NEO4J_PASSWORD', ''),
                    'database': env_vars.get('NEO4J_DATABASE', 'neo4j')
                }
        else:
            logging.warning(".env file not found, using default connection settings")
            return {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'EhsAI2024!',
                'database': 'neo4j'
            }
            
    except Exception as e:
        logging.error(f"Error loading Neo4j credentials: {e}")
        raise

def create_neo4j_connection(credentials):
    """Create Neo4j graph connection"""
    try:
        logging.info(f"Connecting to Neo4j at {credentials['uri']}")
        
        graph = Neo4jGraph(
            url=credentials['uri'],
            username=credentials['username'],
            password=credentials['password'],
            database=credentials['database']
        )
        
        # Test the connection
        test_query = "RETURN 'connection_test' as test"
        result = graph.query(test_query)
        
        if result and result[0]['test'] == 'connection_test':
            logging.info("Neo4j connection successful")
            return graph
        else:
            raise Exception("Connection test failed")
            
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        raise

def migrate_audit_trail_schema(graph):
    """Migrate audit trail schema"""
    try:
        logging.info("=== Starting Audit Trail Schema Migration ===")
        
        audit_schema = AuditTrailSchema(graph)
        result = audit_schema.create_audit_trail_schema()
        
        logging.info(f"Audit Trail Migration Result: {result}")
        
        if result.get('validation_passed', False):
            logging.info("✓ Audit Trail Schema migration completed successfully")
            return True
        else:
            logging.warning("⚠ Audit Trail Schema migration completed with warnings")
            return False
            
    except Exception as e:
        logging.error(f"✗ Audit Trail Schema migration failed: {e}")
        raise

def migrate_prorating_schema(graph):
    """Migrate pro-rating schema"""
    try:
        logging.info("=== Starting Pro-Rating Schema Migration ===")
        
        prorating_schema = ProRatingSchema(graph)
        result = prorating_schema.create_allocation_schema()
        
        logging.info(f"Pro-Rating Migration Result: {result}")
        
        if result.get('schema_validation_passed', False):
            logging.info("✓ Pro-Rating Schema migration completed successfully")
            return True
        else:
            logging.warning("⚠ Pro-Rating Schema migration completed with warnings")
            return False
            
    except Exception as e:
        logging.error(f"✗ Pro-Rating Schema migration failed: {e}")
        raise

def migrate_rejection_tracking_schema(graph):
    """Migrate rejection tracking schema"""
    try:
        logging.info("=== Starting Rejection Tracking Schema Migration ===")
        
        rejection_schema = RejectionTrackingSchema(graph)
        result = rejection_schema.create_rejection_schema()
        
        logging.info(f"Rejection Tracking Migration Result: {result}")
        
        if result.get('schema_validation_passed', False):
            logging.info("✓ Rejection Tracking Schema migration completed successfully")
            return True
        else:
            logging.warning("⚠ Rejection Tracking Schema migration completed with warnings")
            return False
            
    except Exception as e:
        logging.error(f"✗ Rejection Tracking Schema migration failed: {e}")
        raise

def run_migration():
    """Run the complete Phase 1 migration process"""
    
    # Setup logging
    log_filename = setup_logging()
    
    try:
        logging.info("Phase 1 Database Schema Migration Started")
        logging.info(f"Log file: {log_filename}")
        
        # Load credentials
        credentials = load_neo4j_credentials()
        logging.info(f"Loaded Neo4j credentials for database: {credentials['database']}")
        
        # Create connection
        graph = create_neo4j_connection(credentials)
        
        # Track migration results
        migration_results = {
            'audit_trail': False,
            'prorating': False, 
            'rejection_tracking': False
        }
        
        # Run migrations
        try:
            migration_results['audit_trail'] = migrate_audit_trail_schema(graph)
        except Exception as e:
            logging.error(f"Audit trail migration failed: {e}")
        
        try:
            migration_results['prorating'] = migrate_prorating_schema(graph)
        except Exception as e:
            logging.error(f"Pro-rating migration failed: {e}")
        
        try:
            migration_results['rejection_tracking'] = migrate_rejection_tracking_schema(graph)
        except Exception as e:
            logging.error(f"Rejection tracking migration failed: {e}")
        
        # Summary
        successful_migrations = sum(migration_results.values())
        total_migrations = len(migration_results)
        
        logging.info("=== Migration Summary ===")
        logging.info(f"Audit Trail Schema: {'✓ SUCCESS' if migration_results['audit_trail'] else '✗ FAILED'}")
        logging.info(f"Pro-Rating Schema: {'✓ SUCCESS' if migration_results['prorating'] else '✗ FAILED'}")
        logging.info(f"Rejection Tracking Schema: {'✓ SUCCESS' if migration_results['rejection_tracking'] else '✗ FAILED'}")
        logging.info(f"Overall: {successful_migrations}/{total_migrations} migrations successful")
        
        if successful_migrations == total_migrations:
            logging.info("🎉 Phase 1 migration completed successfully!")
            return True
        else:
            logging.warning(f"⚠ Phase 1 migration completed with {total_migrations - successful_migrations} failures")
            return False
            
    except Exception as e:
        logging.error(f"Migration process failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = run_migration()
        if success:
            print("\n✓ Phase 1 schema migration completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Phase 1 schema migration completed with errors. Check the log file for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠ Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        sys.exit(1)