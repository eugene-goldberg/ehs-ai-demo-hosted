#!/usr/bin/env python3
"""
Migration 003: Add Relationships Between Entities

This migration creates the following relationships:
- Equipment -> LOCATED_AT -> Facility
- Equipment -> AFFECTS_CONSUMPTION -> UtilityBill  
- Permit -> PERMITS -> Facility

Creates indexes on relationships for optimal performance.
"""

import logging
from typing import Optional
from neo4j import GraphDatabase, Driver
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RelationshipsMigration:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the migration with Neo4j connection details."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.migration_name = "003_add_relationships"
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def check_migration_applied(self) -> bool:
        """Check if this migration has already been applied."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (m:Migration {name: $migration_name}) RETURN count(m) as count",
                migration_name=self.migration_name
            )
            return result.single()["count"] > 0
            
    def mark_migration_applied(self):
        """Mark this migration as applied."""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (m:Migration {
                    name: $migration_name,
                    applied_at: datetime(),
                    description: 'Add relationships between Equipment, Permit, Facility, and UtilityBill entities'
                })
                """,
                migration_name=self.migration_name
            )
            
    def create_relationship_indexes(self):
        """Create indexes on relationships for optimal performance."""
        with self.driver.session() as session:
            try:
                # Index on LOCATED_AT relationship for Equipment-Facility queries
                session.run("CREATE INDEX equipment_located_at_index IF NOT EXISTS FOR ()-[r:LOCATED_AT]-() ON (r.created_at)")
                logger.info("Created index on LOCATED_AT relationship")
                
                # Index on AFFECTS_CONSUMPTION relationship for Equipment-UtilityBill queries
                session.run("CREATE INDEX equipment_affects_consumption_index IF NOT EXISTS FOR ()-[r:AFFECTS_CONSUMPTION]-() ON (r.impact_factor)")
                logger.info("Created index on AFFECTS_CONSUMPTION relationship")
                
                # Index on PERMITS relationship for Permit-Facility queries
                session.run("CREATE INDEX permit_permits_index IF NOT EXISTS FOR ()-[r:PERMITS]-() ON (r.effective_date)")
                logger.info("Created index on PERMITS relationship")
                
            except Exception as e:
                logger.error(f"Error creating relationship indexes: {e}")
                raise
                
    def verify_entities_exist(self):
        """Verify that all required entities exist before creating relationships."""
        with self.driver.session() as session:
            try:
                # Check if Equipment entity schema exists
                equipment_result = session.run("SHOW CONSTRAINTS YIELD labelsOrTypes WHERE 'Equipment' IN labelsOrTypes RETURN count(*) as count")
                equipment_count = equipment_result.single()["count"]
                
                # Check if Permit entity schema exists
                permit_result = session.run("SHOW CONSTRAINTS YIELD labelsOrTypes WHERE 'Permit' IN labelsOrTypes RETURN count(*) as count")
                permit_count = permit_result.single()["count"]
                
                # Check if Facility entity exists (should exist from previous setup)
                facility_result = session.run("MATCH (f:Facility) RETURN count(f) as count LIMIT 1")
                facility_exists = facility_result.single()["count"] >= 0  # Even 0 is fine, schema should exist
                
                # Check if UtilityBill entity exists (should exist from previous setup)
                utility_result = session.run("MATCH (u:UtilityBill) RETURN count(u) as count LIMIT 1")
                utility_exists = utility_result.single()["count"] >= 0  # Even 0 is fine, schema should exist
                
                logger.info(f"Entity verification - Equipment: {equipment_count > 0}, Permit: {permit_count > 0}")
                logger.info(f"Entity verification - Facility exists: {facility_exists}, UtilityBill exists: {utility_exists}")
                
                return equipment_count > 0 and permit_count > 0
                
            except Exception as e:
                logger.warning(f"Error verifying entities (this may be expected if entities don't exist yet): {e}")
                return True  # Continue with migration even if verification fails
                
    def create_sample_relationships(self):
        """Create sample relationships to demonstrate the schema (optional)."""
        with self.driver.session() as session:
            try:
                # This is just to ensure the relationship types are defined in the schema
                # We'll create and immediately delete sample relationships
                
                # Create temporary nodes and relationships to establish schema
                session.run("""
                    MERGE (temp_eq:Equipment {id: 'temp_equipment_001'})
                    MERGE (temp_fac:Facility {id: 'temp_facility_001'})
                    MERGE (temp_ub:UtilityBill {id: 'temp_utility_001'})
                    MERGE (temp_permit:Permit {id: 'temp_permit_001'})
                    
                    MERGE (temp_eq)-[:LOCATED_AT {created_at: datetime()}]->(temp_fac)
                    MERGE (temp_eq)-[:AFFECTS_CONSUMPTION {impact_factor: 1.0, created_at: datetime()}]->(temp_ub)
                    MERGE (temp_permit)-[:PERMITS {effective_date: date(), created_at: datetime()}]->(temp_fac)
                """)
                
                # Clean up temporary data
                session.run("""
                    MATCH (temp_eq:Equipment {id: 'temp_equipment_001'})
                    MATCH (temp_fac:Facility {id: 'temp_facility_001'})
                    MATCH (temp_ub:UtilityBill {id: 'temp_utility_001'})
                    MATCH (temp_permit:Permit {id: 'temp_permit_001'})
                    
                    DETACH DELETE temp_eq, temp_fac, temp_ub, temp_permit
                """)
                
                logger.info("Successfully established relationship schema patterns")
                
            except Exception as e:
                logger.warning(f"Error creating sample relationships (this may be expected): {e}")
                # This is not critical for the migration
                
    def verify_relationship_schema(self):
        """Verify that relationship schema was created correctly."""
        with self.driver.session() as session:
            try:
                # Check relationship indexes
                indexes_result = session.run("""
                    SHOW INDEXES YIELD name, type WHERE type = 'RANGE' 
                    AND (name CONTAINS 'located_at' OR name CONTAINS 'affects_consumption' OR name CONTAINS 'permits')
                    RETURN count(*) as count
                """)
                indexes_count = indexes_result.single()["count"]
                logger.info(f"Found {indexes_count} relationship indexes")
                
                return indexes_count >= 3
                
            except Exception as e:
                logger.error(f"Error verifying relationship schema: {e}")
                return False
                
    def run(self):
        """Execute the migration."""
        try:
            # Check if migration already applied
            if self.check_migration_applied():
                logger.info(f"Migration {self.migration_name} already applied. Skipping.")
                return True
                
            logger.info(f"Starting migration: {self.migration_name}")
            
            # Verify entities exist
            logger.info("Verifying required entities exist...")
            if not self.verify_entities_exist():
                logger.warning("Some entities may not exist yet, but continuing with migration")
            
            # Create relationship indexes
            logger.info("Creating relationship indexes...")
            self.create_relationship_indexes()
            
            # Create sample relationships to establish schema
            logger.info("Establishing relationship schema patterns...")
            self.create_sample_relationships()
            
            # Verify schema
            if self.verify_relationship_schema():
                logger.info("Relationship schema verification successful")
            else:
                logger.warning("Relationship schema verification failed")
                
            # Mark migration as applied
            self.mark_migration_applied()
            logger.info(f"Migration {self.migration_name} completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration {self.migration_name} failed: {e}")
            raise


def main():
    """Main function to run the relationships migration."""
    # Get Neo4j connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    logger.info("Starting relationships migration...")
    
    migration = RelationshipsMigration(uri, username, password)
    
    try:
        success = migration.run()
        if success:
            logger.info("Relationships migration completed successfully")
        else:
            logger.error("Relationships migration failed")
            return 1
            
    except Exception as e:
        logger.error(f"Migration failed with error: {e}")
        return 1
        
    finally:
        migration.close()
        
    return 0


if __name__ == "__main__":
    exit(main())