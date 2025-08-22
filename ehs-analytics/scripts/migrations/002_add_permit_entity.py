#!/usr/bin/env python3
"""
Migration 002: Add Permit Entity to Neo4j Schema

This migration creates the Permit entity with the following properties:
- id: Unique identifier
- permit_type: Type of permit
- limit: Permit limit value
- unit: Unit of measurement
- expiration_date: When permit expires
- regulatory_authority: Authority that issued the permit

Creates constraints and indexes for optimal performance.
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

class PermitEntityMigration:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the migration with Neo4j connection details."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.migration_name = "002_add_permit_entity"
        
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
                    description: 'Add Permit entity with properties and constraints'
                })
                """,
                migration_name=self.migration_name
            )
    
    def drop_existing_permit_id_index(self):
        """Drop existing permit_id index if it exists."""
        with self.driver.session() as session:
            try:
                # Check if the index exists
                result = session.run(
                    "SHOW INDEXES YIELD name, labelsOrTypes, properties WHERE name = 'permit_id'"
                )
                if result.single():
                    session.run("DROP INDEX permit_id IF EXISTS")
                    logger.info("Dropped existing permit_id index")
            except Exception as e:
                # Index might not exist, which is fine
                logger.info(f"No existing permit_id index to drop: {e}")
            
    def create_permit_constraints(self):
        """Create constraints for Permit entity."""
        with self.driver.session() as session:
            try:
                # Drop existing index first if it exists
                self.drop_existing_permit_id_index()
                
                # Create unique constraint on Permit.id (Community Edition compatible)
                session.run("CREATE CONSTRAINT permit_id_unique IF NOT EXISTS FOR (p:Permit) REQUIRE p.id IS UNIQUE")
                logger.info("Created unique constraint on Permit.id")
                
                # Note: NOT NULL constraints are Enterprise-only features and have been removed
                # The application layer should validate that required fields are present
                
            except Exception as e:
                logger.error(f"Error creating Permit constraints: {e}")
                raise
                
    def create_permit_indexes(self):
        """Create indexes for Permit entity."""
        with self.driver.session() as session:
            try:
                # Index on permit_type for filtering
                session.run("CREATE INDEX permit_type_index IF NOT EXISTS FOR (p:Permit) ON (p.permit_type)")
                logger.info("Created index on Permit.permit_type")
                
                # Index on regulatory_authority for filtering by authority
                session.run("CREATE INDEX permit_authority_index IF NOT EXISTS FOR (p:Permit) ON (p.regulatory_authority)")
                logger.info("Created index on Permit.regulatory_authority")
                
                # Index on expiration_date for temporal queries
                session.run("CREATE INDEX permit_expiration_date_index IF NOT EXISTS FOR (p:Permit) ON (p.expiration_date)")
                logger.info("Created index on Permit.expiration_date")
                
                # Index on unit for filtering by measurement unit
                session.run("CREATE INDEX permit_unit_index IF NOT EXISTS FOR (p:Permit) ON (p.unit)")
                logger.info("Created index on Permit.unit")
                
            except Exception as e:
                logger.error(f"Error creating Permit indexes: {e}")
                raise
                
    def verify_permit_schema(self):
        """Verify that Permit schema was created correctly."""
        with self.driver.session() as session:
            try:
                # Check constraints
                constraints_result = session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties WHERE 'Permit' IN labelsOrTypes")
                constraints = list(constraints_result)
                logger.info(f"Found {len(constraints)} constraints for Permit entity")
                
                # Check indexes
                indexes_result = session.run("SHOW INDEXES YIELD name, labelsOrTypes, properties WHERE 'Permit' IN labelsOrTypes")
                indexes = list(indexes_result)
                logger.info(f"Found {len(indexes)} indexes for Permit entity")
                
                # Updated to expect only 1 constraint (unique) and 5 indexes for Community Edition 
                # (4 created by this migration + 1 for the unique constraint)
                return len(constraints) >= 1 and len(indexes) >= 5
                
            except Exception as e:
                logger.error(f"Error verifying Permit schema: {e}")
                return False
                
    def run(self):
        """Execute the migration."""
        try:
            # Check if migration already applied
            if self.check_migration_applied():
                logger.info(f"Migration {self.migration_name} already applied. Skipping.")
                return True
                
            logger.info(f"Starting migration: {self.migration_name}")
            
            # Create constraints
            logger.info("Creating Permit constraints...")
            self.create_permit_constraints()
            
            # Create indexes
            logger.info("Creating Permit indexes...")
            self.create_permit_indexes()
            
            # Verify schema
            if self.verify_permit_schema():
                logger.info("Permit schema verification successful")
            else:
                logger.warning("Permit schema verification failed")
                
            # Mark migration as applied
            self.mark_migration_applied()
            logger.info(f"Migration {self.migration_name} completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration {self.migration_name} failed: {e}")
            raise


def main():
    """Main function to run the Permit entity migration."""
    # Get Neo4j connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    logger.info("Starting Permit entity migration...")
    
    migration = PermitEntityMigration(uri, username, password)
    
    try:
        success = migration.run()
        if success:
            logger.info("Permit entity migration completed successfully")
        else:
            logger.error("Permit entity migration failed")
            return 1
            
    except Exception as e:
        logger.error(f"Migration failed with error: {e}")
        return 1
        
    finally:
        migration.close()
        
    return 0


if __name__ == "__main__":
    exit(main())