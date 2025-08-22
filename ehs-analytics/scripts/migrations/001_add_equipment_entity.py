#!/usr/bin/env python3
"""
Migration 001: Add Equipment Entity to Neo4j Schema

This migration creates the Equipment entity with the following properties:
- id: Unique identifier
- equipment_type: Type of equipment
- model: Equipment model
- efficiency_rating: Efficiency rating value
- installation_date: Date when equipment was installed

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

class EquipmentEntityMigration:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the migration with Neo4j connection details."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.migration_name = "001_add_equipment_entity"
        
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
                    description: 'Add Equipment entity with properties and constraints'
                })
                """,
                migration_name=self.migration_name
            )
            
    def create_equipment_constraints(self):
        """Create constraints for Equipment entity."""
        with self.driver.session() as session:
            try:
                # Create unique constraint on Equipment.id (Community Edition compatible)
                session.run("CREATE CONSTRAINT equipment_id_unique IF NOT EXISTS FOR (e:Equipment) REQUIRE e.id IS UNIQUE")
                logger.info("Created unique constraint on Equipment.id")
                
                # Note: NOT NULL constraints are Enterprise-only features and have been removed
                # The application layer should validate that required fields are present
                
            except Exception as e:
                logger.error(f"Error creating Equipment constraints: {e}")
                raise
                
    def create_equipment_indexes(self):
        """Create indexes for Equipment entity."""
        with self.driver.session() as session:
            try:
                # Index on equipment_type for filtering
                session.run("CREATE INDEX equipment_type_index IF NOT EXISTS FOR (e:Equipment) ON (e.equipment_type)")
                logger.info("Created index on Equipment.equipment_type")
                
                # Index on model for searching
                session.run("CREATE INDEX equipment_model_index IF NOT EXISTS FOR (e:Equipment) ON (e.model)")
                logger.info("Created index on Equipment.model")
                
                # Index on installation_date for temporal queries
                session.run("CREATE INDEX equipment_installation_date_index IF NOT EXISTS FOR (e:Equipment) ON (e.installation_date)")
                logger.info("Created index on Equipment.installation_date")
                
            except Exception as e:
                logger.error(f"Error creating Equipment indexes: {e}")
                raise
                
    def verify_equipment_schema(self):
        """Verify that Equipment schema was created correctly."""
        with self.driver.session() as session:
            try:
                # Check constraints
                constraints_result = session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties WHERE 'Equipment' IN labelsOrTypes")
                constraints = list(constraints_result)
                logger.info(f"Found {len(constraints)} constraints for Equipment entity")
                
                # Check indexes
                indexes_result = session.run("SHOW INDEXES YIELD name, labelsOrTypes, properties WHERE 'Equipment' IN labelsOrTypes")
                indexes = list(indexes_result)
                logger.info(f"Found {len(indexes)} indexes for Equipment entity")
                
                # Updated to expect only 1 constraint (unique) and 3 indexes for Community Edition
                return len(constraints) >= 1 and len(indexes) >= 3
                
            except Exception as e:
                logger.error(f"Error verifying Equipment schema: {e}")
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
            logger.info("Creating Equipment constraints...")
            self.create_equipment_constraints()
            
            # Create indexes
            logger.info("Creating Equipment indexes...")
            self.create_equipment_indexes()
            
            # Verify schema
            if self.verify_equipment_schema():
                logger.info("Equipment schema verification successful")
            else:
                logger.warning("Equipment schema verification failed")
                
            # Mark migration as applied
            self.mark_migration_applied()
            logger.info(f"Migration {self.migration_name} completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration {self.migration_name} failed: {e}")
            raise


def main():
    """Main function to run the Equipment entity migration."""
    # Get Neo4j connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    logger.info("Starting Equipment entity migration...")
    
    migration = EquipmentEntityMigration(uri, username, password)
    
    try:
        success = migration.run()
        if success:
            logger.info("Equipment entity migration completed successfully")
        else:
            logger.error("Equipment entity migration failed")
            return 1
            
    except Exception as e:
        logger.error(f"Migration failed with error: {e}")
        return 1
        
    finally:
        migration.close()
        
    return 0


if __name__ == "__main__":
    exit(main())