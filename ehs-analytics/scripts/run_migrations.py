#!/usr/bin/env python3
"""
Main Migration Runner for EHS Analytics Neo4j Schema

This script executes all database migrations in the correct order.
Migrations are idempotent and safe to run multiple times.

Usage:
    python3 scripts/run_migrations.py
    
Environment Variables Required:
    NEO4J_URI (default: bolt://localhost:7687)
    NEO4J_USERNAME (default: neo4j)
    NEO4J_PASSWORD (required, default: EhsAI2024!)
"""

import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv
import importlib.util
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'migrations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class MigrationRunner:
    """Main migration runner class."""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the migration runner with Neo4j connection details."""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.migrations_dir = Path(__file__).parent / "migrations"
        
    def connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
            
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
            
    def setup_migration_tracking(self):
        """Create Migration node type to track applied migrations."""
        with self.driver.session() as session:
            try:
                # Create constraint for Migration tracking
                session.run("CREATE CONSTRAINT migration_name_unique IF NOT EXISTS FOR (m:Migration) REQUIRE m.name IS UNIQUE")
                logger.info("Migration tracking setup completed")
            except Exception as e:
                logger.error(f"Failed to setup migration tracking: {e}")
                raise
                
    def get_migration_files(self) -> List[Path]:
        """Get all migration files in order."""
        migration_files = []
        for file_path in self.migrations_dir.glob("*.py"):
            if file_path.name.startswith(("001_", "002_", "003_")) and file_path.name != "__init__.py":
                migration_files.append(file_path)
                
        # Sort by filename to ensure correct order
        migration_files.sort()
        logger.info(f"Found {len(migration_files)} migration files")
        return migration_files
        
    def load_migration_module(self, migration_file: Path):
        """Dynamically load a migration module."""
        try:
            module_name = migration_file.stem
            spec = importlib.util.spec_from_file_location(module_name, migration_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.error(f"Failed to load migration module {migration_file}: {e}")
            raise
            
    def run_migration(self, migration_file: Path) -> bool:
        """Run a single migration."""
        try:
            logger.info(f"Loading migration: {migration_file.name}")
            migration_module = self.load_migration_module(migration_file)
            
            # Call the main function of the migration
            result = migration_module.main()
            
            if result == 0:
                logger.info(f"Migration {migration_file.name} completed successfully")
                return True
            else:
                logger.error(f"Migration {migration_file.name} failed with exit code {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error running migration {migration_file.name}: {e}")
            return False
            
    def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations."""
        with self.driver.session() as session:
            try:
                result = session.run("MATCH (m:Migration) RETURN m.name as name ORDER BY m.applied_at")
                return [record["name"] for record in result]
            except Exception as e:
                logger.warning(f"Could not retrieve applied migrations: {e}")
                return []
                
    def run_all_migrations(self) -> bool:
        """Run all pending migrations."""
        try:
            # Setup migration tracking
            self.setup_migration_tracking()
            
            # Get applied migrations
            applied_migrations = self.get_applied_migrations()
            logger.info(f"Previously applied migrations: {applied_migrations}")
            
            # Get all migration files
            migration_files = self.get_migration_files()
            
            if not migration_files:
                logger.info("No migration files found")
                return True
                
            # Run each migration
            success_count = 0
            for migration_file in migration_files:
                migration_name = migration_file.stem
                
                # Skip if already applied
                if migration_name in applied_migrations:
                    logger.info(f"Skipping already applied migration: {migration_name}")
                    success_count += 1
                    continue
                    
                logger.info(f"Running migration: {migration_name}")
                if self.run_migration(migration_file):
                    success_count += 1
                else:
                    logger.error(f"Migration {migration_name} failed. Stopping.")
                    break
                    
            logger.info(f"Completed {success_count}/{len(migration_files)} migrations")
            return success_count == len(migration_files)
            
        except Exception as e:
            logger.error(f"Error running migrations: {e}")
            return False
            
    def status(self):
        """Show migration status."""
        try:
            # Get applied migrations
            applied_migrations = self.get_applied_migrations()
            
            # Get all migration files
            migration_files = self.get_migration_files()
            
            print("\n" + "="*60)
            print("MIGRATION STATUS")
            print("="*60)
            
            for migration_file in migration_files:
                migration_name = migration_file.stem
                status = "✓ APPLIED" if migration_name in applied_migrations else "✗ PENDING"
                print(f"{migration_name:<30} {status}")
                
            print(f"\nApplied: {len(applied_migrations)}/{len(migration_files)}")
            print("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")


def main():
    """Main function."""
    # Get Neo4j connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    if not password:
        logger.error("NEO4J_PASSWORD environment variable is required")
        return 1
        
    logger.info("Starting EHS Analytics database migrations...")
    logger.info(f"Neo4j URI: {uri}")
    logger.info(f"Username: {username}")
    
    runner = MigrationRunner(uri, username, password)
    
    try:
        # Connect to Neo4j
        runner.connect()
        
        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == "status":
            runner.status()
            return 0
            
        # Run all migrations
        success = runner.run_all_migrations()
        
        # Show final status
        runner.status()
        
        if success:
            logger.info("All migrations completed successfully!")
            return 0
        else:
            logger.error("Some migrations failed. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Migration runner failed: {e}")
        return 1
        
    finally:
        runner.close()


if __name__ == "__main__":
    exit(main())