#!/usr/bin/env python3
"""
Equipment Population Script for EHS Analytics

This script populates the Neo4j database with realistic equipment data for
Apex Manufacturing - Plant A. It creates Equipment entities and establishes
LOCATED_AT relationships with the facility.

This script is idempotent and safe to run multiple times.

Usage:
    python3 scripts/populate_equipment.py

Environment Variables Required:
    NEO4J_URI (default: bolt://localhost:7687)
    NEO4J_USERNAME (default: neo4j)
    NEO4J_PASSWORD (required)
"""

import logging
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv
from datetime import datetime, date
import uuid

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
        logging.FileHandler(f'populate_equipment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class EquipmentPopulator:
    """Equipment population class for Neo4j database."""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the equipment populator with Neo4j connection details."""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.facility_name = "Apex Manufacturing - Plant A"
        
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
    
    def check_facility_exists(self) -> bool:
        """Check if the target facility exists in the database."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    "MATCH (f:Facility {name: $facility_name}) RETURN count(f) as count",
                    facility_name=self.facility_name
                )
                count = result.single()["count"]
                if count > 0:
                    logger.info(f"Found facility: {self.facility_name}")
                    return True
                else:
                    logger.warning(f"Facility not found: {self.facility_name}")
                    return False
            except Exception as e:
                logger.error(f"Error checking facility existence: {e}")
                return False
    
    def get_equipment_data(self) -> List[Dict[str, Any]]:
        """Return realistic equipment data for Apex Manufacturing - Plant A."""
        base_equipment = [
            {
                "equipment_type": "HVAC System",
                "model": "Carrier 50TCQ120",
                "efficiency_rating": 16.5,
                "installation_date": "2019-03-15"
            },
            {
                "equipment_type": "HVAC System",
                "model": "Trane CVG120",
                "efficiency_rating": 15.8,
                "installation_date": "2020-08-22"
            },
            {
                "equipment_type": "Air Compressor",
                "model": "Atlas Copco GA55",
                "efficiency_rating": 92.3,
                "installation_date": "2018-11-10"
            },
            {
                "equipment_type": "Air Compressor", 
                "model": "Ingersoll Rand R75",
                "efficiency_rating": 89.7,
                "installation_date": "2021-05-18"
            },
            {
                "equipment_type": "Steam Boiler",
                "model": "Cleaver-Brooks CB-700",
                "efficiency_rating": 85.2,
                "installation_date": "2017-09-30"
            },
            {
                "equipment_type": "Cooling Tower",
                "model": "Baltimore Aircoil VTI-1500",
                "efficiency_rating": 78.9,
                "installation_date": "2019-06-12"
            },
            {
                "equipment_type": "Cooling Tower",
                "model": "Evapco ATC-2000",
                "efficiency_rating": 81.4,
                "installation_date": "2020-04-25"
            },
            {
                "equipment_type": "CNC Machine",
                "model": "Haas VF-4SS",
                "efficiency_rating": 94.5,
                "installation_date": "2021-10-14"
            },
            {
                "equipment_type": "Injection Molding Machine",
                "model": "Nissei NEX7000",
                "efficiency_rating": 87.8,
                "installation_date": "2020-12-03"
            },
            {
                "equipment_type": "Industrial Furnace",
                "model": "Lindberg MPH2448",
                "efficiency_rating": 73.6,
                "installation_date": "2018-02-28"
            }
        ]
        
        # Generate unique timestamp-based IDs for each equipment item
        equipment_list = []
        for equipment in base_equipment:
            # Generate timestamp-based ID
            equipment_id = f"EQ-{equipment['equipment_type'][:4].upper()}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            # Add the ID to the equipment data
            equipment_with_id = equipment.copy()
            equipment_with_id["id"] = equipment_id
            equipment_list.append(equipment_with_id)
            
            # Add small delay to ensure unique timestamps
            time.sleep(0.001)
        
        return equipment_list
    
    def equipment_exists(self, equipment_id: str) -> bool:
        """Check if equipment with given ID already exists."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    "MATCH (e:Equipment {id: $equipment_id}) RETURN count(e) as count",
                    equipment_id=equipment_id
                )
                return result.single()["count"] > 0
            except Exception as e:
                logger.error(f"Error checking equipment existence: {e}")
                return False
    
    def create_equipment(self, equipment_data: Dict[str, Any]) -> bool:
        """Create a single equipment entity with its relationship to facility."""
        with self.driver.session() as session:
            try:
                # Check if equipment already exists
                if self.equipment_exists(equipment_data["id"]):
                    logger.info(f"Equipment {equipment_data['id']} already exists. Skipping.")
                    return True
                
                # Create equipment and relationship in a single transaction
                query = """
                MATCH (f:Facility {name: $facility_name})
                CREATE (e:Equipment {
                    id: $id,
                    equipment_type: $equipment_type,
                    model: $model,
                    efficiency_rating: $efficiency_rating,
                    installation_date: date($installation_date),
                    created_at: datetime(),
                    updated_at: datetime()
                })
                CREATE (e)-[:LOCATED_AT]->(f)
                RETURN e.id as equipment_id
                """
                
                result = session.run(
                    query,
                    facility_name=self.facility_name,
                    id=equipment_data["id"],
                    equipment_type=equipment_data["equipment_type"],
                    model=equipment_data["model"],
                    efficiency_rating=equipment_data["efficiency_rating"],
                    installation_date=equipment_data["installation_date"]
                )
                
                record = result.single()
                if record:
                    logger.info(f"Created equipment: {record['equipment_id']}")
                    return True
                else:
                    logger.error(f"Failed to create equipment: {equipment_data['id']}")
                    return False
                    
            except Exception as e:
                # Handle constraint violation more gracefully
                if "constraint" in str(e).lower() or "already exists" in str(e).lower():
                    logger.warning(f"Equipment {equipment_data['id']} may already exist due to constraint violation. Continuing...")
                    return True
                else:
                    logger.error(f"Error creating equipment {equipment_data['id']}: {e}")
                    return False
    
    def verify_equipment_creation(self) -> Dict[str, Any]:
        """Verify that equipment was created correctly."""
        with self.driver.session() as session:
            try:
                # Count total equipment
                total_result = session.run("MATCH (e:Equipment) RETURN count(e) as total")
                total_equipment = total_result.single()["total"]
                
                # Count equipment with LOCATED_AT relationships
                relationship_result = session.run(
                    """
                    MATCH (e:Equipment)-[:LOCATED_AT]->(f:Facility {name: $facility_name})
                    RETURN count(e) as with_relationships
                    """,
                    facility_name=self.facility_name
                )
                equipment_with_relationships = relationship_result.single()["with_relationships"]
                
                # Get equipment breakdown by type
                type_result = session.run(
                    """
                    MATCH (e:Equipment)-[:LOCATED_AT]->(f:Facility {name: $facility_name})
                    RETURN e.equipment_type as type, count(e) as count
                    ORDER BY count DESC
                    """,
                    facility_name=self.facility_name
                )
                equipment_by_type = list(type_result)
                
                return {
                    "total_equipment": total_equipment,
                    "equipment_with_relationships": equipment_with_relationships,
                    "equipment_by_type": equipment_by_type
                }
                
            except Exception as e:
                logger.error(f"Error verifying equipment creation: {e}")
                return {}
    
    def populate_equipment(self) -> bool:
        """Populate all equipment data."""
        try:
            # Check if facility exists
            if not self.check_facility_exists():
                logger.error(f"Cannot populate equipment: facility '{self.facility_name}' not found")
                logger.info("Please run facility creation script first or ensure facility exists")
                return False
            
            # Get equipment data
            equipment_list = self.get_equipment_data()
            logger.info(f"Preparing to create {len(equipment_list)} equipment items")
            
            # Create each equipment item
            success_count = 0
            for equipment in equipment_list:
                if self.create_equipment(equipment):
                    success_count += 1
                # Add small delay between creations for timestamp uniqueness
                time.sleep(0.001)
                    
            logger.info(f"Successfully created {success_count}/{len(equipment_list)} equipment items")
            
            # Verify creation
            verification = self.verify_equipment_creation()
            if verification:
                logger.info("Equipment creation verification:")
                logger.info(f"  Total equipment in database: {verification['total_equipment']}")
                logger.info(f"  Equipment with facility relationships: {verification['equipment_with_relationships']}")
                logger.info("  Equipment by type:")
                for type_data in verification['equipment_by_type']:
                    logger.info(f"    {type_data['type']}: {type_data['count']}")
            
            return success_count == len(equipment_list)
            
        except Exception as e:
            logger.error(f"Error populating equipment: {e}")
            return False


def main():
    """Main function."""
    # Get Neo4j connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    if not password:
        logger.error("NEO4J_PASSWORD environment variable is required")
        return 1
        
    logger.info("Starting equipment population for EHS Analytics...")
    logger.info(f"Neo4j URI: {uri}")
    logger.info(f"Username: {username}")
    logger.info(f"Target facility: Apex Manufacturing - Plant A")
    
    populator = EquipmentPopulator(uri, username, password)
    
    try:
        # Connect to Neo4j
        populator.connect()
        
        # Populate equipment
        success = populator.populate_equipment()
        
        if success:
            logger.info("Equipment population completed successfully!")
            return 0
        else:
            logger.error("Equipment population failed. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Equipment population failed: {e}")
        return 1
        
    finally:
        populator.close()


if __name__ == "__main__":
    exit(main())