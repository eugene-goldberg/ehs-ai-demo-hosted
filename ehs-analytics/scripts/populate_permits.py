#!/usr/bin/env python3
"""
Permit Population Script for EHS Analytics

This script populates the Neo4j database with realistic permit data for
Apex Manufacturing - Plant A. It creates Permit entities and establishes
PERMITS relationships with the facility.

This script is idempotent and safe to run multiple times.

Usage:
    python3 scripts/populate_permits.py

Environment Variables Required:
    NEO4J_URI (default: bolt://localhost:7687)
    NEO4J_USERNAME (default: neo4j)
    NEO4J_PASSWORD (required)
"""

import logging
import sys
import os
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
        logging.FileHandler(f'populate_permits_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class PermitPopulator:
    """Permit population class for Neo4j database."""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the permit populator with Neo4j connection details."""
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
    
    def get_permit_data(self) -> List[Dict[str, Any]]:
        """Return realistic permit data for Apex Manufacturing - Plant A."""
        return [
            # Air Emissions Permits
            {
                "id": "AIR-2024-001",
                "permit_type": "Air Quality",
                "limit": 250.0,
                "unit": "tons/year",
                "expiration_date": "2025-12-31",
                "regulatory_authority": "State Environmental Protection Agency"
            },
            {
                "id": "AIR-2024-002", 
                "permit_type": "Air Quality",
                "limit": 50.0,
                "unit": "lbs/hour",
                "expiration_date": "2026-06-30",
                "regulatory_authority": "State Environmental Protection Agency"
            },
            {
                "id": "AIR-2023-003",
                "permit_type": "Volatile Organic Compounds",
                "limit": 15.0,
                "unit": "tons/year",
                "expiration_date": "2025-03-15",
                "regulatory_authority": "EPA Region 5"
            },
            # Water Discharge Permits
            {
                "id": "WTR-2024-001",
                "permit_type": "Water Discharge",
                "limit": 500000.0,
                "unit": "gallons/day",
                "expiration_date": "2027-09-30",
                "regulatory_authority": "State Water Quality Board"
            },
            {
                "id": "WTR-2023-002",
                "permit_type": "Stormwater Discharge",
                "limit": 100.0,
                "unit": "mg/L BOD",
                "expiration_date": "2025-10-15",
                "regulatory_authority": "State Water Quality Board"
            },
            {
                "id": "WTR-2024-003",
                "permit_type": "Industrial Wastewater",
                "limit": 200.0,
                "unit": "mg/L TSS",
                "expiration_date": "2026-12-31",
                "regulatory_authority": "Local Water Authority"
            },
            # Waste Generation Permits
            {
                "id": "WST-2024-001",
                "permit_type": "Hazardous Waste Generator",
                "limit": 2700.0,
                "unit": "kg/month",
                "expiration_date": "2025-08-31",
                "regulatory_authority": "EPA Region 5"
            },
            {
                "id": "WST-2023-002",
                "permit_type": "Solid Waste Disposal",
                "limit": 50.0,
                "unit": "tons/month",
                "expiration_date": "2025-12-15",
                "regulatory_authority": "State Waste Management Division"
            },
            {
                "id": "WST-2024-003",
                "permit_type": "Chemical Storage",
                "limit": 10000.0,
                "unit": "gallons",
                "expiration_date": "2026-07-31",
                "regulatory_authority": "State Fire Marshal"
            },
            {
                "id": "WST-2023-004",
                "permit_type": "Underground Storage Tank",
                "limit": 15000.0,
                "unit": "gallons",
                "expiration_date": "2025-05-20",
                "regulatory_authority": "State Environmental Protection Agency"
            }
        ]
    
    def permit_exists(self, permit_id: str) -> bool:
        """Check if permit with given ID already exists."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    "MATCH (p:Permit {id: $permit_id}) RETURN count(p) as count",
                    permit_id=permit_id
                )
                return result.single()["count"] > 0
            except Exception as e:
                logger.error(f"Error checking permit existence: {e}")
                return False
    
    def create_permit(self, permit_data: Dict[str, Any]) -> bool:
        """Create a single permit entity with its relationship to facility."""
        with self.driver.session() as session:
            try:
                # Check if permit already exists
                if self.permit_exists(permit_data["id"]):
                    logger.info(f"Permit {permit_data['id']} already exists. Skipping.")
                    return True
                
                # Create permit and relationship in a single transaction
                query = """
                MATCH (f:Facility {name: $facility_name})
                CREATE (p:Permit {
                    id: $id,
                    permit_type: $permit_type,
                    limit: $limit,
                    unit: $unit,
                    expiration_date: date($expiration_date),
                    regulatory_authority: $regulatory_authority,
                    created_at: datetime(),
                    updated_at: datetime()
                })
                CREATE (f)-[:PERMITS]->(p)
                RETURN p.id as permit_id
                """
                
                result = session.run(
                    query,
                    facility_name=self.facility_name,
                    id=permit_data["id"],
                    permit_type=permit_data["permit_type"],
                    limit=permit_data["limit"],
                    unit=permit_data["unit"],
                    expiration_date=permit_data["expiration_date"],
                    regulatory_authority=permit_data["regulatory_authority"]
                )
                
                record = result.single()
                if record:
                    logger.info(f"Created permit: {record['permit_id']}")
                    return True
                else:
                    logger.error(f"Failed to create permit: {permit_data['id']}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error creating permit {permit_data['id']}: {e}")
                return False
    
    def verify_permit_creation(self) -> Dict[str, Any]:
        """Verify that permits were created correctly."""
        with self.driver.session() as session:
            try:
                # Count total permits
                total_result = session.run("MATCH (p:Permit) RETURN count(p) as total")
                total_permits = total_result.single()["total"]
                
                # Count permits with PERMITS relationships
                relationship_result = session.run(
                    """
                    MATCH (f:Facility {name: $facility_name})-[:PERMITS]->(p:Permit)
                    RETURN count(p) as with_relationships
                    """,
                    facility_name=self.facility_name
                )
                permits_with_relationships = relationship_result.single()["with_relationships"]
                
                # Get permit breakdown by type
                type_result = session.run(
                    """
                    MATCH (f:Facility {name: $facility_name})-[:PERMITS]->(p:Permit)
                    RETURN p.permit_type as type, count(p) as count
                    ORDER BY count DESC
                    """,
                    facility_name=self.facility_name
                )
                permits_by_type = list(type_result)
                
                # Get permit breakdown by regulatory authority
                authority_result = session.run(
                    """
                    MATCH (f:Facility {name: $facility_name})-[:PERMITS]->(p:Permit)
                    RETURN p.regulatory_authority as authority, count(p) as count
                    ORDER BY count DESC
                    """,
                    facility_name=self.facility_name
                )
                permits_by_authority = list(authority_result)
                
                # Check for expiring permits (within 6 months)
                expiring_result = session.run(
                    """
                    MATCH (f:Facility {name: $facility_name})-[:PERMITS]->(p:Permit)
                    WHERE p.expiration_date <= date() + duration({months: 6})
                    RETURN count(p) as expiring_soon
                    """,
                    facility_name=self.facility_name
                )
                expiring_permits = expiring_result.single()["expiring_soon"]
                
                return {
                    "total_permits": total_permits,
                    "permits_with_relationships": permits_with_relationships,
                    "permits_by_type": permits_by_type,
                    "permits_by_authority": permits_by_authority,
                    "expiring_permits": expiring_permits
                }
                
            except Exception as e:
                logger.error(f"Error verifying permit creation: {e}")
                return {}
    
    def populate_permits(self) -> bool:
        """Populate all permit data."""
        try:
            # Check if facility exists
            if not self.check_facility_exists():
                logger.error(f"Cannot populate permits: facility '{self.facility_name}' not found")
                logger.info("Please run facility creation script first or ensure facility exists")
                return False
            
            # Get permit data
            permit_list = self.get_permit_data()
            logger.info(f"Preparing to create {len(permit_list)} permits")
            
            # Create each permit
            success_count = 0
            for permit in permit_list:
                if self.create_permit(permit):
                    success_count += 1
                    
            logger.info(f"Successfully created {success_count}/{len(permit_list)} permits")
            
            # Verify creation
            verification = self.verify_permit_creation()
            if verification:
                logger.info("Permit creation verification:")
                logger.info(f"  Total permits in database: {verification['total_permits']}")
                logger.info(f"  Permits with facility relationships: {verification['permits_with_relationships']}")
                logger.info("  Permits by type:")
                for type_data in verification['permits_by_type']:
                    logger.info(f"    {type_data['type']}: {type_data['count']}")
                logger.info("  Permits by regulatory authority:")
                for authority_data in verification['permits_by_authority']:
                    logger.info(f"    {authority_data['authority']}: {authority_data['count']}")
                logger.info(f"  Permits expiring within 6 months: {verification['expiring_permits']}")
            
            return success_count == len(permit_list)
            
        except Exception as e:
            logger.error(f"Error populating permits: {e}")
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
        
    logger.info("Starting permit population for EHS Analytics...")
    logger.info(f"Neo4j URI: {uri}")
    logger.info(f"Username: {username}")
    logger.info(f"Target facility: Apex Manufacturing - Plant A")
    
    populator = PermitPopulator(uri, username, password)
    
    try:
        # Connect to Neo4j
        populator.connect()
        
        # Populate permits
        success = populator.populate_permits()
        
        if success:
            logger.info("Permit population completed successfully!")
            return 0
        else:
            logger.error("Permit population failed. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Permit population failed: {e}")
        return 1
        
    finally:
        populator.close()


if __name__ == "__main__":
    exit(main())