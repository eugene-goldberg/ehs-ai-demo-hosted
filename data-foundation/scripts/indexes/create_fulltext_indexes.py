#!/usr/bin/env python3
"""
Script to create fulltext indexes in Neo4j for EHS document search.
"""

import os
import sys
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'src'))

from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FulltextIndexManager:
    """Manager for creating and verifying fulltext indexes in Neo4j."""
    
    def __init__(self):
        """Initialize with Neo4j connection."""
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
    def close(self):
        """Close Neo4j driver."""
        if self.driver:
            self.driver.close()
    
    def create_fulltext_index(self, index_name: str, node_labels: List[str], 
                            properties: List[str]) -> bool:
        """
        Create a fulltext index in Neo4j.
        
        Args:
            index_name: Name of the fulltext index
            node_labels: List of node labels to index
            properties: List of properties to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Format labels and properties for query
                labels_str = ", ".join([f"`{label}`" for label in node_labels])
                properties_str = ", ".join([f"`{prop}`" for prop in properties])
                
                # Create fulltext index
                query = f"""
                CREATE FULLTEXT INDEX `{index_name}` IF NOT EXISTS
                FOR (n:{labels_str})
                ON EACH [{properties_str}]
                """
                
                result = session.run(query)
                logger.info(f"Created fulltext index: {index_name} for {labels_str} on {properties_str}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create fulltext index {index_name}: {str(e)}")
            return False
    
    def create_ehs_fulltext_indexes(self) -> Dict[str, bool]:
        """
        Create all EHS-related fulltext indexes.
        
        Returns:
            Dictionary of index names and their creation status
        """
        indexes = {
            "ehs_document_text_search": {
                "node_labels": ["DocumentChunk", "Document"],
                "properties": ["text", "content", "title"]
            },
            "ehs_entity_search": {
                "node_labels": ["Entity", "Facility", "Equipment"],
                "properties": ["name", "description", "type"]
            },
            "ehs_permit_search": {
                "node_labels": ["Permit", "License"],
                "properties": ["permit_number", "description", "requirements"]
            },
            "ehs_utility_search": {
                "node_labels": ["UtilityBill", "EnergyUsage"],
                "properties": ["meter_number", "account_number", "description"]
            },
            "ehs_waste_search": {
                "node_labels": ["WasteManifest", "WasteRecord"],
                "properties": ["manifest_number", "waste_description", "generator_name"]
            }
        }
        
        results = {}
        for index_name, config in indexes.items():
            success = self.create_fulltext_index(
                index_name=index_name,
                node_labels=config["node_labels"],
                properties=config["properties"]
            )
            results[index_name] = success
            
        return results
    
    def verify_fulltext_indexes(self) -> Dict[str, bool]:
        """
        Verify that fulltext indexes exist and are online.
        
        Returns:
            Dictionary of index names and their status
        """
        try:
            with self.driver.session() as session:
                # Get all fulltext indexes
                query = "SHOW INDEXES YIELD name, type, state WHERE type = 'FULLTEXT'"
                result = session.run(query)
                
                indexes = {}
                for record in result:
                    index_name = record["name"]
                    state = record["state"]
                    indexes[index_name] = (state == "ONLINE")
                    logger.info(f"Fulltext index {index_name}: {state}")
                
                return indexes
                
        except Exception as e:
            logger.error(f"Failed to verify fulltext indexes: {str(e)}")
            return {}
    
    def create_sample_fulltext_data(self) -> bool:
        """
        Create sample fulltext data for testing.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Sample documents
                documents = [
                    {
                        "id": "doc_energy_001",
                        "title": "Energy Consumption Report Q1 2024",
                        "content": "Monthly energy consumption for Building A exceeded baseline by 15%. Peak demand occurred during March with 2,500 kWh usage.",
                        "type": "energy_report"
                    },
                    {
                        "id": "doc_permit_001", 
                        "title": "Air Quality Permit Renewal",
                        "content": "Permit AQ-2024-001 for air emissions monitoring requires quarterly reporting and annual calibration of monitoring equipment.",
                        "type": "permit"
                    }
                ]
                
                for doc in documents:
                    query = """
                    MERGE (d:Document {id: $doc_id})
                    SET d.title = $title,
                        d.content = $content,
                        d.type = $doc_type,
                        d.created_at = datetime()
                    """
                    
                    session.run(query, {
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"],
                        "doc_type": doc["type"]
                    })
                
                # Sample entities
                entities = [
                    {"name": "Building A", "type": "facility", "description": "Main office building"},
                    {"name": "HVAC System 001", "type": "equipment", "description": "Primary heating and cooling system"},
                    {"name": "Permit AQ-2024-001", "type": "permit", "description": "Air quality monitoring permit"}
                ]
                
                for entity in entities:
                    query = """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $entity_type,
                        e.description = $description,
                        e.created_at = datetime()
                    """
                    
                    session.run(query, {
                        "name": entity["name"],
                        "entity_type": entity["type"],
                        "description": entity["description"]
                    })
                
                logger.info(f"Created {len(documents)} sample documents and {len(entities)} sample entities")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create sample fulltext data: {str(e)}")
            return False

def main():
    """Main function to create fulltext indexes."""
    logger.info("Starting fulltext index creation process...")
    
    manager = FulltextIndexManager()
    
    try:
        # Create fulltext indexes
        logger.info("Creating EHS fulltext indexes...")
        creation_results = manager.create_ehs_fulltext_indexes()
        
        # Log results
        for index_name, success in creation_results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Fulltext index creation - {index_name}: {status}")
        
        # Verify indexes
        logger.info("Verifying fulltext indexes...")
        verification_results = manager.verify_fulltext_indexes()
        
        for index_name, online in verification_results.items():
            status = "ONLINE" if online else "OFFLINE"
            logger.info(f"Fulltext index verification - {index_name}: {status}")
        
        # Create sample data
        logger.info("Creating sample fulltext data...")
        sample_success = manager.create_sample_fulltext_data()
        
        if sample_success:
            logger.info("Sample fulltext data created successfully")
        else:
            logger.error("Failed to create sample fulltext data")
        
        # Summary
        total_indexes = len(creation_results)
        successful_indexes = sum(creation_results.values())
        online_indexes = sum(verification_results.values())
        
        logger.info("\n" + "="*50)
        logger.info("FULLTEXT INDEX CREATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total indexes: {total_indexes}")
        logger.info(f"Successfully created: {successful_indexes}")
        logger.info(f"Online indexes: {online_indexes}")
        logger.info(f"Sample data created: {'Yes' if sample_success else 'No'}")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "creation_results": creation_results,
            "verification_results": verification_results,
            "sample_data_created": sample_success
        }
        
        with open("scripts/output/fulltext_index_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Results saved to scripts/output/fulltext_index_results.json")
        
    except Exception as e:
        logger.error(f"Fulltext index creation failed: {str(e)}")
        sys.exit(1)
    
    finally:
        manager.close()

if __name__ == "__main__":
    main()
