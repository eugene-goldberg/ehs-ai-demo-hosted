#!/usr/bin/env python3
"""
Create Fulltext Indexes for EHS Data in Neo4j

This script creates fulltext indexes for efficient text search across EHS entities
including facility names, permit descriptions, equipment models, and document content.

Features:
- Fulltext indexes for facility names and descriptions
- Fulltext indexes for permit descriptions and regulatory text
- Fulltext indexes for equipment models and specifications
- Fulltext indexes for document content and metadata
- Custom analyzer configuration for EHS terminology
- Idempotent operations with existence checks
"""

import logging
import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FulltextIndexCreator:
    """Creates and manages fulltext indexes for EHS data search."""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize with Neo4j connection details."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def check_index_exists(self, index_name: str) -> bool:
        """Check if a fulltext index already exists."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    "SHOW INDEXES YIELD name WHERE name = $index_name",
                    index_name=index_name
                )
                return len(list(result)) > 0
            except Exception as e:
                logger.warning(f"Could not check index existence for {index_name}: {e}")
                return False
                
    def create_facility_fulltext_indexes(self) -> List[str]:
        """Create fulltext indexes for Facility entities."""
        created_indexes = []
        
        with self.driver.session() as session:
            try:
                # Facility name fulltext index
                index_name = "ehs_facility_name_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (f:Facility) 
                        ON EACH [f.name, f.address, f.city]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
                # Facility description and details
                index_name = "ehs_facility_description_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (f:Facility) 
                        ON EACH [f.description, f.facility_type, f.operational_status]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating Facility fulltext indexes: {e}")
                raise
                
        return created_indexes
        
    def create_permit_fulltext_indexes(self) -> List[str]:
        """Create fulltext indexes for Permit entities."""
        created_indexes = []
        
        with self.driver.session() as session:
            try:
                # Permit descriptions and types
                index_name = "ehs_permit_description_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (p:Permit) 
                        ON EACH [p.permit_type, p.description, p.regulatory_authority]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
                # Permit compliance and regulatory text
                index_name = "ehs_permit_compliance_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (p:Permit) 
                        ON EACH [p.compliance_requirements, p.conditions, p.limitations]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating Permit fulltext indexes: {e}")
                raise
                
        return created_indexes
        
    def create_equipment_fulltext_indexes(self) -> List[str]:
        """Create fulltext indexes for Equipment entities."""
        created_indexes = []
        
        with self.driver.session() as session:
            try:
                # Equipment models and types
                index_name = "ehs_equipment_model_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (e:Equipment) 
                        ON EACH [e.model, e.equipment_type, e.manufacturer]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
                # Equipment specifications and descriptions
                index_name = "ehs_equipment_specs_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (e:Equipment) 
                        ON EACH [e.specifications, e.description, e.serial_number]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating Equipment fulltext indexes: {e}")
                raise
                
        return created_indexes
        
    def create_document_fulltext_indexes(self) -> List[str]:
        """Create fulltext indexes for Document entities."""
        created_indexes = []
        
        with self.driver.session() as session:
            try:
                # Document content and title
                index_name = "ehs_document_content_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (d:Document) 
                        ON EACH [d.title, d.content, d.summary]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
                # Document metadata and classification
                index_name = "ehs_document_metadata_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (d:Document) 
                        ON EACH [d.document_type, d.source, d.author, d.keywords]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
                # DocumentChunk content for detailed search
                index_name = "ehs_chunk_content_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (c:DocumentChunk) 
                        ON EACH [c.content, c.section_title, c.chunk_type]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating Document fulltext indexes: {e}")
                raise
                
        return created_indexes
        
    def create_ehs_specialized_indexes(self) -> List[str]:
        """Create specialized fulltext indexes for EHS domain terms."""
        created_indexes = []
        
        with self.driver.session() as session:
            try:
                # Multi-entity compliance and regulatory terms
                index_name = "ehs_compliance_terms_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (n:Permit|Document|Facility) 
                        ON EACH [n.regulatory_authority, n.compliance_requirements, n.regulatory_references]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
                # Environmental and safety terms across entities
                index_name = "ehs_environmental_terms_fulltext_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                        FOR (n:Equipment|Document|Permit) 
                        ON EACH [n.environmental_impact, n.safety_features, n.emission_factors]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'standard-folding',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created fulltext index: {index_name}")
                else:
                    logger.info(f"Fulltext index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating specialized EHS fulltext indexes: {e}")
                raise
                
        return created_indexes
        
    def verify_fulltext_indexes(self, created_indexes: List[str]) -> bool:
        """Verify that fulltext indexes were created successfully."""
        with self.driver.session() as session:
            try:
                # Get all fulltext indexes
                result = session.run("""
                    SHOW INDEXES 
                    YIELD name, type, entityType, labelsOrTypes, properties 
                    WHERE type = 'FULLTEXT'
                """)
                
                fulltext_indexes = list(result)
                logger.info(f"Found {len(fulltext_indexes)} fulltext indexes in database")
                
                # Check if our created indexes are present
                index_names = [idx["name"] for idx in fulltext_indexes]
                missing_indexes = [idx for idx in created_indexes if idx not in index_names]
                
                if missing_indexes:
                    logger.warning(f"Missing fulltext indexes: {missing_indexes}")
                    return False
                    
                logger.info("All fulltext indexes verified successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error verifying fulltext indexes: {e}")
                return False
                
    def create_all_fulltext_indexes(self) -> Dict[str, Any]:
        """Create all fulltext indexes and return summary."""
        logger.info("Starting fulltext index creation for EHS Analytics...")
        
        summary = {
            "total_created": 0,
            "facility_indexes": [],
            "permit_indexes": [],
            "equipment_indexes": [],
            "document_indexes": [],
            "specialized_indexes": [],
            "errors": []
        }
        
        try:
            # Create Facility fulltext indexes
            logger.info("Creating Facility fulltext indexes...")
            facility_indexes = self.create_facility_fulltext_indexes()
            summary["facility_indexes"] = facility_indexes
            
            # Create Permit fulltext indexes
            logger.info("Creating Permit fulltext indexes...")
            permit_indexes = self.create_permit_fulltext_indexes()
            summary["permit_indexes"] = permit_indexes
            
            # Create Equipment fulltext indexes
            logger.info("Creating Equipment fulltext indexes...")
            equipment_indexes = self.create_equipment_fulltext_indexes()
            summary["equipment_indexes"] = equipment_indexes
            
            # Create Document fulltext indexes
            logger.info("Creating Document fulltext indexes...")
            document_indexes = self.create_document_fulltext_indexes()
            summary["document_indexes"] = document_indexes
            
            # Create specialized EHS indexes
            logger.info("Creating specialized EHS fulltext indexes...")
            specialized_indexes = self.create_ehs_specialized_indexes()
            summary["specialized_indexes"] = specialized_indexes
            
            # Calculate total
            all_created = (facility_indexes + permit_indexes + equipment_indexes + 
                          document_indexes + specialized_indexes)
            summary["total_created"] = len(all_created)
            
            # Verify all indexes
            if self.verify_fulltext_indexes(all_created):
                logger.info("Fulltext index creation completed successfully")
                summary["verification"] = "success"
            else:
                logger.warning("Fulltext index verification failed")
                summary["verification"] = "failed"
                
        except Exception as e:
            error_msg = f"Fulltext index creation failed: {e}"
            logger.error(error_msg)
            summary["errors"].append(error_msg)
            
        return summary


def main():
    """Main function to create fulltext indexes."""
    # Get Neo4j connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    logger.info("Starting EHS fulltext index creation...")
    
    creator = FulltextIndexCreator(uri, username, password)
    
    try:
        summary = creator.create_all_fulltext_indexes()
        
        # Print summary
        logger.info("=== Fulltext Index Creation Summary ===")
        logger.info(f"Total indexes created: {summary['total_created']}")
        logger.info(f"Facility indexes: {len(summary['facility_indexes'])}")
        logger.info(f"Permit indexes: {len(summary['permit_indexes'])}")
        logger.info(f"Equipment indexes: {len(summary['equipment_indexes'])}")
        logger.info(f"Document indexes: {len(summary['document_indexes'])}")
        logger.info(f"Specialized indexes: {len(summary['specialized_indexes'])}")
        
        if summary.get("errors"):
            logger.error(f"Errors encountered: {summary['errors']}")
            return 1
            
        logger.info("Fulltext index creation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Fulltext index creation failed: {e}")
        return 1
        
    finally:
        creator.close()


if __name__ == "__main__":
    exit(main())