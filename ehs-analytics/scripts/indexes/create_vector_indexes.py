#!/usr/bin/env python3
"""
Create Vector Indexes for EHS Data in Neo4j

This script creates vector indexes for Document and DocumentChunk nodes with embeddings
to support semantic search and RAG operations in the EHS Analytics system.

Features:
- Vector indexes for Document nodes with embeddings
- Vector indexes for DocumentChunk nodes
- Configuration for OpenAI embedding dimension (1536)
- Support for cosine and euclidean similarity metrics
- EHS-specific naming conventions
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


class VectorIndexCreator:
    """Creates and manages vector indexes for EHS document embeddings."""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize with Neo4j connection details."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def check_index_exists(self, index_name: str) -> bool:
        """Check if a vector index already exists."""
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
                
    def create_document_vector_indexes(self) -> List[str]:
        """Create vector indexes for Document nodes."""
        created_indexes = []
        
        with self.driver.session() as session:
            try:
                # Primary document content vector index
                index_name = "ehs_document_content_vector_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (d:Document) 
                        ON (d.content_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dimension},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created vector index: {index_name}")
                else:
                    logger.info(f"Vector index already exists: {index_name}")
                    
                # Document title vector index for title-based search
                index_name = "ehs_document_title_vector_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (d:Document) 
                        ON (d.title_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dimension},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created vector index: {index_name}")
                else:
                    logger.info(f"Vector index already exists: {index_name}")
                    
                # Document summary vector index for high-level search
                index_name = "ehs_document_summary_vector_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (d:Document) 
                        ON (d.summary_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dimension},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created vector index: {index_name}")
                else:
                    logger.info(f"Vector index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating Document vector indexes: {e}")
                raise
                
        return created_indexes
        
    def create_document_chunk_vector_indexes(self) -> List[str]:
        """Create vector indexes for DocumentChunk nodes."""
        created_indexes = []
        
        with self.driver.session() as session:
            try:
                # Primary chunk content vector index
                index_name = "ehs_chunk_content_vector_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (c:DocumentChunk) 
                        ON (c.content_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dimension},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created vector index: {index_name}")
                else:
                    logger.info(f"Vector index already exists: {index_name}")
                    
                # Euclidean distance index for alternative similarity
                index_name = "ehs_chunk_euclidean_vector_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (c:DocumentChunk) 
                        ON (c.content_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dimension},
                                `vector.similarity_function`: 'euclidean'
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created vector index: {index_name}")
                else:
                    logger.info(f"Vector index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating DocumentChunk vector indexes: {e}")
                raise
                
        return created_indexes
        
    def create_ehs_entity_vector_indexes(self) -> List[str]:
        """Create vector indexes for EHS-specific entities."""
        created_indexes = []
        
        with self.driver.session() as session:
            try:
                # Equipment description embeddings
                index_name = "ehs_equipment_desc_vector_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (e:Equipment) 
                        ON (e.description_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dimension},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created vector index: {index_name}")
                else:
                    logger.info(f"Vector index already exists: {index_name}")
                    
                # Permit description embeddings
                index_name = "ehs_permit_desc_vector_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (p:Permit) 
                        ON (p.description_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dimension},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created vector index: {index_name}")
                else:
                    logger.info(f"Vector index already exists: {index_name}")
                    
                # Facility description embeddings
                index_name = "ehs_facility_desc_vector_idx"
                if not self.check_index_exists(index_name):
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (f:Facility) 
                        ON (f.description_embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {self.embedding_dimension},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    created_indexes.append(index_name)
                    logger.info(f"Created vector index: {index_name}")
                else:
                    logger.info(f"Vector index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating EHS entity vector indexes: {e}")
                raise
                
        return created_indexes
        
    def verify_vector_indexes(self, created_indexes: List[str]) -> bool:
        """Verify that vector indexes were created successfully."""
        with self.driver.session() as session:
            try:
                # Get all vector indexes
                result = session.run("""
                    SHOW INDEXES 
                    YIELD name, type, entityType, labelsOrTypes, properties 
                    WHERE type = 'VECTOR'
                """)
                
                vector_indexes = list(result)
                logger.info(f"Found {len(vector_indexes)} vector indexes in database")
                
                # Check if our created indexes are present
                index_names = [idx["name"] for idx in vector_indexes]
                missing_indexes = [idx for idx in created_indexes if idx not in index_names]
                
                if missing_indexes:
                    logger.warning(f"Missing vector indexes: {missing_indexes}")
                    return False
                    
                logger.info("All vector indexes verified successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error verifying vector indexes: {e}")
                return False
                
    def create_all_vector_indexes(self) -> Dict[str, Any]:
        """Create all vector indexes and return summary."""
        logger.info("Starting vector index creation for EHS Analytics...")
        
        summary = {
            "total_created": 0,
            "document_indexes": [],
            "chunk_indexes": [],
            "entity_indexes": [],
            "errors": []
        }
        
        try:
            # Create Document vector indexes
            logger.info("Creating Document vector indexes...")
            doc_indexes = self.create_document_vector_indexes()
            summary["document_indexes"] = doc_indexes
            
            # Create DocumentChunk vector indexes
            logger.info("Creating DocumentChunk vector indexes...")
            chunk_indexes = self.create_document_chunk_vector_indexes()
            summary["chunk_indexes"] = chunk_indexes
            
            # Create EHS entity vector indexes
            logger.info("Creating EHS entity vector indexes...")
            entity_indexes = self.create_ehs_entity_vector_indexes()
            summary["entity_indexes"] = entity_indexes
            
            # Calculate total
            all_created = doc_indexes + chunk_indexes + entity_indexes
            summary["total_created"] = len(all_created)
            
            # Verify all indexes
            if self.verify_vector_indexes(all_created):
                logger.info("Vector index creation completed successfully")
                summary["verification"] = "success"
            else:
                logger.warning("Vector index verification failed")
                summary["verification"] = "failed"
                
        except Exception as e:
            error_msg = f"Vector index creation failed: {e}"
            logger.error(error_msg)
            summary["errors"].append(error_msg)
            
        return summary


def main():
    """Main function to create vector indexes."""
    # Get Neo4j connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    logger.info("Starting EHS vector index creation...")
    
    creator = VectorIndexCreator(uri, username, password)
    
    try:
        summary = creator.create_all_vector_indexes()
        
        # Print summary
        logger.info("=== Vector Index Creation Summary ===")
        logger.info(f"Total indexes created: {summary['total_created']}")
        logger.info(f"Document indexes: {len(summary['document_indexes'])}")
        logger.info(f"Chunk indexes: {len(summary['chunk_indexes'])}")
        logger.info(f"Entity indexes: {len(summary['entity_indexes'])}")
        
        if summary.get("errors"):
            logger.error(f"Errors encountered: {summary['errors']}")
            return 1
            
        logger.info("Vector index creation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Vector index creation failed: {e}")
        return 1
        
    finally:
        creator.close()


if __name__ == "__main__":
    exit(main())