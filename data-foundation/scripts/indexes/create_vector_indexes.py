#!/usr/bin/env python3
"""
Script to create vector indexes in Neo4j for EHS document search.
"""

import os
import sys
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'src'))

from llama_index.core import Document
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorIndexManager:
    """Manager for creating and verifying vector indexes in Neo4j."""
    
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
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
        
    def close(self):
        """Close Neo4j driver."""
        if self.driver:
            self.driver.close()
    
    def create_vector_index(self, index_name: str, node_label: str, property_name: str, 
                          embedding_dimension: int = 1536) -> bool:
        """
        Create a vector index in Neo4j.
        
        Args:
            index_name: Name of the vector index
            node_label: Neo4j node label
            property_name: Property to index
            embedding_dimension: Dimension of the embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Create vector index
                query = f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
                FOR (n:`{node_label}`)
                ON (n.`{property_name}`)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """
                
                result = session.run(query)
                logger.info(f"Created vector index: {index_name} for {node_label}.{property_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create vector index {index_name}: {str(e)}")
            return False
    
    def create_ehs_vector_indexes(self) -> Dict[str, bool]:
        """
        Create all EHS-related vector indexes.
        
        Returns:
            Dictionary of index names and their creation status
        """
        indexes = {
            "ehs_documents_vector": {
                "node_label": "DocumentChunk",
                "property_name": "embedding",
                "dimension": 1536
            },
            "ehs_entities_vector": {
                "node_label": "Entity",
                "property_name": "embedding",
                "dimension": 1536
            },
            "ehs_summaries_vector": {
                "node_label": "DocumentSummary",
                "property_name": "embedding",
                "dimension": 1536
            }
        }
        
        results = {}
        for index_name, config in indexes.items():
            success = self.create_vector_index(
                index_name=index_name,
                node_label=config["node_label"],
                property_name=config["property_name"],
                embedding_dimension=config["dimension"]
            )
            results[index_name] = success
            
        return results
    
    def verify_vector_indexes(self) -> Dict[str, bool]:
        """
        Verify that vector indexes exist and are online.
        
        Returns:
            Dictionary of index names and their status
        """
        try:
            with self.driver.session() as session:
                # Get all vector indexes
                query = "SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR'"
                result = session.run(query)
                
                indexes = {}
                for record in result:
                    index_name = record["name"]
                    state = record["state"]
                    indexes[index_name] = (state == "ONLINE")
                    logger.info(f"Vector index {index_name}: {state}")
                
                return indexes
                
        except Exception as e:
            logger.error(f"Failed to verify vector indexes: {str(e)}")
            return {}
    
    def create_sample_vector_data(self) -> bool:
        """
        Create sample vector data for testing.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Sample documents for testing
            sample_texts = [
                "Energy consumption report for Building A: 1,250 kWh used in January 2024",
                "Water usage permit #WP-2024-001 expires on December 31, 2024",
                "Air quality monitoring shows PM2.5 levels at 15 μg/m³ in Zone 1",
                "Waste disposal invoice #INV-001: 500 kg hazardous waste removed"
            ]
            
            with self.driver.session() as session:
                for i, text in enumerate(sample_texts):
                    # Generate embedding
                    embedding = self.embedding_model.get_text_embedding(text)
                    
                    # Create sample document chunk
                    query = """
                    MERGE (d:DocumentChunk {id: $chunk_id})
                    SET d.text = $text,
                        d.embedding = $embedding,
                        d.document_type = $doc_type,
                        d.created_at = datetime()
                    """
                    
                    session.run(query, {
                        "chunk_id": f"sample_chunk_{i}",
                        "text": text,
                        "embedding": embedding,
                        "doc_type": "sample"
                    })
                
                logger.info(f"Created {len(sample_texts)} sample vector documents")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create sample vector data: {str(e)}")
            return False

def main():
    """Main function to create vector indexes."""
    logger.info("Starting vector index creation process...")
    
    manager = VectorIndexManager()
    
    try:
        # Create vector indexes
        logger.info("Creating EHS vector indexes...")
        creation_results = manager.create_ehs_vector_indexes()
        
        # Log results
        for index_name, success in creation_results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Vector index creation - {index_name}: {status}")
        
        # Verify indexes
        logger.info("Verifying vector indexes...")
        verification_results = manager.verify_vector_indexes()
        
        for index_name, online in verification_results.items():
            status = "ONLINE" if online else "OFFLINE"
            logger.info(f"Vector index verification - {index_name}: {status}")
        
        # Create sample data
        logger.info("Creating sample vector data...")
        sample_success = manager.create_sample_vector_data()
        
        if sample_success:
            logger.info("Sample vector data created successfully")
        else:
            logger.error("Failed to create sample vector data")
        
        # Summary
        total_indexes = len(creation_results)
        successful_indexes = sum(creation_results.values())
        online_indexes = sum(verification_results.values())
        
        logger.info("\n" + "="*50)
        logger.info("VECTOR INDEX CREATION SUMMARY")
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
        
        with open("scripts/output/vector_index_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Results saved to scripts/output/vector_index_results.json")
        
    except Exception as e:
        logger.error(f"Vector index creation failed: {str(e)}")
        sys.exit(1)
    
    finally:
        manager.close()

if __name__ == "__main__":
    main()
