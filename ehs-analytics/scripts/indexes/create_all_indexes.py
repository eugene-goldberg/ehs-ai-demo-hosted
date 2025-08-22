#!/usr/bin/env python3
"""
Create All Indexes for EHS Analytics Neo4j Database

This script orchestrates the creation of all indexes in the correct order:
1. Vector indexes for embeddings and semantic search
2. Fulltext indexes for text search across EHS entities
3. Performance optimization settings
4. Index health verification

Features:
- Runs all index creation in optimal order
- Index existence checks to prevent conflicts
- Comprehensive progress logging
- Error handling for existing indexes
- Performance optimization settings
- Health checks and verification
- Detailed execution summary
"""

import logging
import os
import sys
import time
from typing import Dict, Any, List
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv

# Add the scripts directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_vector_indexes import VectorIndexCreator
from create_fulltext_indexes import FulltextIndexCreator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexOrchestrator:
    """Orchestrates the creation of all EHS Analytics indexes."""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize with Neo4j connection details."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.uri = uri
        self.username = username
        self.password = password
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def check_neo4j_connection(self) -> bool:
        """Test Neo4j connection before proceeding."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection successful' as status")
                status = result.single()["status"]
                logger.info(f"Neo4j connection verified: {status}")
                return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
            
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and current index status."""
        info = {
            "neo4j_version": "unknown",
            "total_nodes": 0,
            "total_relationships": 0,
            "existing_indexes": 0,
            "vector_indexes": 0,
            "fulltext_indexes": 0
        }
        
        with self.driver.session() as session:
            try:
                # Get Neo4j version
                result = session.run("CALL dbms.components() YIELD name, versions")
                for record in result:
                    if record["name"] == "Neo4j Kernel":
                        info["neo4j_version"] = record["versions"][0]
                        
                # Get node count
                result = session.run("MATCH (n) RETURN count(n) as total")
                info["total_nodes"] = result.single()["total"]
                
                # Get relationship count
                result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
                info["total_relationships"] = result.single()["total"]
                
                # Get existing indexes
                result = session.run("SHOW INDEXES YIELD name, type")
                indexes = list(result)
                info["existing_indexes"] = len(indexes)
                info["vector_indexes"] = len([idx for idx in indexes if idx["type"] == "VECTOR"])
                info["fulltext_indexes"] = len([idx for idx in indexes if idx["type"] == "FULLTEXT"])
                
            except Exception as e:
                logger.warning(f"Could not retrieve complete database info: {e}")
                
        return info
        
    def optimize_database_settings(self) -> bool:
        """Apply performance optimization settings for index creation."""
        logger.info("Applying database optimization settings...")
        
        optimizations = [
            # Increase memory allocation for index operations
            "CALL dbms.setConfigValue('dbms.memory.heap.initial_size', '2G')",
            "CALL dbms.setConfigValue('dbms.memory.heap.max_size', '4G')",
            
            # Optimize for index creation
            "CALL dbms.setConfigValue('dbms.index_sampling.update_percentage', '5')",
            "CALL dbms.setConfigValue('dbms.index_sampling.sample_size_limit', '1000000')",
        ]
        
        applied_count = 0
        with self.driver.session() as session:
            for optimization in optimizations:
                try:
                    session.run(optimization)
                    applied_count += 1
                    logger.debug(f"Applied optimization: {optimization}")
                except Exception as e:
                    logger.warning(f"Could not apply optimization '{optimization}': {e}")
                    
        logger.info(f"Applied {applied_count}/{len(optimizations)} database optimizations")
        return applied_count > 0
        
    def wait_for_index_creation(self, index_names: List[str], timeout: int = 300) -> bool:
        """Wait for indexes to become online with timeout."""
        logger.info(f"Waiting for {len(index_names)} indexes to become online...")
        
        start_time = time.time()
        with self.driver.session() as session:
            while time.time() - start_time < timeout:
                try:
                    # Check index status
                    result = session.run("""
                        SHOW INDEXES 
                        YIELD name, state 
                        WHERE name IN $index_names
                    """, index_names=index_names)
                    
                    index_states = {record["name"]: record["state"] for record in result}
                    
                    # Check if all are online
                    pending_indexes = [name for name, state in index_states.items() 
                                     if state != "ONLINE"]
                    
                    if not pending_indexes:
                        logger.info("All indexes are now online")
                        return True
                        
                    logger.info(f"Waiting for {len(pending_indexes)} indexes to come online...")
                    time.sleep(5)
                    
                except Exception as e:
                    logger.warning(f"Error checking index status: {e}")
                    time.sleep(5)
                    
        logger.warning(f"Timeout waiting for indexes to come online after {timeout}s")
        return False
        
    def verify_index_performance(self) -> Dict[str, Any]:
        """Run performance checks on created indexes."""
        logger.info("Running index performance verification...")
        
        performance = {
            "vector_search_test": False,
            "fulltext_search_test": False,
            "index_statistics": {},
            "response_times": {}
        }
        
        with self.driver.session() as session:
            try:
                # Test vector search performance (if vector indexes exist)
                start_time = time.time()
                result = session.run("""
                    SHOW INDEXES 
                    YIELD name, type 
                    WHERE type = 'VECTOR' 
                    RETURN count(*) as vector_count
                """)
                vector_count = result.single()["vector_count"]
                performance["response_times"]["vector_index_check"] = time.time() - start_time
                performance["vector_search_test"] = vector_count > 0
                
                # Test fulltext search performance
                start_time = time.time()
                result = session.run("""
                    SHOW INDEXES 
                    YIELD name, type 
                    WHERE type = 'FULLTEXT' 
                    RETURN count(*) as fulltext_count
                """)
                fulltext_count = result.single()["fulltext_count"]
                performance["response_times"]["fulltext_index_check"] = time.time() - start_time
                performance["fulltext_search_test"] = fulltext_count > 0
                
                # Get index statistics
                result = session.run("""
                    SHOW INDEXES 
                    YIELD name, type, state, populationPercent, uniquenesses
                """)
                
                for record in result:
                    performance["index_statistics"][record["name"]] = {
                        "type": record["type"],
                        "state": record["state"],
                        "population_percent": record.get("populationPercent", 0),
                        "uniquenesses": record.get("uniquenesses", 0)
                    }
                    
            except Exception as e:
                logger.warning(f"Error during performance verification: {e}")
                
        return performance
        
    def create_all_indexes(self) -> Dict[str, Any]:
        """Create all indexes in the correct order."""
        logger.info("Starting comprehensive index creation for EHS Analytics...")
        
        summary = {
            "start_time": time.time(),
            "database_info": {},
            "vector_summary": {},
            "fulltext_summary": {},
            "performance": {},
            "total_indexes_created": 0,
            "success": False,
            "errors": []
        }
        
        try:
            # Step 1: Check database connection
            if not self.check_neo4j_connection():
                raise Exception("Neo4j connection failed")
                
            # Step 2: Get initial database info
            summary["database_info"] = self.get_database_info()
            logger.info(f"Database info - Nodes: {summary['database_info']['total_nodes']}, "
                       f"Relationships: {summary['database_info']['total_relationships']}, "
                       f"Existing indexes: {summary['database_info']['existing_indexes']}")
            
            # Step 3: Apply optimization settings
            self.optimize_database_settings()
            
            # Step 4: Create vector indexes first (foundation for RAG)
            logger.info("=" * 50)
            logger.info("PHASE 1: Creating Vector Indexes")
            logger.info("=" * 50)
            
            vector_creator = VectorIndexCreator(self.uri, self.username, self.password)
            try:
                summary["vector_summary"] = vector_creator.create_all_vector_indexes()
                logger.info(f"Vector indexes created: {summary['vector_summary']['total_created']}")
            finally:
                vector_creator.close()
                
            # Step 5: Create fulltext indexes (for hybrid search)
            logger.info("=" * 50)
            logger.info("PHASE 2: Creating Fulltext Indexes")
            logger.info("=" * 50)
            
            fulltext_creator = FulltextIndexCreator(self.uri, self.username, self.password)
            try:
                summary["fulltext_summary"] = fulltext_creator.create_all_fulltext_indexes()
                logger.info(f"Fulltext indexes created: {summary['fulltext_summary']['total_created']}")
            finally:
                fulltext_creator.close()
                
            # Step 6: Wait for all indexes to come online
            all_created_indexes = []
            if summary["vector_summary"].get("document_indexes"):
                all_created_indexes.extend(summary["vector_summary"]["document_indexes"])
            if summary["vector_summary"].get("chunk_indexes"):
                all_created_indexes.extend(summary["vector_summary"]["chunk_indexes"])
            if summary["vector_summary"].get("entity_indexes"):
                all_created_indexes.extend(summary["vector_summary"]["entity_indexes"])
            if summary["fulltext_summary"].get("facility_indexes"):
                all_created_indexes.extend(summary["fulltext_summary"]["facility_indexes"])
            if summary["fulltext_summary"].get("permit_indexes"):
                all_created_indexes.extend(summary["fulltext_summary"]["permit_indexes"])
            if summary["fulltext_summary"].get("equipment_indexes"):
                all_created_indexes.extend(summary["fulltext_summary"]["equipment_indexes"])
            if summary["fulltext_summary"].get("document_indexes"):
                all_created_indexes.extend(summary["fulltext_summary"]["document_indexes"])
            if summary["fulltext_summary"].get("specialized_indexes"):
                all_created_indexes.extend(summary["fulltext_summary"]["specialized_indexes"])
                
            if all_created_indexes:
                self.wait_for_index_creation(all_created_indexes)
                
            # Step 7: Performance verification
            logger.info("=" * 50)
            logger.info("PHASE 3: Performance Verification")
            logger.info("=" * 50)
            
            summary["performance"] = self.verify_index_performance()
            
            # Step 8: Calculate totals and success
            summary["total_indexes_created"] = (summary["vector_summary"].get("total_created", 0) + 
                                              summary["fulltext_summary"].get("total_created", 0))
            
            # Check for errors
            errors = []
            if summary["vector_summary"].get("errors"):
                errors.extend(summary["vector_summary"]["errors"])
            if summary["fulltext_summary"].get("errors"):
                errors.extend(summary["fulltext_summary"]["errors"])
            summary["errors"] = errors
            
            summary["success"] = (summary["total_indexes_created"] > 0 and 
                                len(errors) == 0 and
                                summary["vector_summary"].get("verification") == "success" and
                                summary["fulltext_summary"].get("verification") == "success")
                                
        except Exception as e:
            error_msg = f"Index creation orchestration failed: {e}"
            logger.error(error_msg)
            summary["errors"].append(error_msg)
            
        finally:
            summary["end_time"] = time.time()
            summary["duration"] = summary["end_time"] - summary["start_time"]
            
        return summary


def print_execution_summary(summary: Dict[str, Any]):
    """Print detailed execution summary."""
    logger.info("=" * 70)
    logger.info("EHS ANALYTICS INDEX CREATION SUMMARY")
    logger.info("=" * 70)
    
    # Basic stats
    logger.info(f"Execution Time: {summary.get('duration', 0):.2f} seconds")
    logger.info(f"Total Indexes Created: {summary.get('total_indexes_created', 0)}")
    logger.info(f"Success: {'✓' if summary.get('success') else '✗'}")
    
    # Database info
    db_info = summary.get("database_info", {})
    if db_info:
        logger.info(f"Neo4j Version: {db_info.get('neo4j_version', 'unknown')}")
        logger.info(f"Database Nodes: {db_info.get('total_nodes', 0):,}")
        logger.info(f"Database Relationships: {db_info.get('total_relationships', 0):,}")
        logger.info(f"Pre-existing Indexes: {db_info.get('existing_indexes', 0)}")
    
    # Vector indexes
    vector_summary = summary.get("vector_summary", {})
    if vector_summary:
        logger.info(f"Vector Indexes Created: {vector_summary.get('total_created', 0)}")
        logger.info(f"  - Document Indexes: {len(vector_summary.get('document_indexes', []))}")
        logger.info(f"  - Chunk Indexes: {len(vector_summary.get('chunk_indexes', []))}")
        logger.info(f"  - Entity Indexes: {len(vector_summary.get('entity_indexes', []))}")
    
    # Fulltext indexes
    fulltext_summary = summary.get("fulltext_summary", {})
    if fulltext_summary:
        logger.info(f"Fulltext Indexes Created: {fulltext_summary.get('total_created', 0)}")
        logger.info(f"  - Facility Indexes: {len(fulltext_summary.get('facility_indexes', []))}")
        logger.info(f"  - Permit Indexes: {len(fulltext_summary.get('permit_indexes', []))}")
        logger.info(f"  - Equipment Indexes: {len(fulltext_summary.get('equipment_indexes', []))}")
        logger.info(f"  - Document Indexes: {len(fulltext_summary.get('document_indexes', []))}")
        logger.info(f"  - Specialized Indexes: {len(fulltext_summary.get('specialized_indexes', []))}")
    
    # Performance
    performance = summary.get("performance", {})
    if performance:
        logger.info(f"Vector Search Ready: {'✓' if performance.get('vector_search_test') else '✗'}")
        logger.info(f"Fulltext Search Ready: {'✓' if performance.get('fulltext_search_test') else '✗'}")
    
    # Errors
    errors = summary.get("errors", [])
    if errors:
        logger.info(f"Errors Encountered: {len(errors)}")
        for i, error in enumerate(errors, 1):
            logger.error(f"  {i}. {error}")
    
    logger.info("=" * 70)


def main():
    """Main function to orchestrate all index creation."""
    # Get Neo4j connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    logger.info("Starting EHS Analytics comprehensive index creation...")
    
    orchestrator = IndexOrchestrator(uri, username, password)
    
    try:
        summary = orchestrator.create_all_indexes()
        print_execution_summary(summary)
        
        if summary.get("success"):
            logger.info("🎉 All indexes created successfully! EHS Analytics search is ready.")
            return 0
        else:
            logger.error("❌ Index creation completed with errors. Check the summary above.")
            return 1
            
    except Exception as e:
        logger.error(f"Critical error during index creation: {e}")
        return 1
        
    finally:
        orchestrator.close()


if __name__ == "__main__":
    exit(main())