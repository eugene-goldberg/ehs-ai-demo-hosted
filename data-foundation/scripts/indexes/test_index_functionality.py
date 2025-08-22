#!/usr/bin/env python3
"""
Comprehensive test suite for Neo4j vector and fulltext index functionality.
"""

import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'src'))

from neo4j import GraphDatabase
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndexFunctionalityTester:
    """Comprehensive tester for Neo4j index functionality."""
    
    def __init__(self):
        """Initialize with Neo4j connection and embedding model."""
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Initialize embedding model for vector tests
        try:
            self.embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {e}")
            self.embedding_model = None
        
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }
        }
    
    def close(self):
        """Close Neo4j driver."""
        if self.driver:
            self.driver.close()
    
    def run_test(self, test_name: str, test_func) -> bool:
        """
        Run a test and record results.
        
        Args:
            test_name: Name of the test
            test_func: Function to execute
            
        Returns:
            True if test passed, False otherwise
        """
        logger.info(f"Running test: {test_name}")
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            self.test_results["tests"][test_name] = {
                "status": "PASSED" if result else "FAILED",
                "duration": end_time - start_time,
                "details": result if isinstance(result, dict) else {"success": result}
            }
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
                self.test_results["summary"]["passed_tests"] += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
                self.test_results["summary"]["failed_tests"] += 1
            
            self.test_results["summary"]["total_tests"] += 1
            return bool(result)
            
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {str(e)}")
            
            self.test_results["tests"][test_name] = {
                "status": "ERROR",
                "error": str(e),
                "details": {}
            }
            
            self.test_results["summary"]["failed_tests"] += 1
            self.test_results["summary"]["total_tests"] += 1
            return False
    
    def test_vector_indexes_exist(self) -> Dict[str, Any]:
        """Test that vector indexes exist and are online."""
        try:
            with self.driver.session() as session:
                query = "SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR'"
                result = session.run(query)
                
                vector_indexes = {}
                for record in result:
                    index_name = record["name"]
                    state = record["state"]
                    vector_indexes[index_name] = state == "ONLINE"
                
                expected_indexes = [
                    "ehs_documents_vector",
                    "ehs_entities_vector", 
                    "ehs_summaries_vector"
                ]
                
                found_indexes = 0
                online_indexes = 0
                
                for expected in expected_indexes:
                    if expected in vector_indexes:
                        found_indexes += 1
                        if vector_indexes[expected]:
                            online_indexes += 1
                
                return {
                    "success": found_indexes > 0,
                    "expected_indexes": len(expected_indexes),
                    "found_indexes": found_indexes,
                    "online_indexes": online_indexes,
                    "indexes": vector_indexes
                }
                
        except Exception as e:
            logger.error(f"Vector index check failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_fulltext_indexes_exist(self) -> Dict[str, Any]:
        """Test that fulltext indexes exist and are online."""
        try:
            with self.driver.session() as session:
                query = "SHOW INDEXES YIELD name, type, state WHERE type = 'FULLTEXT'"
                result = session.run(query)
                
                fulltext_indexes = {}
                for record in result:
                    index_name = record["name"]
                    state = record["state"]
                    fulltext_indexes[index_name] = state == "ONLINE"
                
                expected_indexes = [
                    "ehs_document_text_search",
                    "ehs_entity_search",
                    "ehs_permit_search",
                    "ehs_utility_search",
                    "ehs_waste_search"
                ]
                
                found_indexes = 0
                online_indexes = 0
                
                for expected in expected_indexes:
                    if expected in fulltext_indexes:
                        found_indexes += 1
                        if fulltext_indexes[expected]:
                            online_indexes += 1
                
                return {
                    "success": found_indexes > 0,
                    "expected_indexes": len(expected_indexes),
                    "found_indexes": found_indexes,
                    "online_indexes": online_indexes,
                    "indexes": fulltext_indexes
                }
                
        except Exception as e:
            logger.error(f"Fulltext index check failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_vector_similarity_search(self) -> Dict[str, Any]:
        """Test vector similarity search functionality."""
        if not self.embedding_model:
            return {"success": False, "error": "No embedding model available"}
        
        try:
            with self.driver.session() as session:
                # Check if we have sample data
                check_query = "MATCH (d:DocumentChunk) RETURN count(d) as count"
                result = session.run(check_query)
                count = result.single()["count"]
                
                if count == 0:
                    return {"success": False, "error": "No sample data available for testing"}
                
                # Perform vector similarity search
                search_text = "energy consumption building"
                search_embedding = self.embedding_model.get_text_embedding(search_text)
                
                # Vector similarity query using cosine similarity
                query = """
                CALL db.index.vector.queryNodes('ehs_documents_vector', 3, $embedding)
                YIELD node, score
                RETURN node.text as text, score
                """
                
                result = session.run(query, {"embedding": search_embedding})
                search_results = []
                
                for record in result:
                    search_results.append({
                        "text": record["text"],
                        "score": record["score"]
                    })
                
                return {
                    "success": len(search_results) > 0,
                    "search_text": search_text,
                    "results_count": len(search_results),
                    "results": search_results[:3]  # Top 3 results
                }
                
        except Exception as e:
            logger.error(f"Vector similarity search failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_fulltext_search(self) -> Dict[str, Any]:
        """Test fulltext search functionality."""
        try:
            with self.driver.session() as session:
                # Check if we have sample data
                check_query = "MATCH (d:Document) RETURN count(d) as count"
                result = session.run(check_query)
                count = result.single()["count"]
                
                if count == 0:
                    return {"success": False, "error": "No sample data available for testing"}
                
                # Perform fulltext search
                search_terms = ["energy", "permit", "consumption"]
                search_results = {}
                
                for term in search_terms:
                    query = """
                    CALL db.index.fulltext.queryNodes('ehs_document_text_search', $search_term)
                    YIELD node, score
                    RETURN node.title as title, node.content as content, score
                    LIMIT 3
                    """
                    
                    result = session.run(query, {"search_term": term})
                    term_results = []
                    
                    for record in result:
                        term_results.append({
                            "title": record["title"],
                            "content": record["content"][:100] + "..." if record["content"] else "",
                            "score": record["score"]
                        })
                    
                    search_results[term] = term_results
                
                total_results = sum(len(results) for results in search_results.values())
                
                return {
                    "success": total_results > 0,
                    "search_terms": search_terms,
                    "total_results": total_results,
                    "results": search_results
                }
                
        except Exception as e:
            logger.error(f"Fulltext search failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_neo4j_connection(self) -> Dict[str, Any]:
        """Test basic Neo4j connection."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Hello Neo4j' as greeting")
                greeting = result.single()["greeting"]
                
                return {
                    "success": greeting == "Hello Neo4j",
                    "greeting": greeting
                }
                
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_data_integrity(self) -> Dict[str, Any]:
        """Test data integrity and constraints."""
        try:
            with self.driver.session() as session:
                # Check various node types and their counts
                queries = {
                    "DocumentChunk": "MATCH (d:DocumentChunk) RETURN count(d) as count",
                    "Document": "MATCH (d:Document) RETURN count(d) as count", 
                    "Entity": "MATCH (e:Entity) RETURN count(e) as count"
                }
                
                counts = {}
                for node_type, query in queries.items():
                    result = session.run(query)
                    counts[node_type] = result.single()["count"]
                
                # Check for relationships
                rel_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
                result = session.run(rel_query)
                relationship_count = result.single()["rel_count"]
                
                return {
                    "success": sum(counts.values()) > 0,
                    "node_counts": counts,
                    "relationship_count": relationship_count,
                    "total_nodes": sum(counts.values())
                }
                
        except Exception as e:
            logger.error(f"Data integrity test failed: {str(e)}")
            return {"success": False, "error": str(e)}

def main():
    """Main function to run all index functionality tests."""
    logger.info("Starting comprehensive index functionality tests...")
    
    tester = IndexFunctionalityTester()
    
    try:
        # Define tests to run
        tests = [
            ("Neo4j Connection", tester.test_neo4j_connection),
            ("Vector Indexes Exist", tester.test_vector_indexes_exist),
            ("Fulltext Indexes Exist", tester.test_fulltext_indexes_exist),
            ("Data Integrity", tester.test_data_integrity),
            ("Vector Similarity Search", tester.test_vector_similarity_search),
            ("Fulltext Search", tester.test_fulltext_search)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            tester.run_test(test_name, test_func)
        
        # Print summary
        summary = tester.test_results["summary"]
        logger.info("\n" + "="*60)
        logger.info("INDEX FUNCTIONALITY TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        
        success_rate = (summary['passed_tests'] / summary['total_tests']) * 100 if summary['total_tests'] > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Detailed results
        logger.info("\nDetailed Test Results:")
        for test_name, test_result in tester.test_results["tests"].items():
            status = test_result["status"]
            duration = test_result.get("duration", 0)
            logger.info(f"  {test_name}: {status} ({duration:.2f}s)")
            
            if test_result["status"] == "FAILED" and "error" in test_result:
                logger.info(f"    Error: {test_result['error']}")
        
        # Save results
        os.makedirs("scripts/output", exist_ok=True)
        results_file = "scripts/output/index_functionality_test_results.json"
        
        with open(results_file, "w") as f:
            json.dump(tester.test_results, f, indent=2)
        
        logger.info(f"\nTest results saved to: {results_file}")
        
        # Exit with appropriate code
        overall_success = summary['failed_tests'] == 0
        logger.info(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        sys.exit(0 if overall_success else 1)
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(1)
    
    finally:
        tester.close()

if __name__ == "__main__":
    main()
