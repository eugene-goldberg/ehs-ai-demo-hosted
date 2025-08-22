#!/usr/bin/env python3
"""
Test script to verify Neo4j indexes are created and functional
"""

import os
import logging
from datetime import datetime
from neo4j import GraphDatabase
from openai import OpenAI
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Neo4jIndexTester:
    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
        self.neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        # Initialize connections
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Initialize OpenAI for embeddings
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            logger.warning("OpenAI API key not found - vector search tests will be skipped")
            self.openai_client = None
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'summary': {}
        }
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def get_embedding(self, text):
        """Get OpenAI embedding for text"""
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def run_cypher_query(self, query, parameters=None):
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(query, parameters or {})
                return [record for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None
    
    def test_database_connection(self):
        """Test basic database connectivity"""
        logger.info("Testing database connection...")
        
        query = """
        MATCH (n)
        RETURN count(n) as node_count
        """
        
        result = self.run_cypher_query(query)
        if result:
            node_count = result[0]['node_count']
            test_result = {
                'test': 'database_connection',
                'status': 'PASS',
                'details': f'Connected successfully. Found {node_count} nodes in database.'
            }
            logger.info(f"✓ Database connection successful. Nodes: {node_count}")
        else:
            test_result = {
                'test': 'database_connection',
                'status': 'FAIL',
                'details': 'Failed to connect to database'
            }
            logger.error("✗ Database connection failed")
        
        self.results['tests'].append(test_result)
        return test_result['status'] == 'PASS'
    
    def test_list_indexes(self):
        """List all indexes in the database"""
        logger.info("Listing all indexes...")
        
        query = "SHOW INDEXES"
        
        result = self.run_cypher_query(query)
        if result:
            vector_indexes = []
            fulltext_indexes = []
            other_indexes = []
            
            for record in result:
                index_name = record['name']
                index_type = record['type']
                index_state = record['state']
                
                if 'vector' in index_name.lower():
                    vector_indexes.append({
                        'name': index_name,
                        'type': index_type,
                        'state': index_state
                    })
                elif 'fulltext' in index_name.lower():
                    fulltext_indexes.append({
                        'name': index_name,
                        'type': index_type,
                        'state': index_state
                    })
                else:
                    other_indexes.append({
                        'name': index_name,
                        'type': index_type,
                        'state': index_state
                    })
            
            test_result = {
                'test': 'list_indexes',
                'status': 'PASS',
                'details': {
                    'total_indexes': len(result),
                    'vector_indexes': len(vector_indexes),
                    'fulltext_indexes': len(fulltext_indexes),
                    'other_indexes': len(other_indexes),
                    'vector_index_list': vector_indexes,
                    'fulltext_index_list': fulltext_indexes
                }
            }
            
            logger.info(f"✓ Found {len(result)} total indexes:")
            logger.info(f"  - Vector indexes: {len(vector_indexes)}")
            logger.info(f"  - Fulltext indexes: {len(fulltext_indexes)}")
            logger.info(f"  - Other indexes: {len(other_indexes)}")
            
        else:
            test_result = {
                'test': 'list_indexes',
                'status': 'FAIL',
                'details': 'Failed to retrieve indexes'
            }
            logger.error("✗ Failed to list indexes")
        
        self.results['tests'].append(test_result)
        return test_result['status'] == 'PASS'
    
    def test_vector_search(self):
        """Test vector similarity search functionality"""
        logger.info("Testing vector similarity search...")
        
        if not self.openai_client:
            test_result = {
                'test': 'vector_search',
                'status': 'SKIP',
                'details': 'OpenAI API key not available'
            }
            self.results['tests'].append(test_result)
            return True
        
        # Test query
        test_query = "environmental compliance requirements"
        embedding = self.get_embedding(test_query)
        
        if not embedding:
            test_result = {
                'test': 'vector_search',
                'status': 'FAIL',
                'details': 'Failed to generate embedding for test query'
            }
            self.results['tests'].append(test_result)
            return False
        
        # Test vector search on DocumentChunk
        query = """
        MATCH (c:DocumentChunk)
        WHERE c.content_embedding IS NOT NULL
        CALL db.index.vector.queryNodes('ehs_chunk_content_vector_idx', 5, $embedding)
        YIELD node, score
        RETURN node.content as content, score
        LIMIT 5
        """
        
        result = self.run_cypher_query(query, {'embedding': embedding})
        
        if result and len(result) > 0:
            test_result = {
                'test': 'vector_search',
                'status': 'PASS',
                'details': {
                    'query': test_query,
                    'results_found': len(result),
                    'top_result_score': float(result[0]['score']) if result else None,
                    'sample_results': [
                        {
                            'content_preview': r['content'][:100] + '...' if r['content'] else 'No content',
                            'score': float(r['score'])
                        } for r in result[:3]
                    ]
                }
            }
            logger.info(f"✓ Vector search successful. Found {len(result)} results")
            logger.info(f"  Top result score: {result[0]['score']:.4f}")
        else:
            test_result = {
                'test': 'vector_search',
                'status': 'FAIL',
                'details': 'No results returned from vector search'
            }
            logger.error("✗ Vector search returned no results")
        
        self.results['tests'].append(test_result)
        return test_result['status'] == 'PASS'
    
    def test_fulltext_search(self):
        """Test fulltext search functionality"""
        logger.info("Testing fulltext search...")
        
        # Test fulltext search on documents
        test_queries = [
            "environmental",
            "compliance",
            "permit",
            "facility"
        ]
        
        successful_tests = 0
        total_tests = len(test_queries)
        test_details = []
        
        for query_term in test_queries:
            query = """
            CALL db.index.fulltext.queryNodes('ehs_document_content_fulltext_idx', $searchTerm)
            YIELD node, score
            RETURN node.title as title, node.document_type as type, score
            LIMIT 5
            """
            
            result = self.run_cypher_query(query, {'searchTerm': query_term})
            
            if result and len(result) > 0:
                successful_tests += 1
                test_details.append({
                    'query': query_term,
                    'results_found': len(result),
                    'top_score': float(result[0]['score']),
                    'sample_titles': [r['title'] for r in result[:2] if r['title']]
                })
                logger.info(f"  ✓ Fulltext search for '{query_term}': {len(result)} results")
            else:
                test_details.append({
                    'query': query_term,
                    'results_found': 0,
                    'error': 'No results found'
                })
                logger.warning(f"  ✗ Fulltext search for '{query_term}': No results")
        
        if successful_tests > 0:
            test_result = {
                'test': 'fulltext_search',
                'status': 'PASS',
                'details': {
                    'successful_queries': successful_tests,
                    'total_queries': total_tests,
                    'success_rate': successful_tests / total_tests,
                    'query_results': test_details
                }
            }
            logger.info(f"✓ Fulltext search: {successful_tests}/{total_tests} queries successful")
        else:
            test_result = {
                'test': 'fulltext_search',
                'status': 'FAIL',
                'details': {
                    'successful_queries': 0,
                    'total_queries': total_tests,
                    'error': 'All fulltext queries failed'
                }
            }
            logger.error("✗ All fulltext search queries failed")
        
        self.results['tests'].append(test_result)
        return test_result['status'] == 'PASS'
    
    def test_hybrid_search(self):
        """Test combined vector + fulltext search"""
        logger.info("Testing hybrid search...")
        
        if not self.openai_client:
            test_result = {
                'test': 'hybrid_search',
                'status': 'SKIP',
                'details': 'OpenAI API key not available'
            }
            self.results['tests'].append(test_result)
            return True
        
        test_query = "environmental permit compliance"
        embedding = self.get_embedding(test_query)
        
        if not embedding:
            test_result = {
                'test': 'hybrid_search',
                'status': 'FAIL',
                'details': 'Failed to generate embedding'
            }
            self.results['tests'].append(test_result)
            return False
        
        # Hybrid search combining vector and fulltext
        query = """
        // Vector search
        CALL db.index.vector.queryNodes('ehs_chunk_content_vector_idx', 10, $embedding)
        YIELD node as vectorNode, score as vectorScore
        
        // Fulltext search  
        CALL db.index.fulltext.queryNodes('ehs_chunk_content_fulltext_idx', $searchTerm)
        YIELD node as fulltextNode, score as fulltextScore
        
        // Combine results
        WITH collect({node: vectorNode, score: vectorScore, type: 'vector'}) as vectorResults,
             collect({node: fulltextNode, score: fulltextScore, type: 'fulltext'}) as fulltextResults
        
        UNWIND vectorResults + fulltextResults as result
        WITH result.node as node, result.score as score, result.type as searchType
        
        RETURN DISTINCT node.content as content, 
               collect(DISTINCT searchType) as searchTypes,
               max(score) as bestScore
        ORDER BY bestScore DESC
        LIMIT 5
        """
        
        result = self.run_cypher_query(query, {
            'embedding': embedding,
            'searchTerm': test_query
        })
        
        if result and len(result) > 0:
            test_result = {
                'test': 'hybrid_search',
                'status': 'PASS',
                'details': {
                    'query': test_query,
                    'results_found': len(result),
                    'sample_results': [
                        {
                            'content_preview': r['content'][:100] + '...' if r['content'] else 'No content',
                            'search_types': r['searchTypes'],
                            'score': float(r['bestScore'])
                        } for r in result[:3]
                    ]
                }
            }
            logger.info(f"✓ Hybrid search successful. Found {len(result)} results")
        else:
            test_result = {
                'test': 'hybrid_search',
                'status': 'FAIL',
                'details': 'No results from hybrid search'
            }
            logger.error("✗ Hybrid search returned no results")
        
        self.results['tests'].append(test_result)
        return test_result['status'] == 'PASS'
    
    def run_all_tests(self):
        """Run all index tests"""
        logger.info("Starting comprehensive Neo4j index testing...")
        
        tests = [
            self.test_database_connection,
            self.test_list_indexes,
            self.test_vector_search,
            self.test_fulltext_search,
            self.test_hybrid_search
        ]
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_func in tests:
            try:
                result = test_func()
                if result is True:
                    passed += 1
                elif result is False:
                    failed += 1
                else:  # None or skipped
                    skipped += 1
            except Exception as e:
                logger.error(f"Test {test_func.__name__} crashed: {e}")
                failed += 1
                self.results['tests'].append({
                    'test': test_func.__name__,
                    'status': 'ERROR',
                    'details': f'Test crashed: {str(e)}'
                })
        
        # Generate summary
        self.results['summary'] = {
            'total_tests': len(tests),
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': passed / len(tests) if len(tests) > 0 else 0
        }
        
        logger.info("=" * 60)
        logger.info("NEO4J INDEX TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {len(tests)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"Success Rate: {passed/len(tests)*100:.1f}%")
        logger.info("=" * 60)
        
        return self.results

def main():
    """Main function to run the tests"""
    tester = Neo4jIndexTester()
    
    try:
        results = tester.run_all_tests()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"neo4j_index_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to: {results_file}")
        
        # Return appropriate exit code
        if results['summary']['failed'] > 0:
            return 1
        else:
            return 0
        
    finally:
        tester.close()

if __name__ == "__main__":
    exit(main())
