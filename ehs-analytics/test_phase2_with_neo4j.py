#!/usr/bin/env python3
"""
Test script for Text2Cypher retriever with actual Neo4j connection.

This script tests the Text2Cypher retriever implementation with real Neo4j data,
focusing on verifying that queries return actual results and logging detailed
information about the process.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import traceback

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Imports from our EHS Analytics system
from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.retrieval.base import QueryType
from neo4j import GraphDatabase

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_phase2_with_neo4j.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class Neo4jConnectionTester:
    """Helper class to test Neo4j connectivity and inspect schema."""
    
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
    
    def connect(self):
        """Establish Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def inspect_schema(self) -> Dict[str, Any]:
        """Inspect the current Neo4j database schema."""
        if not self.driver:
            return {}
        
        schema_info = {
            "node_labels": [],
            "relationship_types": [],
            "properties": {},
            "constraints": [],
            "indexes": []
        }
        
        try:
            with self.driver.session() as session:
                # Get node labels
                result = session.run("CALL db.labels()")
                schema_info["node_labels"] = [record["label"] for record in result]
                logger.info(f"Node labels found: {schema_info['node_labels']}")
                
                # Get relationship types
                result = session.run("CALL db.relationshipTypes()")
                schema_info["relationship_types"] = [record["relationshipType"] for record in result]
                logger.info(f"Relationship types found: {schema_info['relationship_types']}")
                
                # Get property keys for each label (limited to main ones)
                main_labels = ['Facility', 'Equipment', 'Permit', 'WaterBill', 'UtilityBill', 'Emission']
                for label in main_labels:
                    if label in schema_info["node_labels"]:
                        result = session.run(f"MATCH (n:{label}) RETURN DISTINCT keys(n) as props LIMIT 10")
                        props = set()
                        for record in result:
                            props.update(record["props"])
                        schema_info["properties"][label] = list(props)
                        logger.info(f"Properties for {label}: {list(props)}")
                
        except Exception as e:
            logger.error(f"Error inspecting schema: {e}")
        
        return schema_info
    
    def get_sample_data(self) -> Dict[str, List[Dict]]:
        """Get sample data from each node type."""
        sample_data = {}
        
        if not self.driver:
            return sample_data
        
        try:
            with self.driver.session() as session:
                # Get sample facilities
                result = session.run("MATCH (f:Facility) RETURN f LIMIT 3")
                sample_data["facilities"] = [dict(record["f"]) for record in result]
                
                # Get sample equipment
                result = session.run("MATCH (e:Equipment) RETURN e LIMIT 3")
                sample_data["equipment"] = [dict(record["e"]) for record in result]
                
                # Get sample permits
                result = session.run("MATCH (p:Permit) RETURN p LIMIT 3")
                sample_data["permits"] = [dict(record["p"]) for record in result]
                
                # Get sample water bills
                result = session.run("MATCH (w:WaterBill) RETURN w LIMIT 3")
                sample_data["water_bills"] = [dict(record["w"]) for record in result]
                
                # Get sample utility bills
                result = session.run("MATCH (u:UtilityBill) RETURN u LIMIT 3")
                sample_data["utility_bills"] = [dict(record["u"]) for record in result]
                
                # Get sample emissions
                result = session.run("MATCH (e:Emission) RETURN e LIMIT 3")
                sample_data["emissions"] = [dict(record["e"]) for record in result]
                
                # Log counts
                for data_type, records in sample_data.items():
                    logger.info(f"Sample {data_type}: {len(records)} records")
                    if records:
                        logger.debug(f"First {data_type} record: {records[0]}")
                        
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
        
        return sample_data
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

class Text2CypherTester:
    """Main testing class for Text2Cypher retriever."""
    
    def __init__(self):
        self.config = {
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.getenv("NEO4J_USERNAME", "neo4j"),
            "neo4j_password": os.getenv("NEO4J_PASSWORD", "EhsAI2024!"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 2000,
            "cypher_validation": True
        }
        self.retriever = None
        self.neo4j_tester = None
        
    async def setup(self):
        """Set up the test environment."""
        logger.info("Setting up Text2Cypher test environment")
        
        # Test Neo4j connectivity first
        self.neo4j_tester = Neo4jConnectionTester(
            self.config["neo4j_uri"],
            self.config["neo4j_user"],
            self.config["neo4j_password"]
        )
        
        if not self.neo4j_tester.connect():
            raise Exception("Cannot connect to Neo4j database")
        
        # Inspect schema
        logger.info("Inspecting Neo4j schema...")
        schema_info = self.neo4j_tester.inspect_schema()
        
        # Get sample data
        logger.info("Getting sample data...")
        sample_data = self.neo4j_tester.get_sample_data()
        
        # Initialize retriever
        logger.info("Initializing Text2Cypher retriever...")
        self.retriever = Text2CypherRetriever(self.config)
        await self.retriever.initialize()
        
        logger.info("Setup complete")
        return schema_info, sample_data
    
    async def test_query(self, query: str, query_type: QueryType = QueryType.GENERAL) -> Dict[str, Any]:
        """Test a single query and return detailed results."""
        logger.info(f"Testing query: '{query}' (type: {query_type})")
        
        start_time = datetime.now()
        
        try:
            # Execute the query
            result = await self.retriever.retrieve(query, query_type)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract details - check for different result formats
            documents = []
            result_count = 0
            
            if hasattr(result, 'data') and result.data:
                documents = result.data
                result_count = len(result.data)
            elif hasattr(result, 'documents') and result.documents:
                documents = result.documents
                result_count = len(result.documents)
            
            cypher_query = "N/A"
            if hasattr(result, 'metadata') and result.metadata:
                cypher_query = getattr(result.metadata, 'cypher_query', 'N/A')
            
            test_result = {
                "query": query,
                "query_type": query_type.value,
                "success": True,
                "duration_seconds": duration,
                "generated_cypher": cypher_query,
                "result_count": result_count,
                "results": documents[:3] if documents else [],  # First 3 results
                "metadata": result.metadata.__dict__ if hasattr(result, 'metadata') and result.metadata else {},
                "error": None
            }
            
            logger.info(f"Query completed - Duration: {duration:.2f}s, Results: {test_result['result_count']}")
            if test_result['generated_cypher'] != 'N/A':
                logger.info(f"Generated Cypher: {test_result['generated_cypher']}")
            
            if test_result['result_count'] > 0:
                logger.info(f"Sample result: {test_result['results'][0] if test_result['results'] else 'No results'}")
            else:
                logger.warning("Query returned 0 results")
            
            return test_result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Query failed: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            test_result = {
                "query": query,
                "query_type": query_type.value,
                "success": False,
                "duration_seconds": duration,
                "generated_cypher": "N/A",
                "result_count": 0,
                "results": [],
                "metadata": {},
                "error": str(e)
            }
            
            return test_result
    
    async def run_test_suite(self):
        """Run the complete test suite."""
        logger.info("Starting Text2Cypher test suite")
        
        # Test queries based on the requirements - using correct QueryType values
        test_queries = [
            ("Show me all facilities", QueryType.GENERAL),
            ("What equipment do we have?", QueryType.EFFICIENCY),
            ("List all permits", QueryType.COMPLIANCE),
            ("Show water bills", QueryType.CONSUMPTION),
            ("What are the emissions?", QueryType.EMISSIONS),
            ("Find facilities with high water consumption", QueryType.CONSUMPTION),
            ("Show me permits expiring soon", QueryType.COMPLIANCE),
            ("List equipment that needs maintenance", QueryType.EFFICIENCY),
            ("What are our CO2 emissions this year?", QueryType.EMISSIONS),
            ("Show me utility bills for electricity", QueryType.CONSUMPTION),
        ]
        
        test_results = []
        
        for query, query_type in test_queries:
            result = await self.test_query(query, query_type)
            test_results.append(result)
            
            # Add a small delay between queries
            await asyncio.sleep(1)
        
        return test_results
    
    def generate_report(self, test_results: List[Dict[str, Any]], schema_info: Dict[str, Any]):
        """Generate a detailed test report."""
        logger.info("Generating test report")
        
        # Summary statistics
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r['success'])
        failed_tests = total_tests - successful_tests
        tests_with_results = sum(1 for r in test_results if r['result_count'] > 0)
        
        avg_duration = sum(r['duration_seconds'] for r in test_results) / total_tests if total_tests > 0 else 0
        
        report = f"""
=== Text2Cypher Neo4j Integration Test Report ===
Generated: {datetime.now().isoformat()}

SUMMARY STATISTICS:
- Total tests executed: {total_tests}
- Successful executions: {successful_tests} ({successful_tests/total_tests*100:.1f}%)
- Failed executions: {failed_tests} ({failed_tests/total_tests*100:.1f}%)
- Tests returning data: {tests_with_results} ({tests_with_results/total_tests*100:.1f}%)
- Average query duration: {avg_duration:.2f} seconds

SCHEMA INFORMATION:
- Node labels: {', '.join(schema_info.get('node_labels', [])[:10])}...
- Relationship types: {', '.join(schema_info.get('relationship_types', [])[:10])}...
- Total properties mapped: {len(schema_info.get('properties', {}))}

DETAILED TEST RESULTS:
"""
        
        for i, result in enumerate(test_results, 1):
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            data_status = f"({result['result_count']} results)" if result['success'] else ""
            
            report += f"""
{i}. {status} {result['query']} {data_status}
   Type: {result['query_type']}
   Duration: {result['duration_seconds']:.2f}s
   Generated Cypher: {result['generated_cypher'][:100]}{'...' if len(str(result['generated_cypher'])) > 100 else ''}
"""
            
            if not result['success']:
                report += f"   Error: {result['error']}\n"
            elif result['result_count'] == 0:
                report += "   Warning: Query executed but returned no results\n"
            else:
                report += f"   Sample result: {str(result['results'][0])[:100] if result['results'] else 'None'}...\n"
        
        # Recommendations
        report += f"""
RECOMMENDATIONS:
"""
        if tests_with_results < successful_tests:
            report += "- Review Cypher query generation - some queries return no results\n"
            report += "- Check schema alignment between prompt examples and actual database\n"
        
        if failed_tests > 0:
            report += f"- Investigate {failed_tests} failed queries for error patterns\n"
        
        if avg_duration > 5.0:
            report += "- Consider query optimization for better performance\n"
        
        # Write report to file
        report_file = "text2cypher_test_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Test report written to {report_file}")
        print(report)
        
        return report
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up test environment")
        
        if self.retriever and hasattr(self.retriever, 'driver') and self.retriever.driver:
            self.retriever.driver.close()
        
        if self.neo4j_tester:
            self.neo4j_tester.close()
        
        logger.info("Cleanup complete")

async def main():
    """Main test execution function."""
    logger.info("Starting Text2Cypher Neo4j integration test")
    
    tester = Text2CypherTester()
    
    try:
        # Setup
        schema_info, sample_data = await tester.setup()
        
        # Run tests
        test_results = await tester.run_test_suite()
        
        # Generate report
        tester.generate_report(test_results, schema_info)
        
        # Summary
        successful = sum(1 for r in test_results if r['success'])
        with_data = sum(1 for r in test_results if r['result_count'] > 0)
        total = len(test_results)
        
        logger.info(f"Test suite completed: {successful}/{total} successful, {with_data}/{total} returned data")
        
        if successful == total and with_data > 0:
            logger.info("✅ Text2Cypher integration test PASSED")
            return 0
        else:
            logger.warning("⚠️  Text2Cypher integration test COMPLETED WITH ISSUES")
            return 1
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return 1
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
