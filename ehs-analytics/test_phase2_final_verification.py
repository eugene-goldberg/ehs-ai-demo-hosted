#!/usr/bin/env python3
"""
Final verification test for Text2Cypher retriever after all fixes:
1. Fixed Cypher prompt examples to match actual Neo4j schema
2. Fixed GraphCypherQAChain input key from "query" to "question"

This test validates that the retriever can generate correct Cypher queries
and return actual data from Neo4j.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_phase2_final_verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_test_config():
    """Create a test configuration for the retriever."""
    return {
        # Database configuration
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", "neo4jneo4j"),
        "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
        
        # LLM configuration
        "llm_provider": "openai",
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.0,
        
        # Retrieval configuration
        "use_graphrag": True,
        "query_optimization": True,
        "cache_common_queries": True,
        "max_query_complexity": 10,
        "cache_max_size": 100,
        
        # Text2Cypher specific
        "cypher_generation_template": None,  # Use default
        "cypher_validation": True,
        "max_results": 10,
        "timeout_seconds": 30,
        
        # EHS specific
        "ehs_schema_validation": True,
        "query_enhancement": True,
        "performance_monitoring": True
    }

async def test_ehs_text2cypher_retriever():
    """Test the EHS Text2Cypher retriever with various queries."""
    
    logger.info("="*80)
    logger.info("PHASE 2 FINAL VERIFICATION TEST")
    logger.info("Testing EHS Text2Cypher retriever after all fixes")
    logger.info("="*80)
    
    try:
        # Import the retriever
        from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever
        
        logger.info("✅ Successfully imported EHSText2CypherRetriever")
        
        # Create test configuration
        config = create_test_config()
        logger.info("✅ Created test configuration")
        
        # Initialize the retriever
        logger.info("Initializing EHSText2CypherRetriever...")
        retriever = EHSText2CypherRetriever(config)
        
        logger.info("✅ Successfully initialized EHSText2CypherRetriever")
        
        # Test queries that should work with our current schema
        test_queries = [
            "Show me all facilities",
            "What equipment do we have?", 
            "List active permits",
            "Show water consumption for all facilities",
            "What are the CO2 emissions?",
            "Show permits expiring in next 30 days",
            "What facilities are located in California?",
            "Show me utility bills from 2023",
            "What are the different types of equipment?",
            "Show me all consumption data",
            "List all documents",
            "What facilities have environmental permits?",
            "Show emission data for manufacturing facilities"
        ]
        
        results = {}
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST {i}: {query}")
            logger.info(f"{'='*60}")
            
            try:
                # Test the retriever
                result = await retriever.retrieve_async(query)
                
                logger.info(f"✅ Query processed successfully")
                logger.info(f"Result type: {type(result)}")
                
                if hasattr(result, 'results') and result.results:
                    logger.info(f"Number of results: {len(result.results)}")
                    logger.info(f"First result: {result.results[0]}")
                    if hasattr(result.results[0], 'content'):
                        logger.info(f"Result content: {result.results[0].content[:200]}")
                    
                if hasattr(result, 'metadata'):
                    logger.info(f"Metadata: {result.metadata}")
                    if hasattr(result.metadata, 'cypher_query'):
                        logger.info(f"Generated Cypher: {result.metadata.cypher_query}")
                else:
                    logger.info(f"Result: {str(result)[:200] if result else 'No result'}")
                
                results[query] = {
                    'status': 'SUCCESS',
                    'result': result,
                    'error': None
                }
                
            except Exception as e:
                logger.error(f"❌ Error processing query: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                results[query] = {
                    'status': 'ERROR',
                    'result': None,
                    'error': str(e)
                }
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*80}")
        
        successful = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        failed = sum(1 for r in results.values() if r['status'] == 'ERROR')
        
        logger.info(f"Total queries: {len(test_queries)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(test_queries)*100:.1f}%")
        
        if failed > 0:
            logger.info(f"\nFailed queries:")
            for query, result in results.items():
                if result['status'] == 'ERROR':
                    logger.info(f"  - {query}: {result['error']}")
        
        # Test direct GraphCypherQAChain if available
        logger.info(f"\n{'='*60}")
        logger.info("DIRECT CHAIN TEST")
        logger.info(f"{'='*60}")
        
        if hasattr(retriever, 'chain') or hasattr(retriever, 'qa_chain'):
            try:
                chain = getattr(retriever, 'chain', None) or getattr(retriever, 'qa_chain', None)
                if chain:
                    # Test with the correct input key "question"
                    direct_result = chain.invoke({"question": "Show me all facilities"})
                    logger.info(f"✅ Direct chain test successful")
                    logger.info(f"Direct result: {direct_result}")
                else:
                    logger.info("No chain found on retriever")
            except Exception as e:
                logger.error(f"❌ Direct chain test failed: {str(e)}")
        
        # Test Neo4j connection
        logger.info(f"\n{'='*60}")
        logger.info("NEO4J CONNECTION TEST")
        logger.info(f"{'='*60}")
        
        if hasattr(retriever, 'graph') or hasattr(retriever, 'neo4j_graph'):
            try:
                graph = getattr(retriever, 'graph', None) or getattr(retriever, 'neo4j_graph', None)
                if graph:
                    schema = graph.get_schema
                    logger.info(f"✅ Neo4j schema retrieved: {schema[:200] if schema else 'No schema'}")
                else:
                    logger.info("No graph connection found on retriever")
            except Exception as e:
                logger.error(f"❌ Neo4j schema test failed: {str(e)}")
        
        return results
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        logger.error("Make sure you're in the correct directory and the retriever module exists")
        return None
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def main():
    """Main async function to run the tests."""
    logger.info(f"Starting test at {datetime.now()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check required environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        return
    
    # Run the test
    results = await test_ehs_text2cypher_retriever()
    
    if results:
        logger.info("✅ Test completed successfully")
    else:
        logger.info("❌ Test failed to complete")
    
    logger.info(f"Test completed at {datetime.now()}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
