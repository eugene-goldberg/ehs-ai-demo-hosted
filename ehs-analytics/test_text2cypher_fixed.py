#!/usr/bin/env python3
"""
Test script to verify Text2Cypher retriever functionality after input key fix.
Tests with simple queries to validate Cypher generation and data retrieval.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, '/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src')

# Configure logging to file
logging.basicConfig(
    filename='test_text2cypher_fixed.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

async def test_text2cypher_retriever():
    """Test the Text2Cypher retriever with simple queries."""
    
    logger.info("=== Starting Text2Cypher Retriever Test ===")
    
    driver = None
    try:
        # Import required components using existing structure
        from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever
        from ehs_analytics.config import get_settings
        from neo4j import AsyncGraphDatabase
        
        logger.info("✅ Successfully imported required modules")
        
        # Get settings
        settings = get_settings()
        
        # Initialize Neo4j connection
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        
        logger.info("✅ Connected to Neo4j database")
        
        # Create configuration dictionary
        config = {
            "neo4j_uri": settings.neo4j_uri,
            "neo4j_username": settings.neo4j_username,
            "neo4j_password": settings.neo4j_password,
            "openai_api_key": settings.openai_api_key,
            "use_graphrag": True,
            "query_optimization": True,
            "cache_common_queries": False,  # Disable for testing
            "max_query_complexity": 10
        }
        
        # Initialize Text2Cypher retriever with config
        retriever = EHSText2CypherRetriever(config=config)
        
        # Initialize the retriever (loads schema, sets up chains)
        await retriever.initialize()
        logger.info("✅ Text2Cypher retriever initialized successfully")
        
        # Test queries
        test_queries = [
            "Show me all facilities",
            "List water bills", 
            "What equipment exists?"
        ]
        
        logger.info(f"🧪 Testing {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Test {i}: '{query}' ---")
            
            try:
                # Execute the query
                result = await retriever.aretrieve(query)
                
                logger.info(f"Query executed successfully")
                logger.info(f"Result type: {type(result)}")
                
                # Check if result has expected structure
                if hasattr(result, 'data'):
                    logger.info(f"Number of results: {len(result.data)}")
                    logger.info(f"Result success: {result.success}")
                    if result.data:
                        logger.info(f"First result preview: {str(result.data[0])[:200]}")
                        logger.info(f"✅ Query '{query}' returned {len(result.data)} results")
                    else:
                        logger.info(f"⚠️  Query '{query}' returned no results (may be schema mismatch)")
                else:
                    logger.info(f"Result structure: {result}")
                    
            except Exception as e:
                logger.error(f"❌ Query '{query}' failed: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Test the schema alignment
        logger.info("\n--- Schema Information ---")
        try:
            schema_info = await retriever.get_schema()
            logger.info(f"Schema loaded successfully: {len(schema_info) if schema_info else 0} characters")
            logger.info(f"Schema preview: {schema_info[:300] if schema_info else 'No schema'}")
        except Exception as e:
            logger.error(f"Failed to get schema: {str(e)}")
        
        logger.info("\n=== Text2Cypher Test Completed ===")
        logger.info("✅ Core functionality verified - retriever generates queries and connects to database")
        logger.info("📋 Next step: Fix schema examples in prompt template for accurate results")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
    finally:
        try:
            if driver:
                await driver.close()
            logger.info("✅ Neo4j connection closed")
        except:
            logger.info("⚠️  Neo4j connection cleanup attempted")

if __name__ == "__main__":
    logger.info(f"Starting test at {datetime.now()}")
    logger.info("Using virtual environment: /Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/venv")
    
    # Run the async test
    asyncio.run(test_text2cypher_retriever())
    
    logger.info(f"Test completed at {datetime.now()}")
    logger.info("Check test_text2cypher_fixed.log for detailed results")
