#!/usr/bin/env python3
"""
Final test to confirm Text2Cypher functionality with the input key fix.
"""

import sys
import asyncio
import logging
sys.path.insert(0, 'src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_text2cypher_basic():
    """Basic test of Text2Cypher functionality."""
    
    logger.info("=== Testing Text2Cypher Input Key Fix ===")
    
    try:
        # Test basic imports and instantiation
        from ehs_analytics.config import get_settings
        from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
        
        settings = get_settings()
        
        config = {
            "neo4j_uri": settings.neo4j_uri,
            "neo4j_username": settings.neo4j_username,
            "neo4j_password": settings.neo4j_password,
            "openai_api_key": settings.openai_api_key,
            "model_name": "gpt-3.5-turbo",
            "temperature": 0,
            "max_tokens": 1000
        }
        
        # Initialize retriever
        retriever = Text2CypherRetriever(config)
        logger.info("✅ Text2CypherRetriever instantiated successfully")
        
        # Initialize (connect to Neo4j, load schema)
        await retriever.initialize()
        logger.info("✅ Text2CypherRetriever initialized successfully")
        
        # Test a simple query
        test_query = "Show me all facilities"
        logger.info(f"Testing query: '{test_query}'")
        
        result = await retriever.retrieve(test_query, limit=5)
        
        if result.success:
            logger.info(f"✅ Query successful! Results: {len(result.data)}")
            logger.info(f"Execution time: {result.metadata.execution_time_ms:.2f}ms")
            if result.data:
                logger.info(f"Sample result: {result.data[0]}")
            else:
                logger.info("⚠️  No data returned (schema mismatch possible)")
        else:
            logger.error(f"❌ Query failed: {result.error_message}")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_text2cypher_basic())
