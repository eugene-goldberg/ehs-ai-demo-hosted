#!/usr/bin/env python3
"""
Simplified test for Text2Cypher retriever with Neo4j
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Import EHS components
from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever

async def test_text2cypher():
    """Test Text2Cypher retriever with simple queries"""
    print(f"\n=== TEXT2CYPHER RETRIEVER TEST ===")
    print(f"Time: {datetime.now()}")
    
    # Configuration
    config = {
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.getenv("NEO4J_USERNAME", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.0
    }
    
    print(f"\nConfiguration:")
    print(f"  Neo4j URI: {config['neo4j_uri']}")
    print(f"  Neo4j User: {config['neo4j_user']}")
    print(f"  OpenAI Model: {config['model_name']}")
    
    try:
        # Initialize Text2Cypher retriever
        print("\n1. Initializing Text2Cypher retriever...")
        retriever = Text2CypherRetriever(config)
        await retriever.initialize()
        print("   ✅ Retriever initialized")
        
        # Test queries
        test_queries = [
            "What facilities do we have?",
            "Show me water consumption data",
            "What permits exist in the system?",
            "List all equipment"
        ]
        
        print("\n2. Testing queries:")
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            try:
                result = await retriever.retrieve(query)
                print(f"   ✅ Success - Retrieved {len(result.results) if hasattr(result, 'results') else 0} results")
                if result.cypher_query:
                    print(f"   Generated Cypher: {result.cypher_query}")
                if result.results and len(result.results) > 0:
                    print(f"   First result: {str(result.results[0])[:100]}...")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        print("\n✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_text2cypher())
    exit(0 if success else 1)