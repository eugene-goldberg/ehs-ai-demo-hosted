#!/usr/bin/env python3
"""
Final verification test for Phase 2 Text2Cypher retriever
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import EHS components
from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever

async def test_text2cypher_with_new_data():
    """Test Text2Cypher with the newly added test data"""
    print(f"\n=== PHASE 2 TEXT2CYPHER FINAL TEST ===")
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
    
    try:
        # Test regular Text2Cypher
        print("\n1. Testing Text2CypherRetriever...")
        retriever = Text2CypherRetriever(config)
        await retriever.initialize()
        print("   ✅ Text2CypherRetriever initialized")
        
        # Test queries against new data
        test_queries = [
            "Show me all facilities",
            "What equipment do we have?",
            "List active permits",
            "Show water consumption for the past 6 months",
            "What are the CO2 emissions?"
        ]
        
        print("\n2. Testing queries:")
        success_count = 0
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            try:
                result = await retriever.retrieve(query)
                result_count = len(result.results) if hasattr(result, 'results') else 0
                print(f"   ✅ Success - Retrieved {result_count} results")
                if hasattr(result, 'metadata') and hasattr(result.metadata, 'cypher_query'):
                    print(f"   Cypher: {result.metadata.cypher_query}")
                if result_count > 0:
                    print(f"   Sample: {str(result.results[0])[:100]}...")
                success_count += 1
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        # Cleanup
        await retriever.cleanup()
        
        # Test EHS Text2Cypher
        print("\n3. Testing EHSText2CypherRetriever...")
        ehs_retriever = EHSText2CypherRetriever(config)
        await ehs_retriever.initialize()
        print("   ✅ EHSText2CypherRetriever initialized")
        
        # Test one query
        result = await ehs_retriever.retrieve("Show facility water consumption trends")
        print(f"   ✅ EHS retriever test completed")
        
        await ehs_retriever.cleanup()
        
        print(f"\n✅ FINAL TEST COMPLETED")
        print(f"Success rate: {success_count}/{len(test_queries)} ({100*success_count/len(test_queries):.0f}%)")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_text2cypher_with_new_data())
    exit(0 if success else 1)