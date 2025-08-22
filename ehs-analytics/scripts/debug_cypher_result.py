#!/usr/bin/env python3
"""Debug script to inspect GraphCypherQAChain result structure."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ehs_analytics.config import Settings
from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.retrieval.base import QueryType

async def main():
    # Initialize settings
    settings = Settings()
    
    # Create retriever with config
    config = {
        "neo4j_uri": settings.neo4j_uri,
        "neo4j_user": settings.neo4j_username,
        "neo4j_password": settings.neo4j_password,
        "openai_api_key": settings.openai_api_key,
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.0,
        "cypher_validation": True
    }
    
    retriever = Text2CypherRetriever(config)
    
    try:
        print("🔧 Initializing Text2Cypher retriever...")
        await retriever.initialize()
        print("✅ Retriever initialized successfully\n")
        
        # Test query
        query = "Show all facilities"
        print(f"📊 Testing query: '{query}'")
        
        # Directly call the cypher chain to inspect result structure
        print("\n🔍 Calling GraphCypherQAChain directly...")
        
        try:
            # Call the chain directly
            result = retriever.cypher_chain.invoke({"query": query})
            
            print("\n📦 Result type:", type(result))
            print("\n📋 Result keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
            
            # Pretty print the full result
            print("\n🔍 Full result structure:")
            if isinstance(result, dict):
                print(json.dumps(result, indent=2, default=str))
            else:
                print(result)
            
            # Check for intermediate_steps
            if isinstance(result, dict) and "intermediate_steps" in result:
                print("\n📊 Intermediate steps found:")
                steps = result["intermediate_steps"]
                print(f"  - Type: {type(steps)}")
                print(f"  - Length: {len(steps) if hasattr(steps, '__len__') else 'N/A'}")
                
                if isinstance(steps, list) and len(steps) > 0:
                    print("\n  First step:")
                    first_step = steps[0]
                    print(f"    - Type: {type(first_step)}")
                    if isinstance(first_step, dict):
                        print(f"    - Keys: {list(first_step.keys())}")
                        print(f"    - Content: {json.dumps(first_step, indent=6, default=str)}")
            
            # Check for result key
            if isinstance(result, dict) and "result" in result:
                print("\n📊 Result data found:")
                result_data = result["result"]
                print(f"  - Type: {type(result_data)}")
                print(f"  - Value: {result_data}")
            
            # Try different possible paths to the Cypher query
            print("\n🔍 Searching for Cypher query in result...")
            
            # Path 1: Direct cypher_query key
            if isinstance(result, dict) and "cypher_query" in result:
                print(f"  ✅ Found at result['cypher_query']: {result['cypher_query']}")
            
            # Path 2: In intermediate_steps
            if isinstance(result, dict) and "intermediate_steps" in result:
                steps = result["intermediate_steps"]
                if isinstance(steps, list):
                    for i, step in enumerate(steps):
                        if isinstance(step, dict):
                            if "query" in step:
                                print(f"  ✅ Found at intermediate_steps[{i}]['query']: {step['query']}")
                            if "cypher" in step:
                                print(f"  ✅ Found at intermediate_steps[{i}]['cypher']: {step['cypher']}")
            
            # Path 3: Check all string values in the result
            def find_cypher_strings(obj, path=""):
                if isinstance(obj, str) and ("MATCH" in obj.upper() or "RETURN" in obj.upper()):
                    print(f"  🔍 Possible Cypher at {path}: {obj[:100]}...")
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        find_cypher_strings(v, f"{path}['{k}']")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        find_cypher_strings(item, f"{path}[{i}]")
            
            find_cypher_strings(result, "result")
            
        except Exception as e:
            print(f"\n❌ Error calling chain: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await retriever.cleanup()

if __name__ == "__main__":
    asyncio.run(main())