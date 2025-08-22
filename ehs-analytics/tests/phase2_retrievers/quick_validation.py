#!/usr/bin/env python3
"""
Quick validation script for Phase 2 retrievers.
Tests basic functionality and connectivity.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

async def validate_basic_setup():
    """Validate basic setup and connectivity."""
    
    print("🔍 Validating Phase 2 Retriever Setup")
    print("=" * 40)
    
    results = {
        "imports": False,
        "config": False,
        "neo4j": False,
        "openai": False,
        "retrievers": 0
    }
    
    try:
        # Test imports
        from ehs_analytics.config import Settings
        from ehs_analytics.retrieval.base import QueryType, RetrievalStrategy
        from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever
        print("✅ Imports successful")
        results["imports"] = True
        
        # Test configuration
        settings = Settings()
        if settings.openai_api_key and settings.neo4j_uri:
            print("✅ Configuration loaded")
            results["config"] = True
        else:
            print("❌ Missing configuration (API keys or database URI)")
        
        # Test Neo4j connection
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password)
            )
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    print("✅ Neo4j connection successful")
                    results["neo4j"] = True
            driver.close()
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
        
        # Test OpenAI connection
        try:
            import openai
            client = openai.OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            if response.choices:
                print("✅ OpenAI connection successful")
                results["openai"] = True
        except Exception as e:
            print(f"❌ OpenAI connection failed: {e}")
        
        # Test retriever initialization
        if results["config"] and results["neo4j"]:
            try:
                config = {
                    "neo4j_uri": settings.neo4j_uri,
                    "neo4j_user": settings.neo4j_username,
                    "neo4j_password": settings.neo4j_password,
                    "openai_api_key": settings.openai_api_key,
                    "model_name": "gpt-4"
                }
                
                retriever = EHSText2CypherRetriever(config)
                await retriever.initialize()
                print("✅ Text2Cypher retriever initialized")
                results["retrievers"] += 1
                
                # Test a simple query
                test_result = await retriever.retrieve(
                    query="Show me equipment status",
                    query_type=QueryType.GENERAL,
                    limit=5
                )
                
                if test_result:
                    print(f"✅ Test query executed: {len(test_result.data)} results")
                else:
                    print("⚠️  Test query returned no results")
                
                await retriever.cleanup()
                
            except Exception as e:
                print(f"❌ Retriever initialization failed: {e}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    # Print summary
    print("\n📊 Validation Summary")
    print("-" * 20)
    print(f"Imports: {'✅' if results['imports'] else '❌'}")
    print(f"Configuration: {'✅' if results['config'] else '❌'}")
    print(f"Neo4j: {'✅' if results['neo4j'] else '❌'}")
    print(f"OpenAI: {'✅' if results['openai'] else '❌'}")
    print(f"Retrievers: {results['retrievers']} initialized")
    
    ready_for_testing = all([
        results["imports"],
        results["config"],
        results["neo4j"],
        results["openai"],
        results["retrievers"] > 0
    ])
    
    print(f"\n🎯 Ready for comprehensive testing: {'✅' if ready_for_testing else '❌'}")
    
    return ready_for_testing

if __name__ == "__main__":
    success = asyncio.run(validate_basic_setup())
    sys.exit(0 if success else 1)
