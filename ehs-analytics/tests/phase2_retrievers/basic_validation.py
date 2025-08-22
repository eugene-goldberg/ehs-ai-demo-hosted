#!/usr/bin/env python3
"""
Basic validation script for Phase 2 retrievers core functionality.
Tests essential components without external vector store dependencies.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

async def validate_core_functionality():
    """Validate core functionality without external dependencies."""
    
    print("🔍 Validating Core Phase 2 Functionality")
    print("=" * 45)
    
    results = {
        "imports": False,
        "config": False,
        "neo4j": False,
        "openai": False,
        "basic_retriever": False
    }
    
    try:
        # Test core imports
        print("📦 Testing imports...")
        from ehs_analytics.config import Settings
        from ehs_analytics.retrieval.base import QueryType, RetrievalStrategy
        from ehs_analytics.retrieval.orchestrator import RetrievalOrchestrator
        from ehs_analytics.agents.rag_agent import RAGAgent
        print("✅ Core imports successful")
        results["imports"] = True
        
        # Test configuration
        print("⚙️  Testing configuration...")
        settings = Settings()
        if settings.openai_api_key and settings.neo4j_uri:
            print("✅ Configuration loaded successfully")
            results["config"] = True
        else:
            print("⚠️  Some configuration missing (this is OK for basic validation)")
            results["config"] = bool(settings.openai_api_key)
        
        # Test basic Neo4j connection (if configured)
        if settings.neo4j_uri and settings.neo4j_username and settings.neo4j_password:
            print("🗄️  Testing Neo4j connection...")
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
        else:
            print("⏭️  Skipping Neo4j test (not configured)")
        
        # Test basic OpenAI connection (if configured)
        if settings.openai_api_key and len(settings.openai_api_key) > 20:
            print("🤖 Testing OpenAI connection...")
            try:
                import openai
                client = openai.OpenAI(api_key=settings.openai_api_key)
                # Just test client creation, not actual API call to save costs
                print("✅ OpenAI client created successfully")
                results["openai"] = True
            except Exception as e:
                print(f"❌ OpenAI client creation failed: {e}")
        else:
            print("⏭️  Skipping OpenAI test (not configured)")
        
        # Test basic retriever class instantiation
        print("🔧 Testing retriever classes...")
        try:
            from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever
            
            # Test instantiation without initialization
            config = {
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_user": "neo4j", 
                "neo4j_password": "password",
                "openai_api_key": "test_key",
                "model_name": "gpt-4"
            }
            
            retriever = EHSText2CypherRetriever(config)
            print("✅ Basic retriever instantiation successful")
            results["basic_retriever"] = True
            
        except Exception as e:
            print(f"❌ Retriever instantiation failed: {e}")
        
    except Exception as e:
        print(f"❌ Core validation failed: {e}")
        return False
    
    # Print summary
    print("\n📊 Validation Summary")
    print("-" * 25)
    print(f"Core Imports: {'✅' if results['imports'] else '❌'}")
    print(f"Configuration: {'✅' if results['config'] else '❌'}")
    print(f"Neo4j: {'✅' if results['neo4j'] else '⏭️ '}")
    print(f"OpenAI: {'✅' if results['openai'] else '⏭️ '}")
    print(f"Retrievers: {'✅' if results['basic_retriever'] else '❌'}")
    
    # Determine readiness
    essential_ready = all([
        results["imports"],
        results["basic_retriever"]
    ])
    
    full_ready = all([
        results["imports"],
        results["config"],
        results["neo4j"],
        results["openai"],
        results["basic_retriever"]
    ])
    
    print(f"\n🎯 Essential components ready: {'✅' if essential_ready else '❌'}")
    print(f"🎯 Full testing ready: {'✅' if full_ready else '❌'}")
    
    if essential_ready:
        print("\n💡 Ready for basic retriever testing!")
        if not full_ready:
            print("   Note: Some tests may be skipped due to missing configuration")
    else:
        print("\n❌ Core components not ready for testing")
    
    return essential_ready

if __name__ == "__main__":
    success = asyncio.run(validate_core_functionality())
    sys.exit(0 if success else 1)
