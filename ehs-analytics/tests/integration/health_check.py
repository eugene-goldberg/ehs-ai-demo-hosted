#!/usr/bin/env python3
"""
Quick Phase 2 Component Health Check

Validates that all Phase 2 components can be imported and initialized
"""

import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def check_imports():
    """Check all critical imports."""
    print("🔍 Checking Phase 2 component imports...")
    
    try:
        from ehs_analytics.config import get_settings
        print("  ✅ Config")
        
        from ehs_analytics.agents.query_router import QueryRouterAgent
        print("  ✅ Query Router")
        
        from ehs_analytics.agents.rag_agent import RAGAgent
        print("  ✅ RAG Agent")
        
        from ehs_analytics.retrieval.orchestrator import RetrievalOrchestrator
        print("  ✅ Retrieval Orchestrator")
        
        from ehs_analytics.workflows.ehs_workflow import EHSWorkflow
        print("  ✅ EHS Workflow")
        
        from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever
        from ehs_analytics.retrieval.strategies.vector_retriever import EHSVectorRetriever
        from ehs_analytics.retrieval.strategies.hybrid_cypher_retriever import EHSHybridCypherRetriever
        from ehs_analytics.retrieval.strategies.vector_cypher_retriever import EHSVectorCypherRetriever
        print("  ✅ All Retriever Strategies")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def check_basic_initialization():
    """Check basic component initialization."""
    print("\n🔧 Checking basic component initialization...")
    
    try:
        from ehs_analytics.config import get_settings
        settings = get_settings()
        print("  ✅ Settings loaded")
        
        from ehs_analytics.agents.query_router import QueryRouterAgent
        router = QueryRouterAgent()
        print("  ✅ Query Router created")
        
        # Test basic classification
        test_classification = router.classify_query("Test query for health check")
        print(f"  ✅ Classification works (intent: {test_classification.intent_type.value})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return False

def main():
    """Run health check."""
    start_time = time.time()
    
    print("🏥 Phase 2 Component Health Check")
    print("=" * 40)
    
    # Check imports
    imports_ok = check_imports()
    
    # Check basic initialization
    init_ok = check_basic_initialization()
    
    # Overall status
    duration = (time.time() - start_time) * 1000
    
    print(f"\n📊 Health Check Results")
    print("-" * 25)
    print(f"Imports: {'✅ OK' if imports_ok else '❌ FAILED'}")
    print(f"Initialization: {'✅ OK' if init_ok else '❌ FAILED'}")
    print(f"Duration: {duration:.0f}ms")
    
    overall_health = imports_ok and init_ok
    print(f"\nOverall Status: {'✅ HEALTHY' if overall_health else '❌ UNHEALTHY'}")
    
    return 0 if overall_health else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
