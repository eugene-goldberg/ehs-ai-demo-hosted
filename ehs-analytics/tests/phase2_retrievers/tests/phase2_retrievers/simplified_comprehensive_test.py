#!/usr/bin/env python3
"""
Simplified Comprehensive Phase 2 Retriever Tests

This test suite validates the core Phase 2 retriever functionality.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# EHS Analytics imports
try:
    from ehs_analytics.config import Settings
    from ehs_analytics.retrieval.base import QueryType, RetrievalStrategy
    from ehs_analytics.retrieval.orchestrator import RetrievalOrchestrator, OrchestrationConfig
    from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

@dataclass
class TestResult:
    test_name: str
    success: bool
    duration_ms: float
    details: str
    error_message: Optional[str] = None

class Phase2TestSuite:
    def __init__(self):
        self.settings = Settings()
        self.test_results = []
        
    def run_test(self, test_name: str, test_func):
        print(f"\n🧪 Running: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = (time.time() - start_time) * 1000
            
            test_result = TestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration,
                details=str(result) if result else "Test passed",
            )
            print(f"✅ {test_name} - {duration:.0f}ms")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            test_result = TestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration,
                details="Test failed",
                error_message=str(e)
            )
            print(f"❌ {test_name} - {duration:.0f}ms - {e}")
        
        self.test_results.append(test_result)
        return test_result.success

    def test_configuration_loading(self):
        config = Settings()
        assert hasattr(config, 'neo4j_uri'), "Missing neo4j_uri"
        assert hasattr(config, 'openai_api_key'), "Missing openai_api_key"
        assert config.neo4j_uri, "Neo4j URI is empty"
        
        neo4j_config = config.get_neo4j_config()
        assert 'uri' in neo4j_config, "Neo4j config missing URI"
        
        llm_config = config.get_llm_config()
        assert 'model_name' in llm_config, "LLM config missing model"
        
        return f"Config loaded: Neo4j={config.neo4j_uri}, Model={llm_config['model_name']}"

    def test_query_types_and_strategies(self):
        query_types = list(QueryType)
        expected_types = ['CONSUMPTION', 'EFFICIENCY', 'COMPLIANCE', 'EMISSIONS', 'RISK', 'RECOMMENDATION', 'GENERAL']
        
        for expected in expected_types:
            assert any(qt.value.upper() == expected.lower() or qt.name == expected for qt in query_types), f"Missing QueryType: {expected}"
        
        strategies = list(RetrievalStrategy)
        expected_strategies = ['TEXT2CYPHER', 'VECTOR', 'HYBRID', 'VECTOR_CYPHER', 'HYBRID_CYPHER']
        
        for expected in expected_strategies:
            assert any(rs.name == expected for rs in strategies), f"Missing RetrievalStrategy: {expected}"
        
        return f"Found {len(query_types)} query types, {len(strategies)} strategies"

    def test_ehs_text2cypher_instantiation(self):
        config = {
            "neo4j_uri": self.settings.neo4j_uri,
            "neo4j_user": self.settings.neo4j_username,
            "neo4j_password": self.settings.neo4j_password,
            "openai_api_key": self.settings.openai_api_key or "test_key",
            "model_name": "gpt-4"
        }
        
        retriever = EHSText2CypherRetriever(config)
        
        assert retriever.get_strategy() == RetrievalStrategy.TEXT2CYPHER
        assert hasattr(retriever, 'config')
        assert hasattr(retriever, '_initialized')
        
        return f"Text2Cypher retriever created with strategy: {retriever.get_strategy().value}"

    def test_orchestrator_configuration(self):
        config = OrchestrationConfig(
            max_strategies=3,
            min_confidence_threshold=0.6,
            enable_parallel_execution=True,
            max_execution_time_ms=30000
        )
        
        assert config.max_strategies == 3
        assert config.min_confidence_threshold == 0.6
        assert config.enable_parallel_execution == True
        assert config.max_execution_time_ms == 30000
        
        assert RetrievalStrategy.HYBRID_CYPHER in config.strategy_bias_weights
        assert config.strategy_bias_weights[RetrievalStrategy.HYBRID_CYPHER] == 1.3
        
        return f"Orchestrator config: {config.max_strategies} strategies, {config.max_execution_time_ms}ms timeout"

    async def run_all_tests(self):
        print("🚀 Starting Phase 2 Retriever Architecture Tests")
        print("=" * 60)
        
        test_cases = [
            ("Configuration Loading", self.test_configuration_loading),
            ("Query Types & Strategies", self.test_query_types_and_strategies), 
            ("Text2Cypher Instantiation", self.test_ehs_text2cypher_instantiation),
            ("Orchestrator Configuration", self.test_orchestrator_configuration),
        ]
        
        passed = 0
        for test_name, test_func in test_cases:
            if self.run_test(test_name, test_func):
                passed += 1
        
        total = len(test_cases)
        failed = total - passed
        
        print(f"\n📊 Phase 2 Retriever Test Results")
        print("=" * 50)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {passed/total:.1%}")
        
        if passed >= total * 0.8:
            print("🎉 Phase 2 retrievers are ready for comprehensive testing!")
        else:
            print("❌ Phase 2 retrievers need fixes before comprehensive testing")
        
        return passed == total

async def main():
    test_suite = Phase2TestSuite()
    success = await test_suite.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
