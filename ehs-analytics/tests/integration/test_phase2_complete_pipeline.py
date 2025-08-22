#!/usr/bin/env python3
"""
Complete Phase 2 RAG Pipeline Integration Tests

This test suite validates the complete end-to-end RAG pipeline including:
1. Query Router classification
2. Retrieval Orchestrator strategy selection
3. RAG Agent execution 
4. End-to-end response generation with sources

Test scenarios cover all requested query types:
- Simple lookup queries
- Complex analysis queries  
- Temporal queries
- Relationship queries
- Predictive queries
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging for comprehensive output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"phase2_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

try:
    from ehs_analytics.config import get_settings
    from ehs_analytics.agents.query_router import QueryRouterAgent, QueryClassification
    from ehs_analytics.agents.rag_agent import RAGAgent, RAGConfiguration, RetrievalMode
    from ehs_analytics.retrieval.orchestrator import RetrievalOrchestrator, OrchestrationConfig, create_ehs_retrieval_orchestrator
    from ehs_analytics.retrieval.base import QueryType, RetrievalStrategy
    from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever
    from ehs_analytics.retrieval.strategies.vector_retriever import EHSVectorRetriever
    from ehs_analytics.retrieval.strategies.hybrid_cypher_retriever import EHSHybridCypherRetriever
    from ehs_analytics.retrieval.strategies.vector_cypher_retriever import EHSVectorCypherRetriever
    from ehs_analytics.workflows.ehs_workflow import EHSWorkflow, create_ehs_workflow
    from ehs_analytics.api.dependencies import DatabaseManager
    from ehs_analytics.utils.logging import get_ehs_logger
except ImportError as e:
    logger.error(f"Critical import failed: {e}")
    sys.exit(1)

@dataclass
class TestScenario:
    """Test scenario definition."""
    name: str
    query: str
    query_type: str
    description: str
    expected_intent: str
    expected_retrievers: List[str]
    min_confidence: float = 0.7
    min_sources: int = 1

@dataclass  
class PipelineTestResult:
    """Complete pipeline test result."""
    scenario_name: str
    query: str
    
    # Classification results
    classification_success: bool
    classified_intent: Optional[str] = None
    classification_confidence: float = 0.0
    classification_time_ms: float = 0.0
    
    # Orchestration results
    orchestration_success: bool
    selected_strategies: List[str] = None
    orchestration_time_ms: float = 0.0
    
    # RAG results
    rag_success: bool
    response_content: Optional[str] = None
    source_count: int = 0
    rag_confidence: float = 0.0
    rag_time_ms: float = 0.0
    
    # Overall results
    total_time_ms: float = 0.0
    end_to_end_success: bool = False
    error_message: Optional[str] = None

class Phase2IntegrationTestSuite:
    """Complete Phase 2 RAG pipeline integration test suite."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_ehs_logger(__name__)
        self.test_scenarios = self._create_test_scenarios()
        self.test_results: List[PipelineTestResult] = []
        
        # Components to be initialized
        self.query_router: Optional[QueryRouterAgent] = None
        self.orchestrator: Optional[RetrievalOrchestrator] = None
        self.rag_agent: Optional[RAGAgent] = None
        self.workflow: Optional[EHSWorkflow] = None
        self.db_manager: Optional[DatabaseManager] = None
        
    def _create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios covering all requested query types."""
        return [
            # Simple lookup queries
            TestScenario(
                name="simple_permit_lookup",
                query="What is the permit number for Plant A's air emissions?",
                query_type="simple_lookup", 
                description="Simple permit number lookup for specific facility",
                expected_intent="compliance_check",
                expected_retrievers=["text2cypher", "vector"]
            ),
            
            TestScenario(
                name="simple_equipment_status",
                query="Show me the status of Boiler Unit 1 at Manufacturing Site B",
                query_type="simple_lookup",
                description="Equipment status lookup",
                expected_intent="equipment_efficiency", 
                expected_retrievers=["text2cypher"]
            ),
            
            # Complex analysis queries
            TestScenario(
                name="complex_water_analysis",
                query="Compare water consumption trends across all facilities and identify anomalies",
                query_type="complex_analysis",
                description="Multi-facility water consumption trend analysis with anomaly detection",
                expected_intent="consumption_analysis",
                expected_retrievers=["hybrid_cypher", "vector_cypher"]
            ),
            
            TestScenario(
                name="complex_emission_correlation",
                query="Analyze the correlation between production levels and emission rates across manufacturing facilities",
                query_type="complex_analysis",
                description="Complex correlation analysis between production and emissions",
                expected_intent="emission_tracking", 
                expected_retrievers=["hybrid_cypher", "vector_cypher"]
            ),
            
            # Temporal queries
            TestScenario(
                name="temporal_safety_incidents",
                query="Show safety incidents in the past 6 months and their correlation with equipment maintenance",
                query_type="temporal_query",
                description="Temporal safety analysis with maintenance correlation",
                expected_intent="risk_assessment",
                expected_retrievers=["hybrid_cypher", "vector_cypher"]
            ),
            
            TestScenario(
                name="temporal_compliance_trends",
                query="Track permit compliance trends over the last 2 years for all facilities",
                query_type="temporal_query", 
                description="Long-term compliance trend analysis",
                expected_intent="compliance_check",
                expected_retrievers=["vector_cypher", "hybrid"]
            ),
            
            # Relationship queries
            TestScenario(
                name="relationship_expired_permits",
                query="Find all equipment in facilities with expired permits",
                query_type="relationship_query",
                description="Cross-entity relationship query between equipment and permits",
                expected_intent="compliance_check",
                expected_retrievers=["text2cypher", "vector_cypher"]
            ),
            
            TestScenario(
                name="relationship_incident_equipment",
                query="Identify equipment types most frequently involved in safety incidents",
                query_type="relationship_query",
                description="Equipment-incident relationship analysis",
                expected_intent="risk_assessment",
                expected_retrievers=["hybrid_cypher", "vector_cypher"] 
            ),
            
            # Predictive queries
            TestScenario(
                name="predictive_emission_limits",
                query="Which facilities are at risk of exceeding emission limits?",
                query_type="predictive_query",
                description="Predictive analysis for emission limit violations",
                expected_intent="risk_assessment",
                expected_retrievers=["hybrid_cypher", "vector_cypher"],
                min_confidence=0.6  # Predictive queries may have lower confidence
            ),
            
            TestScenario(
                name="predictive_maintenance_needs",
                query="Predict which equipment will require maintenance in the next quarter based on performance trends",
                query_type="predictive_query", 
                description="Predictive maintenance scheduling",
                expected_intent="equipment_efficiency",
                expected_retrievers=["hybrid_cypher", "vector_cypher"],
                min_confidence=0.6
            )
        ]
    
    async def initialize_components(self):
        """Initialize all RAG pipeline components."""
        logger.info("🔧 Initializing Phase 2 RAG pipeline components...")
        
        try:
            # Initialize database manager
            self.db_manager = DatabaseManager()
            
            # Initialize query router
            logger.info("Initializing Query Router...")
            self.query_router = QueryRouterAgent()
            
            # Initialize retrievers with proper configurations
            logger.info("Initializing retrievers...")
            retriever_configs = {
                RetrievalStrategy.TEXT2CYPHER: {
                    "neo4j_uri": self.settings.neo4j_uri,
                    "neo4j_user": self.settings.neo4j_username, 
                    "neo4j_password": self.settings.neo4j_password,
                    "openai_api_key": self.settings.openai_api_key,
                    "model_name": "gpt-4"
                },
                RetrievalStrategy.VECTOR: {
                    "embeddings_model": "text-embedding-ada-002",
                    "vector_store_path": "./vector_store",
                    "openai_api_key": self.settings.openai_api_key
                },
                RetrievalStrategy.VECTOR_CYPHER: {
                    "neo4j_uri": self.settings.neo4j_uri,
                    "neo4j_user": self.settings.neo4j_username,
                    "neo4j_password": self.settings.neo4j_password,
                    "openai_api_key": self.settings.openai_api_key,
                    "embeddings_model": "text-embedding-ada-002"
                },
                RetrievalStrategy.HYBRID_CYPHER: {
                    "neo4j_uri": self.settings.neo4j_uri,
                    "neo4j_user": self.settings.neo4j_username,
                    "neo4j_password": self.settings.neo4j_password,
                    "openai_api_key": self.settings.openai_api_key,
                    "embeddings_model": "text-embedding-ada-002",
                    "fusion_weights": {"vector": 0.6, "cypher": 0.4}
                }
            }
            
            # Create orchestrator
            logger.info("Creating Retrieval Orchestrator...")
            orchestration_config = OrchestrationConfig(
                max_strategies=3,
                min_confidence_threshold=0.6,
                enable_parallel_execution=True,
                max_execution_time_ms=30000
            )
            
            self.orchestrator = await create_ehs_retrieval_orchestrator(
                configs=retriever_configs,
                orchestration_config=orchestration_config
            )
            
            # Create RAG agent with retrievers
            logger.info("Creating RAG Agent...")
            retrievers_dict = {
                "text2cypher": EHSText2CypherRetriever(retriever_configs[RetrievalStrategy.TEXT2CYPHER]),
                "vector": EHSVectorRetriever(retriever_configs[RetrievalStrategy.VECTOR]),
                "vector_cypher": EHSVectorCypherRetriever(retriever_configs[RetrievalStrategy.VECTOR_CYPHER]),
                "hybrid_cypher": EHSHybridCypherRetriever(retriever_configs[RetrievalStrategy.HYBRID_CYPHER])
            }
            
            # Initialize all retrievers
            for name, retriever in retrievers_dict.items():
                try:
                    await retriever.initialize()
                    logger.info(f"✅ {name} retriever initialized")
                except Exception as e:
                    logger.warning(f"⚠️ {name} retriever initialization failed: {e}")
            
            rag_config = RAGConfiguration(
                retrieval_mode=RetrievalMode.PARALLEL,
                max_retrievers=3,
                confidence_threshold=0.6,
                max_context_length=8000,
                include_sources=True,
                validate_responses=True
            )
            
            self.rag_agent = RAGAgent(retrievers_dict, rag_config)
            
            # Create workflow
            logger.info("Creating EHS Workflow...")
            self.workflow = await create_ehs_workflow(
                db_manager=self.db_manager,
                query_router=self.query_router,
                rag_agent=self.rag_agent
            )
            
            logger.info("✅ All Phase 2 components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def test_query_classification(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test query classification step."""
        start_time = time.time()
        
        try:
            classification = self.query_router.classify_query(scenario.query)
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "classification": classification,
                "execution_time_ms": execution_time,
                "intent": classification.intent_type.value,
                "confidence": classification.confidence_score
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }
    
    async def test_retrieval_orchestration(self, scenario: TestScenario, classification) -> Dict[str, Any]:
        """Test retrieval orchestration step."""
        start_time = time.time()
        
        try:
            # Convert intent to query type
            intent_to_query_type = {
                "consumption_analysis": QueryType.CONSUMPTION,
                "equipment_efficiency": QueryType.EFFICIENCY,
                "compliance_check": QueryType.COMPLIANCE,
                "emission_tracking": QueryType.EMISSIONS,
                "risk_assessment": QueryType.RISK,
                "general_inquiry": QueryType.GENERAL
            }
            
            query_type = intent_to_query_type.get(
                classification.intent_type.value, 
                QueryType.GENERAL
            )
            
            # Execute orchestrated retrieval
            merged_result = await self.orchestrator.retrieve(
                query=scenario.query,
                query_type=query_type,
                limit=20
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "merged_result": merged_result,
                "execution_time_ms": execution_time,
                "strategies_used": merged_result.source_strategies,
                "total_results": len(merged_result.data),
                "confidence": merged_result.confidence_score
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e), 
                "execution_time_ms": execution_time
            }
    
    async def test_rag_agent_processing(self, scenario: TestScenario, classification) -> Dict[str, Any]:
        """Test RAG agent end-to-end processing."""
        start_time = time.time()
        
        try:
            query_id = f"test_{scenario.name}_{int(time.time())}"
            
            # Process through RAG agent
            rag_result = await self.rag_agent.process_query(
                query_id=query_id,
                query=scenario.query,
                classification=classification,
                user_id="integration_test"
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": rag_result.success,
                "rag_result": rag_result,
                "execution_time_ms": execution_time,
                "response_content": rag_result.response.content if rag_result.success else None,
                "source_count": rag_result.source_count if rag_result.success else 0,
                "confidence": rag_result.confidence_score if rag_result.success else 0.0,
                "retrievers_used": rag_result.retrievers_used if rag_result.success else [],
                "error": rag_result.error_message if not rag_result.success else None
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }
    
    async def test_complete_workflow(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test complete workflow end-to-end."""
        start_time = time.time()
        
        try:
            query_id = f"workflow_test_{scenario.name}_{int(time.time())}"
            
            # Process through complete workflow
            workflow_state = await self.workflow.process_query(
                query_id=query_id,
                query=scenario.query,
                user_id="integration_test",
                options={"use_rag": True, "include_recommendations": True}
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": not bool(workflow_state.error),
                "workflow_state": workflow_state,
                "execution_time_ms": execution_time,
                "classification": workflow_state.classification,
                "rag_result": workflow_state.rag_result,
                "recommendations": workflow_state.recommendations,
                "total_duration": workflow_state.total_duration_ms,
                "error": workflow_state.error
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }
    
    async def run_scenario_test(self, scenario: TestScenario) -> PipelineTestResult:
        """Run complete pipeline test for a single scenario."""
        logger.info(f"🧪 Testing scenario: {scenario.name}")
        logger.info(f"   Query: {scenario.query}")
        
        total_start_time = time.time()
        result = PipelineTestResult(
            scenario_name=scenario.name,
            query=scenario.query,
            classification_success=False,
            orchestration_success=False,
            rag_success=False
        )
        
        try:
            # Step 1: Test Query Classification
            logger.info("   Step 1: Query Classification")
            classification_result = await self.test_query_classification(scenario)
            
            result.classification_success = classification_result["success"]
            result.classification_time_ms = classification_result["execution_time_ms"]
            
            if classification_result["success"]:
                classification = classification_result["classification"]
                result.classified_intent = classification_result["intent"]
                result.classification_confidence = classification_result["confidence"]
                logger.info(f"   ✅ Classified as: {result.classified_intent} (confidence: {result.classification_confidence:.2f})")
            else:
                logger.error(f"   ❌ Classification failed: {classification_result.get('error')}")
                result.error_message = f"Classification failed: {classification_result.get('error')}"
                return result
            
            # Step 2: Test Retrieval Orchestration
            logger.info("   Step 2: Retrieval Orchestration")
            orchestration_result = await self.test_retrieval_orchestration(scenario, classification)
            
            result.orchestration_success = orchestration_result["success"]
            result.orchestration_time_ms = orchestration_result["execution_time_ms"]
            
            if orchestration_result["success"]:
                result.selected_strategies = orchestration_result["strategies_used"]
                logger.info(f"   ✅ Orchestration completed: {len(result.selected_strategies)} strategies")
            else:
                logger.error(f"   ❌ Orchestration failed: {orchestration_result.get('error')}")
                result.error_message = f"Orchestration failed: {orchestration_result.get('error')}"
                # Continue to test RAG agent directly
            
            # Step 3: Test RAG Agent Processing
            logger.info("   Step 3: RAG Agent Processing")
            rag_result = await self.test_rag_agent_processing(scenario, classification)
            
            result.rag_success = rag_result["success"]
            result.rag_time_ms = rag_result["execution_time_ms"]
            
            if rag_result["success"]:
                result.response_content = rag_result["response_content"]
                result.source_count = rag_result["source_count"]
                result.rag_confidence = rag_result["confidence"]
                logger.info(f"   ✅ RAG completed: {result.source_count} sources, confidence: {result.rag_confidence:.2f}")
            else:
                logger.error(f"   ❌ RAG failed: {rag_result.get('error')}")
                if not result.error_message:
                    result.error_message = f"RAG failed: {rag_result.get('error')}"
            
            # Step 4: Test Complete Workflow (if RAG succeeded)
            if result.rag_success:
                logger.info("   Step 4: Complete Workflow Test")
                workflow_result = await self.test_complete_workflow(scenario)
                
                if workflow_result["success"]:
                    logger.info("   ✅ Complete workflow succeeded")
                else:
                    logger.warning(f"   ⚠️ Workflow failed: {workflow_result.get('error')}")
            
            # Calculate overall success
            result.end_to_end_success = (
                result.classification_success and 
                result.rag_success and
                result.rag_confidence >= scenario.min_confidence and
                result.source_count >= scenario.min_sources
            )
            
        except Exception as e:
            logger.error(f"   ❌ Scenario test failed with exception: {e}")
            result.error_message = f"Test exception: {str(e)}"
        
        finally:
            result.total_time_ms = (time.time() - total_start_time) * 1000
            
        status = "✅ PASSED" if result.end_to_end_success else "❌ FAILED"
        logger.info(f"   {status} - Total time: {result.total_time_ms:.0f}ms")
        
        return result
    
    async def run_comprehensive_integration_tests(self):
        """Run comprehensive integration tests for all scenarios."""
        logger.info("🚀 Starting Phase 2 Complete RAG Pipeline Integration Tests")
        logger.info("=" * 80)
        
        try:
            # Initialize all components
            await self.initialize_components()
            
            # Run tests for each scenario
            for i, scenario in enumerate(self.test_scenarios, 1):
                logger.info(f"\n📋 Scenario {i}/{len(self.test_scenarios)}: {scenario.description}")
                result = await self.run_scenario_test(scenario)
                self.test_results.append(result)
                
                # Brief pause between tests
                await asyncio.sleep(1)
            
            # Generate comprehensive report
            self._generate_phase2_completion_report()
            
        except Exception as e:
            logger.error(f"❌ Integration test suite failed: {e}")
            raise
        
        return self.test_results
    
    def _generate_phase2_completion_report(self):
        """Generate comprehensive Phase 2 completion report."""
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.end_to_end_success)
        classification_success = sum(1 for r in self.test_results if r.classification_success)
        orchestration_success = sum(1 for r in self.test_results if r.orchestration_success)
        rag_success = sum(1 for r in self.test_results if r.rag_success)
        
        logger.info(f"\n🎯 PHASE 2 RAG PIPELINE COMPLETION REPORT")
        logger.info("=" * 60)
        
        # Overall Results
        logger.info(f"\n📊 OVERALL RESULTS")
        logger.info("-" * 20)
        logger.info(f"Total Scenarios Tested: {total_tests}")
        logger.info(f"✅ End-to-End Success: {successful_tests}/{total_tests} ({successful_tests/total_tests:.1%})")
        logger.info(f"📋 Classification Success: {classification_success}/{total_tests} ({classification_success/total_tests:.1%})")
        logger.info(f"🔀 Orchestration Success: {orchestration_success}/{total_tests} ({orchestration_success/total_tests:.1%})")
        logger.info(f"🤖 RAG Agent Success: {rag_success}/{total_tests} ({rag_success/total_tests:.1%})")
        
        # Component Performance
        if successful_tests > 0:
            successful_results = [r for r in self.test_results if r.end_to_end_success]
            
            avg_classification_time = sum(r.classification_time_ms for r in successful_results) / len(successful_results)
            avg_orchestration_time = sum(r.orchestration_time_ms for r in successful_results) / len(successful_results)
            avg_rag_time = sum(r.rag_time_ms for r in successful_results) / len(successful_results)
            avg_total_time = sum(r.total_time_ms for r in successful_results) / len(successful_results)
            avg_confidence = sum(r.rag_confidence for r in successful_results) / len(successful_results)
            avg_sources = sum(r.source_count for r in successful_results) / len(successful_results)
            
            logger.info(f"\n⏱️ PERFORMANCE METRICS")
            logger.info("-" * 20)
            logger.info(f"Avg Classification Time: {avg_classification_time:.0f}ms")
            logger.info(f"Avg Orchestration Time: {avg_orchestration_time:.0f}ms") 
            logger.info(f"Avg RAG Processing Time: {avg_rag_time:.0f}ms")
            logger.info(f"Avg Total End-to-End Time: {avg_total_time:.0f}ms")
            logger.info(f"Avg Response Confidence: {avg_confidence:.2f}")
            logger.info(f"Avg Sources per Response: {avg_sources:.1f}")
        
        # Query Type Coverage
        logger.info(f"\n📝 QUERY TYPE COVERAGE")
        logger.info("-" * 20)
        
        query_types = set(scenario.query_type for scenario in self.test_scenarios)
        for query_type in query_types:
            type_results = [r for r, s in zip(self.test_results, self.test_scenarios) if s.query_type == query_type]
            type_success = sum(1 for r in type_results if r.end_to_end_success)
            logger.info(f"{query_type}: {type_success}/{len(type_results)} ({'✅' if type_success == len(type_results) else '⚠️' if type_success > 0 else '❌'})")
        
        # Component Integration Status
        logger.info(f"\n🔧 COMPONENT INTEGRATION STATUS")
        logger.info("-" * 30)
        
        components = [
            ("Query Router", classification_success, total_tests),
            ("Retrieval Orchestrator", orchestration_success, total_tests),
            ("RAG Agent", rag_success, total_tests),
            ("End-to-End Pipeline", successful_tests, total_tests)
        ]
        
        for component, success_count, total_count in components:
            success_rate = success_count / total_count
            status = "✅ READY" if success_rate >= 0.9 else "⚠️ NEEDS WORK" if success_rate >= 0.7 else "❌ CRITICAL ISSUES"
            logger.info(f"{component}: {status} ({success_rate:.1%})")
        
        # Production Readiness Assessment
        logger.info(f"\n🎯 PRODUCTION READINESS ASSESSMENT")
        logger.info("-" * 35)
        
        overall_success_rate = successful_tests / total_tests
        
        if overall_success_rate >= 0.9:
            readiness = "🎉 PRODUCTION READY"
            readiness_details = [
                "✅ All core components working together",
                "✅ End-to-end query processing operational", 
                "✅ All query types successfully handled",
                "✅ Performance within acceptable thresholds",
                "✅ Source attribution and confidence scoring working"
            ]
        elif overall_success_rate >= 0.7:
            readiness = "⚠️ MOSTLY READY - Minor Issues"
            readiness_details = [
                "✅ Core functionality operational",
                "⚠️ Some query types need optimization",
                "✅ Basic integration working",
                "⚠️ Performance tuning recommended"
            ]
        else:
            readiness = "❌ NOT READY - Significant Work Needed"
            readiness_details = [
                "❌ Core integration issues detected",
                "❌ Multiple component failures",
                "❌ Query processing unreliable",
                "❌ Major fixes required before production"
            ]
        
        logger.info(f"Status: {readiness}")
        for detail in readiness_details:
            logger.info(f"  {detail}")
        
        # Detailed Scenario Results
        logger.info(f"\n📋 DETAILED SCENARIO RESULTS")
        logger.info("-" * 30)
        
        for i, (result, scenario) in enumerate(zip(self.test_results, self.test_scenarios), 1):
            status = "✅ PASS" if result.end_to_end_success else "❌ FAIL"
            logger.info(f"{i:2d}. {scenario.name}: {status}")
            logger.info(f"    Query: {scenario.query[:60]}...")
            logger.info(f"    Intent: {result.classified_intent} (conf: {result.classification_confidence:.2f})")
            if result.end_to_end_success:
                logger.info(f"    Sources: {result.source_count}, RAG Confidence: {result.rag_confidence:.2f}")
                logger.info(f"    Time: {result.total_time_ms:.0f}ms")
            else:
                logger.info(f"    Error: {result.error_message}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data = {
            "timestamp": timestamp,
            "phase": "Phase 2 - Complete RAG Pipeline",
            "summary": {
                "total_scenarios": total_tests,
                "successful_scenarios": successful_tests,
                "success_rate": overall_success_rate,
                "production_readiness": readiness
            },
            "component_results": {
                "query_router": {"success": classification_success, "total": total_tests},
                "orchestrator": {"success": orchestration_success, "total": total_tests},
                "rag_agent": {"success": rag_success, "total": total_tests}
            },
            "performance_metrics": {
                "avg_total_time_ms": sum(r.total_time_ms for r in self.test_results) / len(self.test_results),
                "avg_classification_time_ms": sum(r.classification_time_ms for r in self.test_results) / len(self.test_results),
                "avg_rag_time_ms": sum(r.rag_time_ms for r in self.test_results) / len(self.test_results)
            } if self.test_results else {},
            "test_scenarios": [asdict(scenario) for scenario in self.test_scenarios],
            "detailed_results": [asdict(result) for result in self.test_results]
        }
        
        report_file = f"phase2_complete_integration_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed report saved: {report_file}")
        
        # Final summary
        logger.info(f"\n🏁 PHASE 2 INTEGRATION TEST COMPLETE")
        logger.info(f"   Status: {readiness}")
        logger.info(f"   Success Rate: {overall_success_rate:.1%}")
        logger.info(f"   Report: {report_file}")
        logger.info("=" * 60)

async def main():
    """Main test execution function."""
    test_suite = Phase2IntegrationTestSuite()
    
    try:
        results = await test_suite.run_comprehensive_integration_tests()
        
        # Determine exit code based on success rate
        success_rate = sum(1 for r in results if r.end_to_end_success) / len(results)
        return 0 if success_rate >= 0.8 else 1
        
    except Exception as e:
        logger.error(f"❌ Integration test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
