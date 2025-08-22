"""
Comprehensive tests for all Phase 2 retrievers with EHS-specific queries.

This test suite validates:
1. EHS Text2Cypher with examples
2. Vector Retriever with embeddings
3. Hybrid Retriever with combined search
4. VectorCypher with relationship queries
5. HybridCypher with temporal queries
6. Retrieval Orchestrator with strategy selection
7. RAG Agent with full pipeline

Each retriever is tested with real EHS queries covering:
- Water consumption analysis
- Permit compliance tracking
- Safety incident reporting
- Emission trend analysis
- Risk assessment scenarios
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pytest
import pytest_asyncio

# EHS Analytics imports
from ehs_analytics.retrieval.base import QueryType, RetrievalStrategy
from ehs_analytics.retrieval.orchestrator import (
    RetrievalOrchestrator, 
    OrchestrationConfig,
    OrchestrationMode,
    create_ehs_retrieval_orchestrator
)
from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever
from ehs_analytics.retrieval.strategies.vector_retriever import EHSVectorRetriever
from ehs_analytics.retrieval.strategies.hybrid_retriever import EHSHybridRetriever
from ehs_analytics.retrieval.strategies.vector_cypher_retriever import EHSVectorCypherRetriever
from ehs_analytics.retrieval.strategies.hybrid_cypher_retriever import EHSHybridCypherRetriever
from ehs_analytics.agents.rag_agent import RAGAgent, RAGConfiguration, RetrievalMode
from ehs_analytics.agents.query_router import QueryRouterAgent
from ehs_analytics.config import Settings

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EHSTestQuery:
    """Test query with expected attributes."""
    
    query: str
    query_type: QueryType
    description: str
    expected_strategies: List[RetrievalStrategy]
    min_expected_results: int = 1
    max_execution_time_ms: float = 10000
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

@dataclass
class RetrieverTestResult:
    """Result from testing a specific retriever."""
    
    retriever_name: str
    strategy: RetrievalStrategy
    query: str
    success: bool
    execution_time_ms: float
    result_count: int
    confidence_score: float
    error_message: Optional[str] = None
    data_sample: Optional[Dict] = None

@dataclass
class ComprehensiveTestReport:
    """Complete test report for all retrievers."""
    
    test_timestamp: str
    total_queries: int
    total_retrievers: int
    successful_tests: int
    failed_tests: int
    retriever_results: List[RetrieverTestResult]
    performance_metrics: Dict[str, Any]
    integration_status: Dict[str, bool]
    recommendations: List[str]

class Phase2RetrieverTestSuite:
    """Comprehensive test suite for Phase 2 retrievers."""
    
    def __init__(self):
        self.settings = Settings()
        self.test_queries = self._create_ehs_test_queries()
        self.retrievers = {}
        self.orchestrator = None
        self.rag_agent = None
        self.query_router = None
        self.test_results = []
        
    def _create_ehs_test_queries(self) -> List[EHSTestQuery]:
        """Create comprehensive EHS test queries."""
        return [
            # Water Consumption Analysis
            EHSTestQuery(
                query="What is the water consumption for Plant A in Q4 2024?",
                query_type=QueryType.CONSUMPTION,
                description="Water consumption analysis for specific facility and timeframe",
                expected_strategies=[RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR_CYPHER],
                keywords=["water", "consumption", "Plant A", "Q4", "2024"]
            ),
            
            # Permit Compliance
            EHSTestQuery(
                query="Show me all expired permits for manufacturing facilities",
                query_type=QueryType.COMPLIANCE,
                description="Permit compliance check across manufacturing facilities",
                expected_strategies=[RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR],
                keywords=["expired", "permits", "manufacturing", "facilities"]
            ),
            
            # Safety Incidents
            EHSTestQuery(
                query="Find safety incidents related to equipment failures",
                query_type=QueryType.RISK,
                description="Safety incident analysis with equipment failure correlation",
                expected_strategies=[RetrievalStrategy.HYBRID, RetrievalStrategy.VECTOR_CYPHER],
                keywords=["safety", "incidents", "equipment", "failures"]
            ),
            
            # Emission Trends
            EHSTestQuery(
                query="Analyze emission trends over the past year",
                query_type=QueryType.EMISSIONS,
                description="Temporal emission trend analysis",
                expected_strategies=[RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.VECTOR_CYPHER],
                keywords=["emission", "trends", "past year", "analysis"]
            ),
            
            # Risk Assessment
            EHSTestQuery(
                query="Which facilities are at risk of permit violations?",
                query_type=QueryType.RISK,
                description="Risk assessment for permit compliance violations",
                expected_strategies=[RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.HYBRID],
                keywords=["facilities", "risk", "permit", "violations"]
            ),
            
            # Equipment Efficiency
            EHSTestQuery(
                query="What is the energy efficiency of our HVAC systems?",
                query_type=QueryType.EFFICIENCY,
                description="Equipment efficiency analysis for HVAC systems",
                expected_strategies=[RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR],
                keywords=["energy", "efficiency", "HVAC", "systems"]
            ),
            
            # Waste Management
            EHSTestQuery(
                query="How much hazardous waste was generated last month?",
                query_type=QueryType.EMISSIONS,
                description="Hazardous waste generation tracking",
                expected_strategies=[RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR_CYPHER],
                keywords=["hazardous", "waste", "generated", "last month"]
            ),
            
            # Environmental Impact
            EHSTestQuery(
                query="Compare carbon footprint across all facilities",
                query_type=QueryType.EMISSIONS,
                description="Environmental impact comparison across facilities",
                expected_strategies=[RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.VECTOR],
                keywords=["carbon", "footprint", "facilities", "compare"]
            ),
            
            # Regulatory Compliance
            EHSTestQuery(
                query="What are the upcoming regulatory deadlines?",
                query_type=QueryType.COMPLIANCE,
                description="Regulatory deadline tracking and compliance",
                expected_strategies=[RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR],
                keywords=["regulatory", "deadlines", "upcoming", "compliance"]
            ),
            
            # General Inquiry
            EHSTestQuery(
                query="What is our overall EHS performance this quarter?",
                query_type=QueryType.GENERAL,
                description="General EHS performance overview",
                expected_strategies=[RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.HYBRID],
                keywords=["EHS", "performance", "quarter", "overall"]
            )
        ]
    
    async def setup_retrievers(self) -> bool:
        """Setup and initialize all Phase 2 retrievers."""
        try:
            logger.info("Setting up Phase 2 retrievers...")
            
            # Create Neo4j driver for vector retrievers
            from neo4j import GraphDatabase
            neo4j_driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password)
            )
            
            # Create retriever configurations
            retriever_configs = {
                RetrievalStrategy.TEXT2CYPHER: {
                    "neo4j_uri": self.settings.neo4j_uri,
                    "neo4j_user": self.settings.neo4j_username,
                    "neo4j_password": self.settings.neo4j_password,
                    "openai_api_key": self.settings.openai_api_key,
                    "model_name": "gpt-4",
                    "timeout_seconds": 30
                },
                RetrievalStrategy.VECTOR: {
                    "openai_api_key": self.settings.openai_api_key,
                    "embedding_model": "text-embedding-3-small",
                    "vector_store_type": "chroma",
                    "collection_name": "ehs_documents",
                    "similarity_threshold": 0.7
                },
                RetrievalStrategy.HYBRID: {
                    "text2cypher_config": {
                        "neo4j_uri": self.settings.neo4j_uri,
                        "neo4j_user": self.settings.neo4j_username,
                        "neo4j_password": self.settings.neo4j_password,
                        "openai_api_key": self.settings.openai_api_key
                    },
                    "vector_config": {
                        "openai_api_key": self.settings.openai_api_key,
                        "embedding_model": "text-embedding-3-small"
                    },
                    "fusion_strategy": "reciprocal_rank_fusion",
                    "weights": {"graph": 0.6, "vector": 0.4}
                },
                RetrievalStrategy.VECTOR_CYPHER: {
                    "neo4j_uri": self.settings.neo4j_uri,
                    "neo4j_user": self.settings.neo4j_username,
                    "neo4j_password": self.settings.neo4j_password,
                    "openai_api_key": self.settings.openai_api_key,
                    "embedding_model": "text-embedding-3-small",
                    "traversal_depth": 2,
                    "relationship_weights": {
                        "LOCATED_AT": 1.0,
                        "HAS_PERMIT": 1.2,
                        "MONITORS": 0.8,
                        "CONTAINS": 0.9
                    }
                },
                RetrievalStrategy.HYBRID_CYPHER: {
                    "neo4j_uri": self.settings.neo4j_uri,
                    "neo4j_user": self.settings.neo4j_username,
                    "neo4j_password": self.settings.neo4j_password,
                    "openai_api_key": self.settings.openai_api_key,
                    "embedding_model": "text-embedding-3-small",
                    "temporal_analysis": True,
                    "trend_detection": True,
                    "seasonality_analysis": True,
                    "anomaly_detection": True
                }
            }
            
            # Create orchestrator with all retrievers
            self.orchestrator = await create_ehs_retrieval_orchestrator(
                configs=retriever_configs,
                orchestration_config=OrchestrationConfig(
                    max_strategies=3,
                    min_confidence_threshold=0.6,
                    enable_parallel_execution=True,
                    enable_fallback=True,
                    max_execution_time_ms=30000
                )
            )
            
            # Setup individual retrievers for direct testing
            for strategy, config in retriever_configs.items():
                try:
                    if strategy == RetrievalStrategy.TEXT2CYPHER:
                        retriever = EHSText2CypherRetriever(config)
                    elif strategy == RetrievalStrategy.VECTOR:
                        retriever = EHSVectorRetriever(neo4j_driver, config)
                    elif strategy == RetrievalStrategy.HYBRID:
                        retriever = EHSHybridRetriever(neo4j_driver, config)
                    elif strategy == RetrievalStrategy.VECTOR_CYPHER:
                        retriever = EHSVectorCypherRetriever(neo4j_driver, config)
                    elif strategy == RetrievalStrategy.HYBRID_CYPHER:
                        retriever = EHSHybridCypherRetriever(neo4j_driver, config)
                    else:
                        continue
                    
                    await retriever.initialize()
                    self.retrievers[strategy] = retriever
                    logger.info(f"Initialized {strategy.value} retriever")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {strategy.value} retriever: {e}")
                    continue
            
            # Setup RAG Agent
            if self.retrievers:
                rag_retrievers = {
                    strategy.value: retriever 
                    for strategy, retriever in self.retrievers.items()
                }
                
                self.rag_agent = RAGAgent(
                    retrievers=rag_retrievers,
                    config=RAGConfiguration(
                        retrieval_mode=RetrievalMode.PARALLEL,
                        max_retrievers=3,
                        confidence_threshold=0.6,
                        max_context_length=8000,
                        max_response_length=1000
                    )
                )
                
            # Setup Query Router
            # Initialize LLM for query router
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                api_key=self.settings.openai_api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.0
            )
            
            self.query_router = QueryRouterAgent(
                llm=llm,
                temperature=0.0
            )
            # QueryRouterAgent doesn't need initialization
            
            logger.info(f"Successfully initialized {len(self.retrievers)} retrievers")
            return len(self.retrievers) > 0
            
        except Exception as e:
            logger.error(f"Failed to setup retrievers: {e}")
            return False
    
    async def test_individual_retriever(
        self, 
        retriever, 
        strategy: RetrievalStrategy, 
        test_query: EHSTestQuery
    ) -> RetrieverTestResult:
        """Test an individual retriever with a specific query."""
        start_time = time.time()
        
        try:
            logger.info(f"Testing {strategy.value} with query: {test_query.query[:50]}...")
            
            # Execute retrieval
            result = await retriever.retrieve(
                query=test_query.query,
                query_type=test_query.query_type,
                limit=10
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Extract sample data
            data_sample = None
            if result.data and len(result.data) > 0:
                data_sample = {
                    "first_result": str(result.data[0])[:200],
                    "result_type": type(result.data[0]).__name__
                }
            
            return RetrieverTestResult(
                retriever_name=strategy.value,
                strategy=strategy,
                query=test_query.query,
                success=result.success,
                execution_time_ms=execution_time,
                result_count=len(result.data) if result.data else 0,
                confidence_score=result.metadata.confidence_score if result.metadata else 0.0,
                data_sample=data_sample
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error testing {strategy.value}: {e}")
            
            return RetrieverTestResult(
                retriever_name=strategy.value,
                strategy=strategy,
                query=test_query.query,
                success=False,
                execution_time_ms=execution_time,
                result_count=0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def test_orchestrator(self, test_query: EHSTestQuery) -> RetrieverTestResult:
        """Test the retrieval orchestrator."""
        start_time = time.time()
        
        try:
            logger.info(f"Testing Orchestrator with query: {test_query.query[:50]}...")
            
            result = await self.orchestrator.retrieve(
                query=test_query.query,
                query_type=test_query.query_type,
                mode=OrchestrationMode.ADAPTIVE,
                limit=10
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return RetrieverTestResult(
                retriever_name="Orchestrator",
                strategy=RetrievalStrategy.HYBRID_CYPHER,  # Representing orchestrated approach
                query=test_query.query,
                success=len(result.data) > 0,
                execution_time_ms=execution_time,
                result_count=len(result.data),
                confidence_score=result.confidence_score,
                data_sample={
                    "strategies_used": result.source_strategies,
                    "deduplication_info": str(result.deduplication_info)[:100]
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error testing Orchestrator: {e}")
            
            return RetrieverTestResult(
                retriever_name="Orchestrator",
                strategy=RetrievalStrategy.HYBRID_CYPHER,
                query=test_query.query,
                success=False,
                execution_time_ms=execution_time,
                result_count=0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def test_rag_agent(self, test_query: EHSTestQuery) -> RetrieverTestResult:
        """Test the full RAG pipeline."""
        start_time = time.time()
        
        try:
            logger.info(f"Testing RAG Agent with query: {test_query.query[:50]}...")
            
            # First classify the query
            classification = await self.query_router.classify_query(test_query.query)
            
            # Process through RAG agent
            rag_result = await self.rag_agent.process_query(
                query_id=f"test_{int(time.time())}",
                query=test_query.query,
                classification=classification
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return RetrieverTestResult(
                retriever_name="RAG_Agent",
                strategy=RetrievalStrategy.HYBRID_CYPHER,  # Representing full pipeline
                query=test_query.query,
                success=rag_result.success,
                execution_time_ms=execution_time,
                result_count=rag_result.source_count,
                confidence_score=rag_result.confidence_score,
                data_sample={
                    "response_preview": rag_result.response.content[:200],
                    "retrievers_used": rag_result.retrievers_used,
                    "classification": classification.intent_type.value
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error testing RAG Agent: {e}")
            
            return RetrieverTestResult(
                retriever_name="RAG_Agent",
                strategy=RetrievalStrategy.HYBRID_CYPHER,
                query=test_query.query,
                success=False,
                execution_time_ms=execution_time,
                result_count=0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def run_comprehensive_tests(self) -> ComprehensiveTestReport:
        """Run comprehensive tests on all retrievers with all queries."""
        logger.info("Starting comprehensive Phase 2 retriever tests...")
        
        start_time = datetime.utcnow()
        all_results = []
        
        # Test each retriever with each query
        for test_query in self.test_queries:
            logger.info(f"\n=== Testing Query: {test_query.description} ===")
            
            # Test individual retrievers
            for strategy, retriever in self.retrievers.items():
                result = await self.test_individual_retriever(retriever, strategy, test_query)
                all_results.append(result)
            
            # Test orchestrator
            if self.orchestrator:
                orchestrator_result = await self.test_orchestrator(test_query)
                all_results.append(orchestrator_result)
            
            # Test RAG agent
            if self.rag_agent and self.query_router:
                rag_result = await self.test_rag_agent(test_query)
                all_results.append(rag_result)
        
        # Calculate performance metrics
        successful_tests = sum(1 for r in all_results if r.success)
        failed_tests = len(all_results) - successful_tests
        
        performance_metrics = {
            "avg_execution_time_ms": sum(r.execution_time_ms for r in all_results) / len(all_results),
            "avg_confidence_score": sum(r.confidence_score for r in all_results if r.success) / max(successful_tests, 1),
            "avg_result_count": sum(r.result_count for r in all_results if r.success) / max(successful_tests, 1),
            "success_rate": successful_tests / len(all_results) if all_results else 0,
            "retriever_performance": {}
        }
        
        # Calculate per-retriever performance
        retriever_names = set(r.retriever_name for r in all_results)
        for retriever_name in retriever_names:
            retriever_results = [r for r in all_results if r.retriever_name == retriever_name]
            successful_retriever = [r for r in retriever_results if r.success]
            
            performance_metrics["retriever_performance"][retriever_name] = {
                "success_rate": len(successful_retriever) / len(retriever_results),
                "avg_execution_time_ms": sum(r.execution_time_ms for r in retriever_results) / len(retriever_results),
                "avg_confidence": sum(r.confidence_score for r in successful_retriever) / max(len(successful_retriever), 1),
                "avg_results": sum(r.result_count for r in successful_retriever) / max(len(successful_retriever), 1)
            }
        
        # Check integration status
        integration_status = {
            "neo4j_connection": any(r.success for r in all_results if "Text2Cypher" in r.retriever_name),
            "openai_integration": any(r.success for r in all_results if r.confidence_score > 0),
            "vector_store": any(r.success for r in all_results if "Vector" in r.retriever_name),
            "orchestrator": any(r.success for r in all_results if r.retriever_name == "Orchestrator"),
            "rag_pipeline": any(r.success for r in all_results if r.retriever_name == "RAG_Agent")
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results, performance_metrics, integration_status)
        
        return ComprehensiveTestReport(
            test_timestamp=start_time.isoformat(),
            total_queries=len(self.test_queries),
            total_retrievers=len(self.retrievers) + 2,  # +2 for orchestrator and RAG agent
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            retriever_results=all_results,
            performance_metrics=performance_metrics,
            integration_status=integration_status,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self, 
        results: List[RetrieverTestResult],
        metrics: Dict[str, Any],
        integration: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Success rate recommendations
        if metrics["success_rate"] < 0.7:
            recommendations.append("Overall success rate is below 70%. Review failed retrievers and fix configuration issues.")
        
        # Performance recommendations
        if metrics["avg_execution_time_ms"] > 5000:
            recommendations.append("Average execution time exceeds 5 seconds. Consider optimizing database queries and API calls.")
        
        # Integration recommendations
        if not integration["neo4j_connection"]:
            recommendations.append("Neo4j connection failed. Verify database credentials and network connectivity.")
        
        if not integration["openai_integration"]:
            recommendations.append("OpenAI integration failed. Check API key and rate limits.")
        
        if not integration["vector_store"]:
            recommendations.append("Vector store integration failed. Verify embedding service and vector database setup.")
        
        # Retriever-specific recommendations
        for retriever_name, perf in metrics["retriever_performance"].items():
            if perf["success_rate"] < 0.5:
                recommendations.append(f"{retriever_name} has low success rate ({perf['success_rate']:.1%}). Investigate configuration and dependencies.")
            
            if perf["avg_execution_time_ms"] > 10000:
                recommendations.append(f"{retriever_name} is slow (avg {perf['avg_execution_time_ms']:.0f}ms). Optimize queries and caching.")
        
        # Quality recommendations
        if metrics["avg_confidence_score"] < 0.6:
            recommendations.append("Low average confidence scores. Review query examples and improve retrieval relevance.")
        
        return recommendations
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "retrievers": {},
            "orchestrator": None,
            "rag_agent": None,
            "query_router": None
        }
        
        # Check individual retrievers
        for strategy, retriever in self.retrievers.items():
            try:
                health = await retriever.health_check()
                health_status["retrievers"][strategy.value] = health
            except Exception as e:
                health_status["retrievers"][strategy.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check orchestrator
        if self.orchestrator:
            try:
                health_status["orchestrator"] = await self.orchestrator.health_check()
            except Exception as e:
                health_status["orchestrator"] = {"status": "unhealthy", "error": str(e)}
        
        # Check RAG agent
        if self.rag_agent:
            try:
                health_status["rag_agent"] = await self.rag_agent.health_check()
            except Exception as e:
                health_status["rag_agent"] = {"status": "unhealthy", "error": str(e)}
        
        # Check query router
        if self.query_router:
            try:
                # Simple test classification
                test_result = await self.query_router.classify_query("test query")
                health_status["query_router"] = {
                    "status": "healthy" if test_result else "unhealthy"
                }
            except Exception as e:
                health_status["query_router"] = {"status": "unhealthy", "error": str(e)}
        
        return health_status
    
    def save_test_report(self, report: ComprehensiveTestReport, filename: str = None) -> str:
        """Save test report to file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"phase2_retriever_test_report_{timestamp}.json"
        
        filepath = os.path.join("tests", "phase2_retrievers", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to JSON-serializable format
        report_dict = asdict(report)
        
        # Convert enums to strings
        for result in report_dict["retriever_results"]:
            if "strategy" in result and hasattr(result["strategy"], "value"):
                result["strategy"] = result["strategy"].value
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Test report saved to: {filepath}")
        return filepath


async def main():
    """Main test execution function."""
    print("🚀 Starting Comprehensive Phase 2 Retriever Tests")
    print("=" * 60)
    
    test_suite = Phase2RetrieverTestSuite()
    
    # Setup retrievers
    print("📋 Setting up retrievers...")
    setup_success = await test_suite.setup_retrievers()
    
    if not setup_success:
        print("❌ Failed to setup retrievers. Exiting.")
        return
    
    print(f"✅ Successfully initialized {len(test_suite.retrievers)} retrievers")
    
    # Perform health check
    print("\n🏥 Performing health checks...")
    health_status = await test_suite.health_check_all()
    healthy_retrievers = sum(
        1 for status in health_status["retrievers"].values()
        if status.get("status") == "healthy"
    )
    print(f"✅ Health check complete: {healthy_retrievers}/{len(health_status['retrievers'])} retrievers healthy")
    
    # Run comprehensive tests
    print("\n🧪 Running comprehensive tests...")
    test_report = await test_suite.run_comprehensive_tests()
    
    # Save test report
    report_file = test_suite.save_test_report(test_report)
    
    # Print summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    print(f"Total Tests: {test_report.successful_tests + test_report.failed_tests}")
    print(f"✅ Successful: {test_report.successful_tests}")
    print(f"❌ Failed: {test_report.failed_tests}")
    print(f"📈 Success Rate: {test_report.performance_metrics['success_rate']:.1%}")
    print(f"⏱️  Avg Execution Time: {test_report.performance_metrics['avg_execution_time_ms']:.0f}ms")
    print(f"🎯 Avg Confidence: {test_report.performance_metrics['avg_confidence_score']:.2f}")
    
    print("\n🔧 Integration Status")
    print("-" * 20)
    for component, status in test_report.integration_status.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component.replace('_', ' ').title()}: {'Working' if status else 'Failed'}")
    
    print("\n📈 Retriever Performance")
    print("-" * 25)
    for retriever, perf in test_report.performance_metrics["retriever_performance"].items():
        print(f"{retriever}:")
        print(f"  Success Rate: {perf['success_rate']:.1%}")
        print(f"  Avg Time: {perf['avg_execution_time_ms']:.0f}ms")
        print(f"  Avg Confidence: {perf['avg_confidence']:.2f}")
        print(f"  Avg Results: {perf['avg_results']:.1f}")
    
    if test_report.recommendations:
        print("\n💡 Recommendations")
        print("-" * 15)
        for i, rec in enumerate(test_report.recommendations, 1):
            print(f"{i}. {rec}")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    print("\n🎉 Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())