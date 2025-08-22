#!/usr/bin/env python3
"""
EHS Analytics Retrieval Orchestrator Demo

This script demonstrates how to use the Retrieval Orchestrator to intelligently
coordinate multiple retrieval strategies for EHS queries.
"""

import asyncio
import logging
from typing import Dict, Any

from src.ehs_analytics.retrieval import (
    RetrievalOrchestrator,
    OrchestrationConfig,
    OrchestrationMode,
    RetrievalStrategy,
    QueryType,
    create_ehs_retrieval_orchestrator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main demonstration function."""
    
    logger.info("Starting EHS Analytics Retrieval Orchestrator Demo")
    
    # Configuration for different retrieval strategies
    retriever_configs = {
        RetrievalStrategy.TEXT2CYPHER: {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "openai_api_key": "your-openai-key"
        },
        RetrievalStrategy.VECTOR: {
            "vector_store_type": "chroma",
            "embedding_model": "openai",
            "collection_name": "ehs_documents",
            "openai_api_key": "your-openai-key"
        },
        RetrievalStrategy.HYBRID: {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j", 
            "neo4j_password": "password",
            "vector_store_type": "chroma",
            "embedding_model": "openai",
            "openai_api_key": "your-openai-key"
        }
    }
    
    # Orchestration configuration
    orchestration_config = OrchestrationConfig(
        max_strategies=3,
        min_confidence_threshold=0.6,
        enable_parallel_execution=True,
        enable_caching=True,
        cache_ttl_seconds=300
    )
    
    try:
        # Create the orchestrator
        logger.info("Creating retrieval orchestrator...")
        orchestrator = await create_ehs_retrieval_orchestrator(
            configs=retriever_configs,
            orchestration_config=orchestration_config
        )
        
        # Demo queries with different characteristics
        demo_queries = [
            {
                "query": "What is the status of permit P-001?",
                "query_type": QueryType.COMPLIANCE,
                "mode": OrchestrationMode.SINGLE,
                "description": "Simple lookup query - should use Text2Cypher"
            },
            {
                "query": "Show me water consumption trends for the last 6 months",
                "query_type": QueryType.CONSUMPTION,
                "mode": OrchestrationMode.ADAPTIVE,
                "description": "Temporal analysis - should use HybridCypher"
            },
            {
                "query": "Find equipment related to energy efficiency improvements",
                "query_type": QueryType.EFFICIENCY,
                "mode": OrchestrationMode.PARALLEL,
                "description": "Complex search - should use multiple strategies"
            },
            {
                "query": "Analyze carbon emission patterns and recommend reduction strategies",
                "query_type": QueryType.RECOMMENDATION,
                "mode": OrchestrationMode.PARALLEL,
                "description": "Complex analytical query - should use multiple strategies"
            }
        ]
        
        # Execute demo queries
        for i, demo in enumerate(demo_queries, 1):
            logger.info(f"\n--- Demo Query {i}: {demo['description']} ---")
            logger.info(f"Query: {demo['query']}")
            logger.info(f"Type: {demo['query_type'].value}")
            logger.info(f"Mode: {demo['mode'].value}")
            
            try:
                # Execute the query
                result = await orchestrator.retrieve(
                    query=demo["query"],
                    query_type=demo["query_type"],
                    mode=demo["mode"],
                    limit=10
                )
                
                # Display results
                logger.info(f"✅ Query executed successfully!")
                logger.info(f"Strategies used: {[s.value for s in result.source_strategies]}")
                logger.info(f"Results count: {len(result.data)}")
                logger.info(f"Confidence score: {result.confidence_score:.3f}")
                logger.info(f"Execution time: {result.metrics.total_execution_time_ms:.2f}ms")
                logger.info(f"Ranking method: {result.ranking_explanation}")
                
                # Show deduplication info
                dedup = result.deduplication_info
                logger.info(f"Deduplication: {dedup.original_count} → {dedup.deduplicated_count} "
                           f"({dedup.duplicates_removed} removed)")
                
                # Show top results
                if result.data:
                    logger.info("Top results:")
                    for j, item in enumerate(result.data[:3], 1):
                        score = item.get('score', 0.0)
                        name = item.get('name', item.get('title', 'Unknown'))
                        logger.info(f"  {j}. {name} (score: {score:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Query failed: {e}")
        
        # Show performance metrics
        logger.info("\n--- Performance Metrics ---")
        metrics = await orchestrator.get_performance_metrics()
        if metrics.get('recent_performance'):
            perf = metrics['recent_performance']
            logger.info(f"Average execution time: {perf['avg_execution_time_ms']:.2f}ms")
            logger.info(f"Average strategies used: {perf['avg_strategies_used']:.1f}")
            logger.info(f"Parallel execution rate: {perf['parallel_execution_rate']:.1%}")
            logger.info(f"Cache hit rate: {perf['cache_hit_rate']:.1%}")
        
        # Show strategy performance
        if metrics.get('strategy_performance'):
            logger.info("\nStrategy Performance:")
            for strategy, perf in metrics['strategy_performance'].items():
                logger.info(f"  {strategy}: "
                           f"success={perf['success_rate']:.1%}, "
                           f"avg_time={perf['avg_response_time']:.0f}ms, "
                           f"usage={perf['usage_count']}")
        
        # Health check
        logger.info("\n--- Health Check ---")
        health = await orchestrator.health_check()
        logger.info(f"Orchestrator status: {health['orchestrator']['status']}")
        logger.info(f"Active retrievers: {health['orchestrator']['active_retrievers']}")
        
        for strategy, retriever_health in health['retrievers'].items():
            status = retriever_health.get('status', 'unknown')
            logger.info(f"  {strategy}: {status}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Cleanup
        if 'orchestrator' in locals():
            await orchestrator.cleanup()
            logger.info("Orchestrator cleanup completed")
    
    logger.info("Demo completed successfully!")


def demo_strategy_selection():
    """Demonstrate strategy selection logic."""
    
    from src.ehs_analytics.retrieval.strategy_selector import StrategySelector
    from src.ehs_analytics.retrieval.base import QueryType, RetrievalStrategy
    
    logger.info("\n--- Strategy Selection Demo ---")
    
    selector = StrategySelector()
    
    # Test queries with different characteristics
    test_queries = [
        ("What is facility F-001?", QueryType.GENERAL),
        ("Show me water consumption over time", QueryType.CONSUMPTION),
        ("Find documents about safety procedures", QueryType.GENERAL),
        ("Analyze emission trends and correlations", QueryType.EMISSIONS),
        ("Equipment connected to permits", QueryType.COMPLIANCE)
    ]
    
    available_strategies = list(RetrievalStrategy)
    
    async def run_selection_demo():
        await selector.initialize()
        
        for query, query_type in test_queries:
            result = await selector.select_strategies(
                query=query,
                query_type=query_type,
                available_strategies=available_strategies,
                max_strategies=2
            )
            
            logger.info(f"Query: '{query}'")
            logger.info(f"Selected: {[s.value for s in result.selected_strategies]}")
            logger.info(f"Reasoning: {result.reasoning}")
            logger.info(f"Characteristics: complexity={result.query_characteristics.complexity_score:.2f}, "
                       f"temporal={result.query_characteristics.is_temporal_query}, "
                       f"analytical={result.query_characteristics.is_analytical_query}")
            logger.info("")
    
    asyncio.run(run_selection_demo())


def demo_result_merging():
    """Demonstrate result merging logic."""
    
    from src.ehs_analytics.retrieval.result_merger import ResultMerger, MergerConfig
    from src.ehs_analytics.retrieval.base import RetrievalResult, RetrievalMetadata, RetrievalStrategy, QueryType
    
    logger.info("\n--- Result Merging Demo ---")
    
    # Create mock results from different strategies
    mock_results = [
        RetrievalResult(
            data=[
                {"id": "F-001", "name": "Main Facility", "score": 0.9},
                {"id": "F-002", "name": "Secondary Plant", "score": 0.7}
            ],
            metadata=RetrievalMetadata(
                strategy=RetrievalStrategy.TEXT2CYPHER,
                query_type=QueryType.GENERAL,
                confidence_score=0.85,
                execution_time_ms=150.0
            ),
            success=True
        ),
        RetrievalResult(
            data=[
                {"id": "F-001", "name": "Main Facility", "score": 0.8},  # Duplicate
                {"id": "F-003", "name": "Research Center", "score": 0.6}
            ],
            metadata=RetrievalMetadata(
                strategy=RetrievalStrategy.VECTOR,
                query_type=QueryType.GENERAL,
                confidence_score=0.75,
                execution_time_ms=220.0
            ),
            success=True
        )
    ]
    
    async def run_merging_demo():
        config = MergerConfig(enable_score_normalization=True)
        merger = ResultMerger(config)
        await merger.initialize()
        
        merged = await merger.merge_results(
            results=mock_results,
            query="test query",
            query_type=QueryType.GENERAL,
            max_results=10
        )
        
        logger.info(f"Original results: {sum(len(r.data) for r in mock_results)}")
        logger.info(f"Merged results: {len(merged.data)}")
        logger.info(f"Duplicates removed: {merged.deduplication_info.duplicates_removed}")
        logger.info(f"Confidence score: {merged.confidence_score:.3f}")
        logger.info(f"Ranking: {merged.ranking_explanation}")
        
        for i, item in enumerate(merged.data, 1):
            logger.info(f"  {i}. {item.get('name')} (score: {item.get('score', 0):.3f})")
    
    asyncio.run(run_merging_demo())


if __name__ == "__main__":
    # Run the main demo
    asyncio.run(main())
    
    # Run additional component demos
    demo_strategy_selection()
    demo_result_merging()