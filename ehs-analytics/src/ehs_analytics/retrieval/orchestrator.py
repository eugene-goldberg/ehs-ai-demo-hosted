"""
Retrieval Orchestrator for EHS Analytics.

This module provides intelligent strategy selection and coordination for retrieval operations.
It analyzes queries to select optimal retriever(s), executes them in parallel when beneficial,
and merges results to provide the best possible responses.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import (
    BaseRetriever,
    RetrievalStrategy,
    QueryType,
    RetrievalResult,
    RetrievalMetadata
)
from .strategy_selector import StrategySelector, SelectionResult
from .result_merger import ResultMerger, MergedResult
from .strategies import (
    EHSText2CypherRetriever,
    EHSVectorRetriever,
    EHSHybridRetriever,
    EHSVectorCypherRetriever,
    EHSHybridCypherRetriever
)

logger = logging.getLogger(__name__)


class OrchestrationMode(str, Enum):
    """Orchestration execution modes."""
    SINGLE = "single"           # Use single best retriever
    PARALLEL = "parallel"       # Execute multiple retrievers in parallel
    SEQUENTIAL = "sequential"   # Execute retrievers in sequence with fallback
    ADAPTIVE = "adaptive"       # Dynamically choose based on query complexity


@dataclass
class OrchestrationMetrics:
    """Metrics for orchestration performance tracking."""
    
    total_execution_time_ms: float
    strategy_selection_time_ms: float
    retrieval_execution_time_ms: float
    result_merging_time_ms: float
    strategies_used: List[RetrievalStrategy]
    strategies_attempted: List[RetrievalStrategy]
    strategies_failed: List[RetrievalStrategy] = field(default_factory=list)
    parallel_execution: bool = False
    fallback_triggered: bool = False
    cache_hit: bool = False
    total_results: int = 0
    unique_results: int = 0
    confidence_score: float = 0.0


@dataclass
class OrchestrationConfig:
    """Configuration for retrieval orchestration."""
    
    # Strategy selection
    max_strategies: int = 3
    min_confidence_threshold: float = 0.7
    enable_fallback: bool = True
    enable_parallel_execution: bool = True
    
    # Performance settings
    max_execution_time_ms: float = 30000  # 30 seconds
    parallel_timeout_ms: float = 20000    # 20 seconds
    max_workers: int = 4
    
    # Result merging
    enable_deduplication: bool = True
    max_merged_results: int = 50
    score_normalization: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    
    # Adaptive behavior
    enable_adaptive_selection: bool = True
    performance_learning: bool = True
    strategy_bias_weights: Dict[RetrievalStrategy, float] = field(default_factory=lambda: {
        RetrievalStrategy.TEXT2CYPHER: 1.0,
        RetrievalStrategy.VECTOR: 0.8,
        RetrievalStrategy.HYBRID: 1.2,
        RetrievalStrategy.VECTOR_CYPHER: 1.1,
        RetrievalStrategy.HYBRID_CYPHER: 1.3
    })


class RetrievalOrchestrator:
    """
    Intelligent retrieval orchestrator that coordinates multiple retrieval strategies.
    
    The orchestrator analyzes incoming queries to determine the optimal combination
    of retrieval strategies, executes them efficiently (parallel/sequential), and
    merges results to provide comprehensive and accurate responses.
    """
    
    def __init__(
        self,
        retrievers: Dict[RetrievalStrategy, BaseRetriever],
        config: OrchestrationConfig = None
    ):
        """
        Initialize the retrieval orchestrator.
        
        Args:
            retrievers: Dictionary mapping strategies to retriever instances
            config: Orchestration configuration settings
        """
        self.retrievers = retrievers
        self.config = config or OrchestrationConfig()
        
        # Initialize components
        self.strategy_selector = StrategySelector()
        self.result_merger = ResultMerger()
        
        # Performance tracking
        self.performance_history: List[OrchestrationMetrics] = []
        self.strategy_performance: Dict[RetrievalStrategy, Dict[str, float]] = {}
        
        # Caching
        self.query_cache: Dict[str, Tuple[MergedResult, float]] = {}
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info(f"Initialized RetrievalOrchestrator with {len(retrievers)} strategies")
    
    async def initialize(self) -> None:
        """Initialize all retrievers and components."""
        start_time = time.time()
        
        # Initialize all retrievers
        initialization_tasks = []
        for strategy, retriever in self.retrievers.items():
            if not retriever._initialized:
                task = asyncio.create_task(retriever.initialize())
                initialization_tasks.append((strategy, task))
        
        # Wait for all initializations
        for strategy, task in initialization_tasks:
            try:
                await task
                logger.info(f"Initialized {strategy.value} retriever")
            except Exception as e:
                logger.error(f"Failed to initialize {strategy.value} retriever: {e}")
        
        # Initialize strategy selector and result merger
        await self.strategy_selector.initialize()
        await self.result_merger.initialize()
        
        initialization_time = (time.time() - start_time) * 1000
        logger.info(f"Orchestrator initialization completed in {initialization_time:.2f}ms")
    
    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        mode: OrchestrationMode = OrchestrationMode.ADAPTIVE,
        limit: int = 10,
        **kwargs
    ) -> MergedResult:
        """
        Execute orchestrated retrieval for the given query.
        
        Args:
            query: Natural language query from the user
            query_type: Type of EHS query being processed
            mode: Orchestration execution mode
            limit: Maximum number of results to return
            **kwargs: Additional parameters
            
        Returns:
            MergedResult containing consolidated results and metrics
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(query, query_type, limit)
        if self.config.enable_caching and cache_key in self.query_cache:
            cached_result, cache_time = self.query_cache[cache_key]
            if (time.time() - cache_time) < self.config.cache_ttl_seconds:
                logger.info(f"Cache hit for query: {query[:50]}...")
                cached_result.metrics.cache_hit = True
                return cached_result
        
        try:
            # Step 1: Strategy Selection
            selection_start = time.time()
            selection_result = await self.strategy_selector.select_strategies(
                query=query,
                query_type=query_type,
                available_strategies=list(self.retrievers.keys()),
                max_strategies=self.config.max_strategies,
                min_confidence=self.config.min_confidence_threshold
            )
            selection_time = (time.time() - selection_start) * 1000
            
            # Adjust mode based on selection
            if mode == OrchestrationMode.ADAPTIVE:
                mode = self._determine_optimal_mode(selection_result, query_type)
            
            # Step 2: Execute Retrieval
            retrieval_start = time.time()
            retrieval_results = await self._execute_retrieval(
                query=query,
                query_type=query_type,
                selection_result=selection_result,
                mode=mode,
                limit=limit,
                **kwargs
            )
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            # Step 3: Merge Results
            merging_start = time.time()
            merged_result = await self.result_merger.merge_results(
                results=retrieval_results,
                query=query,
                query_type=query_type,
                max_results=min(limit, self.config.max_merged_results)
            )
            merging_time = (time.time() - merging_start) * 1000
            
            # Step 4: Update Metrics
            total_time = (time.time() - start_time) * 1000
            orchestration_metrics = OrchestrationMetrics(
                total_execution_time_ms=total_time,
                strategy_selection_time_ms=selection_time,
                retrieval_execution_time_ms=retrieval_time,
                result_merging_time_ms=merging_time,
                strategies_used=[r.metadata.strategy for r in retrieval_results if r.success],
                strategies_attempted=selection_result.selected_strategies,
                strategies_failed=[r.metadata.strategy for r in retrieval_results if not r.success],
                parallel_execution=(mode == OrchestrationMode.PARALLEL),
                total_results=sum(len(r.data) for r in retrieval_results),
                unique_results=len(merged_result.data),
                confidence_score=merged_result.confidence_score
            )
            
            merged_result.metrics = orchestration_metrics
            
            # Update performance tracking
            self._update_performance_tracking(orchestration_metrics, selection_result)
            
            # Cache result
            if self.config.enable_caching:
                self.query_cache[cache_key] = (merged_result, time.time())
                self._cleanup_cache()
            
            logger.info(
                f"Orchestrated retrieval completed in {total_time:.2f}ms "
                f"using {len(orchestration_metrics.strategies_used)} strategies"
            )
            
            return merged_result
            
        except Exception as e:
            logger.error(f"Orchestration failed for query '{query[:50]}...': {e}")
            
            # Return empty result with error information
            error_result = MergedResult(
                data=[],
                metadata={},
                confidence_score=0.0,
                source_strategies=[],
                deduplication_info={},
                ranking_explanation=""
            )
            error_result.metrics = OrchestrationMetrics(
                total_execution_time_ms=(time.time() - start_time) * 1000,
                strategy_selection_time_ms=0,
                retrieval_execution_time_ms=0,
                result_merging_time_ms=0,
                strategies_used=[],
                strategies_attempted=[]
            )
            
            return error_result
    
    async def _execute_retrieval(
        self,
        query: str,
        query_type: QueryType,
        selection_result: SelectionResult,
        mode: OrchestrationMode,
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Execute retrieval using selected strategies and mode."""
        
        if mode == OrchestrationMode.PARALLEL:
            return await self._execute_parallel_retrieval(
                query, query_type, selection_result, limit, **kwargs
            )
        elif mode == OrchestrationMode.SEQUENTIAL:
            return await self._execute_sequential_retrieval(
                query, query_type, selection_result, limit, **kwargs
            )
        else:  # SINGLE mode
            return await self._execute_single_retrieval(
                query, query_type, selection_result, limit, **kwargs
            )
    
    async def _execute_parallel_retrieval(
        self,
        query: str,
        query_type: QueryType,
        selection_result: SelectionResult,
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Execute multiple retrievers in parallel."""
        
        tasks = []
        for strategy in selection_result.selected_strategies:
            if strategy in self.retrievers:
                retriever = self.retrievers[strategy]
                task = asyncio.create_task(
                    retriever.retrieve(query, query_type, limit, **kwargs)
                )
                tasks.append((strategy, task))
        
        results = []
        completed_tasks = await asyncio.wait_for(
            asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
            timeout=self.config.parallel_timeout_ms / 1000
        )
        
        for (strategy, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Strategy {strategy.value} failed: {result}")
                # Create error result
                error_result = RetrievalResult(
                    data=[],
                    metadata=RetrievalMetadata(
                        strategy=strategy,
                        query_type=query_type,
                        confidence_score=0.0,
                        execution_time_ms=0.0,
                        error_message=str(result)
                    ),
                    success=False
                )
                results.append(error_result)
            else:
                results.append(result)
        
        return results
    
    async def _execute_sequential_retrieval(
        self,
        query: str,
        query_type: QueryType,
        selection_result: SelectionResult,
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Execute retrievers sequentially with fallback logic."""
        
        results = []
        
        for strategy in selection_result.selected_strategies:
            if strategy not in self.retrievers:
                continue
            
            try:
                retriever = self.retrievers[strategy]
                result = await retriever.retrieve(query, query_type, limit, **kwargs)
                results.append(result)
                
                # Check if result is good enough to stop
                if (result.success and 
                    len(result.data) > 0 and 
                    result.metadata.confidence_score >= self.config.min_confidence_threshold):
                    logger.info(f"Strategy {strategy.value} provided sufficient results")
                    break
                    
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed: {e}")
                error_result = RetrievalResult(
                    data=[],
                    metadata=RetrievalMetadata(
                        strategy=strategy,
                        query_type=query_type,
                        confidence_score=0.0,
                        execution_time_ms=0.0,
                        error_message=str(e)
                    ),
                    success=False
                )
                results.append(error_result)
        
        return results
    
    async def _execute_single_retrieval(
        self,
        query: str,
        query_type: QueryType,
        selection_result: SelectionResult,
        limit: int,
        **kwargs
    ) -> List[RetrievalResult]:
        """Execute single best retriever."""
        
        if not selection_result.selected_strategies:
            return []
        
        best_strategy = selection_result.selected_strategies[0]
        if best_strategy not in self.retrievers:
            return []
        
        try:
            retriever = self.retrievers[best_strategy]
            result = await retriever.retrieve(query, query_type, limit, **kwargs)
            return [result]
        except Exception as e:
            logger.error(f"Single strategy {best_strategy.value} failed: {e}")
            error_result = RetrievalResult(
                data=[],
                metadata=RetrievalMetadata(
                    strategy=best_strategy,
                    query_type=query_type,
                    confidence_score=0.0,
                    execution_time_ms=0.0,
                    error_message=str(e)
                ),
                success=False
            )
            return [error_result]
    
    def _determine_optimal_mode(
        self,
        selection_result: SelectionResult,
        query_type: QueryType
    ) -> OrchestrationMode:
        """Determine optimal orchestration mode based on selection result and query type."""
        
        num_strategies = len(selection_result.selected_strategies)
        
        # Single strategy - use single mode
        if num_strategies <= 1:
            return OrchestrationMode.SINGLE
        
        # Complex queries benefit from parallel execution
        complex_query_types = {QueryType.RISK, QueryType.RECOMMENDATION}
        if query_type in complex_query_types and num_strategies <= 3:
            return OrchestrationMode.PARALLEL
        
        # High confidence in primary strategy - use sequential with fallback
        if selection_result.strategy_confidences[selection_result.selected_strategies[0]] > 0.9:
            return OrchestrationMode.SEQUENTIAL
        
        # Default to parallel for multiple strategies
        if num_strategies <= 3:
            return OrchestrationMode.PARALLEL
        else:
            return OrchestrationMode.SEQUENTIAL
    
    def _update_performance_tracking(
        self,
        metrics: OrchestrationMetrics,
        selection_result: SelectionResult
    ) -> None:
        """Update performance tracking data."""
        
        # Add to history
        self.performance_history.append(metrics)
        
        # Keep only recent history
        max_history = 1000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
        
        # Update strategy performance
        for strategy in metrics.strategies_used:
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {
                    'avg_execution_time': 0.0,
                    'success_rate': 0.0,
                    'avg_results': 0.0,
                    'usage_count': 0
                }
            
            perf = self.strategy_performance[strategy]
            perf['usage_count'] += 1
            
            # Update running averages
            count = perf['usage_count']
            perf['avg_execution_time'] = (
                (perf['avg_execution_time'] * (count - 1) + metrics.retrieval_execution_time_ms) / count
            )
    
    def _generate_cache_key(self, query: str, query_type: QueryType, limit: int) -> str:
        """Generate cache key for query."""
        import hashlib
        key_string = f"{query}|{query_type.value}|{limit}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, cache_time) in self.query_cache.items()
            if (current_time - cache_time) > self.config.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on orchestrator and all retrievers."""
        
        health_status = {
            "orchestrator": {
                "status": "healthy",
                "active_retrievers": len(self.retrievers),
                "cache_size": len(self.query_cache),
                "performance_history_size": len(self.performance_history)
            },
            "retrievers": {}
        }
        
        # Check each retriever
        for strategy, retriever in self.retrievers.items():
            try:
                retriever_health = await retriever.health_check()
                health_status["retrievers"][strategy.value] = retriever_health
            except Exception as e:
                health_status["retrievers"][strategy.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health_status
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics."""
        
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_metrics = self.performance_history[-100:]  # Last 100 operations
        
        return {
            "total_operations": len(self.performance_history),
            "recent_performance": {
                "avg_execution_time_ms": sum(m.total_execution_time_ms for m in recent_metrics) / len(recent_metrics),
                "avg_strategies_used": sum(len(m.strategies_used) for m in recent_metrics) / len(recent_metrics),
                "parallel_execution_rate": sum(1 for m in recent_metrics if m.parallel_execution) / len(recent_metrics),
                "fallback_rate": sum(1 for m in recent_metrics if m.fallback_triggered) / len(recent_metrics),
                "cache_hit_rate": sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)
            },
            "strategy_performance": self.strategy_performance,
            "cache_statistics": {
                "size": len(self.query_cache),
                "hit_rate": sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        
        # Cleanup all retrievers
        cleanup_tasks = []
        for retriever in self.retrievers.values():
            cleanup_tasks.append(asyncio.create_task(retriever.cleanup()))
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Cleanup orchestrator resources
        self.executor.shutdown(wait=True)
        self.query_cache.clear()
        self.performance_history.clear()
        
        logger.info("Orchestrator cleanup completed")


async def create_ehs_retrieval_orchestrator(
    configs: Dict[RetrievalStrategy, Dict[str, Any]],
    orchestration_config: OrchestrationConfig = None
) -> RetrievalOrchestrator:
    """
    Factory function to create an EHS Retrieval Orchestrator with configured retrievers.
    
    Args:
        configs: Dictionary mapping strategies to their configurations
        orchestration_config: Orchestration configuration
        
    Returns:
        Configured RetrievalOrchestrator instance
    """
    
    retrievers = {}
    
    # Create retrievers based on provided configurations
    for strategy, config in configs.items():
        try:
            if strategy == RetrievalStrategy.TEXT2CYPHER:
                retrievers[strategy] = EHSText2CypherRetriever(config)
            elif strategy == RetrievalStrategy.VECTOR:
                retrievers[strategy] = EHSVectorRetriever(config)
            elif strategy == RetrievalStrategy.HYBRID:
                retrievers[strategy] = EHSHybridRetriever(config)
            elif strategy == RetrievalStrategy.VECTOR_CYPHER:
                retrievers[strategy] = EHSVectorCypherRetriever(config)
            elif strategy == RetrievalStrategy.HYBRID_CYPHER:
                retrievers[strategy] = EHSHybridCypherRetriever(config)
            else:
                logger.warning(f"Unknown retrieval strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Failed to create {strategy.value} retriever: {e}")
    
    orchestrator = RetrievalOrchestrator(
        retrievers=retrievers,
        config=orchestration_config or OrchestrationConfig()
    )
    
    await orchestrator.initialize()
    
    logger.info(f"Created EHS Retrieval Orchestrator with {len(retrievers)} strategies")
    return orchestrator