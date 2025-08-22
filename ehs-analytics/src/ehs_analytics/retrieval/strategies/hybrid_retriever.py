"""
Hybrid retriever implementation combining vector and fulltext search for EHS Analytics.

This module provides a sophisticated hybrid search capability that intelligently
combines vector similarity search with fulltext search for optimal EHS document
retrieval across different query types and contexts.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from dataclasses import dataclass

from neo4j_graphrag.retrievers import HybridRetriever as GraphRAGHybridRetriever
from neo4j_graphrag.retrievers import VectorRetriever, Text2CypherRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings

from ..base import BaseRetriever, RetrievalResult, RetrievalMetadata, RetrievalStrategy, QueryType
from ..config import RetrieverConfig
from .vector_retriever import EHSVectorRetriever
from .text2cypher import Text2CypherRetriever as EHSText2CypherRetriever
from .hybrid_config import (
    HybridConfigurationManager, 
    HybridRetrieverConfig, 
    SearchStrategy,
    QueryCharacteristics,
    WeightConfiguration
)
from .fusion_strategies import (
    FusionStrategyFactory,
    SearchResult,
    FusionResult,
    FusionMethod
)

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchMetrics:
    """Metrics for hybrid search performance analysis."""
    
    query_analysis_time: float = 0.0
    vector_search_time: float = 0.0
    fulltext_search_time: float = 0.0
    fusion_time: float = 0.0
    total_time: float = 0.0
    
    vector_results_count: int = 0
    fulltext_results_count: int = 0
    fused_results_count: int = 0
    deduplicated_count: int = 0
    
    strategy_used: Optional[SearchStrategy] = None
    fusion_method: Optional[FusionMethod] = None
    parallel_execution: bool = False


class EHSHybridRetriever(BaseRetriever):
    """
    Advanced hybrid retriever that combines vector similarity and fulltext search
    with EHS-specific optimization strategies.
    
    Features:
    - Intelligent query analysis and strategy selection
    - Parallel execution of vector and fulltext searches
    - Sophisticated result fusion with EHS-specific boosts
    - Dynamic weight adjustment based on query characteristics
    - Performance monitoring and adaptive optimization
    """
    
    def __init__(
        self,
        neo4j_driver,
        config: Optional[Dict[str, Any]] = None,
        vector_retriever: Optional[EHSVectorRetriever] = None,
        text2cypher_retriever: Optional[EHSText2CypherRetriever] = None
    ):
        """
        Initialize the EHS Hybrid Retriever.
        
        Args:
            neo4j_driver: Neo4j database driver
            config: Configuration dictionary
            vector_retriever: Pre-configured vector retriever (optional)
            text2cypher_retriever: Pre-configured text2cypher retriever (optional)
        """
        super().__init__(config or {})
        self.neo4j_driver = neo4j_driver
        
        # Initialize configuration
        self.hybrid_config = HybridRetrieverConfig(**(config or {}))
        self.config_manager = HybridConfigurationManager(self.hybrid_config)
        
        # Initialize component retrievers
        self._init_component_retrievers(vector_retriever, text2cypher_retriever)
        
        # Initialize GraphRAG hybrid retriever (if available)
        self._init_graphrag_hybrid()
        
        # Performance tracking
        self.search_metrics = []
        
        logger.info("Initialized EHS Hybrid Retriever with adaptive search strategies")
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get the retrieval strategy identifier."""
        return RetrievalStrategy.HYBRID
    
    async def validate_query(self, query: str) -> bool:
        """Validate if the query can be processed by this retriever."""
        return len(query.strip()) > 0
    
    def _init_component_retrievers(
        self, 
        vector_retriever: Optional[EHSVectorRetriever],
        text2cypher_retriever: Optional[EHSText2CypherRetriever]
    ):
        """Initialize vector and text2cypher retrievers."""
        try:
            # Initialize vector retriever
            if vector_retriever:
                self.vector_retriever = vector_retriever
            else:
                vector_config = {
                    "embedding_model": self.config.get("embedding_model", "text-embedding-ada-002"),
                    "max_results": self.hybrid_config.vector_top_k,
                    "similarity_threshold": self.hybrid_config.min_vector_score
                }
                self.vector_retriever = EHSVectorRetriever(
                    neo4j_driver=self.neo4j_driver,
                    config=vector_config
                )
            
            # Initialize text2cypher retriever
            if text2cypher_retriever:
                self.text2cypher_retriever = text2cypher_retriever
            else:
                text2cypher_config = {
                    "max_results": self.hybrid_config.fulltext_top_k,
                    "model_name": self.config.get("llm_model", "gpt-4")
                }
                self.text2cypher_retriever = EHSText2CypherRetriever(
                    neo4j_driver=self.neo4j_driver,
                    config=text2cypher_config
                )
            
            logger.info("Component retrievers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize component retrievers: {e}")
            raise
    
    def _init_graphrag_hybrid(self):
        """Initialize neo4j-graphrag-python HybridRetriever if available."""
        try:
            # Configure embeddings
            embedder = OpenAIEmbeddings(
                model=self.config.get("embedding_model", "text-embedding-ada-002")
            )
            
            # Initialize GraphRAG hybrid retriever
            self.graphrag_hybrid = GraphRAGHybridRetriever(
                driver=self.neo4j_driver,
                vector_index_name="ehs_document_chunks",
                fulltext_index_name="ehs_fulltext_index",
                embedder=embedder,
                return_properties=["content", "metadata", "document_id", "chunk_id", "document_type"],
                result_formatter=None  # Use default formatter
            )
            
            logger.info("GraphRAG Hybrid Retriever initialized successfully")
            
        except Exception as e:
            logger.warning(f"GraphRAG Hybrid Retriever not available: {e}")
            self.graphrag_hybrid = None
    
    async def initialize(self) -> None:
        """Initialize the hybrid retriever and its components."""
        try:
            # Initialize component retrievers
            if hasattr(self.vector_retriever, 'initialize'):
                await self.vector_retriever.initialize()
            
            if hasattr(self.text2cypher_retriever, 'initialize'):
                await self.text2cypher_retriever.initialize()
            
            self._initialized = True
            logger.info("Hybrid retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        limit: int = 10,
        **kwargs
    ) -> RetrievalResult:
        """
        Execute hybrid retrieval combining vector and fulltext search.
        
        Args:
            query: Natural language query
            query_type: Type of EHS query
            limit: Maximum number of results to return
            **kwargs: Additional retrieval parameters
            
        Returns:
            RetrievalResult with hybrid search results and comprehensive metadata
        """
        start_time = datetime.now()
        metrics = HybridSearchMetrics()
        
        try:
            # Step 1: Analyze query characteristics
            analysis_start = datetime.now()
            characteristics = self.config_manager.analyze_query(query)
            search_strategy = self.config_manager.get_search_strategy(query_type, query)
            weight_config = self.config_manager.get_weight_configuration(
                search_strategy, query_type, characteristics
            )
            metrics.query_analysis_time = (datetime.now() - analysis_start).total_seconds()
            metrics.strategy_used = search_strategy
            
            logger.info(f"Query analysis: strategy={search_strategy.value}, "
                       f"vector_weight={weight_config.vector_weight:.2f}, "
                       f"fulltext_weight={weight_config.fulltext_weight:.2f}")
            
            # Step 2: Execute searches (parallel or GraphRAG)
            if self.graphrag_hybrid and self.hybrid_config.parallel_search:
                results = await self._execute_graphrag_hybrid_search(
                    query, characteristics, weight_config, limit, metrics, **kwargs
                )
            else:
                results = await self._execute_parallel_search(
                    query, query_type, characteristics, weight_config, limit, metrics, **kwargs
                )
            
            # Step 3: Record metrics and return results
            metrics.total_time = (datetime.now() - start_time).total_seconds()
            self.search_metrics.append(metrics)
            
            # Enhance result metadata
            enhanced_metadata = self._enhance_metadata(results.metadata, metrics, characteristics)
            results.metadata = enhanced_metadata
            
            logger.info(f"Hybrid retrieval completed: {len(results.data)} results in {metrics.total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            metrics.total_time = (datetime.now() - start_time).total_seconds()
            
            return RetrievalResult(
                success=False,
                data=[],
                metadata=RetrievalMetadata(
                    strategy=self.get_strategy(),
                    query_type=query_type,
                    confidence_score=0.0,
                    execution_time_ms=metrics.total_time * 1000,
                    error_message=str(e)
                )
            )
    
    async def _execute_graphrag_hybrid_search(
        self,
        query: str,
        characteristics: QueryCharacteristics,
        weight_config: WeightConfiguration,
        limit: int,
        metrics: HybridSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute hybrid search using GraphRAG HybridRetriever."""
        try:
            search_start = datetime.now()
            
            # Configure GraphRAG hybrid search
            vector_top_k = self.hybrid_config.vector_top_k
            fulltext_top_k = self.hybrid_config.fulltext_top_k
            
            # Build filter conditions
            filter_conditions = self._build_filter_conditions(kwargs)
            
            # Execute GraphRAG hybrid search
            results = self.graphrag_hybrid.search(
                query_text=query,
                vector_top_k=vector_top_k,
                fulltext_top_k=fulltext_top_k,
                vector_weight=weight_config.vector_weight,
                fulltext_weight=weight_config.fulltext_weight,
                filter=filter_conditions
            )
            
            search_time = (datetime.now() - search_start).total_seconds()
            metrics.vector_search_time = search_time / 2  # Approximate split
            metrics.fulltext_search_time = search_time / 2
            metrics.parallel_execution = True
            
            # Convert GraphRAG results to our format
            formatted_results = self._format_graphrag_results(results, query, weight_config)
            
            # Apply final ranking and limits
            final_results = formatted_results[:limit]
            
            return RetrievalResult(
                success=True,
                data=final_results,
                metadata=RetrievalMetadata(
                    strategy=self.get_strategy(),
                    query_type=QueryType.GENERAL,  # Will be enhanced later
                    confidence_score=self._calculate_confidence_score(final_results),
                    execution_time_ms=search_time * 1000,
                    nodes_retrieved=len(final_results)
                )
            )
            
        except Exception as e:
            logger.error(f"GraphRAG hybrid search failed: {e}")
            # Fallback to parallel search
            return await self._execute_parallel_search(
                query, QueryType.GENERAL, characteristics, weight_config, limit, metrics, **kwargs
            )
    
    async def _execute_parallel_search(
        self,
        query: str,
        query_type: QueryType,
        characteristics: QueryCharacteristics,
        weight_config: WeightConfiguration,
        limit: int,
        metrics: HybridSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute parallel vector and fulltext searches with custom fusion."""
        search_tasks = []
        
        # Prepare search parameters
        vector_params = {
            "query": query,
            "query_type": query_type,
            "limit": self.hybrid_config.vector_top_k,
            **kwargs
        }
        
        text2cypher_params = {
            "query": query,
            "query_type": query_type,
            "limit": self.hybrid_config.fulltext_top_k,
            **kwargs
        }
        
        # Execute searches in parallel if enabled
        if self.hybrid_config.parallel_search:
            search_tasks = [
                self._execute_vector_search(vector_params, metrics),
                self._execute_text2cypher_search(text2cypher_params, metrics)
            ]
            
            vector_results, text2cypher_results = await asyncio.gather(
                *search_tasks, return_exceptions=True
            )
            metrics.parallel_execution = True
        else:
            # Execute searches sequentially
            vector_results = await self._execute_vector_search(vector_params, metrics)
            text2cypher_results = await self._execute_text2cypher_search(text2cypher_params, metrics)
            metrics.parallel_execution = False
        
        # Handle any exceptions from parallel execution
        if isinstance(vector_results, Exception):
            logger.error(f"Vector search failed: {vector_results}")
            vector_results = RetrievalResult(success=False, data=[], metadata=None)
        
        if isinstance(text2cypher_results, Exception):
            logger.error(f"Text2Cypher search failed: {text2cypher_results}")
            text2cypher_results = RetrievalResult(success=False, data=[], metadata=None)
        
        # Convert to SearchResult format for fusion
        vector_search_results = self._convert_to_search_results(vector_results, "vector")
        fulltext_search_results = self._convert_to_search_results(text2cypher_results, "fulltext")
        
        # Execute result fusion
        fusion_start = datetime.now()
        fused_results = await self._fuse_search_results(
            vector_search_results, fulltext_search_results, query, weight_config, metrics
        )
        metrics.fusion_time = (datetime.now() - fusion_start).total_seconds()
        
        # Apply final ranking and limits
        final_results = fused_results[:limit]
        
        # Convert back to RetrievalResult format
        formatted_results = [self._format_fusion_result(fr) for fr in final_results]
        
        return RetrievalResult(
            success=True,
            data=formatted_results,
            metadata=RetrievalMetadata(
                strategy=self.get_strategy(),
                query_type=query_type,
                confidence_score=self._calculate_confidence_score(formatted_results),
                execution_time_ms=(datetime.now() - datetime.now()).total_seconds() * 1000,
                nodes_retrieved=len(formatted_results)
            )
        )
    
    async def _execute_vector_search(
        self, 
        params: Dict[str, Any], 
        metrics: HybridSearchMetrics
    ) -> RetrievalResult:
        """Execute vector search with timing."""
        start_time = datetime.now()
        try:
            result = await self.vector_retriever.retrieve(**params)
            metrics.vector_search_time = (datetime.now() - start_time).total_seconds()
            metrics.vector_results_count = len(result.data) if result.success else 0
            return result
        except Exception as e:
            metrics.vector_search_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Vector search error: {e}")
            return RetrievalResult(success=False, data=[], metadata=None)
    
    async def _execute_text2cypher_search(
        self, 
        params: Dict[str, Any], 
        metrics: HybridSearchMetrics
    ) -> RetrievalResult:
        """Execute text2cypher search with timing."""
        start_time = datetime.now()
        try:
            result = await self.text2cypher_retriever.retrieve(**params)
            metrics.fulltext_search_time = (datetime.now() - start_time).total_seconds()
            metrics.fulltext_results_count = len(result.data) if result.success else 0
            return result
        except Exception as e:
            metrics.fulltext_search_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Text2Cypher search error: {e}")
            return RetrievalResult(success=False, data=[], metadata=None)
    
    def _convert_to_search_results(
        self, 
        retrieval_result: RetrievalResult, 
        source: str
    ) -> List[SearchResult]:
        """Convert RetrievalResult to SearchResult list for fusion."""
        if not retrieval_result.success:
            return []
        
        search_results = []
        for i, item in enumerate(retrieval_result.data):
            search_result = SearchResult(
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                metadata=item.get("metadata", {}),
                document_id=item.get("document_id", f"{source}_{i}"),
                chunk_id=item.get("chunk_id"),
                document_type=item.get("document_type", "unknown"),
                source=source,
                rank=i + 1
            )
            search_results.append(search_result)
        
        return search_results
    
    async def _fuse_search_results(
        self,
        vector_results: List[SearchResult],
        fulltext_results: List[SearchResult],
        query: str,
        weight_config: WeightConfiguration,
        metrics: HybridSearchMetrics
    ) -> List[FusionResult]:
        """Fuse vector and fulltext search results."""
        # Create fusion strategy
        fusion_strategy = FusionStrategyFactory.create_strategy(
            method=self.hybrid_config.fusion_method,
            config=weight_config,
            rrf_k=self.hybrid_config.rrf_constant
        )
        metrics.fusion_method = self.hybrid_config.fusion_method
        
        # Execute fusion
        fused_results = fusion_strategy.fuse_results(
            vector_results=vector_results,
            fulltext_results=fulltext_results,
            query=query
        )
        
        metrics.fused_results_count = len(fused_results)
        
        # Apply deduplication if enabled
        if self.hybrid_config.enable_deduplication:
            # Convert to SearchResult for deduplication
            search_results_for_dedup = []
            for fr in fused_results:
                sr = SearchResult(
                    content=fr.content,
                    score=fr.final_score,
                    metadata=fr.metadata,
                    document_id=fr.document_id,
                    chunk_id=fr.chunk_id,
                    document_type=fr.document_type,
                    source="fused"
                )
                search_results_for_dedup.append(sr)
            
            deduplicated = fusion_strategy.deduplicator.deduplicate_results(search_results_for_dedup)
            metrics.deduplicated_count = len(deduplicated)
            
            # Convert back to FusionResult
            final_results = []
            for sr in deduplicated:
                # Find original fusion result to preserve fusion_details
                original_fr = next((fr for fr in fused_results if fr.document_id == sr.document_id), None)
                if original_fr:
                    final_results.append(original_fr)
            
            return final_results
        
        return fused_results
    
    def _format_fusion_result(self, fusion_result: FusionResult) -> Dict[str, Any]:
        """Format FusionResult for RetrievalResult."""
        return {
            "content": fusion_result.content,
            "score": fusion_result.final_score,
            "metadata": {
                **fusion_result.metadata,
                "document_id": fusion_result.document_id,
                "chunk_id": fusion_result.chunk_id,
                "document_type": fusion_result.document_type,
                "fusion_details": fusion_result.fusion_details
            }
        }
    
    def _format_graphrag_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        weight_config: WeightConfiguration
    ) -> List[Dict[str, Any]]:
        """Format GraphRAG results for consistency."""
        formatted = []
        for result in results:
            formatted_result = {
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "metadata": {
                    **result.get("metadata", {}),
                    "document_id": result.get("document_id", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "document_type": result.get("document_type", "unknown"),
                    "search_method": "graphrag_hybrid",
                    "weight_config": {
                        "vector_weight": weight_config.vector_weight,
                        "fulltext_weight": weight_config.fulltext_weight
                    }
                }
            }
            formatted.append(formatted_result)
        
        return formatted
    
    def _build_filter_conditions(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Build filter conditions from kwargs."""
        filters = {}
        
        # Extract standard filters
        if "facility" in kwargs:
            filters["facility"] = kwargs["facility"]
        if "document_types" in kwargs:
            filters["document_type"] = kwargs["document_types"]
        if "date_range" in kwargs:
            start_date, end_date = kwargs["date_range"]
            filters["date_created"] = {
                "gte": start_date.isoformat() if isinstance(start_date, date) else start_date,
                "lte": end_date.isoformat() if isinstance(end_date, date) else end_date
            }
        
        return filters
    
    def _calculate_confidence_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for search results."""
        if not results:
            return 0.0
        
        # Calculate based on score distribution and result count
        scores = [r.get("score", 0.0) for r in results]
        
        if not scores:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Maximum score
        # 2. Score consistency (lower std dev = higher confidence)
        # 3. Number of results
        
        max_score = max(scores)
        score_std = np.std(scores) if len(scores) > 1 else 0.0
        result_count_factor = min(1.0, len(results) / 10.0)  # Normalize to 10 results
        
        # Combine factors
        confidence = (max_score * 0.5) + ((1.0 - score_std) * 0.3) + (result_count_factor * 0.2)
        
        return min(1.0, max(0.0, confidence))
    
    def _enhance_metadata(
        self, 
        base_metadata: RetrievalMetadata, 
        metrics: HybridSearchMetrics,
        characteristics: QueryCharacteristics
    ) -> RetrievalMetadata:
        """Enhance metadata with hybrid search information."""
        # Create new metadata instance with enhanced information
        enhanced = RetrievalMetadata(
            strategy=base_metadata.strategy,
            query_type=base_metadata.query_type,
            confidence_score=base_metadata.confidence_score,
            execution_time_ms=metrics.total_time * 1000,
            nodes_retrieved=base_metadata.nodes_retrieved,
            error_message=base_metadata.error_message,
            # Additional hybrid-specific metadata
            vector_similarity_scores=None  # Could be populated from results
        )
        
        # Add custom metadata for hybrid search
        enhanced.metadata = {
            "hybrid_metrics": {
                "query_analysis_time": metrics.query_analysis_time,
                "vector_search_time": metrics.vector_search_time,
                "fulltext_search_time": metrics.fulltext_search_time,
                "fusion_time": metrics.fusion_time,
                "parallel_execution": metrics.parallel_execution,
                "strategy_used": metrics.strategy_used.value if metrics.strategy_used else None,
                "fusion_method": metrics.fusion_method.value if metrics.fusion_method else None
            },
            "query_characteristics": {
                "contains_keywords": characteristics.contains_keywords,
                "contains_facilities": characteristics.contains_facilities,
                "contains_regulations": characteristics.contains_regulations,
                "semantic_complexity": characteristics.semantic_complexity,
                "query_length": characteristics.query_length
            },
            "result_counts": {
                "vector_results": metrics.vector_results_count,
                "fulltext_results": metrics.fulltext_results_count,
                "fused_results": metrics.fused_results_count,
                "deduplicated_results": metrics.deduplicated_count
            }
        }
        
        return enhanced
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on hybrid retriever."""
        health_status = await super().health_check()
        
        # Check component retrievers
        vector_health = await self.vector_retriever.health_check() if self.vector_retriever else {"status": "not_available"}
        text2cypher_health = await self.text2cypher_retriever.health_check() if self.text2cypher_retriever else {"status": "not_available"}
        
        # Check GraphRAG hybrid
        graphrag_status = "available" if self.graphrag_hybrid else "not_available"
        
        # Performance metrics
        recent_metrics = self.search_metrics[-10:] if self.search_metrics else []
        avg_response_time = np.mean([m.total_time for m in recent_metrics]) if recent_metrics else 0.0
        
        health_status.update({
            "component_health": {
                "vector_retriever": vector_health,
                "text2cypher_retriever": text2cypher_health,
                "graphrag_hybrid": graphrag_status
            },
            "performance": {
                "recent_searches": len(recent_metrics),
                "avg_response_time": avg_response_time,
                "parallel_search_enabled": self.hybrid_config.parallel_search
            },
            "configuration": {
                "default_strategy": self.hybrid_config.default_strategy.value,
                "fusion_method": self.hybrid_config.fusion_method.value,
                "caching_enabled": self.hybrid_config.enable_caching
            }
        })
        
        return health_status
    
    async def cleanup(self) -> None:
        """Clean up hybrid retriever resources."""
        try:
            # Cleanup component retrievers
            if self.vector_retriever and hasattr(self.vector_retriever, 'cleanup'):
                await self.vector_retriever.cleanup()
            
            if self.text2cypher_retriever and hasattr(self.text2cypher_retriever, 'cleanup'):
                await self.text2cypher_retriever.cleanup()
            
            # Clear metrics
            self.search_metrics.clear()
            
            await super().cleanup()
            logger.info("Hybrid retriever cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during hybrid retriever cleanup: {e}")


# Convenience function for easy initialization
def create_ehs_hybrid_retriever(
    neo4j_driver,
    config: Optional[Dict[str, Any]] = None
) -> EHSHybridRetriever:
    """
    Create and initialize an EHS Hybrid Retriever with default configuration.
    
    Args:
        neo4j_driver: Neo4j database driver
        config: Optional configuration overrides
        
    Returns:
        Configured EHSHybridRetriever instance
    """
    default_config = {
        "max_results": 10,
        "parallel_search": True,
        "enable_caching": True,
        "fusion_method": FusionMethod.RECIPROCAL_RANK_FUSION,
        "default_strategy": SearchStrategy.ADAPTIVE,
        "enable_deduplication": True
    }
    
    if config:
        default_config.update(config)
    
    return EHSHybridRetriever(
        neo4j_driver=neo4j_driver,
        config=default_config
    )