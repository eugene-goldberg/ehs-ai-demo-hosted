"""
HybridCypher retriever implementation for complex temporal queries in EHS Analytics.

This module provides an advanced retrieval strategy that combines vector similarity,
fulltext search, and graph traversal with sophisticated temporal awareness for 
complex EHS analytics queries involving time-based patterns and relationships.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum

from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings

from ..base import BaseRetriever, RetrievalResult, RetrievalMetadata, RetrievalStrategy, QueryType, EHSSchemaAware
from ..config import RetrieverConfig
from .vector_retriever import EHSVectorRetriever
from .text2cypher import Text2CypherRetriever as EHSText2CypherRetriever
from .temporal_patterns import (
    TemporalPatternAnalyzer,
    EHSTemporalPatterns,
    TimeWindowType,
    PatternType,
    TemporalAggregationType
)
from .hybrid_cypher_config import (
    HybridCypherConfig,
    TemporalQueryConfiguration,
    TimeWindowConfiguration,
    TemporalWeightDecayFunction
)

logger = logging.getLogger(__name__)


class TemporalQueryType(str, Enum):
    """Types of temporal queries supported by HybridCypher retriever."""
    
    TREND_ANALYSIS = "trend_analysis"
    PATTERN_DETECTION = "pattern_detection"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    SEQUENCE_ANALYSIS = "sequence_analysis"
    COMPLIANCE_TIMELINE = "compliance_timeline"
    RISK_PROGRESSION = "risk_progression"


@dataclass
class TemporalSearchMetrics:
    """Metrics for temporal search performance analysis."""
    
    query_parsing_time: float = 0.0
    temporal_analysis_time: float = 0.0
    vector_search_time: float = 0.0
    graph_traversal_time: float = 0.0
    pattern_matching_time: float = 0.0
    aggregation_time: float = 0.0
    total_time: float = 0.0
    
    temporal_nodes_analyzed: int = 0
    time_windows_processed: int = 0
    patterns_detected: int = 0
    relationships_traversed: int = 0
    
    temporal_query_type: Optional[TemporalQueryType] = None
    time_range_days: Optional[int] = None
    aggregation_granularity: Optional[str] = None


@dataclass
class TemporalContext:
    """Context information for temporal queries."""
    
    time_range: Tuple[datetime, datetime]
    granularity: str  # 'day', 'week', 'month', 'quarter', 'year'
    pattern_types: List[PatternType]
    aggregation_types: List[TemporalAggregationType]
    weight_decay_function: TemporalWeightDecayFunction
    baseline_period: Optional[Tuple[datetime, datetime]] = None
    comparison_periods: List[Tuple[datetime, datetime]] = None


class EHSHybridCypherRetriever(BaseRetriever, EHSSchemaAware):
    """
    Advanced HybridCypher retriever for complex temporal EHS analytics queries.
    
    Features:
    - Combined vector, fulltext, and graph traversal with temporal awareness
    - Time-based query routing and optimization
    - Temporal relationship traversal (e.g., incidents before permit expiration)
    - Time window filtering and aggregation
    - Historical pattern matching and trend analysis
    - Seasonal pattern detection and anomaly identification
    - Event sequence pattern recognition
    """
    
    def __init__(
        self,
        neo4j_driver,
        config: Optional[Dict[str, Any]] = None,
        vector_retriever: Optional[EHSVectorRetriever] = None,
        text2cypher_retriever: Optional[EHSText2CypherRetriever] = None
    ):
        """
        Initialize the EHS HybridCypher Retriever.
        
        Args:
            neo4j_driver: Neo4j database driver
            config: Configuration dictionary
            vector_retriever: Pre-configured vector retriever (optional)
            text2cypher_retriever: Pre-configured text2cypher retriever (optional)
        """
        super().__init__(config or {})
        self.neo4j_driver = neo4j_driver
        
        # Initialize temporal configuration
        self.hybrid_cypher_config = HybridCypherConfig(**(config or {}))
        
        # Initialize component retrievers
        self._init_component_retrievers(vector_retriever, text2cypher_retriever)
        
        # Initialize temporal pattern analyzer
        self.temporal_analyzer = TemporalPatternAnalyzer(
            config=self.hybrid_cypher_config.temporal_config
        )
        
        # Initialize GraphRAG VectorCypher retriever
        self._init_vector_cypher_retriever()
        
        # Performance tracking
        self.search_metrics = []
        
        logger.info("Initialized EHS HybridCypher Retriever with temporal analytics")
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get the retrieval strategy identifier."""
        return RetrievalStrategy.HYBRID_CYPHER
    
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
                    "max_results": self.hybrid_cypher_config.vector_top_k,
                    "similarity_threshold": self.hybrid_cypher_config.min_vector_score
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
                    "max_results": self.hybrid_cypher_config.graph_top_k,
                    "model_name": self.config.get("llm_model", "gpt-4"),
                    "enable_temporal_patterns": True
                }
                self.text2cypher_retriever = EHSText2CypherRetriever(
                    neo4j_driver=self.neo4j_driver,
                    config=text2cypher_config
                )
            
            logger.info("Component retrievers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize component retrievers: {e}")
            raise
    
    def _init_vector_cypher_retriever(self):
        """Initialize neo4j-graphrag-python VectorCypherRetriever."""
        try:
            # Configure embeddings
            embedder = OpenAIEmbeddings(
                model=self.config.get("embedding_model", "text-embedding-ada-002")
            )
            
            # Initialize VectorCypher retriever
            self.vector_cypher = VectorCypherRetriever(
                driver=self.neo4j_driver,
                index_name="ehs_document_chunks",
                embedder=embedder,
                return_properties=["content", "metadata", "document_id", "chunk_id", "document_type", "timestamp"],
                result_formatter=None
            )
            
            logger.info("VectorCypher Retriever initialized successfully")
            
        except Exception as e:
            logger.warning(f"VectorCypher Retriever not available: {e}")
            self.vector_cypher = None
    
    async def initialize(self) -> None:
        """Initialize the hybrid cypher retriever and its components."""
        try:
            # Initialize component retrievers
            if hasattr(self.vector_retriever, 'initialize'):
                await self.vector_retriever.initialize()
            
            if hasattr(self.text2cypher_retriever, 'initialize'):
                await self.text2cypher_retriever.initialize()
            
            # Initialize temporal analyzer
            await self.temporal_analyzer.initialize()
            
            self._initialized = True
            logger.info("HybridCypher retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HybridCypher retriever: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        limit: int = 10,
        **kwargs
    ) -> RetrievalResult:
        """
        Execute HybridCypher retrieval with temporal awareness.
        
        Args:
            query: Natural language query
            query_type: Type of EHS query
            limit: Maximum number of results to return
            **kwargs: Additional retrieval parameters including:
                - time_range: Tuple of (start_date, end_date)
                - granularity: Time granularity ('day', 'week', 'month', etc.)
                - include_trends: Whether to include trend analysis
                - pattern_types: List of patterns to detect
                - aggregation_types: List of aggregation methods
            
        Returns:
            RetrievalResult with temporal analytics and comprehensive metadata
        """
        start_time = datetime.now()
        metrics = TemporalSearchMetrics()
        
        try:
            # Step 1: Parse temporal aspects of the query
            parsing_start = datetime.now()
            temporal_context = await self._parse_temporal_query(query, query_type, **kwargs)
            temporal_query_type = await self._classify_temporal_query(query, temporal_context)
            metrics.query_parsing_time = (datetime.now() - parsing_start).total_seconds()
            metrics.temporal_query_type = temporal_query_type
            
            # Step 2: Analyze temporal patterns and requirements
            analysis_start = datetime.now()
            pattern_requirements = await self.temporal_analyzer.analyze_query_requirements(
                query, temporal_context, temporal_query_type
            )
            metrics.temporal_analysis_time = (datetime.now() - analysis_start).total_seconds()
            
            # Step 3: Execute retrieval strategy based on temporal query type
            results = await self._execute_temporal_retrieval(
                query, query_type, temporal_context, pattern_requirements, limit, metrics, **kwargs
            )
            
            # Step 4: Apply temporal post-processing
            processed_results = await self._apply_temporal_post_processing(
                results, temporal_context, pattern_requirements, metrics
            )
            
            # Step 5: Record metrics and return results
            metrics.total_time = (datetime.now() - start_time).total_seconds()
            self.search_metrics.append(metrics)
            
            # Enhance result metadata with temporal information
            enhanced_metadata = self._enhance_temporal_metadata(
                processed_results.metadata, metrics, temporal_context
            )
            processed_results.metadata = enhanced_metadata
            
            logger.info(f"HybridCypher temporal retrieval completed: {len(processed_results.data)} results in {metrics.total_time:.2f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"HybridCypher temporal retrieval failed: {e}")
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
    
    async def _parse_temporal_query(
        self,
        query: str,
        query_type: QueryType,
        **kwargs
    ) -> TemporalContext:
        """Parse temporal aspects from the query and parameters."""
        # Extract time range
        time_range = kwargs.get("time_range")
        if not time_range:
            # Try to extract from query using NLP
            time_range = await self.temporal_analyzer.extract_time_range_from_query(query)
        
        if not time_range:
            # Default to last 6 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            time_range = (start_date, end_date)
        
        # Extract granularity
        granularity = kwargs.get("granularity", "month")
        if not granularity:
            granularity = await self.temporal_analyzer.infer_granularity_from_query(query, time_range)
        
        # Extract pattern types
        pattern_types = kwargs.get("pattern_types", [])
        if not pattern_types:
            pattern_types = await self.temporal_analyzer.infer_pattern_types_from_query(query, query_type)
        
        # Extract aggregation types
        aggregation_types = kwargs.get("aggregation_types", [])
        if not aggregation_types:
            aggregation_types = await self.temporal_analyzer.infer_aggregation_types_from_query(query, query_type)
        
        # Get weight decay function
        weight_decay_function = self.hybrid_cypher_config.get_temporal_decay_function(query_type)
        
        return TemporalContext(
            time_range=time_range,
            granularity=granularity,
            pattern_types=pattern_types,
            aggregation_types=aggregation_types,
            weight_decay_function=weight_decay_function,
            baseline_period=kwargs.get("baseline_period"),
            comparison_periods=kwargs.get("comparison_periods", [])
        )
    
    async def _classify_temporal_query(
        self,
        query: str,
        temporal_context: TemporalContext
    ) -> TemporalQueryType:
        """Classify the type of temporal query being made."""
        query_lower = query.lower()
        
        # Trend analysis keywords
        if any(keyword in query_lower for keyword in ["trend", "increase", "decrease", "growth", "decline", "over time"]):
            return TemporalQueryType.TREND_ANALYSIS
        
        # Pattern detection keywords
        if any(keyword in query_lower for keyword in ["pattern", "cycle", "seasonal", "recurring", "regular"]):
            return TemporalQueryType.PATTERN_DETECTION
        
        # Correlation analysis keywords
        if any(keyword in query_lower for keyword in ["correlation", "relationship", "impact", "effect", "influence"]):
            return TemporalQueryType.CORRELATION_ANALYSIS
        
        # Anomaly detection keywords
        if any(keyword in query_lower for keyword in ["anomaly", "unusual", "spike", "outlier", "unexpected"]):
            return TemporalQueryType.ANOMALY_DETECTION
        
        # Sequence analysis keywords
        if any(keyword in query_lower for keyword in ["before", "after", "sequence", "order", "timeline", "progression"]):
            return TemporalQueryType.SEQUENCE_ANALYSIS
        
        # Compliance timeline keywords
        if any(keyword in query_lower for keyword in ["compliance", "permit", "deadline", "expiration", "renewal"]):
            return TemporalQueryType.COMPLIANCE_TIMELINE
        
        # Risk progression keywords
        if any(keyword in query_lower for keyword in ["risk", "incident", "safety", "escalation", "deterioration"]):
            return TemporalQueryType.RISK_PROGRESSION
        
        # Default to trend analysis
        return TemporalQueryType.TREND_ANALYSIS
    
    async def _execute_temporal_retrieval(
        self,
        query: str,
        query_type: QueryType,
        temporal_context: TemporalContext,
        pattern_requirements: Dict[str, Any],
        limit: int,
        metrics: TemporalSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute temporal retrieval based on query type and context."""
        temporal_query_type = metrics.temporal_query_type
        
        if temporal_query_type == TemporalQueryType.TREND_ANALYSIS:
            return await self._execute_trend_analysis(
                query, query_type, temporal_context, limit, metrics, **kwargs
            )
        elif temporal_query_type == TemporalQueryType.PATTERN_DETECTION:
            return await self._execute_pattern_detection(
                query, query_type, temporal_context, limit, metrics, **kwargs
            )
        elif temporal_query_type == TemporalQueryType.CORRELATION_ANALYSIS:
            return await self._execute_correlation_analysis(
                query, query_type, temporal_context, limit, metrics, **kwargs
            )
        elif temporal_query_type == TemporalQueryType.SEQUENCE_ANALYSIS:
            return await self._execute_sequence_analysis(
                query, query_type, temporal_context, limit, metrics, **kwargs
            )
        elif temporal_query_type == TemporalQueryType.COMPLIANCE_TIMELINE:
            return await self._execute_compliance_timeline(
                query, query_type, temporal_context, limit, metrics, **kwargs
            )
        elif temporal_query_type == TemporalQueryType.RISK_PROGRESSION:
            return await self._execute_risk_progression(
                query, query_type, temporal_context, limit, metrics, **kwargs
            )
        else:
            # Default to general temporal retrieval
            return await self._execute_general_temporal_retrieval(
                query, query_type, temporal_context, limit, metrics, **kwargs
            )
    
    async def _execute_trend_analysis(
        self,
        query: str,
        query_type: QueryType,
        temporal_context: TemporalContext,
        limit: int,
        metrics: TemporalSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute trend analysis retrieval strategy."""
        # Build temporal Cypher query for trend analysis
        cypher_query = self._build_trend_analysis_cypher(query, temporal_context, query_type)
        
        # Execute graph traversal
        graph_start = datetime.now()
        graph_results = await self._execute_cypher_with_temporal_weights(
            cypher_query, temporal_context, metrics
        )
        metrics.graph_traversal_time = (datetime.now() - graph_start).total_seconds()
        
        # Combine with vector search for semantic relevance
        vector_start = datetime.now()
        vector_results = await self._execute_temporal_vector_search(
            query, temporal_context, limit, **kwargs
        )
        metrics.vector_search_time = (datetime.now() - vector_start).total_seconds()
        
        # Merge and rank results
        merged_results = await self._merge_temporal_results(
            graph_results, vector_results, temporal_context, query_type
        )
        
        return RetrievalResult(
            success=True,
            data=merged_results[:limit],
            metadata=RetrievalMetadata(
                strategy=self.get_strategy(),
                query_type=query_type,
                confidence_score=self._calculate_temporal_confidence(merged_results),
                execution_time_ms=0,  # Will be updated later
                nodes_retrieved=len(merged_results),
                cypher_query=cypher_query
            )
        )
    
    async def _execute_pattern_detection(
        self,
        query: str,
        query_type: QueryType,
        temporal_context: TemporalContext,
        limit: int,
        metrics: TemporalSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute pattern detection retrieval strategy."""
        pattern_start = datetime.now()
        
        # Use temporal analyzer to detect patterns
        patterns = await self.temporal_analyzer.detect_patterns(
            query, temporal_context, query_type
        )
        
        # Build Cypher query incorporating detected patterns
        cypher_query = self._build_pattern_detection_cypher(patterns, temporal_context)
        
        # Execute pattern-aware graph traversal
        results = await self._execute_cypher_with_temporal_weights(
            cypher_query, temporal_context, metrics
        )
        
        metrics.pattern_matching_time = (datetime.now() - pattern_start).total_seconds()
        metrics.patterns_detected = len(patterns)
        
        return RetrievalResult(
            success=True,
            data=results[:limit],
            metadata=RetrievalMetadata(
                strategy=self.get_strategy(),
                query_type=query_type,
                confidence_score=self._calculate_temporal_confidence(results),
                execution_time_ms=0,  # Will be updated later
                nodes_retrieved=len(results),
                cypher_query=cypher_query
            )
        )
    
    async def _execute_sequence_analysis(
        self,
        query: str,
        query_type: QueryType,
        temporal_context: TemporalContext,
        limit: int,
        metrics: TemporalSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute sequence analysis for queries about temporal relationships."""
        # Example: "Show emission trends for facilities that had safety incidents in the past 6 months"
        
        # Parse sequence requirements from query
        sequence_requirements = await self.temporal_analyzer.parse_sequence_requirements(query)
        
        # Build Cypher query for sequence analysis
        cypher_query = self._build_sequence_analysis_cypher(
            sequence_requirements, temporal_context, query_type
        )
        
        # Execute with temporal relationship traversal
        results = await self._execute_cypher_with_temporal_weights(
            cypher_query, temporal_context, metrics
        )
        
        return RetrievalResult(
            success=True,
            data=results[:limit],
            metadata=RetrievalMetadata(
                strategy=self.get_strategy(),
                query_type=query_type,
                confidence_score=self._calculate_temporal_confidence(results),
                execution_time_ms=0,  # Will be updated later
                nodes_retrieved=len(results),
                cypher_query=cypher_query
            )
        )
    
    async def _execute_compliance_timeline(
        self,
        query: str,
        query_type: QueryType,
        temporal_context: TemporalContext,
        limit: int,
        metrics: TemporalSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute compliance timeline analysis."""
        # Example: "Find equipment with increasing failure rates near permit renewal dates"
        
        # Build compliance-focused Cypher query
        cypher_query = self._build_compliance_timeline_cypher(temporal_context, query_type)
        
        # Execute with compliance-specific temporal weights
        results = await self._execute_cypher_with_temporal_weights(
            cypher_query, temporal_context, metrics, compliance_focused=True
        )
        
        return RetrievalResult(
            success=True,
            data=results[:limit],
            metadata=RetrievalMetadata(
                strategy=self.get_strategy(),
                query_type=query_type,
                confidence_score=self._calculate_temporal_confidence(results),
                execution_time_ms=0,  # Will be updated later
                nodes_retrieved=len(results),
                cypher_query=cypher_query
            )
        )
    
    async def _execute_correlation_analysis(
        self,
        query: str,
        query_type: QueryType,
        temporal_context: TemporalContext,
        limit: int,
        metrics: TemporalSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute correlation analysis between temporal events."""
        # Build correlation-focused Cypher query
        cypher_query = self._build_correlation_analysis_cypher(query, temporal_context, query_type)
        
        # Execute with correlation analysis
        results = await self._execute_cypher_with_temporal_weights(
            cypher_query, temporal_context, metrics
        )
        
        return RetrievalResult(
            success=True,
            data=results[:limit],
            metadata=RetrievalMetadata(
                strategy=self.get_strategy(),
                query_type=query_type,
                confidence_score=self._calculate_temporal_confidence(results),
                execution_time_ms=0,  # Will be updated later
                nodes_retrieved=len(results),
                cypher_query=cypher_query
            )
        )
    
    async def _execute_risk_progression(
        self,
        query: str,
        query_type: QueryType,
        temporal_context: TemporalContext,
        limit: int,
        metrics: TemporalSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute risk progression analysis."""
        # Build risk-focused Cypher query
        cypher_query = self._build_risk_progression_cypher(query, temporal_context, query_type)
        
        # Execute with risk-specific temporal analysis
        results = await self._execute_cypher_with_temporal_weights(
            cypher_query, temporal_context, metrics, risk_focused=True
        )
        
        return RetrievalResult(
            success=True,
            data=results[:limit],
            metadata=RetrievalMetadata(
                strategy=self.get_strategy(),
                query_type=query_type,
                confidence_score=self._calculate_temporal_confidence(results),
                execution_time_ms=0,  # Will be updated later
                nodes_retrieved=len(results),
                cypher_query=cypher_query
            )
        )
    
    async def _execute_general_temporal_retrieval(
        self,
        query: str,
        query_type: QueryType,
        temporal_context: TemporalContext,
        limit: int,
        metrics: TemporalSearchMetrics,
        **kwargs
    ) -> RetrievalResult:
        """Execute general temporal retrieval strategy."""
        # Combine all approaches with balanced weights
        tasks = []
        
        # Vector search with temporal filtering
        if self.hybrid_cypher_config.enable_vector_search:
            tasks.append(self._execute_temporal_vector_search(query, temporal_context, limit, **kwargs))
        
        # Graph traversal with temporal relationships
        if self.hybrid_cypher_config.enable_graph_traversal:
            cypher_query = self._build_general_temporal_cypher(query, temporal_context, query_type)
            tasks.append(self._execute_cypher_with_temporal_weights(cypher_query, temporal_context, metrics))
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        merged_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Temporal retrieval task failed: {result}")
                continue
            if isinstance(result, list):
                merged_results.extend(result)
            elif hasattr(result, 'data'):
                merged_results.extend(result.data)
        
        # Apply temporal ranking
        ranked_results = await self._apply_temporal_ranking(
            merged_results, temporal_context, query_type
        )
        
        return RetrievalResult(
            success=True,
            data=ranked_results[:limit],
            metadata=RetrievalMetadata(
                strategy=self.get_strategy(),
                query_type=query_type,
                confidence_score=self._calculate_temporal_confidence(ranked_results),
                execution_time_ms=0,  # Will be updated later
                nodes_retrieved=len(ranked_results)
            )
        )
    
    def _build_trend_analysis_cypher(
        self,
        query: str,
        temporal_context: TemporalContext,
        query_type: QueryType
    ) -> str:
        """Build Cypher query for trend analysis."""
        start_date = temporal_context.time_range[0].isoformat()
        end_date = temporal_context.time_range[1].isoformat()
        granularity = temporal_context.granularity
        
        # Get relevant node types for the query
        relevant_nodes = self.get_relevant_nodes(query_type)
        
        # Build base query with temporal grouping
        if query_type == QueryType.CONSUMPTION:
            cypher = f"""
            MATCH (f:Facility)-[:HAS_UTILITY_BILL]->(ub:UtilityBill)
            WHERE ub.billing_period >= date('{start_date}') 
              AND ub.billing_period <= date('{end_date}')
            WITH f, ub, 
                 apoc.date.format(ub.billing_period.epochMillis, 'ms', 'yyyy-MM') as time_period
            RETURN f.name as facility,
                   ub.utility_type as utility_type,
                   time_period,
                   sum(ub.amount) as total_consumption,
                   avg(ub.amount) as avg_consumption,
                   count(ub) as record_count
            ORDER BY time_period, facility
            """
        elif query_type == QueryType.EMISSIONS:
            cypher = f"""
            MATCH (f:Facility)-[:HAS_EMISSION]->(e:Emission)
            WHERE e.measurement_date >= date('{start_date}')
              AND e.measurement_date <= date('{end_date}')
            WITH f, e,
                 apoc.date.format(e.measurement_date.epochMillis, 'ms', 'yyyy-MM') as time_period
            RETURN f.name as facility,
                   e.emission_type as emission_type,
                   time_period,
                   sum(e.amount) as total_emissions,
                   avg(e.amount) as avg_emissions,
                   count(e) as record_count
            ORDER BY time_period, facility
            """
        else:
            # General temporal query
            cypher = f"""
            MATCH (n)
            WHERE n.created_at >= date('{start_date}')
              AND n.created_at <= date('{end_date}')
              AND labels(n)[0] IN {relevant_nodes}
            WITH n, 
                 apoc.date.format(n.created_at.epochMillis, 'ms', 'yyyy-MM') as time_period
            RETURN labels(n)[0] as node_type,
                   time_period,
                   count(n) as record_count,
                   collect(n.id)[0..5] as sample_ids
            ORDER BY time_period, node_type
            """
        
        return cypher
    
    def _build_sequence_analysis_cypher(
        self,
        sequence_requirements: Dict[str, Any],
        temporal_context: TemporalContext,
        query_type: QueryType
    ) -> str:
        """Build Cypher query for sequence analysis."""
        start_date = temporal_context.time_range[0].isoformat()
        end_date = temporal_context.time_range[1].isoformat()
        
        # Example: incidents before emissions patterns
        cypher = f"""
        MATCH (f:Facility)-[:OCCURRED_AT]-(i:Incident)
        WHERE i.date >= date('{start_date}') AND i.date <= date('{end_date}')
        WITH f, i
        MATCH (f)-[:HAS_EMISSION]->(e:Emission)
        WHERE e.measurement_date > i.date 
          AND e.measurement_date <= date(i.date) + duration('P6M')
        RETURN f.name as facility,
               i.type as incident_type,
               i.date as incident_date,
               e.emission_type as emission_type,
               e.measurement_date as emission_date,
               e.amount as emission_amount,
               duration.between(i.date, e.measurement_date).days as days_after_incident
        ORDER BY facility, incident_date, emission_date
        """
        
        return cypher
    
    def _build_compliance_timeline_cypher(
        self,
        temporal_context: TemporalContext,
        query_type: QueryType
    ) -> str:
        """Build Cypher query for compliance timeline analysis."""
        start_date = temporal_context.time_range[0].isoformat()
        end_date = temporal_context.time_range[1].isoformat()
        
        # Equipment failure rates near permit renewal
        cypher = f"""
        MATCH (f:Facility)-[:HAS_PERMIT]->(p:Permit)
        WHERE p.expiry_date >= date('{start_date}') 
          AND p.expiry_date <= date('{end_date}')
        WITH f, p
        MATCH (f)-[:HAS_EQUIPMENT]->(eq:Equipment)
        OPTIONAL MATCH (eq)-[:INVOLVED_IN]->(i:Incident)
        WHERE i.date >= date(p.expiry_date) - duration('P3M')
          AND i.date <= p.expiry_date
        WITH f, p, eq, count(i) as incident_count,
             duration.between(date(p.expiry_date) - duration('P3M'), p.expiry_date).days as monitoring_days
        WHERE incident_count > 0
        RETURN f.name as facility,
               p.permit_number as permit_number,
               p.type as permit_type,
               p.expiry_date as expiry_date,
               eq.name as equipment_name,
               eq.type as equipment_type,
               incident_count,
               (incident_count * 1.0 / monitoring_days * 30) as monthly_failure_rate
        ORDER BY monthly_failure_rate DESC, expiry_date
        """
        
        return cypher
    
    def _build_pattern_detection_cypher(
        self,
        patterns: List[Dict[str, Any]],
        temporal_context: TemporalContext
    ) -> str:
        """Build Cypher query for pattern detection."""
        start_date = temporal_context.time_range[0].isoformat()
        end_date = temporal_context.time_range[1].isoformat()
        
        # Seasonal consumption patterns
        cypher = f"""
        MATCH (f:Facility)-[:HAS_UTILITY_BILL]->(ub:UtilityBill)
        WHERE ub.billing_period >= date('{start_date}')
          AND ub.billing_period <= date('{end_date}')
        WITH f, ub,
             ub.billing_period.month as month,
             ub.billing_period.quarter as quarter
        RETURN f.name as facility,
               ub.utility_type as utility_type,
               month,
               quarter,
               avg(ub.amount) as avg_monthly_consumption,
               stddev(ub.amount) as consumption_variance,
               count(ub) as record_count
        ORDER BY facility, utility_type, month
        """
        
        return cypher
    
    def _build_correlation_analysis_cypher(
        self,
        query: str,
        temporal_context: TemporalContext,
        query_type: QueryType
    ) -> str:
        """Build Cypher query for correlation analysis."""
        start_date = temporal_context.time_range[0].isoformat()
        end_date = temporal_context.time_range[1].isoformat()
        
        # Correlation between consumption and emissions
        cypher = f"""
        MATCH (f:Facility)-[:HAS_UTILITY_BILL]->(ub:UtilityBill),
              (f)-[:HAS_EMISSION]->(e:Emission)
        WHERE ub.billing_period >= date('{start_date}')
          AND ub.billing_period <= date('{end_date}')
          AND e.measurement_date >= date('{start_date}')
          AND e.measurement_date <= date('{end_date}')
          AND ub.billing_period.month = e.measurement_date.month
          AND ub.billing_period.year = e.measurement_date.year
        RETURN f.name as facility,
               ub.billing_period.year as year,
               ub.billing_period.month as month,
               sum(ub.amount) as total_consumption,
               sum(e.amount) as total_emissions,
               corr(ub.amount, e.amount) as correlation_coefficient
        ORDER BY facility, year, month
        """
        
        return cypher
    
    def _build_risk_progression_cypher(
        self,
        query: str,
        temporal_context: TemporalContext,
        query_type: QueryType
    ) -> str:
        """Build Cypher query for risk progression analysis."""
        start_date = temporal_context.time_range[0].isoformat()
        end_date = temporal_context.time_range[1].isoformat()
        
        # Risk escalation patterns
        cypher = f"""
        MATCH (f:Facility)-[:OCCURRED_AT]-(i:Incident)
        WHERE i.date >= date('{start_date}')
          AND i.date <= date('{end_date}')
        WITH f, i
        ORDER BY f, i.date
        WITH f, collect(i) as incidents
        UNWIND range(0, size(incidents)-2) as idx
        WITH f, incidents[idx] as current_incident, incidents[idx+1] as next_incident
        WHERE current_incident.severity < next_incident.severity
        RETURN f.name as facility,
               current_incident.date as initial_date,
               current_incident.severity as initial_severity,
               next_incident.date as escalation_date,
               next_incident.severity as escalated_severity,
               duration.between(current_incident.date, next_incident.date).days as escalation_days
        ORDER BY facility, initial_date
        """
        
        return cypher
    
    def _build_general_temporal_cypher(
        self,
        query: str,
        temporal_context: TemporalContext,
        query_type: QueryType
    ) -> str:
        """Build general temporal Cypher query."""
        start_date = temporal_context.time_range[0].isoformat()
        end_date = temporal_context.time_range[1].isoformat()
        relevant_nodes = self.get_relevant_nodes(query_type)
        
        cypher = f"""
        MATCH (n)
        WHERE labels(n)[0] IN {relevant_nodes}
          AND EXISTS(n.created_at)
          AND n.created_at >= date('{start_date}')
          AND n.created_at <= date('{end_date}')
        OPTIONAL MATCH (n)-[r]-(related)
        RETURN n, labels(n)[0] as node_type, n.created_at as timestamp,
               collect(DISTINCT {{ type: type(r), node: related.id, timestamp: related.created_at }}) as relationships
        ORDER BY n.created_at DESC
        LIMIT 100
        """
        
        return cypher
    
    async def _execute_cypher_with_temporal_weights(
        self,
        cypher_query: str,
        temporal_context: TemporalContext,
        metrics: TemporalSearchMetrics,
        compliance_focused: bool = False,
        risk_focused: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query and apply temporal weights."""
        try:
            async with self.neo4j_driver.session() as session:
                result = await session.run(cypher_query)
                records = await result.data()
                
                metrics.temporal_nodes_analyzed = len(records)
                metrics.relationships_traversed = sum(
                    len(record.get('relationships', [])) for record in records
                )
                
                # Apply temporal decay weights
                weighted_records = []
                for record in records:
                    weight = self._calculate_temporal_weight(
                        record, temporal_context, compliance_focused, risk_focused
                    )
                    record['temporal_weight'] = weight
                    record['temporal_score'] = weight * record.get('score', 1.0)
                    weighted_records.append(record)
                
                # Sort by temporal score
                weighted_records.sort(key=lambda x: x['temporal_score'], reverse=True)
                
                return weighted_records
                
        except Exception as e:
            logger.error(f"Failed to execute temporal Cypher query: {e}")
            return []
    
    async def _execute_temporal_vector_search(
        self,
        query: str,
        temporal_context: TemporalContext,
        limit: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute vector search with temporal filtering."""
        try:
            # Add temporal constraints to vector search
            temporal_kwargs = {
                **kwargs,
                "time_range": temporal_context.time_range,
                "limit": limit
            }
            
            if self.vector_retriever:
                result = await self.vector_retriever.retrieve(
                    query=query, 
                    **temporal_kwargs
                )
                
                if result.success:
                    # Apply temporal weights to vector results
                    for item in result.data:
                        weight = self._calculate_temporal_weight(
                            item, temporal_context
                        )
                        item['temporal_weight'] = weight
                        item['temporal_score'] = weight * item.get('score', 0.0)
                    
                    return result.data
            
            return []
            
        except Exception as e:
            logger.error(f"Temporal vector search failed: {e}")
            return []
    
    def _calculate_temporal_weight(
        self,
        record: Dict[str, Any],
        temporal_context: TemporalContext,
        compliance_focused: bool = False,
        risk_focused: bool = False
    ) -> float:
        """Calculate temporal weight for a record based on time decay and context."""
        # Extract timestamp from record
        timestamp = None
        if 'timestamp' in record:
            timestamp = record['timestamp']
        elif 'created_at' in record:
            timestamp = record['created_at']
        elif 'date' in record:
            timestamp = record['date']
        elif 'measurement_date' in record:
            timestamp = record['measurement_date']
        
        if not timestamp:
            return 1.0  # Default weight if no timestamp
        
        # Convert to datetime if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return 1.0
        
        # Calculate time difference
        current_time = datetime.now()
        time_diff = (current_time - timestamp).total_seconds()
        
        # Apply decay function
        decay_function = temporal_context.weight_decay_function
        
        if decay_function == TemporalWeightDecayFunction.LINEAR:
            max_age = (temporal_context.time_range[1] - temporal_context.time_range[0]).total_seconds()
            weight = max(0.1, 1.0 - (time_diff / max_age))
        elif decay_function == TemporalWeightDecayFunction.EXPONENTIAL:
            # Exponential decay with half-life of 30 days
            half_life = 30 * 24 * 3600  # 30 days in seconds
            weight = 0.5 ** (time_diff / half_life)
        elif decay_function == TemporalWeightDecayFunction.LOGARITHMIC:
            # Logarithmic decay
            weight = 1.0 / (1.0 + np.log(1.0 + time_diff / (24 * 3600)))
        else:
            weight = 1.0
        
        # Apply context-specific boosts
        if compliance_focused:
            # Boost records near compliance deadlines
            if 'expiry_date' in record:
                try:
                    expiry_date = datetime.fromisoformat(str(record['expiry_date']))
                    days_to_expiry = (expiry_date - current_time).days
                    if 0 <= days_to_expiry <= 90:  # Within 3 months
                        weight *= 1.5
                except:
                    pass
        
        if risk_focused:
            # Boost high-severity incidents
            severity = record.get('severity', record.get('escalated_severity', 0))
            if severity and severity > 3:
                weight *= 1.3
        
        return max(0.1, min(2.0, weight))  # Clamp between 0.1 and 2.0
    
    async def _merge_temporal_results(
        self,
        graph_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        temporal_context: TemporalContext,
        query_type: QueryType
    ) -> List[Dict[str, Any]]:
        """Merge graph and vector results with temporal ranking."""
        all_results = []
        
        # Add source information
        for result in graph_results:
            result['source'] = 'graph'
            all_results.append(result)
        
        for result in vector_results:
            result['source'] = 'vector'
            all_results.append(result)
        
        # Remove duplicates based on content similarity
        deduplicated = await self._deduplicate_temporal_results(all_results)
        
        # Apply final temporal ranking
        ranked_results = await self._apply_temporal_ranking(
            deduplicated, temporal_context, query_type
        )
        
        return ranked_results
    
    async def _deduplicate_temporal_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        if len(results) <= 1:
            return results
        
        deduplicated = []
        seen_content = set()
        
        for result in results:
            # Create a simple hash of key content
            content_key = str(result.get('facility', '')) + str(result.get('equipment_name', '')) + str(result.get('timestamp', ''))
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated.append(result)
        
        return deduplicated
    
    async def _apply_temporal_ranking(
        self,
        results: List[Dict[str, Any]],
        temporal_context: TemporalContext,
        query_type: QueryType
    ) -> List[Dict[str, Any]]:
        """Apply final temporal ranking to results."""
        # Combine temporal score with relevance score
        for result in results:
            temporal_score = result.get('temporal_score', 0.0)
            relevance_score = result.get('score', 0.0)
            
            # Weight combination based on query type
            if query_type in [QueryType.COMPLIANCE, QueryType.RISK]:
                # Prioritize temporal relevance for compliance and risk
                final_score = 0.7 * temporal_score + 0.3 * relevance_score
            else:
                # Balanced approach for other query types
                final_score = 0.5 * temporal_score + 0.5 * relevance_score
            
            result['final_score'] = final_score
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    async def _apply_temporal_post_processing(
        self,
        results: RetrievalResult,
        temporal_context: TemporalContext,
        pattern_requirements: Dict[str, Any],
        metrics: TemporalSearchMetrics
    ) -> RetrievalResult:
        """Apply temporal post-processing to enhance results."""
        if not results.success or not results.data:
            return results
        
        post_processing_start = datetime.now()
        
        # Apply aggregations if requested
        if temporal_context.aggregation_types:
            aggregated_data = await self._apply_temporal_aggregations(
                results.data, temporal_context, metrics
            )
            results.data = aggregated_data
        
        # Add trend information
        results.data = await self._add_trend_information(
            results.data, temporal_context
        )
        
        # Add pattern analysis results
        results.data = await self._add_pattern_analysis(
            results.data, temporal_context, pattern_requirements
        )
        
        metrics.aggregation_time = (datetime.now() - post_processing_start).total_seconds()
        
        return results
    
    async def _apply_temporal_aggregations(
        self,
        data: List[Dict[str, Any]],
        temporal_context: TemporalContext,
        metrics: TemporalSearchMetrics
    ) -> List[Dict[str, Any]]:
        """Apply temporal aggregations to the data."""
        aggregated = []
        
        for aggregation_type in temporal_context.aggregation_types:
            if aggregation_type == TemporalAggregationType.SUM:
                # Group by time period and sum values
                time_groups = {}
                for item in data:
                    time_key = item.get('time_period', 'unknown')
                    if time_key not in time_groups:
                        time_groups[time_key] = []
                    time_groups[time_key].append(item)
                
                for time_key, items in time_groups.items():
                    total_value = sum(item.get('total_consumption', item.get('total_emissions', 0)) for item in items)
                    aggregated.append({
                        'time_period': time_key,
                        'aggregation_type': 'sum',
                        'value': total_value,
                        'count': len(items)
                    })
            
            # Add other aggregation types as needed
        
        return aggregated if aggregated else data
    
    async def _add_trend_information(
        self,
        data: List[Dict[str, Any]],
        temporal_context: TemporalContext
    ) -> List[Dict[str, Any]]:
        """Add trend analysis information to results."""
        # Simple trend calculation for time series data
        if len(data) < 2:
            return data
        
        # Group by facility/entity
        entity_groups = {}
        for item in data:
            entity_key = item.get('facility', item.get('equipment_name', 'unknown'))
            if entity_key not in entity_groups:
                entity_groups[entity_key] = []
            entity_groups[entity_key].append(item)
        
        # Calculate trends for each entity
        for entity_key, items in entity_groups.items():
            if len(items) >= 2:
                # Sort by time
                items.sort(key=lambda x: x.get('timestamp', x.get('time_period', '')))
                
                # Calculate simple trend
                values = [item.get('total_consumption', item.get('total_emissions', 0)) for item in items]
                if len(values) >= 2:
                    trend = (values[-1] - values[0]) / len(values)
                    trend_direction = "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable"
                    
                    for item in items:
                        item['trend_direction'] = trend_direction
                        item['trend_magnitude'] = abs(trend)
        
        return data
    
    async def _add_pattern_analysis(
        self,
        data: List[Dict[str, Any]],
        temporal_context: TemporalContext,
        pattern_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Add pattern analysis information to results."""
        # Add detected patterns to each result
        for item in data:
            item['detected_patterns'] = []
            
            # Check for seasonal patterns
            if 'month' in item:
                month = item['month']
                if month in [12, 1, 2]:  # Winter
                    item['detected_patterns'].append('winter_seasonal')
                elif month in [6, 7, 8]:  # Summer
                    item['detected_patterns'].append('summer_seasonal')
            
            # Check for anomalies (simple threshold-based)
            value = item.get('total_consumption', item.get('total_emissions', 0))
            if value > 0:
                # Simple anomaly detection based on deviation from mean
                all_values = [d.get('total_consumption', d.get('total_emissions', 0)) for d in data]
                mean_value = np.mean(all_values) if all_values else 0
                std_value = np.std(all_values) if all_values else 0
                
                if std_value > 0 and abs(value - mean_value) > 2 * std_value:
                    item['detected_patterns'].append('anomaly')
        
        return data
    
    def _calculate_temporal_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for temporal results."""
        if not results:
            return 0.0
        
        # Factors affecting temporal confidence:
        # 1. Number of temporal data points
        # 2. Temporal coverage (how well the time range is covered)
        # 3. Pattern consistency
        # 4. Data quality (completeness)
        
        num_results = len(results)
        result_count_factor = min(1.0, num_results / 20.0)  # Normalize to 20 results
        
        # Check temporal coverage
        timestamps = []
        for result in results:
            ts = result.get('timestamp', result.get('time_period'))
            if ts:
                timestamps.append(ts)
        
        coverage_factor = 1.0
        if timestamps:
            coverage_factor = min(1.0, len(set(timestamps)) / 12.0)  # Normalize to 12 time periods
        
        # Check pattern consistency (if patterns detected)
        pattern_factor = 1.0
        patterns_detected = sum(1 for r in results if r.get('detected_patterns'))
        if patterns_detected > 0:
            pattern_factor = min(1.0, patterns_detected / num_results)
        
        # Combine factors
        confidence = (result_count_factor * 0.4) + (coverage_factor * 0.4) + (pattern_factor * 0.2)
        
        return min(1.0, max(0.0, confidence))
    
    def _enhance_temporal_metadata(
        self, 
        base_metadata: RetrievalMetadata, 
        metrics: TemporalSearchMetrics,
        temporal_context: TemporalContext
    ) -> RetrievalMetadata:
        """Enhance metadata with temporal information."""
        enhanced = RetrievalMetadata(
            strategy=base_metadata.strategy,
            query_type=base_metadata.query_type,
            confidence_score=base_metadata.confidence_score,
            execution_time_ms=metrics.total_time * 1000,
            nodes_retrieved=base_metadata.nodes_retrieved,
            relationships_retrieved=metrics.relationships_traversed,
            cypher_query=base_metadata.cypher_query,
            error_message=base_metadata.error_message
        )
        
        # Add temporal-specific metadata
        enhanced.temporal_metadata = {
            "temporal_metrics": {
                "query_parsing_time": metrics.query_parsing_time,
                "temporal_analysis_time": metrics.temporal_analysis_time,
                "vector_search_time": metrics.vector_search_time,
                "graph_traversal_time": metrics.graph_traversal_time,
                "pattern_matching_time": metrics.pattern_matching_time,
                "aggregation_time": metrics.aggregation_time
            },
            "temporal_context": {
                "time_range": {
                    "start": temporal_context.time_range[0].isoformat(),
                    "end": temporal_context.time_range[1].isoformat()
                },
                "granularity": temporal_context.granularity,
                "pattern_types": [pt.value for pt in temporal_context.pattern_types],
                "aggregation_types": [at.value for at in temporal_context.aggregation_types]
            },
            "analysis_results": {
                "temporal_query_type": metrics.temporal_query_type.value if metrics.temporal_query_type else None,
                "temporal_nodes_analyzed": metrics.temporal_nodes_analyzed,
                "time_windows_processed": metrics.time_windows_processed,
                "patterns_detected": metrics.patterns_detected,
                "relationships_traversed": metrics.relationships_traversed,
                "time_range_days": metrics.time_range_days
            }
        }
        
        return enhanced
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on HybridCypher retriever."""
        health_status = await super().health_check()
        
        # Check component retrievers
        vector_health = await self.vector_retriever.health_check() if self.vector_retriever else {"status": "not_available"}
        text2cypher_health = await self.text2cypher_retriever.health_check() if self.text2cypher_retriever else {"status": "not_available"}
        
        # Check temporal analyzer
        temporal_health = await self.temporal_analyzer.health_check() if hasattr(self.temporal_analyzer, 'health_check') else {"status": "available"}
        
        # Check VectorCypher
        vector_cypher_status = "available" if self.vector_cypher else "not_available"
        
        # Performance metrics
        recent_metrics = self.search_metrics[-10:] if self.search_metrics else []
        avg_response_time = np.mean([m.total_time for m in recent_metrics]) if recent_metrics else 0.0
        avg_temporal_analysis_time = np.mean([m.temporal_analysis_time for m in recent_metrics]) if recent_metrics else 0.0
        
        health_status.update({
            "component_health": {
                "vector_retriever": vector_health,
                "text2cypher_retriever": text2cypher_health,
                "temporal_analyzer": temporal_health,
                "vector_cypher": vector_cypher_status
            },
            "temporal_performance": {
                "recent_searches": len(recent_metrics),
                "avg_response_time": avg_response_time,
                "avg_temporal_analysis_time": avg_temporal_analysis_time,
                "temporal_queries_processed": sum(1 for m in recent_metrics if m.temporal_query_type)
            },
            "temporal_configuration": {
                "temporal_patterns_enabled": self.hybrid_cypher_config.enable_temporal_patterns,
                "vector_search_enabled": self.hybrid_cypher_config.enable_vector_search,
                "graph_traversal_enabled": self.hybrid_cypher_config.enable_graph_traversal,
                "pattern_detection_enabled": self.hybrid_cypher_config.enable_pattern_detection
            }
        })
        
        return health_status
    
    async def cleanup(self) -> None:
        """Clean up HybridCypher retriever resources."""
        try:
            # Cleanup component retrievers
            if self.vector_retriever and hasattr(self.vector_retriever, 'cleanup'):
                await self.vector_retriever.cleanup()
            
            if self.text2cypher_retriever and hasattr(self.text2cypher_retriever, 'cleanup'):
                await self.text2cypher_retriever.cleanup()
            
            # Cleanup temporal analyzer
            if hasattr(self.temporal_analyzer, 'cleanup'):
                await self.temporal_analyzer.cleanup()
            
            # Clear metrics
            self.search_metrics.clear()
            
            await super().cleanup()
            logger.info("HybridCypher retriever cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during HybridCypher retriever cleanup: {e}")


# Convenience function for easy initialization
def create_ehs_hybrid_cypher_retriever(
    neo4j_driver,
    config: Optional[Dict[str, Any]] = None
) -> EHSHybridCypherRetriever:
    """
    Create and initialize an EHS HybridCypher Retriever with default configuration.
    
    Args:
        neo4j_driver: Neo4j database driver
        config: Optional configuration overrides
        
    Returns:
        Configured EHSHybridCypherRetriever instance
    """
    default_config = {
        "max_results": 15,
        "enable_temporal_patterns": True,
        "enable_vector_search": True,
        "enable_graph_traversal": True,
        "enable_pattern_detection": True,
        "default_time_window_days": 180,
        "default_granularity": "month"
    }
    
    if config:
        default_config.update(config)
    
    return EHSHybridCypherRetriever(
        neo4j_driver=neo4j_driver,
        config=default_config
    )