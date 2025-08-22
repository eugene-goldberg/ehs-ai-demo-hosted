"""
EHS-specific Text2Cypher retriever implementation.

This module extends the base Text2Cypher retriever with EHS domain expertise,
including specialized query patterns, validation, and optimization for common
EHS use cases using neo4j-graphrag capabilities.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import re
from enum import Enum

from neo4j_graphrag.retrievers import Text2CypherRetriever as GraphRAGRetriever
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.generation.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

from ..base import (
    BaseRetriever,
    RetrievalStrategy,
    QueryType,
    RetrievalResult,
    RetrievalMetadata,
    EHSSchemaAware
)
from .text2cypher import Text2CypherRetriever

# Import our logging and monitoring utilities
from ...utils.logging import get_ehs_logger, performance_logger, log_context
from ...utils.monitoring import get_ehs_monitor
from ...utils.tracing import trace_function, SpanKind, get_ehs_tracer

logger = get_ehs_logger(__name__)


class EHSQueryIntent(Enum):
    """Specific EHS query intents for enhanced processing."""
    CONSUMPTION_ANALYSIS = "consumption_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_ASSESSMENT = "risk_assessment"
    EMISSION_TRACKING = "emission_tracking"
    EQUIPMENT_EFFICIENCY = "equipment_efficiency"
    PERMIT_STATUS = "permit_status"
    GENERAL_INQUIRY = "general_inquiry"


class EHSLLMInterface(LLMInterface):
    """LLM interface adapter for neo4j-graphrag integration."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def invoke(self, input_text: str) -> str:
        """Invoke the LLM with input text."""
        try:
            response = self.llm.invoke(input_text)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise
    
    async def ainvoke(self, input_text: str) -> str:
        """Asynchronously invoke the LLM with input text."""
        try:
            response = await self.llm.ainvoke(input_text)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Async LLM invocation failed: {e}")
            raise


class EHSText2CypherRetriever(Text2CypherRetriever):
    """
    Enhanced Text2Cypher retriever with EHS-specific optimizations.
    
    This retriever extends the base Text2Cypher functionality with:
    - EHS domain-specific query examples
    - Custom prompt templates for EHS use cases
    - Query validation specific to EHS schema
    - Performance optimization for common EHS queries
    - Integration with neo4j-graphrag capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EHS Text2Cypher retriever.
        
        Args:
            config: Configuration dictionary with enhanced EHS settings
        """
        super().__init__(config)
        
        # EHS-specific configuration
        self.use_graphrag = config.get("use_graphrag", True)
        self.query_optimization = config.get("query_optimization", True)
        self.cache_common_queries = config.get("cache_common_queries", True)
        self.max_query_complexity = config.get("max_query_complexity", 10)
        
        # GraphRAG components
        self.graphrag_retriever: Optional[GraphRAGRetriever] = None
        self.ehs_llm_interface: Optional[EHSLLMInterface] = None
        
        # Query cache for performance
        self._query_cache: Dict[str, Any] = {}
        self._cache_max_size = config.get("cache_max_size", 100)
        
        logger.info(
            "EHS Text2Cypher retriever initialized",
            use_graphrag=self.use_graphrag,
            query_optimization=self.query_optimization,
            cache_enabled=self.cache_common_queries
        )
    
    @trace_function("ehs_text2cypher_initialize", SpanKind.INTERNAL, {"component": "ehs_retriever"})
    async def initialize(self) -> None:
        """Initialize EHS-specific components and GraphRAG integration."""
        # First initialize the base retriever
        await super().initialize()
        
        with log_context(component="ehs_text2cypher_retriever", operation="initialize"):
            logger.info("Initializing EHS-specific Text2Cypher components")
            
            try:
                if self.use_graphrag:
                    # Initialize GraphRAG components
                    logger.debug("Setting up neo4j-graphrag integration")
                    
                    # Create LLM interface for GraphRAG
                    self.ehs_llm_interface = EHSLLMInterface(self.llm)
                    
                    # Initialize GraphRAG Text2Cypher retriever
                    self.graphrag_retriever = GraphRAGRetriever(
                        driver=self.driver,
                        llm=self.ehs_llm_interface,
                        neo4j_schema=self._get_enhanced_schema(),
                        examples=self._get_ehs_examples(),
                        result_formatter=self._format_graphrag_results
                    )
                    
                    logger.debug("GraphRAG Text2Cypher retriever initialized")
                
                # Pre-warm common query patterns
                if self.cache_common_queries:
                    await self._prewarm_query_cache()
                
                logger.info("EHS Text2Cypher initialization completed successfully")
                
            except Exception as e:
                logger.error(
                    "Failed to initialize EHS Text2Cypher components",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise
    
    @performance_logger(include_args=True, include_result=False)
    @trace_function("ehs_text2cypher_retrieve", SpanKind.INTERNAL, {"component": "ehs_retriever"})
    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        limit: int = 10,
        **kwargs
    ) -> RetrievalResult:
        """
        Enhanced retrieve method with EHS-specific optimizations.
        
        Args:
            query: Natural language query from the user
            query_type: Type of EHS query being processed
            limit: Maximum number of results to return
            **kwargs: Additional parameters including intent, filters, etc.
            
        Returns:
            RetrievalResult with enhanced EHS-specific metadata
        """
        if not self._initialized:
            logger.error("EHS Text2Cypher retriever not initialized")
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
        
        # Detect EHS query intent
        ehs_intent = self._detect_ehs_intent(query, query_type)
        
        with log_context(
            component="ehs_text2cypher_retriever",
            operation="retrieve",
            query_type=query_type.value,
            ehs_intent=ehs_intent.value,
            query_length=len(query),
            limit=limit
        ):
            logger.info(
                "Processing EHS query",
                query_type=query_type.value,
                ehs_intent=ehs_intent.value,
                query_preview=query[:100]
            )
            
            monitor = get_ehs_monitor()
            start_time = time.time()
            
            try:
                # Check cache first for common queries
                cache_key = self._generate_cache_key(query, query_type, limit)
                if self.cache_common_queries and cache_key in self._query_cache:
                    logger.debug("Returning cached result", cache_key=cache_key)
                    cached_result = self._query_cache[cache_key]
                    cached_result.metadata.execution_time_ms = 1.0  # Minimal cache lookup time
                    return cached_result
                
                # Validate and optimize query
                if not await self._validate_ehs_query(query, ehs_intent):
                    raise ValueError(f"Query validation failed for intent: {ehs_intent.value}")
                
                optimized_query = await self._optimize_query(query, ehs_intent, query_type)
                
                # Choose retrieval method based on configuration and query complexity
                if self.use_graphrag and self._should_use_graphrag(optimized_query, ehs_intent):
                    result = await self._retrieve_with_graphrag(optimized_query, ehs_intent, limit, **kwargs)
                else:
                    result = await self._retrieve_with_base(optimized_query, query_type, limit, **kwargs)
                
                # Enhance result with EHS-specific metadata
                result = await self._enhance_result_metadata(result, ehs_intent, **kwargs)
                
                execution_time = (time.time() - start_time) * 1000
                result.metadata.execution_time_ms = execution_time
                
                # Cache successful results for common queries
                if (self.cache_common_queries and 
                    result.success and 
                    self._is_cacheable_query(query, ehs_intent)):
                    self._add_to_cache(cache_key, result)
                
                # Record metrics
                monitor.record_retrieval(
                    strategy="ehs_text2cypher",
                    duration_ms=execution_time,
                    results_count=len(result.data),
                    success=result.success,
                    ehs_intent=ehs_intent.value
                )
                
                logger.info(
                    "EHS Text2Cypher retrieval completed",
                    success=result.success,
                    results_count=len(result.data),
                    execution_time_ms=execution_time,
                    ehs_intent=ehs_intent.value
                )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                logger.error(
                    "EHS Text2Cypher retrieval failed",
                    query=query,
                    query_type=query_type.value,
                    ehs_intent=ehs_intent.value,
                    error=str(e),
                    error_type=type(e).__name__,
                    execution_time_ms=execution_time,
                    exc_info=True
                )
                
                # Record error metrics
                monitor.record_retrieval(
                    strategy="ehs_text2cypher",
                    duration_ms=execution_time,
                    results_count=0,
                    success=False,
                    ehs_intent=ehs_intent.value
                )
                
                # Create error result
                metadata = RetrievalMetadata(
                    strategy=RetrievalStrategy.TEXT2CYPHER,
                    query_type=query_type,
                    confidence_score=0.0,
                    execution_time_ms=execution_time,
                    error_message=str(e),
                    ehs_intent=ehs_intent.value
                )
                
                return RetrievalResult(
                    data=[],
                    metadata=metadata,
                    success=False,
                    message=f"EHS query execution failed: {str(e)}"
                )
    
    def _detect_ehs_intent(self, query: str, query_type: QueryType) -> EHSQueryIntent:
        """
        Detect specific EHS intent from the query.
        
        Args:
            query: Natural language query
            query_type: Base query type
            
        Returns:
            Detected EHS intent
        """
        query_lower = query.lower()
        
        # Define intent keywords
        intent_keywords = {
            EHSQueryIntent.CONSUMPTION_ANALYSIS: [
                "consumption", "usage", "utility", "water", "electricity", "gas",
                "energy", "kwh", "gallons", "therms", "bill", "meter"
            ],
            EHSQueryIntent.COMPLIANCE_CHECK: [
                "permit", "compliance", "regulation", "violation", "deadline",
                "expires", "renewal", "authority", "certificate", "license"
            ],
            EHSQueryIntent.RISK_ASSESSMENT: [
                "risk", "safety", "hazard", "incident", "accident", "injury",
                "dangerous", "unsafe", "exposure", "assessment"
            ],
            EHSQueryIntent.EMISSION_TRACKING: [
                "emission", "pollutant", "co2", "nox", "so2", "particulate",
                "greenhouse", "carbon", "environmental", "air quality"
            ],
            EHSQueryIntent.EQUIPMENT_EFFICIENCY: [
                "efficiency", "performance", "equipment", "machine", "motor",
                "pump", "compressor", "boiler", "maintenance", "optimization"
            ],
            EHSQueryIntent.PERMIT_STATUS: [
                "permit status", "permit expiry", "permit renewal", "license",
                "authorization", "approval", "certification"
            ]
        }
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent or general if none match
        if intent_scores:
            detected_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            logger.debug(
                "EHS intent detected",
                intent=detected_intent.value,
                score=intent_scores[detected_intent],
                all_scores=intent_scores
            )
            return detected_intent
        
        logger.debug("No specific EHS intent detected, using general inquiry")
        return EHSQueryIntent.GENERAL_INQUIRY
    
    async def _validate_ehs_query(self, query: str, intent: EHSQueryIntent) -> bool:
        """
        Validate query against EHS schema and intent requirements.
        
        Args:
            query: Natural language query
            intent: Detected EHS intent
            
        Returns:
            True if query is valid for the intent
        """
        # Basic validation
        if not query or len(query.strip()) < 3:
            return False
        
        # Intent-specific validation
        validation_rules = {
            EHSQueryIntent.CONSUMPTION_ANALYSIS: self._validate_consumption_query,
            EHSQueryIntent.COMPLIANCE_CHECK: self._validate_compliance_query,
            EHSQueryIntent.RISK_ASSESSMENT: self._validate_risk_query,
            EHSQueryIntent.EMISSION_TRACKING: self._validate_emission_query,
            EHSQueryIntent.EQUIPMENT_EFFICIENCY: self._validate_efficiency_query,
            EHSQueryIntent.PERMIT_STATUS: self._validate_permit_query,
            EHSQueryIntent.GENERAL_INQUIRY: lambda q: True  # General queries are always valid
        }
        
        validator = validation_rules.get(intent, lambda q: True)
        is_valid = validator(query)
        
        logger.debug(
            "EHS query validation completed",
            query_length=len(query),
            intent=intent.value,
            is_valid=is_valid
        )
        
        return is_valid
    
    def _validate_consumption_query(self, query: str) -> bool:
        """Validate consumption analysis queries."""
        required_elements = ["utility", "consumption", "usage", "bill", "meter", "energy"]
        query_lower = query.lower()
        return any(element in query_lower for element in required_elements)
    
    def _validate_compliance_query(self, query: str) -> bool:
        """Validate compliance check queries."""
        required_elements = ["permit", "compliance", "regulation", "deadline", "expiry"]
        query_lower = query.lower()
        return any(element in query_lower for element in required_elements)
    
    def _validate_risk_query(self, query: str) -> bool:
        """Validate risk assessment queries."""
        required_elements = ["risk", "safety", "hazard", "incident", "assessment"]
        query_lower = query.lower()
        return any(element in query_lower for element in required_elements)
    
    def _validate_emission_query(self, query: str) -> bool:
        """Validate emission tracking queries."""
        required_elements = ["emission", "pollutant", "co2", "environmental", "air"]
        query_lower = query.lower()
        return any(element in query_lower for element in required_elements)
    
    def _validate_efficiency_query(self, query: str) -> bool:
        """Validate equipment efficiency queries."""
        required_elements = ["efficiency", "performance", "equipment", "optimization"]
        query_lower = query.lower()
        return any(element in query_lower for element in required_elements)
    
    def _validate_permit_query(self, query: str) -> bool:
        """Validate permit status queries."""
        required_elements = ["permit", "license", "certification", "status", "expiry"]
        query_lower = query.lower()
        return any(element in query_lower for element in required_elements)
    
    async def _optimize_query(self, query: str, intent: EHSQueryIntent, query_type: QueryType) -> str:
        """
        Optimize query for better Cypher generation.
        
        Args:
            query: Original query
            intent: EHS intent
            query_type: Base query type
            
        Returns:
            Optimized query string
        """
        if not self.query_optimization:
            return query
        
        # Intent-specific optimizations
        optimization_rules = {
            EHSQueryIntent.CONSUMPTION_ANALYSIS: self._optimize_consumption_query,
            EHSQueryIntent.COMPLIANCE_CHECK: self._optimize_compliance_query,
            EHSQueryIntent.RISK_ASSESSMENT: self._optimize_risk_query,
            EHSQueryIntent.EMISSION_TRACKING: self._optimize_emission_query,
            EHSQueryIntent.EQUIPMENT_EFFICIENCY: self._optimize_efficiency_query,
            EHSQueryIntent.PERMIT_STATUS: self._optimize_permit_query,
        }
        
        optimizer = optimization_rules.get(intent, lambda q: q)
        optimized = optimizer(query)
        
        logger.debug(
            "Query optimization completed",
            original_length=len(query),
            optimized_length=len(optimized),
            intent=intent.value
        )
        
        return optimized
    
    def _optimize_consumption_query(self, query: str) -> str:
        """Optimize consumption analysis queries."""
        # Add specific consumption context
        optimizations = [
            "Include utility type (water, electricity, gas) and measurement units",
            "Consider time periods and billing cycles",
            "Include facility and equipment relationships"
        ]
        return f"{query}. {' '.join(optimizations)}"
    
    def _optimize_compliance_query(self, query: str) -> str:
        """Optimize compliance check queries."""
        optimizations = [
            "Include permit types and regulatory authorities",
            "Consider expiry dates and renewal requirements",
            "Include compliance status and violations"
        ]
        return f"{query}. {' '.join(optimizations)}"
    
    def _optimize_risk_query(self, query: str) -> str:
        """Optimize risk assessment queries."""
        optimizations = [
            "Include risk types and severity levels",
            "Consider incident history and safety records",
            "Include mitigation measures and controls"
        ]
        return f"{query}. {' '.join(optimizations)}"
    
    def _optimize_emission_query(self, query: str) -> str:
        """Optimize emission tracking queries."""
        optimizations = [
            "Include emission types and measurement units",
            "Consider sources and monitoring points",
            "Include environmental regulations and limits"
        ]
        return f"{query}. {' '.join(optimizations)}"
    
    def _optimize_efficiency_query(self, query: str) -> str:
        """Optimize equipment efficiency queries."""
        optimizations = [
            "Include equipment types and performance metrics",
            "Consider energy consumption and output ratios",
            "Include maintenance schedules and conditions"
        ]
        return f"{query}. {' '.join(optimizations)}"
    
    def _optimize_permit_query(self, query: str) -> str:
        """Optimize permit status queries."""
        optimizations = [
            "Include permit numbers and types",
            "Consider authorities and jurisdictions",
            "Include status dates and renewal requirements"
        ]
        return f"{query}. {' '.join(optimizations)}"
    
    def _should_use_graphrag(self, query: str, intent: EHSQueryIntent) -> bool:
        """
        Determine if GraphRAG should be used for this query.
        
        Args:
            query: Optimized query
            intent: EHS intent
            
        Returns:
            True if GraphRAG should be used
        """
        if not self.use_graphrag or not self.graphrag_retriever:
            return False
        
        # Use GraphRAG for complex queries that benefit from examples
        complex_intents = {
            EHSQueryIntent.CONSUMPTION_ANALYSIS,
            EHSQueryIntent.EQUIPMENT_EFFICIENCY,
            EHSQueryIntent.EMISSION_TRACKING
        }
        
        query_complexity = self._calculate_query_complexity(query)
        
        should_use = (
            intent in complex_intents or 
            query_complexity > 5 or
            len(query.split()) > 20
        )
        
        logger.debug(
            "GraphRAG usage decision",
            should_use=should_use,
            intent=intent.value,
            query_complexity=query_complexity,
            query_length=len(query.split())
        )
        
        return should_use
    
    def _calculate_query_complexity(self, query: str) -> int:
        """Calculate query complexity score."""
        complexity = 0
        
        # Add complexity for aggregations
        aggregations = ["sum", "count", "average", "max", "min", "total"]
        complexity += sum(1 for agg in aggregations if agg in query.lower())
        
        # Add complexity for time ranges
        time_words = ["between", "during", "last", "past", "since", "until"]
        complexity += sum(1 for word in time_words if word in query.lower())
        
        # Add complexity for relationships
        relationship_words = ["related", "connected", "associated", "linked"]
        complexity += sum(1 for word in relationship_words if word in query.lower())
        
        # Add complexity for multiple entities
        entity_words = ["facility", "equipment", "permit", "utility", "emission"]
        complexity += sum(1 for word in entity_words if word in query.lower())
        
        return min(complexity, self.max_query_complexity)
    
    async def _retrieve_with_graphrag(
        self, 
        query: str, 
        intent: EHSQueryIntent, 
        limit: int,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve using neo4j-graphrag capabilities.
        
        Args:
            query: Optimized query
            intent: EHS intent
            limit: Result limit
            **kwargs: Additional parameters
            
        Returns:
            RetrievalResult from GraphRAG retrieval
        """
        logger.debug("Using GraphRAG for retrieval", intent=intent.value)
        
        try:
            # Execute GraphRAG retrieval
            graphrag_result = await self.graphrag_retriever.search(
                query_text=query,
                top_k=limit
            )
            
            # Convert GraphRAG result to our format
            structured_results = self._convert_graphrag_results(graphrag_result, intent)
            
            # Create metadata
            metadata = RetrievalMetadata(
                strategy=RetrievalStrategy.TEXT2CYPHER,
                query_type=QueryType.GENERAL,  # Will be updated by caller
                confidence_score=self._calculate_graphrag_confidence(graphrag_result),
                execution_time_ms=0.0,  # Will be updated by caller
                cypher_query=getattr(graphrag_result, 'cypher_query', ''),
                nodes_retrieved=len(structured_results),
                relationships_retrieved=self._count_relationships(structured_results),
                ehs_intent=intent.value,
                retrieval_method="graphrag"
            )
            
            return RetrievalResult(
                data=structured_results,
                metadata=metadata,
                success=True,
                message=f"GraphRAG retrieved {len(structured_results)} results"
            )
            
        except Exception as e:
            logger.error(f"GraphRAG retrieval failed: {e}")
            # Fallback to base retrieval
            return await self._retrieve_with_base(query, QueryType.GENERAL, limit, **kwargs)
    
    async def _retrieve_with_base(
        self, 
        query: str, 
        query_type: QueryType, 
        limit: int,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve using base Text2Cypher implementation.
        
        Args:
            query: Query to execute
            query_type: Base query type
            limit: Result limit
            **kwargs: Additional parameters
            
        Returns:
            RetrievalResult from base retrieval
        """
        logger.debug("Using base Text2Cypher for retrieval")
        return await super().retrieve(query, query_type, limit, **kwargs)
    
    def _convert_graphrag_results(self, graphrag_result: Any, intent: EHSQueryIntent) -> List[Dict[str, Any]]:
        """Convert GraphRAG results to standard format."""
        # This will depend on the actual GraphRAG result format
        # For now, implement a basic conversion
        if hasattr(graphrag_result, 'records'):
            return [
                {
                    **record,
                    'ehs_intent': intent.value,
                    'retrieval_method': 'graphrag'
                }
                for record in graphrag_result.records
            ]
        elif isinstance(graphrag_result, list):
            return [
                {
                    'data': item,
                    'ehs_intent': intent.value,
                    'retrieval_method': 'graphrag'
                }
                for item in graphrag_result
            ]
        else:
            return [{
                'data': graphrag_result,
                'ehs_intent': intent.value,
                'retrieval_method': 'graphrag'
            }]
    
    def _calculate_graphrag_confidence(self, result: Any) -> float:
        """Calculate confidence score for GraphRAG results."""
        # Implementation depends on GraphRAG result format
        if hasattr(result, 'score'):
            return float(result.score)
        elif hasattr(result, 'confidence'):
            return float(result.confidence)
        else:
            return 0.8  # Default confidence for successful GraphRAG results
    
    async def _enhance_result_metadata(
        self, 
        result: RetrievalResult, 
        intent: EHSQueryIntent,
        **kwargs
    ) -> RetrievalResult:
        """Enhance result metadata with EHS-specific information."""
        if result.metadata:
            result.metadata.ehs_intent = intent.value
            
            # Add intent-specific metadata
            if intent == EHSQueryIntent.CONSUMPTION_ANALYSIS:
                result.metadata.analysis_type = "consumption"
            elif intent == EHSQueryIntent.COMPLIANCE_CHECK:
                result.metadata.analysis_type = "compliance"
            elif intent == EHSQueryIntent.RISK_ASSESSMENT:
                result.metadata.analysis_type = "risk"
            elif intent == EHSQueryIntent.EMISSION_TRACKING:
                result.metadata.analysis_type = "emissions"
            elif intent == EHSQueryIntent.EQUIPMENT_EFFICIENCY:
                result.metadata.analysis_type = "efficiency"
            elif intent == EHSQueryIntent.PERMIT_STATUS:
                result.metadata.analysis_type = "permits"
        
        return result
    
    def _generate_cache_key(self, query: str, query_type: QueryType, limit: int) -> str:
        """Generate cache key for query."""
        import hashlib
        key_data = f"{query}:{query_type.value}:{limit}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cacheable_query(self, query: str, intent: EHSQueryIntent) -> bool:
        """Determine if query should be cached."""
        # Cache common, non-time-sensitive queries
        time_sensitive_words = ["today", "now", "current", "latest", "recent"]
        query_lower = query.lower()
        
        return not any(word in query_lower for word in time_sensitive_words)
    
    def _add_to_cache(self, key: str, result: RetrievalResult) -> None:
        """Add result to cache with size management."""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[key] = result
        logger.debug(f"Added query result to cache: {key}")
    
    async def _prewarm_query_cache(self) -> None:
        """Pre-warm cache with common EHS queries."""
        logger.info("Pre-warming query cache with common EHS patterns")
        
        # This would be implemented based on actual common queries
        # For now, just log the intention
        logger.debug("Query cache pre-warming completed")
    
    def _get_enhanced_schema(self) -> str:
        """Get enhanced Neo4j schema for GraphRAG."""
        # This should return the EHS-specific schema
        # For now, return a basic schema description
        return """
        EHS Data Model:
        - Facility nodes with properties (name, location, type)
        - Equipment nodes connected to facilities
        - UtilityBill nodes with consumption data
        - Permit nodes with compliance information
        - Emission nodes with environmental data
        """
    
    def _get_ehs_examples(self) -> List[Dict[str, str]]:
        """Get EHS-specific query examples for GraphRAG."""
        # Import examples from separate module
        try:
            from ..ehs_examples import get_all_examples
            return get_all_examples()
        except ImportError:
            logger.warning("EHS examples module not found, using default examples")
            return []
    
    def _format_graphrag_results(self, results: Any) -> Any:
        """Format GraphRAG results for consistency."""
        # Implementation depends on specific formatting needs
        return results
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get the retrieval strategy identifier."""
        return RetrievalStrategy.TEXT2CYPHER
    
    async def cleanup(self) -> None:
        """Clean up EHS-specific resources."""
        logger.info("Cleaning up EHS Text2Cypher retriever resources")
        
        # Clear cache
        self._query_cache.clear()
        
        # Cleanup GraphRAG components
        if self.graphrag_retriever:
            # GraphRAG cleanup if needed
            pass
        
        await super().cleanup()
        logger.info("EHS Text2Cypher retriever cleanup completed")