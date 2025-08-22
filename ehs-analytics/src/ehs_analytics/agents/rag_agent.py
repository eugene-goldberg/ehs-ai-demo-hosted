"""
RAG Agent for EHS Analytics

This module provides the RAG (Retrieval-Augmented Generation) agent that dynamically
selects retrievers, builds context, generates responses, and validates answers for EHS queries.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import base retriever and types
from ..retrieval.base import BaseRetriever, RetrievalStrategy, QueryType, RetrievalResult
from ..retrieval.strategies.text2cypher import Text2CypherRetriever
from ..retrieval.strategies.vector_retriever import EHSVectorRetriever
from ..retrieval.strategies.hybrid_cypher_retriever import EHSHybridCypherRetriever
from ..retrieval.strategies.vector_cypher_retriever import EHSVectorCypherRetriever

# Import supporting components
from .context_builder import ContextBuilder, ContextWindow
from .response_generator import ResponseGenerator, GeneratedResponse

# Import router for query classification
from .query_router import QueryClassification, IntentType, RetrieverType

# Import logging and monitoring
from ..utils.logging import get_ehs_logger, performance_logger, log_context
from ..utils.monitoring import get_ehs_monitor
from ..utils.tracing import trace_function, SpanKind

logger = get_ehs_logger(__name__)


class RetrievalMode(str, Enum):
    """Different modes for retrieval selection."""
    
    SINGLE_BEST = "single_best"  # Use the single best retriever
    PARALLEL = "parallel"        # Use multiple retrievers in parallel
    FALLBACK = "fallback"        # Try retrievers in sequence with fallback
    ENSEMBLE = "ensemble"        # Combine results from multiple retrievers


@dataclass
class RAGConfiguration:
    """Configuration for the RAG agent."""
    
    # Retrieval settings
    retrieval_mode: RetrievalMode = RetrievalMode.SINGLE_BEST
    max_retrievers: int = 3
    confidence_threshold: float = 0.7
    
    # Context settings
    max_context_length: int = 8000
    context_compression_ratio: float = 0.8
    include_metadata: bool = True
    
    # Response generation settings
    max_response_length: int = 1000
    include_sources: bool = True
    validate_responses: bool = True
    
    # Performance settings
    retrieval_timeout_seconds: float = 30.0
    response_timeout_seconds: float = 45.0
    
    # Quality settings
    min_source_relevance: float = 0.6
    max_sources_per_response: int = 10


@dataclass
class RAGResult:
    """Result from RAG processing."""
    
    query_id: str
    original_query: str
    classification: QueryClassification
    context_window: ContextWindow
    response: GeneratedResponse
    retrieval_results: List[RetrievalResult]
    retrievers_used: List[str]
    confidence_score: float
    source_count: int
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None


class RAGAgent:
    """
    RAG Agent for EHS Analytics with dynamic retriever selection.
    
    This agent orchestrates the complete RAG pipeline:
    1. Dynamic retriever selection based on query classification
    2. Context building from retrieved results
    3. Response generation with source attribution
    4. Answer validation and fact-checking
    5. Confidence scoring for responses
    """
    
    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        config: RAGConfiguration = None
    ):
        """
        Initialize the RAG agent.
        
        Args:
            retrievers: Dictionary of available retrievers by strategy name
            config: RAG configuration settings
        """
        self.retrievers = retrievers
        self.config = config or RAGConfiguration()
        
        # Initialize supporting components
        self.context_builder = ContextBuilder(
            max_length=self.config.max_context_length,
            compression_ratio=self.config.context_compression_ratio,
            include_metadata=self.config.include_metadata
        )
        
        self.response_generator = ResponseGenerator(
            max_length=self.config.max_response_length,
            include_sources=self.config.include_sources,
            validate_responses=self.config.validate_responses
        )
        
        # Performance monitoring
        self.monitor = get_ehs_monitor()
        
        logger.info(
            "RAG Agent initialized",
            available_retrievers=list(self.retrievers.keys()),
            config=self.config.__dict__
        )
    
    @performance_logger(include_args=True, include_result=False)
    @trace_function("rag_process_query", SpanKind.SERVER, {"component": "rag_agent"})
    async def process_query(
        self,
        query_id: str,
        query: str,
        classification: QueryClassification,
        user_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query_id: Unique identifier for the query
            query: Natural language query
            classification: Query classification from router
            user_id: Optional user identifier
            options: Additional processing options
            
        Returns:
            RAGResult containing the complete processing result
        """
        with log_context(
            component="rag_agent",
            operation="process_query",
            query_id=query_id,
            user_id=user_id,
            intent_type=classification.intent_type.value
        ):
            start_time = datetime.utcnow()
            
            logger.info(
                "Starting RAG query processing",
                query_id=query_id,
                query_preview=query[:100],
                intent_type=classification.intent_type.value,
                confidence=classification.confidence_score
            )
            
            try:
                # Step 1: Select retrievers dynamically
                selected_retrievers = await self._select_retrievers(classification, options)
                
                # Step 2: Execute retrieval with selected retrievers
                retrieval_results = await self._execute_retrieval(
                    query, classification, selected_retrievers
                )
                
                # Step 3: Build context from retrieval results
                context_window = await self._build_context(
                    query, classification, retrieval_results
                )
                
                # Step 4: Generate response
                response = await self._generate_response(
                    query, classification, context_window
                )
                
                # Step 5: Calculate confidence and validate
                confidence_score = await self._calculate_confidence(
                    classification, retrieval_results, response
                )
                
                # Create result
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                result = RAGResult(
                    query_id=query_id,
                    original_query=query,
                    classification=classification,
                    context_window=context_window,
                    response=response,
                    retrieval_results=retrieval_results,
                    retrievers_used=[r.strategy.value for r in selected_retrievers],
                    confidence_score=confidence_score,
                    source_count=len(context_window.sources),
                    processing_time_ms=processing_time,
                    success=True
                )
                
                # Log successful completion
                logger.info(
                    "RAG query processing completed successfully",
                    query_id=query_id,
                    retrievers_used=result.retrievers_used,
                    source_count=result.source_count,
                    confidence_score=confidence_score,
                    processing_time_ms=processing_time
                )
                
                # Record metrics
                self.monitor.record_query(
                    query_type="rag_processing",
                    duration_ms=processing_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.error(
                    "RAG query processing failed",
                    query_id=query_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    processing_time_ms=processing_time,
                    exc_info=True
                )
                
                # Create error result
                result = RAGResult(
                    query_id=query_id,
                    original_query=query,
                    classification=classification,
                    context_window=ContextWindow(content="", sources=[], metadata={}),
                    response=GeneratedResponse(
                        content="I apologize, but I encountered an error processing your query.",
                        sources=[],
                        confidence_score=0.0,
                        metadata={"error": str(e)}
                    ),
                    retrieval_results=[],
                    retrievers_used=[],
                    confidence_score=0.0,
                    source_count=0,
                    processing_time_ms=processing_time,
                    success=False,
                    error_message=str(e)
                )
                
                # Record error metrics
                self.monitor.record_query(
                    query_type="rag_processing",
                    duration_ms=processing_time,
                    success=False
                )
                
                return result
    
    @trace_function("select_retrievers", SpanKind.INTERNAL, {"workflow_step": "retriever_selection"})
    async def _select_retrievers(
        self,
        classification: QueryClassification,
        options: Optional[Dict[str, Any]] = None
    ) -> List[BaseRetriever]:
        """
        Dynamically select retrievers based on query classification.
        
        Args:
            classification: Query classification result
            options: Additional selection options
            
        Returns:
            List of selected retrievers
        """
        with log_context(workflow_step="retriever_selection"):
            logger.debug("Selecting retrievers for query classification")
            
            retrieval_mode = options.get("retrieval_mode", self.config.retrieval_mode) if options else self.config.retrieval_mode
            
            # Map intent types to preferred retriever strategies
            intent_strategy_map = {
                IntentType.CONSUMPTION_ANALYSIS: [RetrievalStrategy.VECTOR_CYPHER, RetrievalStrategy.TEXT2CYPHER],
                IntentType.COMPLIANCE_CHECK: [RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR],
                IntentType.RISK_ASSESSMENT: [RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.VECTOR_CYPHER],
                IntentType.EMISSION_TRACKING: [RetrievalStrategy.VECTOR_CYPHER, RetrievalStrategy.TEXT2CYPHER],
                IntentType.EQUIPMENT_EFFICIENCY: [RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR],
                IntentType.PERMIT_STATUS: [RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR],
                IntentType.GENERAL_INQUIRY: [RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.VECTOR]
            }
            
            preferred_strategies = intent_strategy_map.get(
                classification.intent_type,
                [RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR]
            )
            
            selected_retrievers = []
            
            if retrieval_mode == RetrievalMode.SINGLE_BEST:
                # Select the best available retriever
                for strategy in preferred_strategies:
                    retriever = self._get_retriever_by_strategy(strategy)
                    if retriever and retriever._initialized:
                        selected_retrievers.append(retriever)
                        break
                        
            elif retrieval_mode == RetrievalMode.PARALLEL:
                # Select multiple retrievers for parallel execution
                for strategy in preferred_strategies[:self.config.max_retrievers]:
                    retriever = self._get_retriever_by_strategy(strategy)
                    if retriever and retriever._initialized:
                        selected_retrievers.append(retriever)
                        
            elif retrieval_mode == RetrievalMode.FALLBACK:
                # All preferred retrievers for fallback execution
                for strategy in preferred_strategies:
                    retriever = self._get_retriever_by_strategy(strategy)
                    if retriever and retriever._initialized:
                        selected_retrievers.append(retriever)
                        
            elif retrieval_mode == RetrievalMode.ENSEMBLE:
                # Select all available retrievers for ensemble
                for retriever in self.retrievers.values():
                    if retriever._initialized:
                        selected_retrievers.append(retriever)
            
            # Fallback to any available retriever if none selected
            if not selected_retrievers:
                for retriever in self.retrievers.values():
                    if retriever._initialized:
                        selected_retrievers.append(retriever)
                        break
            
            logger.info(
                "Retrievers selected",
                retrieval_mode=retrieval_mode.value,
                selected_strategies=[r.get_strategy().value for r in selected_retrievers],
                total_selected=len(selected_retrievers)
            )
            
            return selected_retrievers
    
    def _get_retriever_by_strategy(self, strategy: RetrievalStrategy) -> Optional[BaseRetriever]:
        """Get retriever instance by strategy."""
        strategy_key_map = {
            RetrievalStrategy.TEXT2CYPHER: "text2cypher",
            RetrievalStrategy.VECTOR: "vector",
            RetrievalStrategy.HYBRID: "hybrid",
            RetrievalStrategy.VECTOR_CYPHER: "vector_cypher",
            RetrievalStrategy.HYBRID_CYPHER: "hybrid_cypher"
        }
        
        key = strategy_key_map.get(strategy)
        return self.retrievers.get(key) if key else None
    
    @trace_function("execute_retrieval", SpanKind.INTERNAL, {"workflow_step": "retrieval"})
    async def _execute_retrieval(
        self,
        query: str,
        classification: QueryClassification,
        retrievers: List[BaseRetriever]
    ) -> List[RetrievalResult]:
        """
        Execute retrieval with selected retrievers.
        
        Args:
            query: Natural language query
            classification: Query classification
            retrievers: Selected retrievers
            
        Returns:
            List of retrieval results
        """
        with log_context(workflow_step="retrieval"):
            logger.debug("Executing retrieval with selected retrievers")
            
            # Convert intent to query type
            intent_to_query_type = {
                IntentType.CONSUMPTION_ANALYSIS: QueryType.CONSUMPTION,
                IntentType.EQUIPMENT_EFFICIENCY: QueryType.EFFICIENCY,
                IntentType.COMPLIANCE_CHECK: QueryType.COMPLIANCE,
                IntentType.EMISSION_TRACKING: QueryType.EMISSIONS,
                IntentType.RISK_ASSESSMENT: QueryType.RISK,
                IntentType.PERMIT_STATUS: QueryType.COMPLIANCE,
                IntentType.GENERAL_INQUIRY: QueryType.GENERAL
            }
            
            query_type = intent_to_query_type.get(classification.intent_type, QueryType.GENERAL)
            query_to_use = classification.query_rewrite or query
            
            retrieval_results = []
            
            if self.config.retrieval_mode == RetrievalMode.PARALLEL:
                # Execute retrievers in parallel
                tasks = []
                for retriever in retrievers:
                    task = asyncio.create_task(
                        self._safe_retrieve(retriever, query_to_use, query_type)
                    )
                    tasks.append(task)
                
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=self.config.retrieval_timeout_seconds
                    )
                    
                    for result in results:
                        if isinstance(result, RetrievalResult):
                            retrieval_results.append(result)
                            
                except asyncio.TimeoutError:
                    logger.warning("Retrieval timeout, using partial results")
                    
            else:
                # Execute retrievers sequentially (fallback mode)
                for retriever in retrievers:
                    try:
                        result = await asyncio.wait_for(
                            self._safe_retrieve(retriever, query_to_use, query_type),
                            timeout=self.config.retrieval_timeout_seconds
                        )
                        
                        if result.success:
                            retrieval_results.append(result)
                            
                            # For fallback mode, stop on first successful retrieval
                            if self.config.retrieval_mode == RetrievalMode.FALLBACK:
                                break
                                
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Retriever timeout",
                            retriever_strategy=retriever.get_strategy().value
                        )
                        continue
            
            logger.info(
                "Retrieval execution completed",
                successful_retrievals=len(retrieval_results),
                total_retrievers=len(retrievers)
            )
            
            return retrieval_results
    
    async def _safe_retrieve(
        self,
        retriever: BaseRetriever,
        query: str,
        query_type: QueryType
    ) -> RetrievalResult:
        """Safely execute retrieval with error handling."""
        try:
            return await retriever.retrieve(query=query, query_type=query_type, limit=20)
        except Exception as e:
            logger.error(
                "Retriever execution failed",
                retriever_strategy=retriever.get_strategy().value,
                error=str(e)
            )
            
            # Return empty result on error
            from ..retrieval.base import RetrievalMetadata
            return RetrievalResult(
                data=[],
                metadata=RetrievalMetadata(
                    strategy=retriever.get_strategy(),
                    query_type=query_type,
                    confidence_score=0.0,
                    execution_time_ms=0.0,
                    error_message=str(e)
                ),
                success=False,
                message=f"Retrieval failed: {str(e)}"
            )
    
    @trace_function("build_context", SpanKind.INTERNAL, {"workflow_step": "context_building"})
    async def _build_context(
        self,
        query: str,
        classification: QueryClassification,
        retrieval_results: List[RetrievalResult]
    ) -> ContextWindow:
        """
        Build context window from retrieval results.
        
        Args:
            query: Original query
            classification: Query classification
            retrieval_results: Results from retrievers
            
        Returns:
            ContextWindow with processed content
        """
        return await self.context_builder.build_context(
            query=query,
            classification=classification,
            retrieval_results=retrieval_results
        )
    
    @trace_function("generate_response", SpanKind.INTERNAL, {"workflow_step": "response_generation"})
    async def _generate_response(
        self,
        query: str,
        classification: QueryClassification,
        context_window: ContextWindow
    ) -> GeneratedResponse:
        """
        Generate response using LLM with context.
        
        Args:
            query: Original query
            classification: Query classification
            context_window: Built context window
            
        Returns:
            Generated response with sources
        """
        return await self.response_generator.generate_response(
            query=query,
            classification=classification,
            context_window=context_window
        )
    
    @trace_function("calculate_confidence", SpanKind.INTERNAL, {"workflow_step": "confidence_calculation"})
    async def _calculate_confidence(
        self,
        classification: QueryClassification,
        retrieval_results: List[RetrievalResult],
        response: GeneratedResponse
    ) -> float:
        """
        Calculate overall confidence score for the RAG result.
        
        Args:
            classification: Query classification
            retrieval_results: Retrieval results
            response: Generated response
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        # Base confidence from classification
        classification_confidence = classification.confidence_score
        
        # Retrieval confidence (average of successful retrievals)
        successful_retrievals = [r for r in retrieval_results if r.success]
        if successful_retrievals:
            retrieval_confidence = sum(
                r.metadata.confidence_score for r in successful_retrievals
            ) / len(successful_retrievals)
        else:
            retrieval_confidence = 0.0
        
        # Response confidence
        response_confidence = response.confidence_score
        
        # Source quality (based on number and relevance of sources)
        source_quality = min(len(response.sources) / self.config.max_sources_per_response, 1.0)
        
        # Weighted combination
        overall_confidence = (
            0.3 * classification_confidence +
            0.3 * retrieval_confidence +
            0.3 * response_confidence +
            0.1 * source_quality
        )
        
        logger.debug(
            "Confidence calculation completed",
            classification_confidence=classification_confidence,
            retrieval_confidence=retrieval_confidence,
            response_confidence=response_confidence,
            source_quality=source_quality,
            overall_confidence=overall_confidence
        )
        
        return min(max(overall_confidence, 0.0), 1.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the RAG agent.
        
        Returns:
            Health status information
        """
        with log_context(component="rag_agent", operation="health_check"):
            retriever_health = {}
            
            for name, retriever in self.retrievers.items():
                try:
                    health = await retriever.health_check()
                    retriever_health[name] = health
                except Exception as e:
                    retriever_health[name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            
            healthy_retrievers = sum(
                1 for health in retriever_health.values()
                if health.get("status") == "healthy"
            )
            
            overall_healthy = healthy_retrievers > 0
            
            return {
                "status": "healthy" if overall_healthy else "unhealthy",
                "retrievers": retriever_health,
                "healthy_retrievers": healthy_retrievers,
                "total_retrievers": len(self.retrievers),
                "config": self.config.__dict__
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG agent statistics."""
        return {
            "available_retrievers": list(self.retrievers.keys()),
            "initialized_retrievers": [
                name for name, retriever in self.retrievers.items()
                if retriever._initialized
            ],
            "config": self.config.__dict__,
            "total_queries_processed": 0,  # Would be tracked in production
            "average_confidence": 0.0,     # Would be calculated from history
            "average_response_time_ms": 0.0  # Would be calculated from history
        }


async def create_rag_agent(
    retrievers: Dict[str, BaseRetriever],
    config: RAGConfiguration = None
) -> RAGAgent:
    """
    Factory function to create a RAG agent.
    
    Args:
        retrievers: Dictionary of available retrievers
        config: RAG configuration
        
    Returns:
        Initialized RAG agent
    """
    with log_context(component="rag_factory", operation="create_agent"):
        logger.info("Creating RAG agent")
        
        agent = RAGAgent(retrievers, config)
        
        logger.info(
            "RAG agent created successfully",
            available_retrievers=list(retrievers.keys())
        )
        
        return agent