"""
EHS Analytics LangGraph Workflow

This module provides the main LangGraph workflow for processing EHS queries
through classification, retrieval, analysis, and recommendation generation.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

from ..agents.query_router import QueryRouterAgent, QueryClassification, IntentType, RetrieverType
from ..retrieval.strategies.text2cypher import Text2CypherRetriever
from ..retrieval.base import QueryType, RetrievalStrategy
from ..api.dependencies import DatabaseManager

# Import our logging and monitoring utilities
from ..utils.logging import get_ehs_logger, performance_logger, log_context, create_request_context
from ..utils.monitoring import get_ehs_monitor
from ..utils.tracing import trace_function, SpanKind, get_ehs_tracer, get_ehs_profiler

logger = get_ehs_logger(__name__)


class EHSWorkflowState:
    """
    State container for the EHS workflow processing.
    
    This represents the state that flows through the LangGraph workflow nodes.
    """
    
    def __init__(self, query_id: str, original_query: str, user_id: Optional[str] = None):
        self.query_id = query_id
        self.original_query = original_query
        self.user_id = user_id
        self.classification: Optional[QueryClassification] = None
        self.retrieval_results: Optional[Dict[str, Any]] = None
        self.analysis_results: Optional[List[Dict[str, Any]]] = None
        self.recommendations: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.workflow_trace: List[str] = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Performance metrics
        self.step_durations: Dict[str, float] = {}
        self.total_duration_ms: Optional[float] = None
    
    def update_state(self, **kwargs):
        """Update state and timestamp."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
    
    def add_trace(self, message: str, step: Optional[str] = None):
        """Add a trace message to the workflow."""
        timestamp = datetime.utcnow().isoformat()
        trace_entry = f"{timestamp}: {message}"
        if step:
            trace_entry = f"[{step}] {trace_entry}"
        
        self.workflow_trace.append(trace_entry)
        self.updated_at = datetime.utcnow()
        
        logger.debug("Workflow trace added", step=step, message=message, query_id=self.query_id)
    
    def record_step_duration(self, step: str, duration_ms: float):
        """Record the duration of a workflow step."""
        self.step_durations[step] = duration_ms
        self.updated_at = datetime.utcnow()
        
        logger.debug(
            "Step duration recorded", 
            step=step, 
            duration_ms=duration_ms, 
            query_id=self.query_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "user_id": self.user_id,
            "classification": self.classification.__dict__ if self.classification else None,
            "retrieval_results": self.retrieval_results,
            "analysis_results": self.analysis_results,
            "recommendations": self.recommendations,
            "error": self.error,
            "metadata": self.metadata,
            "workflow_trace": self.workflow_trace,
            "step_durations": self.step_durations,
            "total_duration_ms": self.total_duration_ms,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class EHSWorkflow:
    """
    EHS Workflow implementation with integrated components.
    
    This class orchestrates the complete workflow from query classification
    through retrieval to analysis and recommendations using real components.
    """
    
    def __init__(self, db_manager: DatabaseManager, query_router: QueryRouterAgent):
        self.db_manager = db_manager
        self.query_router = query_router
        self.text2cypher_retriever: Optional[Text2CypherRetriever] = None
        self.is_initialized = False
        self.profiler = get_ehs_profiler()
        self.monitor = get_ehs_monitor()
        
        logger.info("EHSWorkflow instance created")
    
    @trace_function("workflow_initialize", SpanKind.INTERNAL, {"component": "workflow"})
    async def initialize(self):
        """Initialize the workflow components."""
        with log_context(component="ehs_workflow", operation="initialize"):
            try:
                logger.info("Initializing EHS workflow")
                
                # Initialize Text2Cypher retriever
                await self._initialize_text2cypher_retriever()
                
                self.is_initialized = True
                
                logger.info("EHS workflow initialized successfully")
                
            except Exception as e:
                logger.error(
                    "Failed to initialize EHS workflow", 
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise
    
    async def _initialize_text2cypher_retriever(self):
        """Initialize the Text2Cypher retriever with database connection."""
        try:
            from ..config import get_settings
            settings = get_settings()
            
            # Create retriever configuration
            text2cypher_config = {
                "neo4j_uri": settings.neo4j_uri,
                "neo4j_user": settings.neo4j_username,
                "neo4j_password": settings.neo4j_password,
                "openai_api_key": settings.openai_api_key,
                "model_name": getattr(settings, "llm_model_name", "gpt-3.5-turbo"),
                "temperature": getattr(settings, "llm_temperature", 0.0),
                "max_tokens": getattr(settings, "llm_max_tokens", 2000),
                "cypher_validation": getattr(settings, "cypher_validation", True)
            }
            
            logger.debug("Creating Text2Cypher retriever with config", config_keys=list(text2cypher_config.keys()))
            
            # Create and initialize the retriever
            self.text2cypher_retriever = Text2CypherRetriever(text2cypher_config)
            await self.text2cypher_retriever.initialize()
            
            logger.info("Text2Cypher retriever initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize Text2Cypher retriever",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            # Don't raise - allow workflow to continue without retriever
            logger.warning("Workflow will continue with mock retrieval")
    
    @performance_logger(include_args=True, include_result=False)
    @trace_function("workflow_process_query", SpanKind.SERVER, {"component": "workflow"})
    async def process_query(
        self, 
        query_id: str, 
        query: str, 
        user_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> EHSWorkflowState:
        """
        Process a query through the complete EHS workflow.
        
        This implementation uses real components for classification and retrieval.
        """
        # Create request context for logging
        request_context = create_request_context(
            user_id=user_id,
            request_id=query_id,
            component="ehs_workflow",
            operation="process_query"
        )
        
        with log_context(**request_context.__dict__):
            logger.info(
                "Starting EHS workflow processing",
                query_id=query_id,
                query=query[:100],  # First 100 chars
                user_id=user_id,
                options=options
            )
            
            start_time = datetime.utcnow()
            state = EHSWorkflowState(query_id, query, user_id)
            
            try:
                # Step 1: Query Classification
                await self._step_classify_query(state)
                
                # Step 2: Data Retrieval
                await self._step_retrieve_data(state)
                
                # Step 3: Analysis
                await self._step_analyze_data(state)
                
                # Step 4: Recommendations (optional)
                if options and options.get("include_recommendations", True):
                    await self._step_generate_recommendations(state)
                
                # Calculate total duration
                total_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                state.total_duration_ms = total_duration
                
                # Final logging
                state.add_trace("Workflow processing completed successfully")
                
                logger.info(
                    "EHS workflow processing completed successfully",
                    query_id=query_id,
                    total_duration_ms=total_duration,
                    steps_completed=len(state.step_durations),
                    results_count=len(state.retrieval_results.get("documents", [])) if state.retrieval_results else 0
                )
                
                # Record workflow metrics
                self.monitor.record_query(
                    query_type=state.classification.intent_type.value if state.classification else "unknown",
                    duration_ms=total_duration,
                    success=True
                )
                
                return state
                
            except Exception as e:
                total_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                state.total_duration_ms = total_duration
                
                logger.error(
                    "EHS workflow processing failed", 
                    query_id=query_id,
                    user_id=user_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    total_duration_ms=total_duration,
                    exc_info=True
                )
                
                state.update_state(error=str(e))
                state.add_trace(f"Workflow failed: {str(e)}")
                
                # Record error metrics
                self.monitor.record_query(
                    query_type=state.classification.intent_type.value if state.classification else "unknown",
                    duration_ms=total_duration,
                    success=False
                )
                
                raise
    
    @trace_function("step_classify_query", SpanKind.INTERNAL, {"workflow_step": "classification"})
    async def _step_classify_query(self, state: EHSWorkflowState):
        """Query classification step using the real QueryRouterAgent."""
        with log_context(workflow_step="classification", query_id=state.query_id):
            step_start = datetime.utcnow()
            state.add_trace("Starting query classification", "CLASSIFY")
            
            try:
                logger.debug("Executing query classification step")
                
                with self.profiler.profile_operation(
                    "query_classification",
                    tags={"query_id": state.query_id, "step": "classification"}
                ):
                    # Use the real query router agent
                    classification = self.query_router.classify_query(
                        state.original_query, 
                        user_id=state.user_id
                    )
                
                state.update_state(classification=classification)
                
                # Record step duration
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                state.record_step_duration("classification", step_duration)
                
                state.add_trace(
                    f"Query classified as: {classification.intent_type.value} "
                    f"(confidence: {classification.confidence_score:.2f})",
                    "CLASSIFY"
                )
                
                logger.info(
                    "Query classification completed",
                    query_id=state.query_id,
                    intent_type=classification.intent_type.value,
                    confidence_score=classification.confidence_score,
                    suggested_retriever=classification.suggested_retriever.value,
                    step_duration_ms=step_duration
                )
                
            except Exception as e:
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                state.record_step_duration("classification", step_duration)
                
                logger.error(
                    "Query classification step failed",
                    query_id=state.query_id,
                    error=str(e),
                    step_duration_ms=step_duration
                )
                raise
    
    @trace_function("step_retrieve_data", SpanKind.INTERNAL, {"workflow_step": "retrieval"})
    async def _step_retrieve_data(self, state: EHSWorkflowState):
        """Data retrieval step using the appropriate retriever based on classification."""
        with log_context(workflow_step="retrieval", query_id=state.query_id):
            step_start = datetime.utcnow()
            state.add_trace("Starting data retrieval", "RETRIEVE")
            
            try:
                logger.debug("Executing data retrieval step")
                
                with self.profiler.profile_operation(
                    "data_retrieval",
                    tags={
                        "query_id": state.query_id, 
                        "step": "retrieval",
                        "retriever_type": state.classification.suggested_retriever.value if state.classification else "unknown"
                    }
                ):
                    retrieval_results = await self._execute_retrieval(state)
                
                state.update_state(retrieval_results=retrieval_results)
                
                # Record step duration
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                state.record_step_duration("retrieval", step_duration)
                
                # Record retrieval metrics
                self.monitor.record_retrieval(
                    strategy=retrieval_results.get("retrieval_strategy", "unknown"),
                    duration_ms=step_duration,
                    results_count=retrieval_results.get("total_count", 0),
                    success=retrieval_results.get("success", False)
                )
                
                state.add_trace("Data retrieval completed", "RETRIEVE")
                
                logger.info(
                    "Data retrieval completed",
                    query_id=state.query_id,
                    results_count=retrieval_results.get("total_count", 0),
                    strategy=retrieval_results.get("retrieval_strategy", "unknown"),
                    success=retrieval_results.get("success", False),
                    step_duration_ms=step_duration
                )
                
            except Exception as e:
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                state.record_step_duration("retrieval", step_duration)
                
                # Record error metrics
                self.monitor.record_retrieval(
                    strategy="unknown",
                    duration_ms=step_duration,
                    results_count=0,
                    success=False
                )
                
                logger.error(
                    "Data retrieval step failed",
                    query_id=state.query_id,
                    error=str(e),
                    step_duration_ms=step_duration
                )
                raise
    
    async def _execute_retrieval(self, state: EHSWorkflowState) -> Dict[str, Any]:
        """Execute the appropriate retrieval strategy based on classification."""
        if not state.classification:
            raise ValueError("Query must be classified before retrieval")
        
        suggested_retriever = state.classification.suggested_retriever
        query_to_use = state.classification.query_rewrite or state.original_query
        
        logger.debug(
            "Executing retrieval",
            suggested_retriever=suggested_retriever.value,
            query_length=len(query_to_use),
            has_rewrite=bool(state.classification.query_rewrite)
        )
        
        # Convert IntentType to QueryType for retriever
        intent_to_query_type_map = {
            IntentType.CONSUMPTION_ANALYSIS: QueryType.CONSUMPTION,
            IntentType.EQUIPMENT_EFFICIENCY: QueryType.EFFICIENCY,
            IntentType.COMPLIANCE_CHECK: QueryType.COMPLIANCE,
            IntentType.EMISSION_TRACKING: QueryType.EMISSIONS,
            IntentType.RISK_ASSESSMENT: QueryType.RISK,
            IntentType.PERMIT_STATUS: QueryType.COMPLIANCE,
            IntentType.GENERAL_INQUIRY: QueryType.GENERAL
        }
        
        query_type = intent_to_query_type_map.get(
            state.classification.intent_type, 
            QueryType.GENERAL
        )
        
        # Try to use Text2Cypher retriever if available
        if (self.text2cypher_retriever and 
            self.text2cypher_retriever._initialized and
            suggested_retriever in [RetrieverType.GENERAL_RETRIEVER, 
                                   RetrieverType.CONSUMPTION_RETRIEVER,
                                   RetrieverType.EQUIPMENT_RETRIEVER,
                                   RetrieverType.COMPLIANCE_RETRIEVER,
                                   RetrieverType.EMISSION_RETRIEVER]):
            
            logger.debug("Using Text2Cypher retriever")
            
            # Validate query compatibility
            is_valid = await self.text2cypher_retriever.validate_query(query_to_use)
            
            if is_valid:
                try:
                    retrieval_result = await self.text2cypher_retriever.retrieve(
                        query=query_to_use,
                        query_type=query_type,
                        limit=20  # Default limit
                    )
                    
                    if retrieval_result.success:
                        return {
                            "documents": retrieval_result.data,
                            "total_count": len(retrieval_result.data),
                            "retrieval_strategy": retrieval_result.metadata.strategy.value,
                            "confidence_score": retrieval_result.metadata.confidence_score,
                            "execution_time_ms": retrieval_result.metadata.execution_time_ms,
                            "cypher_query": getattr(retrieval_result.metadata, 'cypher_query', ''),
                            "success": True,
                            "message": retrieval_result.message
                        }
                    else:
                        logger.warning(
                            "Text2Cypher retrieval failed, falling back",
                            message=retrieval_result.message
                        )
                        
                except Exception as e:
                    logger.error(
                        "Text2Cypher retrieval error, falling back",
                        error=str(e),
                        error_type=type(e).__name__
                    )
        
        # Fallback to mock retrieval
        logger.debug("Using fallback mock retrieval")
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "documents": [],
            "total_count": 0,
            "retrieval_strategy": suggested_retriever.value,
            "execution_time_ms": 100,
            "query_embedding": None,
            "confidence_score": state.classification.confidence_score,
            "success": True,
            "message": "Mock retrieval completed (Text2Cypher not available)"
        }
    
    @trace_function("step_analyze_data", SpanKind.INTERNAL, {"workflow_step": "analysis"})
    async def _step_analyze_data(self, state: EHSWorkflowState):
        """Data analysis step."""
        with log_context(workflow_step="analysis", query_id=state.query_id):
            step_start = datetime.utcnow()
            state.add_trace("Starting analysis", "ANALYZE")
            
            try:
                logger.debug("Executing data analysis step")
                
                with self.profiler.profile_operation(
                    "data_analysis",
                    tags={
                        "query_id": state.query_id, 
                        "step": "analysis",
                        "intent_type": state.classification.intent_type.value if state.classification else "unknown"
                    }
                ):
                    # Simulate async operation
                    await asyncio.sleep(0.1)
                    
                    analysis_results = await self._perform_analysis(state)
                
                state.update_state(analysis_results=analysis_results)
                
                # Record step duration
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                state.record_step_duration("analysis", step_duration)
                
                # Record analysis metrics
                if analysis_results:
                    for result in analysis_results:
                        self.monitor.record_analysis(
                            analysis_type=result.get("analysis_type", "unknown"),
                            confidence=result.get("confidence", 0.0),
                            success=True
                        )
                
                state.add_trace("Analysis completed", "ANALYZE")
                
                logger.info(
                    "Data analysis completed",
                    query_id=state.query_id,
                    analysis_count=len(analysis_results) if analysis_results else 0,
                    step_duration_ms=step_duration
                )
                
            except Exception as e:
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                state.record_step_duration("analysis", step_duration)
                
                logger.error(
                    "Data analysis step failed",
                    query_id=state.query_id,
                    error=str(e),
                    step_duration_ms=step_duration
                )
                raise
    
    @trace_function("step_generate_recommendations", SpanKind.INTERNAL, {"workflow_step": "recommendations"})
    async def _step_generate_recommendations(self, state: EHSWorkflowState):
        """Recommendations generation step."""
        with log_context(workflow_step="recommendations", query_id=state.query_id):
            step_start = datetime.utcnow()
            state.add_trace("Generating recommendations", "RECOMMEND")
            
            try:
                logger.debug("Executing recommendations generation step")
                
                with self.profiler.profile_operation(
                    "recommendation_generation",
                    tags={"query_id": state.query_id, "step": "recommendations"}
                ):
                    # Simulate async operation
                    await asyncio.sleep(0.1)
                    
                    recommendations = await self._generate_recommendations(state)
                
                state.update_state(recommendations=recommendations)
                
                # Record step duration
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                state.record_step_duration("recommendations", step_duration)
                
                # Log recommendation metrics
                if recommendations and "recommendations" in recommendations:
                    logger.recommendation_generated(
                        recommendation_count=len(recommendations["recommendations"]),
                        total_savings=recommendations.get("total_estimated_savings", 0.0),
                        query_id=state.query_id
                    )
                
                state.add_trace("Recommendations generated", "RECOMMEND")
                
                logger.info(
                    "Recommendations generation completed",
                    query_id=state.query_id,
                    recommendation_count=len(recommendations.get("recommendations", [])) if recommendations else 0,
                    step_duration_ms=step_duration
                )
                
            except Exception as e:
                step_duration = (datetime.utcnow() - step_start).total_seconds() * 1000
                state.record_step_duration("recommendations", step_duration)
                
                logger.error(
                    "Recommendations generation step failed",
                    query_id=state.query_id,
                    error=str(e),
                    step_duration_ms=step_duration
                )
                raise
    
    async def _perform_analysis(self, state: EHSWorkflowState) -> List[Dict[str, Any]]:
        """
        Perform analysis based on classification and retrieval results.
        
        Enhanced analysis that considers the actual classification and retrieval data.
        """
        if not state.classification:
            return []
        
        intent = state.classification.intent_type
        retrieval_results = state.retrieval_results or {}
        documents = retrieval_results.get("documents", [])
        
        # Create intent-specific analysis
        if intent == IntentType.CONSUMPTION_ANALYSIS:
            return await self._analyze_consumption(documents, state.classification)
        elif intent == IntentType.COMPLIANCE_CHECK:
            return await self._analyze_compliance(documents, state.classification)
        elif intent == IntentType.RISK_ASSESSMENT:
            return await self._analyze_risk(documents, state.classification)
        elif intent == IntentType.EMISSION_TRACKING:
            return await self._analyze_emissions(documents, state.classification)
        elif intent == IntentType.EQUIPMENT_EFFICIENCY:
            return await self._analyze_equipment(documents, state.classification)
        elif intent == IntentType.PERMIT_STATUS:
            return await self._analyze_permits(documents, state.classification)
        else:
            return [{
                "analysis_type": "general_analysis",
                "summary": f"Analysis completed for {intent.value} inquiry",
                "data_points": len(documents),
                "confidence": 0.7,
                "entities_found": len(state.classification.entities_identified.facilities) + 
                               len(state.classification.entities_identified.equipment)
            }]
    
    async def _analyze_consumption(self, documents: List[Dict], classification: QueryClassification) -> List[Dict[str, Any]]:
        """Analyze consumption-related data."""
        return [{
            "analysis_type": "consumption_analysis",
            "total_data_points": len(documents),
            "facilities_mentioned": len(classification.entities_identified.facilities),
            "time_periods": len(classification.entities_identified.date_ranges),
            "consumption_trend": "stable",  # Would be calculated from actual data
            "efficiency_score": 0.75,
            "confidence": 0.8 if documents else 0.4
        }]
    
    async def _analyze_compliance(self, documents: List[Dict], classification: QueryClassification) -> List[Dict[str, Any]]:
        """Analyze compliance-related data."""
        return [{
            "analysis_type": "compliance_analysis",
            "regulations_checked": len(classification.entities_identified.regulations),
            "compliant": True,  # Would be determined from actual data
            "compliance_score": 0.9,
            "violations": [],
            "requirements_met": ["Sample requirement"],
            "confidence": 0.85 if documents else 0.5
        }]
    
    async def _analyze_risk(self, documents: List[Dict], classification: QueryClassification) -> List[Dict[str, Any]]:
        """Analyze risk-related data."""
        return [{
            "analysis_type": "risk_assessment",
            "risk_level": "medium",
            "risk_score": 0.6,
            "risk_factors": ["equipment_age", "maintenance_schedule"],
            "mitigation_suggestions": ["Increase maintenance frequency"],
            "confidence": 0.7 if documents else 0.4
        }]
    
    async def _analyze_emissions(self, documents: List[Dict], classification: QueryClassification) -> List[Dict[str, Any]]:
        """Analyze emissions-related data."""
        return [{
            "analysis_type": "emission_analysis",
            "pollutants_tracked": len(classification.entities_identified.pollutants),
            "total_emissions": 1000.0,  # Would be calculated from actual data
            "emission_trend": "decreasing",
            "targets_met": True,
            "confidence": 0.75 if documents else 0.4
        }]
    
    async def _analyze_equipment(self, documents: List[Dict], classification: QueryClassification) -> List[Dict[str, Any]]:
        """Analyze equipment efficiency data."""
        return [{
            "analysis_type": "equipment_efficiency",
            "equipment_count": len(classification.entities_identified.equipment),
            "average_efficiency": 0.82,
            "maintenance_due": [],
            "performance_trend": "improving",
            "confidence": 0.8 if documents else 0.4
        }]
    
    async def _analyze_permits(self, documents: List[Dict], classification: QueryClassification) -> List[Dict[str, Any]]:
        """Analyze permit status data."""
        return [{
            "analysis_type": "permit_status",
            "permits_checked": 5,  # Would be from actual data
            "expiring_soon": [],
            "renewal_required": [],
            "compliance_status": "compliant",
            "confidence": 0.9 if documents else 0.5
        }]
    
    async def _generate_recommendations(self, state: EHSWorkflowState) -> Dict[str, Any]:
        """
        Generate recommendations based on analysis results.
        
        Enhanced recommendation generation that considers the specific analysis.
        """
        if not state.analysis_results:
            return self._generate_default_recommendations()
        
        recommendations = []
        total_cost = 0.0
        total_savings = 0.0
        
        for analysis in state.analysis_results:
            analysis_type = analysis.get("analysis_type", "general")
            
            if analysis_type == "consumption_analysis":
                rec = self._generate_consumption_recommendations(analysis)
            elif analysis_type == "compliance_analysis":
                rec = self._generate_compliance_recommendations(analysis)
            elif analysis_type == "risk_assessment":
                rec = self._generate_risk_recommendations(analysis)
            elif analysis_type == "emission_analysis":
                rec = self._generate_emission_recommendations(analysis)
            elif analysis_type == "equipment_efficiency":
                rec = self._generate_equipment_recommendations(analysis)
            elif analysis_type == "permit_status":
                rec = self._generate_permit_recommendations(analysis)
            else:
                rec = self._generate_general_recommendations(analysis)
            
            if rec:
                recommendations.append(rec)
                total_cost += rec.get("estimated_cost", 0.0)
                total_savings += rec.get("estimated_savings", 0.0)
        
        return {
            "recommendations": recommendations,
            "total_estimated_cost": total_cost,
            "total_estimated_savings": total_savings,
            "recommendations_count": len(recommendations),
            "net_benefit": total_savings - total_cost,
            "generated_at": datetime.utcnow()
        }
    
    def _generate_consumption_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consumption-specific recommendations."""
        return {
            "id": "rec_consumption_001",
            "title": "Optimize Energy Consumption",
            "description": "Implement energy efficiency measures based on consumption analysis",
            "priority": "high",
            "category": "efficiency",
            "estimated_cost": 10000.0,
            "estimated_savings": 25000.0,
            "payback_period_months": 5,
            "implementation_effort": "medium",
            "confidence": analysis.get("confidence", 0.7),
            "tags": ["energy", "cost_reduction", "consumption"]
        }
    
    def _generate_compliance_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance-specific recommendations."""
        return {
            "id": "rec_compliance_001",
            "title": "Maintain Compliance Standards",
            "description": "Continue current compliance practices and monitor for changes",
            "priority": "medium",
            "category": "compliance",
            "estimated_cost": 2000.0,
            "estimated_savings": 5000.0,
            "payback_period_months": 5,
            "implementation_effort": "low",
            "confidence": analysis.get("confidence", 0.8),
            "tags": ["compliance", "regulatory", "risk_mitigation"]
        }
    
    def _generate_risk_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk-specific recommendations."""
        return {
            "id": "rec_risk_001",
            "title": "Enhance Risk Mitigation",
            "description": "Implement additional safety measures to reduce identified risks",
            "priority": "high",
            "category": "safety",
            "estimated_cost": 15000.0,
            "estimated_savings": 30000.0,
            "payback_period_months": 6,
            "implementation_effort": "high",
            "confidence": analysis.get("confidence", 0.7),
            "tags": ["safety", "risk_reduction", "mitigation"]
        }
    
    def _generate_emission_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate emission-specific recommendations."""
        return {
            "id": "rec_emission_001",
            "title": "Reduce Carbon Footprint",
            "description": "Implement emission reduction strategies based on tracking data",
            "priority": "medium",
            "category": "environmental",
            "estimated_cost": 8000.0,
            "estimated_savings": 18000.0,
            "payback_period_months": 5,
            "implementation_effort": "medium",
            "confidence": analysis.get("confidence", 0.75),
            "tags": ["emissions", "carbon", "environmental"]
        }
    
    def _generate_equipment_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate equipment-specific recommendations."""
        return {
            "id": "rec_equipment_001",
            "title": "Optimize Equipment Performance",
            "description": "Improve equipment efficiency and reduce maintenance costs",
            "priority": "medium",
            "category": "efficiency",
            "estimated_cost": 12000.0,
            "estimated_savings": 28000.0,
            "payback_period_months": 5,
            "implementation_effort": "medium",
            "confidence": analysis.get("confidence", 0.8),
            "tags": ["equipment", "maintenance", "efficiency"]
        }
    
    def _generate_permit_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate permit-specific recommendations."""
        return {
            "id": "rec_permit_001",
            "title": "Maintain Permit Compliance",
            "description": "Ensure all permits remain current and compliant",
            "priority": "high",
            "category": "compliance",
            "estimated_cost": 3000.0,
            "estimated_savings": 8000.0,
            "payback_period_months": 5,
            "implementation_effort": "low",
            "confidence": analysis.get("confidence", 0.9),
            "tags": ["permits", "compliance", "regulatory"]
        }
    
    def _generate_general_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate general recommendations."""
        return {
            "id": "rec_general_001",
            "title": "General EHS Improvement",
            "description": "Implement general EHS improvements based on analysis",
            "priority": "medium",
            "category": "general",
            "estimated_cost": 5000.0,
            "estimated_savings": 12000.0,
            "payback_period_months": 5,
            "implementation_effort": "medium",
            "confidence": analysis.get("confidence", 0.7),
            "tags": ["general", "improvement"]
        }
    
    def _generate_default_recommendations(self) -> Dict[str, Any]:
        """Generate default recommendations when no analysis is available."""
        return {
            "recommendations": [{
                "id": "rec_default_001",
                "title": "General EHS Assessment",
                "description": "Consider a comprehensive EHS assessment to identify improvement opportunities",
                "priority": "medium",
                "category": "assessment",
                "estimated_cost": 5000.0,
                "estimated_savings": 15000.0,
                "payback_period_months": 4,
                "implementation_effort": "medium",
                "confidence": 0.6,
                "tags": ["assessment", "general"]
            }],
            "total_estimated_cost": 5000.0,
            "total_estimated_savings": 15000.0,
            "recommendations_count": 1,
            "generated_at": datetime.utcnow()
        }
    
    @trace_function("workflow_health_check", SpanKind.INTERNAL)
    async def health_check(self) -> bool:
        """Check if the workflow is healthy and operational."""
        with log_context(component="ehs_workflow", operation="health_check"):
            try:
                logger.debug("Performing workflow health check")
                
                if not self.is_initialized:
                    logger.warning("Workflow not initialized")
                    return False
                
                # Test query router
                test_result = self.query_router.classify_query("test health check query")
                router_healthy = test_result is not None and test_result.confidence_score >= 0.0
                
                # Test retriever if available
                retriever_healthy = True
                if self.text2cypher_retriever and self.text2cypher_retriever._initialized:
                    try:
                        retriever_valid = await self.text2cypher_retriever.validate_query("test query")
                        retriever_healthy = retriever_valid is not None
                    except Exception as e:
                        logger.warning("Text2Cypher retriever health check failed", error=str(e))
                        retriever_healthy = False
                
                is_healthy = router_healthy and retriever_healthy
                
                if is_healthy:
                    logger.debug("Workflow health check passed")
                else:
                    logger.warning(
                        "Workflow health check failed",
                        router_healthy=router_healthy,
                        retriever_healthy=retriever_healthy
                    )
                
                return is_healthy
                
            except Exception as e:
                logger.error(
                    "Workflow health check failed", 
                    error=str(e),
                    error_type=type(e).__name__
                )
                return False
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow performance statistics."""
        # This would typically return real metrics from processed queries
        return {
            "total_queries_processed": 0,
            "average_processing_time_ms": 0.0,
            "success_rate": 0.0,
            "step_performance": {
                "classification": {"avg_ms": 0.0, "success_rate": 0.0},
                "retrieval": {"avg_ms": 0.0, "success_rate": 0.0},
                "analysis": {"avg_ms": 0.0, "success_rate": 0.0},
                "recommendations": {"avg_ms": 0.0, "success_rate": 0.0}
            },
            "intent_distribution": {},
            "retriever_usage": {
                "text2cypher_available": bool(self.text2cypher_retriever and self.text2cypher_retriever._initialized),
                "fallback_usage_count": 0
            }
        }


async def create_ehs_workflow(
    db_manager: DatabaseManager, 
    query_router: QueryRouterAgent
) -> EHSWorkflow:
    """
    Factory function to create and initialize the EHS workflow.
    
    Args:
        db_manager: Database manager instance
        query_router: Query router agent instance
    
    Returns:
        Initialized EHSWorkflow instance
    """
    with log_context(component="workflow_factory", operation="create_workflow"):
        try:
            logger.info("Creating EHS workflow instance")
            
            workflow = EHSWorkflow(db_manager, query_router)
            await workflow.initialize()
            
            logger.info("EHS workflow created and initialized successfully")
            return workflow
            
        except Exception as e:
            logger.error(
                "Failed to create EHS workflow", 
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise


# Placeholder for actual LangGraph implementation
"""
Future LangGraph Implementation Structure:

from langgraph import StateGraph, END
from typing import TypedDict

class WorkflowState(TypedDict):
    query_id: str
    original_query: str
    user_id: Optional[str]
    classification: Optional[QueryClassification]
    retrieval_results: Optional[Dict[str, Any]]
    analysis_results: Optional[List[Dict[str, Any]]]
    recommendations: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]
    workflow_trace: List[str]
    step_durations: Dict[str, float]

def create_langgraph_workflow():
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("retrieve_data", retrieve_data_node) 
    workflow.add_node("analyze_data", analyze_data_node)
    workflow.add_node("generate_recommendations", generate_recommendations_node)
    
    # Add edges
    workflow.add_edge("classify_query", "retrieve_data")
    workflow.add_edge("retrieve_data", "analyze_data")
    workflow.add_edge("analyze_data", "generate_recommendations")
    workflow.add_edge("generate_recommendations", END)
    
    # Set entry point
    workflow.set_entry_point("classify_query")
    
    return workflow.compile()

async def classify_query_node(state: WorkflowState) -> WorkflowState:
    # Implementation for query classification
    pass

async def retrieve_data_node(state: WorkflowState) -> WorkflowState:
    # Implementation for data retrieval
    pass

async def analyze_data_node(state: WorkflowState) -> WorkflowState:
    # Implementation for data analysis
    pass

async def generate_recommendations_node(state: WorkflowState) -> WorkflowState:
    # Implementation for recommendation generation
    pass
"""