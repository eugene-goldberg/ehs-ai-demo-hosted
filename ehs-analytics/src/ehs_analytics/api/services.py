"""
EHS Analytics API Services

This module provides high-level services for query processing, result formatting,
and error handling that coordinate between the API layer and the workflow layer.
"""

import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from fastapi import HTTPException, status
import structlog

from ..workflows.ehs_workflow import EHSWorkflow, EHSWorkflowState
from ..agents.query_router import QueryRouterAgent, QueryClassification
from .models import (
    QueryRequest, QueryResponse, QueryResult, ClassificationResponse,
    ErrorResponse, ErrorDetail, ErrorType, HealthResponse
)
from .dependencies import WorkflowManager, DatabaseManager, QuerySessionManager

# Configure structured logging
logger = structlog.get_logger(__name__)


class QueryProcessingService:
    """
    High-level service for processing EHS queries through the complete workflow.
    
    This service orchestrates the query processing pipeline from initial request
    validation through classification, retrieval, analysis, and recommendations.
    """
    
    def __init__(self, workflow: EHSWorkflow, session_manager: QuerySessionManager):
        self.workflow = workflow
        self.session_manager = session_manager
        
        logger.info("QueryProcessingService initialized")
    
    async def process_query(
        self,
        request: QueryRequest,
        user_id: Optional[str] = None
    ) -> QueryResponse:
        """
        Process a complete EHS query through the workflow.
        
        Args:
            request: The query request containing query text and options
            user_id: Optional user identifier for logging and session management
            
        Returns:
            QueryResponse with classification, results, analysis, and recommendations
            
        Raises:
            HTTPException: For various error conditions
        """
        query_id = str(uuid.uuid4())
        
        logger.info(
            "Processing EHS query",
            query_id=query_id,
            user_id=user_id,
            query_length=len(request.query),
            include_recommendations=request.include_recommendations
        )
        
        try:
            # Create session for tracking
            session_id = await self.session_manager.create_session(query_id, user_id or "anonymous")
            
            # Execute workflow
            workflow_state = await self.workflow.process_query(
                query_id=query_id,
                query=request.query,
                user_id=user_id,
                options={
                    "include_recommendations": request.include_recommendations,
                    "limit": getattr(request, 'limit', 20),
                    "timeout": getattr(request, 'timeout', 30)
                }
            )
            
            # Format response
            response = await self._format_query_response(workflow_state, session_id)
            
            logger.info(
                "Query processing completed successfully",
                query_id=query_id,
                session_id=session_id,
                processing_time_ms=workflow_state.total_duration_ms,
                results_count=len(workflow_state.retrieval_results.get("documents", [])) if workflow_state.retrieval_results else 0,
                recommendations_count=len(workflow_state.recommendations.get("recommendations", [])) if workflow_state.recommendations else 0
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Query processing failed",
                query_id=query_id,
                user_id=user_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            
            # Convert to appropriate HTTP exception
            if isinstance(e, HTTPException):
                raise e
            elif "timeout" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail={
                        "error_type": ErrorType.TIMEOUT_ERROR,
                        "message": "Query processing timed out",
                        "details": {"query_id": query_id, "error": str(e)}
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error_type": ErrorType.PROCESSING_ERROR,
                        "message": "Query processing failed",
                        "details": {"query_id": query_id, "error": str(e)}
                    }
                )
    
    async def _format_query_response(
        self,
        state: EHSWorkflowState,
        session_id: str
    ) -> QueryResponse:
        """Format the workflow state into a standardized API response."""
        
        # Format classification
        classification = None
        if state.classification:
            classification = ClassificationResponse(
                intent_type=state.classification.intent_type.value,
                confidence_score=state.classification.confidence_score,
                suggested_retriever=state.classification.suggested_retriever.value,
                reasoning=state.classification.reasoning,
                entities_identified={
                    "facilities": state.classification.entities_identified.facilities,
                    "date_ranges": state.classification.entities_identified.date_ranges,
                    "equipment": state.classification.entities_identified.equipment,
                    "pollutants": state.classification.entities_identified.pollutants,
                    "regulations": state.classification.entities_identified.regulations,
                    "departments": state.classification.entities_identified.departments,
                    "metrics": state.classification.entities_identified.metrics
                },
                query_rewrite=state.classification.query_rewrite
            )
        
        # Format retrieval results
        results = []
        total_count = 0
        if state.retrieval_results:
            documents = state.retrieval_results.get("documents", [])
            total_count = state.retrieval_results.get("total_count", len(documents))
            
            for doc in documents:
                results.append(QueryResult(
                    id=doc.get("id", str(uuid.uuid4())),
                    content=doc.get("content", doc),
                    source=doc.get("source", "database"),
                    confidence_score=doc.get("confidence_score", 0.5),
                    metadata=doc.get("metadata", {})
                ))
        
        # Create response
        return QueryResponse(
            query_id=state.query_id,
            session_id=session_id,
            query=state.original_query,
            classification=classification,
            results=results,
            total_results=total_count,
            analysis=state.analysis_results or [],
            recommendations=state.recommendations.get("recommendations", []) if state.recommendations else [],
            processing_time_ms=state.total_duration_ms or 0,
            success=not bool(state.error),
            message="Query processed successfully" if not state.error else f"Error: {state.error}",
            metadata={
                "step_durations": state.step_durations,
                "workflow_trace": state.workflow_trace[-5:],  # Last 5 trace entries
                "retrieval_strategy": state.retrieval_results.get("retrieval_strategy", "unknown") if state.retrieval_results else "none",
                "created_at": state.created_at.isoformat(),
                "updated_at": state.updated_at.isoformat()
            }
        )


class QueryClassificationService:
    """
    Service for standalone query classification without full workflow processing.
    
    This service provides quick classification of queries for preview purposes
    or to help users understand how their queries will be interpreted.
    """
    
    def __init__(self, query_router: QueryRouterAgent):
        self.query_router = query_router
        
        logger.info("QueryClassificationService initialized")
    
    async def classify_query(
        self,
        query: str,
        user_id: Optional[str] = None
    ) -> ClassificationResponse:
        """
        Classify a query and return the classification details.
        
        Args:
            query: Natural language query to classify
            user_id: Optional user identifier
            
        Returns:
            ClassificationResponse with intent type, confidence, and entities
            
        Raises:
            HTTPException: For invalid queries or processing errors
        """
        logger.info(
            "Classifying query",
            user_id=user_id,
            query_length=len(query)
        )
        
        try:
            # Validate query
            if not query or not query.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_type": ErrorType.VALIDATION_ERROR,
                        "message": "Query cannot be empty",
                        "details": {}
                    }
                )
            
            if len(query) > 10000:  # Reasonable query length limit
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_type": ErrorType.VALIDATION_ERROR,
                        "message": "Query too long (max 10,000 characters)",
                        "details": {"query_length": len(query)}
                    }
                )
            
            # Perform classification
            classification = self.query_router.classify_query(query, user_id)
            
            # Format response
            response = ClassificationResponse(
                intent_type=classification.intent_type.value,
                confidence_score=classification.confidence_score,
                suggested_retriever=classification.suggested_retriever.value,
                reasoning=classification.reasoning,
                entities_identified={
                    "facilities": classification.entities_identified.facilities,
                    "date_ranges": classification.entities_identified.date_ranges,
                    "equipment": classification.entities_identified.equipment,
                    "pollutants": classification.entities_identified.pollutants,
                    "regulations": classification.entities_identified.regulations,
                    "departments": classification.entities_identified.departments,
                    "metrics": classification.entities_identified.metrics
                },
                query_rewrite=classification.query_rewrite
            )
            
            logger.info(
                "Query classification completed",
                user_id=user_id,
                intent_type=classification.intent_type.value,
                confidence_score=classification.confidence_score
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "Query classification failed",
                user_id=user_id,
                query=query[:100],
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_type": ErrorType.PROCESSING_ERROR,
                    "message": "Query classification failed",
                    "details": {"error": str(e)}
                }
            )


class HealthService:
    """
    Service for monitoring system health and providing status information.
    
    This service checks the health of various system components and provides
    detailed status information for monitoring and debugging.
    """
    
    def __init__(
        self,
        workflow_manager: WorkflowManager,
        db_manager: DatabaseManager
    ):
        self.workflow_manager = workflow_manager
        self.db_manager = db_manager
        
        logger.info("HealthService initialized")
    
    async def get_health_status(self) -> HealthResponse:
        """
        Get comprehensive health status of all system components.
        
        Returns:
            HealthResponse with detailed component health information
        """
        logger.debug("Performing comprehensive health check")
        
        start_time = datetime.utcnow()
        
        # Check database health
        db_healthy = False
        db_details = {}
        try:
            db_healthy = await self.db_manager.health_check()
            db_details = {
                "connected": self.db_manager.is_connected,
                "driver_available": bool(self.db_manager._neo4j_driver)
            }
        except Exception as e:
            logger.warning("Database health check failed", error=str(e))
            db_details = {"error": str(e)}
        
        # Check workflow health
        workflow_healthy = False
        workflow_details = {}
        try:
            workflow_healthy = await self.workflow_manager.health_check()
            workflow_details = {
                "initialized": self.workflow_manager.is_initialized,
                "query_router_available": bool(self.workflow_manager._query_router)
            }
            
            # Get workflow stats if available
            if self.workflow_manager._workflow_graph:
                workflow_stats = self.workflow_manager._workflow_graph.get_workflow_stats()
                workflow_details.update(workflow_stats)
                
        except Exception as e:
            logger.warning("Workflow health check failed", error=str(e))
            workflow_details = {"error": str(e)}
        
        # Overall health status
        overall_healthy = db_healthy and workflow_healthy
        
        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        health_response = HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            timestamp=datetime.utcnow(),
            response_time_ms=response_time,
            components={
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "details": db_details
                },
                "workflow": {
                    "status": "healthy" if workflow_healthy else "unhealthy", 
                    "details": workflow_details
                }
            },
            version="1.0.0",  # Would typically come from config
            environment="development"  # Would typically come from config
        )
        
        logger.info(
            "Health check completed",
            overall_status=health_response.status,
            database_healthy=db_healthy,
            workflow_healthy=workflow_healthy,
            response_time_ms=response_time
        )
        
        return health_response
    
    async def get_simple_health_check(self) -> Dict[str, Any]:
        """
        Get a simple health check response for load balancer health probes.
        
        Returns:
            Simple dictionary with status and timestamp
        """
        try:
            # Quick health checks
            db_ok = self.db_manager.is_connected
            workflow_ok = self.workflow_manager.is_initialized
            
            overall_ok = db_ok and workflow_ok
            
            return {
                "status": "ok" if overall_ok else "error",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "database": "ok" if db_ok else "error",
                    "workflow": "ok" if workflow_ok else "error"
                }
            }
        except Exception as e:
            logger.error("Simple health check failed", error=str(e))
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }


class ResultFormattingService:
    """
    Service for formatting and transforming query results into different formats.
    
    This service provides utilities for formatting results for different clients
    and use cases, including CSV export, chart data, and summary statistics.
    """
    
    def __init__(self):
        logger.info("ResultFormattingService initialized")
    
    def format_results_for_dashboard(
        self,
        results: List[QueryResult]
    ) -> Dict[str, Any]:
        """
        Format results for dashboard display with charts and summary data.
        
        Args:
            results: List of query results to format
            
        Returns:
            Dictionary with formatted data for dashboard components
        """
        logger.debug("Formatting results for dashboard", result_count=len(results))
        
        if not results:
            return {
                "summary": {"total_results": 0, "average_confidence": 0.0},
                "chart_data": [],
                "table_data": [],
                "metadata": {"formatted_at": datetime.utcnow().isoformat()}
            }
        
        # Calculate summary statistics
        total_results = len(results)
        average_confidence = sum(r.confidence_score for r in results) / total_results
        confidence_distribution = self._calculate_confidence_distribution(results)
        
        # Prepare chart data (confidence distribution)
        chart_data = [
            {"range": f"{i*0.2:.1f}-{(i+1)*0.2:.1f}", "count": count}
            for i, count in enumerate(confidence_distribution)
        ]
        
        # Prepare table data (top results)
        table_data = [
            {
                "id": result.id,
                "content_preview": str(result.content)[:100] + "..." if len(str(result.content)) > 100 else str(result.content),
                "source": result.source,
                "confidence": f"{result.confidence_score:.2f}",
                "metadata_keys": list(result.metadata.keys()) if result.metadata else []
            }
            for result in results[:10]  # Top 10 results
        ]
        
        formatted = {
            "summary": {
                "total_results": total_results,
                "average_confidence": average_confidence,
                "confidence_distribution": confidence_distribution
            },
            "chart_data": chart_data,
            "table_data": table_data,
            "metadata": {
                "formatted_at": datetime.utcnow().isoformat(),
                "formatter_version": "1.0.0"
            }
        }
        
        logger.debug("Dashboard formatting completed", summary=formatted["summary"])
        return formatted
    
    def format_results_for_export(
        self,
        results: List[QueryResult],
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Format results for export in various formats.
        
        Args:
            results: List of query results to format
            format_type: Export format (json, csv, xlsx)
            
        Returns:
            Dictionary with formatted export data
        """
        logger.debug(
            "Formatting results for export",
            result_count=len(results),
            format_type=format_type
        )
        
        if format_type.lower() == "csv":
            return self._format_as_csv_data(results)
        elif format_type.lower() == "xlsx":
            return self._format_as_excel_data(results)
        else:  # Default to JSON
            return self._format_as_json_data(results)
    
    def _calculate_confidence_distribution(
        self,
        results: List[QueryResult]
    ) -> List[int]:
        """Calculate confidence score distribution in 5 buckets (0-0.2, 0.2-0.4, etc.)."""
        distribution = [0] * 5
        
        for result in results:
            bucket = min(int(result.confidence_score * 5), 4)
            distribution[bucket] += 1
            
        return distribution
    
    def _format_as_csv_data(self, results: List[QueryResult]) -> Dict[str, Any]:
        """Format results as CSV-compatible data."""
        headers = ["id", "content", "source", "confidence_score", "metadata"]
        rows = []
        
        for result in results:
            rows.append([
                result.id,
                str(result.content),
                result.source,
                result.confidence_score,
                str(result.metadata) if result.metadata else ""
            ])
        
        return {
            "format": "csv",
            "headers": headers,
            "rows": rows,
            "total_rows": len(rows)
        }
    
    def _format_as_excel_data(self, results: List[QueryResult]) -> Dict[str, Any]:
        """Format results as Excel-compatible data."""
        # Similar to CSV but with additional formatting info
        csv_data = self._format_as_csv_data(results)
        csv_data["format"] = "xlsx"
        csv_data["sheets"] = [
            {
                "name": "Query Results",
                "headers": csv_data["headers"],
                "rows": csv_data["rows"]
            }
        ]
        return csv_data
    
    def _format_as_json_data(self, results: List[QueryResult]) -> Dict[str, Any]:
        """Format results as structured JSON data."""
        return {
            "format": "json",
            "data": [
                {
                    "id": result.id,
                    "content": result.content,
                    "source": result.source,
                    "confidence_score": result.confidence_score,
                    "metadata": result.metadata
                }
                for result in results
            ],
            "total_results": len(results)
        }


class ErrorHandlingService:
    """
    Service for consistent error handling and logging across the API.
    
    This service provides utilities for converting exceptions to standardized
    HTTP responses and logging errors with proper context.
    """
    
    def __init__(self):
        logger.info("ErrorHandlingService initialized")
    
    def handle_workflow_error(self, error: Exception, query_id: str) -> HTTPException:
        """
        Convert workflow errors to appropriate HTTP exceptions.
        
        Args:
            error: The original exception
            query_id: Query ID for tracking
            
        Returns:
            HTTPException with appropriate status code and details
        """
        logger.error(
            "Handling workflow error",
            query_id=query_id,
            error=str(error),
            error_type=type(error).__name__,
            exc_info=True
        )
        
        if "timeout" in str(error).lower():
            return HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail={
                    "error_type": ErrorType.TIMEOUT_ERROR,
                    "message": "Workflow processing timed out",
                    "details": {"query_id": query_id, "error": str(error)}
                }
            )
        elif "database" in str(error).lower():
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_type": ErrorType.DATABASE_ERROR,
                    "message": "Database service unavailable",
                    "details": {"query_id": query_id, "error": str(error)}
                }
            )
        elif "unauthorized" in str(error).lower() or "authentication" in str(error).lower():
            return HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_type": ErrorType.AUTHORIZATION_ERROR,
                    "message": "Authentication required",
                    "details": {"query_id": query_id}
                }
            )
        else:
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_type": ErrorType.PROCESSING_ERROR,
                    "message": "Workflow processing failed",
                    "details": {"query_id": query_id, "error": str(error)}
                }
            )
    
    def log_api_error(
        self,
        error: Exception,
        endpoint: str,
        user_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None
    ):
        """
        Log API errors with structured context.
        
        Args:
            error: The exception that occurred
            endpoint: API endpoint where the error occurred
            user_id: Optional user identifier
            request_data: Optional request data for debugging
        """
        logger.error(
            "API error occurred",
            endpoint=endpoint,
            user_id=user_id,
            error=str(error),
            error_type=type(error).__name__,
            request_preview=str(request_data)[:200] if request_data else None,
            exc_info=True
        )