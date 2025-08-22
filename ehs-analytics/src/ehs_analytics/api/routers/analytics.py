"""
Analytics Router for EHS Analytics API

This module provides the main analytics endpoints for processing natural language
EHS queries, managing query lifecycle, and providing health checks using the
integrated workflow and services.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import structlog

from ..models import (
    QueryRequest, QueryResponse, QueryResult, ClassificationResponse,
    HealthResponse, ErrorResponse, ErrorDetail, ErrorType,
    BaseResponse
)
from ..dependencies import (
    get_db_manager, get_workflow_manager, get_query_router, get_current_user_id,
    get_session_manager, validate_request_rate_limit, validate_query_request,
    get_workflow, DatabaseManager, WorkflowManager, QuerySessionManager
)
from ..services import (
    QueryProcessingService, QueryClassificationService, HealthService,
    ResultFormattingService, ErrorHandlingService
)
from ...agents.query_router import QueryRouterAgent
from ...workflows.ehs_workflow import EHSWorkflow

# Configure structured logging
logger = structlog.get_logger(__name__)

# Create router instance
router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Analytics"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_ehs_query(
    request: Request,
    query_request: QueryRequest,
    user_id: str = Depends(get_current_user_id),
    workflow: EHSWorkflow = Depends(get_workflow),
    session_manager: QuerySessionManager = Depends(get_session_manager)
) -> QueryResponse:
    """
    Process a natural language EHS query through the complete workflow.
    
    This endpoint accepts natural language queries about EHS data and processes them
    through the complete workflow including classification, data retrieval, analysis,
    and recommendation generation using the integrated Text2Cypher retriever.
    
    - **query**: Natural language query about EHS data
    - **include_recommendations**: Whether to generate recommendations (default: True)
    - **context**: Additional context for the query processing
    - **preferences**: User preferences for processing options
    
    Returns the complete query response with classification, results, analysis, and recommendations.
    """
    error_service = ErrorHandlingService()
    
    try:
        # Validate request
        await validate_query_request(request)
        validate_request_rate_limit(request)
        
        logger.info(
            "Processing EHS query via integrated workflow",
            user_id=user_id,
            query_length=len(query_request.query),
            include_recommendations=query_request.include_recommendations
        )
        
        # Create query processing service
        query_service = QueryProcessingService(workflow, session_manager)
        
        # Process query through the complete workflow
        response = await query_service.process_query(query_request, user_id)
        
        logger.info(
            "EHS query processed successfully",
            query_id=response.query_id,
            user_id=user_id,
            processing_time_ms=response.processing_time_ms,
            intent_type=response.classification.intent_type if response.classification else None,
            results_count=response.total_results
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "EHS query processing failed",
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        
        # Log the error with structured context
        error_service.log_api_error(
            error=e,
            endpoint="/api/v1/analytics/query",
            user_id=user_id,
            request_data={"query_length": len(query_request.query)}
        )
        
        # Convert to appropriate HTTP exception
        query_id = str(uuid.uuid4())
        raise error_service.handle_workflow_error(e, query_id)


@router.post("/classify", response_model=ClassificationResponse, status_code=status.HTTP_200_OK)
async def classify_query(
    query: str,
    user_id: str = Depends(get_current_user_id),
    query_router: QueryRouterAgent = Depends(get_query_router)
) -> ClassificationResponse:
    """
    Classify a natural language query without full processing.
    
    This endpoint provides quick classification of queries to understand how
    they will be interpreted and routed through the system.
    
    - **query**: Natural language query to classify
    
    Returns classification details including intent type, confidence, and entities.
    """
    error_service = ErrorHandlingService()
    
    try:
        logger.info(
            "Classifying query",
            user_id=user_id,
            query_length=len(query)
        )
        
        # Create classification service
        classification_service = QueryClassificationService(query_router)
        
        # Classify the query
        response = await classification_service.classify_query(query, user_id)
        
        logger.info(
            "Query classified successfully",
            user_id=user_id,
            intent_type=response.intent_type,
            confidence_score=response.confidence_score
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
        
        # Log the error
        error_service.log_api_error(
            error=e,
            endpoint="/api/v1/analytics/classify",
            user_id=user_id,
            request_data={"query_length": len(query)}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": ErrorType.PROCESSING_ERROR,
                "message": "Query classification failed",
                "details": {"error": str(e)}
            }
        )


@router.get("/health", response_model=HealthResponse)
async def analytics_health_check(
    db_manager: DatabaseManager = Depends(get_db_manager),
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
) -> HealthResponse:
    """
    Comprehensive health check endpoint for the analytics service.
    
    Checks the health of all service components including:
    - Database connectivity (Neo4j)
    - Workflow engine (LangGraph with Text2Cypher)
    - Query router agent
    - External dependencies (OpenAI API)
    
    Returns detailed health status for monitoring and debugging.
    """
    try:
        logger.debug("Performing comprehensive analytics health check")
        
        # Create health service
        health_service = HealthService(workflow_manager, db_manager)
        
        # Get comprehensive health status
        response = await health_service.get_health_status()
        
        logger.info(
            "Analytics health check completed",
            overall_status=response.status,
            response_time_ms=response.response_time_ms
        )
        
        return response
        
    except Exception as e:
        logger.error("Analytics health check failed", error=str(e), exc_info=True)
        
        # Return minimal error response
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            response_time_ms=0,
            components={
                "error": {
                    "status": "unhealthy",
                    "details": {"error": str(e)}
                }
            },
            version="1.0.0",
            environment="development"
        )


@router.get("/health/simple")
async def simple_health_check(
    db_manager: DatabaseManager = Depends(get_db_manager),
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
) -> Dict[str, Any]:
    """
    Simple health check endpoint for load balancer health probes.
    
    Returns a lightweight health check response suitable for automated monitoring.
    """
    try:
        # Create health service
        health_service = HealthService(workflow_manager, db_manager)
        
        # Get simple health status
        response = await health_service.get_simple_health_check()
        
        logger.debug("Simple health check completed", status=response["status"])
        
        return response
        
    except Exception as e:
        logger.error("Simple health check failed", error=str(e))
        
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/dashboard/format")
async def format_for_dashboard(
    query_id: Optional[str] = None,
    sample_count: int = 10
) -> Dict[str, Any]:
    """
    Format query results for dashboard display.
    
    This endpoint demonstrates result formatting capabilities by providing
    sample data formatted for dashboard consumption.
    
    - **query_id**: Optional query ID to format specific results
    - **sample_count**: Number of sample results to generate for demo
    
    Returns formatted data suitable for charts, tables, and summaries.
    """
    try:
        logger.debug("Formatting results for dashboard", sample_count=sample_count)
        
        # Create result formatting service
        formatter = ResultFormattingService()
        
        # Generate sample results for demo
        sample_results = []
        for i in range(sample_count):
            sample_results.append(QueryResult(
                id=f"sample_{i}",
                content=f"Sample EHS data result {i+1} with various content types and metadata",
                source=["database", "document", "sensor"][i % 3],
                confidence_score=0.6 + (i * 0.04),  # Varying confidence scores
                metadata={
                    "facility": f"Facility_{chr(65 + i % 5)}",
                    "category": ["consumption", "compliance", "emissions", "safety"][i % 4],
                    "timestamp": datetime.utcnow().isoformat()
                }
            ))
        
        # Format for dashboard
        formatted_data = formatter.format_results_for_dashboard(sample_results)
        
        logger.info(
            "Dashboard formatting completed",
            sample_count=sample_count,
            average_confidence=formatted_data["summary"]["average_confidence"]
        )
        
        return {
            "success": True,
            "message": "Results formatted for dashboard successfully",
            "data": formatted_data
        }
        
    except Exception as e:
        logger.error("Dashboard formatting failed", error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": ErrorType.PROCESSING_ERROR,
                "message": "Failed to format results for dashboard",
                "details": {"error": str(e)}
            }
        )


@router.get("/export")
async def export_results(
    format_type: str = "json",
    sample_count: int = 5
) -> Dict[str, Any]:
    """
    Export query results in various formats.
    
    This endpoint demonstrates result export capabilities by providing
    sample data in different export formats.
    
    - **format_type**: Export format (json, csv, xlsx)
    - **sample_count**: Number of sample results to export
    
    Returns formatted export data suitable for download or external use.
    """
    try:
        logger.debug(
            "Exporting results",
            format_type=format_type,
            sample_count=sample_count
        )
        
        # Create result formatting service
        formatter = ResultFormattingService()
        
        # Generate sample results for demo
        sample_results = []
        for i in range(sample_count):
            sample_results.append(QueryResult(
                id=f"export_{i}",
                content=f"Exportable EHS data {i+1}",
                source="database",
                confidence_score=0.8 + (i * 0.02),
                metadata={
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "data_type": "sample"
                }
            ))
        
        # Format for export
        export_data = formatter.format_results_for_export(sample_results, format_type)
        
        logger.info(
            "Results export completed",
            format_type=format_type,
            sample_count=sample_count
        )
        
        return {
            "success": True,
            "message": f"Results exported as {format_type} successfully",
            "export_data": export_data
        }
        
    except Exception as e:
        logger.error("Results export failed", error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": ErrorType.PROCESSING_ERROR,
                "message": "Failed to export results",
                "details": {"error": str(e), "format_type": format_type}
            }
        )


@router.get("/stats")
async def get_workflow_stats(
    workflow: EHSWorkflow = Depends(get_workflow)
) -> Dict[str, Any]:
    """
    Get workflow performance statistics and usage metrics.
    
    Returns detailed statistics about query processing performance,
    success rates, and component availability.
    """
    try:
        logger.debug("Retrieving workflow statistics")
        
        # Get workflow statistics
        stats = workflow.get_workflow_stats()
        
        logger.info("Workflow statistics retrieved", stats_keys=list(stats.keys()))
        
        return {
            "success": True,
            "message": "Workflow statistics retrieved successfully",
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": stats
        }
        
    except Exception as e:
        logger.error("Failed to retrieve workflow statistics", error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": ErrorType.PROCESSING_ERROR,
                "message": "Failed to retrieve workflow statistics",
                "details": {"error": str(e)}
            }
        )


# Demo endpoint to show classification examples
@router.get("/examples")
async def get_query_examples(
    query_router: QueryRouterAgent = Depends(get_query_router)
) -> Dict[str, Any]:
    """
    Get example queries for each intent type.
    
    Returns sample queries that demonstrate the different types of 
    EHS queries supported by the system.
    """
    try:
        logger.debug("Retrieving query examples")
        
        # Get examples from the query router
        examples = query_router.get_intent_examples()
        
        # Format for API response
        formatted_examples = {}
        for intent_type, example_queries in examples.items():
            formatted_examples[intent_type.value] = example_queries
        
        logger.info("Query examples retrieved", intent_count=len(formatted_examples))
        
        return {
            "success": True,
            "message": "Query examples retrieved successfully",
            "examples": formatted_examples,
            "usage_notes": [
                "These examples show different types of EHS queries supported",
                "Use these as templates for your own queries",
                "The system will automatically classify and route your queries",
                "Higher confidence scores indicate better query-intent matching"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to retrieve query examples", error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": ErrorType.PROCESSING_ERROR,
                "message": "Failed to retrieve query examples",
                "details": {"error": str(e)}
            }
        )