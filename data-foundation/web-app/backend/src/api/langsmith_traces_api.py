"""
LangSmith Traces API

FastAPI router providing endpoints for fetching LangSmith traces and project information.
Integrates with the LangSmithClient to provide a RESTful interface for trace data.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import os

# Import the existing LangSmith client
try:
    from ..langsmith_fetcher import LangSmithClient, TraceData, LangSmithFetcherError
except ImportError:
    from langsmith_fetcher import LangSmithClient, TraceData, LangSmithFetcherError

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/langsmith", tags=["LangSmith Traces"])


# ==================== REQUEST/RESPONSE MODELS ====================

class ProjectInfo(BaseModel):
    """Model for project information."""
    id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: str = Field(default="", description="Project description")
    created_at: Optional[str] = Field(None, description="Project creation timestamp")

class ProjectListResponse(BaseModel):
    """Response model for listing projects."""
    success: bool = Field(True, description="Request success status")
    data: List[ProjectInfo] = Field(..., description="List of available projects")
    count: int = Field(..., description="Number of projects returned")

class TraceInfo(BaseModel):
    """Simplified trace information for list views."""
    run_id: str = Field(..., description="Unique run identifier")
    name: str = Field(..., description="Trace name")
    run_type: str = Field(..., description="Type of run (e.g., chain, llm)")
    status: str = Field(..., description="Execution status (completed, error, running)")
    start_time: str = Field(..., description="Start timestamp (ISO format)")
    end_time: Optional[str] = Field(None, description="End timestamp (ISO format)")
    latency_ms: Optional[float] = Field(None, description="Execution latency in milliseconds")
    tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    model_name: Optional[str] = Field(None, description="Model used for execution")
    error: Optional[str] = Field(None, description="Error message if failed")
    has_children: bool = Field(False, description="Whether trace has child runs")

class TracesResponse(BaseModel):
    """Response model for trace listings."""
    success: bool = Field(True, description="Request success status")
    data: List[TraceInfo] = Field(..., description="List of traces")
    count: int = Field(..., description="Number of traces returned")
    project_name: str = Field(..., description="Project name queried")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")

class TraceDetail(BaseModel):
    """Detailed trace information model."""
    run_id: str = Field(..., description="Unique run identifier")
    name: str = Field(..., description="Trace name")
    run_type: str = Field(..., description="Type of run")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    start_time: str = Field(..., description="Start timestamp")
    end_time: Optional[str] = Field(None, description="End timestamp")
    latency_ms: Optional[float] = Field(None, description="Execution latency")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    model_name: Optional[str] = Field(None, description="Model used")
    status: str = Field(..., description="Execution status")
    error: Optional[str] = Field(None, description="Error message")
    parent_run_id: Optional[str] = Field(None, description="Parent run ID")
    child_runs: List[str] = Field(default_factory=list, description="Child run IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class TraceDetailResponse(BaseModel):
    """Response model for detailed trace information."""
    success: bool = Field(True, description="Request success status")
    data: TraceDetail = Field(..., description="Detailed trace information")

class ErrorDetail(BaseModel):
    """Error detail model."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: bool = Field(False, description="Request success status")
    error: ErrorDetail = Field(..., description="Error information")


# ==================== REQUEST VALIDATION ====================

class TracesQueryParams(BaseModel):
    """Query parameters for traces endpoint."""
    start_time: Optional[str] = Field(None, description="Start time (ISO format)")
    end_time: Optional[str] = Field(None, description="End time (ISO format)")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of traces to return")

    @validator('start_time', 'end_time', pre=True)
    def validate_timestamp(cls, v):
        """Validate timestamp format."""
        if v is None:
            return v
        try:
            # Try to parse the timestamp
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError("Timestamp must be in ISO format (e.g., 2024-01-01T00:00:00Z)")


# ==================== DEPENDENCY INJECTION ====================

def get_langsmith_client() -> LangSmithClient:
    """Dependency to get LangSmith client instance."""
    try:
        return LangSmithClient()
    except LangSmithFetcherError as e:
        logger.error(f"Failed to initialize LangSmith client: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "LangSmith service is currently unavailable",
                "details": {"error": str(e)}
            }
        )


# ==================== API ENDPOINTS ====================

@router.get(
    "/projects",
    response_model=ProjectListResponse,
    summary="List Available Projects",
    description="Retrieve a list of all available LangSmith projects.",
    responses={
        200: {"description": "Successfully retrieved project list"},
        503: {"description": "LangSmith service unavailable", "model": ErrorResponse}
    }
)
async def list_projects(client: LangSmithClient = Depends(get_langsmith_client)):
    """
    List all available LangSmith projects.
    
    Returns a list of projects with their basic information including
    ID, name, description, and creation timestamp.
    """
    try:
        logger.info("Fetching available LangSmith projects")
        
        projects_data = client.list_projects()
        projects = [ProjectInfo(**project) for project in projects_data]
        
        logger.info(f"Successfully retrieved {len(projects)} projects")
        
        return ProjectListResponse(
            data=projects,
            count=len(projects)
        )
        
    except LangSmithFetcherError as e:
        logger.error(f"LangSmith error while listing projects: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LANGSMITH_ERROR",
                "message": "Failed to retrieve projects from LangSmith",
                "details": {"error": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while listing projects: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error occurred",
                "details": {"error": str(e)}
            }
        )


@router.get(
    "/traces/{project_name}",
    response_model=TracesResponse,
    summary="Get Project Traces",
    description="Fetch traces for a specific project with optional time filtering.",
    responses={
        200: {"description": "Successfully retrieved traces"},
        400: {"description": "Invalid query parameters", "model": ErrorResponse},
        404: {"description": "Project not found", "model": ErrorResponse},
        502: {"description": "LangSmith service error", "model": ErrorResponse}
    }
)
async def get_project_traces(
    project_name: str = Path(..., description="Name of the project to fetch traces for"),
    start_time: Optional[str] = Query(None, description="Start time for filtering (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time for filtering (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of traces to return"),
    client: LangSmithClient = Depends(get_langsmith_client)
):
    """
    Fetch traces for a specific LangSmith project.
    
    Supports time-based filtering using start_time and end_time parameters.
    Results are limited to prevent excessive data transfer.
    """
    try:
        # Validate query parameters
        params = TracesQueryParams(
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        logger.info(f"Fetching traces for project: {project_name} with limit: {limit}")
        
        # Fetch traces from LangSmith
        traces_data = client.fetch_traces_for_project(
            project_name=project_name,
            start_time=params.start_time,
            end_time=params.end_time,
            limit=params.limit
        )
        
        # Convert to simplified trace info for list view
        traces = []
        for trace_data in traces_data:
            trace_info = TraceInfo(
                run_id=trace_data.run_id,
                name=trace_data.name,
                run_type=trace_data.run_type,
                status=trace_data.status,
                start_time=trace_data.start_time,
                end_time=trace_data.end_time,
                latency_ms=trace_data.latency_ms,
                tokens_used=trace_data.tokens_used,
                model_name=trace_data.model_name,
                error=trace_data.error,
                has_children=len(trace_data.child_runs) > 0
            )
            traces.append(trace_info)
        
        # Prepare filter information for response
        filters = {
            "start_time": params.start_time,
            "end_time": params.end_time,
            "limit": params.limit
        }
        
        logger.info(f"Successfully retrieved {len(traces)} traces for project {project_name}")
        
        return TracesResponse(
            data=traces,
            count=len(traces),
            project_name=project_name,
            filters=filters
        )
        
    except ValueError as e:
        logger.warning(f"Invalid query parameters: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_PARAMETERS",
                "message": "Invalid query parameters provided",
                "details": {"error": str(e)}
            }
        )
    except LangSmithFetcherError as e:
        logger.error(f"LangSmith error while fetching traces: {e}")
        
        # Check if it's a "not found" type error
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "PROJECT_NOT_FOUND",
                    "message": f"Project '{project_name}' not found",
                    "details": {"error": str(e)}
                }
            )
        
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LANGSMITH_ERROR",
                "message": "Failed to retrieve traces from LangSmith",
                "details": {"error": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while fetching traces: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error occurred",
                "details": {"error": str(e)}
            }
        )


@router.get(
    "/trace/{run_id}",
    response_model=TraceDetailResponse,
    summary="Get Trace Details",
    description="Fetch detailed information about a specific trace run.",
    responses={
        200: {"description": "Successfully retrieved trace details"},
        404: {"description": "Trace not found", "model": ErrorResponse},
        502: {"description": "LangSmith service error", "model": ErrorResponse}
    }
)
async def get_trace_details(
    run_id: str = Path(..., description="Unique identifier of the trace run"),
    client: LangSmithClient = Depends(get_langsmith_client)
):
    """
    Get detailed information about a specific trace run.
    
    Returns comprehensive information including inputs, outputs, child runs,
    and metadata for the specified trace.
    """
    try:
        logger.info(f"Fetching details for trace: {run_id}")
        
        # Fetch detailed trace information
        trace_data = client.fetch_trace_details(run_id)
        
        # Convert to detailed response model
        trace_detail = TraceDetail(
            run_id=trace_data.run_id,
            name=trace_data.name,
            run_type=trace_data.run_type,
            inputs=trace_data.inputs,
            outputs=trace_data.outputs,
            start_time=trace_data.start_time,
            end_time=trace_data.end_time,
            latency_ms=trace_data.latency_ms,
            tokens_used=trace_data.tokens_used,
            model_name=trace_data.model_name,
            status=trace_data.status,
            error=trace_data.error,
            parent_run_id=trace_data.parent_run_id,
            child_runs=trace_data.child_runs,
            metadata=trace_data.metadata
        )
        
        logger.info(f"Successfully retrieved details for trace {run_id}")
        
        return TraceDetailResponse(data=trace_detail)
        
    except LangSmithFetcherError as e:
        logger.error(f"LangSmith error while fetching trace details: {e}")
        
        # Check if it's a "not found" type error
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "TRACE_NOT_FOUND",
                    "message": f"Trace with ID '{run_id}' not found",
                    "details": {"error": str(e)}
                }
            )
        
        raise HTTPException(
            status_code=502,
            detail={
                "code": "LANGSMITH_ERROR",
                "message": "Failed to retrieve trace details from LangSmith",
                "details": {"error": str(e)}
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error while fetching trace details: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Internal server error occurred",
                "details": {"error": str(e)}
            }
        )


# ==================== HEALTH CHECK ====================

@router.get(
    "/health",
    summary="Health Check",
    description="Check if LangSmith API integration is healthy.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    }
)
async def health_check():
    """
    Health check endpoint to verify LangSmith integration status.
    
    Returns service status and basic connectivity information.
    """
    try:
        # Try to initialize client to check connectivity
        client = LangSmithClient()
        
        # Try a lightweight operation
        projects = client.list_projects()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "langsmith_traces_api",
                "timestamp": datetime.now().isoformat(),
                "langsmith_available": True,
                "projects_accessible": len(projects) > 0
            }
        )
        
    except LangSmithFetcherError as e:
        logger.warning(f"LangSmith service check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "langsmith_traces_api",
                "timestamp": datetime.now().isoformat(),
                "langsmith_available": False,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during health check: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "langsmith_traces_api",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )


# ==================== ROUTER CONFIGURATION ====================

# Add CORS middleware support (if needed by your application)
# This would typically be configured at the main app level

# Example of how to include this router in your main FastAPI app:
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.langsmith_traces_api import router as langsmith_router

app = FastAPI(title="EHS AI Demo API", version="1.0.0")

# Configure CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the LangSmith traces router
app.include_router(langsmith_router)
"""