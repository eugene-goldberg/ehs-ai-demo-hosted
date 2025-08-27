"""
FastAPI router for serving LLM transcript data.

Provides REST API endpoints for accessing LLM interaction transcript logs
captured during the ingestion and extraction workflow processes.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils.llm_transcript_logger import get_global_logger, TranscriptLogger

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI router with tags (no prefix to avoid duplication)
router = APIRouter(tags=["transcript"])

# Response models
class TranscriptEntry(BaseModel):
    """Individual transcript entry model."""
    timestamp: str = Field(description="ISO timestamp of the interaction")
    unix_timestamp: float = Field(description="Unix timestamp")
    role: str = Field(description="Role of the message sender (system/user/assistant)")
    content: str = Field(description="Content of the message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context information")
    entry_id: int = Field(description="Unique entry identifier")

class TranscriptResponse(BaseModel):
    """Response model for transcript endpoint."""
    transcript: List[TranscriptEntry] = Field(description="Array of transcript entries")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    status: str = Field(default="success", description="Response status")

class TranscriptStats(BaseModel):
    """Statistics about the transcript."""
    total_entries: int = Field(description="Total number of entries")
    role_counts: Dict[str, int] = Field(description="Count by role")
    first_entry_time: Optional[str] = Field(None, description="Timestamp of first entry")
    last_entry_time: Optional[str] = Field(None, description="Timestamp of last entry")
    total_content_length: int = Field(description="Total content length across all entries")

class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    stats: TranscriptStats = Field(description="Transcript statistics")
    status: str = Field(default="success", description="Response status")


def _validate_and_clean_role_filter(role_filter: Any) -> Optional[str]:
    """Helper function to validate and clean role filter parameter."""
    if role_filter is None:
        return None
    
    # Handle the case where role_filter might be a Query object or other type
    try:
        role_str = str(role_filter).strip() if role_filter else ""
        if not role_str or role_str == "None":
            return None
        
        valid_roles = {'system', 'user', 'assistant'}
        if role_str.lower() not in valid_roles:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid role_filter '{role_str}'. Must be one of: {', '.join(valid_roles)}"
            )
        
        return role_str.lower()
        
    except Exception as e:
        # If we can't process it, treat as None
        logger.warning(f"Could not process role_filter parameter: {e}")
        return None


@router.get("/api/data/transcript", response_model=TranscriptResponse)
async def get_transcript(
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Maximum number of entries to return"),
    role_filter: Optional[str] = Query(None, description="Filter by role (system/user/assistant)"),
    start_index: Optional[int] = Query(None, ge=0, description="Starting index for pagination"),
    end_index: Optional[int] = Query(None, ge=0, description="Ending index for pagination")
) -> TranscriptResponse:
    """
    Get LLM transcript data with optional filtering and pagination.
    
    Returns transcript entries in chronological order with support for:
    - Role-based filtering
    - Pagination with start/end indices
    - Limiting the number of results
    
    Args:
        limit: Maximum number of entries to return (1-10000)
        role_filter: Filter entries by role (system/user/assistant)
        start_index: Starting index for slicing (inclusive)
        end_index: Ending index for slicing (exclusive)
    
    Returns:
        JSON response with transcript array and metadata
    """
    try:
        # Get the global transcript logger instance
        transcript_logger = get_global_logger()
        
        # Validate and clean role filter
        filter_role = _validate_and_clean_role_filter(role_filter)
        
        # Validate index range if both are provided
        if start_index is not None and end_index is not None:
            if start_index >= end_index:
                raise HTTPException(
                    status_code=400,
                    detail="start_index must be less than end_index"
                )
        
        # Get transcript data with filtering
        transcript_data = transcript_logger.get_transcript(
            start_index=start_index,
            end_index=end_index,
            role_filter=filter_role
        )
        
        # Apply limit if specified
        if limit and len(transcript_data) > limit:
            transcript_data = transcript_data[:limit]
        
        # Convert to response format
        transcript_entries = [
            TranscriptEntry(**entry) for entry in transcript_data
        ]
        
        # Generate metadata
        metadata = {
            "total_returned": len(transcript_entries),
            "filters_applied": {
                "role_filter": filter_role,
                "limit": limit,
                "start_index": start_index,
                "end_index": end_index
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Returned {len(transcript_entries)} transcript entries with filters: {metadata['filters_applied']}")
        
        return TranscriptResponse(
            transcript=transcript_entries,
            metadata=metadata,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving transcript data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve transcript data: {str(e)}"
        )


@router.get("/api/data/transcript/stats", response_model=StatsResponse)
async def get_transcript_stats() -> StatsResponse:
    """
    Get statistics about the transcript data.
    
    Returns summary information including:
    - Total number of entries
    - Counts by role
    - Time range of entries
    - Total content length
    
    Returns:
        JSON response with transcript statistics
    """
    try:
        # Get the global transcript logger instance
        transcript_logger = get_global_logger()
        
        # Get statistics from the logger
        stats_data = transcript_logger.get_stats()
        
        # Convert to response format
        stats = TranscriptStats(**stats_data)
        
        logger.info(f"Retrieved transcript stats: {stats_data['total_entries']} entries")
        
        return StatsResponse(
            stats=stats,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving transcript stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve transcript statistics: {str(e)}"
        )


@router.delete("/api/data/transcript")
async def clear_transcript() -> JSONResponse:
    """
    Clear all transcript entries.
    
    WARNING: This operation permanently removes all logged interactions.
    Use with caution.
    
    Returns:
        JSON response with the number of entries that were cleared
    """
    try:
        # Get the global transcript logger instance
        transcript_logger = get_global_logger()
        
        # Clear all entries
        cleared_count = transcript_logger.clear_transcript()
        
        logger.warning(f"Cleared {cleared_count} transcript entries")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Cleared {cleared_count} transcript entries",
                "entries_cleared": cleared_count,
                "cleared_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error clearing transcript data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear transcript data: {str(e)}"
        )


@router.get("/api/data/transcript/health")
async def transcript_health_check() -> JSONResponse:
    """
    Health check endpoint for the transcript API.
    
    Verifies that the transcript logger is accessible and functional.
    
    Returns:
        JSON response with health status
    """
    try:
        # Get the global transcript logger instance
        transcript_logger = get_global_logger()
        
        # Basic functionality test
        current_count = len(transcript_logger)
        stats = transcript_logger.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "logger_accessible": True,
                "current_entries": current_count,
                "logger_type": str(type(transcript_logger).__name__),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Transcript health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "logger_accessible": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )