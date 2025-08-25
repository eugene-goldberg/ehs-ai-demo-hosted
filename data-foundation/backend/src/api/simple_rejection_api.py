"""
Simple Rejection API for Phase 1 Compatibility

This module provides a simplified API endpoint that wraps RejectedDocument nodes
from Neo4j and returns them in a format compatible with the Phase 1 rejection tracking API.

The endpoint queries for RejectedDocument nodes and transforms them into the expected
format with document_id, file_name, rejection_reason, rejection_status, created_at, and notes.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the src directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = parent_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from graph_query import get_graphDB_driver
from shared.common_fn import create_graph_database_connection

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter(
    prefix="/api/v1",
    tags=["Simple Rejection API"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


# Pydantic Models
class SimpleRejectedDocument(BaseModel):
    """Simple rejected document model for Phase 1 compatibility."""
    document_id: str = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    rejection_reason: str = Field(..., description="Reason for rejection")
    rejection_status: str = Field(default="rejected", description="Status of rejection")
    created_at: str = Field(..., description="Document creation timestamp")
    notes: Optional[str] = Field(None, description="Additional rejection notes")


class SimpleRejectionResponse(BaseModel):
    """Response model for simple rejected documents endpoint."""
    documents: List[SimpleRejectedDocument] = Field(..., description="List of rejected documents")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")


class PaginationInfo(BaseModel):
    """Pagination information model."""
    total: int = Field(..., description="Total number of documents")
    limit: int = Field(..., description="Items per page limit")
    offset: int = Field(..., description="Pagination offset")
    has_more: bool = Field(..., description="Whether more items are available")


def get_graph_connection():
    """
    Dependency to get Neo4j graph database connection.
    Uses environment variables for connection details.
    """
    try:
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        driver = get_graphDB_driver(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )
        return driver
    except Exception as e:
        logger.error(f"Failed to create graph database connection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database connection failed"
        )


def query_rejected_documents(driver, limit: int = 50, offset: int = 0) -> tuple:
    """
    Query RejectedDocument nodes from Neo4j database.
    
    Args:
        driver: Neo4j driver instance
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        
    Returns:
        tuple: (list of rejected documents, total count)
    """
    try:
        with driver.session() as session:
            # Query to get rejected documents with pagination
            # Updated to use actual property names from RejectedDocument nodes
            query = """
            MATCH (r:RejectedDocument)
            RETURN r.id as document_id,
                   r.original_filename as file_name,
                   r.rejection_reason as rejection_reason,
                   r.rejected_at as rejected_at,
                   r.upload_timestamp as upload_timestamp,
                   r.attempted_type as attempted_type,
                   r.confidence as confidence,
                   r.file_size as file_size,
                   r.page_count as page_count,
                   r.content_length as content_length,
                   r.upload_source as upload_source
            ORDER BY COALESCE(r.rejected_at, r.upload_timestamp) DESC
            SKIP $offset
            LIMIT $limit
            """
            
            # Query to get total count
            count_query = """
            MATCH (r:RejectedDocument)
            RETURN count(r) as total
            """
            
            # Execute queries
            result = session.run(query, {"limit": limit, "offset": offset})
            count_result = session.run(count_query)
            
            documents = []
            for record in result:
                # Create meaningful notes from available metadata
                notes_parts = []
                if record.get("attempted_type"):
                    notes_parts.append(f"Attempted type: {record.get('attempted_type')}")
                if record.get("confidence"):
                    notes_parts.append(f"Confidence: {record.get('confidence'):.3f}")
                if record.get("file_size"):
                    notes_parts.append(f"File size: {record.get('file_size')} bytes")
                if record.get("page_count"):
                    notes_parts.append(f"Pages: {record.get('page_count')}")
                if record.get("upload_source"):
                    notes_parts.append(f"Source: {record.get('upload_source')}")
                
                notes = "; ".join(notes_parts) if notes_parts else None
                
                # Transform Neo4j record to our model format
                doc_data = {
                    "document_id": record.get("document_id", "unknown"),
                    "file_name": record.get("file_name", "unknown.pdf"),
                    "rejection_reason": record.get("rejection_reason", "UNKNOWN"),
                    "rejection_status": "rejected",  # Always "rejected" for RejectedDocument nodes
                    "created_at": record.get("rejected_at") or record.get("upload_timestamp") or datetime.now().isoformat(),
                    "notes": notes
                }
                
                # Ensure document_id is not empty
                if not doc_data["document_id"] or doc_data["document_id"] == "unknown":
                    if doc_data["file_name"] and doc_data["file_name"] != "unknown.pdf":
                        # Use file_name as fallback for document_id if not available
                        doc_data["document_id"] = doc_data["file_name"]
                    else:
                        # Generate a unique ID as last resort
                        doc_data["document_id"] = f"rejected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                documents.append(doc_data)
            
            # Get total count
            total_count = 0
            count_record = count_result.single()
            if count_record:
                total_count = count_record.get("total", 0)
            
            return documents, total_count
            
    except Exception as e:
        logger.error(f"Error querying rejected documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query rejected documents: {str(e)}"
        )


@router.get("/simple-rejected-documents", response_model=SimpleRejectionResponse)
async def get_simple_rejected_documents(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    driver=Depends(get_graph_connection)
):
    """
    Get rejected documents in a format compatible with Phase 1 rejection tracking API.
    
    This endpoint queries RejectedDocument nodes from Neo4j and transforms them
    into the expected format with document_id, file_name, rejection_reason, 
    rejection_status, created_at, and notes.
    
    Args:
        limit: Maximum number of results to return (1-1000)
        offset: Number of results to skip for pagination
        
    Returns:
        SimpleRejectionResponse: List of rejected documents with pagination info
    """
    try:
        logger.info(f"Retrieving rejected documents with limit={limit}, offset={offset}")
        
        # Query rejected documents from Neo4j
        documents_data, total_count = query_rejected_documents(driver, limit, offset)
        
        # Transform to Pydantic models
        documents = []
        for doc_data in documents_data:
            try:
                document = SimpleRejectedDocument(**doc_data)
                documents.append(document)
            except Exception as e:
                logger.warning(f"Failed to create document model for {doc_data.get('document_id', 'unknown')}: {e}")
                # Continue with other documents even if one fails
                continue
        
        # Create pagination info
        pagination = PaginationInfo(
            total=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + limit < total_count)
        )
        
        # Create response
        response = SimpleRejectionResponse(
            documents=documents,
            pagination=pagination.dict()
        )
        
        logger.info(f"Successfully retrieved {len(documents)} rejected documents (total: {total_count})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_simple_rejected_documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve rejected documents: {str(e)}"
        )
    finally:
        # Ensure driver is closed
        if driver:
            try:
                driver.close()
            except:
                pass  # Ignore close errors


@router.get("/simple-rejected-documents/health")
async def simple_rejection_api_health():
    """
    Health check endpoint for the simple rejection API.
    """
    try:
        # Test basic database connectivity
        driver = get_graph_connection()
        
        with driver.session() as session:
            # Simple connectivity test
            result = session.run("RETURN 1 as test")
            test_result = result.single()
            
            # Test RejectedDocument query
            result = session.run("MATCH (r:RejectedDocument) RETURN count(r) as count")
            rejected_count = result.single()['count']
            
        driver.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "simple_rejection_api",
            "version": "1.0.0",
            "database_connection": "ok",
            "rejected_documents_count": rejected_count
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "service": "simple_rejection_api",
                "error": str(e),
                "database_connection": "failed"
            }
        )


# Export the router for integration with main application
simple_rejection_router = router