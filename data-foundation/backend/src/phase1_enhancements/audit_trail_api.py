"""
FastAPI Router for Document Audit Trail API Endpoints

This module provides REST API endpoints for managing document source files and audit information.
It includes functionality for downloading original files, getting file URLs, retrieving audit trails,
and updating source files for documents.

Dependencies:
- FastAPI for REST API framework
- AuditTrailService for file management operations
- Neo4j graph database for document metadata
- Proper error handling and authentication
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_503_SERVICE_UNAVAILABLE

from .audit_trail_service import AuditTrailService
from src.graph_query import get_graphDB_driver
from src.shared.common_fn import create_graph_database_connection

# Initialize logging
logger = logging.getLogger(__name__)

# Create FastAPI router with prefix
router = APIRouter(
    prefix="/documents",
    tags=["Document Audit Trail"],
    responses={
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)

# Service will be initialized in startup event
audit_service = None

def get_graph_connection():
    """
    Dependency to get Neo4j graph database connection.
    Uses environment variables for connection details.
    """
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        return create_graph_database_connection(uri, username, password, database)
    except Exception as e:
        logger.error(f"Failed to create graph connection: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )

def check_service_initialized():
    """
    Check if the audit trail service has been initialized.
    Raises HTTPException if service is not available.
    """
    if audit_service is None:
        logger.error("Audit trail service not initialized")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit trail service is not available. Service initialization may be in progress."
        )
    return audit_service

def get_document_info(document_id: str, graph) -> Optional[Dict[str, Any]]:
    """
    Retrieve document information from Neo4j database.
    
    Args:
        document_id: ID of the document
        graph: Neo4j graph connection
        
    Returns:
        Document information dictionary or None if not found
    """
    try:
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d.id as id, d.fileName as file_name, d.original_filename as original_filename,
               d.source_file_path as source_file_path, d.fileSize as file_size,
               d.fileSource as file_source, d.status as status, d.created_at as created_at,
               d.updated_at as updated_at, d.model as model, d.fileType as file_type
        """
        
        result = graph.query(query, params={"document_id": document_id})
        
        if result and len(result) > 0:
            return result[0]
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving document info for {document_id}: {str(e)}")
        return None

@router.get("/{document_id}/source_file", 
            summary="Download Original Source File",
            description="Download the original source file for a document")
async def download_source_file(
    document_id: str,
    graph=Depends(get_graph_connection)
):
    """
    Download the original source file for a document.
    
    Args:
        document_id: ID of the document
        
    Returns:
        FileResponse containing the original file
        
    Raises:
        HTTPException: If document not found or file not accessible
    """
    try:
        # Check if service is initialized
        service = check_service_initialized()
        
        # Get document information from database
        doc_info = get_document_info(document_id, graph)
        if not doc_info:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Get file path and prepare for serving
        original_filename = doc_info.get('original_filename') or doc_info.get('file_name')
        file_info = service.serve_source_file(document_id, original_filename)
        
        if not file_info:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Source file not found for document {document_id}"
            )
        
        file_path, download_filename, content_type = file_info
        
        # Verify file exists and is readable
        if not Path(file_path).exists():
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail="Source file no longer exists"
            )
        
        logger.info(f"Serving source file for document {document_id}: {download_filename}")
        
        return FileResponse(
            path=file_path,
            filename=download_filename,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{download_filename}\"",
                "X-Document-ID": document_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving source file for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to serve source file"
        )

@router.get("/{document_id}/source_url",
            summary="Get Source File Download URL",
            description="Get a download URL for the document's source file",
            response_model=Dict[str, Any])
async def get_source_url(
    document_id: str,
    graph=Depends(get_graph_connection)
):
    """
    Get a download URL for the document's source file.
    
    Args:
        document_id: ID of the document
        
    Returns:
        JSON response with download URL and file information
        
    Raises:
        HTTPException: If document not found or file not accessible
    """
    try:
        # Check if service is initialized
        service = check_service_initialized()
        
        # Get document information from database
        doc_info = get_document_info(document_id, graph)
        if not doc_info:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Check if source file exists
        original_filename = doc_info.get('original_filename') or doc_info.get('file_name')
        file_path = service.get_source_file_path(document_id, original_filename)
        
        if not file_path:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Source file not found for document {document_id}"
            )
        
        # Generate download URL (relative to API base)
        download_url = f"/api/v1/documents/{document_id}/source_file"
        
        # Get file information
        file_path_obj = Path(file_path)
        file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
        
        response_data = {
            "document_id": document_id,
            "download_url": download_url,
            "filename": original_filename,
            "file_size": file_size,
            "file_type": doc_info.get('file_type'),
            "content_type": service._get_content_type(file_path_obj.suffix.lower()),
            "timestamp": datetime.now().isoformat(),
            "available": file_path_obj.exists()
        }
        
        logger.info(f"Generated download URL for document {document_id}")
        
        return JSONResponse(
            status_code=HTTP_200_OK,
            content=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating source URL for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate source URL"
        )

@router.get("/{document_id}/audit_info",
            summary="Get Document Audit Information",
            description="Retrieve comprehensive audit trail information for a document",
            response_model=Dict[str, Any])
async def get_audit_info(
    document_id: str,
    graph=Depends(get_graph_connection)
):
    """
    Get comprehensive audit trail information for a document.
    
    Args:
        document_id: ID of the document
        
    Returns:
        JSON response with audit trail information
        
    Raises:
        HTTPException: If document not found
    """
    try:
        # Check if service is initialized
        service = check_service_initialized()
        
        # Get document information from database
        doc_info = get_document_info(document_id, graph)
        if not doc_info:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Validate document integrity
        integrity_info = service.validate_document_integrity(document_id)
        
        # Get processing history (chunks, entities, relationships)
        processing_query = """
        MATCH (d:Document {id: $document_id})
        OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
        OPTIONAL MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
        WHERE e1.id CONTAINS $document_id OR e2.id CONTAINS $document_id
        RETURN d.nodeCount as node_count, d.relationshipCount as relationship_count,
               d.chunkNodeCount as chunk_count, d.entityNodeCount as entity_count,
               d.total_chunks as total_chunks, d.processed_chunk as processed_chunks,
               d.processing_time as processing_time, d.status as processing_status,
               count(DISTINCT c) as actual_chunks, count(DISTINCT e) as actual_entities,
               count(DISTINCT r) as actual_relationships
        """
        
        processing_result = graph.query(processing_query, params={"document_id": document_id})
        processing_info = processing_result[0] if processing_result else {}
        
        # Compile comprehensive audit information
        audit_data = {
            "document_id": document_id,
            "document_info": {
                "file_name": doc_info.get('file_name'),
                "original_filename": doc_info.get('original_filename'),
                "file_type": doc_info.get('file_type'),
                "file_size": doc_info.get('file_size'),
                "file_source": doc_info.get('file_source'),
                "model": doc_info.get('model'),
                "status": doc_info.get('status'),
                "created_at": doc_info.get('created_at'),
                "updated_at": doc_info.get('updated_at')
            },
            "file_integrity": {
                "source_file_exists": integrity_info.get('is_valid', False),
                "files_found": integrity_info.get('files_found', []),
                "total_files": integrity_info.get('total_files', 0),
                "total_size": integrity_info.get('total_size', 0)
            },
            "processing_info": {
                "status": processing_info.get('processing_status'),
                "total_chunks": processing_info.get('total_chunks', 0),
                "processed_chunks": processing_info.get('processed_chunks', 0),
                "actual_chunks": processing_info.get('actual_chunks', 0),
                "node_count": processing_info.get('node_count', 0),
                "entity_count": processing_info.get('entity_count', 0),
                "relationship_count": processing_info.get('relationship_count', 0),
                "actual_entities": processing_info.get('actual_entities', 0),
                "actual_relationships": processing_info.get('actual_relationships', 0),
                "processing_time": processing_info.get('processing_time')
            },
            "audit_trail": {
                "has_source_file": bool(doc_info.get('source_file_path')),
                "source_file_path": doc_info.get('source_file_path'),
                "integrity_validated": True,
                "validation_timestamp": integrity_info.get('timestamp'),
                "audit_timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Generated audit info for document {document_id}")
        
        return JSONResponse(
            status_code=HTTP_200_OK,
            content=audit_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audit info for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit information"
        )

@router.post("/{document_id}/update_source",
             summary="Update Source File for Document",
             description="Upload a new source file to replace the existing one for a document",
             response_model=Dict[str, Any])
async def update_source_file(
    document_id: str,
    file: UploadFile = File(..., description="New source file to upload"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    graph=Depends(get_graph_connection)
):
    """
    Update the source file for an existing document.
    
    Args:
        document_id: ID of the document to update
        file: New source file to upload
        background_tasks: Background tasks for cleanup operations
        
    Returns:
        JSON response with update status and file information
        
    Raises:
        HTTPException: If document not found, file invalid, or update fails
    """
    try:
        # Check if service is initialized
        service = check_service_initialized()
        
        # Get document information from database
        doc_info = get_document_info(document_id, graph)
        if not doc_info:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Validate uploaded file
        if not file.filename:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Check file size (limit to 100MB)
        max_file_size = 100 * 1024 * 1024  # 100MB
        file_content = await file.read()
        if len(file_content) > max_file_size:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="File size exceeds maximum limit (100MB)"
            )
        
        # Create temporary file for processing
        temp_dir = Path("/tmp/audit_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        import uuid
        temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        
        try:
            # Write uploaded content to temporary file
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            
            # Backup existing source file if it exists
            existing_file_path = service.get_source_file_path(document_id)
            backup_path = None
            
            if existing_file_path:
                backup_path = f"{existing_file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                Path(existing_file_path).rename(backup_path)
                logger.info(f"Backed up existing file to: {backup_path}")
            
            # Store new source file
            stored_document_id, stored_file_path = service.store_source_file(
                str(temp_file_path), file.filename, document_id
            )
            
            # Update document metadata in Neo4j
            update_query = """
            MATCH (d:Document {id: $document_id})
            SET d.original_filename = $original_filename,
                d.source_file_path = $source_file_path,
                d.fileSize = $file_size,
                d.updated_at = $updated_at,
                d.fileType = $file_type
            RETURN d.id as id
            """
            
            file_extension = Path(file.filename).suffix.lower().lstrip('.')
            
            update_result = graph.query(update_query, params={
                "document_id": document_id,
                "original_filename": file.filename,
                "source_file_path": stored_file_path,
                "file_size": len(file_content),
                "file_type": file_extension,
                "updated_at": datetime.now().isoformat()
            })
            
            if not update_result:
                # Rollback file storage if database update fails
                if Path(stored_file_path).exists():
                    Path(stored_file_path).unlink()
                
                # Restore backup if it exists
                if backup_path and Path(backup_path).exists():
                    Path(backup_path).rename(existing_file_path)
                
                raise HTTPException(
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update document metadata"
                )
            
            # Schedule cleanup of backup file (after 24 hours)
            if backup_path:
                background_tasks.add_task(cleanup_backup_file, backup_path)
            
            response_data = {
                "document_id": document_id,
                "status": "success",
                "message": "Source file updated successfully",
                "file_info": {
                    "filename": file.filename,
                    "file_size": len(file_content),
                    "file_type": file_extension,
                    "content_type": file.content_type,
                    "stored_path": stored_file_path
                },
                "backup_created": backup_path is not None,
                "updated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully updated source file for document {document_id}")
            
            return JSONResponse(
                status_code=HTTP_201_CREATED,
                content=response_data
            )
            
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating source file for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update source file"
        )

async def cleanup_backup_file(backup_path: str):
    """
    Background task to clean up backup files after a delay.
    
    Args:
        backup_path: Path to backup file to remove
    """
    try:
        # Wait 24 hours before cleanup (in production, this would be handled by a job scheduler)
        import asyncio
        await asyncio.sleep(24 * 60 * 60)  # 24 hours
        
        backup_file = Path(backup_path)
        if backup_file.exists():
            backup_file.unlink()
            logger.info(f"Cleaned up backup file: {backup_path}")
    except Exception as e:
        logger.error(f"Error cleaning up backup file {backup_path}: {str(e)}")

# Health check endpoint for the audit trail API
@router.get("/audit/health",
            summary="Audit Trail API Health Check",
            description="Check the health status of the audit trail API and services",
            response_model=Dict[str, Any])
async def audit_health_check():
    """
    Health check endpoint for the audit trail API.
    
    Returns:
        JSON response with health status information
    """
    try:
        # Check audit service health
        service_healthy = audit_service is not None
        storage_stats = {}
        
        if service_healthy:
            storage_stats = audit_service.get_storage_stats()
        else:
            storage_stats = {"error": "Service not initialized"}
        
        # Check database connectivity
        try:
            graph = get_graph_connection()
            db_healthy = True
            db_error = None
        except Exception as e:
            db_healthy = False
            db_error = str(e)
        
        overall_healthy = service_healthy and db_healthy and "error" not in storage_stats
        
        health_data = {
            "service": "audit_trail_api",
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "file_storage": {
                    "healthy": service_healthy and "error" not in storage_stats,
                    "initialized": audit_service is not None,
                    "stats": storage_stats
                },
                "database": {
                    "healthy": db_healthy,
                    "error": db_error
                }
            }
        }
        
        status_code = HTTP_200_OK if overall_healthy else HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content=health_data
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "service": "audit_trail_api",
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )
# Export router as audit_trail_router for external imports
audit_trail_router = router