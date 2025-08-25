"""
Audit Trail Integration Module

This module provides integration components for the audit trail system,
enabling seamless integration with the existing document processing pipeline.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase

from ..audit_trail.schema import AuditTrailSchema
from ..audit_trail.service import AuditTrailService
from ..audit_trail.api import audit_trail_router


logger = logging.getLogger(__name__)


class AuditTrailIntegration:
    """
    Main integration class for audit trail functionality.
    
    This class orchestrates the integration of audit trail components
    with the existing document processing pipeline.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize the audit trail integration.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        self.schema = None
        self.service = None
        self.initialized = False
        
        # Docker volume paths configuration
        self.source_files_path = Path("/app/data/source_files")
        self.processed_files_path = Path("/app/data/processed_files")
        
    def initialize_audit_trail(self) -> bool:
        """
        Initialize the audit trail system components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize Neo4j driver
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
                
            logger.info("Neo4j connection established successfully")
            
            # Initialize schema
            self.schema = AuditTrailSchema(self.driver)
            self.schema.create_schema()
            logger.info("Audit trail schema created successfully")
            
            # Initialize service
            self.service = AuditTrailService(self.driver)
            logger.info("Audit trail service initialized successfully")
            
            # Ensure directory structure exists
            self._ensure_directory_structure()
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audit trail: {str(e)}")
            return False
    
    def _ensure_directory_structure(self):
        """Ensure required directory structure exists."""
        self.source_files_path.mkdir(parents=True, exist_ok=True)
        self.processed_files_path.mkdir(parents=True, exist_ok=True)
        logger.info("Directory structure verified")
    
    def process_document_with_audit(
        self, 
        file_path: str, 
        original_filename: str,
        user_id: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document with full audit trail integration.
        
        This method wraps the existing document processing pipeline
        with audit trail functionality.
        
        Args:
            file_path: Path to the uploaded file
            original_filename: Original name of the uploaded file
            user_id: ID of the user uploading the document
            metadata: Additional metadata for the document
            
        Returns:
            Dict containing processing results and audit information
        """
        if not self.initialized:
            raise RuntimeError("Audit trail system not initialized")
            
        try:
            # Store source file with audit trail
            source_audit_id = self._store_source_file(
                file_path, original_filename, user_id, metadata
            )
            
            # Process document (placeholder for actual processing pipeline)
            processed_result = self._process_document_pipeline(file_path)
            
            # Create document node with audit properties
            document_id = self._create_document_node(
                original_filename, 
                source_audit_id,
                processed_result,
                user_id,
                metadata
            )
            
            # Record processing completion
            self.service.record_action(
                entity_id=document_id,
                entity_type="Document",
                action="processing_completed",
                user_id=user_id,
                details={
                    "processing_result": processed_result,
                    "source_audit_id": source_audit_id
                }
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "source_audit_id": source_audit_id,
                "processed_result": processed_result
            }
            
        except Exception as e:
            logger.error(f"Error in document processing with audit: {str(e)}")
            # Record error in audit trail
            if hasattr(self, 'service') and self.service:
                self.service.record_action(
                    entity_id="unknown",
                    entity_type="Document",
                    action="processing_failed",
                    user_id=user_id,
                    details={"error": str(e), "file": original_filename}
                )
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    
    def _store_source_file(
        self, 
        file_path: str, 
        original_filename: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store source file and create audit trail entry.
        
        Args:
            file_path: Path to the source file
            original_filename: Original filename
            user_id: User who uploaded the file
            metadata: Additional metadata
            
        Returns:
            str: Audit trail ID for the stored file
        """
        # Generate unique filename for source storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stored_filename = f"{timestamp}_{original_filename}"
        stored_path = self.source_files_path / stored_filename
        
        # Copy source file to audit storage
        import shutil
        shutil.copy2(file_path, stored_path)
        
        # Create audit trail entry
        audit_id = self.service.store_file(
            file_path=str(stored_path),
            original_filename=original_filename,
            file_type="source",
            user_id=user_id,
            metadata=metadata or {}
        )
        
        logger.info(f"Source file stored with audit ID: {audit_id}")
        return audit_id
    
    def _process_document_pipeline(self, file_path: str) -> Dict[str, Any]:
        """
        Placeholder for actual document processing pipeline.
        
        This should be replaced with calls to the actual document
        processing components.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            Dict containing processing results
        """
        # TODO: Replace with actual document processing pipeline
        # This is a placeholder implementation
        
        file_size = os.path.getsize(file_path)
        file_extension = Path(file_path).suffix.lower()
        
        return {
            "status": "processed",
            "file_size": file_size,
            "file_type": file_extension,
            "processing_time": datetime.now().isoformat(),
            "extracted_entities": [],  # Placeholder
            "processed_text_length": 0,  # Placeholder
        }
    
    def _create_document_node(
        self,
        original_filename: str,
        source_audit_id: str,
        processed_result: Dict[str, Any],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create document node in Neo4j with audit properties.
        
        Args:
            original_filename: Original filename
            source_audit_id: ID of the source file audit entry
            processed_result: Results from document processing
            user_id: User who uploaded the document
            metadata: Additional metadata
            
        Returns:
            str: Document node ID
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (d:Document {
                    id: randomUUID(),
                    original_filename: $original_filename,
                    source_audit_id: $source_audit_id,
                    uploaded_by: $user_id,
                    upload_timestamp: datetime(),
                    file_size: $file_size,
                    file_type: $file_type,
                    processing_status: $status,
                    processed_at: datetime($processed_at),
                    metadata: $metadata
                })
                RETURN d.id as document_id
                """,
                original_filename=original_filename,
                source_audit_id=source_audit_id,
                user_id=user_id,
                file_size=processed_result.get("file_size", 0),
                file_type=processed_result.get("file_type", "unknown"),
                status=processed_result.get("status", "unknown"),
                processed_at=processed_result.get("processing_time"),
                metadata=metadata or {}
            )
            
            record = result.single()
            document_id = record["document_id"]
            
            logger.info(f"Document node created with ID: {document_id}")
            return document_id
    
    def integrate_with_main_app(self, app: FastAPI) -> bool:
        """
        Integrate audit trail API router with the main FastAPI application.
        
        Args:
            app: FastAPI application instance
            
        Returns:
            bool: True if integration successful
        """
        try:
            if not self.initialized:
                logger.warning("Audit trail not initialized, skipping API integration")
                return False
                
            # Include the audit trail router
            app.include_router(
                audit_trail_router,
                prefix="/api/v1/audit",
                tags=["audit-trail"]
            )
            
            # Add audit trail service to app state
            app.state.audit_service = self.service
            
            logger.info("Audit trail API integrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate audit trail API: {str(e)}")
            return False
    
    def configure_docker_volumes(self, source_path: str, processed_path: str):
        """
        Configure Docker volume paths for file storage.
        
        Args:
            source_path: Path for source files storage
            processed_path: Path for processed files storage
        """
        self.source_files_path = Path(source_path)
        self.processed_files_path = Path(processed_path)
        self._ensure_directory_structure()
        logger.info(f"Docker volumes configured: source={source_path}, processed={processed_path}")
    
    def migrate_existing_documents(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Migration script for existing documents in the system.
        
        This method helps migrate existing documents to include
        audit trail information.
        
        Args:
            batch_size: Number of documents to process in each batch
            
        Returns:
            Dict containing migration statistics
        """
        if not self.initialized:
            raise RuntimeError("Audit trail system not initialized")
            
        try:
            migration_stats = {
                "total_processed": 0,
                "successful_migrations": 0,
                "failed_migrations": 0,
                "errors": []
            }
            
            # Query existing documents without audit trail
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document) 
                    WHERE NOT EXISTS(d.source_audit_id)
                    RETURN d.id as doc_id, d.original_filename as filename
                    LIMIT $batch_size
                    """,
                    batch_size=batch_size
                )
                
                documents = list(result)
                migration_stats["total_processed"] = len(documents)
                
                for doc in documents:
                    try:
                        # Create retrospective audit entry
                        audit_id = self.service.record_action(
                            entity_id=doc["doc_id"],
                            entity_type="Document",
                            action="migrated_to_audit_trail",
                            user_id="system",
                            details={
                                "migration_timestamp": datetime.now().isoformat(),
                                "original_filename": doc["filename"]
                            }
                        )
                        
                        # Update document node with audit information
                        session.run(
                            """
                            MATCH (d:Document {id: $doc_id})
                            SET d.source_audit_id = $audit_id,
                                d.migrated_at = datetime()
                            """,
                            doc_id=doc["doc_id"],
                            audit_id=audit_id
                        )
                        
                        migration_stats["successful_migrations"] += 1
                        
                    except Exception as e:
                        error_msg = f"Failed to migrate document {doc['doc_id']}: {str(e)}"
                        migration_stats["errors"].append(error_msg)
                        migration_stats["failed_migrations"] += 1
                        logger.error(error_msg)
                
            logger.info(f"Migration completed: {migration_stats}")
            return migration_stats
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            raise
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """
        Get audit trail statistics and health information.
        
        Returns:
            Dict containing audit system statistics
        """
        if not self.initialized:
            return {"status": "not_initialized"}
            
        try:
            with self.driver.session() as session:
                # Get basic statistics
                result = session.run(
                    """
                    MATCH (f:FileAudit)
                    RETURN 
                        count(f) as total_files,
                        count(CASE WHEN f.file_type = 'source' THEN 1 END) as source_files,
                        count(CASE WHEN f.file_type = 'processed' THEN 1 END) as processed_files
                    """
                )
                
                stats = result.single()
                
                # Get action statistics
                action_result = session.run(
                    """
                    MATCH (a:AuditAction)
                    RETURN a.action as action_type, count(a) as count
                    ORDER BY count DESC
                    LIMIT 10
                    """
                )
                
                action_stats = [dict(record) for record in action_result]
                
                return {
                    "status": "initialized",
                    "total_files": stats["total_files"],
                    "source_files": stats["source_files"],
                    "processed_files": stats["processed_files"],
                    "top_actions": action_stats,
                    "system_health": "operational"
                }
                
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def cleanup_old_files(self, days_old: int = 90) -> Dict[str, Any]:
        """
        Clean up old audit files based on age.
        
        Args:
            days_old: Files older than this many days will be cleaned up
            
        Returns:
            Dict containing cleanup statistics
        """
        if not self.initialized:
            raise RuntimeError("Audit trail system not initialized")
            
        cleanup_stats = {
            "files_removed": 0,
            "space_freed": 0,
            "errors": []
        }
        
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with self.driver.session() as session:
                # Find old files
                result = session.run(
                    """
                    MATCH (f:FileAudit)
                    WHERE f.created_at < datetime($cutoff_date)
                    RETURN f.id as file_id, f.file_path as file_path
                    """,
                    cutoff_date=cutoff_date.isoformat()
                )
                
                old_files = list(result)
                
                for file_record in old_files:
                    try:
                        file_path = Path(file_record["file_path"])
                        if file_path.exists():
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            cleanup_stats["files_removed"] += 1
                            cleanup_stats["space_freed"] += file_size
                            
                        # Update database record
                        session.run(
                            """
                            MATCH (f:FileAudit {id: $file_id})
                            SET f.file_cleaned_up = true,
                                f.cleanup_date = datetime()
                            """,
                            file_id=file_record["file_id"]
                        )
                        
                    except Exception as e:
                        error_msg = f"Failed to cleanup file {file_record['file_path']}: {str(e)}"
                        cleanup_stats["errors"].append(error_msg)
                        logger.error(error_msg)
                        
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
    
    def close(self):
        """Close database connections and cleanup resources."""
        if self.driver:
            self.driver.close()
            logger.info("Audit trail integration closed")


def create_audit_integration(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    auto_initialize: bool = True
) -> AuditTrailIntegration:
    """
    Factory function to create and optionally initialize audit trail integration.
    
    Args:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username  
        neo4j_password: Neo4j password
        auto_initialize: Whether to automatically initialize the system
        
    Returns:
        AuditTrailIntegration: Configured integration instance
    """
    integration = AuditTrailIntegration(neo4j_uri, neo4j_user, neo4j_password)
    
    if auto_initialize:
        success = integration.initialize_audit_trail()
        if not success:
            logger.warning("Auto-initialization failed, manual initialization required")
    
    return integration