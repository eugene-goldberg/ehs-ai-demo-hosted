import os
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

class AuditTrailService:
    """
    Service for managing file storage with UUID-based directory structure.
    Handles original filename preservation, file retrieval, and secure file serving.
    """

    def __init__(self, base_storage_path: str = "/app/storage/"):
        """
        Initialize the audit trail service.
        
        Args:
            base_storage_path: Base path for file storage (Docker volume path)
        """
        self.base_storage_path = Path(base_storage_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure base storage directory exists
        try:
            self.base_storage_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Storage directory initialized at: {self.base_storage_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize storage directory: {str(e)}")
            raise

    async def initialize_schema(self):
        """Initialize audit trail schema (placeholder for database schema if needed)."""
        try:
            # For file-based audit trail, ensure directory structure exists
            self.base_storage_path.mkdir(parents=True, exist_ok=True)
            self.logger.info("Audit trail schema initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize audit trail schema: {str(e)}")
            raise

    async def test_connection(self) -> bool:
        """Test service connectivity and basic functionality."""
        try:
            # Test storage directory accessibility
            if not self.base_storage_path.exists():
                self.logger.error("Audit trail service connection test failed: Storage directory does not exist")
                return False
            
            # Test write permissions by creating and removing a test file
            test_file = self.base_storage_path / f"test_connection_{uuid.uuid4().hex[:8]}.tmp"
            try:
                # Test write access
                test_file.write_text("connection test", encoding='utf-8')
                
                # Test read access
                content = test_file.read_text(encoding='utf-8')
                if content != "connection test":
                    self.logger.error("Audit trail service connection test failed: Read/write test failed")
                    return False
                
                # Clean up test file
                test_file.unlink()
                
                self.logger.debug("Audit trail service storage directory access test successful")
                return True
                
            except Exception as e:
                # Clean up test file if it exists
                if test_file.exists():
                    try:
                        test_file.unlink()
                    except:
                        pass
                self.logger.error(f"Audit trail service connection test failed: Storage access error: {str(e)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Audit trail service connection test failed: {str(e)}")
            return False

    def store_source_file(self, uploaded_file_path: str, original_filename: str, document_id: str = None) -> Tuple[str, str]:
        """
        Store uploaded file with UUID-based directory structure while preserving original filename.
        
        Args:
            uploaded_file_path: Path to the uploaded file
            original_filename: Original name of the file
            document_id: Optional document ID, will generate UUID if not provided
            
        Returns:
            Tuple of (document_id, stored_file_path)
            
        Raises:
            Exception: If file storage fails
        """
        try:
            # Generate document ID if not provided
            if document_id is None:
                document_id = str(uuid.uuid4())
            
            # Create UUID-based directory structure
            uuid_dir = self.base_storage_path / document_id
            uuid_dir.mkdir(parents=True, exist_ok=True)
            
            # Preserve original filename in storage
            stored_file_path = uuid_dir / original_filename
            
            # Copy file to storage location
            shutil.copy2(uploaded_file_path, stored_file_path)
            
            # Verify file was stored successfully
            if not stored_file_path.exists():
                raise Exception(f"File verification failed: {stored_file_path}")
                
            self.logger.info(f"File stored successfully: {document_id} -> {stored_file_path}")
            
            return document_id, str(stored_file_path)
            
        except Exception as e:
            error_message = f"Error storing source file {original_filename}: {str(e)}"
            self.logger.error(error_message)
            raise Exception(error_message)

    def get_source_file_path(self, document_id: str, original_filename: str = None) -> Optional[str]:
        """
        Retrieve the file path for a document by its ID.
        
        Args:
            document_id: UUID of the document
            original_filename: Original filename (optional, for verification)
            
        Returns:
            Full path to the stored file, or None if not found
        """
        try:
            uuid_dir = self.base_storage_path / document_id
            
            if not uuid_dir.exists():
                self.logger.warning(f"Document directory not found: {document_id}")
                return None
            
            # If original filename is provided, check for that specific file
            if original_filename:
                file_path = uuid_dir / original_filename
                if file_path.exists():
                    return str(file_path)
                else:
                    self.logger.warning(f"Specific file not found: {file_path}")
                    return None
            
            # Otherwise, find the first file in the directory
            files = list(uuid_dir.glob("*"))
            if files:
                file_path = files[0]  # Return first file found
                self.logger.info(f"File found for document {document_id}: {file_path}")
                return str(file_path)
            else:
                self.logger.warning(f"No files found in document directory: {document_id}")
                return None
                
        except Exception as e:
            error_message = f"Error retrieving file path for document {document_id}: {str(e)}"
            self.logger.error(error_message)
            return None

    def serve_source_file(self, document_id: str, original_filename: str = None) -> Optional[Tuple[str, str, str]]:
        """
        Prepare file for secure serving/download.
        
        Args:
            document_id: UUID of the document
            original_filename: Original filename (optional)
            
        Returns:
            Tuple of (file_path, filename_for_download, content_type) or None if not found
        """
        try:
            file_path = self.get_source_file_path(document_id, original_filename)
            
            if not file_path:
                return None
                
            file_path_obj = Path(file_path)
            
            # Determine filename for download (use original or actual filename)
            if original_filename:
                download_filename = original_filename
            else:
                download_filename = file_path_obj.name
            
            # Determine content type based on file extension
            content_type = self._get_content_type(file_path_obj.suffix.lower())
            
            # Verify file still exists and is readable
            if not file_path_obj.exists() or not os.access(file_path_obj, os.R_OK):
                self.logger.error(f"File not accessible: {file_path}")
                return None
            
            self.logger.info(f"File prepared for serving: {document_id} -> {download_filename}")
            
            return str(file_path), download_filename, content_type
            
        except Exception as e:
            error_message = f"Error preparing file for serving {document_id}: {str(e)}"
            self.logger.error(error_message)
            return None

    def delete_source_file(self, document_id: str) -> bool:
        """
        Remove file and directory when document is deleted.
        
        Args:
            document_id: UUID of the document to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            uuid_dir = self.base_storage_path / document_id
            
            if not uuid_dir.exists():
                self.logger.warning(f"Document directory not found for deletion: {document_id}")
                return True  # Consider it successful if already gone
            
            # Remove entire directory and its contents
            shutil.rmtree(uuid_dir)
            
            # Verify deletion
            if uuid_dir.exists():
                self.logger.error(f"Failed to delete document directory: {document_id}")
                return False
            
            self.logger.info(f"Document files deleted successfully: {document_id}")
            return True
            
        except Exception as e:
            error_message = f"Error deleting source file for document {document_id}: {str(e)}"
            self.logger.error(error_message)
            return False

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics for monitoring and maintenance.
        
        Returns:
            Dictionary containing storage statistics
        """
        try:
            total_documents = 0
            total_files = 0
            total_size = 0
            
            if self.base_storage_path.exists():
                for doc_dir in self.base_storage_path.iterdir():
                    if doc_dir.is_dir():
                        total_documents += 1
                        for file_path in doc_dir.rglob("*"):
                            if file_path.is_file():
                                total_files += 1
                                total_size += file_path.stat().st_size
            
            stats = {
                "total_documents": total_documents,
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "storage_path": str(self.base_storage_path),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Storage stats generated: {stats}")
            return stats
            
        except Exception as e:
            error_message = f"Error generating storage statistics: {str(e)}"
            self.logger.error(error_message)
            return {"error": error_message}

    def validate_document_integrity(self, document_id: str) -> dict:
        """
        Validate that a document's files are intact and accessible.
        
        Args:
            document_id: UUID of the document to validate
            
        Returns:
            Dictionary containing validation results
        """
        try:
            uuid_dir = self.base_storage_path / document_id
            
            result = {
                "document_id": document_id,
                "directory_exists": uuid_dir.exists(),
                "files_found": [],
                "total_files": 0,
                "total_size": 0,
                "is_valid": False,
                "timestamp": datetime.now().isoformat()
            }
            
            if not uuid_dir.exists():
                result["error"] = "Document directory not found"
                return result
            
            for file_path in uuid_dir.rglob("*"):
                if file_path.is_file():
                    file_stat = file_path.stat()
                    file_info = {
                        "filename": file_path.name,
                        "size": file_stat.st_size,
                        "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        "readable": os.access(file_path, os.R_OK)
                    }
                    result["files_found"].append(file_info)
                    result["total_files"] += 1
                    result["total_size"] += file_stat.st_size
            
            # Document is valid if directory exists and has at least one readable file
            result["is_valid"] = result["total_files"] > 0 and all(
                f["readable"] for f in result["files_found"]
            )
            
            return result
            
        except Exception as e:
            error_message = f"Error validating document integrity for {document_id}: {str(e)}"
            self.logger.error(error_message)
            return {
                "document_id": document_id,
                "error": error_message,
                "is_valid": False,
                "timestamp": datetime.now().isoformat()
            }

    def _get_content_type(self, file_extension: str) -> str:
        """
        Get MIME content type based on file extension.
        
        Args:
            file_extension: File extension (with dot)
            
        Returns:
            MIME content type string
        """
        content_types = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.xml': 'application/xml',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.zip': 'application/zip',
            '.rar': 'application/x-rar-compressed',
            '.7z': 'application/x-7z-compressed'
        }
        
        return content_types.get(file_extension, 'application/octet-stream')

    def cleanup_orphaned_files(self, valid_document_ids: list) -> dict:
        """
        Clean up files for documents that no longer exist in the database.
        
        Args:
            valid_document_ids: List of document IDs that should be preserved
            
        Returns:
            Dictionary containing cleanup results
        """
        try:
            cleanup_result = {
                "directories_checked": 0,
                "directories_removed": 0,
                "files_removed": 0,
                "space_freed_bytes": 0,
                "errors": [],
                "timestamp": datetime.now().isoformat()
            }
            
            if not self.base_storage_path.exists():
                return cleanup_result
            
            for doc_dir in self.base_storage_path.iterdir():
                if doc_dir.is_dir():
                    cleanup_result["directories_checked"] += 1
                    doc_id = doc_dir.name
                    
                    # If this document ID is not in the valid list, remove it
                    if doc_id not in valid_document_ids:
                        try:
                            # Calculate space to be freed
                            for file_path in doc_dir.rglob("*"):
                                if file_path.is_file():
                                    cleanup_result["space_freed_bytes"] += file_path.stat().st_size
                                    cleanup_result["files_removed"] += 1
                            
                            # Remove the directory
                            shutil.rmtree(doc_dir)
                            cleanup_result["directories_removed"] += 1
                            
                            self.logger.info(f"Removed orphaned document directory: {doc_id}")
                            
                        except Exception as e:
                            error_msg = f"Failed to remove orphaned directory {doc_id}: {str(e)}"
                            cleanup_result["errors"].append(error_msg)
                            self.logger.error(error_msg)
            
            cleanup_result["space_freed_mb"] = round(
                cleanup_result["space_freed_bytes"] / (1024 * 1024), 2
            )
            
            self.logger.info(f"Cleanup completed: {cleanup_result}")
            return cleanup_result
            
        except Exception as e:
            error_message = f"Error during orphaned files cleanup: {str(e)}"
            self.logger.error(error_message)
            return {
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }