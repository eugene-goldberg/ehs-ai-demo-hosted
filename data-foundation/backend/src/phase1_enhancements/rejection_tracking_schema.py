import logging
import os
from datetime import datetime
from enum import Enum
from neo4j.exceptions import TransientError
from langchain_neo4j import Neo4jGraph
import uuid

class DocumentStatus(Enum):
    """Document processing status enumeration"""
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    REJECTED = "REJECTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"

class RejectionReason(Enum):
    """Standardized rejection reason codes"""
    INVALID_FORMAT = "INVALID_FORMAT"
    NOT_RELEVANT = "NOT_RELEVANT"
    DUPLICATE = "DUPLICATE"
    POOR_QUALITY = "POOR_QUALITY"
    INCOMPLETE_DATA = "INCOMPLETE_DATA"
    TECHNICAL_ERROR = "TECHNICAL_ERROR"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    SIZE_EXCEEDED = "SIZE_EXCEEDED"
    OTHER = "OTHER"

class RejectionTrackingSchema:
    """
    Schema management class for rejection tracking enhancements to ProcessedDocument nodes.
    Adds status tracking, rejection reasons, and audit trail for document processing workflow.
    Creates relationships to User nodes for rejected documents.
    """

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def create_constraints_and_indexes(self):
        """
        Create constraints and indexes for rejection tracking properties.
        """
        try:
            logging.info("Creating constraints and indexes for rejection tracking schema")
            
            # Create index for document status for efficient querying
            status_index_query = """
                CREATE INDEX idx_document_status IF NOT EXISTS
                FOR (d:Document) ON (d.status)
            """
            self.graph.query(status_index_query, session_params={"database": self.graph._database})
            
            # Create index for rejection reason for analytics
            rejection_reason_index_query = """
                CREATE INDEX idx_rejection_reason IF NOT EXISTS
                FOR (d:Document) ON (d.rejection_reason)
            """
            self.graph.query(rejection_reason_index_query, session_params={"database": self.graph._database})
            
            # Create index for rejected_at timestamp for time-based queries
            rejected_at_index_query = """
                CREATE INDEX idx_rejected_at IF NOT EXISTS
                FOR (d:Document) ON (d.rejected_at)
            """
            self.graph.query(rejected_at_index_query, session_params={"database": self.graph._database})
            
            # Create index for rejected_by_user_id for user-based queries
            rejected_by_index_query = """
                CREATE INDEX idx_rejected_by_user_id IF NOT EXISTS
                FOR (d:Document) ON (d.rejected_by_user_id)
            """
            self.graph.query(rejected_by_index_query, session_params={"database": self.graph._database})
            
            # Create constraint for User nodes if it doesn't exist
            user_constraint_query = """
                CREATE CONSTRAINT unique_user_id IF NOT EXISTS
                FOR (u:User) REQUIRE u.user_id IS UNIQUE
            """
            self.graph.query(user_constraint_query, session_params={"database": self.graph._database})
            
            logging.info("Constraints and indexes created successfully")
            
        except Exception as e:
            error_message = f"Error creating constraints and indexes: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def add_rejection_properties_to_documents(self):
        """
        Add rejection tracking properties to existing ProcessedDocument nodes.
        Sets default status to PROCESSING for existing documents.
        """
        try:
            logging.info("Adding rejection tracking properties to existing ProcessedDocument nodes")
            
            # Query to add properties to existing documents that don't have them
            migration_query = """
                MATCH (d:Document)
                WHERE d.status IS NULL
                SET d.status = $default_status,
                    d.rejection_reason = null,
                    d.rejection_notes = null,
                    d.rejected_at = null,
                    d.rejected_by_user_id = null,
                    d.rejection_migration_date = $migration_date
                RETURN count(d) as updated_count
            """
            
            params = {
                "default_status": DocumentStatus.PROCESSING.value,
                "migration_date": datetime.now().isoformat()
            }
            result = self.graph.query(migration_query, params, session_params={"database": self.graph._database})
            
            updated_count = result[0]['updated_count'] if result else 0
            logging.info(f"Updated {updated_count} documents with rejection tracking properties")
            
            return updated_count
            
        except Exception as e:
            error_message = f"Error adding rejection properties to documents: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def create_rejection_schema(self):
        """
        Create the complete rejection tracking schema including constraints, indexes, and property migration.
        """
        try:
            logging.info("Creating complete rejection tracking schema")
            
            # Step 1: Create constraints and indexes
            self.create_constraints_and_indexes()
            
            # Step 2: Migrate existing documents
            updated_count = self.add_rejection_properties_to_documents()
            
            # Step 3: Verify the schema changes
            validation_result = self.validate_rejection_schema()
            
            result = {
                "constraints_created": True,
                "indexes_created": True,
                "documents_migrated": updated_count,
                "schema_validation_passed": validation_result
            }
            
            logging.info(f"Rejection tracking schema creation completed: {result}")
            return result
            
        except Exception as e:
            error_message = f"Error creating rejection tracking schema: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def validate_rejection_schema(self):
        """
        Validate that the rejection tracking schema is properly created.
        Returns True if validation passes, False otherwise.
        """
        try:
            logging.info("Validating rejection tracking schema")
            
            # Check if indexes exist
            index_check_query = """
                SHOW INDEXES
                YIELD name, type, entityType, labelsOrTypes, properties
                WHERE name CONTAINS 'rejection' OR name CONTAINS 'status' OR name CONTAINS 'rejected'
                RETURN count(*) as index_count
            """
            
            index_result = self.graph.query(index_check_query, session_params={"database": self.graph._database})
            index_count = index_result[0]['index_count'] if index_result else 0
            
            # Check if documents have status property
            status_check_query = """
                MATCH (d:Document)
                WITH count(d) as total_docs,
                     count(CASE WHEN d.status IS NOT NULL THEN 1 END) as with_status
                RETURN total_docs, with_status,
                       CASE WHEN total_docs > 0 AND total_docs = with_status 
                            THEN true 
                            ELSE false 
                       END as status_validation_passed
            """
            
            status_result = self.graph.query(status_check_query, session_params={"database": self.graph._database})
            status_validation = status_result[0]['status_validation_passed'] if status_result else False
            
            # Schema is valid if we have the required indexes and status properties
            validation_passed = index_count >= 4 and status_validation
            
            logging.info(f"Schema validation - Indexes: {index_count}, Status validation: {status_validation}, Passed: {validation_passed}")
            return validation_passed
                
        except Exception as e:
            error_message = f"Error validating rejection tracking schema: {str(e)}"
            logging.error(error_message)
            return False

    def update_document_status(self, document_file_name, status):
        """
        Update the status of a document.
        
        Args:
            document_file_name: The fileName of the ProcessedDocument
            status: New status (DocumentStatus enum value)
            
        Returns:
            Boolean indicating success
        """
        try:
            logging.info(f"Updating status for document: {document_file_name} to {status}")
            
            # Validate status
            if isinstance(status, str):
                try:
                    status = DocumentStatus(status)
                except ValueError:
                    raise ValueError(f"Invalid status: {status}. Must be one of {[s.value for s in DocumentStatus]}")
            elif not isinstance(status, DocumentStatus):
                raise ValueError(f"Status must be DocumentStatus enum or valid string")
            
            query = """
                MATCH (d:Document {fileName: $document_file_name})
                SET d.status = $status,
                    d.updatedAt = $updated_at
                RETURN d.fileName as updated_document
            """
            
            params = {
                "document_file_name": document_file_name,
                "status": status.value,
                "updated_at": datetime.now().isoformat()
            }
            
            result = self.graph.query(query, params, session_params={"database": self.graph._database})
            
            if result:
                logging.info(f"Successfully updated status for document: {document_file_name}")
                return True
            else:
                logging.warning(f"Document not found: {document_file_name}")
                return False
                
        except Exception as e:
            error_message = f"Error updating document status for {document_file_name}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def reject_document(self, document_file_name, rejection_reason, rejection_notes=None, rejected_by_user_id=None):
        """
        Mark a document as rejected with detailed information and create relationship to User.
        
        Args:
            document_file_name: The fileName of the ProcessedDocument
            rejection_reason: Reason for rejection (RejectionReason enum value)
            rejection_notes: Optional free text notes about the rejection
            rejected_by_user_id: Optional ID of the user who rejected the document
            
        Returns:
            Dictionary containing rejection information
        """
        try:
            logging.info(f"Rejecting document: {document_file_name} with reason: {rejection_reason}")
            
            # Validate rejection reason
            if isinstance(rejection_reason, str):
                try:
                    rejection_reason = RejectionReason(rejection_reason)
                except ValueError:
                    raise ValueError(f"Invalid rejection reason: {rejection_reason}. Must be one of {[r.value for r in RejectionReason]}")
            elif not isinstance(rejection_reason, RejectionReason):
                raise ValueError(f"Rejection reason must be RejectionReason enum or valid string")
            
            current_time = datetime.now().isoformat()
            rejection_id = str(uuid.uuid4())
            
            # Base query to reject the document
            base_query = """
                MATCH (d:Document {fileName: $document_file_name})
                SET d.status = $rejected_status,
                    d.rejection_reason = $rejection_reason,
                    d.rejection_notes = $rejection_notes,
                    d.rejected_at = $rejected_at,
                    d.rejected_by_user_id = $rejected_by_user_id,
                    d.rejection_id = $rejection_id,
                    d.updatedAt = $updated_at
            """
            
            params = {
                "document_file_name": document_file_name,
                "rejected_status": DocumentStatus.REJECTED.value,
                "rejection_reason": rejection_reason.value,
                "rejection_notes": rejection_notes,
                "rejected_at": current_time,
                "rejected_by_user_id": rejected_by_user_id,
                "rejection_id": rejection_id,
                "updated_at": current_time
            }
            
            # If user ID is provided, create/update User node and REJECTED relationship
            if rejected_by_user_id:
                query = base_query + """
                    WITH d
                    MERGE (u:User {user_id: $rejected_by_user_id})
                    ON CREATE SET u.created_at = $rejected_at
                    ON MATCH SET u.last_activity_at = $rejected_at
                    MERGE (u)-[r:REJECTED]->(d)
                    ON CREATE SET r.rejected_at = $rejected_at,
                                  r.rejection_reason = $rejection_reason,
                                  r.rejection_id = $rejection_id
                    RETURN d.fileName as rejected_document,
                           d.rejection_id as rejection_id,
                           d.rejected_at as rejected_at,
                           u.user_id as rejected_by
                """
            else:
                query = base_query + """
                    RETURN d.fileName as rejected_document,
                           d.rejection_id as rejection_id,
                           d.rejected_at as rejected_at,
                           null as rejected_by
                """
            
            result = self.graph.query(query, params, session_params={"database": self.graph._database})
            
            if result:
                rejection_info = result[0]
                logging.info(f"Successfully rejected document: {document_file_name} with ID: {rejection_id}")
                return rejection_info
            else:
                raise Exception(f"Failed to reject document - document may not exist: {document_file_name}")
                
        except Exception as e:
            error_message = f"Error rejecting document {document_file_name}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_rejected_documents(self, limit=None, offset=None):
        """
        Retrieve all rejected documents with their rejection details.
        
        Args:
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            
        Returns:
            List of rejected document dictionaries
        """
        try:
            query = """
                MATCH (d:Document {status: $rejected_status})
                OPTIONAL MATCH (u:User {user_id: d.rejected_by_user_id})
                RETURN d.fileName as document_file_name,
                       d.rejection_reason as rejection_reason,
                       d.rejection_notes as rejection_notes,
                       d.rejected_at as rejected_at,
                       d.rejected_by_user_id as rejected_by_user_id,
                       d.rejection_id as rejection_id,
                       u.user_id as rejector_user_id,
                       d.createdAt as original_created_at,
                       d.fileSource as file_source
                ORDER BY d.rejected_at DESC
            """
            
            if limit is not None:
                query += f" LIMIT {limit}"
            if offset is not None:
                query += f" SKIP {offset}"
            
            param = {"rejected_status": DocumentStatus.REJECTED.value}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            return result
            
        except Exception as e:
            error_message = f"Error retrieving rejected documents: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_documents_by_status(self, status, limit=None, offset=None):
        """
        Retrieve documents filtered by status.
        
        Args:
            status: Document status to filter by (DocumentStatus enum value)
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            
        Returns:
            List of document dictionaries
        """
        try:
            # Validate status
            if isinstance(status, str):
                try:
                    status = DocumentStatus(status)
                except ValueError:
                    raise ValueError(f"Invalid status: {status}. Must be one of {[s.value for s in DocumentStatus]}")
            elif not isinstance(status, DocumentStatus):
                raise ValueError(f"Status must be DocumentStatus enum or valid string")
            
            query = """
                MATCH (d:Document {status: $status})
                RETURN d.fileName as document_file_name,
                       d.status as status,
                       d.rejection_reason as rejection_reason,
                       d.rejection_notes as rejection_notes,
                       d.rejected_at as rejected_at,
                       d.rejected_by_user_id as rejected_by_user_id,
                       d.createdAt as created_at,
                       d.updatedAt as updated_at,
                       d.fileSource as file_source
                ORDER BY d.updatedAt DESC
            """
            
            if limit is not None:
                query += f" LIMIT {limit}"
            if offset is not None:
                query += f" SKIP {offset}"
            
            param = {"status": status.value}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            return result
            
        except Exception as e:
            error_message = f"Error retrieving documents by status {status}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_rejection_statistics(self):
        """
        Get aggregated statistics about rejected documents.
        
        Returns:
            Dictionary containing rejection statistics
        """
        try:
            query = """
                MATCH (d:Document)
                WITH count(d) as total_documents,
                     count(CASE WHEN d.status = $rejected_status THEN 1 END) as rejected_count,
                     count(CASE WHEN d.status = $processed_status THEN 1 END) as processed_count,
                     count(CASE WHEN d.status = $processing_status THEN 1 END) as processing_count,
                     count(CASE WHEN d.status = $review_required_status THEN 1 END) as review_required_count
                
                OPTIONAL MATCH (rejected:Document {status: $rejected_status})
                WITH total_documents, rejected_count, processed_count, processing_count, review_required_count,
                     collect(rejected.rejection_reason) as rejection_reasons
                
                UNWIND rejection_reasons as reason
                WITH total_documents, rejected_count, processed_count, processing_count, review_required_count,
                     reason, count(reason) as reason_count
                
                RETURN total_documents,
                       rejected_count,
                       processed_count,
                       processing_count,
                       review_required_count,
                       CASE WHEN total_documents > 0 
                            THEN round(rejected_count * 100.0 / total_documents, 2)
                            ELSE 0 
                       END as rejection_rate_percentage,
                       collect({reason: reason, count: reason_count}) as rejection_reasons_breakdown
            """
            
            params = {
                "rejected_status": DocumentStatus.REJECTED.value,
                "processed_status": DocumentStatus.PROCESSED.value,
                "processing_status": DocumentStatus.PROCESSING.value,
                "review_required_status": DocumentStatus.REVIEW_REQUIRED.value
            }
            
            result = self.graph.query(query, params, session_params={"database": self.graph._database})
            
            if result:
                return result[0]
            else:
                return {
                    "total_documents": 0,
                    "rejected_count": 0,
                    "processed_count": 0,
                    "processing_count": 0,
                    "review_required_count": 0,
                    "rejection_rate_percentage": 0,
                    "rejection_reasons_breakdown": []
                }
                
        except Exception as e:
            error_message = f"Error getting rejection statistics: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_user_rejection_history(self, user_id, limit=None):
        """
        Get rejection history for a specific user.
        
        Args:
            user_id: ID of the user to get rejection history for
            limit: Optional limit on number of results
            
        Returns:
            List of documents rejected by the user
        """
        try:
            query = """
                MATCH (u:User {user_id: $user_id})-[r:REJECTED]->(d:Document)
                RETURN d.fileName as document_file_name,
                       d.rejection_reason as rejection_reason,
                       d.rejection_notes as rejection_notes,
                       r.rejected_at as rejected_at,
                       r.rejection_id as rejection_id,
                       d.fileSource as file_source
                ORDER BY r.rejected_at DESC
            """
            
            if limit is not None:
                query += f" LIMIT {limit}"
            
            param = {"user_id": user_id}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            return result
            
        except Exception as e:
            error_message = f"Error getting rejection history for user {user_id}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def unreject_document(self, document_file_name, new_status=DocumentStatus.PROCESSING):
        """
        Remove rejection status from a document and reset to a new status.
        
        Args:
            document_file_name: The fileName of the ProcessedDocument
            new_status: New status to set (default: PROCESSING)
            
        Returns:
            Boolean indicating success
        """
        try:
            logging.info(f"Un-rejecting document: {document_file_name}")
            
            # Validate new status
            if isinstance(new_status, str):
                try:
                    new_status = DocumentStatus(new_status)
                except ValueError:
                    raise ValueError(f"Invalid status: {new_status}. Must be one of {[s.value for s in DocumentStatus]}")
            elif not isinstance(new_status, DocumentStatus):
                raise ValueError(f"Status must be DocumentStatus enum or valid string")
            
            query = """
                MATCH (d:Document {fileName: $document_file_name})
                WHERE d.status = $rejected_status
                SET d.status = $new_status,
                    d.rejection_reason = null,
                    d.rejection_notes = null,
                    d.rejected_at = null,
                    d.rejected_by_user_id = null,
                    d.rejection_id = null,
                    d.updatedAt = $updated_at
                WITH d
                OPTIONAL MATCH (u:User)-[r:REJECTED]->(d)
                DELETE r
                RETURN d.fileName as unrejected_document
            """
            
            params = {
                "document_file_name": document_file_name,
                "rejected_status": DocumentStatus.REJECTED.value,
                "new_status": new_status.value,
                "updated_at": datetime.now().isoformat()
            }
            
            result = self.graph.query(query, params, session_params={"database": self.graph._database})
            
            if result:
                logging.info(f"Successfully un-rejected document: {document_file_name}")
                return True
            else:
                logging.warning(f"Rejected document not found: {document_file_name}")
                return False
                
        except Exception as e:
            error_message = f"Error un-rejecting document {document_file_name}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def validate_document_workflow(self, document_file_name):
        """
        Validate the workflow state of a document for audit purposes.
        
        Args:
            document_file_name: The fileName of the ProcessedDocument
            
        Returns:
            Dictionary containing workflow validation information
        """
        try:
            query = """
                MATCH (d:Document {fileName: $document_file_name})
                OPTIONAL MATCH (u:User)-[r:REJECTED]->(d)
                RETURN d.fileName as document_file_name,
                       d.status as current_status,
                       d.rejection_reason as rejection_reason,
                       d.rejected_at as rejected_at,
                       d.rejected_by_user_id as rejected_by_user_id,
                       d.createdAt as created_at,
                       d.updatedAt as updated_at,
                       count(r) as rejection_relationships_count,
                       collect(u.user_id) as rejecting_users
            """
            
            param = {"document_file_name": document_file_name}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            if result:
                validation_info = result[0]
                
                # Add validation checks
                validation_info['is_valid_workflow'] = True
                validation_info['validation_issues'] = []
                
                # Check for inconsistencies
                if validation_info['current_status'] == DocumentStatus.REJECTED.value:
                    if not validation_info['rejection_reason']:
                        validation_info['is_valid_workflow'] = False
                        validation_info['validation_issues'].append("Rejected document missing rejection reason")
                    
                    if not validation_info['rejected_at']:
                        validation_info['is_valid_workflow'] = False
                        validation_info['validation_issues'].append("Rejected document missing rejection timestamp")
                
                elif validation_info['rejection_reason']:
                    validation_info['is_valid_workflow'] = False
                    validation_info['validation_issues'].append("Non-rejected document has rejection reason")
                
                return validation_info
            else:
                return {"error": f"Document not found: {document_file_name}"}
                
        except Exception as e:
            error_message = f"Error validating workflow for document {document_file_name}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)
# Stub classes for compatibility
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class RejectionEntry(BaseModel):
    document_id: str
    reason: str
    timestamp: datetime
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ValidationRule(BaseModel):
    rule_name: str
    rule_type: str
    criteria: Dict[str, Any]
    enabled: bool = True
