import logging
import os
from datetime import datetime
from neo4j.exceptions import TransientError
from langchain_neo4j import Neo4jGraph

class AuditTrailSchema:
    """
    Schema management class for audit trail enhancements to ProcessedDocument nodes.
    Adds original_filename and source_file_path properties to track document provenance.
    """

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def create_constraints_and_indexes(self):
        """
        Create constraints and indexes for the new audit trail properties.
        """
        try:
            logging.info("Creating constraints and indexes for audit trail properties")
            
            # Create constraint for original_filename if it doesn't exist
            constraint_query = """
                CREATE CONSTRAINT unique_original_filename IF NOT EXISTS
                FOR (d:Document) REQUIRE d.original_filename IS UNIQUE
            """
            self.graph.query(constraint_query, session_params={"database": self.graph._database})
            
            # Create index for source_file_path for better query performance
            index_query = """
                CREATE INDEX idx_source_file_path IF NOT EXISTS
                FOR (d:Document) ON (d.source_file_path)
            """
            self.graph.query(index_query, session_params={"database": self.graph._database})
            
            logging.info("Constraints and indexes created successfully")
            
        except Exception as e:
            error_message = f"Error creating constraints and indexes: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def add_audit_properties_to_documents(self):
        """
        Add original_filename and source_file_path properties to existing ProcessedDocument nodes.
        This method migrates existing documents to include these properties.
        """
        try:
            logging.info("Adding audit trail properties to existing ProcessedDocument nodes")
            
            # Query to add properties to existing documents that don't have them
            migration_query = """
                MATCH (d:Document)
                WHERE d.original_filename IS NULL OR d.source_file_path IS NULL
                SET d.original_filename = COALESCE(d.original_filename, d.fileName),
                    d.source_file_path = COALESCE(d.source_file_path, 
                        CASE 
                            WHEN d.fileSource = 'local file' THEN '/local/uploads/' + d.fileName
                            WHEN d.fileSource = 'web-url' THEN d.url
                            WHEN d.fileSource = 'gcs' THEN d.gcsBucket + '/' + d.gcsBucketFolder + '/' + d.fileName
                            ELSE 'unknown'
                        END
                    ),
                    d.audit_migration_date = $migration_date
                RETURN count(d) as updated_count
            """
            
            param = {"migration_date": datetime.now().isoformat()}
            result = self.graph.query(migration_query, param, session_params={"database": self.graph._database})
            
            updated_count = result[0]['updated_count'] if result else 0
            logging.info(f"Updated {updated_count} documents with audit trail properties")
            
            return updated_count
            
        except Exception as e:
            error_message = f"Error adding audit properties to documents: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def create_audit_trail_schema(self):
        """
        Create the complete audit trail schema including constraints, indexes, and property migration.
        """
        try:
            logging.info("Creating complete audit trail schema")
            
            # Step 1: Create constraints and indexes
            self.create_constraints_and_indexes()
            
            # Step 2: Migrate existing documents
            updated_count = self.add_audit_properties_to_documents()
            
            # Step 3: Verify the schema changes
            validation_result = self.validate_audit_properties()
            
            result = {
                "constraints_created": True,
                "indexes_created": True,
                "documents_migrated": updated_count,
                "validation_passed": validation_result
            }
            
            logging.info(f"Audit trail schema creation completed: {result}")
            return result
            
        except Exception as e:
            error_message = f"Error creating audit trail schema: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def validate_audit_properties(self):
        """
        Validate that the audit trail properties exist and are properly populated.
        Returns True if validation passes, False otherwise.
        """
        try:
            logging.info("Validating audit trail properties")
            
            # Check if all documents have the required properties
            validation_query = """
                MATCH (d:Document)
                WITH count(d) as total_docs,
                     count(CASE WHEN d.original_filename IS NOT NULL THEN 1 END) as with_original,
                     count(CASE WHEN d.source_file_path IS NOT NULL THEN 1 END) as with_source_path
                RETURN total_docs, with_original, with_source_path,
                       CASE WHEN total_docs = with_original AND total_docs = with_source_path 
                            THEN true 
                            ELSE false 
                       END as validation_passed
            """
            
            result = self.graph.query(validation_query, session_params={"database": self.graph._database})
            
            if result:
                validation_data = result[0]
                logging.info(f"Validation results: {validation_data}")
                return validation_data['validation_passed']
            else:
                logging.warning("No validation results returned")
                return False
                
        except Exception as e:
            error_message = f"Error validating audit properties: {str(e)}"
            logging.error(error_message)
            return False

    def get_audit_trail_info(self, file_name):
        """
        Retrieve audit trail information for a specific document.
        
        Args:
            file_name: The name of the file to get audit info for
            
        Returns:
            Dictionary containing audit trail information
        """
        try:
            query = """
                MATCH (d:Document {fileName: $file_name})
                RETURN d.fileName as current_filename,
                       d.original_filename as original_filename,
                       d.source_file_path as source_file_path,
                       d.fileSource as file_source,
                       d.createdAt as created_at,
                       d.updatedAt as updated_at,
                       d.audit_migration_date as migration_date
            """
            
            param = {"file_name": file_name}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            if result:
                return result[0]
            else:
                logging.warning(f"No audit trail information found for file: {file_name}")
                return None
                
        except Exception as e:
            error_message = f"Error retrieving audit trail info for {file_name}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def update_document_audit_properties(self, file_name, original_filename=None, source_file_path=None):
        """
        Update audit trail properties for a specific document.
        
        Args:
            file_name: The current name of the file
            original_filename: The original filename (optional)
            source_file_path: The source file path (optional)
        """
        try:
            logging.info(f"Updating audit properties for document: {file_name}")
            
            # Build dynamic SET clause based on provided parameters
            set_clauses = []
            params = {"file_name": file_name, "updated_at": datetime.now().isoformat()}
            
            if original_filename is not None:
                set_clauses.append("d.original_filename = $original_filename")
                params["original_filename"] = original_filename
                
            if source_file_path is not None:
                set_clauses.append("d.source_file_path = $source_file_path")
                params["source_file_path"] = source_file_path
            
            if not set_clauses:
                logging.warning("No properties to update")
                return
                
            set_clauses.append("d.updatedAt = $updated_at")
            
            query = f"""
                MATCH (d:Document {{fileName: $file_name}})
                SET {', '.join(set_clauses)}
                RETURN d.fileName as updated_file
            """
            
            result = self.graph.query(query, params, session_params={"database": self.graph._database})
            
            if result:
                logging.info(f"Successfully updated audit properties for: {file_name}")
            else:
                logging.warning(f"Document not found: {file_name}")
                
        except Exception as e:
            error_message = f"Error updating audit properties for {file_name}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_documents_by_original_filename(self, original_filename):
        """
        Find all documents with a specific original filename.
        This can help identify potential duplicates or renamed files.
        
        Args:
            original_filename: The original filename to search for
            
        Returns:
            List of documents matching the original filename
        """
        try:
            query = """
                MATCH (d:Document {original_filename: $original_filename})
                RETURN d.fileName as current_filename,
                       d.original_filename as original_filename,
                       d.source_file_path as source_file_path,
                       d.fileSource as file_source,
                       d.status as status,
                       d.createdAt as created_at
                ORDER BY d.createdAt DESC
            """
            
            param = {"original_filename": original_filename}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            return result
            
        except Exception as e:
            error_message = f"Error finding documents by original filename {original_filename}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_documents_by_source_path(self, source_path_pattern):
        """
        Find all documents with source paths matching a pattern.
        
        Args:
            source_path_pattern: Pattern to match against source_file_path
            
        Returns:
            List of documents matching the source path pattern
        """
        try:
            query = """
                MATCH (d:Document)
                WHERE d.source_file_path CONTAINS $source_path_pattern
                RETURN d.fileName as current_filename,
                       d.original_filename as original_filename,
                       d.source_file_path as source_file_path,
                       d.fileSource as file_source,
                       d.status as status,
                       d.createdAt as created_at
                ORDER BY d.createdAt DESC
            """
            
            param = {"source_path_pattern": source_path_pattern}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            return result
            
        except Exception as e:
            error_message = f"Error finding documents by source path pattern {source_path_pattern}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)
# Stub classes for compatibility
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class AuditTrailEntry(BaseModel):
    document_id: str
    action: str
    timestamp: datetime
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentInfo(BaseModel):
    document_id: str
    title: Optional[str] = None
    status: Optional[str] = None
