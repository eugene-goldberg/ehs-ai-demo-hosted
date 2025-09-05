#!/usr/bin/env python3

"""
Script to delete exactly 2 documents with document_type = None (Unknown) from Neo4j
while preserving Water Bill, Electric Bill, and Waste Manifest documents.
"""

import os
import sys
from neo4j import GraphDatabase
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/delete_unknown_docs_v2.log'),
        logging.StreamHandler()
    ]
)

class Neo4jManager:
    def __init__(self, uri, username, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
    def close(self):
        if self.driver:
            self.driver.close()
    
    def query_all_documents(self):
        """Query all documents in the database"""
        query = """
        MATCH (d:Document)
        RETURN d.id as id, 
               d.fileName as fileName, 
               d.document_type as document_type
        ORDER BY d.id
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                documents = []
                for record in result:
                    doc = {
                        'id': record['id'],
                        'fileName': record['fileName'],
                        'document_type': record['document_type']
                    }
                    documents.append(doc)
                    logging.info(f"Found document: {doc}")
                return documents
        except Exception as e:
            logging.error(f"Error querying documents: {e}")
            return []
    
    def get_unknown_documents(self):
        """Get documents with document_type = None (Unknown documents)"""
        query = """
        MATCH (d:Document)
        WHERE d.document_type IS NULL OR d.document_type = 'Unknown'
        RETURN d.id as id, 
               d.fileName as fileName, 
               d.document_type as document_type
        ORDER BY d.id
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                documents = []
                for record in result:
                    doc = {
                        'id': record['id'],
                        'fileName': record['fileName'],
                        'document_type': record['document_type']
                    }
                    documents.append(doc)
                    logging.info(f"Found Unknown/NULL document: {doc}")
                return documents
        except Exception as e:
            logging.error(f"Error querying Unknown documents: {e}")
            return []
    
    def delete_document_by_id(self, doc_id):
        """Delete a document and all its relationships by ID"""
        query = """
        MATCH (d:Document)
        WHERE d.id = $doc_id OR (d.id IS NULL AND ID(d) = $doc_id)
        DETACH DELETE d
        RETURN count(*) as deleted_count
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                # First try with the provided ID
                result = session.run(query, doc_id=doc_id)
                record = result.single()
                deleted_count = record['deleted_count'] if record else 0
                
                # If no document was deleted with ID, try with node ID
                if deleted_count == 0 and doc_id is None:
                    # For documents with NULL id, we need to use a different approach
                    query_null = """
                    MATCH (d:Document)
                    WHERE d.id IS NULL AND d.document_type IS NULL
                    WITH d LIMIT 1
                    DETACH DELETE d
                    RETURN count(*) as deleted_count
                    """
                    result = session.run(query_null)
                    record = result.single()
                    deleted_count = record['deleted_count'] if record else 0
                
                logging.info(f"Deleted document with ID {doc_id}, deleted_count: {deleted_count}")
                return deleted_count
        except Exception as e:
            logging.error(f"Error deleting document {doc_id}: {e}")
            return 0
    
    def delete_null_documents(self, count_to_delete=2):
        """Delete exactly 'count_to_delete' documents with NULL document_type"""
        query = """
        MATCH (d:Document)
        WHERE d.document_type IS NULL
        WITH d LIMIT $count_to_delete
        DETACH DELETE d
        RETURN count(*) as deleted_count
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, count_to_delete=count_to_delete)
                record = result.single()
                deleted_count = record['deleted_count'] if record else 0
                logging.info(f"Deleted {deleted_count} NULL document_type documents")
                return deleted_count
        except Exception as e:
            logging.error(f"Error deleting NULL documents: {e}")
            return 0
    
    def count_documents_by_type(self):
        """Count documents by type"""
        query = """
        MATCH (d:Document)
        RETURN CASE 
                 WHEN d.document_type IS NULL THEN 'NULL/Unknown'
                 ELSE d.document_type 
               END as document_type, 
               count(*) as count
        ORDER BY document_type
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                counts = {}
                for record in result:
                    doc_type = record['document_type']
                    count = record['count']
                    counts[doc_type] = count
                    logging.info(f"Document type '{doc_type}': {count} documents")
                return counts
        except Exception as e:
            logging.error(f"Error counting documents by type: {e}")
            return {}

def main():
    # Load environment variables
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
    neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    logging.info("Starting Neo4j document deletion process (v2)")
    logging.info(f"Connecting to {neo4j_uri} with database {neo4j_database}")
    
    # Initialize Neo4j manager
    neo4j_manager = None
    try:
        neo4j_manager = Neo4jManager(neo4j_uri, neo4j_username, neo4j_password, neo4j_database)
        
        # Step 1: Query all documents before deletion
        logging.info("=== STEP 1: QUERYING ALL DOCUMENTS BEFORE DELETION ===")
        all_docs_before = neo4j_manager.query_all_documents()
        logging.info(f"Total documents found: {len(all_docs_before)}")
        
        # Step 2: Count documents by type before deletion
        logging.info("=== STEP 2: COUNTING DOCUMENTS BY TYPE BEFORE DELETION ===")
        counts_before = neo4j_manager.count_documents_by_type()
        
        # Step 3: Get Unknown documents (NULL document_type)
        logging.info("=== STEP 3: IDENTIFYING UNKNOWN DOCUMENTS (NULL document_type) ===")
        unknown_docs = neo4j_manager.get_unknown_documents()
        logging.info(f"Found {len(unknown_docs)} Unknown/NULL documents")
        
        if len(unknown_docs) < 2:
            logging.error(f"Expected at least 2 Unknown documents, but found only {len(unknown_docs)}")
            return False
            
        # Step 4: Delete exactly 2 Unknown documents using bulk delete
        logging.info("=== STEP 4: DELETING EXACTLY 2 UNKNOWN DOCUMENTS ===")
        deleted_count = neo4j_manager.delete_null_documents(count_to_delete=2)
        
        logging.info(f"Total documents deleted: {deleted_count}")
        
        # Step 5: Verify remaining documents
        logging.info("=== STEP 5: VERIFYING REMAINING DOCUMENTS ===")
        all_docs_after = neo4j_manager.query_all_documents()
        logging.info(f"Total documents remaining: {len(all_docs_after)}")
        
        # Step 6: Count documents by type after deletion
        logging.info("=== STEP 6: COUNTING DOCUMENTS BY TYPE AFTER DELETION ===")
        counts_after = neo4j_manager.count_documents_by_type()
        
        # Step 7: Verify that important documents are preserved
        logging.info("=== STEP 7: VERIFYING IMPORTANT DOCUMENTS ARE PRESERVED ===")
        important_doc_types = ['water_bill', 'electricity_bill', 'waste_manifest']
        preserved_docs = []
        
        for doc in all_docs_after:
            if doc['document_type'] in important_doc_types:
                preserved_docs.append(doc)
                logging.info(f"PRESERVED: {doc['document_type']} - ID: {doc['id']}")
        
        # Final validation
        success = True
        if deleted_count != 2:
            logging.error(f"Expected to delete 2 documents, but deleted {deleted_count}")
            success = False
            
        if len(preserved_docs) != 3:
            logging.error(f"Expected 3 important documents to be preserved, but found {len(preserved_docs)}")
            success = False
        else:
            preserved_types = [doc['document_type'] for doc in preserved_docs]
            for important_doc in important_doc_types:
                if important_doc not in preserved_types:
                    logging.error(f"Important document type '{important_doc}' was not preserved!")
                    success = False
        
        # Check that no NULL documents remain (should be 0 after deleting 2)
        remaining_nulls = counts_after.get('NULL/Unknown', 0)
        expected_nulls = counts_before.get('NULL/Unknown', 0) - 2
        if remaining_nulls != expected_nulls:
            logging.error(f"Expected {expected_nulls} NULL documents to remain, but found {remaining_nulls}")
            success = False
        
        # Summary
        logging.info("=== SUMMARY ===")
        logging.info(f"Documents before deletion: {len(all_docs_before)}")
        logging.info(f"Documents after deletion: {len(all_docs_after)}")
        logging.info(f"Documents deleted: {deleted_count}")
        logging.info(f"Important documents preserved: {len(preserved_docs)}")
        logging.info(f"NULL/Unknown documents before: {counts_before.get('NULL/Unknown', 0)}")
        logging.info(f"NULL/Unknown documents after: {counts_after.get('NULL/Unknown', 0)}")
        logging.info(f"Operation successful: {success}")
        
        return success
        
    except Exception as e:
        logging.error(f"Error during deletion process: {e}")
        return False
    finally:
        if neo4j_manager:
            neo4j_manager.close()
            logging.info("Neo4j connection closed")

if __name__ == "__main__":
    success = main()
    if success:
        logging.info("Document deletion completed successfully!")
        sys.exit(0)
    else:
        logging.error("Document deletion failed!")
        sys.exit(1)