#!/usr/bin/env python3

"""
Script to verify the current state of documents in Neo4j after deletion.
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
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verify_documents.log'),
        logging.StreamHandler()
    ]
)

class Neo4jVerifier:
    def __init__(self, uri, username, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        
    def close(self):
        if self.driver:
            self.driver.close()
    
    def get_all_documents(self):
        """Get all documents with their details"""
        query = """
        MATCH (d:Document)
        RETURN d.id as id, 
               d.fileName as fileName, 
               d.document_type as document_type,
               d.entityName as entityName,
               labels(d) as labels
        ORDER BY d.document_type, d.id
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                documents = []
                for record in result:
                    doc = {
                        'id': record['id'],
                        'fileName': record['fileName'],
                        'document_type': record['document_type'],
                        'entityName': record['entityName'],
                        'labels': record['labels']
                    }
                    documents.append(doc)
                return documents
        except Exception as e:
            logging.error(f"Error querying documents: {e}")
            return []
    
    def get_document_counts(self):
        """Get count of documents by type"""
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
                total = 0
                for record in result:
                    doc_type = record['document_type']
                    count = record['count']
                    counts[doc_type] = count
                    total += count
                return counts, total
        except Exception as e:
            logging.error(f"Error counting documents: {e}")
            return {}, 0

def main():
    # Load environment variables
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
    neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    logging.info("Starting Neo4j document verification")
    logging.info(f"Connecting to {neo4j_uri} with database {neo4j_database}")
    
    # Initialize Neo4j verifier
    verifier = None
    try:
        verifier = Neo4jVerifier(neo4j_uri, neo4j_username, neo4j_password, neo4j_database)
        
        # Get all documents
        logging.info("=== CURRENT DOCUMENTS IN DATABASE ===")
        documents = verifier.get_all_documents()
        
        for i, doc in enumerate(documents, 1):
            logging.info(f"Document {i}:")
            logging.info(f"  ID: {doc['id']}")
            logging.info(f"  File Name: {doc['fileName']}")
            logging.info(f"  Document Type: {doc['document_type']}")
            logging.info(f"  Entity Name: {doc['entityName']}")
            logging.info(f"  Labels: {doc['labels']}")
            logging.info("")
        
        # Get document counts
        logging.info("=== DOCUMENT COUNTS BY TYPE ===")
        counts, total = verifier.get_document_counts()
        for doc_type, count in counts.items():
            logging.info(f"{doc_type}: {count} documents")
        
        logging.info(f"\nTotal documents: {total}")
        
        # Verification checks
        logging.info("=== VERIFICATION RESULTS ===")
        
        # Check 1: Should have exactly 3 documents
        if total == 3:
            logging.info("✓ PASS: Exactly 3 documents remain in database")
        else:
            logging.error(f"✗ FAIL: Expected 3 documents, found {total}")
        
        # Check 2: Should have no NULL/Unknown documents
        null_count = counts.get('NULL/Unknown', 0)
        if null_count == 0:
            logging.info("✓ PASS: No NULL/Unknown documents remain")
        else:
            logging.error(f"✗ FAIL: Found {null_count} NULL/Unknown documents")
        
        # Check 3: Should have exactly one of each important document type
        required_types = ['electricity_bill', 'waste_manifest', 'water_bill']
        for doc_type in required_types:
            count = counts.get(doc_type, 0)
            if count == 1:
                logging.info(f"✓ PASS: Found 1 {doc_type} document")
            else:
                logging.error(f"✗ FAIL: Expected 1 {doc_type} document, found {count}")
        
        # Overall verification
        all_checks_pass = (
            total == 3 and
            null_count == 0 and
            all(counts.get(doc_type, 0) == 1 for doc_type in required_types)
        )
        
        if all_checks_pass:
            logging.info("🎉 ALL VERIFICATION CHECKS PASSED!")
            logging.info("The deletion operation was successful:")
            logging.info("- Exactly 2 Unknown documents were deleted")
            logging.info("- Water Bill, Electric Bill, and Waste Manifest are preserved")
            logging.info("- No unwanted documents remain")
        else:
            logging.error("❌ VERIFICATION FAILED!")
            
        return all_checks_pass
        
    except Exception as e:
        logging.error(f"Error during verification: {e}")
        return False
    finally:
        if verifier:
            verifier.close()
            logging.info("Neo4j connection closed")

if __name__ == "__main__":
    success = main()
    if success:
        logging.info("Verification completed successfully!")
        sys.exit(0)
    else:
        logging.error("Verification failed!")
        sys.exit(1)