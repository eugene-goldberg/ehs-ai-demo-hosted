#!/usr/bin/env python3
"""
Neo4j Database Ingestion Status Checker

This script connects to the local Neo4j database and checks the status of document ingestion.
It provides comprehensive statistics about:
1. Total number of Document nodes
2. Count of each document type (utility_bill, water_bill, waste_manifest, etc.)
3. Any RejectedDocument nodes
4. Recent documents (check timestamps if available)
"""

import os
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jIngestionChecker:
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            print(f"✅ Successfully connected to Neo4j at {self.uri}")
            print(f"   Database: {self.database}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()

    def run_query(self, query, parameters=None):
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return result.data()
        except Exception as e:
            print(f"❌ Error executing query: {e}")
            return []

    def check_total_documents(self):
        """Check total number of Document nodes"""
        print("\n" + "="*60)
        print("📊 TOTAL DOCUMENT NODES")
        print("="*60)
        
        query = "MATCH (d:Document) RETURN count(d) as totalDocuments"
        results = self.run_query(query)
        
        if results:
            total = results[0]['totalDocuments']
            print(f"Total Document nodes: {total}")
            return total
        return 0

    def check_document_types(self):
        """Check count of each document type"""
        print("\n" + "="*60)
        print("📋 DOCUMENT TYPES BREAKDOWN")
        print("="*60)
        
        # First check if there's a documentType field
        query = """
        MATCH (d:Document) 
        WHERE d.documentType IS NOT NULL
        RETURN d.documentType as docType, count(d) as count
        ORDER BY count DESC
        """
        results = self.run_query(query)
        
        if results:
            print("Documents by documentType field:")
            for record in results:
                print(f"  {record['docType']}: {record['count']}")
        else:
            print("No documentType field found in Document nodes.")
        
        # Also check by file types if available
        query = """
        MATCH (d:Document) 
        WHERE d.fileType IS NOT NULL
        RETURN d.fileType as fileType, count(d) as count
        ORDER BY count DESC
        """
        results = self.run_query(query)
        
        if results:
            print("\nDocuments by fileType field:")
            for record in results:
                print(f"  {record['fileType']}: {record['count']}")
        
        # Check for specific EHS document types in fileName patterns
        print("\nEHS Document Type Analysis (based on fileName patterns):")
        ehs_patterns = [
            ('utility_bill', 'utility'),
            ('water_bill', 'water'),
            ('waste_manifest', 'waste'),
            ('environmental_report', 'environment'),
            ('safety_report', 'safety'),
            ('compliance_document', 'compliance'),
            ('audit_report', 'audit'),
            ('inspection_report', 'inspection')
        ]
        
        for doc_type, pattern in ehs_patterns:
            query = f"""
            MATCH (d:Document) 
            WHERE toLower(d.fileName) CONTAINS toLower('{pattern}')
            RETURN count(d) as count
            """
            results = self.run_query(query)
            if results and results[0]['count'] > 0:
                print(f"  {doc_type}: {results[0]['count']}")

    def check_rejected_documents(self):
        """Check for RejectedDocument nodes"""
        print("\n" + "="*60)
        print("🚫 REJECTED DOCUMENTS")
        print("="*60)
        
        # Check for RejectedDocument label
        query = "MATCH (r:RejectedDocument) RETURN count(r) as rejectedCount"
        results = self.run_query(query)
        
        if results and results[0]['rejectedCount'] > 0:
            print(f"RejectedDocument nodes: {results[0]['rejectedCount']}")
            
            # Get details of rejected documents
            query = """
            MATCH (r:RejectedDocument) 
            RETURN r.fileName as fileName, r.reason as reason, r.errorMessage as errorMessage
            LIMIT 10
            """
            details = self.run_query(query)
            
            if details:
                print("\nRecent rejected documents:")
                for doc in details:
                    print(f"  File: {doc.get('fileName', 'N/A')}")
                    print(f"    Reason: {doc.get('reason', 'N/A')}")
                    print(f"    Error: {doc.get('errorMessage', 'N/A')}")
        else:
            print("✅ No RejectedDocument nodes found")
        
        # Also check for Document nodes with Failed status
        query = """
        MATCH (d:Document) 
        WHERE d.status = 'Failed' 
        RETURN count(d) as failedCount
        """
        results = self.run_query(query)
        
        if results and results[0]['failedCount'] > 0:
            print(f"\nDocument nodes with 'Failed' status: {results[0]['failedCount']}")
            
            # Get details of failed documents
            query = """
            MATCH (d:Document) 
            WHERE d.status = 'Failed' 
            RETURN d.fileName as fileName, d.errorMessage as errorMessage, d.updatedAt as updatedAt
            ORDER BY d.updatedAt DESC
            LIMIT 10
            """
            details = self.run_query(query)
            
            if details:
                print("\nRecent failed documents:")
                for doc in details:
                    print(f"  File: {doc.get('fileName', 'N/A')}")
                    print(f"    Error: {doc.get('errorMessage', 'N/A')}")
                    print(f"    Updated: {doc.get('updatedAt', 'N/A')}")
        else:
            print("✅ No failed Document nodes found")

    def check_recent_documents(self):
        """Check recent documents with timestamps"""
        print("\n" + "="*60)
        print("🕒 RECENT DOCUMENTS")
        print("="*60)
        
        # Check documents with creation timestamps
        query = """
        MATCH (d:Document) 
        WHERE d.createdAt IS NOT NULL 
        RETURN d.fileName as fileName, d.createdAt as createdAt, d.status as status,
               d.fileType as fileType, d.fileSize as fileSize
        ORDER BY d.createdAt DESC 
        LIMIT 10
        """
        results = self.run_query(query)
        
        if results:
            print("Most recently created documents:")
            for doc in results:
                created_at = doc.get('createdAt', 'N/A')
                if isinstance(created_at, (int, float)):
                    # Convert timestamp to readable format
                    try:
                        created_at = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                
                print(f"  📄 {doc.get('fileName', 'N/A')}")
                print(f"     Status: {doc.get('status', 'N/A')}")
                print(f"     Type: {doc.get('fileType', 'N/A')}")
                print(f"     Size: {doc.get('fileSize', 'N/A')} bytes")
                print(f"     Created: {created_at}")
                print()
        else:
            print("No documents with creation timestamps found")

    def check_processing_status(self):
        """Check processing status of all documents"""
        print("\n" + "="*60)
        print("⚙️ PROCESSING STATUS")
        print("="*60)
        
        query = """
        MATCH (d:Document) 
        RETURN d.status as status, count(d) as count
        ORDER BY count DESC
        """
        results = self.run_query(query)
        
        if results:
            print("Documents by processing status:")
            for record in results:
                status = record['status'] or 'NULL'
                count = record['count']
                print(f"  {status}: {count}")
        else:
            print("No status information available")

    def check_graph_structure(self):
        """Check basic graph structure and relationships"""
        print("\n" + "="*60)
        print("🕸️ GRAPH STRUCTURE OVERVIEW")
        print("="*60)
        
        # Check node types
        query = "CALL db.labels() YIELD label RETURN label ORDER BY label"
        results = self.run_query(query)
        
        if results:
            print("Node labels in the database:")
            for record in results:
                label = record['label']
                
                # Count nodes for each label
                count_query = f"MATCH (n:{label}) RETURN count(n) as count"
                count_results = self.run_query(count_query)
                count = count_results[0]['count'] if count_results else 0
                
                print(f"  {label}: {count} nodes")
        
        # Check relationship types
        query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
        results = self.run_query(query)
        
        if results:
            print("\nRelationship types in the database:")
            for record in results:
                rel_type = record['relationshipType']
                
                # Count relationships for each type
                count_query = f"MATCH ()-[r:{rel_type}]-() RETURN count(r) as count"
                count_results = self.run_query(count_query)
                count = count_results[0]['count'] if count_results else 0
                
                print(f"  {rel_type}: {count} relationships")

    def run_comprehensive_check(self):
        """Run all checks and provide a comprehensive report"""
        print("🔍 Neo4j Database Ingestion Status Report")
        print("=" * 80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.connect():
            return False
        
        try:
            # Run all checks
            total_docs = self.check_total_documents()
            self.check_document_types()
            self.check_rejected_documents()
            self.check_recent_documents()
            self.check_processing_status()
            self.check_graph_structure()
            
            # Summary
            print("\n" + "="*60)
            print("📈 SUMMARY")
            print("="*60)
            print(f"Total documents in database: {total_docs}")
            
            if total_docs == 0:
                print("⚠️  No documents found in the database.")
                print("   This could mean:")
                print("   - No documents have been ingested yet")
                print("   - Documents are stored with a different structure")
                print("   - Connection to the wrong database")
            else:
                print("✅ Documents have been successfully ingested into the database")
            
        except Exception as e:
            print(f"❌ Error during comprehensive check: {e}")
            return False
        finally:
            self.close()
        
        return True

def main():
    """Main function to run the ingestion check"""
    checker = Neo4jIngestionChecker()
    success = checker.run_comprehensive_check()
    
    if success:
        print("\n✅ Ingestion check completed successfully!")
    else:
        print("\n❌ Ingestion check failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())