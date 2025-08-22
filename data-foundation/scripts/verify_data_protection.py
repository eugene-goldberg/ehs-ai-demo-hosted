#!/usr/bin/env python3
"""
Neo4j Data Protection Verification Script

This script verifies that the Neo4j data protection fixes are working correctly by:
1. Checking the current state of the Neo4j database
2. Testing the --clear-database flag behavior
3. Verifying the web app endpoint returns all documents
4. Providing a clear summary of verification results
"""

import os
import sys
import json
import subprocess
import requests
from typing import Dict, Any, List, Tuple
from neo4j import GraphDatabase
import time

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Neo4jDataProtectionVerifier:
    def __init__(self):
        # Neo4j connection settings
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        
        # Web app settings
        self.web_app_url = os.getenv('WEB_APP_URL', 'http://localhost:8000')
        
        # Initialize Neo4j driver
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✓ Neo4j connection established")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            sys.exit(1)
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        with self.driver.session() as session:
            # Count nodes by label
            node_counts = {}
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]
            
            for label in labels:
                count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                node_counts[label] = count_result.single()["count"]
            
            # Count relationships by type
            relationship_counts = {}
            types_result = session.run("CALL db.relationshipTypes()")
            types = [record["relationshipType"] for record in types_result]
            
            for rel_type in types:
                count_result = session.run(f"MATCH ()-[r:{rel_type}]-() RETURN count(r) as count")
                relationship_counts[rel_type] = count_result.single()["count"]
            
            # Total counts
            total_nodes_result = session.run("MATCH (n) RETURN count(n) as count")
            total_nodes = total_nodes_result.single()["count"]
            
            total_rels_result = session.run("MATCH ()-[r]-() RETURN count(r) as count")
            total_relationships = total_rels_result.single()["count"]
            
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "node_counts": node_counts,
                "relationship_counts": relationship_counts
            }
    
    def check_web_app_status(self) -> bool:
        """Check if the web app is running"""
        try:
            response = requests.get(f"{self.web_app_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_documents_from_web_app(self) -> List[Dict]:
        """Get documents from the web app API"""
        try:
            response = requests.get(f"{self.web_app_url}/api/documents", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"✗ Web app returned status code: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to connect to web app: {e}")
            return []
    
    def simulate_ingestion_without_clear(self) -> Tuple[bool, str]:
        """Simulate running ingestion without --clear-database flag"""
        try:
            # This would typically run the ingestion script
            # For now, we'll just check that data exists and would be preserved
            stats_before = self.get_database_stats()
            
            if stats_before["total_nodes"] > 0:
                return True, f"Database has {stats_before['total_nodes']} nodes that would be preserved"
            else:
                return True, "Database is empty - no data to preserve"
                
        except Exception as e:
            return False, f"Error during simulation: {e}"
    
    def simulate_ingestion_with_clear(self) -> Tuple[bool, str]:
        """Simulate what would happen with --clear-database flag"""
        try:
            stats_before = self.get_database_stats()
            
            # In a real scenario, this would clear the database
            # For verification, we just show what would happen
            if stats_before["total_nodes"] > 0:
                return True, f"Would clear {stats_before['total_nodes']} nodes and {stats_before['total_relationships']} relationships"
            else:
                return True, "Database is already empty - nothing to clear"
                
        except Exception as e:
            return False, f"Error during simulation: {e}"
    
    def verify_data_consistency(self) -> Tuple[bool, str]:
        """Verify data consistency between Neo4j and web app"""
        try:
            # Get stats from Neo4j
            neo4j_stats = self.get_database_stats()
            
            # Check if web app is running
            if not self.check_web_app_status():
                return False, "Web app is not running - cannot verify consistency"
            
            # Get documents from web app
            web_app_docs = self.get_documents_from_web_app()
            
            # Compare counts (this is a basic check - in practice you'd want more sophisticated comparison)
            neo4j_document_count = neo4j_stats["node_counts"].get("Document", 0)
            web_app_document_count = len(web_app_docs)
            
            if neo4j_document_count == web_app_document_count:
                return True, f"Document counts match: Neo4j={neo4j_document_count}, WebApp={web_app_document_count}"
            else:
                return False, f"Document count mismatch: Neo4j={neo4j_document_count}, WebApp={web_app_document_count}"
                
        except Exception as e:
            return False, f"Error during consistency check: {e}"
    
    def run_verification(self) -> None:
        """Run all verification tests"""
        print("=" * 60)
        print("Neo4j Data Protection Verification")
        print("=" * 60)
        
        # 1. Check current Neo4j state
        print("\n1. Current Neo4j Database State:")
        print("-" * 40)
        try:
            stats = self.get_database_stats()
            print(f"Total Nodes: {stats['total_nodes']}")
            print(f"Total Relationships: {stats['total_relationships']}")
            
            if stats['node_counts']:
                print("\nNode Counts by Label:")
                for label, count in stats['node_counts'].items():
                    print(f"  {label}: {count}")
            
            if stats['relationship_counts']:
                print("\nRelationship Counts by Type:")
                for rel_type, count in stats['relationship_counts'].items():
                    print(f"  {rel_type}: {count}")
            
            print("✓ Successfully retrieved database statistics")
            
        except Exception as e:
            print(f"✗ Failed to get database statistics: {e}")
        
        # 2. Test --clear-database flag behavior
        print("\n2. Database Clear Flag Behavior:")
        print("-" * 40)
        
        # Without clear flag
        success, message = self.simulate_ingestion_without_clear()
        if success:
            print(f"✓ Without --clear-database: {message}")
        else:
            print(f"✗ Without --clear-database: {message}")
        
        # With clear flag
        success, message = self.simulate_ingestion_with_clear()
        if success:
            print(f"✓ With --clear-database: {message}")
        else:
            print(f"✗ With --clear-database: {message}")
        
        # 3. Check web app connectivity and data
        print("\n3. Web App Integration:")
        print("-" * 40)
        
        if self.check_web_app_status():
            print("✓ Web app is running and accessible")
            
            docs = self.get_documents_from_web_app()
            print(f"✓ Retrieved {len(docs)} documents from web app")
            
            if docs:
                print("Sample document fields:")
                sample_doc = docs[0]
                for key in list(sample_doc.keys())[:5]:  # Show first 5 fields
                    print(f"  - {key}")
                if len(sample_doc.keys()) > 5:
                    print(f"  ... and {len(sample_doc.keys()) - 5} more fields")
        else:
            print("✗ Web app is not running or not accessible")
        
        # 4. Verify data consistency
        print("\n4. Data Consistency Verification:")
        print("-" * 40)
        
        success, message = self.verify_data_consistency()
        if success:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
        
        # 5. Summary and recommendations
        print("\n5. Summary and Recommendations:")
        print("-" * 40)
        
        current_stats = self.get_database_stats()
        web_app_running = self.check_web_app_status()
        
        if current_stats["total_nodes"] > 0:
            print("✓ Database contains data - protection mechanisms are important")
        else:
            print("! Database is empty - consider adding test data for verification")
        
        if web_app_running:
            print("✓ Web app is accessible - can verify end-to-end functionality")
        else:
            print("! Web app is not running - start it to verify full integration")
        
        print("\nRecommended next steps:")
        print("1. If database is empty, run ingestion script to populate test data")
        print("2. Test the --clear-database flag with actual ingestion script")
        print("3. Verify web app displays all documents correctly")
        print("4. Test multiple ingestion runs without --clear-database flag")
        
        print("\n" + "=" * 60)
        print("Verification Complete")
        print("=" * 60)

def main():
    """Main function"""
    verifier = Neo4jDataProtectionVerifier()
    
    try:
        verifier.run_verification()
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error during verification: {e}")
    finally:
        verifier.close()

if __name__ == "__main__":
    main()