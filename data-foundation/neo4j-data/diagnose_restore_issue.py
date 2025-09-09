#!/usr/bin/env python3
"""
Neo4j Restore Issue Diagnostic Script

This script diagnoses issues with Neo4j database restore by:
1. Connecting to Neo4j database
2. Checking for nodes with __backup_id__ property
3. Showing sample nodes with their properties
4. Looking for specific backup IDs (130, 163)
5. Analyzing potential mismatches between backup file and database
"""

import os
import sys
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import json

class Neo4jRestoreDiagnostics:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = None):
        """Initialize Neo4j connection"""
        if password is None:
            # Try to get password from environment
            password = os.getenv('NEO4J_PASSWORD', 'password')
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print(f"✓ Connected to Neo4j at {uri}")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            sys.exit(1)
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def run_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get basic database statistics"""
        print("
=== DATABASE STATISTICS ===")
        
        queries = {
            'total_nodes': "MATCH (n) RETURN count(n) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count",
            'node_labels': "CALL db.labels() YIELD label RETURN collect(label) as labels",
            'relationship_types': "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        }
        
        stats = {}
        for key, query in queries.items():
            try:
                result = self.run_query(query)
                if key in ['node_labels', 'relationship_types']:
                    stats[key] = result[0][list(result[0].keys())[0]] if result else []
                else:
                    stats[key] = result[0]['count'] if result else 0
                print(f"{key.replace('_', ' ').title()}: {stats[key]}")
            except Exception as e:
                print(f"Error getting {key}: {e}")
                stats[key] = None
        
        return stats
    
    def check_backup_id_nodes(self) -> Dict[str, Any]:
        """Check for nodes with __backup_id__ property"""
        print("
=== BACKUP ID ANALYSIS ===")
        
        # Count nodes with __backup_id__
        query = "MATCH (n) WHERE exists(n.__backup_id__) RETURN count(n) as count"
        result = self.run_query(query)
        backup_id_count = result[0]['count'] if result else 0
        print(f"Nodes with __backup_id__: {backup_id_count}")
        
        if backup_id_count == 0:
            print("✗ No nodes found with __backup_id__ property")
            return {'count': 0, 'samples': [], 'id_range': None}
        
        # Get sample nodes with __backup_id__
        query = """
        MATCH (n) WHERE exists(n.__backup_id__) 
        RETURN labels(n) as labels, n.__backup_id__ as backup_id, 
               keys(n) as properties, n
        LIMIT 10
        """
        samples = self.run_query(query)
        print(f"
Sample nodes with __backup_id__:")
        for i, sample in enumerate(samples[:5], 1):
            print(f"  {i}. Labels: {sample['labels']}")
            print(f"     Backup ID: {sample['backup_id']}")
            print(f"     Properties: {sample['properties']}")
            print(f"     Node data: {dict(sample['n'])}")
            print()
        
        # Get backup ID range
        query = """
        MATCH (n) WHERE exists(n.__backup_id__) 
        RETURN min(n.__backup_id__) as min_id, max(n.__backup_id__) as max_id
        """
        range_result = self.run_query(query)
        id_range = range_result[0] if range_result else {'min_id': None, 'max_id': None}
        print(f"Backup ID range: {id_range['min_id']} to {id_range['max_id']}")
        
        return {
            'count': backup_id_count,
            'samples': samples,
            'id_range': id_range
        }
    
    def search_specific_backup_ids(self, backup_ids: List[int]) -> Dict[int, Any]:
        """Search for specific backup IDs"""
        print(f"
=== SEARCHING FOR SPECIFIC BACKUP IDs: {backup_ids} ===")
        
        results = {}
        for backup_id in backup_ids:
            query = "MATCH (n {__backup_id__: }) RETURN n, labels(n) as labels"
            result = self.run_query(query, {'backup_id': backup_id})
            
            if result:
                print(f"✓ Found node with backup_id {backup_id}:")
                node_data = result[0]
                print(f"  Labels: {node_data['labels']}")
                print(f"  Properties: {dict(node_data['n'])}")
                results[backup_id] = node_data
            else:
                print(f"✗ No node found with backup_id {backup_id}")
                results[backup_id] = None
            print()
        
        return results
    
    def analyze_backup_id_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of backup IDs by label"""
        print("
=== BACKUP ID DISTRIBUTION BY LABEL ===")
        
        query = """
        MATCH (n) WHERE exists(n.__backup_id__)
        UNWIND labels(n) as label
        RETURN label, count(n) as node_count, 
               min(n.__backup_id__) as min_backup_id,
               max(n.__backup_id__) as max_backup_id
        ORDER BY node_count DESC
        """
        
        results = self.run_query(query)
        distribution = {}
        
        for result in results:
            label = result['label']
            distribution[label] = {
                'count': result['node_count'],
                'min_backup_id': result['min_backup_id'],
                'max_backup_id': result['max_backup_id']
            }
            print(f"{label}: {result['node_count']} nodes (backup IDs {result['min_backup_id']}-{result['max_backup_id']})")
        
        return distribution
    
    def check_nodes_without_backup_id(self) -> Dict[str, Any]:
        """Check for nodes without __backup_id__ property"""
        print("
=== NODES WITHOUT BACKUP ID ===")
        
        # Count nodes without __backup_id__
        query = "MATCH (n) WHERE NOT exists(n.__backup_id__) RETURN count(n) as count"
        result = self.run_query(query)
        no_backup_id_count = result[0]['count'] if result else 0
        print(f"Nodes without __backup_id__: {no_backup_id_count}")
        
        if no_backup_id_count > 0:
            # Get sample nodes without __backup_id__
            query = """
            MATCH (n) WHERE NOT exists(n.__backup_id__) 
            RETURN labels(n) as labels, keys(n) as properties, n
            LIMIT 5
            """
            samples = self.run_query(query)
            print(f"
Sample nodes without __backup_id__:")
            for i, sample in enumerate(samples, 1):
                print(f"  {i}. Labels: {sample['labels']}")
                print(f"     Properties: {sample['properties']}")
                print(f"     Node data: {dict(sample['n'])}")
                print()
            
            return {'count': no_backup_id_count, 'samples': samples}
        
        return {'count': 0, 'samples': []}
    
    def check_duplicate_backup_ids(self) -> List[Dict]:
        """Check for duplicate backup IDs"""
        print("
=== CHECKING FOR DUPLICATE BACKUP IDs ===")
        
        query = """
        MATCH (n) WHERE exists(n.__backup_id__)
        WITH n.__backup_id__ as backup_id, collect(n) as nodes
        WHERE size(nodes) > 1
        RETURN backup_id, size(nodes) as duplicate_count
        ORDER BY duplicate_count DESC
        LIMIT 10
        """
        
        duplicates = self.run_query(query)
        
        if duplicates:
            print(f"Found {len(duplicates)} backup IDs with duplicates:")
            for dup in duplicates:
                print(f"  Backup ID {dup['backup_id']}: {dup['duplicate_count']} nodes")
        else:
            print("✓ No duplicate backup IDs found")
        
        return duplicates
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report"""
        print("
" + "="*60)
        print("NEO4J RESTORE ISSUE DIAGNOSTIC REPORT")
        print("="*60)
        
        # Collect all diagnostic data
        stats = self.get_database_stats()
        backup_id_analysis = self.check_backup_id_nodes()
        specific_ids = self.search_specific_backup_ids([130, 163])
        distribution = self.analyze_backup_id_distribution()
        no_backup_id = self.check_nodes_without_backup_id()
        duplicates = self.check_duplicate_backup_ids()
        
        # Summary and recommendations
        print("
=== SUMMARY AND RECOMMENDATIONS ===")
        
        total_nodes = stats.get('total_nodes', 0)
        backup_id_nodes = backup_id_analysis.get('count', 0)
        no_backup_id_nodes = no_backup_id.get('count', 0)
        
        print(f"Total nodes in database: {total_nodes}")
        print(f"Nodes with __backup_id__: {backup_id_nodes}")
        print(f"Nodes without __backup_id__: {no_backup_id_nodes}")
        
        if backup_id_nodes == 0:
            print("
⚠️  ISSUE: No nodes have __backup_id__ property")
            print("   This suggests the restore process may not have preserved backup metadata")
            print("   Recommendation: Check backup file format and restore procedure")
        
        elif backup_id_nodes > 0 and no_backup_id_nodes > 0:
            print("
⚠️  MIXED STATE: Some nodes have __backup_id__, others don't")
            print("   This suggests partial restore or mixed data sources")
            print("   Recommendation: Verify restore completeness")
        
        if specific_ids[130] is None:
            print("
⚠️  SPECIFIC ISSUE: Node with backup_id 130 not found")
            print("   This node should exist if restore was complete")
        
        if specific_ids[163] is None:
            print("
⚠️  SPECIFIC ISSUE: Node with backup_id 163 not found")
            print("   This node should exist if restore was complete")
        
        if duplicates:
            print("
⚠️  DATA INTEGRITY ISSUE: Duplicate backup IDs found")
            print("   This suggests data corruption or multiple restore attempts")
        
        print("
=== DIAGNOSTIC COMPLETE ===")

def main():
    """Main diagnostic function"""
    # Check for environment variables
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    if not neo4j_password:
        print("Warning: NEO4J_PASSWORD environment variable not set, using default 'password'")
    
    # Initialize diagnostics
    diagnostics = None
    try:
        diagnostics = Neo4jRestoreDiagnostics(
            uri="bolt://localhost:7687",
            user="neo4j",
            password=neo4j_password
        )
        
        # Run diagnostic report
        diagnostics.generate_diagnostic_report()
        
    except KeyboardInterrupt:
        print("
Diagnostic interrupted by user")
    except Exception as e:
        print(f"
Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if diagnostics:
            diagnostics.close()

if __name__ == "__main__":
    main()
