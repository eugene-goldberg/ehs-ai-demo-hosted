#!/usr/bin/env python3
"""
Test script to examine the actual properties of nodes in Neo4j database
to understand what data we have and what's missing.
"""

import os
import sys
from datetime import datetime
from neo4j import GraphDatabase

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NodePropertiesExaminer:
    def __init__(self):
        self.driver = None
        self.session = None
        
        # Neo4j connection details from .env
        self.uri = "bolt://localhost:7687"
        self.username = "neo4j"
        self.password = "EhsAI2024!"
        self.database = "neo4j"
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            self.session = self.driver.session(database=self.database)
            print(f"✅ Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.session:
            self.session.close()
        if self.driver:
            self.driver.close()
        print("🔌 Disconnected from Neo4j")
    
    def examine_node_type(self, node_type, limit=3):
        """Examine properties of a specific node type"""
        print(f"\n=== {node_type.upper()} NODE PROPERTIES ===")
        
        query = f"""
        MATCH (n:{node_type})
        RETURN n
        LIMIT {limit}
        """
        
        try:
            result = self.session.run(query)
            nodes = list(result)
            
            if not nodes:
                print(f"⚠️  No {node_type} nodes found")
                return
            
            print(f"Examining first {len(nodes)} {node_type} nodes:")
            
            for i, record in enumerate(nodes, 1):
                node = record['n']
                print(f"\n  Node {i}:")
                print(f"    Labels: {list(node.labels)}")
                print(f"    Properties:")
                
                for key, value in node.items():
                    print(f"      {key}: {repr(value)}")
                    
        except Exception as e:
            print(f"❌ Error examining {node_type} nodes: {e}")
    
    def examine_all_relationships(self):
        """Examine all relationships in the database"""
        print("\n=== ALL RELATIONSHIPS ===")
        
        query = """
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) as relationship_type, count(r) as count
        ORDER BY relationship_type
        """
        
        try:
            result = self.session.run(query)
            relationships = list(result)
            
            if not relationships:
                print("⚠️  No relationships found in the database")
                return
            
            print(f"Found {len(relationships)} relationship types:")
            for rel in relationships:
                print(f"  - {rel['relationship_type']}: {rel['count']} instances")
                
        except Exception as e:
            print(f"❌ Error examining relationships: {e}")
    
    def examine_sample_relationships(self):
        """Examine sample relationships to understand structure"""
        print("\n=== SAMPLE RELATIONSHIPS ===")
        
        query = """
        MATCH (a)-[r]->(b)
        RETURN labels(a)[0] as from_type, type(r) as relationship, 
               labels(b)[0] as to_type, a.name as from_name, b.name as to_name
        LIMIT 10
        """
        
        try:
            result = self.session.run(query)
            relationships = list(result)
            
            if not relationships:
                print("⚠️  No relationships found")
                return
            
            print(f"Sample relationships ({len(relationships)}):")
            for rel in relationships:
                print(f"  {rel['from_name']} ({rel['from_type']}) -[{rel['relationship']}]-> {rel['to_name']} ({rel['to_type']})")
                
        except Exception as e:
            print(f"❌ Error examining sample relationships: {e}")
    
    def examine_all_labels(self):
        """Examine all node labels in the database"""
        print("\n=== ALL NODE LABELS ===")
        
        query = """
        CALL db.labels() YIELD label
        RETURN label
        ORDER BY label
        """
        
        try:
            result = self.session.run(query)
            labels = list(result)
            
            if not labels:
                print("⚠️  No labels found")
                return
            
            print(f"Found {len(labels)} node labels:")
            for label in labels:
                print(f"  - {label['label']}")
                
        except Exception as e:
            print(f"❌ Error examining labels: {e}")
    
    def count_nodes_by_label(self):
        """Count nodes by label"""
        print("\n=== NODE COUNTS BY LABEL ===")
        
        query = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) 
        YIELD value
        RETURN label, value.count as count
        ORDER BY value.count DESC
        """
        
        try:
            result = self.session.run(query)
            counts = list(result)
            
            if not counts:
                # Fallback method if APOC is not available
                print("Using fallback method to count nodes...")
                self.count_nodes_fallback()
                return
            
            print(f"Node counts by label:")
            for count in counts:
                print(f"  {count['label']}: {count['count']} nodes")
                
        except Exception as e:
            print(f"❌ Error counting nodes (trying fallback): {e}")
            self.count_nodes_fallback()
    
    def count_nodes_fallback(self):
        """Fallback method to count nodes by label"""
        labels = ['Site', 'Building', 'Floor', 'Area', 'Document', 'Chunk', 'Entity']
        
        for label in labels:
            try:
                query = f"MATCH (n:{label}) RETURN count(n) as count"
                result = self.session.run(query)
                count = result.single()['count']
                if count > 0:
                    print(f"  {label}: {count} nodes")
            except Exception as e:
                print(f"  {label}: Error counting - {e}")
    
    def run_examination(self):
        """Run all examinations"""
        print("=" * 60)
        print(f"NODE PROPERTIES EXAMINATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        if not self.connect():
            return False
        
        try:
            # Examine all labels
            self.examine_all_labels()
            
            # Count nodes by label
            self.count_nodes_by_label()
            
            # Examine node properties for hierarchy types
            self.examine_node_type('Site', 2)
            self.examine_node_type('Building', 3)
            self.examine_node_type('Floor', 3)
            self.examine_node_type('Area', 3)
            
            # Examine relationships
            self.examine_all_relationships()
            self.examine_sample_relationships()
            
            return True
            
        finally:
            self.close()

def main():
    """Main function"""
    examiner = NodePropertiesExaminer()
    success = examiner.run_examination()
    
    if success:
        print("\n✅ Node properties examination completed successfully")
    else:
        print("\n❌ Node properties examination failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
