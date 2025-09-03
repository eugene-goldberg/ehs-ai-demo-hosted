#!/usr/bin/env python3
"""
Detailed Neo4j Database Analysis

This script provides detailed information about the documents and data structures
in the Neo4j database.
"""

import os
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DetailedNeo4jChecker:
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

    def analyze_document_properties(self):
        """Analyze all properties of Document nodes"""
        print("\n" + "="*60)
        print("📄 DOCUMENT NODE PROPERTIES ANALYSIS")
        print("="*60)
        
        # Get all Document nodes with all their properties
        query = """
        MATCH (d:Document) 
        RETURN d
        ORDER BY d.fileName
        """
        results = self.run_query(query)
        
        if results:
            print(f"Found {len(results)} Document nodes:")
            for i, record in enumerate(results, 1):
                doc = record['d']
                print(f"\n📄 Document {i}:")
                print(f"  Properties available:")
                
                # Display all properties
                for key, value in doc.items():
                    if value is not None:
                        # Convert timestamps if they look like timestamps
                        if key in ['createdAt', 'updatedAt'] and isinstance(value, (int, float)):
                            try:
                                readable_time = datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
                                print(f"    {key}: {value} ({readable_time})")
                            except:
                                print(f"    {key}: {value}")
                        else:
                            print(f"    {key}: {value}")
        else:
            print("No Document nodes found")

    def analyze_specific_document_types(self):
        """Look for specific EHS document types based on node labels"""
        print("\n" + "="*60)
        print("📋 SPECIFIC EHS DOCUMENT TYPES")
        print("="*60)
        
        # Check for specific EHS-related node types
        ehs_labels = [
            'UtilityBill', 'WaterBill', 'WasteManifest', 'Electricitybill', 'Waterbill', 'Wastemanifest'
        ]
        
        for label in ehs_labels:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            results = self.run_query(query)
            if results and results[0]['count'] > 0:
                count = results[0]['count']
                print(f"{label}: {count} nodes")
                
                # Get sample properties for this document type
                sample_query = f"MATCH (n:{label}) RETURN n LIMIT 3"
                samples = self.run_query(sample_query)
                
                if samples:
                    print(f"  Sample properties from {label} nodes:")
                    for i, sample in enumerate(samples, 1):
                        node = sample['n']
                        print(f"    Sample {i}:")
                        for key, value in list(node.items())[:5]:  # Show first 5 properties
                            print(f"      {key}: {value}")
                        if len(node.items()) > 5:
                            print(f"      ... and {len(node.items()) - 5} more properties")
                print()

    def analyze_rejected_documents(self):
        """Analyze rejected documents in detail"""
        print("\n" + "="*60)
        print("🚫 REJECTED DOCUMENTS DETAIL")
        print("="*60)
        
        query = "MATCH (r:RejectedDocument) RETURN r"
        results = self.run_query(query)
        
        if results:
            print(f"Found {len(results)} RejectedDocument nodes:")
            for i, record in enumerate(results, 1):
                rejected = record['r']
                print(f"\n🚫 RejectedDocument {i}:")
                for key, value in rejected.items():
                    print(f"    {key}: {value}")
        else:
            print("No RejectedDocument nodes found")

    def analyze_data_relationships(self):
        """Analyze relationships between documents and other data"""
        print("\n" + "="*60)
        print("🔗 DOCUMENT RELATIONSHIPS")
        print("="*60)
        
        # Check relationships from Document nodes
        query = """
        MATCH (d:Document)-[r]->(n)
        RETURN type(r) as relationshipType, labels(n) as targetLabels, count(*) as count
        ORDER BY count DESC
        """
        results = self.run_query(query)
        
        if results:
            print("Outgoing relationships from Document nodes:")
            for record in results:
                print(f"  {record['relationshipType']} -> {record['targetLabels']}: {record['count']}")
        
        # Check relationships to Document nodes
        query = """
        MATCH (n)-[r]->(d:Document)
        RETURN type(r) as relationshipType, labels(n) as sourceLabels, count(*) as count
        ORDER BY count DESC
        """
        results = self.run_query(query)
        
        if results:
            print("\nIncoming relationships to Document nodes:")
            for record in results:
                print(f"  {record['sourceLabels']} -{record['relationshipType']}-> Document: {record['count']}")

    def analyze_data_volume(self):
        """Analyze the volume of different types of data"""
        print("\n" + "="*60)
        print("📊 DATA VOLUME ANALYSIS")
        print("="*60)
        
        # Key EHS data types
        key_labels = [
            'EHSMetric', 'ElectricityConsumption', 'WaterConsumption', 'WasteGeneration',
            'Incident', 'ComplianceRecord', 'EnvironmentalKPI', 'Recommendation',
            'Facility', 'Area', 'Building'
        ]
        
        print("Key EHS data volumes:")
        for label in key_labels:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
            results = self.run_query(query)
            if results and results[0]['count'] > 0:
                count = results[0]['count']
                print(f"  {label}: {count:,} nodes")

    def check_recent_activity(self):
        """Check for recent activity in the database"""
        print("\n" + "="*60)
        print("⏰ RECENT ACTIVITY ANALYSIS")
        print("="*60)
        
        # Look for any nodes with timestamp properties
        timestamp_props = ['createdAt', 'updatedAt', 'timestamp', 'created', 'modified']
        
        for prop in timestamp_props:
            query = f"""
            MATCH (n) 
            WHERE n.{prop} IS NOT NULL 
            RETURN labels(n) as nodeLabels, n.{prop} as timestamp
            ORDER BY n.{prop} DESC 
            LIMIT 5
            """
            results = self.run_query(query)
            
            if results:
                print(f"\nMost recent nodes by {prop}:")
                for record in results:
                    timestamp = record['timestamp']
                    labels = record['nodeLabels']
                    
                    if isinstance(timestamp, (int, float)):
                        try:
                            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            print(f"  {labels}: {readable_time}")
                        except:
                            print(f"  {labels}: {timestamp}")
                    else:
                        print(f"  {labels}: {timestamp}")

    def run_detailed_analysis(self):
        """Run all detailed analyses"""
        print("🔍 Detailed Neo4j Database Analysis")
        print("=" * 80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.connect():
            return False
        
        try:
            self.analyze_document_properties()
            self.analyze_specific_document_types()
            self.analyze_rejected_documents()
            self.analyze_data_relationships()
            self.analyze_data_volume()
            self.check_recent_activity()
            
            print("\n" + "="*60)
            print("✅ DETAILED ANALYSIS COMPLETED")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error during detailed analysis: {e}")
            return False
        finally:
            self.close()
        
        return True

def main():
    """Main function to run the detailed analysis"""
    checker = DetailedNeo4jChecker()
    success = checker.run_detailed_analysis()
    
    if success:
        print("\n✅ Detailed analysis completed successfully!")
    else:
        print("\n❌ Detailed analysis failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())