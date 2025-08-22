#!/usr/bin/env python3
"""Direct Neo4j connectivity and entity query test"""

import os
import sys
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test Neo4j connection and query for Equipment and Permit entities"""
    
    # Try multiple common passwords for Neo4j
    passwords_to_try = [
        os.getenv('NEO4J_PASSWORD', 'your_neo4j_password'),
        'neo4j',
        'password',
        'test',
        'admin',
        '123456'
    ]
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    database = os.getenv('NEO4J_DATABASE', 'ehs')
    
    print(f"Testing Neo4j connection to {uri}")
    print(f"Username: {username}")
    print(f"Database: {database}")
    print(f"Time: {datetime.now()}")
    print("-" * 50)
    
    driver = None
    success = False
    
    for password in passwords_to_try:
        if password == 'your_neo4j_password':
            continue
            
        try:
            print(f"Trying password: {'*' * len(password)}")
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    print(f"✅ Connected successfully with password: {'*' * len(password)}")
                    success = True
                    break
                    
        except Exception as e:
            print(f"❌ Failed with password {'*' * len(password)}: {str(e)}")
            if driver:
                driver.close()
            continue
    
    if not success:
        print("❌ Could not connect to Neo4j with any common passwords")
        return
    
    try:
        print("\n" + "=" * 50)
        print("QUERYING NEO4J DATABASE ENTITIES")
        print("=" * 50)
        
        with driver.session(database=database) as session:
            # Check what labels exist
            print("\n1. Available Node Labels:")
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            for label in labels:
                print(f"   - {label}")
            
            # Check what relationship types exist
            print("\n2. Available Relationship Types:")
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]
            for rel_type in rel_types:
                print(f"   - {rel_type}")
            
            # Count Equipment nodes
            print("\n3. Equipment Entities:")
            result = session.run("MATCH (e:Equipment) RETURN count(e) as count")
            equipment_count = result.single()["count"]
            print(f"   Total Equipment nodes: {equipment_count}")
            
            if equipment_count > 0:
                # Sample Equipment nodes
                result = session.run("MATCH (e:Equipment) RETURN e LIMIT 5")
                print("   Sample Equipment nodes:")
                for record in result:
                    node = record["e"]
                    print(f"     - ID: {node.get('id', 'N/A')}, Name: {node.get('name', 'N/A')}")
            
            # Count Permit nodes
            print("\n4. Permit Entities:")
            result = session.run("MATCH (p:Permit) RETURN count(p) as count")
            permit_count = result.single()["count"]
            print(f"   Total Permit nodes: {permit_count}")
            
            if permit_count > 0:
                # Sample Permit nodes
                result = session.run("MATCH (p:Permit) RETURN p LIMIT 5")
                print("   Sample Permit nodes:")
                for record in result:
                    node = record["p"]
                    print(f"     - ID: {node.get('id', 'N/A')}, Number: {node.get('permit_number', 'N/A')}")
            
            # Check for any other entities
            print("\n5. All Node Types with Counts:")
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()["count"]
                print(f"   {label}: {count} nodes")
            
            # Check relationships
            print("\n6. Relationship Counts:")
            for rel_type in rel_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]-() RETURN count(r) as count")
                count = result.single()["count"]
                print(f"   {rel_type}: {count} relationships")
                
    except Exception as e:
        print(f"❌ Error querying database: {str(e)}")
    
    finally:
        if driver:
            driver.close()
            print("\n✅ Database connection closed")

if __name__ == "__main__":
    test_neo4j_connection()
