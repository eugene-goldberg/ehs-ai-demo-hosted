#!/usr/bin/env python3
"""Corrected Neo4j connectivity test with proper password and database handling"""

import os
import sys
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test Neo4j connection with correct password and handle database selection"""
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
    
    print(f"Neo4j Database Status Check - Corrected Version")
    print(f"Time: {datetime.now()}")
    print(f"URI: {uri}")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password)}")
    print("=" * 60)
    
    driver = None
    
    try:
        print(f"Connecting to Neo4j...")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # First try to connect to default database
        print("\n1. TESTING CONNECTION TO DEFAULT DATABASE:")
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_result = result.single()
            if test_result:
                print(f"✅ Connected successfully to default database")
                
                # Check available databases
                print("\n2. AVAILABLE DATABASES:")
                try:
                    result = session.run("SHOW DATABASES")
                    databases = list(result)
                    for db in databases:
                        print(f"   - {dict(db)}")
                except Exception as e:
                    print(f"   Could not list databases: {str(e)}")
        
        # Try to connect to 'ehs' database specifically
        print("\n3. TESTING CONNECTION TO 'EHS' DATABASE:")
        try:
            with driver.session(database='ehs') as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    print(f"✅ Connected successfully to 'ehs' database")
                    database_to_use = 'ehs'
                else:
                    print(f"❌ Could not connect to 'ehs' database")
                    database_to_use = None
        except Exception as e:
            print(f"❌ Failed to connect to 'ehs' database: {str(e)}")
            print("   Will use default database instead")
            database_to_use = None
        
        # Perform analysis on the appropriate database
        print("\n" + "=" * 60)
        print(f"ANALYZING DATABASE: {'ehs' if database_to_use else 'default (neo4j)'}")
        print("=" * 60)
        
        with driver.session(database=database_to_use) as session:
            # 1. Check what labels exist
            print("\n1. AVAILABLE NODE LABELS:")
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            print(f"   Found {len(labels)} node types:")
            for label in sorted(labels):
                print(f"   - {label}")
            
            # 2. Check what relationship types exist
            print("\n2. AVAILABLE RELATIONSHIP TYPES:")
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]
            print(f"   Found {len(rel_types)} relationship types:")
            for rel_type in sorted(rel_types):
                print(f"   - {rel_type}")
            
            # 3. Count Equipment nodes
            print("\n3. EQUIPMENT ENTITIES:")
            try:
                result = session.run("MATCH (e:Equipment) RETURN count(e) as count")
                equipment_count = result.single()["count"]
                print(f"   Total Equipment nodes: {equipment_count}")
                
                if equipment_count > 0:
                    result = session.run("MATCH (e:Equipment) RETURN e LIMIT 3")
                    print("   Sample Equipment nodes:")
                    for record in result:
                        node = record["e"]
                        props = dict(node.items())
                        print(f"     - {props}")
            except Exception as e:
                print(f"   Equipment query failed: {str(e)}")
            
            # 4. Count Permit nodes
            print("\n4. PERMIT ENTITIES:")
            try:
                result = session.run("MATCH (p:Permit) RETURN count(p) as count")
                permit_count = result.single()["count"]
                print(f"   Total Permit nodes: {permit_count}")
                
                if permit_count > 0:
                    result = session.run("MATCH (p:Permit) RETURN p LIMIT 3")
                    print("   Sample Permit nodes:")
                    for record in result:
                        node = record["p"]
                        props = dict(node.items())
                        print(f"     - {props}")
            except Exception as e:
                print(f"   Permit query failed: {str(e)}")
            
            # 5. Check Migration nodes
            print("\n5. MIGRATION STATUS:")
            try:
                result = session.run("MATCH (m:Migration) RETURN count(m) as count")
                migration_count = result.single()["count"]
                print(f"   Total Migration nodes: {migration_count}")
                
                if migration_count > 0:
                    result = session.run("MATCH (m:Migration) RETURN m ORDER BY m.applied_at DESC LIMIT 10")
                    print("   Recent migrations:")
                    for record in result:
                        node = record["m"]
                        props = dict(node.items())
                        print(f"     - {props}")
                else:
                    print("   No migration tracking nodes found")
            except Exception as e:
                print(f"   Migration check failed: {str(e)}")
            
            # 6. All node types with counts
            print("\n6. COMPLETE NODE INVENTORY:")
            total_nodes = 0
            for label in sorted(labels):
                try:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    count = result.single()["count"]
                    total_nodes += count
                    print(f"   {label}: {count} nodes")
                except Exception as e:
                    print(f"   {label}: Error counting - {str(e)}")
            print(f"   TOTAL: {total_nodes} nodes")
            
            # 7. Relationship inventory
            print("\n7. RELATIONSHIP INVENTORY:")
            total_relationships = 0
            if rel_types:
                for rel_type in sorted(rel_types):
                    try:
                        result = session.run(f"MATCH ()-[r:{rel_type}]-() RETURN count(r) as count")
                        count = result.single()["count"]
                        total_relationships += count
                        print(f"   {rel_type}: {count} relationships")
                    except Exception as e:
                        print(f"   {rel_type}: Error counting - {str(e)}")
                print(f"   TOTAL: {total_relationships} relationships")
            else:
                print("   No relationships found in database")
            
            # 8. Sample some data to understand structure
            print("\n8. SAMPLE DATA EXPLORATION:")
            if total_nodes > 0:
                try:
                    result = session.run("MATCH (n) RETURN n LIMIT 5")
                    print("   Sample nodes from database:")
                    for i, record in enumerate(result, 1):
                        node = record["n"]
                        labels_str = ":".join(node.labels)
                        props = dict(node.items())
                        print(f"     {i}. ({labels_str}) {props}")
                except Exception as e:
                    print(f"   Could not retrieve sample data: {str(e)}")
                
        print("\n" + "=" * 60)
        print("DATABASE ANALYSIS COMPLETE")
        print("=" * 60)
                
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if driver:
            driver.close()
            print("\n✅ Database connection closed")

if __name__ == "__main__":
    test_neo4j_connection()
