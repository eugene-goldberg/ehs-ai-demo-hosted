#!/usr/bin/env python3
"""Enhanced Neo4j connectivity test with migration status check"""

import os
import sys
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test Neo4j connection and query for Equipment, Permit entities, and Migration status"""
    
    # Try multiple common passwords for Neo4j
    passwords_to_try = [
        os.getenv('NEO4J_PASSWORD', 'your_neo4j_password'),
        'neo4j123',  # Updated password
        'neo4j',
        'password',
        'test',
        'admin',
        '123456'
    ]
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    database = os.getenv('NEO4J_DATABASE', 'ehs')
    
    print(f"Enhanced Neo4j Database Status Check")
    print(f"Time: {datetime.now()}")
    print(f"URI: {uri}")
    print(f"Username: {username}")
    print(f"Database: {database}")
    print("=" * 60)
    
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
        print("\n" + "=" * 60)
        print("DATABASE SCHEMA AND CONTENT ANALYSIS")
        print("=" * 60)
        
        with driver.session(database=database) as session:
            # 1. Check what labels exist
            print("\n1. AVAILABLE NODE LABELS:")
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            print(f"   Found {len(labels)} node types:")
            for label in labels:
                print(f"   - {label}")
            
            # 2. Check what relationship types exist
            print("\n2. AVAILABLE RELATIONSHIP TYPES:")
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]
            print(f"   Found {len(rel_types)} relationship types:")
            for rel_type in rel_types:
                print(f"   - {rel_type}")
            
            # 3. Count Equipment nodes
            print("\n3. EQUIPMENT ENTITIES:")
            result = session.run("MATCH (e:Equipment) RETURN count(e) as count")
            equipment_count = result.single()["count"]
            print(f"   Total Equipment nodes: {equipment_count}")
            
            if equipment_count > 0:
                result = session.run("MATCH (e:Equipment) RETURN e LIMIT 5")
                print("   Sample Equipment nodes:")
                for record in result:
                    node = record["e"]
                    props = dict(node.items())
                    print(f"     - {props}")
            
            # 4. Count Permit nodes
            print("\n4. PERMIT ENTITIES:")
            result = session.run("MATCH (p:Permit) RETURN count(p) as count")
            permit_count = result.single()["count"]
            print(f"   Total Permit nodes: {permit_count}")
            
            if permit_count > 0:
                result = session.run("MATCH (p:Permit) RETURN p LIMIT 5")
                print("   Sample Permit nodes:")
                for record in result:
                    node = record["p"]
                    props = dict(node.items())
                    print(f"     - {props}")
            
            # 5. Check Migration nodes (NEW)
            print("\n5. MIGRATION STATUS:")
            try:
                result = session.run("MATCH (m:Migration) RETURN count(m) as count")
                migration_count = result.single()["count"]
                print(f"   Total Migration nodes: {migration_count}")
                
                if migration_count > 0:
                    result = session.run("MATCH (m:Migration) RETURN m ORDER BY m.applied_at DESC")
                    print("   Applied migrations:")
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
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()["count"]
                print(f"   {label}: {count} nodes")
            
            # 7. Relationship inventory
            print("\n7. RELATIONSHIP INVENTORY:")
            if rel_types:
                for rel_type in rel_types:
                    result = session.run(f"MATCH ()-[r:{rel_type}]-() RETURN count(r) as count")
                    count = result.single()["count"]
                    print(f"   {rel_type}: {count} relationships")
            else:
                print("   No relationships found in database")
            
            # 8. Database constraints and indexes
            print("\n8. DATABASE CONSTRAINTS:")
            try:
                result = session.run("SHOW CONSTRAINTS")
                constraints = list(result)
                if constraints:
                    for constraint in constraints:
                        print(f"   - {dict(constraint)}")
                else:
                    print("   No constraints defined")
            except Exception as e:
                print(f"   Could not retrieve constraints: {str(e)}")
            
            print("\n9. DATABASE INDEXES:")
            try:
                result = session.run("SHOW INDEXES")
                indexes = list(result)
                if indexes:
                    for index in indexes:
                        print(f"   - {dict(index)}")
                else:
                    print("   No indexes defined")
            except Exception as e:
                print(f"   Could not retrieve indexes: {str(e)}")
                
        print("\n" + "=" * 60)
        print("DATABASE ANALYSIS COMPLETE")
        print("=" * 60)
                
    except Exception as e:
        print(f"❌ Error querying database: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if driver:
            driver.close()
            print("\n✅ Database connection closed")

if __name__ == "__main__":
    test_neo4j_connection()
