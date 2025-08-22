#!/usr/bin/env python3
"""
Verify test data in Neo4j
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_test_data():
    """Check what data exists in Neo4j"""
    
    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    print(f"Connected to Neo4j at {uri}")
    
    with driver.session() as session:
        # Count all nodes
        print("\n=== NODE COUNT BY LABEL ===")
        result = session.run("""
            MATCH (n)
            WITH labels(n) as labels, n
            UNWIND labels as label
            RETURN label, count(n) as count
            ORDER BY label
        """)
        
        total_nodes = 0
        for record in result:
            print(f"{record['label']}: {record['count']}")
            total_nodes += record['count']
        
        print(f"\nTotal nodes: {total_nodes}")
        
        # Check specific test data
        print("\n=== TEST DATA CHECK ===")
        
        # Facilities
        result = session.run("MATCH (f:Facility) RETURN f.name as name, f.test_data as test")
        facilities = list(result)
        print(f"\nFacilities ({len(facilities)}):")
        for f in facilities:
            print(f"  - {f['name']} (test_data: {f['test']})")
        
        # Equipment
        result = session.run("MATCH (e:Equipment) RETURN e.name as name, e.test_data as test")
        equipment = list(result)
        print(f"\nEquipment ({len(equipment)}):")
        for e in equipment:
            print(f"  - {e['name']} (test_data: {e['test']})")
        
        # Permits
        result = session.run("MATCH (p:Permit) RETURN p.permit_number as num, p.test_data as test")
        permits = list(result)
        print(f"\nPermits ({len(permits)}):")
        for p in permits:
            print(f"  - {p['num']} (test_data: {p['test']})")
        
        # Water Bills
        result = session.run("MATCH (w:WaterBill) RETURN count(w) as count")
        water_bills = result.single()['count']
        print(f"\nWater Bills: {water_bills}")
        
        # Check relationships
        print("\n=== RELATIONSHIPS ===")
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY type
        """)
        for record in result:
            print(f"{record['type']}: {record['count']}")
    
    driver.close()

if __name__ == "__main__":
    verify_test_data()