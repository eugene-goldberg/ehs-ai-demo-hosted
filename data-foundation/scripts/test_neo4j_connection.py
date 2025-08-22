#!/usr/bin/env python3
"""
Test Neo4j connection for EHS AI Platform
"""

from neo4j import GraphDatabase
import sys

# Connection details
uri = "bolt://localhost:7687"
username = "neo4j"
password = "EhsAI2024!"

def test_connection():
    """Test Neo4j connection and create initial indexes"""
    driver = None
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record["test"] == 1:
                print("✅ Successfully connected to Neo4j!")
            
            # Get database info
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            for record in result:
                print(f"📊 Neo4j {record['edition']} - Version: {record['versions'][0]}")
            
            # Create initial indexes for EHS schema
            print("\n🔧 Creating EHS schema indexes...")
            
            indexes = [
                "CREATE INDEX facility_id IF NOT EXISTS FOR (f:Facility) ON (f.id)",
                "CREATE INDEX facility_name IF NOT EXISTS FOR (f:Facility) ON (f.name)",
                "CREATE INDEX utility_bill_id IF NOT EXISTS FOR (u:UtilityBill) ON (u.id)",
                "CREATE INDEX permit_id IF NOT EXISTS FOR (p:Permit) ON (p.id)",
                "CREATE INDEX equipment_id IF NOT EXISTS FOR (e:Equipment) ON (e.id)",
                "CREATE INDEX emission_id IF NOT EXISTS FOR (e:Emission) ON (e.id)",
                "CREATE INDEX utility_bill_time IF NOT EXISTS FOR (u:UtilityBill) ON (u.period_start, u.period_end)",
                "CREATE INDEX emission_date IF NOT EXISTS FOR (e:Emission) ON (e.date)"
            ]
            
            for index_query in indexes:
                session.run(index_query)
                print(f"  ✓ {index_query.split('FOR')[1].split('ON')[0].strip()}")
            
            print("\n✅ All indexes created successfully!")
            
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        sys.exit(1)
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    test_connection()