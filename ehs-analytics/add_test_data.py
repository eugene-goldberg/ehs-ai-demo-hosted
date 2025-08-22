#!/usr/bin/env python3
"""
Add test data to Neo4j for Phase 2 retriever testing
"""

import os
from datetime import datetime, timedelta
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def add_test_data():
    """Add comprehensive test data for EHS retrievers"""
    
    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    print(f"Connected to Neo4j at {uri}")
    
    with driver.session() as session:
        # Clear existing test data (optional)
        print("\nClearing existing test data...")
        session.run("MATCH (n) WHERE n.test_data = true DETACH DELETE n")
        
        # Add Facilities
        print("\nAdding facilities...")
        facilities = [
            {"name": "Main Manufacturing Plant", "location": "Houston, TX", "type": "manufacturing"},
            {"name": "Distribution Center East", "location": "Atlanta, GA", "type": "warehouse"},
            {"name": "Research Lab Alpha", "location": "San Jose, CA", "type": "research"}
        ]
        
        for facility in facilities:
            session.run("""
                CREATE (f:Facility {
                    name: $name,
                    location: $location,
                    type: $type,
                    test_data: true
                })
            """, **facility)
        
        # Add Equipment
        print("Adding equipment...")
        equipment = [
            {"name": "Boiler-01", "type": "boiler", "manufacturer": "Acme", "model": "B-2000"},
            {"name": "HVAC-Main", "type": "hvac", "manufacturer": "CoolTech", "model": "AC-500"},
            {"name": "Compressor-A", "type": "compressor", "manufacturer": "AirFlow", "model": "C-300"}
        ]
        
        for eq in equipment:
            session.run("""
                MATCH (f:Facility {name: 'Main Manufacturing Plant'})
                CREATE (e:Equipment {
                    name: $name,
                    type: $type,
                    manufacturer: $manufacturer,
                    model: $model,
                    test_data: true
                })
                CREATE (f)-[:HAS_EQUIPMENT]->(e)
            """, **eq)
        
        # Add Permits
        print("Adding permits...")
        permits = [
            {"permit_number": "AIR-2024-001", "type": "air_emissions", "status": "active", "expiry_date": "2025-12-31"},
            {"permit_number": "WATER-2024-002", "type": "water_discharge", "status": "active", "expiry_date": "2024-06-30"},
            {"permit_number": "WASTE-2024-003", "type": "hazardous_waste", "status": "expired", "expiry_date": "2024-01-31"}
        ]
        
        for permit in permits:
            session.run("""
                MATCH (f:Facility {name: 'Main Manufacturing Plant'})
                CREATE (p:Permit {
                    permit_number: $permit_number,
                    type: $type,
                    status: $status,
                    expiry_date: date($expiry_date),
                    test_data: true
                })
                CREATE (f)-[:HAS_PERMIT]->(p)
            """, **permit)
        
        # Add Water Bills with proper relationships
        print("Adding water bills...")
        base_date = datetime.now() - timedelta(days=90)
        
        for i in range(6):
            bill_date = base_date + timedelta(days=30*i)
            consumption = 1000 + (i * 100)  # Increasing consumption
            
            session.run("""
                MATCH (f:Facility {name: 'Main Manufacturing Plant'})
                CREATE (w:WaterBill:UtilityBill {
                    billing_period: $billing_period,
                    consumption_amount: $consumption,
                    unit: 'gallons',
                    cost: $cost,
                    utility_type: 'water',
                    test_data: true
                })
                CREATE (f)-[:HAS_UTILITY_BILL]->(w)
            """, 
            billing_period=bill_date.strftime("%Y-%m"),
            consumption=consumption,
            cost=consumption * 0.005)  # $0.005 per gallon
        
        # Add Emissions data
        print("Adding emissions data...")
        emissions = [
            {"source": "Boiler-01", "pollutant": "CO2", "amount": 1500, "unit": "kg"},
            {"source": "Boiler-01", "pollutant": "NOx", "amount": 25, "unit": "kg"},
            {"source": "Main Manufacturing Plant", "pollutant": "CO2", "amount": 5000, "unit": "kg"}
        ]
        
        for emission in emissions:
            session.run("""
                CREATE (e:Emission {
                    source: $source,
                    pollutant: $pollutant,
                    amount: $amount,
                    unit: $unit,
                    date: date(),
                    test_data: true
                })
            """, **emission)
        
        # Create vector embeddings for documents (mock)
        print("Adding document chunks for vector search...")
        documents = [
            {"content": "The main manufacturing plant water consumption increased by 15% in Q1 2024", "type": "report"},
            {"content": "Equipment efficiency report shows boiler operating at 85% capacity", "type": "analysis"},
            {"content": "Air emissions permit requires quarterly monitoring and reporting", "type": "compliance"}
        ]
        
        for i, doc in enumerate(documents):
            session.run("""
                CREATE (d:DocumentChunk {
                    chunk_id: $chunk_id,
                    content: $content,
                    document_type: $type,
                    embedding: $embedding,
                    test_data: true
                })
            """, 
            chunk_id=f"chunk_{i}",
            content=doc["content"],
            type=doc["type"],
            embedding=[0.1] * 1536)  # Mock embedding vector
        
        print("\n✅ Test data added successfully!")
        
        # Verify data
        result = session.run("MATCH (n) WHERE n.test_data = true RETURN labels(n)[0] as label, count(n) as count")
        print("\nTest data summary:")
        for record in result:
            print(f"  {record['label']}: {record['count']}")
    
    driver.close()

if __name__ == "__main__":
    add_test_data()