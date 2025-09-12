#!/usr/bin/env python3
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

print("=== HOUSTON WASTE DATA INVESTIGATION ===\n")

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    print("=== All Sites and their waste data ===")
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Site) 
            OPTIONAL MATCH (s)-[:GENERATES_WASTE]->(w:WasteGeneration) 
            RETURN s.name, s.site_id, s.id, count(w) as waste_count 
            ORDER BY s.name
        """)
        for record in result:
            print("Name: {}, Site ID: {}, ID: {}, Waste Count: {}".format(
                record["s.name"], record["s.site_id"], record["s.id"], record["waste_count"]))
    
    print("\n=== Detailed Houston site information ===")
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Site) 
            WHERE toLower(s.name) CONTAINS houston OR toLower(toString(s.id)) CONTAINS houston OR toLower(toString(s.site_id)) CONTAINS houston
            OPTIONAL MATCH (s)-[:GENERATES_WASTE]->(w:WasteGeneration)
            RETURN s.name, s.site_id, s.id, count(w) as waste_count, 
                   collect(w.waste_type)[0..5] as sample_waste_types
        """)
        for record in result:
            print("Houston Site Details:")
            print("  Name: {}".format(record["s.name"]))
            print("  Site ID: {}".format(record["s.site_id"]))
            print("  ID: {}".format(record["s.id"]))
            print("  Waste Count: {}".format(record["waste_count"]))
            print("  Sample Waste Types: {}".format(record["sample_waste_types"]))
    
    print("\n=== Sample Houston waste data ===")
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Site)-[:GENERATES_WASTE]->(w:WasteGeneration)
            WHERE toLower(s.name) CONTAINS houston
            RETURN w.waste_type, w.quantity, w.unit, w.date
            LIMIT 10
        """)
        print("First 10 Houston waste records:")
        for record in result:
            print("  Type: {}, Quantity: {}, Unit: {}, Date: {}".format(
                record["w.waste_type"], record["w.quantity"], record["w.unit"], record["w.date"]))

    print("\n=== Total waste generation nodes ===")
    with driver.session() as session:
        result = session.run("MATCH (w:WasteGeneration) RETURN count(w) as total_count")
        for record in result:
            print("Total WasteGeneration nodes: {}".format(record["total_count"]))

finally:
    driver.close()
