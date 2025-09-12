#!/usr/bin/env python3
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

print("Connecting to Neo4j at {} with user {}".format(uri, username))

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    print("=== Query 1: All Sites and their site_ids ===")
    with driver.session() as session:
        result = session.run("MATCH (s:Site) RETURN s.name, s.site_id, s.id ORDER BY s.name")
        for record in result:
            print("Name: {}, Site ID: {}, ID: {}".format(record["s.name"], record["s.site_id"], record["s.id"]))
    
    print("\n=== Query 2: Count waste data for each site ===")
    with driver.session() as session:
        result = session.run("MATCH (s:Site)-[:GENERATES_WASTE]->(w:WasteGeneration) RETURN s.site_id, s.name, count(w) as waste_count")
        records = list(result)
        if records:
            for record in records:
                print("Site ID: {}, Name: {}, Waste Count: {}".format(record["s.site_id"], record["s.name"], record["waste_count"]))
        else:
            print("No waste data found for any sites")
    
    print("\n=== Query 3: Check Houston site specifically ===")
    with driver.session() as session:
        result = session.run("MATCH (s:Site) WHERE toLower(s.name) CONTAINS houston OR toLower(s.id) CONTAINS houston OR toLower(s.site_id) CONTAINS houston RETURN s.name, s.site_id, s.id")
        records = list(result)
        if records:
            for record in records:
                print("Houston Site - Name: {}, Site ID: {}, ID: {}".format(record["s.name"], record["s.site_id"], record["s.id"]))
        else:
            print("No sites found containing Houston in name, id, or site_id")
    
    print("\n=== Query 4: Check waste data connected to Houston ===")
    with driver.session() as session:
        result = session.run("MATCH (s:Site)-[:GENERATES_WASTE]->(w:WasteGeneration) WHERE toLower(s.name) CONTAINS houston OR toLower(s.id) CONTAINS houston OR toLower(s.site_id) CONTAINS houston RETURN count(w) as houston_waste_count")
        for record in result:
            print("Houston waste count: {}".format(record["houston_waste_count"]))
    
    print("\n=== Query 5: All Sites and their waste generation connections ===")
    with driver.session() as session:
        result = session.run("MATCH (s:Site) OPTIONAL MATCH (s)-[:GENERATES_WASTE]->(w:WasteGeneration) RETURN s.site_id, s.name, s.id, count(w) as waste_count ORDER BY s.name")
        print("All Sites and their waste counts:")
        for record in result:
            print("Site ID: {}, Name: {}, ID: {}, Waste Count: {}".format(record["s.site_id"], record["s.name"], record["s.id"], record["waste_count"]))

    print("\n=== Query 6: Check if WasteGeneration nodes exist at all ===")
    with driver.session() as session:
        result = session.run("MATCH (w:WasteGeneration) RETURN count(w) as total_waste_nodes")
        for record in result:
            print("Total WasteGeneration nodes in database: {}".format(record["total_waste_nodes"]))

    print("\n=== Query 7: Check for any relationship types involving Sites ===")
    with driver.session() as session:
        result = session.run("MATCH (s:Site)-[r]->(n) RETURN type(r) as relationship_type, labels(n) as target_labels, count(*) as count ORDER BY count DESC")
        print("Relationships from Site nodes:")
        for record in result:
            print("Relationship: {} -> {}, Count: {}".format(record["relationship_type"], record["target_labels"], record["count"]))

finally:
    driver.close()
