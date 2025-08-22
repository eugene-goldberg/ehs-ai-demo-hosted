#!/usr/bin/env python3
"""Check existing facilities in Neo4j"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Connect to Neo4j
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

print(f"Connected to Neo4j at {uri}")

with driver.session() as session:
    # List all facilities
    result = session.run("MATCH (f:Facility) RETURN f.name as name, f.location as location, f.type as type ORDER BY f.name")
    facilities = list(result)
    
    print(f"\nFound {len(facilities)} facilities:")
    for f in facilities:
        print(f"  - {f['name']} ({f['type']}) at {f['location']}")
    
    # Check if Apex Manufacturing exists
    result = session.run("MATCH (f:Facility {name: 'Apex Manufacturing - Plant A'}) RETURN f")
    apex = list(result)
    
    if apex:
        print("\n✅ Apex Manufacturing - Plant A exists")
    else:
        print("\n❌ Apex Manufacturing - Plant A not found")

driver.close()