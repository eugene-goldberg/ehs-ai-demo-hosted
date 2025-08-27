#!/usr/bin/env python3

import requests
from neo4j import GraphDatabase

# Get the document ID from Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "EhsAI2024!"))
doc_id = None

with driver.session() as session:
    result = session.run("""
        MATCH (d:Document)
        WHERE 'Electricitybill' IN labels(d) OR 'ElectricityBill' IN labels(d)
        RETURN d.id as id
        LIMIT 1
    """)
    record = result.single()
    if record:
        doc_id = record['id']
        print(f"Found electricity bill document: {doc_id}")

driver.close()

if doc_id:
    # Call the prorating API
    url = f"http://localhost:8000/api/v1/prorating/process/{doc_id}"
    payload = {
        "document_id": doc_id,
        "facility_info": [
            {"facility_id": "facility_001", "percentage": 60.0},
            {"facility_id": "facility_002", "percentage": 40.0}
        ],
        "method": "custom"
    }
    
    print(f"\nCalling prorating API: {url}")
    print(f"Payload: {payload}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"\nResponse status: {response.status_code}")
        print(f"Response body: {response.json()}")
    except Exception as e:
        print(f"Error calling API: {e}")
else:
    print("No electricity bill documents found in database")