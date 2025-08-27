#!/usr/bin/env python3
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import uuid
import requests

load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
)

test_doc_id = str(uuid.uuid4())
print(f"Test Document ID: {test_doc_id}")

# Create a test document
with driver.session(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as session:
    result = session.run("""
        CREATE (d:Document {
            id: $id,
            fileName: $fileName,
            total_amount: $total_amount,
            status: $status,
            created_at: $created_at
        })
        RETURN d.id as doc_id
    """, {
        "id": test_doc_id,
        "fileName": f"simple_test_{test_doc_id[:8]}.pdf",
        "total_amount": 100.0,
        "status": "processed",
        "created_at": "2025-08-25T20:20:00"
    })
    
    record = result.single()
    if record:
        print(f"✅ Document created: {record['doc_id']}")
    else:
        print("❌ Failed to create document")
        exit(1)

# Verify document exists
with driver.session(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as session:
    result = session.run("""
        MATCH (d:Document {id: $document_id})
        RETURN d.id as id, d.fileName as file_name, d.status as status,
               d.created_at as created_at, d.total_amount as total_amount
    """, {"document_id": test_doc_id})
    
    record = result.single()
    if record:
        print(f"✅ Document found: {record}")
    else:
        print("❌ Document not found")

# Test API call with minimal data
print("\nTesting API call...")
response = requests.post(f"http://localhost:8000/api/v1/prorating/process/{test_doc_id}", 
                        headers={"Content-Type": "application/json"},
                        json={
                            "document_id": test_doc_id,
                            "method": "headcount",
                            "facility_info": [{
                                "facility_id": "test_facility",
                                "name": "Test Facility",
                                "headcount": 10
                            }]
                        })

print(f"Response status: {response.status_code}")
print(f"Response body: {response.text}")

# Cleanup
with driver.session(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as session:
    session.run("MATCH (d:Document {id: $document_id}) DELETE d", {"document_id": test_doc_id})
    print("✅ Cleaned up test document")

driver.close()
