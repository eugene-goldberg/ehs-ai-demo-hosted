#!/bin/bash
# Full ingestion and extraction test

LOG_FILE="full_ingestion_extraction_test_$(date +%Y%m%d_%H%M%S).log"
echo "Starting full ingestion and extraction test - $(date)" > "$LOG_FILE"

# Activate virtual environment
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation
source .venv/bin/activate

# Step 1: Clear all Neo4j data
echo -e "\n=== Step 1: Clearing all Neo4j data ===" >> "$LOG_FILE"
python3 << 'EOF' >> "$LOG_FILE" 2>&1
from neo4j import GraphDatabase
import os

uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
username = os.getenv('NEO4J_USERNAME', 'neo4j')
password = os.getenv('NEO4J_PASSWORD', 'Neo4j123!')

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    with driver.session() as session:
        # Delete all nodes and relationships
        result = session.run("MATCH (n) DETACH DELETE n")
        print("Cleared all Neo4j data")
        
        # Verify database is empty
        count = session.run("MATCH (n) RETURN COUNT(n) as count").single()["count"]
        print(f"Node count after clearing: {count}")
except Exception as e:
    print(f"Error clearing Neo4j: {e}")
finally:
    driver.close()
EOF

# Step 2: Run ingestion for all 3 documents
echo -e "\n=== Step 2: Running ingestion for all documents ===" >> "$LOG_FILE"

# Electric bill
echo -e "\n--- Ingesting Electric Bill ---" >> "$LOG_FILE"
curl -X POST http://localhost:8005/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "utility_bill",
    "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/sample_data/electric_bill.pdf"
  }' >> "$LOG_FILE" 2>&1
sleep 5

# Water bill  
echo -e "\n--- Ingesting Water Bill ---" >> "$LOG_FILE"
curl -X POST http://localhost:8005/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "water_bill", 
    "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/sample_data/water_bill.pdf"
  }' >> "$LOG_FILE" 2>&1
sleep 5

# Waste manifest
echo -e "\n--- Ingesting Waste Manifest ---" >> "$LOG_FILE"
curl -X POST http://localhost:8005/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "waste_manifest",
    "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/sample_data/waste_manifest.pdf"
  }' >> "$LOG_FILE" 2>&1
sleep 5

# Step 3: Verify all documents are in Neo4j
echo -e "\n=== Step 3: Verifying data in Neo4j ===" >> "$LOG_FILE"
python3 << 'EOF' >> "$LOG_FILE" 2>&1
from neo4j import GraphDatabase
import os

uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
username = os.getenv('NEO4J_USERNAME', 'neo4j')
password = os.getenv('NEO4J_PASSWORD', 'Neo4j123!')

driver = GraphDatabase.driver(uri, auth=(username, password))

try:
    with driver.session() as session:
        # Check documents
        print("\n--- Documents in Neo4j ---")
        docs = session.run("MATCH (d:Document) RETURN d.id as id, labels(d) as labels, d.type as type").data()
        for doc in docs:
            print(f"Document: {doc['id']}, Labels: {doc['labels']}, Type: {doc['type']}")
        
        # Check electric bill data
        print("\n--- Electric Bill Data ---")
        electric = session.run("""
            MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)
            WHERE 'Utilitybill' IN labels(d) OR 'UtilityBill' IN labels(d)
            RETURN b.total_kwh as kwh, b.total_cost as cost, b.billing_period_start as start, b.billing_period_end as end
        """).data()
        for bill in electric:
            print(f"Electric: {bill['kwh']} kWh, ${bill['cost']}, Period: {bill['start']} to {bill['end']}")
        
        # Check water bill data  
        print("\n--- Water Bill Data ---")
        water = session.run("""
            MATCH (d:Document)-[:EXTRACTED_TO]->(w:WaterBill)
            WHERE 'Waterbill' IN labels(d) OR 'WaterBill' IN labels(d)
            RETURN w.total_gallons as gallons, w.total_cost as cost, w.billing_period_start as start, w.billing_period_end as end
        """).data()
        for bill in water:
            print(f"Water: {bill['gallons']} gallons, ${bill['cost']}, Period: {bill['start']} to {bill['end']}")
        
        # Check waste manifest data
        print("\n--- Waste Manifest Data ---")
        waste = session.run("""
            MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
            MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
            RETURN wm.manifest_number as manifest_no, ws.shipment_date as date, 
                   wi.description as waste_desc, wi.quantity as quantity, wi.unit as unit
        """).data()
        for item in waste:
            print(f"Manifest: {item['manifest_no']}, Date: {item['date']}, Waste: {item['waste_desc']}, Quantity: {item['quantity']} {item['unit']}")
            
except Exception as e:
    print(f"Error verifying data: {e}")
finally:
    driver.close()
EOF

# Step 4: Test extraction endpoints
echo -e "\n=== Step 4: Testing extraction endpoints ===" >> "$LOG_FILE"

# Test electrical consumption
echo -e "\n--- Testing Electrical Consumption ---" >> "$LOG_FILE"
curl -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}' 2>&1 | jq '.data.report_data.query_results[0] | {query, record_count, results: .results[0]}' >> "$LOG_FILE"

# Test water consumption  
echo -e "\n--- Testing Water Consumption ---" >> "$LOG_FILE"
curl -X POST http://localhost:8005/api/v1/extract/water-consumption \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}' 2>&1 | jq '.data.report_data.query_results[0] | {query, record_count, results: .results[0]}' >> "$LOG_FILE"

# Test waste generation
echo -e "\n--- Testing Waste Generation ---" >> "$LOG_FILE"
curl -X POST http://localhost:8005/api/v1/extract/waste-generation \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}' 2>&1 | jq '.data.report_data.query_results[0] | {query, record_count, results: .results[0]}' >> "$LOG_FILE"

echo -e "\n=== Test completed at $(date) ===" >> "$LOG_FILE"
