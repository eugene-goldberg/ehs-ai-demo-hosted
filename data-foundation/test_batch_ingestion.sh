#!/bin/bash
# Test the batch ingestion endpoint

LOG_FILE="batch_ingestion_test_$(date +%Y%m%d_%H%M%S).log"
echo "Testing batch ingestion endpoint - $(date)" > "$LOG_FILE"

# Make sure the API is running on port 8005
echo "Checking if API is running on port 8005..." >> "$LOG_FILE"
curl -s http://localhost:8005/health >> "$LOG_FILE" 2>&1

# Test 1: Call batch ingestion endpoint with clear_database=true
echo -e "\n=== Test 1: Batch Ingestion with Database Clear ===" >> "$LOG_FILE"
echo "Calling POST /api/v1/ingest/batch with clear_database=true" >> "$LOG_FILE"

curl -X POST http://localhost:8005/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"clear_database": true}' \
  -w "\nHTTP Status: %{http_code}\n" 2>&1 | tee -a "$LOG_FILE" | jq '.'

# Wait for ingestion to complete
echo -e "\nWaiting for ingestion to complete..." >> "$LOG_FILE"
sleep 30

# Verify data in Neo4j
echo -e "\n=== Verifying Data in Neo4j ===" >> "$LOG_FILE"
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation
source .venv/bin/activate

python3 << 'EOP' >> "$LOG_FILE" 2>&1
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'Neo4j123!'))

try:
    with driver.session() as session:
        # Count nodes
        result = session.run("MATCH (n) RETURN COUNT(n) as count")
        node_count = result.single()['count']
        print(f"Total nodes in database: {node_count}")
        
        # Count by type
        docs = session.run("MATCH (d:Document) RETURN COUNT(d) as count").single()['count']
        bills = session.run("MATCH (b:UtilityBill) RETURN COUNT(b) as count").single()['count']
        water = session.run("MATCH (w:WaterBill) RETURN COUNT(w) as count").single()['count']
        waste = session.run("MATCH (wm:WasteManifest) RETURN COUNT(wm) as count").single()['count']
        
        print(f"Documents: {docs}")
        print(f"Utility Bills: {bills}")
        print(f"Water Bills: {water}")
        print(f"Waste Manifests: {waste}")
        
finally:
    driver.close()
EOP

# Test extraction endpoints to verify data is accessible
echo -e "\n=== Testing Extraction Endpoints ===" >> "$LOG_FILE"

echo -e "\n--- Electrical Consumption ---" >> "$LOG_FILE"
curl -s -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}' | jq '.data.report_data.query_results[0].record_count' >> "$LOG_FILE"

echo -e "\n--- Water Consumption ---" >> "$LOG_FILE"
curl -s -X POST http://localhost:8005/api/v1/extract/water-consumption \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}' | jq '.data.report_data.query_results[0].record_count' >> "$LOG_FILE"

echo -e "\n--- Waste Generation ---" >> "$LOG_FILE"
curl -s -X POST http://localhost:8005/api/v1/extract/waste-generation \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}' | jq '.data.report_data.query_results[0].record_count' >> "$LOG_FILE"

echo -e "\n=== Test completed at $(date) ===" >> "$LOG_FILE"
