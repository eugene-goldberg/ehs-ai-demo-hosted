#!/bin/bash
# Test the batch ingestion endpoint with correct credentials

LOG_FILE="batch_ingestion_final_test_$(date +%Y%m%d_%H%M%S).log"
echo "Testing batch ingestion endpoint - $(date)" > "$LOG_FILE"

cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation

# First, test the script directly to see if it runs
echo -e "\n=== Test 1: Run ingestion script directly ===" >> "$LOG_FILE"
source .venv/bin/activate
export NEO4J_PASSWORD=EhsAI2024!
export NEO4J_USERNAME=neo4j
export NEO4J_URI=bolt://localhost:7687

# Run the script in the background and capture output
python3 scripts/ingest_all_documents.py > direct_ingestion_output.txt 2>&1 &
SCRIPT_PID=$!

# Wait for script to complete (max 2 minutes)
TIMEOUT=120
ELAPSED=0
while kill -0 $SCRIPT_PID 2>/dev/null && [ $ELAPSED -lt $TIMEOUT ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "Waiting for script... ($ELAPSED seconds)" >> "$LOG_FILE"
done

if kill -0 $SCRIPT_PID 2>/dev/null; then
    echo "Script still running after timeout, killing it..." >> "$LOG_FILE"
    kill $SCRIPT_PID
fi

echo -e "\n--- Direct Script Output ---" >> "$LOG_FILE"
cat direct_ingestion_output.txt >> "$LOG_FILE"

# Test 2: Verify data in Neo4j with correct password
echo -e "\n=== Test 2: Verify Data in Neo4j ===" >> "$LOG_FILE"
python3 << 'PYTHONEOF' >> "$LOG_FILE" 2>&1
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'EhsAI2024!'))

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
        
        # Get sample data
        print("\n--- Sample Electric Bill Data ---")
        electric_data = session.run("""
            MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)
            WHERE 'Utilitybill' IN labels(d) OR 'UtilityBill' IN labels(d)
            RETURN b.total_kwh as kwh, b.total_cost as cost
            LIMIT 1
        """).data()
        for record in electric_data:
            print(f"kWh: {record['kwh']}, Cost: ${record['cost']}")
            
finally:
    driver.close()
PYTHONEOF

# Test 3: Call batch ingestion API endpoint
echo -e "\n=== Test 3: Batch Ingestion API Endpoint ===" >> "$LOG_FILE"
echo "Calling POST /api/v1/ingest/batch" >> "$LOG_FILE"

# Make the API call and save full response
API_RESPONSE=$(curl -X POST http://localhost:8005/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"clear_database": true}' \
  -w "\nHTTP_CODE:%{http_code}" 2>&1)

# Extract HTTP code
HTTP_CODE=$(echo "$API_RESPONSE" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
RESPONSE_BODY=$(echo "$API_RESPONSE" | sed 's/HTTP_CODE:[0-9]*//g')

echo "HTTP Status Code: $HTTP_CODE" >> "$LOG_FILE"
echo "Response Body:" >> "$LOG_FILE"
echo "$RESPONSE_BODY" >> "$LOG_FILE"

# Try to parse as JSON if possible
echo -e "\n--- Parsed Response ---" >> "$LOG_FILE"
echo "$RESPONSE_BODY" | jq '.' >> "$LOG_FILE" 2>&1 || echo "Could not parse as JSON" >> "$LOG_FILE"

# Test 4: Verify extraction endpoints work
echo -e "\n=== Test 4: Testing Extraction Endpoints ===" >> "$LOG_FILE"

echo -e "\n--- Electrical Consumption ---" >> "$LOG_FILE"
ELECTRIC_RESPONSE=$(curl -s -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}')
echo "$ELECTRIC_RESPONSE" | jq '.data.report_data.query_results[0] | {record_count, results: .results[0]}' >> "$LOG_FILE" 2>&1

echo -e "\n--- Water Consumption ---" >> "$LOG_FILE"
WATER_RESPONSE=$(curl -s -X POST http://localhost:8005/api/v1/extract/water-consumption \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}')
echo "$WATER_RESPONSE" | jq '.data.report_data.query_results[0] | {record_count, results: .results[0]}' >> "$LOG_FILE" 2>&1

echo -e "\n--- Waste Generation ---" >> "$LOG_FILE"
WASTE_RESPONSE=$(curl -s -X POST http://localhost:8005/api/v1/extract/waste-generation \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}')
echo "$WASTE_RESPONSE" | jq '.data.report_data.query_results[0] | {record_count, results: .results[0]}' >> "$LOG_FILE" 2>&1

echo -e "\n=== Test completed at $(date) ===" >> "$LOG_FILE"

# Clean up
rm -f direct_ingestion_output.txt
