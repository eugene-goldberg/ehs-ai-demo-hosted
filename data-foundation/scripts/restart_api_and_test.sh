#!/bin/bash
# Restart API and test batch endpoint

LOG_FILE="api_restart_test_$(date +%Y%m%d_%H%M%S).log"
echo "Restarting API and testing batch endpoint - $(date)" > "$LOG_FILE"

cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation

# Kill existing API process
echo "Killing existing API process..." >> "$LOG_FILE"
pkill -f "ehs_extraction_api.py" || echo "No existing process found" >> "$LOG_FILE"
sleep 2

# Start API in the background
echo "Starting API on port 8005..." >> "$LOG_FILE"
source .venv/bin/activate
export PYTHONPATH=/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation
export NEO4J_PASSWORD=EhsAI2024!
export NEO4J_USERNAME=neo4j
export NEO4J_URI=bolt://localhost:7687

python3 -m uvicorn backend.src.ehs_extraction_api:app --host 0.0.0.0 --port 8005 >> api_output.log 2>&1 &
API_PID=$!
echo "API started with PID: $API_PID" >> "$LOG_FILE"

# Wait for API to start
echo "Waiting for API to start..." >> "$LOG_FILE"
for i in {1..30}; do
    if curl -s http://localhost:8005/health > /dev/null; then
        echo "API is ready after $i seconds" >> "$LOG_FILE"
        break
    fi
    sleep 1
done

# Check available endpoints
echo -e "\n=== Checking API Documentation ===" >> "$LOG_FILE"
curl -s http://localhost:8005/api/docs | grep -o "batch" | head -5 >> "$LOG_FILE" 2>&1

# Test batch endpoint
echo -e "\n=== Testing Batch Ingestion Endpoint ===" >> "$LOG_FILE"
RESPONSE=$(curl -X POST http://localhost:8005/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"clear_database": false}' \
  -w "\nHTTP_CODE:%{http_code}" 2>&1)

HTTP_CODE=$(echo "$RESPONSE" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
echo "HTTP Status Code: $HTTP_CODE" >> "$LOG_FILE"

# Show response
echo -e "\nResponse:" >> "$LOG_FILE"
echo "$RESPONSE" | sed 's/HTTP_CODE:[0-9]*//g' | jq '.' >> "$LOG_FILE" 2>&1 || echo "$RESPONSE" >> "$LOG_FILE"

# Keep API running
echo -e "\nAPI is running with PID: $API_PID" >> "$LOG_FILE"
echo "To stop it: kill $API_PID" >> "$LOG_FILE"

echo -e "\n=== Test completed at $(date) ===" >> "$LOG_FILE"
