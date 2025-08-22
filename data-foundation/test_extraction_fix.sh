#!/bin/bash
# Start API and test the electrical consumption endpoint after fixing case sensitivity

LOG_FILE="extraction_test_$(date +%Y%m%d_%H%M%S).log"
echo "Starting API and testing extraction endpoint - $(date)" > "$LOG_FILE"

# Start the API server on port 8005 in the background
echo "Starting API server on port 8005..." >> "$LOG_FILE"
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation
source .venv/bin/activate

# Use the correct module path - the FastAPI app is in backend/src/ehs_extraction_api.py
export PYTHONPATH="/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/src:$PYTHONPATH"
python3 -m uvicorn backend.src.ehs_extraction_api:app --host 0.0.0.0 --port 8005 >> "$LOG_FILE" 2>&1 &
API_PID=$!

# Wait for API to start
echo "Waiting for API to start..." >> "$LOG_FILE"
sleep 15

# Test electrical consumption endpoint using the correct endpoint
echo -e "\n=== Testing Electrical Consumption Endpoint (Direct) ===" >> "$LOG_FILE"
curl -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -d '{
    "output_format": "json",
    "include_emissions": true,
    "include_cost_analysis": true
  }' 2>&1 | tee -a "$LOG_FILE" | jq '.' || echo "jq parse error - checking raw response"

# Test using custom endpoint with electrical-consumption query type
echo -e "\n=== Testing Custom Endpoint with electrical-consumption query type ===" >> "$LOG_FILE"
curl -X POST "http://localhost:8005/api/v1/extract/custom?query_type=electrical-consumption&output_format=json" \
  -H "Content-Type: application/json" 2>&1 | tee -a "$LOG_FILE" | jq '.' || echo "jq parse error - checking raw response"

# Also capture the raw response for detailed analysis
echo -e "\n=== Raw Response (Custom endpoint) ===" >> "$LOG_FILE"
curl -X POST "http://localhost:8005/api/v1/extract/custom?query_type=electrical-consumption&output_format=json" \
  -H "Content-Type: application/json" 2>&1 >> "$LOG_FILE"

# Test health endpoint
echo -e "\n=== Testing Health Endpoint ===" >> "$LOG_FILE"
curl -X GET http://localhost:8005/health 2>&1 | tee -a "$LOG_FILE" | jq '.' || echo "Health check failed"

# Check if we got actual data (not 0 records) from either endpoint
echo -e "\n=== Checking for actual data ===" >> "$LOG_FILE"
RECORD_COUNT=$(curl -s -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -d '{
    "output_format": "json",
    "include_emissions": true,
    "include_cost_analysis": true
  }' | jq '.metadata.total_records // 0' 2>/dev/null || echo "0")

echo "Record count: $RECORD_COUNT" >> "$LOG_FILE"

if [ "$RECORD_COUNT" -gt 0 ]; then
    echo "SUCCESS: Found $RECORD_COUNT records!" >> "$LOG_FILE"
    echo "Extracting consumption data..." >> "$LOG_FILE"
    curl -s -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
      -H "Content-Type: application/json" \
      -d '{
        "output_format": "json",
        "include_emissions": true,
        "include_cost_analysis": true
      }' | jq '.data.report_data' >> "$LOG_FILE" 2>/dev/null || echo "Could not parse consumption data"
else
    echo "INFO: Record count is $RECORD_COUNT - checking if API is working correctly" >> "$LOG_FILE"
    
    # Test if we can get valid response structure even if no data
    RESPONSE_STATUS=$(curl -s -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
      -H "Content-Type: application/json" \
      -d '{
        "output_format": "json",
        "include_emissions": true,
        "include_cost_analysis": true
      }' | jq '.status // "unknown"' 2>/dev/null)
    
    echo "Response status: $RESPONSE_STATUS" >> "$LOG_FILE"
    
    if [ "$RESPONSE_STATUS" = '"success"' ]; then
        echo "API working correctly, but no data found in database" >> "$LOG_FILE"
    else
        echo "API may have issues - status: $RESPONSE_STATUS" >> "$LOG_FILE"
    fi
fi

echo -e "\n=== Test completed at $(date) ===" >> "$LOG_FILE"

# Keep the API running
echo "API server is running with PID: $API_PID" >> "$LOG_FILE"
echo "To stop it later, run: kill $API_PID" >> "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo "API running on: http://localhost:8005"
echo "API docs available at: http://localhost:8005/api/docs"
