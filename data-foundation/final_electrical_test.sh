#!/bin/bash

echo "==========================================="
echo "Final Electrical Consumption Endpoint Test"
echo "==========================================="
echo "Endpoint: POST /api/v1/extract/electrical-consumption"
echo "URL: http://localhost:8005/api/v1/extract/electrical-consumption"
echo "Time: $(date)"
echo ""

echo "Making API request with curl..."
curl -X POST \
  http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{}' \
  -o electrical_consumption_test.txt \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\nSize: %{size_download} bytes\n" \
  -s

echo ""
echo "Response saved to: electrical_consumption_test.txt"
echo ""
echo "Response summary:"
if [ -f electrical_consumption_test.txt ] && [ -s electrical_consumption_test.txt ]; then
    # Extract key fields from the response
    echo "Status: $(cat electrical_consumption_test.txt | jq -r '.status // "unknown"' 2>/dev/null)"
    echo "Message: $(cat electrical_consumption_test.txt | jq -r '.message // "unknown"' 2>/dev/null)"
    echo "Query Type: $(cat electrical_consumption_test.txt | jq -r '.data.query_type // "unknown"' 2>/dev/null)"
    echo "Total Queries: $(cat electrical_consumption_test.txt | jq -r '.data.metadata.total_queries // "unknown"' 2>/dev/null)"
    echo "Processing Time: $(cat electrical_consumption_test.txt | jq -r '.data.processing_time // "unknown"' 2>/dev/null)s"
    echo ""
    echo "File size: $(wc -c < electrical_consumption_test.txt) bytes"
else
    echo "ERROR: No response received or empty file"
fi

echo ""
echo "Test completed at: $(date)"
echo "==========================================="
