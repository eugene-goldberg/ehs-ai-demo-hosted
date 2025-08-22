#!/bin/bash

# Test script for electrical consumption endpoint
echo "Testing electrical consumption endpoint..."
echo "Endpoint: POST /api/v1/extract/electrical-consumption"
echo "Time: $(date)"
echo "========================================="

# Make the API call
curl -X POST \
  http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{}' \
  -w "\n\nHTTP Status: %{http_code}\nTotal Time: %{time_total}s\n" \
  -s | tee electrical_consumption_test.txt

echo ""
echo "Response saved to electrical_consumption_test.txt"

# Format the JSON output if jq is available
if command -v jq &> /dev/null; then
  echo ""
  echo "Formatted JSON response:"
  echo "========================"
  # Extract just the JSON part (before the HTTP status info)
  head -n -3 electrical_consumption_test.txt | jq '.' 2>/dev/null || echo "Response is not valid JSON"
else
  echo "jq not available - raw response saved to file"
fi

echo ""
echo "Test completed at $(date)"
