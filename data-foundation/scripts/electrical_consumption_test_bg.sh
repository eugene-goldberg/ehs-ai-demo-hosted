#!/bin/bash

LOG_FILE="electrical_consumption_test.log"
OUTPUT_FILE="electrical_consumption_test.txt"

{
    echo "========================================="
    echo "Electrical Consumption Endpoint Test"
    echo "========================================="
    echo "Endpoint: POST /api/v1/extract/electrical-consumption"
    echo "Time: $(date)"
    echo "URL: http://localhost:8005/api/v1/extract/electrical-consumption"
    echo ""
    
    echo "Making API request..."
    
    # Capture both response and curl stats
    RESPONSE=$(curl -X POST \
      http://localhost:8005/api/v1/extract/electrical-consumption \
      -H "Content-Type: application/json" \
      -H "Accept: application/json" \
      -d '{}' \
      -w "CURL_INFO:HTTP_STATUS:%{http_code}:TOTAL_TIME:%{time_total}:SIZE:%{size_download}" \
      -s)
    
    # Extract curl info
    CURL_INFO=$(echo "$RESPONSE" | grep "CURL_INFO:" | tail -1)
    JSON_RESPONSE=$(echo "$RESPONSE" | sed '/CURL_INFO:/d')
    
    # Parse curl info
    HTTP_STATUS=$(echo "$CURL_INFO" | cut -d: -f3)
    TOTAL_TIME=$(echo "$CURL_INFO" | cut -d: -f5)
    SIZE=$(echo "$CURL_INFO" | cut -d: -f7)
    
    echo "Response received:"
    echo "HTTP Status: $HTTP_STATUS"
    echo "Total Time: ${TOTAL_TIME}s"
    echo "Response Size: $SIZE bytes"
    echo ""
    
    # Save raw response to file
    echo "$JSON_RESPONSE" > "$OUTPUT_FILE"
    
    echo "Raw response saved to: $OUTPUT_FILE"
    
    # Format JSON if possible
    if command -v jq &> /dev/null; then
        echo ""
        echo "Formatted JSON response:"
        echo "------------------------"
        echo "$JSON_RESPONSE" | jq '.' 2>/dev/null || echo "Invalid JSON response"
    fi
    
    echo ""
    echo "Test completed at: $(date)"
    echo "========================================="
    
} 2>&1 | tee "$LOG_FILE"
