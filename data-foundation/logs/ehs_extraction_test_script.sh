#!/bin/bash

# EHS API Extraction Test Script
# Tests all three extraction endpoints with comprehensive output capture
# Author: Claude Code
# Date: $(date)

OUTPUT_FILE="ehs_extraction_test_results.txt"
API_BASE_URL="http://localhost:8005/api/v1/extract"

# Function to print separator
print_separator() {
    echo "=================================================================" >> "$OUTPUT_FILE"
    echo "=================================================================" 
}

# Function to print section header
print_header() {
    local title="$1"
    echo "" >> "$OUTPUT_FILE"
    print_separator
    echo "TEST: $title" >> "$OUTPUT_FILE"
    echo "TIMESTAMP: $(date)" >> "$OUTPUT_FILE"
    print_separator
    echo "" >> "$OUTPUT_FILE"
    
    echo ""
    echo "Testing: $title"
    echo "Timestamp: $(date)"
}

# Function to test endpoint
test_endpoint() {
    local endpoint="$1"
    local description="$2"
    
    print_header "$description"
    
    echo "Endpoint: $endpoint" >> "$OUTPUT_FILE"
    echo "Request: POST $endpoint" >> "$OUTPUT_FILE"
    echo "Payload: {}" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    echo "Making request to: $endpoint"
    
    # Make the curl request and capture response
    local response_file=$(mktemp)
    local status_code=$(curl -s -w "%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d '{}' \
        "$endpoint" \
        -o "$response_file")
    
    echo "HTTP Status Code: $status_code" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    if [ "$status_code" = "200" ]; then
        echo "✅ SUCCESS - Status: $status_code"
        echo "RESPONSE (formatted with jq):" >> "$OUTPUT_FILE"
        if command -v jq &> /dev/null; then
            cat "$response_file" | jq '.' >> "$OUTPUT_FILE" 2>/dev/null || {
                echo "Raw response (jq formatting failed):" >> "$OUTPUT_FILE"
                cat "$response_file" >> "$OUTPUT_FILE"
            }
        else
            echo "Raw response (jq not available):" >> "$OUTPUT_FILE"
            cat "$response_file" >> "$OUTPUT_FILE"
        fi
    else
        echo "❌ FAILED - Status: $status_code"
        echo "ERROR RESPONSE:" >> "$OUTPUT_FILE"
        cat "$response_file" >> "$OUTPUT_FILE"
    fi
    
    echo "" >> "$OUTPUT_FILE"
    echo "---End of $description test---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Cleanup
    rm -f "$response_file"
    
    # Brief pause between requests
    sleep 2
}

# Initialize output file
echo "EHS API EXTRACTION ENDPOINT TEST RESULTS" > "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "API Base URL: $API_BASE_URL" >> "$OUTPUT_FILE"
echo "Test Script: $0" >> "$OUTPUT_FILE"
print_separator

echo "Starting EHS API extraction endpoint tests..."
echo "Results will be saved to: $OUTPUT_FILE"

# Test 1: Utility Consumption (Electrical)
test_endpoint "$API_BASE_URL/utility-consumption" "Utility Consumption Extraction (Electrical Data)"

# Test 2: Water Consumption
test_endpoint "$API_BASE_URL/water-consumption" "Water Consumption Extraction"

# Test 3: Waste Generation
test_endpoint "$API_BASE_URL/waste-generation" "Waste Generation Extraction"

# Summary
print_header "TEST SUMMARY"
echo "All extraction endpoint tests completed." >> "$OUTPUT_FILE"
echo "Total endpoints tested: 3" >> "$OUTPUT_FILE"
echo "- POST /api/v1/extract/utility-consumption" >> "$OUTPUT_FILE"
echo "- POST /api/v1/extract/water-consumption" >> "$OUTPUT_FILE"
echo "- POST /api/v1/extract/waste-generation" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Test completed at: $(date)" >> "$OUTPUT_FILE"

print_separator

echo ""
echo "Test script completed successfully!"
echo "Results saved to: $(pwd)/$OUTPUT_FILE"
echo "You can examine the results with: cat $OUTPUT_FILE"

