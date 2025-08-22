#!/bin/bash

# Comprehensive EHS API Endpoints Test Script
# Tests all three extraction endpoints on port 8005

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Output file
OUTPUT_FILE="all_endpoints_test_results.txt"

# Clear previous results
> "$OUTPUT_FILE"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$OUTPUT_FILE"
}

# Function to test an endpoint
test_endpoint() {
    local endpoint_name=$1
    local url=$2
    local data_type=$3
    
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
    log_with_timestamp "TESTING: $endpoint_name"
    log_with_timestamp "URL: $url"
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
    
    # Make the curl request and capture response
    log_with_timestamp "Sending POST request with empty JSON body..."
    
    # Capture both response and HTTP status
    local response_file=$(mktemp)
    local status_file=$(mktemp)
    
    curl -s -w "%{http_code}" \
         -X POST \
         -H "Content-Type: application/json" \
         -H "Accept: application/json" \
         -d '{}' \
         "$url" > "$response_file" 2>&1
    
    local http_status=$(tail -c 3 "$response_file")
    local response_body=$(head -c -3 "$response_file")
    
    log_with_timestamp "HTTP Status Code: $http_status"
    
    if [ "$http_status" = "200" ]; then
        echo -e "${GREEN}✅ SUCCESS: $endpoint_name responded successfully${NC}" | tee -a "$OUTPUT_FILE"
        
        # Try to format with jq if response is valid JSON
        if echo "$response_body" | jq . >/dev/null 2>&1; then
            log_with_timestamp "Response (formatted JSON):"
            echo "$response_body" | jq . | tee -a "$OUTPUT_FILE"
            
            # Extract record count if available
            local record_count=$(echo "$response_body" | jq -r '.data | length' 2>/dev/null || echo "N/A")
            if [ "$record_count" != "N/A" ] && [ "$record_count" != "null" ]; then
                log_with_timestamp "Records returned: $record_count"
            else
                # Try alternative count methods
                record_count=$(echo "$response_body" | jq -r '. | length' 2>/dev/null || echo "N/A")
                if [ "$record_count" != "N/A" ] && [ "$record_count" != "null" ]; then
                    log_with_timestamp "Records returned: $record_count"
                fi
            fi
        else
            log_with_timestamp "Response (raw):"
            echo "$response_body" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo -e "${RED}❌ FAILED: $endpoint_name returned HTTP $http_status${NC}" | tee -a "$OUTPUT_FILE"
        log_with_timestamp "Error response:"
        echo "$response_body" | tee -a "$OUTPUT_FILE"
    fi
    
    # Cleanup temp files
    rm -f "$response_file"
    
    echo "" | tee -a "$OUTPUT_FILE"
}

# Main execution
main() {
    log_with_timestamp "Starting comprehensive EHS API endpoints test"
    log_with_timestamp "Testing against server on port 8005"
    echo "" | tee -a "$OUTPUT_FILE"
    
    # Check if server is responding
    log_with_timestamp "Checking if server is accessible..."
    if curl -s --connect-timeout 5 http://localhost:8005/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is accessible on port 8005${NC}" | tee -a "$OUTPUT_FILE"
    else
        echo -e "${YELLOW}⚠️  Warning: Server may not be running on port 8005${NC}" | tee -a "$OUTPUT_FILE"
        log_with_timestamp "Continuing with tests anyway..."
    fi
    echo "" | tee -a "$OUTPUT_FILE"
    
    # Test all three endpoints
    test_endpoint "Electrical Consumption Extraction" \
                  "http://localhost:8005/api/v1/extract/electrical-consumption" \
                  "electrical"
    
    test_endpoint "Water Consumption Extraction" \
                  "http://localhost:8005/api/v1/extract/water-consumption" \
                  "water"
    
    test_endpoint "Waste Generation Extraction" \
                  "http://localhost:8005/api/v1/extract/waste-generation" \
                  "waste"
    
    # Summary
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
    log_with_timestamp "TEST SUMMARY"
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
    
    # Count successes and failures
    local success_count=$(grep -c "✅ SUCCESS" "$OUTPUT_FILE" || echo "0")
    local failure_count=$(grep -c "❌ FAILED" "$OUTPUT_FILE" || echo "0")
    
    log_with_timestamp "Endpoints tested: 3"
    log_with_timestamp "Successful responses: $success_count"
    log_with_timestamp "Failed responses: $failure_count"
    
    if [ "$success_count" -eq 3 ]; then
        echo -e "${GREEN}🎉 All endpoints are working correctly!${NC}" | tee -a "$OUTPUT_FILE"
    elif [ "$success_count" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Some endpoints are working, but $failure_count failed${NC}" | tee -a "$OUTPUT_FILE"
    else
        echo -e "${RED}💥 All endpoints failed - check server status${NC}" | tee -a "$OUTPUT_FILE"
    fi
    
    log_with_timestamp "Test completed. Full results saved to: $OUTPUT_FILE"
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
}

# Run the test
main "$@"
