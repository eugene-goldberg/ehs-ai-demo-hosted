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
    
    # Use a better approach to capture response and status
    local response=$(curl -s -w "\n%{http_code}" \
                         -X POST \
                         -H "Content-Type: application/json" \
                         -H "Accept: application/json" \
                         -d '{}' \
                         "$url")
    
    # Split response and status
    local http_status=$(echo "$response" | tail -n1)
    local response_body=$(echo "$response" | head -n -1)
    
    log_with_timestamp "HTTP Status Code: $http_status"
    
    if [ "$http_status" = "200" ]; then
        echo -e "${GREEN}✅ SUCCESS: $endpoint_name responded successfully${NC}" | tee -a "$OUTPUT_FILE"
        
        # Try to format with jq if response is valid JSON
        if echo "$response_body" | jq . >/dev/null 2>&1; then
            log_with_timestamp "Response (formatted JSON):"
            echo "$response_body" | jq . | tee -a "$OUTPUT_FILE"
            
            # Extract record count from different possible locations
            local total_records=$(echo "$response_body" | jq -r '.metadata.total_records // .data.metadata.total_records // "N/A"' 2>/dev/null)
            local data_length=$(echo "$response_body" | jq -r '.data | length // "N/A"' 2>/dev/null)
            local processing_time=$(echo "$response_body" | jq -r '.processing_time // "N/A"' 2>/dev/null)
            
            if [ "$total_records" != "N/A" ] && [ "$total_records" != "null" ]; then
                log_with_timestamp "Total records processed: $total_records"
            fi
            
            if [ "$processing_time" != "N/A" ] && [ "$processing_time" != "null" ]; then
                log_with_timestamp "Processing time: ${processing_time}s"
            fi
            
            # Extract status and message
            local status=$(echo "$response_body" | jq -r '.status // "N/A"' 2>/dev/null)
            local message=$(echo "$response_body" | jq -r '.message // "N/A"' 2>/dev/null)
            
            if [ "$status" != "N/A" ]; then
                log_with_timestamp "Status: $status"
            fi
            if [ "$message" != "N/A" ]; then
                log_with_timestamp "Message: $message"
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
    
    # Also show a quick summary of what each endpoint returned
    echo "" | tee -a "$OUTPUT_FILE"
    log_with_timestamp "ENDPOINT DATA SUMMARY:"
    
    local electrical_records=$(grep -A 20 "Electrical Consumption Extraction" "$OUTPUT_FILE" | grep "Total records processed:" | head -1 | cut -d: -f3 | xargs || echo "0")
    local water_records=$(grep -A 20 "Water Consumption Extraction" "$OUTPUT_FILE" | grep "Total records processed:" | head -1 | cut -d: -f3 | xargs || echo "0")
    local waste_records=$(grep -A 20 "Waste Generation Extraction" "$OUTPUT_FILE" | grep "Total records processed:" | head -1 | cut -d: -f3 | xargs || echo "0")
    
    log_with_timestamp "- Electrical consumption: $electrical_records records"
    log_with_timestamp "- Water consumption: $water_records records" 
    log_with_timestamp "- Waste generation: $waste_records records"
    
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
}

# Run the test
main "$@"
