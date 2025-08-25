#!/bin/bash

# Final Comprehensive EHS API Endpoints Test Script
# Tests all three extraction endpoints on port 8005 with complete output

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

# Function to test an endpoint and capture full response
test_endpoint() {
    local endpoint_name=$1
    local url=$2
    
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
    log_with_timestamp "TESTING: $endpoint_name"
    log_with_timestamp "URL: $url"
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
    
    log_with_timestamp "Sending POST request with empty JSON body: '{}'"
    
    # Use curl to get full response with timing
    local start_time=$(date +%s.%N)
    
    local response=$(curl -s \
                         -w "CURL_STATUS:%{http_code};CURL_TIME:%{time_total}" \
                         -X POST \
                         -H "Content-Type: application/json" \
                         -H "Accept: application/json" \
                         -d '{}' \
                         "$url")
    
    local end_time=$(date +%s.%N)
    local request_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")
    
    # Extract curl metrics
    local curl_metrics=$(echo "$response" | grep -o "CURL_STATUS:[0-9]*;CURL_TIME:[0-9.]*" | tail -1)
    local http_status=$(echo "$curl_metrics" | sed -n 's/.*CURL_STATUS:\([0-9]*\).*/\1/p')
    local curl_time=$(echo "$curl_metrics" | sed -n 's/.*CURL_TIME:\([0-9.]*\).*/\1/p')
    
    # Get response body (everything before the curl metrics)
    local response_body=$(echo "$response" | sed 's/CURL_STATUS:[0-9]*;CURL_TIME:[0-9.]*$//')
    
    log_with_timestamp "HTTP Status Code: $http_status"
    log_with_timestamp "Request Time: ${curl_time}s"
    
    if [ "$http_status" = "200" ]; then
        echo -e "${GREEN}✅ SUCCESS: $endpoint_name responded successfully${NC}" | tee -a "$OUTPUT_FILE"
        
        # Check if response is valid JSON
        if echo "$response_body" | jq . >/dev/null 2>&1; then
            log_with_timestamp "Response (formatted JSON):"
            echo "$response_body" | jq . | tee -a "$OUTPUT_FILE"
            
            # Extract specific metrics
            local status=$(echo "$response_body" | jq -r '.status // "unknown"' 2>/dev/null)
            local message=$(echo "$response_body" | jq -r '.message // "unknown"' 2>/dev/null)
            local total_records=$(echo "$response_body" | jq -r '.metadata.total_records // "unknown"' 2>/dev/null)
            local processing_time=$(echo "$response_body" | jq -r '.processing_time // "unknown"' 2>/dev/null)
            local successful_queries=$(echo "$response_body" | jq -r '.metadata.successful_queries // "unknown"' 2>/dev/null)
            local total_queries=$(echo "$response_body" | jq -r '.metadata.total_queries // "unknown"' 2>/dev/null)
            
            echo "" | tee -a "$OUTPUT_FILE"
            log_with_timestamp "RESPONSE SUMMARY:"
            log_with_timestamp "- Status: $status"
            log_with_timestamp "- Message: $message"
            log_with_timestamp "- Total Records: $total_records"
            log_with_timestamp "- Processing Time: ${processing_time}s"
            log_with_timestamp "- Successful Queries: $successful_queries/$total_queries"
            
        else
            log_with_timestamp "Response (raw text):"
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
    log_with_timestamp "=== COMPREHENSIVE EHS API ENDPOINTS TEST ==="
    log_with_timestamp "Testing all three EHS extraction endpoints on port 8005"
    log_with_timestamp "Test Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "" | tee -a "$OUTPUT_FILE"
    
    # Check server health
    log_with_timestamp "Checking server health..."
    if curl -s --connect-timeout 5 http://localhost:8005/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is accessible on port 8005${NC}" | tee -a "$OUTPUT_FILE"
    else
        echo -e "${YELLOW}⚠️  Warning: Health check failed, but continuing with tests${NC}" | tee -a "$OUTPUT_FILE"
    fi
    echo "" | tee -a "$OUTPUT_FILE"
    
    # Test all three endpoints
    log_with_timestamp "Starting endpoint tests..."
    echo "" | tee -a "$OUTPUT_FILE"
    
    test_endpoint "Electrical Consumption Extraction" \
                  "http://localhost:8005/api/v1/extract/electrical-consumption"
    
    test_endpoint "Water Consumption Extraction" \
                  "http://localhost:8005/api/v1/extract/water-consumption"
    
    test_endpoint "Waste Generation Extraction" \
                  "http://localhost:8005/api/v1/extract/waste-generation"
    
    # Final Summary
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
    log_with_timestamp "FINAL TEST SUMMARY"
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
    
    local success_count=$(grep -c "✅ SUCCESS" "$OUTPUT_FILE" || echo "0")
    local failure_count=$(grep -c "❌ FAILED" "$OUTPUT_FILE" || echo "0")
    local total_endpoints=3
    
    log_with_timestamp "Total Endpoints Tested: $total_endpoints"
    log_with_timestamp "Successful Responses: $success_count"
    log_with_timestamp "Failed Responses: $failure_count"
    
    # Extract record counts from successful responses
    if [ "$success_count" -gt 0 ]; then
        echo "" | tee -a "$OUTPUT_FILE"
        log_with_timestamp "DATA EXTRACTION RESULTS:"
        
        # Electrical
        local electrical_records=$(grep -A 10 "Electrical Consumption" "$OUTPUT_FILE" | grep "Total Records:" | head -1 | cut -d: -f3 | xargs || echo "0")
        # Water
        local water_records=$(grep -A 10 "Water Consumption" "$OUTPUT_FILE" | grep "Total Records:" | head -1 | cut -d: -f3 | xargs || echo "0")
        # Waste
        local waste_records=$(grep -A 10 "Waste Generation" "$OUTPUT_FILE" | grep "Total Records:" | head -1 | cut -d: -f3 | xargs || echo "0")
        
        log_with_timestamp "- Electrical Consumption: $electrical_records records"
        log_with_timestamp "- Water Consumption: $water_records records" 
        log_with_timestamp "- Waste Generation: $waste_records records"
    fi
    
    echo "" | tee -a "$OUTPUT_FILE"
    
    if [ "$success_count" -eq "$total_endpoints" ]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED! All $total_endpoints endpoints are working correctly.${NC}" | tee -a "$OUTPUT_FILE"
    elif [ "$success_count" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  PARTIAL SUCCESS: $success_count/$total_endpoints endpoints working, $failure_count failed${NC}" | tee -a "$OUTPUT_FILE"
    else
        echo -e "${RED}💥 ALL TESTS FAILED! Check server status and configuration.${NC}" | tee -a "$OUTPUT_FILE"
    fi
    
    echo "" | tee -a "$OUTPUT_FILE"
    log_with_timestamp "Test completed. Complete results saved to: $OUTPUT_FILE"
    log_with_timestamp "=== END OF TEST ==="
    echo -e "${BLUE}========================================${NC}" | tee -a "$OUTPUT_FILE"
}

# Run the test
main "$@"
