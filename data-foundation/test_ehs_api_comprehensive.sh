#!/bin/bash

# Comprehensive EHS API Test Script
# Tests all extraction endpoints on the EHS API
# Author: Test Runner Agent
# Date: $(date)

set -e  # Exit on any error

# Configuration
API_BASE_URL_8001="http://localhost:8001"
API_BASE_URL_8005="http://localhost:8005"
OUTPUT_FILE="api_extraction_test_results.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$OUTPUT_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}" | tee -a "$OUTPUT_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $1${NC}" | tee -a "$OUTPUT_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $1${NC}" | tee -a "$OUTPUT_FILE"
}

# Function to test API connectivity
test_connectivity() {
    local base_url=$1
    local port=$(echo $base_url | grep -o '[0-9]*$')
    
    log "Testing connectivity to $base_url..."
    
    if curl -s --connect-timeout 5 "$base_url/health" > /dev/null 2>&1; then
        log_success "API is reachable on port $port"
        return 0
    else
        log_error "API is not reachable on port $port"
        return 1
    fi
}

# Function to format JSON output
format_json() {
    if command -v jq &> /dev/null; then
        echo "$1" | jq .
    else
        echo "$1"
    fi
}

# Function to test an endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4
    
    echo "" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "Testing: $description" >> "$OUTPUT_FILE"
    echo "Method: $method" >> "$OUTPUT_FILE"
    echo "Endpoint: $endpoint" >> "$OUTPUT_FILE"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    
    log "Testing $description..."
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}\nTIME_TOTAL:%{time_total}" \
            -H "Content-Type: application/json" \
            "$API_BASE_URL$endpoint" 2>&1)
    else
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}\nTIME_TOTAL:%{time_total}" \
            -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_BASE_URL$endpoint" 2>&1)
    fi
    
    # Extract HTTP status and time from response
    http_status=$(echo "$response" | grep "HTTP_STATUS:" | cut -d: -f2)
    time_total=$(echo "$response" | grep "TIME_TOTAL:" | cut -d: -f2)
    response_body=$(echo "$response" | sed '/HTTP_STATUS:/d' | sed '/TIME_TOTAL:/d')
    
    echo "Request Data:" >> "$OUTPUT_FILE"
    if [ -n "$data" ]; then
        echo "$data" | format_json >> "$OUTPUT_FILE"
    else
        echo "No request body (GET request)" >> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"
    
    echo "Response Status: $http_status" >> "$OUTPUT_FILE"
    echo "Response Time: ${time_total}s" >> "$OUTPUT_FILE"
    echo "Response Body:" >> "$OUTPUT_FILE"
    
    if [ "$http_status" = "200" ]; then
        log_success "$description - Status: $http_status, Time: ${time_total}s"
        format_json "$response_body" >> "$OUTPUT_FILE"
    elif [ "$http_status" = "422" ]; then
        log_warning "$description - Validation Error: $http_status, Time: ${time_total}s"
        format_json "$response_body" >> "$OUTPUT_FILE"
    else
        log_error "$description - Status: $http_status, Time: ${time_total}s"
        echo "$response_body" >> "$OUTPUT_FILE"
    fi
    
    echo "" >> "$OUTPUT_FILE"
    echo "----------------------------------------" >> "$OUTPUT_FILE"
}

# Initialize output file
echo "=================================================================" > "$OUTPUT_FILE"
echo "EHS API Comprehensive Test Results" >> "$OUTPUT_FILE"
echo "Started: $TIMESTAMP" >> "$OUTPUT_FILE"
echo "=================================================================" >> "$OUTPUT_FILE"

log "Starting comprehensive EHS API test suite..."

# Test connectivity on both ports
API_BASE_URL=""
if test_connectivity "$API_BASE_URL_8005"; then
    API_BASE_URL="$API_BASE_URL_8005"
    log_success "Using API on port 8005"
elif test_connectivity "$API_BASE_URL_8001"; then
    API_BASE_URL="$API_BASE_URL_8001"
    log_success "Using API on port 8001"
else
    log_error "API is not reachable on either port 8001 or 8005"
    echo "API connectivity test failed. Please ensure the EHS API server is running." >> "$OUTPUT_FILE"
    exit 1
fi

# Test 1: Health Check
test_endpoint "GET" "/health" "" "Health Check"

# Test 2: Get Query Types
test_endpoint "GET" "/api/v1/query-types" "" "Get Available Query Types"

# Test 3: Extract Facility Emissions
facility_emissions_data='{
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/facility-emissions" "$facility_emissions_data" "Extract Facility Emissions"

# Test 4: Extract Utility Consumption (Electrical)
utility_consumption_data='{
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31"
  },
  "include_emissions": true,
  "include_cost_analysis": true,
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/utility-consumption" "$utility_consumption_data" "Extract Utility Consumption"

# Test 5: Extract Water Consumption
water_consumption_data='{
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31"
  },
  "include_meter_details": true,
  "include_emissions": true,
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/water-consumption" "$water_consumption_data" "Extract Water Consumption"

# Test 6: Extract Waste Generation
waste_generation_data='{
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-03-31"
  },
  "include_disposal_details": true,
  "include_transport_details": true,
  "include_emissions": true,
  "hazardous_only": false,
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/waste-generation" "$waste_generation_data" "Extract Waste Generation"

# Test 7: Extract Compliance Status
compliance_status_data='{
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/compliance-status" "$compliance_status_data" "Extract Compliance Status"

# Test 8: Extract Trend Analysis
trend_analysis_data='{
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/trend-analysis" "$trend_analysis_data" "Extract Trend Analysis"

# Test 9: Custom Extraction with specific facility filter
custom_extraction_data='{
  "query_type": "facility_emissions",
  "facility_filter": {
    "facility_name": "Manufacturing Plant"
  },
  "date_range": {
    "start_date": "2024-06-01",
    "end_date": "2024-06-30"
  },
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/custom" "$custom_extraction_data" "Custom Extraction with Facility Filter"

# Test 10: Invalid Date Range (Error Test)
invalid_date_data='{
  "date_range": {
    "start_date": "2024-12-31",
    "end_date": "2024-01-01"
  },
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/utility-consumption" "$invalid_date_data" "Invalid Date Range Test (Error Case)"

# Test 11: Empty Request Body (Error Test)
test_endpoint "POST" "/api/v1/extract/water-consumption" "{}" "Empty Request Body Test (Error Case)"

# Test 12: Large Date Range Test
large_range_data='{
  "date_range": {
    "start_date": "2020-01-01",
    "end_date": "2024-12-31"
  },
  "output_format": "json"
}'
test_endpoint "POST" "/api/v1/extract/facility-emissions" "$large_range_data" "Large Date Range Test"

# Test 13: Text Output Format Test
text_format_data='{
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  },
  "output_format": "txt"
}'
test_endpoint "POST" "/api/v1/extract/utility-consumption" "$text_format_data" "Text Output Format Test"

# Summary
echo "" >> "$OUTPUT_FILE"
echo "=================================================================" >> "$OUTPUT_FILE"
echo "Test Suite Completed: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "=================================================================" >> "$OUTPUT_FILE"

# Count results
total_tests=$(grep -c "Testing:" "$OUTPUT_FILE" || echo "0")
successful_tests=$(grep -c "✓" "$OUTPUT_FILE" || echo "0")
failed_tests=$(grep -c "✗" "$OUTPUT_FILE" || echo "0")
warning_tests=$(grep -c "⚠" "$OUTPUT_FILE" || echo "0")

echo "Summary:" >> "$OUTPUT_FILE"
echo "  Total Tests: $total_tests" >> "$OUTPUT_FILE"
echo "  Successful: $successful_tests" >> "$OUTPUT_FILE"
echo "  Failed: $failed_tests" >> "$OUTPUT_FILE"
echo "  Warnings: $warning_tests" >> "$OUTPUT_FILE"

log "Test suite completed!"
log_success "Results saved to: $OUTPUT_FILE"
log "Summary: $total_tests total, $successful_tests successful, $failed_tests failed, $warning_tests warnings"

# Display final summary
echo ""
echo "================================================================="
echo "FINAL SUMMARY"
echo "================================================================="
echo "Total Tests: $total_tests"
echo "Successful: $successful_tests"
echo "Failed: $failed_tests"
echo "Warnings: $warning_tests"
echo "Results file: $OUTPUT_FILE"
echo "================================================================="

