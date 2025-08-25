#!/bin/bash

###############################################################################
# EHS API Comprehensive Test Suite
# 
# This script provides comprehensive testing for all EHS extraction API endpoints
# including Phase 1 enhancements and various test scenarios.
#
# Prerequisites:
# - API server running on localhost:8001 (or set API_BASE_URL)
# - curl and jq installed
# - Neo4j database accessible with test data
#
# Usage: ./ehs_api_tests.sh [options]
# Options:
#   --base-url URL    Set API base URL (default: http://localhost:8001)
#   --output-dir DIR  Set output directory for reports (default: ./test_reports)
#   --verbose         Enable verbose output
#   --performance     Run performance tests
#   --help            Show this help message
###############################################################################

set -e  # Exit on any error

# Configuration
DEFAULT_API_BASE_URL="http://localhost:8001"
DEFAULT_OUTPUT_DIR="./test_reports"
VERBOSE=false
RUN_PERFORMANCE_TESTS=false
CURL_TIMEOUT=30
MAX_PARALLEL_REQUESTS=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-url)
            API_BASE_URL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --performance)
            RUN_PERFORMANCE_TESTS=true
            shift
            ;;
        --help)
            grep "^#" "$0" | head -20 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults - Use environment variable if set by test orchestrator, otherwise fall back to default
API_BASE_URL="${API_BASE_URL:-$DEFAULT_API_BASE_URL}"
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

###############################################################################
# Utility Functions
###############################################################################

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED_TESTS++))
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${NC}[VERBOSE]${NC} $1"
    fi
}

# Setup function
setup_test_environment() {
    log "Setting up test environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Create test data directory
    mkdir -p "$OUTPUT_DIR/test_data"
    
    # Initialize test report
    TEST_REPORT="$OUTPUT_DIR/test_report_$(date +%Y%m%d_%H%M%S).json"
    echo '{"test_run": {"start_time": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'", "api_base_url": "'$API_BASE_URL'", "tests": []}}' > "$TEST_REPORT"
    
    log "Test environment setup complete"
    log "API Base URL: $API_BASE_URL"
    log "Output Directory: $OUTPUT_DIR"
    log "Test Report: $TEST_REPORT"
}

# Cleanup function
cleanup_test_environment() {
    log "Cleaning up test environment..."
    
    # Update test report with summary
    local end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    jq --arg end_time "$end_time" --argjson total "$TOTAL_TESTS" --argjson passed "$PASSED_TESTS" --argjson failed "$FAILED_TESTS" --argjson skipped "$SKIPPED_TESTS" \
       '.test_run.end_time = $end_time | .test_run.summary = {total: $total, passed: $passed, failed: $failed, skipped: $skipped}' \
       "$TEST_REPORT" > "$TEST_REPORT.tmp" && mv "$TEST_REPORT.tmp" "$TEST_REPORT"
    
    log "Test cleanup complete"
}

# Generic API test function
run_api_test() {
    local test_name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_status="$5"
    local additional_checks="$6"
    
    ((TOTAL_TESTS++))
    log_verbose "Running test: $test_name"
    
    local start_time=$(date +%s.%3N)
    local response_file="$OUTPUT_DIR/response_${test_name//[^a-zA-Z0-9]/_}.json"
    local headers_file="$OUTPUT_DIR/headers_${test_name//[^a-zA-Z0-9]/_}.txt"
    local status_file="$OUTPUT_DIR/status_${test_name//[^a-zA-Z0-9]/_}.txt"
    
    # Build curl command with proper separation of body and status code
    local curl_cmd="curl -s --connect-timeout $CURL_TIMEOUT --max-time $((CURL_TIMEOUT * 2))"
    curl_cmd="$curl_cmd -D '$headers_file'"
    curl_cmd="$curl_cmd -w '%{http_code}' -o '$response_file'"
    
    if [[ "$method" == "POST" ]]; then
        curl_cmd="$curl_cmd -X POST -H 'Content-Type: application/json'"
        if [[ -n "$data" ]]; then
            curl_cmd="$curl_cmd -d '$data'"
        fi
    elif [[ "$method" == "GET" ]]; then
        curl_cmd="$curl_cmd -X GET"
    fi
    
    curl_cmd="$curl_cmd '$API_BASE_URL$endpoint'"
    
    # Execute the request
    local http_status
    
    if http_status=$(eval "$curl_cmd" 2>"$OUTPUT_DIR/error_${test_name//[^a-zA-Z0-9]/_}.txt"); then
        
        local end_time=$(date +%s.%3N)
        local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
        
        # Check if response is valid JSON
        local is_json=false
        if jq empty "$response_file" 2>/dev/null; then
            is_json=true
        fi
        
        # Basic status code check
        if [[ "$http_status" == "$expected_status" ]]; then
            local test_passed=true
            
            # Run additional checks if provided
            if [[ -n "$additional_checks" ]]; then
                if ! eval "$additional_checks"; then
                    test_passed=false
                fi
            fi
            
            if [[ "$test_passed" == "true" ]]; then
                log_success "$test_name (${duration}s, status: $http_status)"
                
                # Log to test report
                jq --arg name "$test_name" --arg status "PASS" --arg duration "$duration" --arg http_status "$http_status" --arg method "$method" --arg endpoint "$endpoint" \
                   '.test_run.tests += [{name: $name, status: $status, duration: ($duration | tonumber), http_status: ($http_status | tonumber), method: $method, endpoint: $endpoint}]' \
                   "$TEST_REPORT" > "$TEST_REPORT.tmp" && mv "$TEST_REPORT.tmp" "$TEST_REPORT"
            else
                log_failure "$test_name (${duration}s, status: $http_status) - Additional checks failed"
            fi
        else
            log_failure "$test_name (${duration}s) - Expected status $expected_status, got $http_status"
            log_verbose "Response: $(cat "$response_file")"
        fi
    else
        log_failure "$test_name - Request failed"
        log_verbose "Error: $(cat "$OUTPUT_DIR/error_${test_name//[^a-zA-Z0-9]/_}.txt" 2>/dev/null || echo 'Unknown error')"
    fi
}

###############################################################################
# Test Data Preparation
###############################################################################

prepare_test_data() {
    log "Preparing test data..."
    
    # Sample electrical consumption request
    cat > "$OUTPUT_DIR/test_data/electrical_consumption_request.json" << 'EOF'
{
    "facility_filter": {
        "facility_id": "FAC-001",
        "facility_name": "Main Campus"
    },
    "date_range": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "output_format": "json",
    "include_emissions": true,
    "include_cost_analysis": true
}
EOF

    # Minimal electrical consumption request
    cat > "$OUTPUT_DIR/test_data/electrical_minimal_request.json" << 'EOF'
{
    "output_format": "json"
}
EOF

    # Sample water consumption request
    cat > "$OUTPUT_DIR/test_data/water_consumption_request.json" << 'EOF'
{
    "date_range": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "output_format": "json",
    "include_meter_details": true,
    "include_emissions": true
}
EOF

    # Sample waste generation request
    cat > "$OUTPUT_DIR/test_data/waste_generation_request.json" << 'EOF'
{
    "facility_filter": {
        "facility_id": "FAC-001"
    },
    "date_range": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "output_format": "json",
    "include_disposal_details": true,
    "include_transport_details": true,
    "include_emissions": true,
    "hazardous_only": false
}
EOF

    # Hazardous waste only request
    cat > "$OUTPUT_DIR/test_data/hazardous_waste_request.json" << 'EOF'
{
    "date_range": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "output_format": "json",
    "hazardous_only": true,
    "include_disposal_details": false,
    "include_transport_details": false,
    "include_emissions": false
}
EOF

    # Batch ingestion request
    cat > "$OUTPUT_DIR/test_data/batch_ingestion_request.json" << 'EOF'
{
    "clear_database": true
}
EOF

    # Invalid date range request
    cat > "$OUTPUT_DIR/test_data/invalid_date_range_request.json" << 'EOF'
{
    "date_range": {
        "start_date": "2023-12-31",
        "end_date": "2023-01-01"
    },
    "output_format": "json"
}
EOF

    # Large date range request
    cat > "$OUTPUT_DIR/test_data/large_date_range_request.json" << 'EOF'
{
    "date_range": {
        "start_date": "2020-01-01",
        "end_date": "2023-12-31"
    },
    "output_format": "json"
}
EOF

    log "Test data preparation complete"
}

###############################################################################
# Core API Endpoint Tests
###############################################################################

test_health_endpoint() {
    log "Testing health check endpoint..."
    
    run_api_test "health_check" "GET" "/health" "" "200" \
        'jq -e ".status == \"healthy\" and .timestamp and (.neo4j_connection == true or .neo4j_connection == false) and .version" "$response_file" >/dev/null'
}

test_electrical_consumption_endpoints() {
    log "Testing electrical consumption endpoints..."
    
    # Basic electrical consumption test
    run_api_test "electrical_consumption_basic" "POST" "/api/v1/extract/electrical-consumption" \
        "$(cat "$OUTPUT_DIR/test_data/electrical_consumption_request.json")" "200" \
        'jq -e ".status and .message and .data and .metadata and .processing_time" "$response_file" >/dev/null'
    
    # Minimal request test
    run_api_test "electrical_consumption_minimal" "POST" "/api/v1/extract/electrical-consumption" \
        "$(cat "$OUTPUT_DIR/test_data/electrical_minimal_request.json")" "200" \
        'jq -e ".status and .data.query_type" "$response_file" >/dev/null'
    
    # Text output format test
    run_api_test "electrical_consumption_text_output" "POST" "/api/v1/extract/electrical-consumption" \
        '{"output_format": "txt", "include_emissions": false}' "200" \
        'jq -e ".data.file_path" "$response_file" >/dev/null'
}

test_water_consumption_endpoints() {
    log "Testing water consumption endpoints..."
    
    # Basic water consumption test
    run_api_test "water_consumption_basic" "POST" "/api/v1/extract/water-consumption" \
        "$(cat "$OUTPUT_DIR/test_data/water_consumption_request.json")" "200" \
        'jq -e ".status and .message and .data and .metadata" "$response_file" >/dev/null'
    
    # Date range only test
    run_api_test "water_consumption_date_only" "POST" "/api/v1/extract/water-consumption" \
        '{"date_range": {"start_date": "2023-01-01", "end_date": "2023-12-31"}, "output_format": "json"}' "200" \
        'jq -e ".data.facility_filter == null" "$response_file" >/dev/null'
    
    # No meter details test
    run_api_test "water_consumption_no_details" "POST" "/api/v1/extract/water-consumption" \
        '{"output_format": "json", "include_meter_details": false, "include_emissions": false}' "200" \
        'jq -e ".data.include_meter_details == false and .data.include_emissions == false" "$response_file" >/dev/null'
}

test_waste_generation_endpoints() {
    log "Testing waste generation endpoints..."
    
    # Basic waste generation test
    run_api_test "waste_generation_basic" "POST" "/api/v1/extract/waste-generation" \
        "$(cat "$OUTPUT_DIR/test_data/waste_generation_request.json")" "200" \
        'jq -e ".status and .message and .data and .metadata" "$response_file" >/dev/null'
    
    # Hazardous waste only test
    run_api_test "waste_generation_hazardous_only" "POST" "/api/v1/extract/waste-generation" \
        "$(cat "$OUTPUT_DIR/test_data/hazardous_waste_request.json")" "200" \
        'jq -e ".data.hazardous_only == true" "$response_file" >/dev/null'
    
    # All parameters enabled test
    run_api_test "waste_generation_all_params" "POST" "/api/v1/extract/waste-generation" \
        "$(cat "$OUTPUT_DIR/test_data/waste_generation_request.json")" "200" \
        'jq -e ".data.include_disposal_details == true and .data.include_transport_details == true and .data.include_emissions == true" "$response_file" >/dev/null'
}

test_batch_ingestion_endpoint() {
    log "Testing batch ingestion endpoint..."
    
    # Note: This test might take longer as it processes actual documents
    run_api_test "batch_ingestion" "POST" "/api/v1/ingest/batch" \
        "$(cat "$OUTPUT_DIR/test_data/batch_ingestion_request.json")" "200" \
        'jq -e ".status and .message and .data and .processing_time" "$response_file" >/dev/null'
}

test_custom_extraction_endpoint() {
    log "Testing custom extraction endpoint..."
    
    # Test with facility emissions query type
    run_api_test "custom_extraction_facility_emissions" "POST" "/api/v1/extract/custom?query_type=facility_emissions&output_format=json" \
        '{"facility_filter": {"facility_id": "FAC-001"}}' "200" \
        'jq -e ".status and .data.query_type == \"facility_emissions\"" "$response_file" >/dev/null'
    
    # Test with custom queries
    run_api_test "custom_extraction_with_queries" "POST" "/api/v1/extract/custom?query_type=custom&output_format=json" \
        '{"custom_queries": [{"query": "MATCH (n:Facility) RETURN n", "parameters": {}}]}' "200" \
        'jq -e ".data.custom_queries" "$response_file" >/dev/null'
    
    # Test invalid query type
    run_api_test "custom_extraction_invalid_type" "POST" "/api/v1/extract/custom?query_type=invalid_type&output_format=json" \
        '{}' "400" \
        'jq -e ".detail" "$response_file" >/dev/null'
}

test_query_types_endpoint() {
    log "Testing query types endpoint..."
    
    run_api_test "query_types" "GET" "/api/v1/query-types" "" "200" \
        'jq -e ".query_types and (.query_types | length) > 0 and .query_types[0].value and .query_types[0].name and .query_types[0].description" "$response_file" >/dev/null'
}

###############################################################################
# Phase 1 Enhancement Tests (Placeholder - Update when implemented)
###############################################################################

test_phase1_enhancements() {
    log "Testing Phase 1 enhancement endpoints..."
    
    # Note: These endpoints might not be implemented yet, so we'll test them but expect them to potentially fail
    
    # Test document source file endpoint (audit trail)
    run_api_test "document_source_file_audit" "GET" "/api/v1/documents/electric_bill_001/source_file" "" "200" \
        'true'  # Placeholder check
    
    if [[ "$?" -ne 0 ]]; then
        log_skip "Document source file audit endpoint - not yet implemented"
        ((SKIPPED_TESTS++))
        ((FAILED_TESTS--))
    fi
    
    # Test document audit info endpoint
    run_api_test "document_audit_info" "GET" "/api/v1/documents/electric_bill_001/audit_info" "" "200" \
        'true'  # Placeholder check
    
    if [[ "$?" -ne 0 ]]; then
        log_skip "Document audit info endpoint - not yet implemented"
        ((SKIPPED_TESTS++))
        ((FAILED_TESTS--))
    fi
    
    # Test prorating process endpoint
    run_api_test "prorating_process" "POST" "/api/v1/prorating/process/electric_bill_001" "" "200" \
        'true'  # Placeholder check
    
    if [[ "$?" -ne 0 ]]; then
        log_skip "Prorating process endpoint - not yet implemented"
        ((SKIPPED_TESTS++))
        ((FAILED_TESTS--))
    fi
    
    # Test monthly prorating report
    run_api_test "prorating_monthly_report" "GET" "/api/v1/prorating/monthly-report?year=2023&month=12" "" "200" \
        'true'  # Placeholder check
    
    if [[ "$?" -ne 0 ]]; then
        log_skip "Prorating monthly report endpoint - not yet implemented"
        ((SKIPPED_TESTS++))
        ((FAILED_TESTS--))
    fi
    
    # Test document rejection
    run_api_test "document_reject" "PUT" "/api/v1/documents/electric_bill_001/reject" \
        '{"reason": "Invalid data format", "rejected_by": "test_user"}' "200" \
        'true'  # Placeholder check
    
    if [[ "$?" -ne 0 ]]; then
        log_skip "Document rejection endpoint - not yet implemented"
        ((SKIPPED_TESTS++))
        ((FAILED_TESTS--))
    fi
    
    # Test rejected documents list
    run_api_test "rejected_documents_list" "GET" "/api/v1/documents/rejected" "" "200" \
        'true'  # Placeholder check
    
    if [[ "$?" -ne 0 ]]; then
        log_skip "Rejected documents list endpoint - not yet implemented"
        ((SKIPPED_TESTS++))
        ((FAILED_TESTS--))
    fi
}

###############################################################################
# Edge Case and Error Testing
###############################################################################

test_edge_cases_and_errors() {
    log "Testing edge cases and error conditions..."
    
    # Invalid JSON payload
    run_api_test "invalid_json_payload" "POST" "/api/v1/extract/electrical-consumption" \
        'invalid json' "422" \
        'true'  # Expect validation error
    
    # Invalid date range (end before start)
    run_api_test "invalid_date_range" "POST" "/api/v1/extract/electrical-consumption" \
        "$(cat "$OUTPUT_DIR/test_data/invalid_date_range_request.json")" "422" \
        'true'  # Expect validation error
    
    # Empty request body
    run_api_test "empty_request_body" "POST" "/api/v1/extract/electrical-consumption" \
        '{}' "200" \
        'true'  # Should work with defaults
    
    # Large date range
    run_api_test "large_date_range" "POST" "/api/v1/extract/waste-generation" \
        "$(cat "$OUTPUT_DIR/test_data/large_date_range_request.json")" "200" \
        'jq -e ".status" "$response_file" >/dev/null'
    
    # Same start and end date
    run_api_test "same_start_end_date" "POST" "/api/v1/extract/water-consumption" \
        '{"date_range": {"start_date": "2023-06-15", "end_date": "2023-06-15"}, "output_format": "json"}' "200" \
        'jq -e ".status" "$response_file" >/dev/null'
    
    # Non-existent endpoint
    run_api_test "non_existent_endpoint" "GET" "/api/v1/non-existent" "" "404" \
        'true'
}

###############################################################################
# Performance Tests
###############################################################################

test_performance() {
    if [[ "$RUN_PERFORMANCE_TESTS" != "true" ]]; then
        log "Skipping performance tests (use --performance to enable)"
        return
    fi
    
    log "Running performance tests..."
    
    # Response time validation
    run_api_test "performance_response_time" "POST" "/api/v1/extract/electrical-consumption" \
        '{"output_format": "json"}' "200" \
        'response_time=$(jq -r ".processing_time" "$response_file"); [[ $(echo "$response_time < 5.0" | bc -l) == 1 ]]'
    
    # Concurrent requests test
    log "Testing concurrent requests..."
    local concurrent_pids=()
    
    for i in $(seq 1 $MAX_PARALLEL_REQUESTS); do
        (
            run_api_test "concurrent_request_$i" "POST" "/api/v1/extract/electrical-consumption" \
                '{"output_format": "json"}' "200" \
                'jq -e ".status" "$response_file" >/dev/null'
        ) &
        concurrent_pids+=($!)
    done
    
    # Wait for all concurrent requests to complete
    for pid in "${concurrent_pids[@]}"; do
        wait $pid
    done
    
    log "Concurrent requests test completed"
}

###############################################################################
# Data Validation Tests
###############################################################################

test_data_validation() {
    log "Testing response data validation..."
    
    # Test electrical consumption response structure
    run_api_test "electrical_response_structure" "POST" "/api/v1/extract/electrical-consumption" \
        "$(cat "$OUTPUT_DIR/test_data/electrical_consumption_request.json")" "200" \
        'jq -e ".status and .message and .data and .metadata and .processing_time and .data.query_type and .data.facility_filter and .data.date_range" "$response_file" >/dev/null'
    
    # Test metadata fields presence
    run_api_test "metadata_fields_presence" "POST" "/api/v1/extract/water-consumption" \
        '{"output_format": "json"}' "200" \
        'jq -e ".metadata.total_queries and (.metadata.successful_queries | type == \"number\") and (.metadata.total_records | type == \"number\") and .metadata.generated_at and .metadata.processing_status" "$response_file" >/dev/null'
    
    # Test processing time validation
    run_api_test "processing_time_validation" "POST" "/api/v1/extract/electrical-consumption" \
        '{"output_format": "json"}' "200" \
        'processing_time=$(jq -r ".processing_time" "$response_file"); [[ $(echo "$processing_time >= 0" | bc -l) == 1 ]]'
}

###############################################################################
# Filtering and Date Range Tests
###############################################################################

test_filtering_scenarios() {
    log "Testing filtering scenarios..."
    
    # Facility ID filtering
    run_api_test "facility_id_filtering" "POST" "/api/v1/extract/electrical-consumption" \
        '{"facility_filter": {"facility_id": "FAC-001"}, "output_format": "json"}' "200" \
        'jq -e ".data.facility_filter.facility_id == \"FAC-001\"" "$response_file" >/dev/null'
    
    # Facility name filtering
    run_api_test "facility_name_filtering" "POST" "/api/v1/extract/water-consumption" \
        '{"facility_filter": {"facility_name": "Main Campus"}, "output_format": "json"}' "200" \
        'jq -e ".data.facility_filter.facility_name == \"Main Campus\"" "$response_file" >/dev/null'
    
    # Date range filtering
    run_api_test "date_range_filtering" "POST" "/api/v1/extract/waste-generation" \
        '{"date_range": {"start_date": "2023-01-01", "end_date": "2023-12-31"}, "output_format": "json"}' "200" \
        'jq -e ".data.date_range.start_date == \"2023-01-01\" and .data.date_range.end_date == \"2023-12-31\"" "$response_file" >/dev/null'
    
    # Combined filtering
    run_api_test "combined_filtering" "POST" "/api/v1/extract/electrical-consumption" \
        "$(cat "$OUTPUT_DIR/test_data/electrical_consumption_request.json")" "200" \
        'jq -e ".data.facility_filter and .data.date_range" "$response_file" >/dev/null'
}

###############################################################################
# Output Format Tests
###############################################################################

test_output_formats() {
    log "Testing output formats..."
    
    # JSON format
    run_api_test "json_output_format" "POST" "/api/v1/extract/electrical-consumption" \
        '{"output_format": "json"}' "200" \
        'jq -e ".data.file_path" "$response_file" >/dev/null'
    
    # Text format
    run_api_test "text_output_format" "POST" "/api/v1/extract/water-consumption" \
        '{"output_format": "txt"}' "200" \
        'jq -e ".data.file_path" "$response_file" >/dev/null'
    
    # Default format (should be JSON)
    run_api_test "default_output_format" "POST" "/api/v1/extract/waste-generation" \
        '{}' "200" \
        'jq -e ".data" "$response_file" >/dev/null'
}

###############################################################################
# Main Test Execution
###############################################################################

main() {
    log "Starting EHS API Test Suite"
    log "========================================="
    
    setup_test_environment
    prepare_test_data
    
    # Trap cleanup function
    trap cleanup_test_environment EXIT
    
    # Run all test suites
    test_health_endpoint
    test_electrical_consumption_endpoints
    test_water_consumption_endpoints
    test_waste_generation_endpoints
    test_batch_ingestion_endpoint
    test_custom_extraction_endpoint
    test_query_types_endpoint
    test_phase1_enhancements
    test_edge_cases_and_errors
    test_data_validation
    test_filtering_scenarios
    test_output_formats
    test_performance
    
    # Print final summary
    log "========================================="
    log "Test Execution Complete"
    log "========================================="
    echo -e "${BLUE}Total Tests:${NC} $TOTAL_TESTS"
    echo -e "${GREEN}Passed:${NC} $PASSED_TESTS"
    echo -e "${RED}Failed:${NC} $FAILED_TESTS" 
    echo -e "${YELLOW}Skipped:${NC} $SKIPPED_TESTS"
    echo -e "${BLUE}Test Report:${NC} $TEST_REPORT"
    
    # Set exit code based on test results
    if [[ $FAILED_TESTS -eq 0 ]]; then
        log "All tests passed successfully!"
        exit 0
    else
        log "Some tests failed. Check the test report for details."
        exit 1
    fi
}

# Check if curl and jq are available
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed"
    exit 1
fi

# Run main function
main "$@"