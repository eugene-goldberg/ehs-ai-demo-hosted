#!/bin/bash

# Rejection Tracking API Test Script
# Tests rejection tracking API endpoints with real Neo4j database
# Author: Claude Code Assistant
# Created: 2025-08-23

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
API_HOST="http://localhost"
API_PORT="8000"
BASE_URL="${API_HOST}:${API_PORT}"
TIMEOUT=10
MAX_RETRIES=5
RETRY_DELAY=2

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Real document IDs from database
DOCUMENT_IDS=(
    "electric_bill_20250823_095948_505"
    "water_bill_20250823_100019_766" 
    "waste_manifest_20250823_100047_703"
)

# Test user ID for rejection operations
TEST_USER_ID="test-user-001"

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    
    case $status in
        "PASS")
            echo -e "${GREEN}✓ PASS${NC}: $message"
            ((PASSED_TESTS++))
            ;;
        "FAIL")
            echo -e "${RED}✗ FAIL${NC}: $message"
            ((FAILED_TESTS++))
            ;;
        "INFO")
            echo -e "${BLUE}ℹ INFO${NC}: $message"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠ WARN${NC}: $message"
            ;;
        "ACTION")
            echo -e "${CYAN}▶ ACTION${NC}: $message"
            ;;
    esac
}

# Function to print section headers
print_header() {
    local header=$1
    echo ""
    echo -e "${BLUE}=================================================${NC}"
    echo -e "${BLUE}$header${NC}"
    echo -e "${BLUE}=================================================${NC}"
}

# Function to print test headers
print_test_header() {
    local test_name=$1
    echo ""
    echo -e "${CYAN}--- $test_name ---${NC}"
}

# Function to check if API is running
check_api_connection() {
    print_header "API Connection Test"
    
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s --max-time $TIMEOUT "${BASE_URL}/docs" > /dev/null 2>&1; then
            print_status "PASS" "API is running on ${BASE_URL}"
            return 0
        else
            if [ $i -eq $MAX_RETRIES ]; then
                print_status "FAIL" "API is not responding on ${BASE_URL} after $MAX_RETRIES attempts"
                print_status "INFO" "Please ensure the API server is running with: python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000"
                return 1
            else
                print_status "WARN" "API not ready, retrying in ${RETRY_DELAY}s (attempt $i/$MAX_RETRIES)"
                sleep $RETRY_DELAY
            fi
        fi
    done
}

# Function to test GET endpoint
test_get_endpoint() {
    local endpoint=$1
    local description=$2
    local expected_status=${3:-200}
    local full_url="${BASE_URL}${endpoint}"
    
    ((TOTAL_TESTS++))
    
    print_test_header "$description"
    print_status "ACTION" "GET $endpoint"
    
    # Make the request and capture response
    response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT "$full_url" 2>/dev/null)
    
    # Extract HTTP status code (last line) and response body (everything else)
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    # Check if curl command was successful
    if [ $? -eq 0 ]; then
        if [ "$http_code" -eq "$expected_status" ]; then
            print_status "PASS" "$description - HTTP $http_code"
            if [ ! -z "$response_body" ] && [ "$response_body" != "null" ]; then
                echo -e "  ${GREEN}Response:${NC} $response_body"
            fi
        else
            print_status "FAIL" "$description - Expected HTTP $expected_status, got $http_code"
            if [ ! -z "$response_body" ]; then
                echo -e "  ${RED}Response:${NC} $response_body"
            fi
        fi
    else
        print_status "FAIL" "$description - Connection failed"
    fi
}

# Function to test POST endpoint
test_post_endpoint() {
    local endpoint=$1
    local description=$2
    local data=$3
    local expected_status=${4:-200}
    local full_url="${BASE_URL}${endpoint}"
    
    ((TOTAL_TESTS++))
    
    print_test_header "$description"
    print_status "ACTION" "POST $endpoint"
    if [ ! -z "$data" ]; then
        echo -e "  ${BLUE}Data:${NC} $data"
    fi
    
    # Make the request and capture response
    if [ ! -z "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
            -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$full_url" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
            -X POST \
            "$full_url" 2>/dev/null)
    fi
    
    # Extract HTTP status code (last line) and response body (everything else)
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    # Check if curl command was successful
    if [ $? -eq 0 ]; then
        if [ "$http_code" -eq "$expected_status" ]; then
            print_status "PASS" "$description - HTTP $http_code"
            if [ ! -z "$response_body" ] && [ "$response_body" != "null" ]; then
                echo -e "  ${GREEN}Response:${NC} $response_body"
            fi
        else
            print_status "FAIL" "$description - Expected HTTP $expected_status, got $http_code"
            if [ ! -z "$response_body" ]; then
                echo -e "  ${RED}Response:${NC} $response_body"
            fi
        fi
    else
        print_status "FAIL" "$description - Connection failed"
    fi
}

# Function to print final summary
print_summary() {
    print_header "Test Summary"
    
    echo -e "${BLUE}Total Tests:${NC} $TOTAL_TESTS"
    echo -e "${GREEN}Passed:${NC} $PASSED_TESTS"
    echo -e "${RED}Failed:${NC} $FAILED_TESTS"
    
    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo -e "${BLUE}Success Rate:${NC} $success_rate%"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All tests passed!${NC}"
        exit 0
    else
        echo -e "\n${YELLOW}⚠️  Some tests failed. Check the API implementation and database state.${NC}"
        exit 1
    fi
}

# Function to show available documents
show_available_documents() {
    print_header "Available Test Documents"
    echo -e "${BLUE}The following document IDs will be used for testing:${NC}"
    for doc_id in "${DOCUMENT_IDS[@]}"; do
        echo -e "  • $doc_id"
    done
}

# Main test execution
main() {
    print_header "Rejection Tracking API Test Suite"
    echo -e "${BLUE}Testing EHS AI Demo Rejection Tracking API${NC}"
    echo -e "${BLUE}Target API: ${BASE_URL}${NC}"
    echo -e "${BLUE}Using real Neo4j database for testing${NC}"
    echo ""
    
    # Check if API is running
    if ! check_api_connection; then
        exit 1
    fi
    
    # Show available documents for testing
    show_available_documents
    
    # Test 1: Health Check
    print_header "1. Health Check Tests"
    test_get_endpoint "/api/v1/documents/rejection-tracking/health" "Rejection Tracking Health Check"
    
    # Test 2: Get Initial Rejected Documents (should be empty initially)
    print_header "2. Initial State Tests"
    test_get_endpoint "/api/v1/documents/rejected" "Get Initial Rejected Documents List"
    test_get_endpoint "/api/v1/documents/rejection-statistics" "Get Initial Rejection Statistics"
    
    # Test 3: Reject Documents
    print_header "3. Document Rejection Tests"
    
    # Reject first document
    doc_id="${DOCUMENT_IDS[0]}"
    reject_data="{\"rejection_reason\": \"Document quality too low for processing\", \"rejected_by_user_id\": \"$TEST_USER_ID\"}"
    test_post_endpoint "/api/v1/documents/$doc_id/reject" "Reject Document: $doc_id" "$reject_data"
    
    # Reject second document
    doc_id="${DOCUMENT_IDS[1]}"
    reject_data="{\"rejection_reason\": \"Missing required information\", \"rejected_by_user_id\": \"$TEST_USER_ID\"}"
    test_post_endpoint "/api/v1/documents/$doc_id/reject" "Reject Document: $doc_id" "$reject_data"
    
    # Test 4: Verify Rejected Documents
    print_header "4. Post-Rejection Verification Tests"
    test_get_endpoint "/api/v1/documents/rejected" "Get Rejected Documents After Rejections"
    test_get_endpoint "/api/v1/documents/rejection-statistics" "Get Updated Rejection Statistics"
    
    # Test 5: Unreject a Document
    print_header "5. Document Un-rejection Tests"
    
    # Unreject first document
    doc_id="${DOCUMENT_IDS[0]}"
    unreject_data="{\"unrejected_by_user_id\": \"$TEST_USER_ID\"}"
    test_post_endpoint "/api/v1/documents/$doc_id/unreject" "Unreject Document: $doc_id" "$unreject_data"
    
    # Test 6: Final State Verification
    print_header "6. Final State Verification Tests"
    test_get_endpoint "/api/v1/documents/rejected" "Get Final Rejected Documents List"
    test_get_endpoint "/api/v1/documents/rejection-statistics" "Get Final Rejection Statistics"
    
    # Test 7: Error Handling Tests
    print_header "7. Error Handling Tests"
    
    # Test with non-existent document
    test_post_endpoint "/api/v1/documents/non-existent-doc-id/reject" "Reject Non-existent Document" "$reject_data" 404
    test_post_endpoint "/api/v1/documents/non-existent-doc-id/unreject" "Unreject Non-existent Document" "$unreject_data" 404
    
    # Test with invalid JSON
    test_post_endpoint "/api/v1/documents/${DOCUMENT_IDS[2]}/reject" "Reject with Invalid JSON" "{invalid json}" 422
    
    # Test 8: Edge Cases
    print_header "8. Edge Case Tests"
    
    # Try to reject already rejected document
    doc_id="${DOCUMENT_IDS[1]}"
    test_post_endpoint "/api/v1/documents/$doc_id/reject" "Re-reject Already Rejected Document" "$reject_data"
    
    # Try to unreject non-rejected document  
    doc_id="${DOCUMENT_IDS[2]}"
    test_post_endpoint "/api/v1/documents/$doc_id/unreject" "Unreject Non-rejected Document" "$unreject_data"
    
    # Print final summary
    print_summary
}

# Help function
show_help() {
    echo "Rejection Tracking API Test Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -u, --url URL  Set custom API base URL (default: http://localhost:8000)"
    echo "  -t, --timeout  Set request timeout in seconds (default: 10)"
    echo ""
    echo "Examples:"
    echo "  $0                              # Test using default settings"
    echo "  $0 -u http://api.example.com    # Test against custom URL"
    echo "  $0 --timeout 30                 # Use 30 second timeout"
    echo ""
    echo "Document IDs that will be used for testing:"
    for doc_id in "${DOCUMENT_IDS[@]}"; do
        echo "  • $doc_id"
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--url)
            BASE_URL="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run the tests
main