#!/bin/bash

# Comprehensive Phase 1 Features Test Script
# Tests all three Phase 1 features: Audit Trail, Pro-Rating, and Rejection Tracking
# Author: Claude Code Assistant
# Created: 2025-08-23

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
API_HOST="http://localhost"
API_PORT="8000"
BASE_URL="${API_HOST}:${API_PORT}"
TIMEOUT=15
MAX_RETRIES=5
RETRY_DELAY=2

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
FEATURE_RESULTS=()

# Test document IDs
DOC1="electric_bill_20250823_095948_505"
DOC2="water_bill_20250823_100019_766"

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
        "SUCCESS")
            echo -e "${GREEN}🎉 SUCCESS${NC}: $message"
            ;;
        "ERROR")
            echo -e "${RED}❌ ERROR${NC}: $message"
            ;;
    esac
}

# Function to print section headers
print_header() {
    local header=$1
    echo ""
    echo -e "${CYAN}=============================================${NC}"
    echo -e "${CYAN}$header${NC}"
    echo -e "${CYAN}=============================================${NC}"
}

# Function to print feature headers
print_feature_header() {
    local feature=$1
    echo ""
    echo -e "${MAGENTA}╔════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║  $feature${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════╝${NC}"
}

# Function to print response safely
print_response() {
    local response=$1
    local lines_to_show=${2:-3}
    
    if [ ! -z "$response" ]; then
        echo "$response" | head -n "$lines_to_show"
    fi
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

# Function to test a GET endpoint
test_get_endpoint() {
    local endpoint=$1
    local description=$2
    local full_url="${BASE_URL}${endpoint}"
    
    ((TOTAL_TESTS++))
    
    print_status "INFO" "Testing $description"
    print_status "INFO" "URL: $full_url"
    
    # Make the request and capture response
    response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT "$full_url" 2>/dev/null)
    
    # Extract HTTP status code (last line) and response body (everything else)
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | sed '$d')
    
    # Check if curl command was successful
    if [ $? -eq 0 ]; then
        case $http_code in
            200)
                print_status "PASS" "$description - HTTP $http_code"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${GREEN}Response:${NC}"
                    print_response "$response_body" 3
                fi
                return 0
                ;;
            404)
                print_status "FAIL" "$description - HTTP $http_code (Not Found)"
                print_status "INFO" "Endpoint may not be implemented yet"
                return 1
                ;;
            500)
                print_status "FAIL" "$description - HTTP $http_code (Internal Server Error)"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${RED}Error:${NC} $response_body"
                fi
                return 1
                ;;
            *)
                print_status "FAIL" "$description - HTTP $http_code"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${YELLOW}Response:${NC}"
                    print_response "$response_body" 2
                fi
                return 1
                ;;
        esac
    else
        print_status "FAIL" "$description - Connection failed"
        return 1
    fi
    
    echo "" # Add spacing between tests
}

# Function to test a POST endpoint
test_post_endpoint() {
    local endpoint=$1
    local description=$2
    local json_payload=$3
    local full_url="${BASE_URL}${endpoint}"
    
    ((TOTAL_TESTS++))
    
    print_status "INFO" "Testing $description"
    print_status "INFO" "URL: $full_url"
    print_status "INFO" "Payload: $json_payload"
    
    # Make the request and capture response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        -w "\n%{http_code}" \
        --max-time $TIMEOUT \
        "$full_url" 2>/dev/null)
    
    # Extract HTTP status code (last line) and response body (everything else)
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | sed '$d')
    
    # Check if curl command was successful
    if [ $? -eq 0 ]; then
        case $http_code in
            200|201)
                print_status "PASS" "$description - HTTP $http_code"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${GREEN}Response:${NC}"
                    print_response "$response_body" 3
                fi
                return 0
                ;;
            404)
                print_status "FAIL" "$description - HTTP $http_code (Not Found)"
                print_status "INFO" "Endpoint may not be implemented yet"
                return 1
                ;;
            422)
                print_status "FAIL" "$description - HTTP $http_code (Validation Error)"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${RED}Validation Error:${NC} $response_body"
                fi
                return 1
                ;;
            500)
                print_status "FAIL" "$description - HTTP $http_code (Internal Server Error)"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${RED}Error:${NC} $response_body"
                fi
                return 1
                ;;
            *)
                print_status "FAIL" "$description - HTTP $http_code"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${YELLOW}Response:${NC}"
                    print_response "$response_body" 2
                fi
                return 1
                ;;
        esac
    else
        print_status "FAIL" "$description - Connection failed"
        return 1
    fi
    
    echo "" # Add spacing between tests
}

# Function to test health endpoints
test_health_endpoints() {
    print_header "Health Check Endpoints"
    
    local health_passed=0
    local health_total=5
    
    # Test main API health endpoints
    if test_get_endpoint "/health" "Main API Health Check"; then
        ((health_passed++))
    fi
    
    if test_get_endpoint "/api/health" "API Health Check"; then
        ((health_passed++))
    fi
    
    # Test Phase 1 health endpoints - using correct paths from OpenAPI schema
    if test_get_endpoint "/api/v1/audit-trail/documents/audit/health" "Audit Trail Health Check"; then
        ((health_passed++))
    fi
    
    if test_get_endpoint "/api/v1/prorating/prorating/health" "Prorating Health Check"; then
        ((health_passed++))
    fi
    
    if test_get_endpoint "/api/v1/rejection-tracking/documents/rejection-tracking/health" "Rejection Tracking Health Check"; then
        ((health_passed++))
    fi
    
    echo -e "${BLUE}Health Checks Summary: ${GREEN}$health_passed${NC}/${BLUE}5 passed${NC}"
    return $health_passed
}

# Function to test Audit Trail API
test_audit_trail_api() {
    print_feature_header "AUDIT TRAIL API TESTS"
    
    local feature_passed=0
    local feature_total=6
    
    # Test 1: Get audit info for first document
    if test_get_endpoint "/api/v1/audit-trail/documents/${DOC1}/audit_info" "Get audit info for ${DOC1}"; then
        ((feature_passed++))
    fi
    
    # Test 2: Get audit info for second document
    if test_get_endpoint "/api/v1/audit-trail/documents/${DOC2}/audit_info" "Get audit info for ${DOC2}"; then
        ((feature_passed++))
    fi
    
    # Test 3: Update source for first document
    local update_payload1='{
        "source_type": "file",
        "source_location": "/uploads/electric_bill_updated.pdf",
        "source_metadata": {
            "filename": "electric_bill_updated.pdf",
            "size": 245760,
            "mime_type": "application/pdf",
            "uploaded_by": "test_user",
            "updated_reason": "corrected billing period"
        }
    }'
    
    if test_post_endpoint "/api/v1/audit-trail/documents/${DOC1}/update_source" "Update source for ${DOC1}" "$update_payload1"; then
        ((feature_passed++))
    fi
    
    # Test 4: Update source for second document
    local update_payload2='{
        "source_type": "url",
        "source_location": "https://water-utility.com/bills/2025/08/bill_766.pdf",
        "source_metadata": {
            "url": "https://water-utility.com/bills/2025/08/bill_766.pdf",
            "retrieved_at": "2025-08-23T10:00:19Z",
            "content_type": "application/pdf",
            "updated_by": "test_user",
            "updated_reason": "refreshed from source"
        }
    }'
    
    if test_post_endpoint "/api/v1/audit-trail/documents/${DOC2}/update_source" "Update source for ${DOC2}" "$update_payload2"; then
        ((feature_passed++))
    fi
    
    # Test 5: Get source file for first document
    if test_get_endpoint "/api/v1/audit-trail/documents/${DOC1}/source_file" "Get source file for ${DOC1}"; then
        ((feature_passed++))
    fi
    
    # Test 6: Get source URL for first document
    if test_get_endpoint "/api/v1/audit-trail/documents/${DOC1}/source_url" "Get source URL for ${DOC1}"; then
        ((feature_passed++))
    fi
    
    echo -e "${MAGENTA}Audit Trail Summary: ${GREEN}$feature_passed${NC}/${MAGENTA}$feature_total passed${NC}"
    FEATURE_RESULTS+=("Audit Trail: $feature_passed/$feature_total")
    return $feature_passed
}

# Function to test Pro-Rating API
test_prorating_api() {
    print_feature_header "PRO-RATING API TESTS"
    
    local feature_passed=0
    local feature_total=8
    
    # Test 1: Process first document with correct payload format
    local process_payload1='{
        "document_id": "'$DOC1'",
        "method": "headcount",
        "facility_info": [
            {
                "facility_id": "facility_001",
                "name": "Main Office",
                "headcount": 100,
                "floor_area": 10000
            }
        ]
    }'
    
    if test_post_endpoint "/api/v1/prorating/prorating/process/${DOC1}" "Process document ${DOC1}" "$process_payload1"; then
        ((feature_passed++))
    fi
    
    # Test 2: Process second document with area-based method
    local process_payload2='{
        "document_id": "'$DOC2'",
        "method": "area",
        "facility_info": [
            {
                "facility_id": "facility_002",
                "name": "Secondary Office",
                "headcount": 50,
                "floor_area": 5000
            }
        ]
    }'
    
    if test_post_endpoint "/api/v1/prorating/prorating/process/${DOC2}" "Process document ${DOC2}" "$process_payload2"; then
        ((feature_passed++))
    fi
    
    # Test 3: Get allocations for first document
    if test_get_endpoint "/api/v1/prorating/prorating/allocations/${DOC1}" "Get allocations for ${DOC1}"; then
        ((feature_passed++))
    fi
    
    # Test 4: Get allocations for second document
    if test_get_endpoint "/api/v1/prorating/prorating/allocations/${DOC2}" "Get allocations for ${DOC2}"; then
        ((feature_passed++))
    fi
    
    # Test 5: Get facility allocations
    if test_get_endpoint "/api/v1/prorating/prorating/facility/facility_001/allocations" "Get facility allocations"; then
        ((feature_passed++))
    fi
    
    # Test 6: Generate monthly report
    local report_payload='{
        "facility_id": "facility_001",
        "year": 2025,
        "month": 8,
        "report_format": "detailed"
    }'
    
    if test_post_endpoint "/api/v1/prorating/prorating/monthly-report" "Generate monthly report" "$report_payload"; then
        ((feature_passed++))
    fi
    
    # Test 7: Batch process documents
    local batch_payload='{
        "document_ids": [
            "'$DOC1'",
            "'$DOC2'"
        ],
        "method": "headcount",
        "facility_mapping": {
            "'$DOC1'": "facility_001",
            "'$DOC2'": "facility_002"
        }
    }'
    
    if test_post_endpoint "/api/v1/prorating/prorating/batch-process" "Batch process documents" "$batch_payload"; then
        ((feature_passed++))
    fi
    
    # Test 8: Calculate pro-rating with comprehensive payload
    local calculate_payload='{
        "document_id": "'$DOC1'",
        "method": "headcount", 
        "facility_info": [
            {
                "facility_id": "facility_001",
                "name": "Main Office",
                "headcount": 100,
                "floor_area": 10000
            },
            {
                "facility_id": "facility_003",
                "name": "Branch Office", 
                "headcount": 25,
                "floor_area": 2500
            }
        ]
    }'
    
    if test_post_endpoint "/api/v1/prorating/prorating/calculate" "Calculate pro-rating" "$calculate_payload"; then
        ((feature_passed++))
    fi
    
    echo -e "${MAGENTA}Pro-Rating Summary: ${GREEN}$feature_passed${NC}/${MAGENTA}$feature_total passed${NC}"
    FEATURE_RESULTS+=("Pro-Rating: $feature_passed/$feature_total")
    return $feature_passed
}

# Function to test Rejection Tracking API
test_rejection_tracking_api() {
    print_feature_header "REJECTION TRACKING API TESTS"
    
    local feature_passed=0
    local feature_total=4
    
    # Test 1: Reject first document with correct rejection reason
    local reject_payload1='{"reason": "incomplete"}'
    
    if test_post_endpoint "/api/v1/rejection-tracking/documents/${DOC1}/reject" "Reject document ${DOC1}" "$reject_payload1"; then
        ((feature_passed++))
    fi
    
    # Test 2: Reject second document
    local reject_payload2='{"reason": "wrong_format"}'
    
    if test_post_endpoint "/api/v1/rejection-tracking/documents/${DOC2}/reject" "Reject document ${DOC2}" "$reject_payload2"; then
        ((feature_passed++))
    fi
    
    # Test 3: Get all rejected documents
    if test_get_endpoint "/api/v1/rejection-tracking/documents/rejected" "Get all rejected documents"; then
        ((feature_passed++))
    fi
    
    # Test 4: Get rejection details for specific document
    if test_get_endpoint "/api/v1/rejection-tracking/documents/${DOC1}/rejection-details" "Get rejection details for ${DOC1}"; then
        ((feature_passed++))
    fi
    
    echo -e "${MAGENTA}Rejection Tracking Summary: ${GREEN}$feature_passed${NC}/${MAGENTA}$feature_total passed${NC}"
    FEATURE_RESULTS+=("Rejection Tracking: $feature_passed/$feature_total")
    return $feature_passed
}

# Function to print comprehensive summary
print_comprehensive_summary() {
    print_header "COMPREHENSIVE TEST SUMMARY"
    
    echo -e "${BLUE}📊 Overall Test Results:${NC}"
    echo -e "  ${BLUE}Total Tests Run:${NC} $TOTAL_TESTS"
    echo -e "  ${GREEN}✓ Passed:${NC} $PASSED_TESTS"
    echo -e "  ${RED}✗ Failed:${NC} $FAILED_TESTS"
    
    local success_rate=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    echo -e "  ${CYAN}📈 Success Rate:${NC} ${success_rate}%"
    
    echo ""
    echo -e "${BLUE}🎯 Feature-by-Feature Results:${NC}"
    for result in "${FEATURE_RESULTS[@]}"; do
        echo -e "  ${CYAN}•${NC} $result"
    done
    
    echo ""
    echo -e "${BLUE}🔧 Test Documents Used:${NC}"
    echo -e "  ${CYAN}•${NC} $DOC1"
    echo -e "  ${CYAN}•${NC} $DOC2"
    
    echo ""
    if [ $FAILED_TESTS -eq 0 ]; then
        print_status "SUCCESS" "All Phase 1 features are working correctly!"
        echo -e "${GREEN}🚀 Ready for production deployment!${NC}"
        return 0
    elif [ $PASSED_TESTS -gt $(( TOTAL_TESTS / 2 )) ]; then
        print_status "WARN" "Most features are working, but some need attention"
        echo -e "${YELLOW}🔨 Review failed tests and fix issues${NC}"
        return 1
    else
        print_status "ERROR" "Many features are failing - significant issues detected"
        echo -e "${RED}🚨 Major fixes needed before deployment${NC}"
        return 2
    fi
}

# Function to show usage help
show_help() {
    echo "Comprehensive Phase 1 Features Test Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help         Show this help message"
    echo "  -u, --url URL      Set custom API base URL (default: http://localhost:8000)"
    echo "  -t, --timeout SEC  Set request timeout in seconds (default: 15)"
    echo "  --doc1 ID         Set first test document ID (default: electric_bill_20250823_095948_505)"
    echo "  --doc2 ID         Set second test document ID (default: water_bill_20250823_100019_766)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Test using default settings"
    echo "  $0 -u http://api.example.com          # Test against custom URL"
    echo "  $0 --timeout 30                       # Use 30 second timeout"
    echo "  $0 --doc1 custom_doc_123               # Use custom document ID"
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
        --doc1)
            DOC1="$2"
            shift 2
            ;;
        --doc2)
            DOC2="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution function
main() {
    print_header "PHASE 1 COMPREHENSIVE TEST SUITE"
    echo -e "${BLUE}Testing EHS AI Demo Phase 1 Features${NC}"
    echo -e "${BLUE}Target API: ${BASE_URL}${NC}"
    echo -e "${BLUE}Test Documents: ${DOC1}, ${DOC2}${NC}"
    echo -e "${BLUE}Timeout: ${TIMEOUT}s${NC}"
    echo ""
    
    # Check if API is running
    if ! check_api_connection; then
        exit 1
    fi
    
    # Test health endpoints first
    test_health_endpoints
    
    # Test each Phase 1 feature
    test_audit_trail_api
    test_prorating_api
    test_rejection_tracking_api
    
    # Print comprehensive summary and exit with appropriate code
    print_comprehensive_summary
    exit $?
}

# Run the comprehensive test suite
main "$@"