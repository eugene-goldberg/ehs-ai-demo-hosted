#!/bin/bash

# Phase 1 Endpoints Test Script
# Tests health endpoints for audit-trail, prorating, and rejection-tracking features
# Author: Claude Code Assistant
# Created: 2025-08-23

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    esac
}

# Function to print section headers
print_header() {
    local header=$1
    echo ""
    echo -e "${BLUE}===========================================${NC}"
    echo -e "${BLUE}$header${NC}"
    echo -e "${BLUE}===========================================${NC}"
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

# Function to test an endpoint
test_endpoint() {
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
    response_body=$(echo "$response" | head -n -1)
    
    # Check if curl command was successful
    if [ $? -eq 0 ]; then
        case $http_code in
            200)
                print_status "PASS" "$description - HTTP $http_code"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${GREEN}Response:${NC} $response_body"
                fi
                ;;
            404)
                print_status "FAIL" "$description - HTTP $http_code (Not Found)"
                print_status "INFO" "Endpoint may not be implemented yet"
                ;;
            500)
                print_status "FAIL" "$description - HTTP $http_code (Internal Server Error)"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${RED}Error:${NC} $response_body"
                fi
                ;;
            *)
                print_status "FAIL" "$description - HTTP $http_code"
                if [ ! -z "$response_body" ]; then
                    echo -e "  ${YELLOW}Response:${NC} $response_body"
                fi
                ;;
        esac
    else
        print_status "FAIL" "$description - Connection failed"
    fi
    
    echo "" # Add spacing between tests
}

# Function to print final summary
print_summary() {
    print_header "Test Summary"
    
    echo -e "${BLUE}Total Tests:${NC} $TOTAL_TESTS"
    echo -e "${GREEN}Passed:${NC} $PASSED_TESTS"
    echo -e "${RED}Failed:${NC} $FAILED_TESTS"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All tests passed!${NC}"
        exit 0
    else
        echo -e "\n${RED}❌ Some tests failed. Please check the endpoints and API implementation.${NC}"
        exit 1
    fi
}

# Main execution
main() {
    print_header "Phase 1 Endpoints Test Suite"
    echo -e "${BLUE}Testing EHS AI Demo Phase 1 health endpoints${NC}"
    echo -e "${BLUE}Target API: ${BASE_URL}${NC}"
    echo ""
    
    # Check if API is running
    if ! check_api_connection; then
        exit 1
    fi
    
    # Test Phase 1 health endpoints
    print_header "Phase 1 Health Endpoints"
    
    test_endpoint "/api/v1/audit-trail/health" "Audit Trail Health Check"
    test_endpoint "/api/v1/prorating/health" "Prorating Health Check"  
    test_endpoint "/api/v1/rejection-tracking/health" "Rejection Tracking Health Check"
    
    # Optional: Test main API health if available
    print_header "General API Health"
    test_endpoint "/health" "Main API Health Check"
    test_endpoint "/api/health" "API Health Check"
    
    # Print final summary
    print_summary
}

# Help function
show_help() {
    echo "Phase 1 Endpoints Test Script"
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