#!/bin/bash

# Quick Test Script for Simple Test API
# Tests basic functionality of simple_test_api.py endpoints

# Configuration
API_BASE="http://localhost:8000"
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print test results
print_test_result() {
    local test_name="$1"
    local status_code="$2"
    local expected_code="$3"
    local response="$4"
    
    echo -e "\n${BOLD}Test: $test_name${NC}"
    echo "Expected: $expected_code, Got: $status_code"
    
    if [ "$status_code" == "$expected_code" ]; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        echo -e "${RED}✗ FAIL${NC}"
    fi
    
    echo "Response: $response"
    echo "----------------------------------------"
}

# Function to test an endpoint
test_endpoint() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local expected_code="$4"
    local test_name="$5"
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$API_BASE$endpoint")
    else
        response=$(curl -s -w "HTTPSTATUS:%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_BASE$endpoint")
    fi
    
    # Extract HTTP status code and body
    status_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    body=$(echo "$response" | sed 's/HTTPSTATUS:[0-9]*$//')
    
    print_test_result "$test_name" "$status_code" "$expected_code" "$body"
}

# Main testing function
main() {
    echo -e "${BOLD}${YELLOW}Quick API Test Script${NC}"
    echo -e "${BOLD}Testing API at: $API_BASE${NC}"
    echo "========================================"
    
    # Check if API is reachable
    echo -e "\n${BOLD}Checking API availability...${NC}"
    if ! curl -s "$API_BASE/health" > /dev/null; then
        echo -e "${RED}✗ API is not reachable at $API_BASE${NC}"
        echo -e "${YELLOW}Make sure the API server is running:${NC}"
        echo "  python3 simple_test_api.py"
        exit 1
    fi
    echo -e "${GREEN}✓ API is reachable${NC}"
    
    # Test 1: Health check
    test_endpoint "GET" "/health" "" "200" "Health Check"
    
    # Test 2: Root endpoint
    test_endpoint "GET" "/" "" "200" "Root Endpoint"
    
    # Test 3: Status endpoint
    test_endpoint "GET" "/status" "" "200" "Status Endpoint"
    
    # Test 4: Echo endpoint
    test_endpoint "POST" "/echo" '{"message": "Hello World", "test": true}' "200" "Echo Endpoint"
    
    # Test 5: Get item endpoint (valid ID)
    test_endpoint "GET" "/test/123" "" "200" "Get Item (Valid ID)"
    
    # Test 6: Get item endpoint (invalid ID)
    test_endpoint "GET" "/test/-1" "" "400" "Get Item (Invalid ID)"
    
    # Test 7: Create item endpoint (valid data)
    test_endpoint "POST" "/test/items" '{"name": "Test Item", "description": "A test item"}' "200" "Create Item (Valid Data)"
    
    # Test 8: Create item endpoint (missing name)
    test_endpoint "POST" "/test/items" '{"description": "A test item without name"}' "400" "Create Item (Missing Name)"
    
    # Test 9: Error endpoint
    test_endpoint "GET" "/test/error" "" "500" "Error Endpoint"
    
    echo -e "\n${BOLD}${YELLOW}Test Summary${NC}"
    echo "========================================"
    echo -e "${GREEN}All tests completed!${NC}"
    echo -e "${YELLOW}Note: Check individual test results above for pass/fail status${NC}"
    echo -e "\n${BOLD}API Endpoints Tested:${NC}"
    echo "  GET  /health"
    echo "  GET  /"
    echo "  GET  /status"
    echo "  POST /echo"
    echo "  GET  /test/{item_id}"
    echo "  POST /test/items"
    echo "  GET  /test/error"
}

# Run the tests
main "$@"