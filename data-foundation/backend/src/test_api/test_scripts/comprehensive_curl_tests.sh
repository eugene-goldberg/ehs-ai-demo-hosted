#!/bin/bash

# Comprehensive API Testing Script
# Created: 2025-08-23
# Purpose: Test all API endpoints with curl commands and generate detailed reports

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
TEST_REPORT_DIR="./test_reports"
TEST_REPORT_FILE="$TEST_REPORT_DIR/api_test_report_$(date +%Y%m%d_%H%M%S).json"
HTML_REPORT_FILE="$TEST_REPORT_DIR/api_test_report_$(date +%Y%m%d_%H%M%S).html"
LOG_FILE="$TEST_REPORT_DIR/test_execution.log"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
TEST_RESULTS=()

# Utility Functions
print_header() {
    echo -e "\n${CYAN}================================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${CYAN}================================================${NC}\n"
}

print_section() {
    echo -e "\n${PURPLE}--- $1 ---${NC}\n"
}

print_test() {
    echo -e "${BLUE}Testing: $1${NC}"
}

print_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

print_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING${NC}: $1"
}

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Setup and Cleanup Functions
setup_test_environment() {
    print_header "Setting up Test Environment"
    
    # Create test report directory
    mkdir -p "$TEST_REPORT_DIR"
    
    # Initialize log file
    echo "API Test Execution Log - $(date)" > "$LOG_FILE"
    log_message "Test environment setup started"
    
    # Check if API is running
    if ! curl -s "$API_BASE_URL/health" > /dev/null 2>&1; then
        print_warning "API not responding at $API_BASE_URL"
        print_warning "Please ensure the API server is running"
        exit 1
    fi
    
    print_pass "Test environment ready"
    log_message "Test environment setup completed"
}

cleanup_test_environment() {
    print_header "Cleaning up Test Environment"
    log_message "Test cleanup started"
    
    # Additional cleanup can be added here
    print_pass "Cleanup completed"
    log_message "Test cleanup completed"
}

# Test Execution Functions
execute_test() {
    local test_name="$1"
    local test_description="$2"
    local curl_command="$3"
    local expected_status="$4"
    local validation_function="$5"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    local start_time=$(date +%s.%N)
    
    print_test "$test_name: $test_description"
    log_message "Starting test: $test_name"
    
    # Execute curl command and capture response
    local response
    local http_status
    local curl_exit_code
    
    response=$(eval "$curl_command" 2>/dev/null) || curl_exit_code=$?
    http_status=$(eval "$curl_command -w '%{http_code}' -o /dev/null -s" 2>/dev/null)
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
    
    # Validate response
    local test_passed=false
    local error_message=""
    
    if [[ ${curl_exit_code:-0} -eq 0 && "$http_status" == "$expected_status" ]]; then
        if [[ -n "$validation_function" ]]; then
            if $validation_function "$response"; then
                test_passed=true
            else
                error_message="Response validation failed"
            fi
        else
            test_passed=true
        fi
    else
        error_message="HTTP Status: $http_status (expected: $expected_status), Curl exit code: ${curl_exit_code:-0}"
    fi
    
    # Record results
    local result_object
    result_object=$(cat <<EOF
{
    "test_name": "$test_name",
    "description": "$test_description",
    "curl_command": "$curl_command",
    "expected_status": "$expected_status",
    "actual_status": "$http_status",
    "response": $(echo "$response" | jq -R . 2>/dev/null || echo "\"$response\""),
    "duration": "$duration",
    "timestamp": "$(date -Iseconds)",
    "passed": $test_passed,
    "error_message": "$error_message"
}
EOF
    )
    
    TEST_RESULTS+=("$result_object")
    
    if [[ "$test_passed" == "true" ]]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        print_pass "$test_name (${duration}s)"
        log_message "Test passed: $test_name"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_fail "$test_name - $error_message"
        log_message "Test failed: $test_name - $error_message"
    fi
    
    echo "  Response: $response"
    echo ""
}

# Validation Functions
validate_json_response() {
    local response="$1"
    echo "$response" | jq . >/dev/null 2>&1
}

validate_health_response() {
    local response="$1"
    echo "$response" | jq -e '.status == "healthy"' >/dev/null 2>&1
}

validate_array_response() {
    local response="$1"
    echo "$response" | jq -e 'type == "array"' >/dev/null 2>&1
}

validate_object_response() {
    local response="$1"
    echo "$response" | jq -e 'type == "object"' >/dev/null 2>&1
}

validate_error_response() {
    local response="$1"
    echo "$response" | jq -e 'has("error") or has("detail")' >/dev/null 2>&1
}

# Test Categories

run_original_features_tests() {
    print_section "Original Features Tests"
    
    # Health check
    execute_test \
        "health_check" \
        "API health status" \
        "curl -s '$API_BASE_URL/health'" \
        "200" \
        "validate_health_response"
    
    # Basic data endpoints
    execute_test \
        "get_all_incidents" \
        "Get all incidents" \
        "curl -s '$API_BASE_URL/incidents'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "get_all_employees" \
        "Get all employees" \
        "curl -s '$API_BASE_URL/employees'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "get_all_departments" \
        "Get all departments" \
        "curl -s '$API_BASE_URL/departments'" \
        "200" \
        "validate_array_response"
    
    # Specific record access
    execute_test \
        "get_incident_by_id" \
        "Get specific incident" \
        "curl -s '$API_BASE_URL/incidents/1'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "get_employee_by_id" \
        "Get specific employee" \
        "curl -s '$API_BASE_URL/employees/1'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "get_department_by_id" \
        "Get specific department" \
        "curl -s '$API_BASE_URL/departments/1'" \
        "200" \
        "validate_object_response"
}

run_phase1_enhancement_tests() {
    print_section "Phase 1 Enhancement Tests"
    
    # Analytics endpoints
    execute_test \
        "analytics_summary" \
        "Get analytics summary" \
        "curl -s '$API_BASE_URL/analytics/summary'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "analytics_trends" \
        "Get analytics trends" \
        "curl -s '$API_BASE_URL/analytics/trends'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "analytics_by_department" \
        "Get analytics by department" \
        "curl -s '$API_BASE_URL/analytics/by-department'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "analytics_risk_assessment" \
        "Get risk assessment" \
        "curl -s '$API_BASE_URL/analytics/risk-assessment'" \
        "200" \
        "validate_object_response"
    
    # Enhanced search and filtering
    execute_test \
        "search_incidents" \
        "Search incidents" \
        "curl -s '$API_BASE_URL/incidents/search?q=safety'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "filter_incidents_by_severity" \
        "Filter incidents by severity" \
        "curl -s '$API_BASE_URL/incidents?severity=high'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "filter_incidents_by_date" \
        "Filter incidents by date range" \
        "curl -s '$API_BASE_URL/incidents?start_date=2024-01-01&end_date=2024-12-31'" \
        "200" \
        "validate_array_response"
    
    # Report generation
    execute_test \
        "generate_incident_report" \
        "Generate incident report" \
        "curl -s '$API_BASE_URL/reports/incidents'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "generate_department_report" \
        "Generate department safety report" \
        "curl -s '$API_BASE_URL/reports/departments/1'" \
        "200" \
        "validate_object_response"
}

run_integration_tests() {
    print_section "Integration Tests"
    
    # Cross-entity relationships
    execute_test \
        "employee_incidents" \
        "Get incidents for employee" \
        "curl -s '$API_BASE_URL/employees/1/incidents'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "department_incidents" \
        "Get incidents for department" \
        "curl -s '$API_BASE_URL/departments/1/incidents'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "department_employees" \
        "Get employees in department" \
        "curl -s '$API_BASE_URL/departments/1/employees'" \
        "200" \
        "validate_array_response"
    
    # Complex queries
    execute_test \
        "incident_analytics_with_filters" \
        "Analytics with multiple filters" \
        "curl -s '$API_BASE_URL/analytics/incidents?department_id=1&severity=high&year=2024'" \
        "200" \
        "validate_object_response"
}

run_performance_tests() {
    print_section "Performance Tests"
    
    # Large dataset queries
    execute_test \
        "large_incident_query" \
        "Query large incident dataset" \
        "curl -s '$API_BASE_URL/incidents?limit=1000'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "concurrent_analytics_1" \
        "Concurrent analytics query 1" \
        "curl -s '$API_BASE_URL/analytics/summary'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "concurrent_analytics_2" \
        "Concurrent analytics query 2" \
        "curl -s '$API_BASE_URL/analytics/trends'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "complex_report_generation" \
        "Complex report generation" \
        "curl -s '$API_BASE_URL/reports/comprehensive?include_analytics=true'" \
        "200" \
        "validate_object_response"
}

run_error_handling_tests() {
    print_section "Error Handling Tests"
    
    # 404 errors
    execute_test \
        "nonexistent_incident" \
        "Get nonexistent incident" \
        "curl -s '$API_BASE_URL/incidents/99999'" \
        "404" \
        "validate_error_response"
    
    execute_test \
        "nonexistent_employee" \
        "Get nonexistent employee" \
        "curl -s '$API_BASE_URL/employees/99999'" \
        "404" \
        "validate_error_response"
    
    execute_test \
        "invalid_endpoint" \
        "Invalid endpoint" \
        "curl -s '$API_BASE_URL/invalid-endpoint'" \
        "404" \
        "validate_error_response"
    
    # 400 errors
    execute_test \
        "invalid_date_format" \
        "Invalid date format in query" \
        "curl -s '$API_BASE_URL/incidents?start_date=invalid-date'" \
        "400" \
        "validate_error_response"
    
    execute_test \
        "invalid_severity_filter" \
        "Invalid severity filter" \
        "curl -s '$API_BASE_URL/incidents?severity=invalid-severity'" \
        "400" \
        "validate_error_response"
}

run_end_to_end_tests() {
    print_section "End-to-End Workflow Tests"
    
    # Typical user workflows
    execute_test \
        "workflow_dashboard_load" \
        "Dashboard data loading workflow" \
        "curl -s '$API_BASE_URL/analytics/summary'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "workflow_incident_investigation" \
        "Incident investigation workflow" \
        "curl -s '$API_BASE_URL/incidents/1'" \
        "200" \
        "validate_object_response"
    
    execute_test \
        "workflow_department_review" \
        "Department safety review workflow" \
        "curl -s '$API_BASE_URL/departments/1/incidents'" \
        "200" \
        "validate_array_response"
    
    execute_test \
        "workflow_report_generation" \
        "Report generation workflow" \
        "curl -s '$API_BASE_URL/reports/incidents'" \
        "200" \
        "validate_object_response"
}

# Report Generation Functions
generate_json_report() {
    print_header "Generating JSON Test Report"
    
    local report_data
    report_data=$(cat <<EOF
{
    "test_execution": {
        "timestamp": "$(date -Iseconds)",
        "total_tests": $TOTAL_TESTS,
        "passed_tests": $PASSED_TESTS,
        "failed_tests": $FAILED_TESTS,
        "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
    },
    "test_results": [
        $(IFS=','; echo "${TEST_RESULTS[*]}")
    ]
}
EOF
    )
    
    echo "$report_data" | jq . > "$TEST_REPORT_FILE"
    print_pass "JSON report saved to: $TEST_REPORT_FILE"
}

generate_html_report() {
    print_header "Generating HTML Test Report"
    
    cat > "$HTML_REPORT_FILE" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: flex; justify-content: space-around; margin-bottom: 30px; }
        .metric { text-align: center; padding: 15px; border-radius: 5px; }
        .total { background-color: #e3f2fd; }
        .passed { background-color: #e8f5e8; }
        .failed { background-color: #ffebee; }
        .success-rate { background-color: #f3e5f5; }
        .test-section { margin-bottom: 30px; }
        .test-item { margin-bottom: 15px; padding: 15px; border-left: 4px solid #ddd; background-color: #fafafa; }
        .test-item.passed { border-left-color: #4caf50; }
        .test-item.failed { border-left-color: #f44336; }
        .test-name { font-weight: bold; color: #333; }
        .test-description { color: #666; margin: 5px 0; }
        .test-details { font-size: 0.9em; color: #777; }
        .curl-command { background: #f8f8f8; padding: 8px; border-radius: 3px; font-family: monospace; margin: 5px 0; word-break: break-all; }
        .response { background: #f0f0f0; padding: 8px; border-radius: 3px; font-family: monospace; max-height: 150px; overflow-y: auto; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>API Test Report</h1>
            <p>Generated on: <span id="timestamp"></span></p>
        </div>
        <div class="summary">
            <div class="metric total">
                <h3 id="total-tests">0</h3>
                <p>Total Tests</p>
            </div>
            <div class="metric passed">
                <h3 id="passed-tests">0</h3>
                <p>Passed</p>
            </div>
            <div class="metric failed">
                <h3 id="failed-tests">0</h3>
                <p>Failed</p>
            </div>
            <div class="metric success-rate">
                <h3 id="success-rate">0%</h3>
                <p>Success Rate</p>
            </div>
        </div>
        <div id="test-results"></div>
    </div>

    <script>
        // Test data will be injected here
        const testData = TEST_DATA_PLACEHOLDER;
        
        // Update summary
        document.getElementById('timestamp').textContent = testData.test_execution.timestamp;
        document.getElementById('total-tests').textContent = testData.test_execution.total_tests;
        document.getElementById('passed-tests').textContent = testData.test_execution.passed_tests;
        document.getElementById('failed-tests').textContent = testData.test_execution.failed_tests;
        document.getElementById('success-rate').textContent = testData.test_execution.success_rate + '%';
        
        // Render test results
        const resultsContainer = document.getElementById('test-results');
        testData.test_results.forEach(test => {
            const testItem = document.createElement('div');
            testItem.className = `test-item ${test.passed ? 'passed' : 'failed'}`;
            testItem.innerHTML = `
                <div class="test-name">${test.test_name} ${test.passed ? '✓' : '✗'}</div>
                <div class="test-description">${test.description}</div>
                <div class="test-details">
                    <strong>Duration:</strong> ${test.duration}s | 
                    <strong>Status:</strong> ${test.actual_status} | 
                    <strong>Time:</strong> ${test.timestamp}
                </div>
                <div class="curl-command"><strong>Command:</strong> ${test.curl_command}</div>
                <div class="response"><strong>Response:</strong> ${JSON.stringify(JSON.parse(test.response), null, 2)}</div>
                ${!test.passed ? `<div style="color: red; font-weight: bold;">Error: ${test.error_message}</div>` : ''}
            `;
            resultsContainer.appendChild(testItem);
        });
    </script>
</body>
</html>
EOF

    # Inject test data into HTML
    local json_data
    json_data=$(cat "$TEST_REPORT_FILE")
    sed -i.bak "s/TEST_DATA_PLACEHOLDER/$json_data/g" "$HTML_REPORT_FILE" && rm "$HTML_REPORT_FILE.bak"
    
    print_pass "HTML report saved to: $HTML_REPORT_FILE"
}

display_summary() {
    print_header "Test Execution Summary"
    
    echo -e "Total Tests: ${BLUE}$TOTAL_TESTS${NC}"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        local success_rate
        success_rate=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
        echo -e "Success Rate: ${YELLOW}$success_rate%${NC}"
    fi
    
    echo -e "\nReports generated:"
    echo -e "  JSON: ${CYAN}$TEST_REPORT_FILE${NC}"
    echo -e "  HTML: ${CYAN}$HTML_REPORT_FILE${NC}"
    echo -e "  Log: ${CYAN}$LOG_FILE${NC}"
}

# Main execution functions
run_all_tests() {
    setup_test_environment
    
    run_original_features_tests
    run_phase1_enhancement_tests
    run_integration_tests
    run_performance_tests
    run_error_handling_tests
    run_end_to_end_tests
    
    generate_json_report
    generate_html_report
    display_summary
    
    cleanup_test_environment
}

# Command line interface
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo "Comprehensive API testing script with curl commands"
    echo ""
    echo "Options:"
    echo "  all                    Run all test categories (default)"
    echo "  original               Run original features tests"
    echo "  phase1                 Run Phase 1 enhancement tests"
    echo "  integration            Run integration tests"
    echo "  performance            Run performance tests"
    echo "  errors                 Run error handling tests"
    echo "  e2e                    Run end-to-end tests"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  API_BASE_URL          Base URL for API (default: http://localhost:8000)"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run all tests"
    echo "  $0 original            # Run only original features tests"
    echo "  API_BASE_URL=http://localhost:3000 $0 phase1"
}

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    command -v curl >/dev/null 2>&1 || missing_deps+=("curl")
    command -v jq >/dev/null 2>&1 || missing_deps+=("jq")
    command -v bc >/dev/null 2>&1 || missing_deps+=("bc")
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo -e "${RED}Error: Missing required dependencies: ${missing_deps[*]}${NC}"
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
}

# Main script execution
main() {
    check_dependencies
    
    case "${1:-all}" in
        "all")
            run_all_tests
            ;;
        "original")
            setup_test_environment
            run_original_features_tests
            generate_json_report
            generate_html_report
            display_summary
            cleanup_test_environment
            ;;
        "phase1")
            setup_test_environment
            run_phase1_enhancement_tests
            generate_json_report
            generate_html_report
            display_summary
            cleanup_test_environment
            ;;
        "integration")
            setup_test_environment
            run_integration_tests
            generate_json_report
            generate_html_report
            display_summary
            cleanup_test_environment
            ;;
        "performance")
            setup_test_environment
            run_performance_tests
            generate_json_report
            generate_html_report
            display_summary
            cleanup_test_environment
            ;;
        "errors")
            setup_test_environment
            run_error_handling_tests
            generate_json_report
            generate_html_report
            display_summary
            cleanup_test_environment
            ;;
        "e2e")
            setup_test_environment
            run_end_to_end_tests
            generate_json_report
            generate_html_report
            display_summary
            cleanup_test_environment
            ;;
        "-h"|"--help")
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"