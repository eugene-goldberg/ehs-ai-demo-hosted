#!/bin/bash

################################################################################
# Phase 1 Integration Tests for EHS AI Demo
################################################################################
#
# This script tests the complete Phase 1 enhanced workflows and validates
# integration between Phase 1 enhancements and existing system components.
#
# Test Coverage:
# - Document upload with audit trail creation
# - Utility bill processing with pro-rating
# - Document rejection workflow
# - End-to-end workflow with all Phase 1 features
# - Integration between Phase 1 components
# - Original features compatibility with Phase 1
# - Performance comparison (with/without Phase 1)
#
# Usage:
#   ./phase1_integration_tests.sh [options]
#
# Options:
#   --api-url URL          Base URL for API (default: http://localhost:8001)
#   --output-dir DIR       Output directory for test reports (default: ./test_results)
#   --verbose              Enable verbose output
#   --performance-only     Run only performance comparison tests
#   --skip-performance     Skip performance comparison tests
#   --timeout SECONDS      Request timeout (default: 30)
#   --help                 Show this help message
#
################################################################################

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_START_TIME=$(date +%s)
TEST_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default configuration
# Use environment variable if set by test orchestrator, otherwise fall back to default
API_URL="${API_BASE_URL:-http://localhost:8001}"
OUTPUT_DIR="${SCRIPT_DIR}/test_results"
VERBOSE=false
PERFORMANCE_ONLY=false
SKIP_PERFORMANCE=false
REQUEST_TIMEOUT=30
TEST_REPORT_FILE=""
PERFORMANCE_LOG=""

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_help() {
    head -40 "$0" | tail -20
}

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        ERROR)   echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN)    echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO)    echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
        DEBUG)   if [[ "$VERBOSE" == "true" ]]; then echo -e "[$timestamp] DEBUG: $message"; fi ;;
        *)       echo "[$timestamp] $message" ;;
    esac
    
    # Log to file if specified
    if [[ -n "$TEST_REPORT_FILE" ]]; then
        echo "[$timestamp] [$level] $message" >> "$TEST_REPORT_FILE"
    fi
}

check_dependencies() {
    log "INFO" "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required tools
    for tool in curl jq python3; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        log "INFO" "Please install: sudo apt-get install curl jq python3"
        exit 1
    fi
    
    log "SUCCESS" "All dependencies available"
}

setup_test_environment() {
    log "INFO" "Setting up test environment..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Initialize test report file
    TEST_REPORT_FILE="${OUTPUT_DIR}/phase1_integration_test_report_${TEST_TIMESTAMP}.txt"
    PERFORMANCE_LOG="${OUTPUT_DIR}/phase1_performance_comparison_${TEST_TIMESTAMP}.json"
    
    # Create test report header
    cat > "$TEST_REPORT_FILE" << EOF
================================================================================
Phase 1 Integration Tests Report
================================================================================
Test Suite: EHS AI Demo Phase 1 Enhanced Workflows
Start Time: $(date)
API URL: $API_URL
Output Directory: $OUTPUT_DIR
================================================================================

EOF

    log "INFO" "Test report: $TEST_REPORT_FILE"
    log "INFO" "Performance log: $PERFORMANCE_LOG"
}

check_api_health() {
    log "INFO" "Checking API health..."
    
    local health_response
    if health_response=$(curl -s --max-time "$REQUEST_TIMEOUT" "$API_URL/test/health"); then
        if echo "$health_response" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
            log "SUCCESS" "API is healthy"
            return 0
        else
            log "ERROR" "API health check failed: $health_response"
            return 1
        fi
    else
        log "ERROR" "Failed to connect to API at $API_URL"
        log "INFO" "Make sure the comprehensive test API is running:"
        log "INFO" "  cd $PROJECT_ROOT/data-foundation/backend/src/test_api"
        log "INFO" "  python3 comprehensive_test_api.py"
        return 1
    fi
}

make_api_request() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local description="$4"
    
    log "DEBUG" "Making $method request to $endpoint"
    log "DEBUG" "Request data: $data"
    
    local response
    local http_code
    local start_time end_time duration
    
    start_time=$(date +%s.%N)
    
    if [[ "$method" == "GET" ]]; then
        response=$(curl -s -w "\n%{http_code}" --max-time "$REQUEST_TIMEOUT" "$API_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" --max-time "$REQUEST_TIMEOUT" \
                       -X "$method" \
                       -H "Content-Type: application/json" \
                       -d "$data" \
                       "$API_URL$endpoint")
    fi
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    
    # Extract HTTP code (last line)
    http_code=$(echo "$response" | tail -n1)
    # Extract response body (all but last line)
    response=$(echo "$response" | head -n -1)
    
    log "DEBUG" "Response HTTP code: $http_code"
    log "DEBUG" "Response time: ${duration}s"
    log "DEBUG" "Response body: $response"
    
    if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
        echo "$response"
        return 0
    else
        log "ERROR" "$description failed (HTTP $http_code)"
        log "ERROR" "Response: $response"
        return 1
    fi
}

record_test_result() {
    local test_name="$1"
    local status="$2"
    local execution_time="$3"
    local details="$4"
    local error_message="${5:-}"
    
    ((TOTAL_TESTS++))
    
    case $status in
        PASS)
            ((PASSED_TESTS++))
            log "SUCCESS" "✓ $test_name (${execution_time}ms)"
            ;;
        FAIL)
            ((FAILED_TESTS++))
            log "ERROR" "✗ $test_name (${execution_time}ms)"
            if [[ -n "$error_message" ]]; then
                log "ERROR" "  Error: $error_message"
            fi
            ;;
        SKIP)
            ((SKIPPED_TESTS++))
            log "WARN" "○ $test_name (SKIPPED)"
            ;;
        *)
            ((FAILED_TESTS++))
            log "ERROR" "? $test_name (UNKNOWN STATUS: $status)"
            ;;
    esac
    
    # Log to report file
    cat >> "$TEST_REPORT_FILE" << EOF

Test: $test_name
Status: $status
Execution Time: ${execution_time}ms
Details: $details
$(if [[ -n "$error_message" ]]; then echo "Error: $error_message"; fi)
$(echo "----------------------------------------")

EOF
}

################################################################################
# Phase 1 Integration Test Functions
################################################################################

test_document_upload_with_audit_trail() {
    log "INFO" "Testing document upload with audit trail creation..."
    
    local test_name="Document Upload with Audit Trail"
    local start_time=$(date +%s.%N)
    
    # Test data for document upload
    local test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "document_type": "utility_bill",
        "filename": "test_utility_bill.pdf",
        "user_id": "test_user_001",
        "enable_audit": true,
        "enable_phase1": true
    },
    "mock_data": {
        "document_content": "Mock utility bill with consumption data: 250 kWh usage, billing period Jan 2024",
        "file_size": 2048,
        "mime_type": "application/pdf"
    }
}
EOF
    )
    
    # Make API request to document upload endpoint
    local response
    if response=$(make_api_request "POST" "/test/original/document-upload" "$test_data" "document upload with audit trail"); then
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        
        # Parse response and check for audit trail
        local status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
        local audit_present=$(echo "$response" | jq -e '.details | has("audit_trail")' > /dev/null 2>&1 && echo "true" || echo "false")
        
        if [[ "$status" == "PASS" && "$audit_present" == "true" ]]; then
            # Follow up with audit trail verification
            test_audit_trail_verification "$response"
            record_test_result "$test_name" "PASS" "$duration_ms" "Document uploaded successfully with audit trail"
        else
            record_test_result "$test_name" "FAIL" "$duration_ms" "Upload failed or audit trail missing" "Status: $status, Audit present: $audit_present"
        fi
    else
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        record_test_result "$test_name" "FAIL" "$duration_ms" "API request failed" "Could not connect to upload endpoint"
    fi
}

test_audit_trail_verification() {
    log "INFO" "Verifying audit trail functionality..."
    
    local upload_response="$1"
    local test_name="Audit Trail Verification"
    local start_time=$(date +%s.%N)
    
    # Test audit trail endpoint
    local audit_test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "action_type": "document_processed",
        "user_id": "test_user_001",
        "resource_id": "test_doc_001"
    }
}
EOF
    )
    
    local response
    if response=$(make_api_request "POST" "/test/phase1/audit-trail" "$audit_test_data" "audit trail verification"); then
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        
        local status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
        local audit_id=$(echo "$response" | jq -r '.details.audit_entry.id // "none"')
        
        if [[ "$status" == "PASS" && "$audit_id" != "none" ]]; then
            record_test_result "$test_name" "PASS" "$duration_ms" "Audit trail created successfully (ID: $audit_id)"
        else
            record_test_result "$test_name" "FAIL" "$duration_ms" "Audit trail creation failed" "Status: $status, Audit ID: $audit_id"
        fi
    else
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        record_test_result "$test_name" "FAIL" "$duration_ms" "API request failed" "Could not connect to audit trail endpoint"
    fi
}

test_utility_bill_processing_with_pro_rating() {
    log "INFO" "Testing utility bill processing with pro-rating..."
    
    local test_name="Utility Bill Processing with Pro-Rating"
    local start_time=$(date +%s.%N)
    
    # Test data for utility bill with pro-rating
    local test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "document_type": "utility_bill",
        "billing_period": "2024-01",
        "usage_amount": 350.5,
        "rate_per_unit": 0.15,
        "enable_pro_rating": true,
        "user_allocation": {
            "department_a": 0.6,
            "department_b": 0.4
        }
    },
    "mock_data": {
        "bill_content": "Electric bill - Usage: 350.5 kWh, Rate: $0.15/kWh, Total: $52.58",
        "account_number": "ACCT-123456",
        "service_address": "123 Industrial Blvd"
    }
}
EOF
    )
    
    # Test pro-rating functionality
    local response
    if response=$(make_api_request "POST" "/test/phase1/pro-rating" "$test_data" "utility bill pro-rating"); then
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        
        local status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
        local usage_percentage=$(echo "$response" | jq -r '.details.pro_rating_result.usage_percentage // 0')
        local total_cost=$(echo "$response" | jq -r '.details.pro_rating_result.total_cost // 0')
        
        if [[ "$status" == "PASS" ]] && (( $(echo "$usage_percentage > 0" | bc -l) )); then
            log "INFO" "Pro-rating calculation: ${usage_percentage}% usage, \$${total_cost} total cost"
            record_test_result "$test_name" "PASS" "$duration_ms" "Pro-rating calculated successfully (${usage_percentage}% usage)"
        else
            record_test_result "$test_name" "FAIL" "$duration_ms" "Pro-rating calculation failed" "Status: $status, Usage: ${usage_percentage}%"
        fi
    else
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        record_test_result "$test_name" "FAIL" "$duration_ms" "API request failed" "Could not connect to pro-rating endpoint"
    fi
}

test_document_rejection_workflow() {
    log "INFO" "Testing document rejection workflow..."
    
    local test_name="Document Rejection Workflow"
    local start_time=$(date +%s.%N)
    
    # Test data for document rejection
    local test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "document_id": "test_doc_reject_001",
        "rejection_trigger": "quality_threshold",
        "quality_score": 0.45,
        "quality_threshold": 0.7,
        "auto_retry": true,
        "max_retries": 3
    },
    "mock_data": {
        "document_content": "Low quality scan with illegible text and poor image resolution",
        "rejection_categories": ["poor_image_quality", "illegible_text", "incomplete_data"]
    }
}
EOF
    )
    
    local response
    if response=$(make_api_request "POST" "/test/phase1/rejection-tracking" "$test_data" "document rejection workflow"); then
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        
        local status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
        local rejection_id=$(echo "$response" | jq -r '.details.rejection_record.rejection_id // "none"')
        local rejection_reason=$(echo "$response" | jq -r '.details.rejection_record.rejection_reason // "unknown"')
        
        if [[ "$status" == "PASS" && "$rejection_id" != "none" ]]; then
            log "INFO" "Document rejected: ID=$rejection_id, Reason=$rejection_reason"
            record_test_result "$test_name" "PASS" "$duration_ms" "Rejection workflow completed (ID: $rejection_id, Reason: $rejection_reason)"
        else
            record_test_result "$test_name" "FAIL" "$duration_ms" "Rejection workflow failed" "Status: $status, Rejection ID: $rejection_id"
        fi
    else
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        record_test_result "$test_name" "FAIL" "$duration_ms" "API request failed" "Could not connect to rejection tracking endpoint"
    fi
}

test_end_to_end_workflow_with_phase1() {
    log "INFO" "Testing end-to-end workflow with Phase 1 features..."
    
    local test_name="End-to-End Workflow with Phase 1 Features"
    local start_time=$(date +%s.%N)
    
    # Test comprehensive workflow
    local test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "workflow_type": "complete_document_processing",
        "enable_phase1_features": true,
        "document_type": "environmental_permit",
        "user_id": "integration_test_user",
        "track_performance": true
    },
    "mock_data": {
        "document": {
            "filename": "environmental_permit_2024.pdf",
            "content": "Environmental discharge permit for facility XYZ with emission limits and compliance requirements",
            "metadata": {
                "permit_type": "air_emission",
                "facility_id": "FAC-001",
                "expiry_date": "2025-12-31"
            }
        }
    }
}
EOF
    )
    
    local response
    if response=$(make_api_request "POST" "/test/e2e/complete-user-journey" "$test_data" "end-to-end Phase 1 workflow"); then
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        
        local status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
        local completed_steps=$(echo "$response" | jq -r '.details.journey_steps | keys | length')
        local audit_logged=$(echo "$response" | jq -e '.details.journey_steps.audit_logged.status == "completed"' > /dev/null 2>&1 && echo "true" || echo "false")
        
        if [[ "$status" == "PASS" && "$audit_logged" == "true" ]] && (( completed_steps > 5 )); then
            log "INFO" "End-to-end workflow completed: $completed_steps steps, audit trail created"
            record_test_result "$test_name" "PASS" "$duration_ms" "Complete workflow with Phase 1 features ($completed_steps steps completed)"
        else
            record_test_result "$test_name" "FAIL" "$duration_ms" "Workflow incomplete or missing Phase 1 features" "Status: $status, Steps: $completed_steps, Audit: $audit_logged"
        fi
    else
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        record_test_result "$test_name" "FAIL" "$duration_ms" "API request failed" "Could not connect to E2E workflow endpoint"
    fi
}

test_phase1_component_integration() {
    log "INFO" "Testing integration between Phase 1 components..."
    
    local test_name="Phase 1 Component Integration"
    local start_time=$(date +%s.%N)
    
    # Test integration between audit trail, pro-rating, and rejection tracking
    local test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "integration_scenario": "multi_component_workflow",
        "components": ["audit_trail", "pro_rating", "rejection_tracking"],
        "document_type": "utility_invoice",
        "simulate_rejection": false,
        "enable_all_phase1": true
    }
}
EOF
    )
    
    local response
    if response=$(make_api_request "POST" "/test/integration/db-llm-integration" "$test_data" "Phase 1 component integration"); then
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        
        local status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
        local integration_steps=$(echo "$response" | jq -r '.details.integration_steps | length')
        local all_successful=$(echo "$response" | jq -e '.details.integration_steps | all(.success == true)' > /dev/null 2>&1 && echo "true" || echo "false")
        
        if [[ "$status" == "PASS" && "$all_successful" == "true" ]]; then
            log "INFO" "All Phase 1 components integrated successfully ($integration_steps steps)"
            record_test_result "$test_name" "PASS" "$duration_ms" "All Phase 1 components integrated ($integration_steps integration points)"
        else
            record_test_result "$test_name" "FAIL" "$duration_ms" "Component integration failed" "Status: $status, Steps: $integration_steps, All successful: $all_successful"
        fi
    else
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        record_test_result "$test_name" "FAIL" "$duration_ms" "API request failed" "Could not connect to integration test endpoint"
    fi
}

test_original_features_compatibility() {
    log "INFO" "Testing original features compatibility with Phase 1..."
    
    local test_name="Original Features Compatibility"
    local start_time=$(date +%s.%N)
    
    # Test that original features still work with Phase 1 enabled
    local test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "compatibility_mode": "phase1_enabled",
        "test_original_features": true,
        "features_to_test": [
            "document_upload",
            "entity_extraction",
            "qa_system"
        ]
    }
}
EOF
    )
    
    local response
    if response=$(make_api_request "POST" "/test/regression/feature-compatibility" "$test_data" "original features compatibility"); then
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        
        local status=$(echo "$response" | jq -r '.status // "UNKNOWN"')
        local compatibility_tests=$(echo "$response" | jq -r '.details.compatibility_tests | length')
        local all_compatible=$(echo "$response" | jq -e '.details.compatibility_tests | all(.compatible == true)' > /dev/null 2>&1 && echo "true" || echo "false")
        
        if [[ "$status" == "PASS" && "$all_compatible" == "true" ]]; then
            log "SUCCESS" "All original features compatible with Phase 1 ($compatibility_tests tests passed)"
            record_test_result "$test_name" "PASS" "$duration_ms" "Original features fully compatible with Phase 1 ($compatibility_tests compatibility tests)"
        else
            record_test_result "$test_name" "FAIL" "$duration_ms" "Compatibility issues found" "Status: $status, Tests: $compatibility_tests, All compatible: $all_compatible"
        fi
    else
        local end_time=$(date +%s.%N)
        local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
        record_test_result "$test_name" "FAIL" "$duration_ms" "API request failed" "Could not connect to compatibility test endpoint"
    fi
}

test_performance_comparison() {
    log "INFO" "Running performance comparison (with/without Phase 1)..."
    
    local test_name="Performance Comparison (Phase 1 vs Original)"
    local start_time=$(date +%s.%N)
    
    # Test performance without Phase 1
    log "INFO" "Testing performance without Phase 1 enhancements..."
    local original_test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "concurrent_requests": 5,
        "request_duration": 0.1,
        "phase1_enabled": false,
        "test_type": "baseline_performance"
    }
}
EOF
    )
    
    local original_response
    if original_response=$(make_api_request "POST" "/test/performance/load-test" "$original_test_data" "original performance test"); then
        local original_success_rate=$(echo "$original_response" | jq -r '.details.success_rate // 0')
        local original_time=$(echo "$original_response" | jq -r '.execution_time_ms // 0')
        
        log "DEBUG" "Original performance: ${original_success_rate}% success rate, ${original_time}ms execution time"
        
        # Test performance with Phase 1
        log "INFO" "Testing performance with Phase 1 enhancements..."
        local phase1_test_data=$(cat << 'EOF'
{
    "test_parameters": {
        "concurrent_requests": 5,
        "request_duration": 0.1,
        "phase1_enabled": true,
        "test_type": "enhanced_performance"
    }
}
EOF
        )
        
        local phase1_response
        if phase1_response=$(make_api_request "POST" "/test/performance/load-test" "$phase1_test_data" "Phase 1 performance test"); then
            local phase1_success_rate=$(echo "$phase1_response" | jq -r '.details.success_rate // 0')
            local phase1_time=$(echo "$phase1_response" | jq -r '.execution_time_ms // 0')
            
            log "DEBUG" "Phase 1 performance: ${phase1_success_rate}% success rate, ${phase1_time}ms execution time"
            
            # Calculate performance impact
            local time_increase=$(echo "scale=2; ($phase1_time - $original_time) / $original_time * 100" | bc -l)
            local acceptable_overhead=$(echo "$time_increase < 50" | bc -l)  # Less than 50% overhead is acceptable
            
            # Log performance comparison to JSON file
            cat > "$PERFORMANCE_LOG" << EOF
{
    "test_timestamp": "$(date -Iseconds)",
    "original_performance": {
        "success_rate": $original_success_rate,
        "execution_time_ms": $original_time,
        "concurrent_requests": 5
    },
    "phase1_performance": {
        "success_rate": $phase1_success_rate,
        "execution_time_ms": $phase1_time,
        "concurrent_requests": 5
    },
    "comparison": {
        "time_increase_percent": $time_increase,
        "acceptable_overhead": $acceptable_overhead,
        "success_rate_maintained": $(echo "$phase1_success_rate >= $original_success_rate" | bc -l)
    }
}
EOF
            
            local end_time=$(date +%s.%N)
            local duration_ms=$(echo "($end_time - $start_time) * 1000" | bc -l | cut -d. -f1)
            
            if [[ "$acceptable_overhead" == "1" ]] && (( $(echo "$phase1_success_rate >= $original_success_rate" | bc -l) )); then
                log "SUCCESS" "Performance impact acceptable: +${time_increase}% execution time"
                record_test_result "$test_name" "PASS" "$duration_ms" "Performance impact acceptable (+${time_increase}% time, ${phase1_success_rate}% vs ${original_success_rate}% success rate)"
            else
                log "WARN" "Performance impact significant: +${time_increase}% execution time"
                record_test_result "$test_name" "FAIL" "$duration_ms" "Performance impact too high" "Time increase: +${time_increase}%, Success rates: ${phase1_success_rate}% vs ${original_success_rate}%"
            fi
        else
            record_test_result "$test_name" "FAIL" "0" "Phase 1 performance test failed" "Could not test Phase 1 performance"
        fi
    else
        record_test_result "$test_name" "FAIL" "0" "Original performance test failed" "Could not test baseline performance"
    fi
    
    log "INFO" "Performance comparison results saved to: $PERFORMANCE_LOG"
}

################################################################################
# Main Test Execution
################################################################################

run_integration_tests() {
    log "INFO" "Starting Phase 1 integration tests..."
    
    if [[ "$PERFORMANCE_ONLY" == "true" ]]; then
        log "INFO" "Running performance comparison only..."
        test_performance_comparison
        return
    fi
    
    # Core integration tests
    test_document_upload_with_audit_trail
    test_utility_bill_processing_with_pro_rating
    test_document_rejection_workflow
    test_end_to_end_workflow_with_phase1
    test_phase1_component_integration
    test_original_features_compatibility
    
    # Performance comparison (unless skipped)
    if [[ "$SKIP_PERFORMANCE" != "true" ]]; then
        test_performance_comparison
    fi
}

generate_test_summary() {
    local test_end_time=$(date +%s)
    local total_duration=$((test_end_time - TEST_START_TIME))
    
    log "INFO" "Generating test summary..."
    
    cat >> "$TEST_REPORT_FILE" << EOF

================================================================================
TEST SUMMARY
================================================================================
Total Tests: $TOTAL_TESTS
Passed: $PASSED_TESTS
Failed: $FAILED_TESTS  
Skipped: $SKIPPED_TESTS

Success Rate: $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)%
Total Duration: ${total_duration}s

Test Results Location: $OUTPUT_DIR
Performance Comparison: $PERFORMANCE_LOG

================================================================================
End Time: $(date)
================================================================================
EOF

    # Console summary
    echo ""
    log "INFO" "=================================================================================="
    log "INFO" "PHASE 1 INTEGRATION TESTS COMPLETE"
    log "INFO" "=================================================================================="
    log "INFO" "Total Tests: $TOTAL_TESTS"
    if [[ $PASSED_TESTS -gt 0 ]]; then
        log "SUCCESS" "Passed: $PASSED_TESTS"
    fi
    if [[ $FAILED_TESTS -gt 0 ]]; then
        log "ERROR" "Failed: $FAILED_TESTS"
    fi
    if [[ $SKIPPED_TESTS -gt 0 ]]; then
        log "WARN" "Skipped: $SKIPPED_TESTS"
    fi
    
    local success_rate=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)
    log "INFO" "Success Rate: ${success_rate}%"
    log "INFO" "Total Duration: ${total_duration}s"
    log "INFO" "=================================================================================="
    log "INFO" "Detailed report: $TEST_REPORT_FILE"
    if [[ "$SKIP_PERFORMANCE" != "true" ]]; then
        log "INFO" "Performance comparison: $PERFORMANCE_LOG"
    fi
    log "INFO" "=================================================================================="
    
    # Return appropriate exit code
    if [[ $FAILED_TESTS -gt 0 ]]; then
        return 1
    else
        return 0
    fi
}

################################################################################
# Script Entry Point
################################################################################

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --api-url)
                API_URL="$2"
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
            --performance-only)
                PERFORMANCE_ONLY=true
                shift
                ;;
            --skip-performance)
                SKIP_PERFORMANCE=true
                shift
                ;;
            --timeout)
                REQUEST_TIMEOUT="$2"
                shift 2
                ;;
            --help)
                print_help
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                print_help
                exit 1
                ;;
        esac
    done
    
    # Validate configuration
    if [[ "$PERFORMANCE_ONLY" == "true" && "$SKIP_PERFORMANCE" == "true" ]]; then
        log "ERROR" "Cannot use --performance-only and --skip-performance together"
        exit 1
    fi
    
    # Main execution flow
    log "INFO" "Phase 1 Integration Tests Starting..."
    log "INFO" "API URL: $API_URL"
    log "INFO" "Output Directory: $OUTPUT_DIR"
    log "INFO" "Request Timeout: ${REQUEST_TIMEOUT}s"
    
    check_dependencies
    setup_test_environment
    
    if ! check_api_health; then
        log "ERROR" "API health check failed. Cannot proceed with tests."
        exit 1
    fi
    
    run_integration_tests
    generate_test_summary
    
    # Exit with appropriate code
    if [[ $FAILED_TESTS -gt 0 ]]; then
        log "ERROR" "Some tests failed. Check the detailed report for more information."
        exit 1
    else
        log "SUCCESS" "All tests passed successfully!"
        exit 0
    fi
}

# Execute main function with all arguments
main "$@"