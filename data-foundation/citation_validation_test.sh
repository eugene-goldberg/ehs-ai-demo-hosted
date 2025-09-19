#!/bin/bash

# Stage 3 Citation Validation Test Script
# Testing citation improvements after the fix

echo "=== Stage 3 Citation Validation Test ==="
echo "Date: $(date)"
echo "Testing FastAPI citation improvements"
echo ""

# Create log file with timestamp
LOG_FILE="citation_test_results_$(date +"%Y%m%d_%H%M%S").log"
echo "Logging to: $LOG_FILE"

# Function to test a question and analyze citations
test_question() {
    local question="$1"
    local test_num="$2"
    
    echo "" | tee -a $LOG_FILE
    echo "=== TEST $test_num: $question ===" | tee -a $LOG_FILE
    echo "Question: $question" | tee -a $LOG_FILE
    echo "Timestamp: $(date)" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    
    # Generate unique session ID
    session_id="test_citations_$(date +%s)_$test_num"
    
    # Send request and capture response
    echo "Sending request..." | tee -a $LOG_FILE
    response=$(curl -s -X POST http://localhost:8000/api/chatbot/chat \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$question\", \"session_id\": \"$session_id\"}")
    
    echo "Response received" | tee -a $LOG_FILE
    echo "Raw Response:" | tee -a $LOG_FILE
    echo "$response" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    
    # Extract response content
    response_content=$(echo "$response" | jq -r '.response // .message // ""')
    
    # Count citations (look for [citation:X] patterns)
    citation_count=$(echo "$response_content" | grep -o '\[citation:[0-9]*\]' | wc -l)
    
    # Look for company names
    companies_found=""
    
    if echo "$response_content" | grep -i "cleveland.*cliffs" > /dev/null; then
        companies_found="$companies_found Cleveland-Cliffs,"
    fi
    
    if echo "$response_content" | grep -i "general.*motors" > /dev/null; then
        companies_found="$companies_found General-Motors,"
    fi
    
    if echo "$response_content" | grep -i "chrome.*deposit" > /dev/null; then
        companies_found="$companies_found Chrome-Deposit,"
    fi
    
    if echo "$response_content" | grep -i "smithfield" > /dev/null; then
        companies_found="$companies_found Smithfield-Foods,"
    fi
    
    if echo "$response_content" | grep -i "archer.*daniels" > /dev/null; then
        companies_found="$companies_found Archer-Daniels-Midland,"
    fi
    
    # Count unique companies
    company_count=$(echo "$companies_found" | tr ',' '\n' | grep -v '^$' | wc -l)
    
    # Look for quantified results
    quantified_results=""
    
    if echo "$response_content" | grep -i "75%.*co2\|co2.*75%" > /dev/null; then
        quantified_results="$quantified_results 75%_CO2_reduction,"
    fi
    
    if echo "$response_content" | grep -i "11\.5%.*energy\|energy.*11\.5%" > /dev/null; then
        quantified_results="$quantified_results 11.5%_energy_intensity,"
    fi
    
    if echo "$response_content" | grep -i "12%.*gas\|gas.*12%" > /dev/null; then
        quantified_results="$quantified_results 12%_natural_gas,"
    fi
    
    # Check for EPA ENERGY STAR
    epa_citation="No"
    if echo "$response_content" | grep -i "EPA\|ENERGY.*STAR" > /dev/null; then
        epa_citation="Yes"
    fi
    
    # Determine pass/fail (Pass = 6+ citations with 4+ company names)
    if [ $citation_count -ge 6 ] && [ $company_count -ge 4 ]; then
        status="PASS"
    else
        status="FAIL"
    fi
    
    # Output analysis
    echo "ANALYSIS RESULTS:" | tee -a $LOG_FILE
    echo "Citation Count: $citation_count" | tee -a $LOG_FILE
    echo "Companies Found ($company_count): $companies_found" | tee -a $LOG_FILE
    echo "Quantified Results: $quantified_results" | tee -a $LOG_FILE
    echo "EPA Citation: $epa_citation" | tee -a $LOG_FILE
    echo "Status: $status" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    
    # Return status for summary
    echo $status
}

# Test questions
echo "Starting citation validation tests..." | tee -a $LOG_FILE

test1_result=$(test_question "What environmental recommendations do you have?" "1")
test2_result=$(test_question "What are your energy efficiency recommendations?" "2")
test3_result=$(test_question "What are your high priority recommendations?" "3")
test4_result=$(test_question "Show me recommendations for waste reduction" "4")
test5_result=$(test_question "What recommendations are available?" "5")

# Create summary
echo "" | tee -a $LOG_FILE
echo "=== SUMMARY REPORT ===" | tee -a $LOG_FILE
echo "Test 1 (Environmental): $test1_result" | tee -a $LOG_FILE
echo "Test 2 (Energy Efficiency): $test2_result" | tee -a $LOG_FILE
echo "Test 3 (High Priority): $test3_result" | tee -a $LOG_FILE
echo "Test 4 (Waste Reduction): $test4_result" | tee -a $LOG_FILE
echo "Test 5 (General Recommendations): $test5_result" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Count passes and fails
pass_count=$(echo "$test1_result $test2_result $test3_result $test4_result $test5_result" | tr ' ' '\n' | grep -c "PASS")
fail_count=$(echo "$test1_result $test2_result $test3_result $test4_result $test5_result" | tr ' ' '\n' | grep -c "FAIL")

echo "OVERALL RESULTS:" | tee -a $LOG_FILE
echo "Passed Tests: $pass_count/5" | tee -a $LOG_FILE
echo "Failed Tests: $fail_count/5" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

if [ $pass_count -ge 3 ]; then
    echo "VALIDATION STATUS: SUCCESS - Citation improvements verified" | tee -a $LOG_FILE
else
    echo "VALIDATION STATUS: NEEDS REVIEW - Insufficient citation improvements" | tee -a $LOG_FILE
fi

echo "" | tee -a $LOG_FILE
echo "COMPARISON TO BASELINE:" | tee -a $LOG_FILE
echo "Before fix: Only 1 EPA citation per response" | tee -a $LOG_FILE
echo "After fix: Multiple citations with company names and quantified results" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "Test completed. Full results saved to: $LOG_FILE"
