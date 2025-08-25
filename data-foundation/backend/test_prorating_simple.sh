#!/bin/bash

# Simple Pro-Rating API Test Script
# Tests the actual double-prefix endpoints with real document IDs

echo "=== Testing Pro-Rating API ==="
echo

# Base URL
BASE_URL="http://localhost:8000"

# Document IDs for testing
DOC1="electric_bill_20250823_095948_505"
DOC2="water_bill_20250823_100019_766"

# Facility ID for testing
FACILITY_ID="facility_001"

# Test 1: Health Check
echo "1. Testing Health Check..."
curl -X GET "${BASE_URL}/api/v1/prorating/api/v1/prorating/health" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 2: Process first document
echo "2. Processing document: ${DOC1}"
curl -X POST "${BASE_URL}/api/v1/prorating/api/v1/prorating/process/${DOC1}" \
  -H "Content-Type: application/json" \
  -d '{
    "billing_period": {
      "start_date": "2025-07-01",
      "end_date": "2025-07-31"
    },
    "facility_id": "facility_001",
    "allocation_method": "equal_distribution"
  }' \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 3: Process second document
echo "3. Processing document: ${DOC2}"
curl -X POST "${BASE_URL}/api/v1/prorating/api/v1/prorating/process/${DOC2}" \
  -H "Content-Type: application/json" \
  -d '{
    "billing_period": {
      "start_date": "2025-07-01",
      "end_date": "2025-07-31"
    },
    "facility_id": "facility_002",
    "allocation_method": "consumption_based"
  }' \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 4: Get allocations for first document
echo "4. Getting allocations for document: ${DOC1}"
curl -X GET "${BASE_URL}/api/v1/prorating/api/v1/prorating/allocations/${DOC1}" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 5: Get allocations for second document
echo "5. Getting allocations for document: ${DOC2}"
curl -X GET "${BASE_URL}/api/v1/prorating/api/v1/prorating/allocations/${DOC2}" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 6: Get facility allocations
echo "6. Getting facility allocations for facility: ${FACILITY_ID}"
curl -X GET "${BASE_URL}/api/v1/prorating/api/v1/prorating/facility/${FACILITY_ID}/allocations" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 7: Generate monthly report
echo "7. Generating monthly report..."
curl -X POST "${BASE_URL}/api/v1/prorating/api/v1/prorating/monthly-report" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_id": "facility_001",
    "year": 2025,
    "month": 7,
    "report_format": "detailed"
  }' \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 8: Batch process documents
echo "8. Batch processing documents..."
curl -X POST "${BASE_URL}/api/v1/prorating/api/v1/prorating/batch-process" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": [
      "electric_bill_20250823_095948_505",
      "water_bill_20250823_100019_766"
    ],
    "billing_period": {
      "start_date": "2025-08-01",
      "end_date": "2025-08-31"
    },
    "allocation_method": "equal_distribution",
    "facility_mapping": {
      "electric_bill_20250823_095948_505": "facility_001",
      "water_bill_20250823_100019_766": "facility_002"
    }
  }' \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

echo "=== Test Complete ==="