#!/bin/bash

# Simple Rejection Tracking API Test Script
# Tests the actual double-prefix endpoints with real document IDs

echo "=== Testing Rejection Tracking API ==="
echo

# Base URL
BASE_URL="http://localhost:8000"

# Test 1: Health Check
echo "1. Testing Health Check..."
curl -X GET "${BASE_URL}/api/v1/rejection-tracking/api/v1/documents/rejection-tracking/health" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 2: Reject first document
echo "2. Rejecting document: electric_bill_20250823_095948_505"
curl -X POST "${BASE_URL}/api/v1/rejection-tracking/api/v1/documents/electric_bill_20250823_095948_505/reject" \
  -H "Content-Type: application/json" \
  -d '{"reason": "incomplete"}' \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 3: Reject second document
echo "3. Rejecting document: water_bill_20250823_100019_766"
curl -X POST "${BASE_URL}/api/v1/rejection-tracking/api/v1/documents/water_bill_20250823_100019_766/reject" \
  -H "Content-Type: application/json" \
  -d '{"reason": "wrong_format"}' \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 4: Get all rejected documents
echo "4. Retrieving all rejected documents..."
curl -X GET "${BASE_URL}/api/v1/rejection-tracking/api/v1/documents/rejected" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

echo "=== Test Complete ==="