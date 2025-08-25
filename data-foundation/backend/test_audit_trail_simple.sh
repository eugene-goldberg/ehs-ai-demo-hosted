#!/bin/bash

# Simple Audit Trail API Test Script
# Tests the actual double-prefix endpoints with real document IDs

echo "=== Testing Audit Trail API ==="
echo

# Base URL
BASE_URL="http://localhost:8000"

# Document IDs for testing
DOC1="electric_bill_20250823_095948_505"
DOC2="water_bill_20250823_100019_766"

# Test 1: Health Check
echo "1. Testing Health Check..."
curl -X GET "${BASE_URL}/api/v1/audit-trail/api/v1/documents/audit/health" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 2: Get audit info for first document
echo "2. Getting audit info for document: ${DOC1}"
curl -X GET "${BASE_URL}/api/v1/audit-trail/api/v1/documents/${DOC1}/audit_info" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 3: Get audit info for second document
echo "3. Getting audit info for document: ${DOC2}"
curl -X GET "${BASE_URL}/api/v1/audit-trail/api/v1/documents/${DOC2}/audit_info" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 4: Update source for first document
echo "4. Updating source for document: ${DOC1}"
curl -X POST "${BASE_URL}/api/v1/audit-trail/api/v1/documents/${DOC1}/update_source" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "file",
    "source_location": "/uploads/electric_bill_updated.pdf",
    "source_metadata": {
      "filename": "electric_bill_updated.pdf",
      "size": 245760,
      "mime_type": "application/pdf",
      "uploaded_by": "test_user",
      "updated_reason": "corrected billing period"
    }
  }' \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 5: Update source for second document
echo "5. Updating source for document: ${DOC2}"
curl -X POST "${BASE_URL}/api/v1/audit-trail/api/v1/documents/${DOC2}/update_source" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "url",
    "source_location": "https://water-utility.com/bills/2025/08/bill_766.pdf",
    "source_metadata": {
      "url": "https://water-utility.com/bills/2025/08/bill_766.pdf",
      "retrieved_at": "2025-08-23T10:00:19Z",
      "content_type": "application/pdf",
      "updated_by": "test_user",
      "updated_reason": "refreshed from source"
    }
  }' \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 6: Get source file for first document
echo "6. Getting source file for document: ${DOC1}"
curl -X GET "${BASE_URL}/api/v1/audit-trail/api/v1/documents/${DOC1}/source_file" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 7: Get source file for second document
echo "7. Getting source file for document: ${DOC2}"
curl -X GET "${BASE_URL}/api/v1/audit-trail/api/v1/documents/${DOC2}/source_file" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 8: Get source URL for first document
echo "8. Getting source URL for document: ${DOC1}"
curl -X GET "${BASE_URL}/api/v1/audit-trail/api/v1/documents/${DOC1}/source_url" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

# Test 9: Get source URL for second document
echo "9. Getting source URL for document: ${DOC2}"
curl -X GET "${BASE_URL}/api/v1/audit-trail/api/v1/documents/${DOC2}/source_url" \
  -H "Content-Type: application/json" \
  -w "\nStatus: %{http_code}\n" \
  -s
echo
echo "---"

echo "=== Test Complete ==="