#!/bin/bash
echo "Starting API endpoint test at Tue Aug 26 19:29:53 CDT 2025"
echo "Testing endpoint: http://localhost:8001/api/documents/933a00d8-ae1a-4ec8-821a-45400d6b192e"
echo "Looking for prorated_monthly_usage field with value 48390.0"
echo ""

# Test the API endpoint
echo "Making curl request..."
response="HTTP_STATUS:000"

# Extract HTTP status code
http_status=
response_body=""

echo "HTTP Status Code: "
echo ""
echo "Response Body:"
echo ""
echo ""

# Check if the response contains the expected prorated_monthly_usage field
if echo "" | grep -q "prorated_monthly_usage"; then
    echo "✓ Found prorated_monthly_usage field in response"
    
    # Extract the prorated_monthly_usage value
    usage_value=
    echo "prorated_monthly_usage value: "
    
    if [ "" = "48390.0" ]; then
        echo "✓ PASS: prorated_monthly_usage has expected value of 48390.0"
    else
        echo "✗ FAIL: prorated_monthly_usage value () does not match expected value (48390.0)"
    fi
else
    echo "✗ FAIL: prorated_monthly_usage field not found in response"
fi

echo ""
echo "Test completed at Tue Aug 26 19:29:53 CDT 2025"
