#!/bin/bash

echo "=== EHS Analytics API Testing ==="
echo "Testing Enhanced RiskAssessment API endpoint"
echo "Timestamp: $(date)"
echo ""

# Test 1: Health check
echo "Test 1: Health Check"
echo "Command: curl http://localhost:8000/health"
echo "Response:"
health_response=$(curl -s http://localhost:8000/health -w "\nHTTP Status: %{http_code}\n" 2>/dev/null)
echo "$health_response"
echo ""

# Test 2: Enhanced RiskAssessment with Phase 3 fields
echo "Test 2: Enhanced RiskAssessment API with Phase 3 fields"
echo "Command: curl -X POST http://localhost:8000/api/v1/analytics/query"
echo "Request body:"
cat << 'JSON'
{
  "query": "What are the risk factors for water consumption at our facilities?",
  "include_recommendations": true,
  "risk_domains": ["water"],
  "include_forecast": true,
  "anomaly_detection": true,
  "time_range": "last_30_days"
}
JSON
echo ""
echo "Response:"
analytics_response=$(curl -s -X POST "http://localhost:8000/api/v1/analytics/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the risk factors for water consumption at our facilities?",
       "include_recommendations": true,
       "risk_domains": ["water"],
       "include_forecast": true,
       "anomaly_detection": true,
       "time_range": "last_30_days"           
     }' -w "\nHTTP Status: %{http_code}\n" 2>/dev/null)

echo "$analytics_response"

# Try to parse JSON if possible
echo ""
echo "Attempting JSON parsing:"
echo "$analytics_response" | head -n -1 | python3 -m json.tool 2>/dev/null || echo "JSON parsing failed - response may not be valid JSON"

echo ""
echo "=== Test Analysis ==="
echo "1. Health check status: $(echo "$health_response" | tail -1 | grep -o '[0-9]*')"
echo "2. Analytics API status: $(echo "$analytics_response" | tail -1 | grep -o '[0-9]*')"
echo "3. Response contains JSON: $(echo "$analytics_response" | head -n -1 | python3 -c "import json, sys; json.load(sys.stdin); print('Yes')" 2>/dev/null || echo 'No')"
echo ""
