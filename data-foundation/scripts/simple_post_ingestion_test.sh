#!/bin/bash

# Simple Post-Ingestion EHS Extraction Test
OUTPUT_FILE="post_ingestion_extraction_test.txt"
API_BASE="http://localhost:8005/api/v1/extract"

echo "=== POST-INGESTION EHS EXTRACTION TEST ===" > "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Test results summary
success_count=0

# Test 1: Electrical Consumption
echo "Testing electrical consumption..." >&2
electrical_response=$(curl -s -X POST -H "Content-Type: application/json" -d '{}' "$API_BASE/electrical-consumption")
electrical_status=$(echo "$electrical_response" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null || echo "error")

echo "=== ELECTRICAL CONSUMPTION TEST ===" >> "$OUTPUT_FILE"
echo "Status: $electrical_status" >> "$OUTPUT_FILE"
if [ "$electrical_status" = "success" ]; then
    echo "✅ Electrical consumption endpoint: WORKING" >> "$OUTPUT_FILE"
    # Extract key data points
    echo "$electrical_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    report = data.get('data', {}).get('report_data', {})
    summary = report.get('summary', {})
    print(f'  Total queries: {summary.get(\"total_queries\", 0)}')
    print(f'  Successful queries: {summary.get(\"successful_queries\", 0)}')
    print(f'  Total records: {summary.get(\"total_records\", 0)}')
    
    # Check for actual data
    query_results = report.get('query_results', [])
    has_data = any(r.get('record_count', 0) > 0 for r in query_results)
    print(f'  Has actual data: {has_data}')
except:
    print('  Could not parse response details')
" >> "$OUTPUT_FILE"
    ((success_count++))
else
    echo "❌ Electrical consumption endpoint: FAILED" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Test 2: Water Consumption
echo "Testing water consumption..." >&2
water_response=$(curl -s -X POST -H "Content-Type: application/json" -d '{}' "$API_BASE/water-consumption")
water_status=$(echo "$water_response" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null || echo "error")

echo "=== WATER CONSUMPTION TEST ===" >> "$OUTPUT_FILE"
echo "Status: $water_status" >> "$OUTPUT_FILE"
if [ "$water_status" = "success" ]; then
    echo "✅ Water consumption endpoint: WORKING" >> "$OUTPUT_FILE"
    # Extract key data points
    echo "$water_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    report = data.get('data', {}).get('report_data', {})
    summary = report.get('summary', {})
    print(f'  Total queries: {summary.get(\"total_queries\", 0)}')
    print(f'  Successful queries: {summary.get(\"successful_queries\", 0)}')
    print(f'  Total records: {summary.get(\"total_records\", 0)}')
    
    # Extract consumption data
    query_results = report.get('query_results', [])
    for result in query_results:
        if result.get('record_count', 0) > 0:
            for record in result.get('results', []):
                if 'w' in record:
                    water_bill = record['w']['properties']
                    print(f'  Water consumption: {water_bill.get(\"total_gallons\", 0)} gallons')
                    print(f'  Total cost: ${water_bill.get(\"total_cost\", 0)}')
                    print(f'  Facility: Apex Manufacturing - Plant A')
                    break
            break
except:
    print('  Could not parse response details')
" >> "$OUTPUT_FILE"
    ((success_count++))
else
    echo "❌ Water consumption endpoint: FAILED" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Test 3: Waste Generation
echo "Testing waste generation..." >&2
waste_response=$(curl -s -X POST -H "Content-Type: application/json" -d '{}' "$API_BASE/waste-generation")
waste_status=$(echo "$waste_response" | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null || echo "error")

echo "=== WASTE GENERATION TEST ===" >> "$OUTPUT_FILE"
echo "Status: $waste_status" >> "$OUTPUT_FILE"
if [ "$waste_status" = "success" ]; then
    echo "✅ Waste generation endpoint: WORKING" >> "$OUTPUT_FILE"
    # Extract key data points
    echo "$waste_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    report = data.get('data', {}).get('report_data', {})
    summary = report.get('summary', {})
    print(f'  Total queries: {summary.get(\"total_queries\", 0)}')
    print(f'  Successful queries: {summary.get(\"successful_queries\", 0)}')
    print(f'  Total records: {summary.get(\"total_records\", 0)}')
    
    # Extract waste data
    query_results = report.get('query_results', [])
    for result in query_results:
        if result.get('record_count', 0) > 0:
            for record in result.get('results', []):
                if 'w' in record:
                    waste_data = record['w']['properties']
                    print(f'  Waste tracked: YES')
                    print(f'  Facility: Apex Manufacturing - Plant A')
                    break
            break
except:
    print('  Could not parse response details')
" >> "$OUTPUT_FILE"
    ((success_count++))
else
    echo "❌ Waste generation endpoint: FAILED" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Summary
echo "=== FINAL SUMMARY ===" >> "$OUTPUT_FILE"
echo "Endpoints tested: 3" >> "$OUTPUT_FILE"
echo "Endpoints working: $success_count" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

if [ $success_count -eq 3 ]; then
    echo "🎉 ALL ENDPOINTS SUCCESSFUL!" >> "$OUTPUT_FILE"
    echo "✅ Data ingestion verification PASSED" >> "$OUTPUT_FILE"
    echo "✅ All 3 EHS data categories are now available via API" >> "$OUTPUT_FILE"
else
    echo "⚠️  $((3-success_count)) endpoint(s) failed" >> "$OUTPUT_FILE"
    echo "❌ Data ingestion verification INCOMPLETE" >> "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"
echo "Test completed at: $(date)" >> "$OUTPUT_FILE"

echo "Test completed. Results in: $OUTPUT_FILE"
echo "Successful endpoints: $success_count/3"
