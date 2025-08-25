#!/bin/bash

# Post-Ingestion EHS Extraction Test Script
# Tests all three EHS extraction endpoints for actual data after ingestion

OUTPUT_FILE="post_ingestion_extraction_test.txt"
API_BASE="http://localhost:8005/api/v1/extract"

echo "=== POST-INGESTION EHS EXTRACTION TEST ===" > "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "Testing EHS extraction endpoints after successful data ingestion" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Function to extract key metrics from JSON response
extract_metrics() {
    local response="$1"
    local endpoint_name="$2"
    
    echo "Processing $endpoint_name response..." >&2
    
    # Check if response is valid JSON
    if ! echo "$response" | python3 -m json.tool > /dev/null 2>&1; then
        echo "Invalid JSON response for $endpoint_name" >> "$OUTPUT_FILE"
        echo "Raw response: $response" >> "$OUTPUT_FILE"
        return 1
    fi
    
    # Extract metrics using Python
    python3 << PYTHON_SCRIPT
import json
import sys

try:
    data = json.loads('''$response''')
    
    print("\n--- $endpoint_name ANALYSIS ---")
    
    # Check if data exists
    if not data or (isinstance(data, dict) and not data.get('data')):
        print("❌ NO DATA RETURNED")
        print("Response structure:", str(data)[:200] + "..." if len(str(data)) > 200 else str(data))
    else:
        # Handle different response structures
        records = []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            if 'data' in data:
                records = data['data'] if isinstance(data['data'], list) else [data['data']]
            elif 'results' in data:
                records = data['results'] if isinstance(data['results'], list) else [data['results']]
            else:
                records = [data]
        
        print(f"✅ Records returned: {len(records)}")
        
        if records:
            # Extract consumption/generation totals
            total_values = []
            facilities = set()
            emissions = []
            
            for record in records:
                if isinstance(record, dict):
                    # Look for consumption/generation values
                    for key, value in record.items():
                        if any(term in key.lower() for term in ['consumption', 'generation', 'amount', 'quantity', 'total']):
                            if isinstance(value, (int, float)):
                                total_values.append(value)
                        
                        # Look for facility information
                        if 'facility' in key.lower() and isinstance(value, str):
                            facilities.add(value)
                        
                        # Look for emissions
                        if 'emission' in key.lower() and isinstance(value, (int, float)):
                            emissions.append(value)
            
            if total_values:
                print(f"📊 Total consumption/generation values: {len(total_values)}")
                print(f"📈 Sum of values: {sum(total_values):.2f}")
                print(f"📉 Average value: {sum(total_values)/len(total_values):.2f}")
            
            if facilities:
                print(f"🏢 Facilities involved: {len(facilities)}")
                print(f"🏭 Facility names: {', '.join(list(facilities)[:5])}")
            
            if emissions:
                print(f"🌍 Emissions calculated: {len(emissions)}")
                print(f"💨 Total emissions: {sum(emissions):.2f}")
            
            # Show sample record structure
            print(f"📋 Sample record keys: {list(records[0].keys())[:10]}")
        
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing error: {e}")
except Exception as e:
    print(f"❌ Analysis error: {e}")
PYTHON_SCRIPT
}

# Test 1: Electrical Consumption
echo "=== TESTING ELECTRICAL CONSUMPTION ENDPOINT ===" >> "$OUTPUT_FILE"
echo "Endpoint: POST $API_BASE/electrical-consumption" >> "$OUTPUT_FILE"
echo "Payload: {}" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Testing electrical consumption endpoint..." >&2
electrical_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{}' \
    "$API_BASE/electrical-consumption" 2>&1)

echo "Raw electrical response:" >> "$OUTPUT_FILE"
echo "$electrical_response" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Extract metrics for electrical consumption
extract_metrics "$electrical_response" "ELECTRICAL CONSUMPTION" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Test 2: Water Consumption  
echo "=== TESTING WATER CONSUMPTION ENDPOINT ===" >> "$OUTPUT_FILE"
echo "Endpoint: POST $API_BASE/water-consumption" >> "$OUTPUT_FILE"
echo "Payload: {}" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Testing water consumption endpoint..." >&2
water_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{}' \
    "$API_BASE/water-consumption" 2>&1)

echo "Raw water response:" >> "$OUTPUT_FILE"
echo "$water_response" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Extract metrics for water consumption
extract_metrics "$water_response" "WATER CONSUMPTION" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Test 3: Waste Generation
echo "=== TESTING WASTE GENERATION ENDPOINT ===" >> "$OUTPUT_FILE"
echo "Endpoint: POST $API_BASE/waste-generation" >> "$OUTPUT_FILE"
echo "Payload: {}" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Testing waste generation endpoint..." >&2
waste_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{}' \
    "$API_BASE/waste-generation" 2>&1)

echo "Raw waste response:" >> "$OUTPUT_FILE"
echo "$waste_response" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Extract metrics for waste generation
extract_metrics "$waste_response" "WASTE GENERATION" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Summary
echo "=== TEST SUMMARY ===" >> "$OUTPUT_FILE"
echo "Timestamp: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Count successful endpoints (non-empty responses)
success_count=0

if echo "$electrical_response" | python3 -c "import json,sys; data=json.load(sys.stdin); exit(0 if data and (isinstance(data,list) and len(data)>0 or isinstance(data,dict) and data.get('data')) else 1)" 2>/dev/null; then
    echo "✅ Electrical consumption: SUCCESS (data returned)" >> "$OUTPUT_FILE"
    ((success_count++))
else
    echo "❌ Electrical consumption: FAILED (no data or error)" >> "$OUTPUT_FILE"
fi

if echo "$water_response" | python3 -c "import json,sys; data=json.load(sys.stdin); exit(0 if data and (isinstance(data,list) and len(data)>0 or isinstance(data,dict) and data.get('data')) else 1)" 2>/dev/null; then
    echo "✅ Water consumption: SUCCESS (data returned)" >> "$OUTPUT_FILE"
    ((success_count++))
else
    echo "❌ Water consumption: FAILED (no data or error)" >> "$OUTPUT_FILE"
fi

if echo "$waste_response" | python3 -c "import json,sys; data=json.load(sys.stdin); exit(0 if data and (isinstance(data,list) and len(data)>0 or isinstance(data,dict) and data.get('data')) else 1)" 2>/dev/null; then
    echo "✅ Waste generation: SUCCESS (data returned)" >> "$OUTPUT_FILE"
    ((success_count++))
else
    echo "❌ Waste generation: FAILED (no data or error)" >> "$OUTPUT_FILE"
fi

echo "" >> "$OUTPUT_FILE"
echo "Overall result: $success_count/3 endpoints returning actual data" >> "$OUTPUT_FILE"

if [ $success_count -eq 3 ]; then
    echo "🎉 ALL ENDPOINTS SUCCESSFUL - Data ingestion verified!" >> "$OUTPUT_FILE"
    exit 0
else
    echo "⚠️  Some endpoints failed - Check ingestion process" >> "$OUTPUT_FILE"
    exit 1
fi
