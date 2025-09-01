# Environmental Endpoints Verification

This directory contains comprehensive verification scripts to test all environmental endpoints after schema mapping fixes.

## Files

1. **`verify_all_environmental_endpoints.py`** - Main verification script
2. **`run_verification.sh`** - Shell script runner with environment setup
3. **`README_verification.md`** - This documentation

## What Gets Tested

### Environmental Data Endpoints (9 total)
- `/environmental/electricity/facts`
- `/environmental/electricity/risks` 
- `/environmental/electricity/recommendations`
- `/environmental/water/facts`
- `/environmental/water/risks`
- `/environmental/water/recommendations`
- `/environmental/waste/facts`
- `/environmental/waste/risks`
- `/environmental/waste/recommendations`

### Additional Tests
- Each endpoint tested with and without location filtering
- LLM assessment endpoint with various parameters
- Response structure validation
- Performance timing
- Data count verification

## Usage

### Option 1: Using the Shell Runner (Recommended)
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
./tmp/run_verification.sh
```

### Option 2: Direct Python Execution
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
source venv/bin/activate
python3 tmp/verify_all_environmental_endpoints.py
```

## Prerequisites

1. **FastAPI Server Running**: The server must be running on `localhost:8000`
   ```bash
   cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
   source venv/bin/activate
   python3 app.py
   ```

2. **Virtual Environment**: Python virtual environment with required packages

3. **Neo4j Database**: Database must be running with environmental data loaded

## Output Files

After running verification, you'll get:

1. **`tmp/verification_summary.json`** - Detailed JSON results
2. **`tmp/endpoint_verification_results.log`** - Full execution log

## Interpreting Results

### Success Indicators
- ✅ All endpoints return data (not empty lists)
- ✅ Response times are reasonable (< 5 seconds for data, < 60 seconds for LLM)
- ✅ Expected fields are present in responses
- ✅ Location filtering works when applied

### Failure Indicators
- ❌ Empty data returned (indicates schema mapping issues)
- ❌ HTTP errors (4xx, 5xx status codes)
- ❌ Missing expected fields
- ❌ Request timeouts or connection errors

## Example Output

```
Environmental Endpoints Verification Script
==================================================
[2025-01-31 10:15:30] INFO: Starting comprehensive environmental endpoints verification
[2025-01-31 10:15:30] INFO: Base URL: http://localhost:8000
[2025-01-31 10:15:31] INFO: ✓ Server is running
[2025-01-31 10:15:32] INFO: ✓ /environmental/electricity/facts - SUCCESS (15 items, 234ms)
[2025-01-31 10:15:33] INFO: ✓ /environmental/electricity/risks - SUCCESS (12 items, 198ms)
...
=== VERIFICATION SUMMARY ===
Total endpoints tested: 21
Passed: 21
Failed: 0
Success rate: 100.0%
```

## Troubleshooting

### Server Not Running
```
Error: Cannot connect to server: Connection refused
```
**Solution**: Start the FastAPI server first

### Empty Data Returned
```
✗ /environmental/electricity/facts - FAILED: No data returned
```
**Solution**: Check Neo4j database and schema mappings

### Virtual Environment Issues
```
Error: Virtual environment not found
```
**Solution**: Create and activate virtual environment as shown in prerequisites
