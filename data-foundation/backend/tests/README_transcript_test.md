# Transcript Integration Test

This directory contains the test script to verify the transcript functionality.

## Overview

The `test_transcript_integration.py` script comprehensively tests the LLM transcript logging and API functionality to ensure:

1. **Transcript Logger**: Verifies that LLM interactions are properly captured and stored
2. **API Endpoints**: Tests all transcript API endpoints for correct functionality  
3. **Data Validation**: Ensures response data structure and content are valid
4. **Integration**: Confirms the frontend can successfully retrieve transcript data

## Prerequisites

1. **API Server Running**: The FastAPI server must be running before executing the test
   ```bash
   cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
   python3 src/ehs_extraction_api.py
   ```

2. **Dependencies**: Ensure required Python packages are installed
   ```bash
   python3 -m pip install requests
   ```

## Running the Test

### Command Line Execution
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
python3 tests/test_transcript_integration.py
```

### Making the Script Executable
```bash
chmod +x tests/test_transcript_integration.py
./tests/test_transcript_integration.py
```

## What the Test Does

### 1. Transcript Logging Test
- Clears any existing transcript entries
- Logs 6 test LLM interactions (system, user, assistant roles)
- Verifies all entries are captured correctly
- Displays transcript statistics

### 2. API Health Check Test  
- Tests the `/api/data/transcript/health` endpoint
- Verifies the transcript logger is accessible
- Confirms healthy service status

### 3. Get Transcript Test
- Tests the `/api/data/transcript` endpoint
- Retrieves all transcript entries
- Tests filtering by role (assistant only)
- Tests pagination limits
- Validates response structure

### 4. Get Stats Test
- Tests the `/api/data/transcript/stats` endpoint
- Verifies statistics calculation
- Checks for required fields (total_entries, role_counts, etc.)

### 5. Data Validation Test
- Validates transcript entry structure
- Checks data types and required fields
- Ensures role values are valid (system/user/assistant)
- Verifies timestamp and context handling

### 6. Clear Transcript Test  
- Tests the `DELETE /api/data/transcript` endpoint
- Verifies entries are properly cleared
- Confirms count accuracy after clearing

### 7. Cleanup
- Removes any remaining test data
- Ensures clean state after testing

## Expected Output

### Success Case
```
Starting Transcript Functionality Integration Test
Timestamp: 2025-01-29T10:30:00.000000
API Base URL: http://localhost:8000

============================================================
 Testing Transcript Logging
============================================================
Logging 6 test interactions...
[PASS] Transcript Logging
       Successfully logged 6 interactions

============================================================
 Testing API Health Check  
============================================================
[PASS] API Health Check
       Logger accessible: True
Current entries: 6

============================================================
 Testing Get Transcript API
============================================================
Retrieved 6 transcript entries
Filtered transcript (assistant only, limit 3): 3 entries
[PASS] Get Transcript API
       Basic and filtered retrieval working

============================================================
 Testing Get Stats API
============================================================
[PASS] Get Stats API
       Total entries: 6
Role counts: {'system': 1, 'user': 2, 'assistant': 3}
Content length: 387

============================================================
 Testing Data Validation
============================================================
[PASS] Data Validation
       All data structure and types valid
Sample entry fields: ['timestamp', 'unix_timestamp', 'role', 'content', 'context', 'entry_id']

============================================================
 Testing Clear Transcript API
============================================================
Initial entry count: 6
[PASS] Clear Transcript API
       Cleared 6 entries

============================================================
 Cleanup
============================================================
[PASS] Cleanup
       Cleared 0 remaining entries

============================================================
 Final Results
============================================================
Tests Passed: 7/7
Success Rate: 100.0%

🎉 ALL TESTS PASSED! The transcript functionality is working correctly.

The system is ready for:
- Logging LLM interactions during the ingestion workflow
- Serving transcript data to the frontend via REST API
- Providing real-time visibility into LLM processing
```

## Available API Endpoints

The test validates these transcript API endpoints:

- `GET /api/data/transcript` - Retrieve transcript entries
- `GET /api/data/transcript/stats` - Get transcript statistics  
- `GET /api/data/transcript/health` - Health check
- `DELETE /api/data/transcript` - Clear all entries

### Query Parameters for GET /api/data/transcript:
- `limit`: Maximum number of entries to return (1-10000)
- `role_filter`: Filter by role (system/user/assistant)
- `start_index`: Starting index for pagination
- `end_index`: Ending index for pagination

## Troubleshooting

### Common Issues

1. **API Server Not Running**
   ```
   ❌ Cannot connect to API server at http://localhost:8000
   Please ensure the FastAPI server is running
   ```
   **Solution**: Start the API server first

2. **Import Errors**
   ```
   ERROR: Could not import transcript logger
   ```
   **Solution**: Ensure you're running from the backend directory

3. **Port Conflicts**
   **Solution**: Check if port 8000 is available or modify the API_BASE_URL in the test script

### Debug Mode

For more detailed output, modify the logging level in the test script:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Frontend

Once this test passes, the frontend can:

1. **Real-time Monitoring**: Poll `/api/data/transcript` to show LLM interactions
2. **Filtered Views**: Use `role_filter` to show only user queries or assistant responses
3. **Statistics Dashboard**: Display transcript stats from `/api/data/transcript/stats`
4. **Pagination**: Handle large transcript logs with `start_index`/`end_index`

## Next Steps

After successful testing:

1. **Production Configuration**: Configure CORS and rate limiting appropriately
2. **Authentication**: Add authentication to transcript endpoints if needed
3. **Persistence**: Consider implementing transcript persistence to disk/database
4. **Frontend Integration**: Connect the React frontend to these endpoints
5. **Monitoring**: Set up logging and monitoring for the transcript API