# EHS Goals API Testing Suite

This directory contains a comprehensive testing suite for the EHS Goals API endpoints, designed to verify that all goals-related functionality is working correctly.

## Files

- **`test_goals_api.py`** - Main test script that comprehensively tests all EHS Goals API endpoints
- **`start_goals_api_server.py`** - Helper script to start the API server for testing
- **`README_goals_api_testing.md`** - This documentation file

## Test Coverage

The test suite verifies the following EHS Goals API endpoints:

### Core Endpoints
1. **Health Check** (`GET /api/goals/health`)
   - Verifies the goals service is running
   - Checks configuration validity
   - Returns service status

2. **Annual Goals** (`GET /api/goals/annual`)
   - Returns all annual EHS goals for all sites and categories
   - Validates response structure and data completeness
   - Confirms all expected sites (Algonquin Illinois, Houston Texas) are present
   - Confirms all expected categories (CO2 emissions, water consumption, waste generation) are present

3. **Site-Specific Goals** (`GET /api/goals/annual/{site_id}`)
   - Tests goals retrieval for Algonquin Illinois (`algonquin_illinois`)
   - Tests goals retrieval for Houston Texas (`houston_texas`)
   - Validates that returned goals match the requested site
   - Ensures all categories are present for each site

4. **Goals Summary** (`GET /api/goals/summary`)
   - Returns high-level summary of all goals
   - Organized by site and category
   - Validates summary structure and completeness

5. **Progress Tracking** (`GET /api/goals/progress/{site_id}`)
   - Tests progress calculation for both sites
   - Validates progress metrics structure
   - Handles cases with insufficient data gracefully
   - Returns simulated progress data when real data is unavailable

### Validation Tests
- **Error Handling**: Tests invalid site IDs return appropriate 404 errors
- **Data Consistency**: Verifies consistency between annual and site-specific endpoints
- **Response Structure**: Validates JSON structure and required fields
- **API Standards**: Checks response headers and HTTP status codes

## Expected Goals Configuration

The test suite expects the following configuration:

### Sites
- **Algonquin Illinois** (`algonquin_illinois`)
- **Houston Texas** (`houston_texas`)

### Categories (for each site)
- **CO2 Emissions** (`co2_emissions`) - Reduction targets for carbon footprint
- **Water Consumption** (`water_consumption`) - Water usage reduction goals
- **Waste Generation** (`waste_generation`) - Waste reduction objectives

### Goal Structure
Each goal should include:
- Site identifier
- Category type
- Reduction percentage target
- Baseline year (2024)
- Target year (2025)
- Unit of measurement
- Description

## Prerequisites

### Python Dependencies
```bash
pip install requests tabulate fastapi uvicorn
```

### API Server
The EHS Goals API server must be running. You can either:
1. Start your existing API server
2. Use the provided helper script: `python3 start_goals_api_server.py`

## Usage

### Basic Testing
```bash
# Run all tests against localhost:8000
python3 test_goals_api.py

# Run tests against a different server
python3 test_goals_api.py --base-url http://your-server:8080

# Enable verbose logging
python3 test_goals_api.py --verbose
```

### Starting Test Server
```bash
# Start the goals API server on default port 8000
python3 start_goals_api_server.py

# Start on custom port
python3 start_goals_api_server.py --port 8080

# Enable auto-reload for development
python3 start_goals_api_server.py --reload
```

### Full Testing Workflow
```bash
# Terminal 1: Start the API server
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
python3 -m venv test_venv
source test_venv/bin/activate
pip install fastapi uvicorn requests tabulate
python3 tmp/start_goals_api_server.py

# Terminal 2: Run the tests
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
source test_venv/bin/activate  # Use the same venv
python3 tmp/test_goals_api.py
```

## Test Output

The test script provides comprehensive output including:

### Console Output
- Real-time test execution progress
- Pass/fail status for each endpoint
- Response times and performance metrics
- Detailed error messages for failed tests

### Summary Report
- Overall test statistics (total, passed, failed, success rate)
- Detailed results table with endpoints, status codes, and response times
- API endpoint coverage summary
- Performance analysis
- Recommendations for fixing issues

### Result Files
- **Log File**: `test_goals_api.log` - Detailed execution log
- **Results File**: `test_goals_api_results.json` - Machine-readable test results

## Sample Expected Output

```
================================================================================
EHS GOALS API TEST RESULTS SUMMARY
================================================================================
Total Tests: 11
Passed: 11
Failed: 0
Success Rate: 100.0%

+--------+--------+----------------------------------+------+--------+-------------------+
| Status | Method | Endpoint                         | Code | Time   | Details           |
+========+========+==================================+======+========+===================+
| ✅ PASS | GET    | /api/goals/health               | 200  | 0.045s | Status: healthy   |
| ✅ PASS | GET    | /api/goals/annual               | 200  | 0.089s | Goals: 6          |
| ✅ PASS | GET    | /api/goals/annual/algonquin...  | 200  | 0.067s | Goals: 3          |
| ✅ PASS | GET    | /api/goals/annual/houston_texas | 200  | 0.071s | Goals: 3          |
| ✅ PASS | GET    | /api/goals/summary              | 200  | 0.054s | Sites: 2          |
+--------+--------+----------------------------------+------+--------+-------------------+

================================================================================
API ENDPOINT COVERAGE
================================================================================
Health Check             ✅ Working
Annual Goals             ✅ Working
Site Goals - Algonquin   ✅ Working
Site Goals - Houston     ✅ Working
Goals Summary            ✅ Working
Progress - Algonquin     ✅ Working
Progress - Houston       ✅ Working
```

## Troubleshooting

### Common Issues

1. **Connection Error**
   ```
   ❌ Cannot connect to server: Connection refused
   ```
   **Solution**: Ensure the API server is running on the specified port

2. **Missing Dependencies**
   ```
   Missing required dependencies: No module named 'requests'
   ```
   **Solution**: Install required packages: `pip install requests tabulate`

3. **404 Errors on Valid Endpoints**
   - Check that the EHS Goals API router is properly included
   - Verify the API server is using the correct base URL
   - Ensure the goals configuration is loaded

4. **Test Failures with "insufficient_data" Status**
   - This is expected behavior when the environmental assessment service is unavailable
   - The progress endpoints should still return valid structure with simulated data

### Debug Mode
Enable verbose logging to see detailed request/response information:
```bash
python3 test_goals_api.py --verbose
```

## Integration with CI/CD

The test script returns appropriate exit codes:
- **0**: All tests passed
- **1**: Some tests failed

This makes it suitable for integration with continuous integration systems:

```bash
# Run tests and capture result
if python3 test_goals_api.py; then
    echo "All EHS Goals API tests passed"
else
    echo "Some EHS Goals API tests failed"
    exit 1
fi
```

## Extending the Tests

To add new test cases:

1. Add a new test method to the `EHSGoalsAPITester` class
2. Include it in the `test_methods` list in `run_all_tests()`
3. Follow the existing pattern for creating `TestResult` objects
4. Update the expected coverage in the summary report

Example new test method:
```python
def test_new_endpoint(self):
    """Test a new goals endpoint"""
    endpoint = "/api/goals/new-endpoint"
    start_time = time.time()
    
    try:
        response = self.session.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
        response_time = time.time() - start_time
        
        # Your validation logic here
        success = response.status_code == 200
        
        self.test_results.append(TestResult(
            endpoint=endpoint,
            method="GET",
            status_code=response.status_code,
            success=success,
            response_time=response_time,
            data_summary="Your summary here"
        ))
    except Exception as e:
        # Handle errors
        pass
```
