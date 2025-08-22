# EHS Data Foundation - Test Suite

This directory contains comprehensive tests for the EHS Data Extraction API and related components.

## Test Structure

- `test_ehs_extraction_api.py` - Main API endpoint tests
- `__init__.py` - Package initialization

## Test Categories

### Unit Tests
- Individual function and method tests
- Mock-based testing without external dependencies
- Fast execution

### Integration Tests  
- Component interaction tests
- Database integration tests
- Workflow integration tests

### API Tests
- HTTP endpoint testing
- Request/response validation
- Error handling tests

### Performance Tests
- Response time validation
- Concurrent request handling
- Large dataset processing

## Running Tests

### Using the Test Runner Script
```bash
# Run all tests
python3 run_tests.py

# Run specific test types
python3 run_tests.py --test-type api
python3 run_tests.py --test-type unit

# Run with coverage
python3 run_tests.py --coverage

# Run specific test file
python3 run_tests.py --file tests/test_ehs_extraction_api.py

# Run specific test function
python3 run_tests.py --function test_health_check_success

# Install dependencies and run tests
python3 run_tests.py --install-deps --coverage
```

### Using pytest directly
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test class
pytest tests/test_ehs_extraction_api.py::TestHealthEndpoint

# Run tests matching pattern
pytest -k "test_electrical"

# Run tests in parallel
pytest -n auto
```

## Test Features

### Fixtures
- `test_client` - FastAPI test client
- `mock_workflow` - Mocked DataExtractionWorkflow
- `mock_neo4j_env` - Mocked environment variables
- `sample_*_request` - Sample request data for different endpoints

### Test Coverage

#### Health Check Endpoint
- ✅ Successful health check
- ✅ Neo4j connection failure handling

#### Electrical Consumption Endpoint
- ✅ Successful extraction with all parameters
- ✅ Minimal request data handling
- ✅ Workflow failure scenarios
- ✅ Failed extraction status handling

#### Water Consumption Endpoint
- ✅ Successful extraction
- ✅ Date range only requests
- ✅ Different output formats

#### Waste Generation Endpoint
- ✅ Successful extraction with all parameters
- ✅ Hazardous waste filtering
- ✅ Parameter validation

#### Custom Extraction Endpoint
- ✅ Valid query types
- ✅ Custom queries
- ✅ Invalid query type handling

#### Query Types Endpoint
- ✅ Available query types listing
- ✅ Content validation

#### Edge Cases & Validation
- ✅ Invalid date ranges
- ✅ Missing parameters
- ✅ Empty results handling
- ✅ Invalid JSON payloads

#### Error Handling
- ✅ Neo4j connection failures
- ✅ Workflow initialization errors
- ✅ Processing errors
- ✅ Partial query failures

#### Response Validation
- ✅ Response structure validation
- ✅ Metadata fields presence
- ✅ Error response format
- ✅ Processing time validation

#### Performance Tests
- ✅ Response time validation
- ✅ Concurrent request handling
- ✅ Large date range handling

## Mock Strategy

The tests use comprehensive mocking to avoid dependencies on:
- Neo4j database connections
- LLM API calls
- File system operations
- Network requests

This ensures tests are:
- Fast and reliable
- Independent of external services
- Predictable in results

## Environment Setup

Tests automatically mock environment variables but you can set real ones for integration testing:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password
export LLM_MODEL=gpt-4
```

## Continuous Integration

The test suite is designed to run in CI/CD pipelines with:
- Parallel execution support
- Coverage reporting
- JUnit XML output
- HTML reports

## Adding New Tests

When adding new functionality:

1. Add corresponding test classes and methods
2. Use appropriate fixtures for setup
3. Follow naming conventions (`test_*`)
4. Add appropriate markers (`@pytest.mark.slow`, etc.)
5. Update this README if adding new test categories

## Test Markers

- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests  
- `@pytest.mark.api` - API tests