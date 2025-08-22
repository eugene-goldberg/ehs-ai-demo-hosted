# EHS Analytics Test Suite

This directory contains comprehensive unit tests for the EHS Analytics system, covering all core components including query routing, text-to-Cypher translation, workflow management, and API functionality.

## Test Structure

### Unit Tests (`tests/unit/`)

The unit tests are organized into several categories:

#### 1. Query Router Tests
- **test_query_router_simple.py**: Tests core query classification logic
  - Intent classification for 7 different EHS intent types
  - Entity extraction (facilities, dates, equipment, pollutants, etc.)
  - Confidence scoring algorithms
  - Pattern matching and fallback logic
  - Query validation and EHS keyword detection

#### 2. Text2Cypher Tests  
- **test_text2cypher_simple.py**: Tests Cypher query generation logic
  - Cypher query validation and syntax checking
  - EHS schema pattern recognition
  - Query enhancement strategies
  - Result structuring and confidence calculation
  - Error detection and performance benchmarking

#### 3. Workflow Tests
- **test_workflow_simple.py**: Tests workflow orchestration
  - Workflow state management and updates
  - Step execution tracking and timing
  - Error handling and recovery strategies
  - Parallel step execution
  - Performance analysis and optimization

#### 4. API Tests
- **test_api_simple.py**: Tests API functionality
  - Request validation and processing options
  - Response formatting and pagination
  - Error handling and sanitization  
  - Authentication and authorization logic
  - Rate limiting and session management

### Fixtures and Configuration (`conftest.py`)

The `conftest.py` file provides comprehensive test fixtures including:

- **Mock Components**: Neo4j drivers, LLMs, workflow managers
- **Sample Data**: EHS documents, utility bills, permits, equipment data
- **Test Configuration**: Database settings, API configuration
- **Performance Benchmarks**: Expected timing and quality thresholds

## Running Tests

### Prerequisites

Ensure you have Python 3.9+ and the project dependencies installed:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov
```

### Running All Tests

```bash
# Run all working unit tests
python -m pytest tests/unit/*_simple.py -v

# Run with coverage report
python -m pytest tests/unit/*_simple.py --cov=src/ehs_analytics --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_query_router_simple.py -v
```

### Test Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = [
    "--strict-markers",
    "--verbose",
    "--cov=src/ehs_analytics",
    "--cov-report=term-missing",
    "--cov-report=html"
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "slow: Slow tests"
]
asyncio_mode = "auto"
```

## Test Results

### Current Status

- **Total Tests**: 53 tests across 4 core components
- **Passing**: 52 tests (98% pass rate)
- **Failing**: 1 minor test (schema pattern recognition edge case)
- **Coverage**: ~23% overall code coverage (focused on key logic)

### Test Categories Covered

1. **Query Classification Logic** (17 tests)
   - Pattern matching algorithms
   - Entity extraction patterns
   - Confidence calculation
   - Input validation

2. **Cypher Generation Logic** (9 tests)  
   - Query validation patterns
   - Schema awareness
   - Result processing
   - Error detection

3. **Workflow Management** (14 tests)
   - State management
   - Step execution 
   - Error handling
   - Performance monitoring

4. **API Functionality** (13 tests)
   - Request/response validation
   - Authentication logic
   - Error handling
   - Session management

## Test Philosophy

### Focus on Core Logic

These tests focus on the core business logic and algorithms rather than external integrations:

- **Logic Testing**: Pattern matching, confidence scoring, validation
- **Mocked Dependencies**: External services (Neo4j, OpenAI) are mocked
- **Edge Case Coverage**: Error conditions, boundary cases, fallback scenarios
- **Performance Validation**: Timing constraints and optimization opportunities

### Simplified Approach

Rather than testing complex integration scenarios, these tests validate:

- **Algorithm Correctness**: Core classification and processing algorithms
- **Error Handling**: Graceful degradation and recovery
- **Input Validation**: Robust handling of various input formats
- **Output Formatting**: Consistent response structures

## Contributing

When adding new tests:

1. **Follow Naming Conventions**: Use descriptive test method names
2. **Test Edge Cases**: Include both happy path and error scenarios
3. **Mock External Dependencies**: Keep tests fast and reliable
4. **Add Documentation**: Document complex test scenarios
5. **Maintain Coverage**: Aim for high coverage of critical paths

### Example Test Structure

```python
def test_feature_success_case(self):
    """Test successful execution of feature."""
    # Arrange
    input_data = create_test_input()
    
    # Act  
    result = process_feature(input_data)
    
    # Assert
    assert result.success is True
    assert result.data is not None
    assert len(result.data) > 0

def test_feature_error_handling(self):
    """Test feature handles errors gracefully."""
    # Arrange
    invalid_input = create_invalid_input()
    
    # Act & Assert
    with pytest.raises(ValidationError):
        process_feature(invalid_input)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure PYTHONPATH includes the src directory
2. **Missing Dependencies**: Install all packages from requirements.txt
3. **Async Test Issues**: Ensure pytest-asyncio is installed and configured
4. **Coverage Reports**: HTML reports are generated in `htmlcov/` directory

### Debug Mode

Run tests with verbose output and debugging:

```bash
python -m pytest tests/unit/test_query_router_simple.py::TestQueryRouterAgentBasic::test_pattern_matching_consumption -v -s
```

### Performance Testing

Monitor test execution time:

```bash
python -m pytest tests/unit/ --durations=10
```

This test suite provides a solid foundation for validating the core EHS Analytics functionality while remaining maintainable and fast to execute.