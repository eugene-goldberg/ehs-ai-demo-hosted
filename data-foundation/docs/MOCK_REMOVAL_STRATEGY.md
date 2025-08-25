# Mock Removal Strategy

> Created: 2025-08-23
> Version: 1.0.0
> Status: Strategy Document

## Overview

This document outlines a comprehensive strategy for removing mocks from the EHS AI Demo test suite and replacing them with real implementations. The goal is to ensure all tests validate actual functionality rather than simulated behavior while maintaining test reliability and comprehensive coverage.

## Current Mock Analysis

### 1. Mock Categories Identified

Based on analysis of the test suite, mocks fall into these primary categories:

#### A. Database Connection Mocks
- **Location**: `test_ehs_extraction_api.py`, `test_text2cypher.py`, `test_neo4j_connection.py`
- **Purpose**: Mock Neo4j driver, sessions, and connections
- **Pattern**: `Mock()`, `patch('neo4j.GraphDatabase.driver')`

#### B. LLM Integration Mocks
- **Location**: `test_text2cypher.py`, `test_phase2_cypher_generation.py`
- **Purpose**: Mock OpenAI ChatGPT, LangChain components
- **Pattern**: `ChatOpenAI(openai_api_key="mock-key")`, `patch('ChatOpenAI')`

#### C. Workflow Component Mocks
- **Location**: `test_ehs_extraction_api.py`, workflow tests
- **Purpose**: Mock DataExtractionWorkflow, Text2CypherRetriever
- **Pattern**: `Mock(spec=DataExtractionWorkflow)`, `patch('ehs_extraction_api.get_workflow')`

#### D. API Response Mocks
- **Location**: FastAPI test files
- **Purpose**: Mock external API responses and HTTP clients
- **Pattern**: `Mock()` return values, side effects

### 2. Mock Inventory Table

| Mock Type | File Location | Mock Target | Real Implementation | Priority |
|-----------|---------------|-------------|-------------------|----------|
| Neo4j Driver | `test_ehs_extraction_api.py:43-84` | `GraphDatabase.driver` | Real Neo4j connection | High |
| Neo4j Graph | `test_text2cypher.py:64-80` | `Neo4jGraph` | LangChain Neo4j integration | High |
| OpenAI LLM | `test_text2cypher.py:83-87` | `ChatOpenAI` | Real OpenAI API calls | High |
| Cypher Chain | `test_text2cypher.py:90-102` | `GraphCypherQAChain` | Real chain execution | High |
| Workflow | `test_ehs_extraction_api.py:44-72` | `DataExtractionWorkflow` | Real workflow instance | Medium |
| FastAPI Client | Multiple files | `TestClient` | Real HTTP client | Low |
| Environment Variables | `test_ehs_extraction_api.py:76-84` | `os.environ` | Real env config | Medium |

## Replacement Strategy

### Phase 1: Infrastructure Setup (Week 1)

#### Step 1.1: Test Environment Configuration
```bash
# Create test-specific environment configuration
cp .env.example .env.test
# Update .env.test with test database credentials
NEO4J_URI=bolt://localhost:7688  # Test instance
NEO4J_USERNAME=neo4j_test
NEO4J_PASSWORD=test_password
OPENAI_API_KEY=test_api_key_with_limits
```

#### Step 1.2: Test Database Setup
```python
# Create test database utilities
# File: tests/utils/test_database.py

import pytest
from neo4j import GraphDatabase
from contextlib import contextmanager

@pytest.fixture(scope="session")
def test_neo4j_driver():
    """Real Neo4j driver for testing."""
    uri = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7688")
    user = os.getenv("NEO4J_TEST_USER", "neo4j")
    password = os.getenv("NEO4J_TEST_PASSWORD", "testpass")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    yield driver
    driver.close()

@contextmanager
def clean_test_database(driver):
    """Context manager for clean test database state."""
    with driver.session() as session:
        # Clean up before test
        session.run("MATCH (n) DETACH DELETE n")
        yield session
        # Clean up after test
        session.run("MATCH (n) DETACH DELETE n")
```

### Phase 2: Mock Replacement (Weeks 2-4)

#### Step 2.1: Neo4j Connection Mocks → Real Connections

**Before (test_ehs_extraction_api.py:43-84):**
```python
@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    driver = Mock()
    session = Mock()
    result = Mock()
    
    session.run.return_value = result
    driver.session.return_value = session
    return driver
```

**After:**
```python
@pytest.fixture
def test_neo4j_driver():
    """Real Neo4j driver for testing."""
    from tests.utils.test_database import test_neo4j_driver
    return test_neo4j_driver()

@pytest.fixture
def clean_test_session(test_neo4j_driver):
    """Clean test database session."""
    with clean_test_database(test_neo4j_driver) as session:
        # Setup test data
        session.run("""
        CREATE (f:Facility {name: 'Test Plant', location: 'Test City'})
        CREATE (e:Equipment {name: 'Test Boiler', type: 'heating'})
        CREATE (u:UtilityBill {amount: 1500.0, utility_type: 'electricity'})
        CREATE (f)-[:CONTAINS]->(e)
        CREATE (u)-[:RECORDED_AT]->(f)
        """)
        yield session
```

#### Step 2.2: LLM Integration Mocks → Real API Calls

**Before (test_text2cypher.py:83-87):**
```python
@pytest.fixture
def mock_llm():
    """Mock OpenAI LLM."""
    llm = Mock()
    llm.model_name = "gpt-3.5-turbo"
    return llm
```

**After:**
```python
@pytest.fixture
def test_llm():
    """Real OpenAI LLM with test configuration."""
    from langchain.chat_models import ChatOpenAI
    
    api_key = os.getenv("OPENAI_TEST_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_TEST_API_KEY not configured")
    
    return ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=100,  # Limit for cost control
        request_timeout=10  # Fail fast
    )
```

#### Step 2.3: Workflow Component Mocks → Real Instances

**Before (test_ehs_extraction_api.py:44-72):**
```python
@pytest.fixture
def mock_workflow():
    """Mock DataExtractionWorkflow for testing without Neo4j."""
    mock_instance = Mock(spec=DataExtractionWorkflow)
    mock_instance.extract_data.return_value = {
        "status": "completed",
        "report_data": {"test_data": "sample_data"}
    }
    return mock_instance
```

**After:**
```python
@pytest.fixture
def test_workflow(test_neo4j_driver, clean_test_session):
    """Real DataExtractionWorkflow with test database."""
    config = {
        "neo4j_uri": os.getenv("NEO4J_TEST_URI"),
        "neo4j_user": os.getenv("NEO4J_TEST_USER"),
        "neo4j_password": os.getenv("NEO4J_TEST_PASSWORD"),
        "llm_model": "gpt-3.5-turbo"
    }
    
    workflow = DataExtractionWorkflow(config)
    yield workflow
    workflow.close()
```

### Phase 3: Test Data Management (Week 5)

#### Step 3.1: Test Data Fixtures
```python
# File: tests/fixtures/test_data_fixtures.py

@pytest.fixture
def sample_ehs_data():
    """Create sample EHS data in test database."""
    return {
        "facilities": [
            {"name": "North Plant", "location": "Seattle", "facility_id": "FAC-001"},
            {"name": "South Plant", "location": "Portland", "facility_id": "FAC-002"}
        ],
        "equipment": [
            {"name": "Boiler A", "type": "heating", "facility": "North Plant"},
            {"name": "Chiller B", "type": "cooling", "facility": "South Plant"}
        ],
        "utility_bills": [
            {"amount": 1500.0, "utility_type": "electricity", "facility": "North Plant", "date": "2023-01-01"},
            {"amount": 2200.0, "utility_type": "gas", "facility": "South Plant", "date": "2023-01-01"}
        ]
    }

@pytest.fixture
def populated_test_database(clean_test_session, sample_ehs_data):
    """Database populated with test data."""
    session = clean_test_session
    
    # Create facilities
    for facility in sample_ehs_data["facilities"]:
        session.run("""
        CREATE (f:Facility {name: $name, location: $location, facility_id: $facility_id})
        """, **facility)
    
    # Create equipment and relationships
    for equipment in sample_ehs_data["equipment"]:
        session.run("""
        MATCH (f:Facility {name: $facility})
        CREATE (e:Equipment {name: $name, type: $type})
        CREATE (f)-[:CONTAINS]->(e)
        """, **equipment)
    
    # Create utility bills and relationships
    for bill in sample_ehs_data["utility_bills"]:
        session.run("""
        MATCH (f:Facility {name: $facility})
        CREATE (u:UtilityBill {amount: $amount, utility_type: $utility_type, date: date($date)})
        CREATE (u)-[:RECORDED_AT]->(f)
        """, **bill)
    
    return session
```

### Phase 4: Test Validation and Assertion Updates (Week 6)

#### Step 4.1: Update Test Assertions

**Before (Mock-based assertions):**
```python
def test_electrical_consumption_success(self, test_client, mock_workflow):
    # Mock returns predetermined data
    assert data["status"] == "success"
    assert "data" in data
    mock_workflow.extract_data.assert_called_once()
```

**After (Real data assertions):**
```python
def test_electrical_consumption_success(self, test_client, test_workflow, populated_test_database):
    # Real data from test database
    response = test_client.post("/api/v1/extract/electrical-consumption", 
                              json={"output_format": "json"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Validate actual data structure and content
    assert len(data["data"]["report_data"]) > 0
    assert data["metadata"]["total_records"] == 2  # Based on test data
    
    # Validate actual Cypher queries were executed
    assert "MATCH" in data["queries"][0]["query"]
    assert "UtilityBill" in data["queries"][0]["query"]
```

#### Step 4.2: Performance and Integration Validation

```python
@pytest.mark.integration
def test_end_to_end_query_execution(populated_test_database, test_workflow):
    """Test complete query execution pipeline."""
    query = "Show electricity consumption for all facilities"
    
    # Measure actual execution time
    start_time = time.time()
    result = await test_workflow.process_query(query)
    execution_time = time.time() - start_time
    
    # Validate performance
    assert execution_time < 30.0  # Reasonable timeout
    
    # Validate results structure
    assert result["success"] is True
    assert len(result["data"]) >= 1  # At least one facility
    
    # Validate data accuracy
    for record in result["data"]:
        assert "facility_name" in record
        assert "consumption" in record
        assert isinstance(record["consumption"], (int, float))
```

## Implementation Plan

### Week 1: Infrastructure Setup
- [ ] Set up test Neo4j database instance
- [ ] Configure test environment variables
- [ ] Create database utility functions
- [ ] Set up API key management for tests

### Week 2: Core Component Replacement
- [ ] Replace Neo4j connection mocks
- [ ] Update database interaction tests
- [ ] Verify connection pooling and cleanup

### Week 3: LLM Integration Replacement
- [ ] Replace OpenAI LLM mocks
- [ ] Implement rate limiting for API calls
- [ ] Add fallback for API failures

### Week 4: Workflow Integration
- [ ] Replace workflow component mocks
- [ ] Update end-to-end test scenarios
- [ ] Verify data flow integrity

### Week 5: Test Data Management
- [ ] Implement test data fixtures
- [ ] Create database seeding utilities
- [ ] Add data cleanup mechanisms

### Week 6: Validation and Optimization
- [ ] Update all test assertions
- [ ] Add performance benchmarks
- [ ] Optimize test execution time

## Quality Assurance Measures

### Test Isolation
```python
@pytest.fixture(autouse=True)
def isolate_test_database():
    """Ensure each test has clean database state."""
    # Cleanup before test
    yield
    # Cleanup after test
```

### API Cost Control
```python
@pytest.fixture
def api_rate_limiter():
    """Limit API calls during testing."""
    import time
    last_call = getattr(api_rate_limiter, '_last_call', 0)
    current_time = time.time()
    
    if current_time - last_call < 1.0:  # 1 second minimum between calls
        time.sleep(1.0 - (current_time - last_call))
    
    api_rate_limiter._last_call = time.time()
```

### Error Handling Validation
```python
@pytest.mark.integration
def test_database_connection_failure_handling():
    """Test graceful handling of database failures."""
    # Temporarily disconnect database
    # Verify error handling and fallback behavior
    # Restore connection
```

## Rollback Plan

### Emergency Rollback Procedure
1. **Immediate Rollback**: Revert to previous commit with mocks
2. **Partial Rollback**: Keep successful replacements, revert problematic ones
3. **Staged Rollback**: Disable specific test categories while fixing issues

### Rollback Commands
```bash
# Revert to last known good state
git revert <commit-hash>

# Disable specific test categories
pytest -m "not integration" --ignore=tests/real_database/

# Run only mock-based tests
pytest tests/unit/ -k "mock"
```

## Success Criteria

### Completion Metrics
- [ ] 100% of database connection mocks replaced
- [ ] 100% of LLM integration mocks replaced  
- [ ] 100% of workflow component mocks replaced
- [ ] All tests pass with real implementations
- [ ] Test execution time < 2x original duration
- [ ] No test flakiness introduced
- [ ] API usage within budget limits

### Validation Checklist
- [ ] All test scenarios cover real code paths
- [ ] Error conditions properly tested with real failures
- [ ] Performance benchmarks established
- [ ] Documentation updated with new test patterns
- [ ] CI/CD pipeline updated for new requirements

## Post-Implementation Maintenance

### Monitoring
- Track API usage costs
- Monitor test execution times
- Watch for flaky tests
- Validate test database performance

### Documentation Updates
- Update README with new test setup requirements
- Document real API key management
- Create troubleshooting guide for test failures
- Update development workflow documentation

## Risk Mitigation

### Technical Risks
- **API Rate Limits**: Implement request throttling and caching
- **Database Performance**: Use connection pooling and query optimization
- **Test Flakiness**: Add retries and better error handling
- **Cost Overruns**: Set up API usage monitoring and alerts

### Process Risks
- **Timeline Delays**: Implement phased rollout with rollback options
- **Knowledge Transfer**: Document all changes and provide team training
- **Integration Issues**: Maintain backward compatibility during transition

This strategy ensures a systematic, safe, and thorough transition from mocked to real implementations while maintaining test reliability and comprehensive coverage.