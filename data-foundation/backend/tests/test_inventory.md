# Test Inventory - EHS AI Demo Data Foundation Backend

This document maintains a complete inventory of all tests related to this project.

## Executive Dashboard API v2 Tests

### Comprehensive Integration Tests
- **Location**: `/tests/api/test_executive_dashboard_v2.py`
- **Type**: Comprehensive API Integration Test Suite
- **Coverage**: Complete Executive Dashboard API v2 functionality
- **Test Classes**:
  1. `TestExecutiveDashboardAPI` - Core API functionality tests
  2. `TestExecutiveDashboardIntegration` - Real service integration tests
  3. `TestPerformance` - Performance and load testing
- **Test Cases**:
  1. **API Endpoint Testing**:
     - Executive dashboard endpoint (`/api/v2/executive-dashboard`)
     - Dashboard summary endpoint (`/api/v2/dashboard-summary`)
     - Real-time metrics endpoint (`/api/v2/real-time-metrics`)
     - KPIs endpoint (`/api/v2/kpis`)
     - Locations endpoint (`/api/v2/locations`)
     - Health check endpoint (`/api/v2/health`)
     - Cache management endpoint (`/api/v2/cache/clear`)
     - Static dashboard files endpoint (`/api/v2/static-dashboard/{filename}`)
     - Legacy compatibility endpoint (`/api/v2/dashboard`)
  2. **Parameter Validation Testing**:
     - DashboardRequest model validation
     - Location filter parsing
     - Date range parsing and validation
     - Aggregation period validation
     - Response format validation
     - Cache timeout validation
  3. **Dynamic Data Generation Testing**:
     - Service initialization and dependency injection
     - Mock service integration
     - Real data retrieval (when available)
     - Response format filtering
  4. **Static Fallback Testing**:
     - Static dashboard file serving
     - Fallback when service unavailable
     - Minimal fallback response generation
  5. **Error Handling Testing**:
     - Service initialization failures
     - Invalid parameter handling
     - Database connection errors
     - Graceful degradation scenarios
  6. **Cache Functionality Testing**:
     - Cache hit/miss performance
     - Cache clear operations
     - Performance improvements validation
  7. **Performance Benchmarking**:
     - Individual endpoint performance
     - Concurrent request handling
     - Load testing (50 requests)
     - Response time thresholds (5s max)
  8. **Backward Compatibility Testing**:
     - Legacy API parameter mapping
     - v1 to v2 API compatibility
     - Response format consistency

### Test Features:
- **Mock Service Integration**: Comprehensive mocking of ExecutiveDashboardService
- **Fixtures and Utilities**: Reusable test fixtures for consistency
- **Metrics Tracking**: Built-in performance metrics collection
- **Static File Testing**: Temporary file creation for fallback testing
- **Concurrent Testing**: Multi-threading request validation
- **Edge Case Testing**: Boundary conditions and special scenarios
- **Response Validation**: Strict response format and content validation
- **Error Scenario Simulation**: Service failures and recovery testing

### Performance Thresholds:
- **Individual Endpoint Response Time**: 5 seconds maximum
- **Cache Performance Improvement**: 50% faster on cache hits
- **Load Test Average**: 1 second per request under load
- **Concurrent Requests**: 5 simultaneous requests supported

### Dependencies:
- FastAPI TestClient
- pytest framework
- httpx for HTTP testing
- unittest.mock for service mocking
- threading for concurrency tests
- Executive Dashboard Service components
- Environment variables from .env file

### Test Environment Requirements:
- Python 3.8+ with pytest
- FastAPI and dependencies installed
- Mock-based testing (no external dependencies required for most tests)
- Optional: Real Neo4j and OpenAI API for integration tests
- Environment variables:
  - NEO4J_URI (for real service tests)
  - NEO4J_USERNAME
  - NEO4J_PASSWORD  
  - OPENAI_API_KEY (for real service tests)

## Risk Assessment Workflow Tests

### Manual Risk Assessment Storage Test
- **Location**: `/tmp/test_risk_storage_manual.py` (created 2025-01-29)
- **Type**: Manual Integration Test for Risk Assessment Storage
- **Coverage**: Complete risk assessment data storage workflow
- **Test Cases**:
  1. **Neo4j Connection Setup**: Establishes connection to Neo4j database
  2. **Test Document Creation**: Creates test document for relationship validation
  3. **Risk Assessment Storage**: Tests complete storage of risk assessment results
  4. **Storage Verification**: Validates all stored data and relationships
  5. **Graph Structure Validation**: Verifies complete graph structure integrity

### Test Results (Latest Run: 2025-01-29 13:39:21):
- **Status**: ✅ SUCCESS - All tests passed
- **Duration**: ~0.5 seconds
- **Nodes Created**: 5 (1 RiskAssessment, 2 RiskFactor, 2 RiskRecommendation)
- **Relationships Created**: 6 total relationships
- **Test Coverage**:
  - ✅ RiskAssessment node creation and storage
  - ✅ RiskFactor nodes with IDENTIFIES relationships
  - ✅ RiskRecommendation nodes with RECOMMENDS relationships
  - ✅ Document-RiskAssessment HAS_RISK_ASSESSMENT relationship
  - ✅ Facility-RiskAssessment HAS_RISK_ASSESSMENT relationship
  - ✅ Complete graph structure verification

### Test Features:
- **Direct Cypher Query Testing**: Uses the exact same Cypher queries as the workflow
- **Realistic Test Data**: Creates representative risk assessment data
- **Comprehensive Verification**: Validates all nodes, relationships, and data integrity
- **Auto-cleanup**: Removes test data after completion
- **Detailed Logging**: Comprehensive logging to `/tmp/test_risk_storage_manual.log`
- **Environment Integration**: Loads configuration from project .env file

### Storage Schema Validated:
```
RiskAssessment {
  id, document_id, assessment_date, risk_level, risk_score,
  assessment_status, methodology, processing_time, created_by, updated_at
}

RiskFactor {
  id, name, category, description, severity, probability, confidence,
  created_at, updated_at
}

RiskRecommendation {
  id, title, description, priority, estimated_impact,
  implementation_timeline, created_at, updated_at
}

Relationships:
- (RiskAssessment)-[:IDENTIFIES]->(RiskFactor)
- (RiskAssessment)-[:RECOMMENDS]->(RiskRecommendation)
- (Document)-[:HAS_RISK_ASSESSMENT]->(RiskAssessment)
- (Facility)-[:HAS_RISK_ASSESSMENT]->(RiskAssessment)
```

### Running the Test:
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
source venv/bin/activate
python3 /tmp/test_risk_storage_manual.py
```

### Test Environment Requirements:
- Neo4j database running on localhost:7687
- Environment variables from .env file:
  - NEO4J_URI
  - NEO4J_USERNAME
  - NEO4J_PASSWORD
- Python 3.8+ with neo4j driver installed
- Test facility (DEMO_FACILITY_001) should exist in Neo4j for optimal results

### Integration Tests (Existing - Currently Has Import Issues)
- **Location**: `/tests/agents/risk_assessment/test_workflow_integration.py`
- **Type**: End-to-End Integration Test
- **Coverage**: Complete risk assessment workflow
- **Status**: ⚠️ Import dependency issues prevent execution
- **Test Cases**:
  1. Complete risk assessment workflow (test_complete_workflow)
  2. Error handling scenarios (test_error_handling)
  3. Different document types (test_document_types)
  4. Configuration management (test_configuration)
  5. Performance benchmarks (test_performance_benchmarks)
- **Dependencies**: Neo4j, OpenAI API, LangSmith (optional)
- **Test Data**: Creates test facility and associated EHS data
- **Performance Thresholds**:
  - Total assessment time: 120 seconds
  - Data collection time: 30 seconds
  - Risk analysis time: 45 seconds
  - Recommendation generation time: 30 seconds

### Test Features (Original Integration Tests):
- **Neo4j Data Validation**: Tests data collection from Neo4j graph database
- **LangSmith Trace Capture**: Validates trace metadata and session tracking
- **Risk Assessment Results**: Validates risk levels, scores, and recommendations
- **Error Handling**: Tests graceful failure scenarios and retry logic
- **Configuration Testing**: Validates different agent configurations
- **Performance Benchmarking**: Measures and validates execution times
- **Document Type Support**: Tests different assessment scopes and types

## Existing Tests (Legacy)

### API Tests
- **Location**: `/tests/test_ehs_extraction_api.py`
- **Type**: API Integration Test
- **Coverage**: EHS extraction API endpoints

### Database Tests
- **Location**: `/tests/test_database.py`
- **Type**: Database Integration Test
- **Coverage**: Database connectivity and operations

### Transcript Tests
- **Location**: `/tests/test_transcript_integration.py`
- **Type**: Integration Test
- **Coverage**: Document upload and transcript generation workflow

- **Location**: `/tests/test_transcript_simple.py`
- **Type**: Simple Integration Test
- **Coverage**: Basic transcript functionality

- **Location**: `/tests/test_transcript_diagnostic.py`
- **Type**: Diagnostic Test
- **Coverage**: Transcript troubleshooting and debugging

### Neo4j Enhancement Tests
- **Location**: `/tests/neo4j_enhancements/test_enhanced_schema_integration.py`
- **Type**: Schema Integration Test
- **Coverage**: Enhanced Neo4j schema functionality

### Workflow Tests
- **Location**: `/tests/test_workflow_prorating_integration.py`
- **Type**: Workflow Integration Test
- **Coverage**: Prorating workflow functionality

### Allocation Tests
- **Location**: `/tests/test_monthly_boundary_allocation.py`
- **Type**: Business Logic Test
- **Coverage**: Monthly boundary allocation logic

### Rejection Tracking Tests
- **Location**: `/tests/test_rejection_tracking_real.py`
- **Type**: Real-world Integration Test
- **Coverage**: Document rejection tracking functionality

### Document Recognition Tests
- **Location**: `/test/test_document_recognition_service.py`
- **Type**: Service Test
- **Coverage**: Document recognition and classification

## Test Execution Instructions

### Running Executive Dashboard API v2 Tests

#### Using pytest (recommended):
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
python3 -m pytest tests/api/test_executive_dashboard_v2.py -v
```

#### Running specific test classes:
```bash
# Core API functionality tests
python3 -m pytest tests/api/test_executive_dashboard_v2.py::TestExecutiveDashboardAPI -v

# Performance tests
python3 -m pytest tests/api/test_executive_dashboard_v2.py::TestPerformance -v

# Integration tests (requires real service)
python3 -m pytest tests/api/test_executive_dashboard_v2.py::TestExecutiveDashboardIntegration -v
```

#### Running specific test methods:
```bash
python3 -m pytest tests/api/test_executive_dashboard_v2.py::TestExecutiveDashboardAPI::test_executive_dashboard_endpoint_success -v
python3 -m pytest tests/api/test_executive_dashboard_v2.py::TestExecutiveDashboardAPI::test_performance_benchmarks -v
```

#### Running as standalone script:
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
python3 tests/api/test_executive_dashboard_v2.py
```

#### With virtual environment:
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
source venv/bin/activate
python3 tests/api/test_executive_dashboard_v2.py
```

#### Performance and load testing:
```bash
# Run performance tests specifically
python3 -m pytest tests/api/test_executive_dashboard_v2.py -k "performance" -v

# Run with detailed timing
python3 -m pytest tests/api/test_executive_dashboard_v2.py --durations=10 -v
```

### Running Risk Assessment Storage Tests

#### Manual Risk Assessment Storage Test:
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
source venv/bin/activate
python3 /tmp/test_risk_storage_manual.py
```

#### Risk Assessment Integration Tests (Original - Currently Non-Functional):
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
python3 -m pytest tests/agents/risk_assessment/test_workflow_integration.py -v
```

**Note**: The original integration tests currently have import dependency issues that prevent execution. The manual storage test provides validated coverage of the core risk assessment storage functionality.

### Test Output and Logging

#### Executive Dashboard API v2 Tests:
- Console output: Detailed test progress and results
- Performance metrics: Response times and throughput measurements
- Mock service validation: Service interaction verification
- Temporary files: Cleaned up automatically after tests

#### Risk Assessment Storage Tests:
- Console output: Shows test progress and results with emoji indicators
- Log file: `/tmp/test_risk_storage_manual.log` (detailed logging)
- Test artifacts: Temporary test data cleaned up automatically
- Graph verification: Comprehensive Neo4j data validation

### Prerequisites Checklist

#### For Executive Dashboard API v2 Tests:
- [ ] Python 3.8+ installed
- [ ] pytest and FastAPI dependencies installed
- [ ] Virtual environment activated (recommended)
- [ ] For integration tests: Neo4j and OpenAI API access

#### For Risk Assessment Storage Tests:
- [ ] Neo4j database is running and accessible
- [ ] .env file is properly configured with Neo4j credentials
- [ ] Python dependencies are installed (neo4j driver)
- [ ] Virtual environment is activated (if using)
- [ ] Network connectivity to Neo4j database

## Test Maintenance Notes

### Executive Dashboard API v2 Tests:
- Tests use comprehensive mocking for reliable execution
- Performance thresholds may need adjustment based on system capabilities
- Static fallback files are created temporarily during tests
- Concurrent testing validates multi-threading support
- Mock service responses can be customized for different scenarios

### Risk Assessment Storage Tests:
- Test data is automatically created and cleaned up with `test_data: true` flag
- Uses realistic risk assessment data matching production schema
- All test artifacts are stored in /tmp directory
- Performance measurement included (typically completes in <1 second)
- Schema validation ensures compatibility with workflow storage methods

## Future Test Additions

### Risk Assessment Workflow:
- **Priority**: Fix import issues in existing integration tests
- Unit tests for individual risk assessment workflow methods
- Mock-based testing for offline scenarios
- Load testing with multiple concurrent risk assessments
- Cross-validation with manual risk assessments
- Performance benchmarking under various data loads

### Executive Dashboard API v2:
- Authentication and authorization testing
- WebSocket real-time updates testing
- Database-specific testing for different Neo4j configurations
- API versioning and migration testing
- Integration with external monitoring systems

### General:
- Integration between risk assessment and dashboard display
- End-to-end workflow testing from document upload to risk visualization
- Automated performance regression testing
- Data consistency validation across all components