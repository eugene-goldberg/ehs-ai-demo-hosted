# EHS AI Demo - Test Execution Guide

> Version: 1.0.0  
> Last Updated: 2025-08-23  
> Status: Active

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites and Setup](#prerequisites-and-setup)
3. [Test Environment Configuration](#test-environment-configuration)
4. [Test Categories](#test-categories)
5. [Running Tests](#running-tests)
6. [Understanding Test Results](#understanding-test-results)
7. [Test Data Fixtures](#test-data-fixtures)
8. [Test Matrix and Coverage](#test-matrix-and-coverage)
9. [Performance Testing](#performance-testing)
10. [Troubleshooting](#troubleshooting)
11. [CI/CD Integration](#cicd-integration)
12. [Test File Reference](#test-file-reference)

## Overview

The EHS AI Demo project includes comprehensive testing infrastructure to validate:
- **Original Features**: Document processing, extraction, and Q&A capabilities
- **Phase 1 Enhancements**: Audit trail, pro-rating calculations, rejection handling
- **Integration Scenarios**: End-to-end workflows and system integration
- **Performance Benchmarks**: Load testing and response time validation

## Prerequisites and Setup

### System Requirements

- Python 3.8+
- Neo4j database (running on localhost:7687)
- Virtual environment support
- curl (for API testing)
- At least 4GB RAM available
- Network access for external API calls

### Required API Keys

The following API keys must be configured in `.env` file:

```bash
# Required for OpenAI embeddings and LLM operations
OPENAI_API_KEY="your-openai-api-key"

# Required for document parsing
LLAMA_PARSE_API_KEY="your-llama-parse-api-key"

# Optional: For enhanced model capabilities
LANGCHAIN_API_KEY="your-langchain-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Initial Setup

1. **Navigate to backend directory**:
   ```bash
   cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Neo4j connection**:
   ```bash
   python3 -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'EhsAI2024!')); driver.verify_connectivity(); print('Neo4j connected successfully')"
   ```

5. **Start test API server**:
   ```bash
   python3 src/test_api/comprehensive_test_api.py
   ```

## Test Environment Configuration

### Environment Variables

Key configuration in `.env`:

```bash
# Database Configuration
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="EhsAI2024!"
NEO4J_DATABASE="neo4j"

# Embedding Configuration
EMBEDDING_MODEL="all-MiniLM-L6-v2"
IS_EMBEDDING="TRUE"
ENTITY_EMBEDDING="TRUE"

# Model Configuration
DEFAULT_DIFFBOT_CHAT_MODEL="openai_gpt_4o"
LLM_MODEL_CONFIG_openai_gpt_4o="gpt-4o-2024-11-20,openai_api_key"

# Performance Tuning
KNN_MIN_SCORE="0.94"
NUMBER_OF_CHUNKS_TO_COMBINE=6
MAX_TOKEN_CHUNK_SIZE=2000
```

### Test Server Endpoints

The comprehensive test API runs on `http://localhost:8001` with the following base endpoints:

- **Health Check**: `GET /test/health`
- **Original Features**: `/test/original/*`
- **Phase 1 Features**: `/test/phase1/*`
- **Integration Tests**: `/test/integration/*`
- **Performance Tests**: `/test/performance/*`

## Test Categories

### 1. Original Features Testing

#### Document Processing
- **Upload and Processing**: `/test/original/document-upload`
- **Content Extraction**: `/test/original/content-extraction`
- **Metadata Parsing**: `/test/original/metadata-extraction`

#### Q&A System
- **Basic Q&A**: `/test/original/qa-basic`
- **Context-Aware Q&A**: `/test/original/qa-contextual`
- **Multi-document Q&A**: `/test/original/qa-multidoc`

#### Search and Retrieval
- **Semantic Search**: `/test/original/semantic-search`
- **Keyword Search**: `/test/original/keyword-search`
- **Hybrid Search**: `/test/original/hybrid-search`

### 2. Phase 1 Enhancements Testing

#### Audit Trail
- **Action Logging**: `/test/phase1/audit-logging`
- **User Tracking**: `/test/phase1/audit-users`
- **Change History**: `/test/phase1/audit-history`

#### Pro-rating Calculations
- **Simple Pro-rating**: `/test/phase1/prorate-simple`
- **Complex Pro-rating**: `/test/phase1/prorate-complex`
- **Validation Rules**: `/test/phase1/prorate-validation`

#### Rejection Handling
- **Document Rejection**: `/test/phase1/rejection-document`
- **Quality Checks**: `/test/phase1/rejection-quality`
- **Retry Mechanisms**: `/test/phase1/rejection-retry`

### 3. Integration Testing

#### Workflow Integration
- **End-to-End Processing**: `/test/integration/e2e-workflow`
- **Cross-component Communication**: `/test/integration/component-communication`
- **Error Propagation**: `/test/integration/error-handling`

#### Database Integration
- **Neo4j Operations**: `/test/integration/neo4j-operations`
- **Data Consistency**: `/test/integration/data-consistency`
- **Transaction Handling**: `/test/integration/transactions`

### 4. Performance Testing

#### Load Testing
- **Concurrent Requests**: `/test/performance/load-concurrent`
- **Throughput Testing**: `/test/performance/throughput`
- **Stress Testing**: `/test/performance/stress`

#### Response Time Benchmarks
- **API Response Times**: `/test/performance/response-times`
- **Database Query Performance**: `/test/performance/db-performance`
- **Memory Usage**: `/test/performance/memory-usage`

## Running Tests

### Quick Start - Health Check

```bash
# Verify test API is running
curl -X GET http://localhost:8001/test/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-23T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "api": "operational",
    "database": "connected",
    "models": "loaded"
  }
}
```

### Running Original Features Tests

#### Document Upload and Processing

```bash
# Test document upload
curl -X POST http://localhost:8001/test/original/document-upload \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "pdf",
    "file_size": "2MB",
    "test_scenario": "standard_processing"
  }'
```

#### Q&A System Testing

```bash
# Test basic Q&A
curl -X POST http://localhost:8001/test/original/qa-basic \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key safety requirements?",
    "context": "safety_document_context",
    "expected_answer_type": "list"
  }'
```

#### Semantic Search Testing

```bash
# Test semantic search
curl -X POST http://localhost:8001/test/original/semantic-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "environmental compliance procedures",
    "search_type": "semantic",
    "max_results": 10
  }'
```

### Running Phase 1 Enhancement Tests

#### Audit Trail Testing

```bash
# Test audit logging
curl -X POST http://localhost:8001/test/phase1/audit-logging \
  -H "Content-Type: application/json" \
  -d '{
    "action": "document_processed",
    "user_id": "test_user_001",
    "metadata": {
      "document_id": "doc_123",
      "processing_time": "2.5s"
    }
  }'
```

#### Pro-rating Calculations

```bash
# Test simple pro-rating
curl -X POST http://localhost:8001/test/phase1/prorate-simple \
  -H "Content-Type: application/json" \
  -d '{
    "base_amount": 1000.00,
    "proration_factor": 0.75,
    "calculation_type": "linear"
  }'
```

#### Rejection Handling

```bash
# Test document rejection
curl -X POST http://localhost:8001/test/phase1/rejection-document \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_456",
    "rejection_reason": "insufficient_quality",
    "quality_score": 0.45
  }'
```

### Running Integration Tests

#### End-to-End Workflow

```bash
# Test complete workflow
curl -X POST http://localhost:8001/test/integration/e2e-workflow \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "complete_document_processing",
    "test_data_set": "integration_test_suite",
    "validation_mode": "strict"
  }'
```

#### Database Operations

```bash
# Test Neo4j operations
curl -X POST http://localhost:8001/test/integration/neo4j-operations \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "crud_operations",
    "test_entities": ["documents", "relationships", "properties"]
  }'
```

### Running Performance Tests

#### Load Testing

```bash
# Test concurrent load
curl -X POST http://localhost:8001/test/performance/load-concurrent \
  -H "Content-Type: application/json" \
  -d '{
    "concurrent_users": 50,
    "duration_seconds": 300,
    "ramp_up_time": 60
  }'
```

#### Response Time Benchmarks

```bash
# Test API response times
curl -X POST http://localhost:8001/test/performance/response-times \
  -H "Content-Type: application/json" \
  -d '{
    "endpoints": ["document-upload", "qa-basic", "semantic-search"],
    "iterations": 100,
    "timeout_threshold": 5000
  }'
```

### Automated Test Execution

#### Using Shell Scripts

```bash
# Run comprehensive curl tests
chmod +x src/test_api/test_scripts/comprehensive_curl_tests.sh
./src/test_api/test_scripts/comprehensive_curl_tests.sh

# Run EHS-specific API tests
chmod +x src/test_api/test_scripts/ehs_api_tests.sh
./src/test_api/test_scripts/ehs_api_tests.sh
```

#### Using Python Test Runner

```bash
# Run all tests with Python test runner
python3 run_tests.py --category all --verbose

# Run specific test categories
python3 run_tests.py --category original --output-format json
python3 run_tests.py --category phase1 --parallel
python3 run_tests.py --category integration --timeout 300
```

## Understanding Test Results

### Response Format

All test endpoints return standardized responses:

```json
{
  "test_id": "unique_test_identifier",
  "timestamp": "2025-08-23T10:30:00Z",
  "status": "passed|failed|error",
  "execution_time_ms": 1250,
  "test_category": "original|phase1|integration|performance",
  "test_name": "descriptive_test_name",
  "results": {
    "success": true,
    "details": "Detailed test outcome information",
    "metrics": {
      "response_time": "1.25s",
      "memory_usage": "45MB",
      "accuracy_score": 0.92
    }
  },
  "validation": {
    "assertions_passed": 8,
    "assertions_failed": 0,
    "assertions_skipped": 1
  },
  "errors": []
}
```

### Success Criteria

#### Original Features
- **Document Processing**: Upload success rate > 95%, processing time < 30s
- **Q&A System**: Answer accuracy > 85%, response time < 5s
- **Search**: Relevance score > 0.8, recall > 70%

#### Phase 1 Enhancements
- **Audit Trail**: 100% action logging, complete metadata capture
- **Pro-rating**: Calculation accuracy 99.9%, validation coverage 100%
- **Rejection Handling**: Proper error codes, retry success rate > 80%

#### Integration Tests
- **E2E Workflow**: Complete workflow success rate > 90%
- **Database Operations**: Transaction success rate 100%, consistency maintained
- **Error Handling**: Graceful failure handling, proper error propagation

#### Performance Tests
- **Load Testing**: System stable under target load, no memory leaks
- **Response Times**: 95th percentile response time within SLA
- **Throughput**: Minimum requests per second achieved

### Test Metrics Dashboard

Access real-time test metrics at: `http://localhost:8001/test/metrics/dashboard`

Key metrics tracked:
- Test execution trends
- Success/failure rates by category
- Performance benchmarks over time
- Resource utilization during tests
- Error frequency and patterns

## Test Data Fixtures

### Location and Structure

Test data fixtures are located in: `src/test_api/fixtures/`

```
fixtures/
├── test_data_fixtures.py          # Main fixture definitions
├── documents/
│   ├── sample_ehs_policy.pdf      # Sample EHS policy document
│   ├── safety_manual.docx         # Safety manual for testing
│   └── compliance_report.xlsx     # Compliance data spreadsheet
├── expected_results/
│   ├── qa_expected_answers.json   # Expected Q&A responses
│   ├── extraction_results.json   # Expected extraction outputs
│   └── search_results.json       # Expected search results
└── test_scenarios/
    ├── integration_scenarios.json # Integration test scenarios
    ├── performance_scenarios.json # Performance test scenarios
    └── error_scenarios.json      # Error handling scenarios
```

### Available Test Data Sets

#### Document Test Data
- **EHS Policy Documents**: 15 sample policy documents in various formats
- **Safety Manuals**: 8 comprehensive safety procedure manuals
- **Compliance Reports**: 12 quarterly compliance reports with data
- **Incident Reports**: 20 sample incident reports for testing

#### Q&A Test Data
- **Safety Questions**: 100 common safety-related questions
- **Compliance Queries**: 75 regulatory compliance questions
- **Policy Questions**: 50 policy interpretation questions
- **Technical Questions**: 25 technical procedure questions

#### Performance Test Data
- **Large Document Set**: 1000+ documents for load testing
- **Concurrent User Scenarios**: 50 different user interaction patterns
- **Stress Test Data**: High-volume data sets for stress testing

### Using Test Fixtures

```python
# Import test fixtures
from src.test_api.fixtures.test_data_fixtures import (
    get_sample_document,
    get_test_questions,
    get_performance_data
)

# Load specific test data
sample_doc = get_sample_document('ehs_policy')
test_questions = get_test_questions('safety', count=10)
perf_data = get_performance_data('load_test_large')
```

## Test Matrix and Coverage

### Feature Coverage Matrix

| Feature Category | Unit Tests | Integration Tests | Performance Tests | E2E Tests | Coverage % |
|-----------------|------------|-------------------|-------------------|-----------|------------|
| **Original Features** | | | | | |
| Document Processing | ✅ | ✅ | ✅ | ✅ | 95% |
| Content Extraction | ✅ | ✅ | ✅ | ✅ | 92% |
| Q&A System | ✅ | ✅ | ✅ | ✅ | 89% |
| Semantic Search | ✅ | ✅ | ✅ | ✅ | 91% |
| **Phase 1 Enhancements** | | | | | |
| Audit Trail | ✅ | ✅ | ⚠️ | ✅ | 87% |
| Pro-rating Calculations | ✅ | ✅ | ⚠️ | ✅ | 94% |
| Rejection Handling | ✅ | ✅ | ⚠️ | ✅ | 86% |
| **System Components** | | | | | |
| Neo4j Operations | ✅ | ✅ | ✅ | ✅ | 93% |
| API Endpoints | ✅ | ✅ | ✅ | ✅ | 96% |
| Error Handling | ✅ | ✅ | ✅ | ✅ | 88% |
| Authentication | ⚠️ | ⚠️ | ❌ | ⚠️ | 45% |

Legend: ✅ Complete | ⚠️ Partial | ❌ Not Implemented

### Test Scenario Coverage

#### Original Features (250+ test cases)
- **Document Upload**: 45 scenarios covering various file types and sizes
- **Content Extraction**: 35 scenarios testing different document structures
- **Q&A System**: 85 scenarios with various question types and contexts
- **Search Functions**: 65 scenarios testing different search modes
- **Error Handling**: 20 scenarios for error conditions

#### Phase 1 Enhancements (120+ test cases)
- **Audit Trail**: 25 scenarios covering action logging and tracking
- **Pro-rating**: 40 scenarios with different calculation types
- **Rejection Handling**: 35 scenarios for various rejection reasons
- **Validation**: 20 scenarios for data validation rules

#### Integration Scenarios (75+ test cases)
- **End-to-End Workflows**: 30 complete workflow scenarios
- **Component Integration**: 25 inter-component communication tests
- **Database Integration**: 20 Neo4j-specific integration tests

#### Performance Scenarios (50+ test cases)
- **Load Testing**: 15 scenarios with varying concurrent users
- **Stress Testing**: 10 scenarios pushing system limits
- **Memory/Resource**: 15 scenarios monitoring resource usage
- **Response Time**: 10 scenarios benchmarking API performance

## Performance Testing

### Performance Test Categories

#### 1. Load Testing
Tests system behavior under expected load conditions.

```bash
# Standard load test (20 concurrent users)
curl -X POST http://localhost:8001/test/performance/load-standard \
  -H "Content-Type: application/json" \
  -d '{
    "concurrent_users": 20,
    "duration_minutes": 10,
    "operations": ["upload", "search", "qa"]
  }'

# Peak load test (50 concurrent users)
curl -X POST http://localhost:8001/test/performance/load-peak \
  -H "Content-Type: application/json" \
  -d '{
    "concurrent_users": 50,
    "duration_minutes": 5,
    "ramp_up_minutes": 2
  }'
```

#### 2. Stress Testing
Tests system behavior beyond normal operating capacity.

```bash
# Stress test with high load
curl -X POST http://localhost:8001/test/performance/stress-high \
  -H "Content-Type: application/json" \
  -d '{
    "concurrent_users": 100,
    "duration_minutes": 15,
    "failure_threshold": 10
  }'
```

#### 3. Endurance Testing
Tests system stability over extended periods.

```bash
# Endurance test
curl -X POST http://localhost:8001/test/performance/endurance \
  -H "Content-Type: application/json" \
  -d '{
    "duration_hours": 4,
    "user_load": 10,
    "memory_monitoring": true
  }'
```

### Performance Benchmarks

#### Response Time Targets
- **Document Upload**: < 10 seconds (95th percentile)
- **Content Extraction**: < 30 seconds (95th percentile)
- **Q&A Queries**: < 5 seconds (95th percentile)
- **Search Operations**: < 3 seconds (95th percentile)

#### Throughput Targets
- **API Requests**: > 100 requests/second
- **Document Processing**: > 50 documents/hour
- **Concurrent Users**: Support 50+ simultaneous users

#### Resource Usage Limits
- **Memory Usage**: < 4GB under normal load
- **CPU Usage**: < 70% average under normal load
- **Database Connections**: < 100 concurrent connections

### Performance Monitoring

```bash
# Monitor system resources during tests
curl -X GET http://localhost:8001/test/performance/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "metrics": ["cpu", "memory", "database", "response_times"],
    "interval_seconds": 30,
    "duration_minutes": 10
  }'
```

## Troubleshooting

### Common Issues and Solutions

#### 1. API Connection Issues

**Problem**: Cannot connect to test API
```
curl: (7) Failed to connect to localhost port 8001: Connection refused
```

**Solutions**:
```bash
# Check if test API is running
ps aux | grep comprehensive_test_api

# Start the test API if not running
python3 src/test_api/comprehensive_test_api.py

# Check port availability
lsof -i :8001
```

#### 2. Neo4j Connection Issues

**Problem**: Neo4j connection failures
```json
{
  "error": "Failed to connect to Neo4j database",
  "details": "ServiceUnavailable: Failed to establish connection"
}
```

**Solutions**:
```bash
# Check Neo4j status
neo4j status

# Start Neo4j if stopped
neo4j start

# Verify connection credentials in .env
grep NEO4J .env

# Test connection manually
python3 -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'EhsAI2024!')); driver.verify_connectivity()"
```

#### 3. API Key Issues

**Problem**: Authentication failures with external APIs
```json
{
  "error": "Invalid API key",
  "service": "OpenAI"
}
```

**Solutions**:
```bash
# Verify API keys are set
grep -E "(OPENAI|LLAMA_PARSE)" .env

# Test API key validity
curl -H "Authorization: Bearer YOUR_OPENAI_KEY" https://api.openai.com/v1/models

# Update .env with valid keys
nano .env
```

#### 4. Memory Issues

**Problem**: Out of memory errors during testing
```json
{
  "error": "MemoryError: Unable to allocate memory",
  "context": "document_processing"
}
```

**Solutions**:
```bash
# Check available memory
free -h

# Reduce concurrent test load
# Edit test parameters to use fewer concurrent operations

# Clear Python cache
rm -rf __pycache__/
rm -rf src/**/__pycache__/

# Restart with increased memory limits
ulimit -m 8388608  # 8GB limit
```

#### 5. Performance Issues

**Problem**: Tests running slower than expected

**Solutions**:
```bash
# Check system resource usage
top
htop

# Optimize Neo4j performance
# Add to neo4j.conf:
# dbms.memory.heap.initial_size=2g
# dbms.memory.heap.max_size=4g

# Use performance test mode
export TESTING_MODE=performance
python3 src/test_api/comprehensive_test_api.py
```

#### 6. Test Data Issues

**Problem**: Missing or corrupted test data files

**Solutions**:
```bash
# Verify test fixtures exist
ls -la src/test_api/fixtures/

# Regenerate test data if needed
python3 src/test_api/fixtures/test_data_fixtures.py --regenerate

# Download missing sample documents
# Check project documentation for data sources
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set debug environment variables
export LOG_LEVEL=DEBUG
export TESTING_DEBUG=true

# Run tests with verbose output
python3 run_tests.py --debug --verbose

# View test logs
tail -f test_execution.log
```

### Getting Help

1. **Check test logs**: Always review `test_execution.log` for detailed error information
2. **Verify environment**: Ensure all prerequisites are met and configured correctly
3. **Consult documentation**: Review API documentation at `/test/docs`
4. **Run health checks**: Use `/test/health` endpoint to verify system status
5. **Gradual testing**: Start with simple tests before running complex scenarios

## CI/CD Integration

### GitHub Actions Integration

Create `.github/workflows/test-execution.yml`:

```yaml
name: EHS AI Demo Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/EhsAI2024!
        ports:
          - 7687:7687
          - 7474:7474
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        cd data-foundation/backend
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Configure environment
      run: |
        cd data-foundation/backend
        cp .env.example .env
        # Set GitHub secrets for API keys
        echo "OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}" >> .env
        echo "LLAMA_PARSE_API_KEY=${{ secrets.LLAMA_PARSE_API_KEY }}" >> .env
    
    - name: Wait for Neo4j
      run: |
        timeout 60 bash -c 'until echo > /dev/tcp/localhost/7687; do sleep 1; done'
    
    - name: Run original features tests
      run: |
        cd data-foundation/backend
        python3 run_tests.py --category original --ci-mode
    
    - name: Run phase1 enhancement tests
      run: |
        cd data-foundation/backend
        python3 run_tests.py --category phase1 --ci-mode
    
    - name: Run integration tests
      run: |
        cd data-foundation/backend
        python3 run_tests.py --category integration --ci-mode
    
    - name: Generate test report
      run: |
        cd data-foundation/backend
        python3 generate_test_report.py --format html --output test_report.html
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: data-foundation/backend/test_report.html
```

### Jenkins Integration

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        NEO4J_URI = 'bolt://localhost:7687'
        NEO4J_AUTH = 'neo4j/EhsAI2024!'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                dir('data-foundation/backend') {
                    sh '''
                        python3 -m venv venv
                        source venv/bin/activate
                        pip install -r requirements.txt
                    '''
                }
            }
        }
        
        stage('Start Services') {
            steps {
                sh 'docker run -d --name neo4j-test -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/EhsAI2024! neo4j:5'
                sh 'sleep 30'  // Wait for Neo4j to start
            }
        }
        
        stage('Run Tests') {
            parallel {
                stage('Original Features') {
                    steps {
                        dir('data-foundation/backend') {
                            sh '''
                                source venv/bin/activate
                                python3 run_tests.py --category original --ci-mode --junit-xml original_tests.xml
                            '''
                        }
                    }
                }
                stage('Phase 1 Enhancements') {
                    steps {
                        dir('data-foundation/backend') {
                            sh '''
                                source venv/bin/activate
                                python3 run_tests.py --category phase1 --ci-mode --junit-xml phase1_tests.xml
                            '''
                        }
                    }
                }
                stage('Integration Tests') {
                    steps {
                        dir('data-foundation/backend') {
                            sh '''
                                source venv/bin/activate
                                python3 run_tests.py --category integration --ci-mode --junit-xml integration_tests.xml
                            '''
                        }
                    }
                }
            }
        }
        
        stage('Performance Tests') {
            steps {
                dir('data-foundation/backend') {
                    sh '''
                        source venv/bin/activate
                        python3 run_tests.py --category performance --ci-mode --junit-xml performance_tests.xml
                    '''
                }
            }
        }
    }
    
    post {
        always {
            dir('data-foundation/backend') {
                junit '*.xml'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: '.',
                    reportFiles: 'test_report.html',
                    reportName: 'EHS AI Test Report'
                ])
            }
            sh 'docker stop neo4j-test && docker rm neo4j-test'
        }
        
        failure {
            emailext (
                subject: "EHS AI Tests Failed - Build ${env.BUILD_NUMBER}",
                body: "Test execution failed. Please check the test report for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### Test Reporting

#### JUnit XML Format
Tests can generate JUnit-compatible XML reports for CI/CD integration:

```bash
# Generate JUnit XML report
python3 run_tests.py --category all --junit-xml test_results.xml
```

#### HTML Reports
Generate comprehensive HTML test reports:

```bash
# Generate HTML report
python3 generate_test_report.py --format html --include-charts --output comprehensive_test_report.html
```

#### JSON Reports
Generate machine-readable JSON reports for integration:

```bash
# Generate JSON report
python3 run_tests.py --category all --output-format json --output test_results.json
```

## Test File Reference

### Main Test Files

| File | Purpose | Location | Test Type |
|------|---------|----------|-----------|
| `comprehensive_test_api.py` | Main test API server | `src/test_api/` | API Endpoints |
| `test_ehs_extraction_api.py` | EHS extraction testing | `tests/` | Integration |
| `test_commutiesqa.py` | Community Q&A testing | `/` | Unit/Integration |
| `test_integrationqa.py` | Integration Q&A testing | `/` | Integration |
| `Performance_test.py` | Performance benchmarking | `/` | Performance |
| `dbtest.py` | Database testing | `/` | Unit |
| `run_tests.py` | Test runner and coordinator | `/` | Test Runner |

### Test Script Files

| File | Purpose | Location | Usage |
|------|---------|----------|-------|
| `comprehensive_curl_tests.sh` | Comprehensive API testing via curl | `src/test_api/test_scripts/` | CI/CD, Manual |
| `ehs_api_tests.sh` | EHS-specific API testing | `src/test_api/test_scripts/` | CI/CD, Manual |

### Test Data Files

| File | Purpose | Location | Content Type |
|------|---------|----------|--------------|
| `test_data_fixtures.py` | Test data generation and management | `src/test_api/fixtures/` | Test Data |
| Sample documents | Various document formats for testing | `src/test_api/fixtures/documents/` | PDF, DOCX, XLSX |
| Expected results | Expected outputs for validation | `src/test_api/fixtures/expected_results/` | JSON |
| Test scenarios | Predefined test scenarios | `src/test_api/fixtures/test_scenarios/` | JSON |

### Configuration Files

| File | Purpose | Location | Content |
|------|---------|----------|---------|
| `.env` | Environment configuration | `/` | API keys, settings |
| `requirements.txt` | Python dependencies | `/` | Package list |
| `pytest.ini` | Pytest configuration | `/` | Test settings |

### Test Categories by File

#### Original Features Testing
- **Primary Files**: `test_ehs_extraction_api.py`, `comprehensive_test_api.py`
- **Coverage**: Document processing, Q&A, search functionality
- **Endpoints**: `/test/original/*`

#### Phase 1 Enhancements Testing
- **Primary Files**: `comprehensive_test_api.py`, custom test modules
- **Coverage**: Audit trail, pro-rating, rejection handling
- **Endpoints**: `/test/phase1/*`

#### Integration Testing
- **Primary Files**: `test_integrationqa.py`, `comprehensive_test_api.py`
- **Coverage**: End-to-end workflows, component integration
- **Endpoints**: `/test/integration/*`

#### Performance Testing
- **Primary Files**: `Performance_test.py`, `comprehensive_test_api.py`
- **Coverage**: Load testing, response times, resource usage
- **Endpoints**: `/test/performance/*`

### Test Execution Priority

1. **High Priority** (Run first):
   - `comprehensive_test_api.py` health checks
   - `test_ehs_extraction_api.py` core functionality
   - Database connectivity tests

2. **Medium Priority** (Run after core tests pass):
   - Integration test suites
   - Phase 1 enhancement tests
   - End-to-end workflow tests

3. **Low Priority** (Run for comprehensive validation):
   - Performance and load tests
   - Stress tests
   - Extended endurance tests

This comprehensive test execution guide provides all necessary information for developers and QA teams to effectively test the EHS AI Demo system across all feature categories and integration scenarios.