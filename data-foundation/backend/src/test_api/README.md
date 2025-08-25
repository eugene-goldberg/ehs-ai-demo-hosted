# EHS AI Demo - Test Orchestrator

## Overview

The `run_all_tests.py` script serves as the single entry point for running all tests in the EHS AI Demo system. It orchestrates bash test scripts, Python unit tests, and generates comprehensive reports.

## Quick Start

```bash
# Run all tests with default settings
python3 run_all_tests.py

# Run all tests with HTML report
python3 run_all_tests.py --html-report

# Run only API tests in parallel
python3 run_all_tests.py --categories api --parallel

# Run with coverage metrics and verbose output
python3 run_all_tests.py --coverage --verbose
```

## Features

### Test Orchestration
- **Automatic API Server Management**: Starts and stops the test API server
- **Multi-Category Testing**: Supports API, integration, and unit test categories
- **Parallel Execution**: Option to run tests in parallel for faster execution
- **Error Handling**: Robust error handling with proper cleanup

### Test Integration
- **Bash Script Integration**: Runs existing bash test scripts:
  - `comprehensive_curl_tests.sh`
  - `ehs_api_tests.sh` 
  - `phase1_integration_tests.sh`
- **Python Unit Tests**: Discovers and runs pytest-based tests
- **Mixed Test Suites**: Combines different test types into unified reports

### Reporting
- **JSON Reports**: Machine-readable test results
- **HTML Reports**: Human-friendly visual reports with:
  - Test suite summaries
  - Individual test results
  - Performance metrics
  - Interactive expandable sections
- **Console Output**: Real-time progress and summary information
- **Coverage Metrics**: Test coverage calculation and reporting

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--categories` | Run specific test categories | `--categories api integration` |
| `--parallel` | Execute tests in parallel | `--parallel` |
| `--html-report` | Generate HTML report | `--html-report` |
| `--coverage` | Include coverage metrics | `--coverage` |
| `--no-api-server` | Don't start API server | `--no-api-server` |
| `--verbose` | Enable verbose logging | `--verbose` |

## Test Categories

### API Tests (`api`)
- Tests all API endpoints using curl commands
- Validates request/response formats
- Checks error handling
- Includes performance measurements

**Scripts:**
- `comprehensive_curl_tests.sh`
- `ehs_api_tests.sh`

### Integration Tests (`integration`)
- End-to-end workflow testing
- Multi-component interaction tests
- Data flow validation

**Scripts:**
- `phase1_integration_tests.sh`

### Unit Tests (`unit`)
- Python unit tests using pytest
- Isolated component testing
- Mock-based testing

**Discovery:** Automatically finds `test_*.py` and `*_test.py` files

## Installation

### Prerequisites
```bash
# Ensure Python 3.8+ is installed
python3 --version

# Install core dependencies (already in main requirements.txt)
pip3 install fastapi uvicorn
```

### Optional Testing Dependencies
```bash
# Install enhanced testing tools
pip3 install -r test_requirements.txt
```

## Usage Examples

### Basic Test Run
```bash
python3 run_all_tests.py
```

### API-Only Testing
```bash
python3 run_all_tests.py --categories api
```

### Parallel Execution with Reports
```bash
python3 run_all_tests.py --parallel --html-report --coverage
```

### CI/CD Integration
```bash
# For automated environments
python3 run_all_tests.py --no-api-server --verbose
echo "Exit code: $?"
```

## Output Structure

### Console Output
```
================================================================================
TEST EXECUTION SUMMARY
================================================================================
Total Test Suites: 4
  ✓ Passed: 3
  ✗ Failed: 1
  ⊝ Skipped: 0

Individual Tests: 45
  ✓ Passed: 42
  ✗ Failed: 3
  ⊝ Skipped: 0

Test Coverage: 72.5%

Total Duration: 127.3 seconds
```

### Generated Files
- **JSON Report**: `test_reports/test_report_YYYYMMDD_HHMMSS.json`
- **HTML Report**: `test_reports/test_report_YYYYMMDD_HHMMSS.html`
- **Log File**: `test_orchestrator.log`

### JSON Report Structure
```json
{
  "metadata": {
    "timestamp": "2025-08-23T07:33:15.123456",
    "total_duration": 127.3,
    "test_orchestrator_version": "1.0.0"
  },
  "summary": {
    "total_tests": 45,
    "total_suites": 4,
    "passed_tests": 42,
    "failed_tests": 3,
    "passed_suites": 3,
    "failed_suites": 1
  },
  "coverage": {
    "coverage_percentage": 72.5,
    "total_files": 40,
    "covered_files": 29
  },
  "results": [...],
  "categories": {...}
}
```

## Architecture

### Class Structure

#### `TestResult`
Represents individual test suite results with:
- Timing information
- Status tracking
- Output capture
- Metadata storage

#### `TestAPIServer`
Manages test API server lifecycle:
- Automatic startup/shutdown
- Health checking
- Port management

#### `TestOrchestrator`
Main orchestration class:
- Test discovery and execution
- Result aggregation
- Report generation
- Cleanup handling

### Error Handling
- Graceful server shutdown on interruption
- Test timeout handling (15 minutes for bash scripts, 5 minutes for Python)
- Resource cleanup on failure
- Detailed error reporting

### Performance Considerations
- Parallel execution for independent tests
- Configurable timeouts
- Memory-efficient output handling
- Incremental result processing

## Extending the Orchestrator

### Adding New Test Scripts
1. Place script in `test_scripts/` directory
2. Add to `bash_scripts` mapping in `TestOrchestrator.__init__()`
3. Ensure script outputs parseable test counts

### Adding New Test Categories
1. Update `choices` in argument parser
2. Add category handling in `run_tests_*` methods
3. Update documentation

### Custom Report Formats
Extend the `TestOrchestrator` class and override:
- `generate_json_report()`
- `generate_html_report()`
- Add new format methods

## Troubleshooting

### Common Issues

#### API Server Won't Start
```bash
# Check if port is in use
lsof -i :8001

# Use different port or kill existing process
python3 run_all_tests.py --no-api-server
```

#### Tests Time Out
```bash
# Run with verbose logging
python3 run_all_tests.py --verbose

# Check individual script execution
bash test_scripts/comprehensive_curl_tests.sh
```

#### Permission Errors
```bash
# Ensure scripts are executable
chmod +x test_scripts/*.sh
chmod +x run_all_tests.py
```

### Debug Mode
```bash
# Enable maximum verbosity
python3 run_all_tests.py --verbose 2>&1 | tee debug.log
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run All Tests
  run: |
    cd data-foundation/backend/src/test_api
    python3 run_all_tests.py --coverage --verbose
    
- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: test_reports/
```

### Jenkins Pipeline
```groovy
stage('Test') {
    steps {
        sh '''
            cd data-foundation/backend/src/test_api
            python3 run_all_tests.py --html-report --coverage
        '''
        publishHTML([
            allowMissing: false,
            alwaysLinkToLastBuild: true,
            keepAll: true,
            reportDir: 'test_reports',
            reportFiles: '*.html',
            reportName: 'Test Report'
        ])
    }
}
```

## Contributing

When adding new tests or modifying the orchestrator:

1. Follow existing patterns for test script integration
2. Ensure proper error handling and cleanup
3. Update documentation for new features
4. Test with both sequential and parallel execution
5. Verify report generation works correctly

## License

This test orchestrator is part of the EHS AI Demo project and follows the same licensing terms.