# Real PDF Ingestion Test Script

This script (`test_real_pdf_ingestion.py`) provides comprehensive testing of the complete PDF ingestion workflow including PDF parsing with LlamaParse, data extraction, and risk assessment.

## Features

✅ **Complete Workflow Testing**
- PDF parsing with LlamaParse
- Data extraction and processing
- Neo4j database storage
- Risk assessment integration
- Error handling and logging

✅ **Recursion Error Detection**
- Low recursion limit (100) to catch infinite recursion quickly
- Detailed stack trace logging
- Specific recursion error handling

✅ **Comprehensive Logging**
- Detailed logs to `/tmp/real_pdf_ingestion_test.log`
- Console output with progress updates
- JSON results file for analysis

✅ **Real Data Testing**
- Uses actual PDF file: `document-1.pdf` (46KB)
- Tests with real API keys and environment
- Creates test facility in Neo4j

## Usage

### 1. Navigate to Backend Directory
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3. Run the Test
```bash
python3 tmp/test_real_pdf_ingestion.py
```

### 4. Monitor Progress
The script will show real-time progress and create detailed logs:

```
🚀 STARTING REAL PDF INGESTION TEST
🔧 Step 1: Validating environment...
✅ OPENAI_API_KEY: ********...
✅ LLAMA_PARSE_API_KEY: ********...
📄 Step 2: Validating test file...
✅ Test file found: document-1.pdf (46,123 bytes)
🔗 Step 3: Testing Neo4j connection...
✅ Neo4j connection successful
🏢 Step 4: Setting up test facility...
✅ Test facility created: TEST_FACILITY_PDF_001
🤖 Step 5: Initializing ingestion workflow...
✅ Workflow initialized successfully
📊 Step 6: Running document ingestion with risk assessment...
🔄 Starting workflow execution...
✅ Workflow execution completed in 45.23 seconds
🔍 Step 7: Validating results in Neo4j...
✅ Document found in Neo4j
```

## Output Files

### 1. Detailed Log
```
/tmp/real_pdf_ingestion_test.log
```
Contains comprehensive debug information including:
- Environment validation details
- Workflow step-by-step execution
- API calls and responses
- Error stack traces
- Performance timing

### 2. Results JSON
```
/tmp/real_pdf_ingestion_results.json
```
Contains structured test results including:
- Test status and timing
- Workflow execution results
- Risk assessment outcomes
- Error and warning details

## Test Configuration

The script is configured to:

- **Test File**: `/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/document-1.pdf`
- **Risk Assessment**: Enabled (set to `True`)
- **Recursion Limit**: 100 (to catch recursion errors quickly)
- **Timeout**: 300 seconds (5 minutes)
- **LLM Model**: GPT-4o for optimal performance

## Key Test Steps

1. **Environment Validation**: Checks all required API keys and environment variables
2. **File Validation**: Verifies test PDF file exists and is readable
3. **Neo4j Connection**: Tests database connectivity
4. **Test Facility Setup**: Creates a test facility for document association
5. **Workflow Initialization**: Sets up the ingestion workflow with risk assessment
6. **Document Ingestion**: Runs the complete workflow with real PDF processing
7. **Results Validation**: Verifies data was properly stored in Neo4j

## Expected Results

### Successful Test
- Status: `completed`
- Document parsed and processed
- Chunks created and stored in Neo4j
- Entities extracted
- Risk assessment completed (if enabled)
- Test facility linked to document

### Recursion Error Detection
If infinite recursion occurs:
- Status: `failed`
- `recursion_error_detected`: `true`
- Detailed stack trace in log file
- Error occurs quickly due to low recursion limit

## Troubleshooting

### Missing API Keys
Ensure your `.env` file contains:
```
OPENAI_API_KEY=your_key_here
LLAMA_PARSE_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Neo4j Connection Issues
1. Verify Neo4j is running: `sudo systemctl status neo4j`
2. Check connection details in `.env`
3. Test with Neo4j Browser: http://localhost:7474

### Permission Issues
Ensure the script is executable:
```bash
chmod +x tmp/test_real_pdf_ingestion.py
```

### Virtual Environment
Always run from activated virtual environment:
```bash
source venv/bin/activate
```

## Integration with Main Workflow

This test script replicates the same workflow used by:
- `run_batch_ingestion.py`
- API endpoints for document ingestion
- Risk assessment integrated workflows

The test results should match the behavior of the production ingestion pipeline.