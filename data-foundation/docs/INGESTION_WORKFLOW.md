# EHS Document Ingestion Workflow

This document provides comprehensive documentation for the EHS (Environmental, Health, Safety) document ingestion system that processes utility bills, water bills, and waste manifests and stores them in Neo4j.

## Table of Contents

1. [Overview](#overview)
2. [Current Status](#current-status)
3. [Architecture and Components](#architecture-and-components)
4. [Directory Structure](#directory-structure)
5. [Supported Document Types](#supported-document-types)
6. [API Endpoints and Usage](#api-endpoints-and-usage)
7. [Batch Ingestion API Endpoint](#batch-ingestion-api-endpoint)
8. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
9. [Testing Procedures](#testing-procedures)
10. [Data Flow and Processing Steps](#data-flow-and-processing-steps)
11. [Neo4j Schema and Relationships](#neo4j-schema-and-relationships)
12. [Configuration and Environment Variables](#configuration-and-environment-variables)
13. [Error Handling and Troubleshooting](#error-handling-and-troubleshooting)
14. [Sample Curl Commands and Responses](#sample-curl-commands-and-responses)

## Overview

The EHS Document Ingestion Workflow is a LangGraph-based system that orchestrates the complete pipeline from document upload to knowledge graph storage. It processes EHS documents (utility bills, water bills, waste manifests) through multiple stages: validation, parsing, data extraction, transformation, validation, and loading into Neo4j.

### Key Features

- **LangGraph-based workflow orchestration** for reliable, stateful processing
- **Multi-format document support** with LlamaParse integration
- **Comprehensive data extraction** using specialized extractors
- **Neo4j knowledge graph storage** with rich entity relationships
- **Automatic emissions calculations** for environmental impact tracking
- **Robust error handling and retry mechanisms**
- **Real-time processing status tracking**
- **Comprehensive validation** at multiple stages

## Current Status

**Last Updated**: August 18, 2025

### ✅ Fully Operational Features

1. **Document Ingestion Pipeline**
   - Electric bill ingestion: Working (130,000 kWh, $15,432.89)
   - Water bill ingestion: Working (250,000 gallons, $4,891.50)
   - Waste manifest ingestion: Working (15 tons)
   - Neo4j knowledge graph population: Working
   - Emission calculations: Automatically generated for all documents

2. **Batch Ingestion Script** (`scripts/ingest_all_documents.py`)
   - Database clearing functionality: Implemented and tested
   - Comprehensive logging and error handling: Working
   - Processing time: ~90-120 seconds for all 3 documents

3. **API Endpoints**
   - Upload API (port 8001): `/upload` - Working
   - Extraction API (port 8005): All endpoints operational
   - **NEW** Batch Ingestion API: `/api/v1/ingest/batch` - Working

4. **Data Extraction**
   - Case-insensitive label matching: Fixed (handles both 'Utilitybill' and 'UtilityBill')
   - All extraction workflows: Operational
   - Query performance: Optimized

### 🔧 Recent Updates

- **August 18, 2025**: 
  - Added database clearing to `ingest_all_documents.py`
  - Implemented batch ingestion API endpoint
  - Fixed case sensitivity issues in extraction queries
  - Updated documentation with comprehensive examples

### 📊 Test Results

- All 3 documents ingest successfully
- Extraction endpoints return correct data
- Neo4j populated with 70+ nodes and 80+ relationships
- Batch API endpoint tested and operational

### 🚀 Quick Start

For immediate batch ingestion of all documents:

```bash
curl -X POST http://localhost:8005/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"clear_database": true}'
```

This will clear the database and ingest all three sample documents in ~90 seconds.

## Architecture and Components

### Core Components

1. **IngestionWorkflow** (`backend/src/workflows/ingestion_workflow.py`)
   - Main LangGraph workflow orchestrator
   - Manages document processing pipeline
   - Handles state transitions and error recovery

2. **Document Parser** (`backend/src/parsers/llama_parser.py`)
   - Uses LlamaParse API for PDF processing
   - Extracts text and table data
   - Document type detection

3. **Data Extractors** (`backend/src/extractors/ehs_extractors.py`)
   - UtilityBillExtractor
   - WaterBillExtractor
   - WasteManifestExtractor
   - PermitExtractor
   - InvoiceExtractor

4. **Document Indexer** (`backend/src/indexing/document_indexer.py`)
   - Creates hybrid search indexes
   - Handles document retrieval
   - Vector and text-based search

5. **API Services**
   - EHS extraction API (`backend/src/ehs_extraction_api.py`)

### Workflow Stages

The ingestion workflow follows these stages:

```
validate → parse → extract → transform → validate_data → load → index → complete
     ↓         ↓         ↓          ↓           ↓          ↓       ↓        ↓
   [error handling and retry logic throughout]
```

## Directory Structure

```
data-foundation/
├── backend/
│   ├── src/
│   │   ├── workflows/
│   │   │   └── ingestion_workflow.py      # Main ingestion workflow
│   │   ├── parsers/
│   │   │   └── llama_parser.py            # Document parsing
│   │   ├── extractors/
│   │   │   └── ehs_extractors.py          # Data extraction logic
│   │   ├── indexing/
│   │   │   └── document_indexer.py        # Search indexing
│   │   ├── ehs_extraction_api.py          # EHS extraction API
│   ├── example.env                        # Environment configuration template
│   └── requirements.txt                   # Python dependencies
├── data/                                  # Sample documents
│   ├── electric_bill.pdf                 # Sample utility bill
│   ├── water_bill.pdf                    # Sample water bill
│   └── waste_manifest.pdf                # Sample waste manifest
├── scripts/
│   ├── ingest_all_documents.py           # Batch ingestion script
│   └── logs/                             # Processing logs
└── docs/
    └── INGESTION_WORKFLOW.md             # This documentation
```

## Supported Document Types

### 1. Utility Bills (`utility_bill`)
- **File**: `data/electric_bill.pdf`
- **Extracts**: 
  - Account information
  - Billing period and consumption (kWh)
  - Peak and off-peak usage
  - Cost breakdown and charges
  - Meter readings
  - Customer and facility information
  - Provider details
- **Entities Created**: Document, UtilityBill, Facility, Customer, UtilityProvider, Meter, Emission

### 2. Water Bills (`water_bill`)
- **File**: `data/water_bill.pdf`
- **Extracts**:
  - Water consumption (gallons/cubic meters)
  - Billing period and costs
  - Service charges and fees
  - Meter readings
  - Customer and facility information
- **Entities Created**: Document, WaterBill, Facility, Customer, UtilityProvider, Meter, Emission

### 3. Waste Manifests (`waste_manifest`)
- **File**: `data/waste_manifest.pdf`
- **Extracts**:
  - Manifest tracking information
  - Waste types and quantities
  - Generator information
  - Transporter details
  - Disposal facility information
  - Waste item specifications
- **Entities Created**: Document, WasteManifest, WasteShipment, WasteGenerator, Transporter, DisposalFacility, WasteItem, Emission

## API Endpoints and Usage

### Main Document Upload API

**Endpoint**: `POST /upload`
**Port**: 8005 (default)
**Host**: `http://localhost:8005`

#### Upload Parameters
- `file`: Document file (PDF)
- `chunkNumber`: Chunk number for large file uploads
- `totalChunks`: Total number of chunks
- `originalname`: Original filename
- `model`: LLM model to use
- `uri`: Neo4j connection URI
- `userName`: Neo4j username
- `password`: Neo4j password
- `database`: Neo4j database name

### EHS Data Extraction API

**Base URL**: `http://localhost:8005`

#### Available Endpoints

1. **Health Check**
   ```
   GET /health
   ```

2. **Extract Electrical Consumption**
   ```
   POST /api/v1/extract/electrical-consumption
   ```

3. **Extract Water Consumption**
   ```
   POST /api/v1/extract/water-consumption
   ```

4. **Extract Waste Generation**
   ```
   POST /api/v1/extract/waste-generation
   ```

5. **Custom Extraction**
   ```
   POST /api/v1/extract/custom
   ```

6. **Get Query Types**
   ```
   GET /api/v1/query-types
   ```

## Batch Ingestion API Endpoint

### Batch Document Ingestion
```
POST /api/v1/ingest/batch
```

This endpoint runs batch ingestion of all three EHS documents (electric bill, water bill, and waste manifest) in a single operation.

#### Request Body

```json
{
  "clear_database": true  // Optional, defaults to true
}
```

#### Parameters
- `clear_database` (boolean, optional): Whether to clear the Neo4j database before ingestion. Default: `true`

#### Response

```json
{
  "status": "success",
  "message": "Batch ingestion completed. 3/3 documents processed successfully.",
  "data": {
    "documents_processed": [
      {"document_type": "electric_bill", "status": "success"},
      {"document_type": "water_bill", "status": "success"},
      {"document_type": "waste_manifest", "status": "success"}
    ],
    "successful_ingestions": 3,
    "total_nodes_created": 23,
    "total_relationships_created": 23,
    "database_cleared": true
  },
  "metadata": {
    "script_path": "/path/to/scripts/ingest_all_documents.py",
    "return_code": 0,
    "python_version": "3.11.x",
    "generated_at": "2025-08-18T10:30:45.123456"
  },
  "processing_time": 93.45,
  "errors": []
}
```

#### Example Usage

```bash
# Ingest all documents with database clearing
curl -X POST http://localhost:8005/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"clear_database": true}'

# Ingest all documents without clearing database
curl -X POST http://localhost:8005/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"clear_database": false}'
```

#### Important Notes

1. **Database Clearing**: By default, the endpoint clears all Neo4j data before ingestion. Set `clear_database` to `false` to preserve existing data.

2. **Processing Time**: The batch ingestion typically takes 90-120 seconds to complete, processing all three documents sequentially.

3. **Script Execution**: This endpoint executes the `scripts/ingest_all_documents.py` script, which:
   - Clears the database (if requested)
   - Ingests electric_bill.pdf
   - Ingests water_bill.pdf
   - Ingests waste_manifest.pdf
   - Returns comprehensive statistics

4. **Error Handling**: If any document fails to process, the endpoint will continue with remaining documents and report partial success.

## Step-by-Step Execution Guide

### Prerequisites

1. **Environment Setup**
   ```bash
   cd data-foundation
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

2. **Environment Variables**
   ```bash
   cp backend/example.env backend/.env
   # Edit .env with your configuration
   ```

3. **Neo4j Database**
   - Ensure Neo4j is running on `bolt://localhost:7687`
   - Default credentials: `neo4j / EhsAI2024!`

### Method 1: Using the Batch Ingestion Script (Recommended)

```bash
# From the data-foundation directory
python3 scripts/ingest_all_documents.py
```

This script will:
- Process all three sample documents
- Show detailed progress and validation results
- Display comprehensive database statistics
- Generate detailed logs

### Method 2: Using the Web API

1. **Start the API Server**
   ```bash
   cd backend
   python3 score.py
   ```
   Server runs on `http://localhost:8005`

2. **Option A: Batch Ingestion (Recommended)**
   ```bash
   # Ingest all documents in one operation
   curl -X POST "http://localhost:8005/api/v1/ingest/batch" \
     -H "Content-Type: application/json" \
     -d '{"clear_database": true}'
   ```

3. **Option B: Upload Documents Individually**
   ```bash
   # Upload electric bill
   curl -X POST "http://localhost:8005/upload" \
     -F "file=@../data/electric_bill.pdf" \
     -F "chunkNumber=1" \
     -F "totalChunks=1" \
     -F "originalname=electric_bill.pdf" \
     -F "model=gpt-4o" \
     -F "uri=bolt://localhost:7687" \
     -F "userName=neo4j" \
     -F "password=EhsAI2024!" \
     -F "database=neo4j"
   ```

### Method 3: Using the Ingestion Workflow Directly

```python
from backend.src.workflows.ingestion_workflow import IngestionWorkflow
import os

# Initialize workflow
workflow = IngestionWorkflow(
    llama_parse_api_key=os.getenv("LLAMA_PARSE_API_KEY"),
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="EhsAI2024!",
    llm_model="gpt-4o-2024-11-20"
)

# Process document
result = workflow.process_document(
    file_path="data/electric_bill.pdf",
    document_id="electric_bill_001",
    metadata={"source": "api_upload", "document_type": "utility_bill"}
)

print(f"Status: {result['status']}")
print(f"Nodes created: {len(result.get('neo4j_nodes', []))}")
print(f"Relationships created: {len(result.get('neo4j_relationships', []))}")
```

## Testing Procedures

### 1. Unit Testing

```bash
cd backend
python3 -m pytest tests/ -v
```

### 2. Integration Testing

```bash
# Test complete pipeline
./test_complete_pipeline.py

# Test specific document types
python3 scripts/test_document_pipeline.py
```

### 3. API Testing

```bash
# Test all endpoints
./test_all_endpoints.sh

# Test EHS extraction API
./test_ehs_api_comprehensive.sh
```

### 4. Performance Testing

```bash
python3 backend/Performance_test.py
```

### 5. Manual Verification

```cypher
// Check document ingestion
MATCH (d:Document) RETURN count(d) as document_count;

// Check utility bill data
MATCH (d:Document)-[:EXTRACTED_TO]->(ub:UtilityBill)-[:BILLED_TO]->(f:Facility)
RETURN d.id, ub.total_kwh, ub.total_cost, f.name;

// Check emissions calculations
MATCH (e:Emission) RETURN e.amount, e.unit, e.source_type, e.calculation_method;

// Check complete data model
MATCH (n)-[r]->(m) 
RETURN labels(n), type(r), labels(m), count(*) 
ORDER BY count(*) DESC;
```

## Data Flow and Processing Steps

### 1. Document Validation
- File existence and size checks
- Document type detection using LlamaParse
- Format validation

### 2. Document Parsing
- PDF text extraction using LlamaParse API
- Table extraction and structure preservation
- Metadata extraction

### 3. Data Extraction
- Document-specific extractor selection
- Structured data extraction using LLM
- Field mapping and normalization

### 4. Data Transformation
- Neo4j schema mapping
- Entity and relationship creation
- Property assignment and type conversion

### 5. Data Validation
- Required field verification
- Data range and format validation
- Business rule enforcement

### 6. Neo4j Loading
- Node creation with labels and properties
- Relationship establishment
- Transaction management

### 7. Document Indexing
- Vector embedding generation
- Search index creation
- Hybrid retrieval setup

## Neo4j Schema and Relationships

### Node Types

#### Core Entities
- **Document**: Base document node
- **Facility**: Physical locations
- **Customer**: Billing entities
- **UtilityProvider**: Service providers

#### Bill-Specific Entities
- **UtilityBill**: Electric bill data
- **WaterBill**: Water bill data
- **WasteManifest**: Waste tracking documents

#### Waste-Specific Entities
- **WasteShipment**: Waste transportation records
- **WasteGenerator**: Waste origin facilities
- **Transporter**: Waste transportation companies
- **DisposalFacility**: Waste disposal locations
- **WasteItem**: Individual waste components

#### Measurement Entities
- **Meter**: Utility measurement devices
- **Emission**: Environmental impact calculations

### Relationship Types

- **EXTRACTED_TO**: Document → Bill entities
- **TRACKS**: Document → WasteManifest
- **BILLED_TO**: Bill → Facility
- **BILLED_FOR**: Bill → Customer
- **PROVIDED_BY**: Bill → UtilityProvider
- **MONITORS**: Meter → Facility
- **RECORDED_IN**: Meter → Bill
- **RESULTED_IN**: Bill/Shipment → Emission
- **DOCUMENTS**: WasteManifest → WasteShipment
- **GENERATED_BY**: WasteShipment → WasteGenerator
- **TRANSPORTED_BY**: WasteShipment → Transporter
- **DISPOSED_AT**: WasteShipment → DisposalFacility
- **CONTAINS_WASTE**: WasteShipment → WasteItem

### Sample Schema Visualization

```cypher
// Utility Bill Flow
(Document)-[:EXTRACTED_TO]->(UtilityBill)-[:BILLED_TO]->(Facility)
(UtilityBill)-[:BILLED_FOR]->(Customer)
(UtilityBill)-[:PROVIDED_BY]->(UtilityProvider)
(UtilityBill)-[:RESULTED_IN]->(Emission)
(Meter)-[:MONITORS]->(Facility)
(Meter)-[:RECORDED_IN]->(UtilityBill)

// Waste Manifest Flow
(Document)-[:TRACKS]->(WasteManifest)-[:DOCUMENTS]->(WasteShipment)
(WasteShipment)-[:GENERATED_BY]->(WasteGenerator)
(WasteShipment)-[:TRANSPORTED_BY]->(Transporter)
(WasteShipment)-[:DISPOSED_AT]->(DisposalFacility)
(WasteShipment)-[:CONTAINS_WASTE]->(WasteItem)
(WasteShipment)-[:RESULTED_IN]->(Emission)
```

## Configuration and Environment Variables

### Required Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=""                    # Required for LLM processing

# LlamaParse Configuration  
LLAMA_PARSE_API_KEY=""               # Required for PDF parsing

# Neo4j Configuration
NEO4J_URI="bolt://localhost:7687"    # Neo4j connection string
NEO4J_USERNAME="neo4j"               # Neo4j username
NEO4J_PASSWORD="EhsAI2024!"          # Neo4j password
NEO4J_DATABASE="neo4j"               # Neo4j database name

# LLM Model Configuration
LLM_MODEL_CONFIG_openai_gpt_4o="gpt-4o-2024-11-20,openai_api_key"

# Processing Configuration
NUMBER_OF_CHUNKS_TO_COMBINE=6        # Chunk processing batch size
UPDATE_GRAPH_CHUNKS_PROCESSED=20     # Graph update frequency
MAX_TOKEN_CHUNK_SIZE=2000            # Maximum tokens per chunk

# Embedding Configuration
EMBEDDING_MODEL="all-MiniLM-L6-v2"   # Embedding model for indexing
IS_EMBEDDING="TRUE"                   # Enable embedding generation
```

### Optional Configuration

```bash
# Logging and Monitoring
LANGCHAIN_TRACING_V2=""              # Enable LangChain tracing
LANGCHAIN_API_KEY=""                 # LangChain API key
LANGCHAIN_PROJECT=""                 # LangChain project name

# File Storage
GCS_FILE_CACHE="False"               # Use Google Cloud Storage
AWS_ACCESS_KEY_ID=""                 # AWS credentials
AWS_SECRET_ACCESS_KEY=""             # AWS credentials

# Performance Tuning
KNN_MIN_SCORE="0.94"                 # Similarity search threshold
DUPLICATE_SCORE_VALUE=0.97           # Duplicate detection threshold
```

### Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Environment verification
python3 -c "import os; print('✓ Python environment ready')"
```

## Error Handling and Troubleshooting

### Common Issues and Solutions

#### 1. Environment Setup Issues

**Issue**: Missing environment variables
```
ERROR: Missing required environment variables: LLAMA_PARSE_API_KEY
```

**Solution**:
```bash
# Check environment file
cat backend/.env

# Verify required variables
python3 -c "import os; print(os.getenv('LLAMA_PARSE_API_KEY'))"

# Update .env file with correct values
```

#### 2. Neo4j Connection Issues

**Issue**: Neo4j connection failures
```
ERROR: Failed to connect to Neo4j at bolt://localhost:7687
```

**Solutions**:
```bash
# Check Neo4j status
systemctl status neo4j

# Verify Neo4j is running
neo4j status

# Test connection
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'EhsAI2024!'))
with driver.session() as session:
    result = session.run('RETURN 1')
    print('✓ Neo4j connection successful')
driver.close()
"
```

#### 3. Document Processing Errors

**Issue**: LlamaParse API errors
```
ERROR: Parsing error: API key invalid or quota exceeded
```

**Solutions**:
```bash
# Verify API key
curl -H "Authorization: Bearer $LLAMA_PARSE_API_KEY" \
     https://api.cloud.llamaindex.ai/v1/parsing/

# Check quota usage
# Monitor API usage in LlamaParse dashboard
```

**Issue**: File not found errors
```
ERROR: Validation error: File not found: data/electric_bill.pdf
```

**Solutions**:
```bash
# Check file existence
ls -la data/

# Verify file permissions
chmod 644 data/*.pdf

# Check absolute vs relative paths
pwd
realpath data/electric_bill.pdf
```

#### 4. Extraction and Validation Issues

**Issue**: Missing required fields during validation
```
ERROR: Missing required field: total_kwh
```

**Solutions**:
- Review document content quality
- Check LLM extraction prompts
- Verify document type detection
- Examine parsed content for completeness

**Issue**: Validation warnings
```
WARNING: Unusual kWh value detected
```

**Solutions**:
- Review extracted values for accuracy
- Adjust validation thresholds if appropriate
- Check document parsing for extraction errors

#### 5. Performance Issues

**Issue**: Slow processing times
```
INFO: Processing took 120 seconds for single document
```

**Solutions**:
```bash
# Check system resources
htop
nvidia-smi  # If using GPU

# Optimize chunk processing
export UPDATE_GRAPH_CHUNKS_PROCESSED=10
export NUMBER_OF_CHUNKS_TO_COMBINE=3

# Use faster LLM model
export LLM_MODEL="gpt-4o-mini"
```

### Debugging Tools

#### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. Neo4j Browser Queries

```cypher
// Check processing status
MATCH (d:Document) 
RETURN d.file_path, d.document_type, d.uploaded_at
ORDER BY d.uploaded_at DESC;

// Check for incomplete processing
MATCH (d:Document) 
WHERE NOT (d)-[:EXTRACTED_TO]->()
RETURN d;

// Examine data quality
MATCH (ub:UtilityBill) 
WHERE ub.total_kwh IS NULL OR ub.total_cost IS NULL
RETURN ub;
```

#### 3. Workflow State Inspection

```python
# Check workflow state during processing
result = workflow.process_document(...)
print(f"Status: {result['status']}")
print(f"Errors: {result['errors']}")
print(f"Validation: {result['validation_results']}")
print(f"Extracted data keys: {list(result.get('extracted_data', {}).keys())}")
```

### Log Analysis

#### Processing Logs Location
- Script logs: `scripts/logs/ingestion_YYYYMMDD_HHMMSS.log`
- API logs: `backend/api.log`
- Test logs: Various `*test*.log` files

#### Key Log Patterns

```bash
# Check for successful processing
grep "PROCESSED SUCCESSFULLY" scripts/logs/*.log

# Find validation issues
grep "WARNING\|ERROR" scripts/logs/*.log

# Check Neo4j operations
grep "Created node\|Created relationship" scripts/logs/*.log

# Monitor processing times
grep "Processing Time:" scripts/logs/*.log
```

## Sample Curl Commands and Responses

### 1. Upload Electric Bill

```bash
curl -X POST "http://localhost:8005/upload" \
  -F "file=@data/electric_bill.pdf" \
  -F "chunkNumber=1" \
  -F "totalChunks=1" \
  -F "originalname=electric_bill.pdf" \
  -F "model=gpt-4o" \
  -F "uri=bolt://localhost:7687" \
  -F "userName=neo4j" \
  -F "password=EhsAI2024!" \
  -F "database=neo4j"
```

**Expected Response**:
```json
{
  "status": "Success",
  "message": "Source Node Created Successfully",
  "data": {
    "file_size": 2458392,
    "file_name": "electric_bill.pdf",
    "file_extension": "pdf",
    "message": "Chunk 1/1 saved"
  }
}
```

### 2. Extract Document Processing

```bash
curl -X POST "http://localhost:8005/extract" \
  -F "uri=bolt://localhost:7687" \
  -F "userName=neo4j" \
  -F "password=EhsAI2024!" \
  -F "database=neo4j" \
  -F "fileName=electric_bill.pdf" \
  -F "model=gpt-4o" \
  -F "allowedNodes=" \
  -F "allowedRelationship="
```

**Expected Response**:
```json
{
  "fileName": "electric_bill.pdf",
  "nodeCount": 12,
  "relationshipCount": 18,
  "total_processing_time": 45.23,
  "status": "Completed",
  "model": "gpt-4o",
  "success_count": 1
}
```

### 3. Extract Electrical Consumption Data

```bash
curl -X POST "http://localhost:8001/api/v1/extract/electrical-consumption" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_filter": {
      "facility_name": "Apex Manufacturing"
    },
    "date_range": {
      "start_date": "2025-07-01",
      "end_date": "2025-07-31"
    },
    "include_emissions": true,
    "include_cost_analysis": true,
    "output_format": "json"
  }'
```

**Expected Response**:
```json
{
  "status": "success",
  "message": "Electrical consumption data extracted successfully",
  "data": {
    "query_type": "utility_consumption",
    "facility_filter": {
      "facility_name": "Apex Manufacturing"
    },
    "date_range": {
      "start_date": "2025-07-01",
      "end_date": "2025-07-31"
    },
    "include_emissions": true,
    "include_cost_analysis": true,
    "report_data": {
      "total_kwh": 130000,
      "total_cost": 18245.67,
      "peak_kwh": 70000,
      "off_peak_kwh": 60000,
      "emissions_kg_co2": 52000,
      "billing_periods": [
        {
          "start_date": "2025-07-01",
          "end_date": "2025-07-31",
          "kwh": 130000,
          "cost": 18245.67
        }
      ]
    },
    "file_path": "./reports/ehs_report_utility_consumption_20250818_120000.json"
  },
  "metadata": {
    "total_queries": 5,
    "successful_queries": 5,
    "total_records": 1,
    "processing_status": "completed",
    "generated_at": "2025-08-18T12:00:00.000Z"
  },
  "processing_time": 2.45,
  "errors": null
}
```

### 4. Extract Water Consumption Data

```bash
curl -X POST "http://localhost:8001/api/v1/extract/water-consumption" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_filter": {
      "facility_name": "Apex Manufacturing"
    },
    "include_meter_details": true,
    "include_emissions": true,
    "output_format": "json"
  }'
```

**Expected Response**:
```json
{
  "status": "success",
  "message": "Water consumption data extracted successfully",
  "data": {
    "query_type": "water_consumption",
    "report_data": {
      "total_gallons": 13470,
      "total_cost": 425.89,
      "billing_periods": [
        {
          "start_date": "2025-07-01",
          "end_date": "2025-07-31",
          "gallons": 13470,
          "cost": 425.89
        }
      ],
      "meter_details": [
        {
          "meter_id": "WTR-5521-A",
          "type": "water",
          "usage": 13470,
          "unit": "gallons"
        }
      ],
      "emissions_kg_co2": 2.694
    }
  },
  "processing_time": 1.82
}
```

### 5. Extract Waste Generation Data

```bash
curl -X POST "http://localhost:8001/api/v1/extract/waste-generation" \
  -H "Content-Type: application/json" \
  -d '{
    "include_disposal_details": true,
    "include_transport_details": true,
    "include_emissions": true,
    "hazardous_only": false,
    "output_format": "json"
  }'
```

**Expected Response**:
```json
{
  "status": "success",
  "message": "Waste generation data extracted successfully",
  "data": {
    "query_type": "waste_generation",
    "report_data": {
      "total_weight": 15.5,
      "weight_unit": "tons",
      "manifests": [
        {
          "manifest_tracking_number": "CA8888813579",
          "issue_date": "2025-08-15",
          "total_quantity": 15.5,
          "disposal_method": "landfill",
          "generator": "Apex Manufacturing",
          "transporter": "SafeHaul Environmental Services",
          "disposal_facility": "Metro Waste Solutions"
        }
      ],
      "waste_items": [
        {
          "waste_type": "Industrial Solid Waste - Non-Hazardous",
          "quantity": 15.5,
          "container_type": "Roll-off Container",
          "container_count": 3
        }
      ],
      "emissions_metric_tons_co2e": 7.75
    }
  },
  "processing_time": 3.12
}
```

### 6. Health Check

```bash
curl -X GET "http://localhost:8001/health"
```

**Expected Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-18T12:00:00.000Z",
  "neo4j_connection": true,
  "version": "1.0.0"
}
```

### 7. Get Available Query Types

```bash
curl -X GET "http://localhost:8001/api/v1/query-types"
```

**Expected Response**:
```json
{
  "query_types": [
    {
      "value": "facility_emissions",
      "name": "Facility Emissions",
      "description": "Extract facility-level emission data and calculations"
    },
    {
      "value": "utility_consumption",
      "name": "Utility Consumption",
      "description": "Extract electrical and utility consumption data from bills"
    },
    {
      "value": "water_consumption",
      "name": "Water Consumption",
      "description": "Extract water usage data from water bills"
    },
    {
      "value": "waste_generation",
      "name": "Waste Generation",
      "description": "Extract waste generation data from manifests"
    },
    {
      "value": "compliance_status",
      "name": "Compliance Status",
      "description": "Extract compliance and permit status information"
    },
    {
      "value": "trend_analysis",
      "name": "Trend Analysis",
      "description": "Extract data for trend analysis over time"
    },
    {
      "value": "custom",
      "name": "Custom",
      "description": "Custom queries provided by user"
    }
  ]
}
```

### Error Response Examples

#### 1. Missing Required Parameters

```bash
curl -X POST "http://localhost:8005/upload" \
  -F "file=@data/electric_bill.pdf"
```

**Response**:
```json
{
  "status": "Failed",
  "message": "Missing required parameters",
  "error": "Missing form fields: model, uri, userName, password, database"
}
```

#### 2. Neo4j Connection Error

```bash
curl -X POST "http://localhost:8001/api/v1/extract/electrical-consumption" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response**:
```json
{
  "status": "Failed",
  "message": "Extraction failed: Failed to connect to Neo4j",
  "error": "Unable to retrieve routing information",
  "processing_time": null
}
```

#### 3. Invalid Document Format

```bash
curl -X POST "http://localhost:8005/upload" \
  -F "file=@invalid_file.txt" \
  -F "originalname=invalid_file.txt" \
  # ... other parameters
```

**Response**:
```json
{
  "status": "Failed",
  "message": "Unable to upload file: Invalid file format. Only PDF files are supported.",
  "error": "Validation error: Unsupported file type",
  "file_name": "invalid_file.txt"
}
```

---

## Additional Resources

- **Neo4j Browser**: `http://localhost:7474` (for database exploration)
- **API Documentation**: `http://localhost:8001/api/docs` (Swagger UI)
- **Test Scripts**: Located in `scripts/` directory
- **Sample Data**: Located in `data/` directory
- **Log Files**: Generated in `scripts/logs/` and `backend/`

For additional support or questions, refer to the implementation files in the `backend/src/` directory or examine the test scripts for usage examples.