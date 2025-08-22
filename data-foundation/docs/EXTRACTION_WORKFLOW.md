# EHS Data Extraction Workflow Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture and Components](#architecture-and-components)
3. [File Locations and Directory Structure](#file-locations-and-directory-structure)
4. [Query Types and Categories](#query-types-and-categories)
5. [API Endpoints and Usage](#api-endpoints-and-usage)
6. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
7. [Testing Procedures](#testing-procedures)
8. [Query Patterns and Neo4j Cypher Queries](#query-patterns-and-neo4j-cypher-queries)
9. [Response Formats and Data Structures](#response-formats-and-data-structures)
10. [Configuration and Environment Variables](#configuration-and-environment-variables)
11. [Error Handling and Troubleshooting](#error-handling-and-troubleshooting)
12. [Sample curl Commands and Responses](#sample-curl-commands-and-responses)

---

## Overview

The EHS Data Extraction Workflow is a comprehensive system designed to query and retrieve Environmental, Health, and Safety (EHS) data from a Neo4j graph database. The system utilizes LangGraph for workflow orchestration and provides a RESTful API for accessing various types of EHS data including utility consumption, water consumption, and waste generation.

### Key Features

- **LangGraph-based workflow orchestration** for complex data extraction processes
- **Multiple query types** supporting various EHS data categories
- **RESTful API** with FastAPI for easy integration
- **Structured data extraction** with Pydantic models for validation
- **LLM-powered analysis** for generating insights from extracted data
- **Flexible output formats** (JSON, TXT) for different use cases
- **Comprehensive error handling** and logging throughout the workflow

### Expected Data Values

The system is designed to extract and process the following types of data:
- **Electrical consumption**: 130,000 kWh
- **Water consumption**: 250,000 gallons
- **Waste generation**: 15 tons

---

## Architecture and Components

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangGraph      │    │   Neo4j         │
│   REST API      │───▶│   Workflow       │───▶│   Graph DB      │
│                 │    │   Orchestration  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Models    │    │   State Machine  │    │   Cypher        │
│   (Pydantic)    │    │   Nodes          │    │   Queries       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │   LLM Analysis   │
                      │   (OpenAI/       │
                      │   Anthropic)     │
                      └──────────────────┘
```

### Core Components

1. **DataExtractionWorkflow** (`backend/src/workflows/extraction_workflow.py`)
   - Main workflow orchestrator using LangGraph
   - Manages state transitions and error handling
   - Coordinates between Neo4j queries and LLM analysis

2. **EHS Extraction API** (`backend/src/ehs_extraction_api.py`)
   - FastAPI-based REST service
   - Provides endpoints for different data extraction types
   - Handles request validation and response formatting

3. **Query Templates** (Built into workflow)
   - Pre-defined Cypher queries for various data types
   - Parameterized for flexible filtering
   - Optimized for EHS data relationships

4. **State Management**
   - Typed state definitions using TypedDict
   - Comprehensive error tracking and recovery
   - Processing time and metadata collection

---

## File Locations and Directory Structure

```
data-foundation/
├── backend/
│   └── src/
│       ├── workflows/
│       │   ├── extraction_workflow.py      # Main workflow implementation
│       │   └── ingestion_workflow.py       # Document ingestion workflow
│       ├── ehs_extraction_api.py           # FastAPI service
│       ├── api_response.py                 # Response utilities
│       └── reports/                        # Generated reports
│           ├── ehs_report_QueryType.UTILITY_CONSUMPTION_*.json
│           ├── ehs_report_QueryType.WATER_CONSUMPTION_*.json
│           └── ehs_report_QueryType.WASTE_GENERATION_*.json
├── docs/
│   ├── EXTRACTION_WORKFLOW.md             # This documentation
│   ├── implementation-status.md           # Project status
│   └── ehs_extraction_api_guide.md        # API guide
├── scripts/
│   ├── test_extraction_workflow.py        # Test scripts
│   └── test_complete_pipeline.py          # End-to-end tests
├── test_extraction_fix.sh                 # Extraction testing script
├── full_ingestion_extraction_test.sh      # Complete pipeline test
└── .venv/                                 # Python virtual environment
```

### Key Files

- **Main workflow**: `backend/src/workflows/extraction_workflow.py`
- **API service**: `backend/src/ehs_extraction_api.py`
- **Test scripts**: `test_extraction_fix.sh`, `full_ingestion_extraction_test.sh`
- **Reports directory**: `backend/src/reports/`

---

## Query Types and Categories

The system supports seven primary query types defined in the `QueryType` enum:

### 1. FACILITY_EMISSIONS
- **Purpose**: Extract facility-level emission data and calculations
- **Data Sources**: Facilities, UtilityBills, Emissions
- **Typical Use**: Environmental impact reporting, carbon footprint analysis

### 2. UTILITY_CONSUMPTION
- **Purpose**: Extract electrical and utility consumption data from bills
- **Data Sources**: UtilityBill, Document (with 'Utilitybill' or 'UtilityBill' labels)
- **Key Metrics**: Total kWh, costs, billing periods
- **Case Sensitivity Fix**: Handles both `Utilitybill` and `UtilityBill` labels

### 3. WATER_CONSUMPTION
- **Purpose**: Extract water usage data from water bills
- **Data Sources**: WaterBill, Meter, Customer, UtilityProvider, Emissions
- **Key Metrics**: Total gallons, meter readings, billing information

### 4. WASTE_GENERATION
- **Purpose**: Extract waste generation data from manifests
- **Data Sources**: WasteManifest, WasteShipment, WasteGenerator, Transporter, DisposalFacility, WasteItem
- **Key Metrics**: Waste quantities, disposal methods, transportation details

### 5. COMPLIANCE_STATUS
- **Purpose**: Extract compliance and permit status information
- **Data Sources**: Permit, Facility
- **Focus**: Permit expiry dates, compliance requirements

### 6. TREND_ANALYSIS
- **Purpose**: Extract data for trend analysis over time
- **Data Sources**: UtilityBill, Emissions
- **Use Case**: Time-series analysis, performance tracking

### 7. CUSTOM
- **Purpose**: Custom queries provided by user
- **Flexibility**: Allows arbitrary Cypher queries
- **Use Case**: Ad-hoc analysis, specialized reporting

---

## API Endpoints and Usage

### Base Configuration

- **Host**: `localhost` / `0.0.0.0`
- **Port**: `8005` (configurable via environment variable)
- **Base URL**: `http://localhost:8005`
- **API Documentation**: `http://localhost:8005/api/docs`

### Available Endpoints

#### 1. Health Check
```
GET /health
```
**Purpose**: Check API status and Neo4j connectivity

#### 2. Electrical Consumption Extraction
```
POST /api/v1/extract/electrical-consumption
```
**Purpose**: Extract electrical consumption data from utility bills
**Request Body**:
```json
{
  "facility_filter": {
    "facility_id": "optional_facility_id",
    "facility_name": "optional_facility_name"
  },
  "date_range": {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  },
  "output_format": "json",
  "include_emissions": true,
  "include_cost_analysis": true
}
```

#### 3. Water Consumption Extraction
```
POST /api/v1/extract/water-consumption
```
**Purpose**: Extract water consumption data from water bills
**Request Body**:
```json
{
  "facility_filter": {
    "facility_id": "optional_facility_id"
  },
  "date_range": {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  },
  "output_format": "json",
  "include_meter_details": true,
  "include_emissions": true
}
```

#### 4. Waste Generation Extraction
```
POST /api/v1/extract/waste-generation
```
**Purpose**: Extract waste generation data from waste manifests
**Request Body**:
```json
{
  "date_range": {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  },
  "output_format": "json",
  "include_disposal_details": true,
  "include_transport_details": true,
  "include_emissions": true,
  "hazardous_only": false
}
```

#### 5. Custom Data Extraction
```
POST /api/v1/extract/custom?query_type={query_type}&output_format={format}
```
**Purpose**: Extract custom EHS data using predefined or custom queries
**Parameters**:
- `query_type`: One of the QueryType enum values
- `output_format`: "json" or "txt"

#### 6. Query Types Information
```
GET /api/v1/query-types
```
**Purpose**: Get list of available query types and their descriptions

---

## Step-by-Step Execution Guide

### Prerequisites

1. **Environment Setup**:
   ```bash
   cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation
   source .venv/bin/activate
   ```

2. **Environment Variables**:
   - `NEO4J_URI`
   - `NEO4J_USERNAME`
   - `NEO4J_PASSWORD`
   - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
   - `LLM_MODEL` (optional, defaults to 'gpt-4')

### Starting the API Server

1. **Set Python Path**:
   ```bash
   export PYTHONPATH="/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/src:$PYTHONPATH"
   ```

2. **Start the Server**:
   ```bash
   python3 -m uvicorn backend.src.ehs_extraction_api:app --host 0.0.0.0 --port 8005
   ```

3. **Verify Server Status**:
   ```bash
   curl http://localhost:8005/health
   ```

### Running Extraction Workflows

#### Step 1: Prepare Query Configuration
```python
query_config = {
    "type": "utility_consumption",
    "parameters": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
}
```

#### Step 2: Initialize Workflow
```python
from workflows.extraction_workflow import DataExtractionWorkflow

workflow = DataExtractionWorkflow(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    llm_model="gpt-4"
)
```

#### Step 3: Execute Extraction
```python
result = workflow.extract_data(
    query_type="utility_consumption",
    parameters={"start_date": "2023-01-01"},
    output_format="json"
)
```

#### Step 4: Process Results
```python
if result["status"] == "completed":
    report_path = result["report_file_path"]
    print(f"Report saved to: {report_path}")
else:
    print(f"Errors: {result['errors']}")
```

### Workflow State Transitions

The LangGraph workflow follows this state machine:

```
[prepare_queries] → [identify_objects] → [execute_queries] → [analyze_results] → [generate_report] → [save_report] → [complete] → [END]
                                    ↓
                            [handle_error] ← (on error)
                                    ↓
                            (retry or fail)
```

---

## Testing Procedures

### Automated Test Scripts

#### 1. Basic Extraction Test (`test_extraction_fix.sh`)
```bash
./test_extraction_fix.sh
```
**Purpose**: Tests the electrical consumption endpoint with case sensitivity fixes
**Features**:
- Starts API server on port 8005
- Tests both direct and custom endpoints
- Validates response structure
- Checks for actual data records

#### 2. Full Pipeline Test (`full_ingestion_extraction_test.sh`)
```bash
./full_ingestion_extraction_test.sh
```
**Purpose**: End-to-end testing of ingestion and extraction
**Features**:
- Document ingestion testing
- Multiple query type validation
- Data integrity verification

### Manual Testing Commands

#### Test Health Endpoint
```bash
curl -X GET http://localhost:8005/health
```

#### Test Electrical Consumption
```bash
curl -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -d '{
    "output_format": "json",
    "include_emissions": true,
    "include_cost_analysis": true
  }'
```

#### Test Water Consumption
```bash
curl -X POST http://localhost:8005/api/v1/extract/water-consumption \
  -H "Content-Type: application/json" \
  -d '{
    "output_format": "json",
    "include_meter_details": true,
    "include_emissions": true
  }'
```

#### Test Waste Generation
```bash
curl -X POST http://localhost:8005/api/v1/extract/waste-generation \
  -H "Content-Type: application/json" \
  -d '{
    "output_format": "json",
    "include_disposal_details": true,
    "include_transport_details": true,
    "include_emissions": true,
    "hazardous_only": false
  }'
```

### Test Data Validation

The system expects to find specific data values:
- **Electrical consumption**: ~130,000 kWh
- **Water consumption**: ~250,000 gallons
- **Waste generation**: ~15 tons

### Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

---

## Query Patterns and Neo4j Cypher Queries

### Case Sensitivity Handling

The system handles case sensitivity issues in Neo4j labels:

```cypher
-- Handles both 'Utilitybill' and 'UtilityBill' labels
MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)
WHERE 'Utilitybill' IN labels(d) OR 'UtilityBill' IN labels(d)
RETURN d, b
ORDER BY b.billing_period_end DESC
```

### Utility Consumption Queries

#### Basic Utility Data Extraction
```cypher
MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)
WHERE 'Utilitybill' IN labels(d) OR 'UtilityBill' IN labels(d)
RETURN d, b
ORDER BY b.billing_period_end DESC
```

#### Aggregated Consumption Analysis
```cypher
MATCH (b:UtilityBill)
WHERE b.billing_period_start >= $start_date 
  AND b.billing_period_end <= $end_date
RETURN SUM(b.total_kwh) as total_consumption,
       AVG(b.total_cost) as avg_cost,
       COUNT(b) as bill_count
```

### Water Consumption Queries

#### Comprehensive Water Data
```cypher
MATCH (w:WaterBill)
OPTIONAL MATCH (w)-[:BILLED_TO]->(f:Facility)
OPTIONAL MATCH (w)-[:BILLED_FOR]->(c:Customer)
OPTIONAL MATCH (w)-[:PROVIDED_BY]->(p:UtilityProvider)
OPTIONAL MATCH (w)-[:RESULTED_IN]->(e:Emission)
OPTIONAL MATCH (m:Meter)-[:RECORDED_IN]->(w)
RETURN w, f, c, p, e, collect(m) as meters
ORDER BY w.billing_period_end DESC
```

#### Water Usage Aggregation
```cypher
MATCH (w:WaterBill)
WHERE w.billing_period_start >= $start_date 
  AND w.billing_period_end <= $end_date
RETURN SUM(w.total_gallons) as total_water_usage,
       AVG(w.total_cost) as avg_cost,
       COUNT(w) as bill_count
```

### Waste Generation Queries

#### Complete Waste Manifest Data
```cypher
MATCH (d:Document)-[:TRACKS]->(wm:WasteManifest)
WHERE 'Wastemanifest' IN labels(d) OR 'WasteManifest' IN labels(d)
MATCH (wm)-[:DOCUMENTS]->(ws:WasteShipment)
MATCH (ws)-[:GENERATED_BY]->(g:WasteGenerator)
MATCH (ws)-[:TRANSPORTED_BY]->(t:Transporter)  
MATCH (ws)-[:DISPOSED_AT]->(df:DisposalFacility)
OPTIONAL MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
OPTIONAL MATCH (ws)-[:RESULTED_IN]->(e:Emission)
RETURN d, wm, ws, g, t, df, collect(wi) as waste_items, collect(e) as emissions
ORDER BY ws.shipment_date DESC
```

#### Waste Generator Analysis
```cypher
MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
MATCH (ws)-[:GENERATED_BY]->(g:WasteGenerator)
MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
OPTIONAL MATCH (ws)-[:RESULTED_IN]->(e:Emission)
WHERE ws.shipment_date >= $start_date 
  AND ws.shipment_date <= $end_date
RETURN g.name as generator,
       g.epa_id as generator_epa_id,
       SUM(wi.quantity) as total_waste_quantity,
       wi.unit as quantity_unit,
       COUNT(DISTINCT wm) as manifest_count,
       COUNT(wi) as waste_item_count,
       SUM(e.amount) as total_emissions
ORDER BY total_waste_quantity DESC
```

#### Daily Waste Trends
```cypher
MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
WHERE ws.shipment_date >= $start_date 
  AND ws.shipment_date <= $end_date
WITH date(ws.shipment_date) as shipment_date, 
     wm, 
     ws,
     [(ws)-[:CONTAINS_WASTE]->(wi:WasteItem) | wi] as waste_items
RETURN shipment_date,
       COUNT(DISTINCT wm) as manifests_generated,
       SUM(reduce(total = 0, wi in waste_items | total + wi.quantity)) as daily_waste_total,
       AVG(ws.total_weight) as avg_shipment_weight
ORDER BY shipment_date DESC
```

### Facility Emissions Queries

#### Basic Facility Emissions
```cypher
MATCH (f:Facility)
OPTIONAL MATCH (f)<-[:BILLED_TO]-(b:UtilityBill)
OPTIONAL MATCH (b)-[:RESULTED_IN]->(e:Emission)
RETURN f, b, e
ORDER BY b.billing_period_end DESC
```

#### Emissions Aggregation
```cypher
MATCH (f:Facility)
MATCH (f)<-[:BILLED_TO]-(b:UtilityBill)-[:RESULTED_IN]->(e:Emission)
RETURN f.name as facility, 
       SUM(e.amount) as total_emissions,
       e.unit as unit,
       COUNT(b) as bill_count
```

---

## Response Formats and Data Structures

### Standard API Response Structure

```json
{
  "status": "success|failed",
  "message": "Descriptive message about the operation",
  "data": {
    "query_type": "utility_consumption",
    "facility_filter": null,
    "date_range": null,
    "report_data": { /* Full report structure */ },
    "file_path": "/path/to/generated/report.json"
  },
  "metadata": {
    "total_queries": 2,
    "successful_queries": 1,
    "total_records": 150,
    "processing_status": "completed",
    "generated_at": "2025-08-18T22:25:30.242553"
  },
  "processing_time": 2.45,
  "errors": []
}
```

### Report Data Structure

#### Metadata Section
```json
{
  "metadata": {
    "title": "EHS Data Extraction Report - utility_consumption",
    "generated_at": "2025-08-18T22:25:30.242553",
    "query_type": "utility_consumption",
    "parameters": {
      "start_date": "2023-01-01",
      "end_date": "2023-12-31"
    },
    "output_format": "json"
  }
}
```

#### Summary Section
```json
{
  "summary": {
    "total_queries": 2,
    "successful_queries": 1,
    "failed_queries": 1,
    "total_records": 150,
    "graph_objects": [
      {
        "nodes": [
          {"labels": ["Document", "Utilitybill"], "count": 1},
          {"labels": ["UtilityBill"], "count": 1},
          {"labels": ["Facility"], "count": 2}
        ],
        "relationships": [
          {"type": "EXTRACTED_TO", "count": 1},
          {"type": "BILLED_TO", "count": 2}
        ],
        "total_nodes": 6,
        "total_relationships": 8
      }
    ]
  }
}
```

#### Query Results Section
```json
{
  "query_results": [
    {
      "query": "MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)...",
      "parameters": {"start_date": "2023-01-01"},
      "results": [
        {
          "b": {
            "id": 123,
            "labels": ["UtilityBill"],
            "properties": {
              "total_kwh": 130000,
              "total_cost": 15600.50,
              "billing_period_start": "2023-01-01",
              "billing_period_end": "2023-01-31"
            }
          }
        }
      ],
      "record_count": 1,
      "status": "success"
    }
  ]
}
```

### Utility Bill Data Structure

```json
{
  "b": {
    "id": 123,
    "labels": ["UtilityBill"],
    "properties": {
      "account_number": "1234567890",
      "total_kwh": 130000,
      "total_cost": 15600.50,
      "demand_charge": 2500.00,
      "energy_charge": 13100.50,
      "billing_period_start": "2023-01-01",
      "billing_period_end": "2023-01-31",
      "service_address": "123 Main St, City, State"
    }
  }
}
```

### Water Bill Data Structure

```json
{
  "w": {
    "id": 456,
    "labels": ["WaterBill"],
    "properties": {
      "account_number": "W9876543210",
      "total_gallons": 250000,
      "total_cost": 890.75,
      "service_charge": 45.00,
      "usage_charge": 845.75,
      "billing_period_start": "2023-01-01",
      "billing_period_end": "2023-01-31"
    }
  },
  "meters": [
    {
      "id": 789,
      "labels": ["Meter"],
      "properties": {
        "meter_id": "M123456",
        "previous_reading": 1000000,
        "current_reading": 1250000,
        "usage": 250000
      }
    }
  ]
}
```

### Waste Manifest Data Structure

```json
{
  "wm": {
    "id": 789,
    "labels": ["WasteManifest"],
    "properties": {
      "manifest_number": "WM-2023-001",
      "issue_date": "2023-01-15",
      "status": "completed"
    }
  },
  "waste_items": [
    {
      "id": 1001,
      "labels": ["WasteItem"],
      "properties": {
        "waste_code": "D001",
        "description": "Ignitable waste",
        "quantity": 15.0,
        "unit": "tons",
        "hazardous": true,
        "disposal_method": "Incineration"
      }
    }
  ],
  "generator": {
    "id": 1002,
    "labels": ["WasteGenerator"],
    "properties": {
      "name": "Industrial Facility Inc.",
      "epa_id": "NYD123456789",
      "address": "789 Industrial Blvd, City, State"
    }
  }
}
```

### Error Response Structure

```json
{
  "status": "failed",
  "message": "Extraction failed: Connection timeout",
  "data": null,
  "metadata": {
    "total_queries": 0,
    "successful_queries": 0,
    "total_records": 0,
    "processing_status": "failed",
    "generated_at": "2025-08-18T22:30:15.123456"
  },
  "processing_time": 30.0,
  "errors": [
    "Neo4j connection timeout after 30 seconds",
    "Query execution failed: MATCH syntax error"
  ]
}
```

---

## Configuration and Environment Variables

### Required Environment Variables

#### Neo4j Database Configuration
```bash
# Neo4j connection settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# For AuraDB (cloud Neo4j)
# NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
```

#### LLM Configuration
```bash
# OpenAI configuration (default)
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4

# Or Anthropic configuration
ANTHROPIC_API_KEY=your_anthropic_api_key
LLM_MODEL=claude-3-sonnet-20240229
```

#### API Server Configuration
```bash
# Server settings
PORT=8005
HOST=0.0.0.0

# Enable development mode
RELOAD=true
RELOAD_DIRS=["src"]
```

### Optional Environment Variables

```bash
# Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Output directory for reports
REPORTS_OUTPUT_DIR=./reports

# API documentation settings
DOCS_URL=/api/docs
REDOC_URL=/api/redoc

# CORS settings (for production)
ALLOWED_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
ALLOWED_METHODS=["GET", "POST"]
ALLOWED_HEADERS=["*"]
```

### Environment File (.env)

Create a `.env` file in the project root:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# LLM Configuration
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4

# API Configuration
PORT=8005
HOST=0.0.0.0

# Development Settings
RELOAD=true
LOG_LEVEL=INFO
```

### Docker Environment

For Docker deployments, use environment variables or Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  ehs-api:
    build: .
    ports:
      - "8005:8005"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=password
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_MODEL=gpt-4
    depends_on:
      - neo4j
```

### Configuration Validation

The API validates required environment variables at startup:

```python
# Startup validation
required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    logger.error(f"Missing required environment variables: {missing_vars}")
    raise RuntimeError(f"Missing required environment variables: {missing_vars}")
```

---

## Error Handling and Troubleshooting

### Common Error Types

#### 1. Neo4j Connection Errors

**Error**: `Neo4j connection timeout`
```json
{
  "status": "failed",
  "message": "Workflow initialization error: Failed to connect to Neo4j",
  "errors": ["Neo4j connection timeout after 30 seconds"]
}
```

**Solutions**:
- Verify Neo4j server is running
- Check `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- Test connection: `curl http://localhost:7474`
- Check firewall settings for port 7687

#### 2. Query Execution Errors

**Error**: `Cypher syntax error`
```json
{
  "query_results": [
    {
      "query": "MATCH (invalid syntax...",
      "error": "Invalid input 'invalid': expected whitespace...",
      "status": "failed"
    }
  ]
}
```

**Solutions**:
- Review Cypher query syntax
- Check node labels and relationship types
- Verify parameter substitution
- Test queries in Neo4j Browser

#### 3. Data Not Found Errors

**Error**: `No records found`
```json
{
  "metadata": {
    "total_records": 0,
    "successful_queries": 1,
    "failed_queries": 0
  }
}
```

**Solutions**:
- Verify data has been ingested
- Check label case sensitivity (`Utilitybill` vs `UtilityBill`)
- Adjust date range filters
- Review relationship patterns

#### 4. LLM Analysis Errors

**Error**: `LLM analysis failed`
```json
{
  "analysis_results": {
    "error": "Analysis failed",
    "summary": "Could not parse structured response"
  }
}
```

**Solutions**:
- Verify API keys (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- Check token limits and data size
- Review prompt structure
- Consider using smaller data samples

### Troubleshooting Workflow

#### Step 1: Health Check
```bash
curl -X GET http://localhost:8005/health
```
Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-18T22:30:00.000Z",
  "neo4j_connection": true,
  "version": "1.0.0"
}
```

#### Step 2: Neo4j Connectivity Test
```bash
# Test Neo4j directly
curl -u neo4j:password http://localhost:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"MATCH (n) RETURN count(n) as node_count"}]}'
```

#### Step 3: API Endpoint Test
```bash
# Test simple extraction
curl -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -d '{"output_format": "json"}' | jq '.'
```

#### Step 4: Data Verification
```cypher
// Check available node types
MATCH (n) 
RETURN DISTINCT labels(n) as node_types, count(n) as count
ORDER BY count DESC

// Check utility bill data
MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)
WHERE 'Utilitybill' IN labels(d) OR 'UtilityBill' IN labels(d)
RETURN count(b) as utility_bill_count

// Check case sensitivity issues
MATCH (d:Document)
WHERE any(label IN labels(d) WHERE toLower(label) CONTAINS 'utility')
RETURN labels(d) as document_labels, count(d) as count
```

### Error Recovery Strategies

#### 1. Automatic Retry Logic
The workflow includes built-in retry mechanisms:
```python
workflow.add_conditional_edges(
    "handle_error",
    lambda state: "fail" if len(state["errors"]) > 3 else "retry",
    {
        "fail": END,
        "retry": "prepare_queries"
    }
)
```

#### 2. Graceful Degradation
- Continue processing successful queries when some fail
- Provide partial results with error information
- Generate reports even with incomplete data

#### 3. State Recovery
- Maintain processing state for resume capability
- Log all state transitions for debugging
- Enable workflow restart from specific points

### Debugging Tools

#### 1. Verbose Logging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python3 -m uvicorn backend.src.ehs_extraction_api:app --log-level debug
```

#### 2. Query Profiling
```cypher
// Profile query performance
PROFILE MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)
RETURN d, b
```

#### 3. State Inspection
```python
# Inspect workflow state
import json
print(json.dumps(workflow_state, indent=2))
```

### Performance Optimization

#### 1. Query Optimization
- Add indexes on frequently queried properties
- Use LIMIT clauses for large datasets
- Optimize relationship traversals

#### 2. Connection Pooling
```python
# Configure connection pool
driver = GraphDatabase.driver(
    uri,
    auth=(username, password),
    max_connection_lifetime=3600,
    max_connection_pool_size=50
)
```

#### 3. Batch Processing
- Process multiple documents in batches
- Use UNWIND for bulk operations
- Implement parallel query execution

---

## Sample curl Commands and Responses

### Health Check

#### Request
```bash
curl -X GET http://localhost:8005/health
```

#### Response
```json
{
  "status": "healthy",
  "timestamp": "2025-08-18T22:30:00.123456",
  "neo4j_connection": true,
  "version": "1.0.0"
}
```

### Electrical Consumption Extraction

#### Request
```bash
curl -X POST http://localhost:8005/api/v1/extract/electrical-consumption \
  -H "Content-Type: application/json" \
  -d '{
    "facility_filter": {
      "facility_name": "Main Campus"
    },
    "date_range": {
      "start_date": "2023-01-01",
      "end_date": "2023-12-31"
    },
    "output_format": "json",
    "include_emissions": true,
    "include_cost_analysis": true
  }'
```

#### Response
```json
{
  "status": "success",
  "message": "Electrical consumption data extracted successfully",
  "data": {
    "query_type": "utility_consumption",
    "facility_filter": {
      "facility_name": "Main Campus"
    },
    "date_range": {
      "start_date": "2023-01-01",
      "end_date": "2023-12-31"
    },
    "include_emissions": true,
    "include_cost_analysis": true,
    "report_data": {
      "metadata": {
        "title": "EHS Data Extraction Report - utility_consumption",
        "generated_at": "2025-08-18T22:25:30.242553",
        "query_type": "utility_consumption",
        "parameters": {
          "start_date": "2023-01-01",
          "end_date": "2023-12-31",
          "facility_name": "Main Campus"
        },
        "output_format": "json"
      },
      "summary": {
        "total_queries": 2,
        "successful_queries": 2,
        "failed_queries": 0,
        "total_records": 12,
        "graph_objects": [
          {
            "nodes": [
              {"labels": ["Document", "Utilitybill"], "count": 1},
              {"labels": ["UtilityBill"], "count": 12},
              {"labels": ["Facility"], "count": 1}
            ],
            "total_nodes": 14,
            "total_relationships": 24
          }
        ]
      },
      "query_results": [
        {
          "query": "MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)...",
          "results": [
            {
              "d": {
                "id": 1,
                "labels": ["Document", "Utilitybill"],
                "properties": {
                  "filename": "electric_bill_2023_01.pdf",
                  "status": "processed"
                }
              },
              "b": {
                "id": 101,
                "labels": ["UtilityBill"],
                "properties": {
                  "account_number": "1234567890",
                  "total_kwh": 130000,
                  "total_cost": 15600.50,
                  "billing_period_start": "2023-01-01",
                  "billing_period_end": "2023-01-31"
                }
              }
            }
          ],
          "record_count": 12,
          "status": "success"
        }
      ]
    },
    "file_path": "/path/to/reports/ehs_report_utility_consumption_20250818_222530.json"
  },
  "metadata": {
    "total_queries": 2,
    "successful_queries": 2,
    "total_records": 12,
    "processing_status": "completed",
    "generated_at": "2025-08-18T22:25:30.242553"
  },
  "processing_time": 3.45,
  "errors": []
}
```

### Water Consumption Extraction

#### Request
```bash
curl -X POST http://localhost:8005/api/v1/extract/water-consumption \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "2023-06-01",
      "end_date": "2023-06-30"
    },
    "output_format": "json",
    "include_meter_details": true,
    "include_emissions": true
  }'
```

#### Response
```json
{
  "status": "success",
  "message": "Water consumption data extracted successfully",
  "data": {
    "query_type": "water_consumption",
    "date_range": {
      "start_date": "2023-06-01",
      "end_date": "2023-06-30"
    },
    "include_meter_details": true,
    "include_emissions": true,
    "report_data": {
      "summary": {
        "total_queries": 4,
        "successful_queries": 3,
        "total_records": 8
      },
      "query_results": [
        {
          "results": [
            {
              "w": {
                "id": 201,
                "labels": ["WaterBill"],
                "properties": {
                  "account_number": "W9876543210",
                  "total_gallons": 250000,
                  "total_cost": 890.75,
                  "billing_period_start": "2023-06-01",
                  "billing_period_end": "2023-06-30"
                }
              },
              "meters": [
                {
                  "id": 301,
                  "labels": ["Meter"],
                  "properties": {
                    "meter_id": "M123456",
                    "previous_reading": 1000000,
                    "current_reading": 1250000,
                    "usage": 250000
                  }
                }
              ]
            }
          ],
          "record_count": 1,
          "status": "success"
        }
      ]
    }
  },
  "processing_time": 2.1
}
```

### Waste Generation Extraction

#### Request
```bash
curl -X POST http://localhost:8005/api/v1/extract/waste-generation \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "2023-03-01",
      "end_date": "2023-03-31"
    },
    "output_format": "json",
    "include_disposal_details": true,
    "include_transport_details": true,
    "include_emissions": true,
    "hazardous_only": false
  }'
```

#### Response
```json
{
  "status": "success",
  "message": "Waste generation data extracted successfully",
  "data": {
    "query_type": "waste_generation",
    "date_range": {
      "start_date": "2023-03-01",
      "end_date": "2023-03-31"
    },
    "hazardous_only": false,
    "report_data": {
      "summary": {
        "total_queries": 8,
        "successful_queries": 7,
        "total_records": 45
      },
      "query_results": [
        {
          "results": [
            {
              "wm": {
                "id": 401,
                "labels": ["WasteManifest"],
                "properties": {
                  "manifest_number": "WM-2023-001",
                  "issue_date": "2023-03-15",
                  "status": "completed"
                }
              },
              "ws": {
                "id": 501,
                "labels": ["WasteShipment"],
                "properties": {
                  "shipment_date": "2023-03-15",
                  "total_weight": 15.0,
                  "transport_method": "truck"
                }
              },
              "generator": {
                "id": 601,
                "labels": ["WasteGenerator"],
                "properties": {
                  "name": "Industrial Facility Inc.",
                  "epa_id": "NYD123456789"
                }
              },
              "transporter": {
                "id": 701,
                "labels": ["Transporter"],
                "properties": {
                  "name": "Waste Transport LLC",
                  "epa_id": "NYT987654321"
                }
              },
              "disposal_facility": {
                "id": 801,
                "labels": ["DisposalFacility"],
                "properties": {
                  "name": "Secure Disposal Inc.",
                  "epa_id": "NYF555666777",
                  "state": "NY"
                }
              },
              "waste_items": [
                {
                  "id": 901,
                  "labels": ["WasteItem"],
                  "properties": {
                    "waste_code": "D001",
                    "description": "Ignitable waste",
                    "quantity": 15.0,
                    "unit": "tons",
                    "hazardous": true,
                    "disposal_method": "Incineration"
                  }
                }
              ]
            }
          ],
          "record_count": 5,
          "status": "success"
        }
      ]
    }
  },
  "processing_time": 4.2
}
```

### Custom Query Extraction

#### Request
```bash
curl -X POST "http://localhost:8005/api/v1/extract/custom?query_type=facility_emissions&output_format=json" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_filter": {
      "facility_id": "FAC-001"
    },
    "date_range": {
      "start_date": "2023-01-01",
      "end_date": "2023-12-31"
    }
  }'
```

#### Response
```json
{
  "status": "success",
  "message": "Custom data extraction completed",
  "data": {
    "query_type": "facility_emissions",
    "facility_filter": {
      "facility_id": "FAC-001"
    },
    "date_range": {
      "start_date": "2023-01-01",
      "end_date": "2023-12-31"
    },
    "custom_queries": null,
    "report_data": {
      "summary": {
        "total_queries": 2,
        "successful_queries": 2,
        "total_records": 18
      }
    }
  },
  "processing_time": 1.8
}
```

### Query Types Information

#### Request
```bash
curl -X GET http://localhost:8005/api/v1/query-types
```

#### Response
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

#### Missing Environment Variables
```json
{
  "status": "Failed",
  "message": "Missing required Neo4j connection configuration",
  "data": null,
  "metadata": null,
  "processing_time": null,
  "errors": ["Missing required environment variables: ['NEO4J_PASSWORD']"]
}
```

#### Neo4j Connection Failure
```json
{
  "status": "failed",
  "message": "Extraction failed: Neo4j connection timeout",
  "data": null,
  "metadata": {
    "total_queries": 0,
    "successful_queries": 0,
    "total_records": 0,
    "processing_status": "failed",
    "generated_at": "2025-08-18T22:35:00.000000"
  },
  "processing_time": 30.0,
  "errors": [
    "Query execution error: Failed to connect to server at bolt://localhost:7687"
  ]
}
```

#### Invalid Query Type
```json
{
  "status": "Failed",
  "message": "Invalid query type: invalid_type",
  "data": null,
  "metadata": null,
  "processing_time": null,
  "errors": ["Invalid query type: invalid_type"]
}
```

---

## Appendix

### Useful Commands Summary

```bash
# Start API server
export PYTHONPATH="/path/to/backend/src:$PYTHONPATH"
python3 -m uvicorn backend.src.ehs_extraction_api:app --host 0.0.0.0 --port 8005

# Run test scripts
./test_extraction_fix.sh
./full_ingestion_extraction_test.sh

# Check server status
curl http://localhost:8005/health

# View API documentation
open http://localhost:8005/api/docs
```

### Key File Locations

- **Main workflow**: `backend/src/workflows/extraction_workflow.py`
- **API service**: `backend/src/ehs_extraction_api.py`
- **Reports**: `backend/src/reports/`
- **Test scripts**: `test_extraction_fix.sh`, `full_ingestion_extraction_test.sh`
- **Virtual environment**: `.venv/`

### Important URLs

- **API Base**: `http://localhost:8005`
- **Health Check**: `http://localhost:8005/health`
- **API Documentation**: `http://localhost:8005/api/docs`
- **ReDoc Documentation**: `http://localhost:8005/api/redoc`

---

This documentation provides comprehensive coverage of the EHS Data Extraction Workflow system, enabling users to understand, deploy, test, and troubleshoot the system effectively.