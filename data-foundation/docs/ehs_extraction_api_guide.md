# EHS Extraction API Documentation

## Overview

The EHS (Environmental, Health, Safety) Extraction API provides a comprehensive REST interface for extracting and analyzing EHS data from a Neo4j graph database. The API leverages the DataExtractionWorkflow, which uses LangGraph orchestration and LLM-powered analysis to query structured EHS data and generate detailed reports.

### Key Features

- **Multi-format Output**: JSON and text report formats
- **LLM-powered Analysis**: Intelligent data insights using OpenAI or Anthropic models
- **Graph Database Integration**: Direct Neo4j queries with structured data extraction
- **Flexible Filtering**: Date range and facility-based filtering
- **Real-time Processing**: Synchronous data extraction with performance metrics
- **Comprehensive Reporting**: Detailed reports with metadata, analysis, and error tracking

### Architecture

```
FastAPI REST API → DataExtractionWorkflow → Neo4j Database
      ↓                        ↓                    ↓
  Pydantic Models       LangGraph Workflow    Cypher Queries
      ↓                        ↓                    ↓
   Validation           LLM Analysis          Data Extraction
      ↓                        ↓                    ↓
   Responses              Report Generation    Structured Output
```

## Getting Started

### Prerequisites

- Python 3.8+
- Neo4j database with EHS data
- OpenAI or Anthropic API key for LLM analysis
- FastAPI and related dependencies

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd data-foundation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Set up environment variables** (see [Configuration](#configuration) section)

4. **Run the API server**:
   ```bash
   cd backend/src
   python ehs_extraction_api.py
   ```

The API will be available at `http://localhost:8001` with interactive documentation at `http://localhost:8001/api/docs`.

### Quick Start Example

```bash
# Health check
curl -X GET "http://localhost:8001/health"

# Extract electrical consumption data
curl -X POST "http://localhost:8001/api/v1/extract/electrical-consumption" \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    },
    "include_emissions": true,
    "output_format": "json"
  }'
```

## Configuration

### Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

#### Required Variables

```bash
# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# LLM Configuration (choose one)
OPENAI_API_KEY=your_openai_key
# OR
ANTHROPIC_API_KEY=your_anthropic_key

# LLM Model Selection
LLM_MODEL=gpt-4                    # Options: gpt-4, gpt-3.5-turbo, claude-3-sonnet
```

#### Optional Variables

```bash
# API Configuration
PORT=8001                          # API server port

# Logging
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_PROJECT=ehs-extraction

# Performance Tuning
MAX_TOKEN_CHUNK_SIZE=2000
NUMBER_OF_CHUNKS_TO_COMBINE=6
```

### Model Configuration

The API supports multiple LLM providers. Configure them using the `LLM_MODEL_CONFIG_*` format:

```bash
# OpenAI Models
LLM_MODEL_CONFIG_openai_gpt_4o="gpt-4o-2024-11-20,openai_api_key"
LLM_MODEL_CONFIG_openai_gpt_4o_mini="gpt-4o-mini-2024-07-18,openai_api_key"

# Anthropic Models
LLM_MODEL_CONFIG_anthropic_claude_4_sonnet="claude-sonnet-4-20250514,anthropic_api_key"

# Google Models
LLM_MODEL_CONFIG_gemini_2.0_flash="gemini-2.0-flash-001"
```

## API Endpoints Reference

### Health Check

**GET** `/health`

Check the API and database connectivity status.

#### Response Schema

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "neo4j_connection": true,
  "version": "1.0.0"
}
```

#### Example Request

```bash
curl -X GET "http://localhost:8001/health"
```

### Extract Electrical Consumption Data

**POST** `/api/v1/extract/electrical-consumption`

Extract electrical consumption data from utility bills, including usage, costs, and emissions calculations.

#### Request Schema

```json
{
  "facility_filter": {
    "facility_id": "string",        // Optional: Specific facility ID
    "facility_name": "string"       // Optional: Facility name pattern
  },
  "date_range": {
    "start_date": "2024-01-01",     // Optional: Start date (YYYY-MM-DD)
    "end_date": "2024-12-31"        // Optional: End date (YYYY-MM-DD)
  },
  "output_format": "json",          // Output format: "json" or "txt"
  "include_emissions": true,        // Include emissions calculations
  "include_cost_analysis": true     // Include cost breakdown analysis
}
```

#### Response Schema

```json
{
  "status": "success",
  "message": "Electrical consumption data extracted successfully",
  "data": {
    "query_type": "utility_consumption",
    "facility_filter": {...},
    "date_range": {...},
    "include_emissions": true,
    "include_cost_analysis": true,
    "report_data": {
      "metadata": {...},
      "summary": {...},
      "queries_executed": [...],
      "query_results": [...],
      "analysis": {...}
    },
    "file_path": "/path/to/report/file"
  },
  "metadata": {
    "total_queries": 2,
    "successful_queries": 2,
    "total_records": 156,
    "processing_status": "completed",
    "generated_at": "2024-01-15T10:30:00.000Z"
  },
  "processing_time": 3.45,
  "errors": []
}
```

#### Example Request

```bash
curl -X POST "http://localhost:8001/api/v1/extract/electrical-consumption" \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "2024-01-01",
      "end_date": "2024-03-31"
    },
    "include_emissions": true,
    "include_cost_analysis": true,
    "output_format": "json"
  }'
```

### Extract Water Consumption Data

**POST** `/api/v1/extract/water-consumption`

Extract water consumption data from water bills, including usage, costs, meter information, and emissions.

#### Request Schema

```json
{
  "facility_filter": {
    "facility_id": "string",        // Optional: Specific facility ID
    "facility_name": "string"       // Optional: Facility name pattern
  },
  "date_range": {
    "start_date": "2024-01-01",     // Optional: Start date (YYYY-MM-DD)
    "end_date": "2024-12-31"        // Optional: End date (YYYY-MM-DD)
  },
  "output_format": "json",          // Output format: "json" or "txt"
  "include_meter_details": true,    // Include water meter information
  "include_emissions": true         // Include emissions calculations
}
```

#### Example Request

```bash
curl -X POST "http://localhost:8001/api/v1/extract/water-consumption" \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "2024-06-01",
      "end_date": "2024-06-30"
    },
    "include_meter_details": true,
    "include_emissions": true
  }'
```

### Extract Waste Generation Data

**POST** `/api/v1/extract/waste-generation`

Extract waste generation data from waste manifests, including quantities, disposal information, transporter details, and emissions.

#### Request Schema

```json
{
  "facility_filter": {
    "facility_id": "string",        // Optional: Specific facility ID
    "facility_name": "string"       // Optional: Facility name pattern
  },
  "date_range": {
    "start_date": "2024-01-01",     // Optional: Start date (YYYY-MM-DD)
    "end_date": "2024-12-31"        // Optional: End date (YYYY-MM-DD)
  },
  "output_format": "json",          // Output format: "json" or "txt"
  "include_disposal_details": true, // Include disposal facility details
  "include_transport_details": true,// Include transporter information
  "include_emissions": true,        // Include emissions calculations
  "hazardous_only": false          // Filter for hazardous waste only
}
```

#### Example Request

```bash
curl -X POST "http://localhost:8001/api/v1/extract/waste-generation" \
  -H "Content-Type: application/json" \
  -d '{
    "date_range": {
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    },
    "include_disposal_details": true,
    "include_transport_details": true,
    "include_emissions": true,
    "hazardous_only": false
  }'
```

### Custom Data Extraction

**POST** `/api/v1/extract/custom`

Execute custom data extraction using predefined query types or custom Cypher queries.

#### Request Parameters

- `query_type` (string): Query type from available options
- `facility_filter` (object, optional): Facility filtering criteria
- `date_range` (object, optional): Date range filtering
- `output_format` (string): Output format ("json" or "txt")
- `custom_queries` (array, optional): Custom Cypher queries

#### Example Request

```bash
curl -X POST "http://localhost:8001/api/v1/extract/custom" \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "facility_emissions",
    "date_range": {
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    },
    "output_format": "json"
  }'
```

### Get Available Query Types

**GET** `/api/v1/query-types`

Retrieve a list of all available query types for extraction.

#### Response Schema

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
    // ... more query types
  ]
}
```

#### Example Request

```bash
curl -X GET "http://localhost:8001/api/v1/query-types"
```

## Data Models

### Request Models

#### FacilityFilter

```python
class FacilityFilter(BaseModel):
    facility_id: Optional[str] = None      # Specific facility ID
    facility_name: Optional[str] = None    # Facility name pattern
```

#### DateRangeFilter

```python
class DateRangeFilter(BaseModel):
    start_date: Optional[date] = None      # Start date (YYYY-MM-DD)
    end_date: Optional[date] = None        # End date (YYYY-MM-DD)
```

**Validation Rules:**
- `end_date` must be after `start_date` if both are provided
- Dates must be in `YYYY-MM-DD` format

#### BaseExtractionRequest

```python
class BaseExtractionRequest(BaseModel):
    facility_filter: Optional[FacilityFilter] = None
    date_range: Optional[DateRangeFilter] = None
    output_format: str = "json"            # "json" or "txt"
```

#### ElectricalConsumptionRequest

```python
class ElectricalConsumptionRequest(BaseExtractionRequest):
    include_emissions: bool = True         # Include emissions data
    include_cost_analysis: bool = True     # Include cost analysis
```

#### WaterConsumptionRequest

```python
class WaterConsumptionRequest(BaseExtractionRequest):
    include_meter_details: bool = True     # Include meter information
    include_emissions: bool = True         # Include emissions data
```

#### WasteGenerationRequest

```python
class WasteGenerationRequest(BaseExtractionRequest):
    include_disposal_details: bool = True  # Include disposal facility details
    include_transport_details: bool = True # Include transporter details
    include_emissions: bool = True         # Include emissions data
    hazardous_only: bool = False          # Filter for hazardous waste only
```

### Response Models

#### ExtractionResponse

```python
class ExtractionResponse(BaseModel):
    status: str                           # "success" or "failed"
    message: str                          # Response message
    data: Optional[Dict[str, Any]] = None # Extracted data
    metadata: Optional[Dict[str, Any]] = None # Query metadata
    processing_time: Optional[float] = None # Processing time in seconds
    errors: Optional[List[str]] = None    # Any errors encountered
```

#### HealthResponse

```python
class HealthResponse(BaseModel):
    status: str = "healthy"               # Health status
    timestamp: str                        # Current timestamp
    neo4j_connection: bool               # Neo4j connectivity status
    version: str = "1.0.0"              # API version
```

### Data Structures

#### Report Data Structure

```json
{
  "metadata": {
    "title": "EHS Data Extraction Report - Utility Consumption",
    "generated_at": "2024-01-15T10:30:00.000Z",
    "query_type": "utility_consumption",
    "parameters": {
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    },
    "output_format": "json"
  },
  "summary": {
    "total_queries": 2,
    "successful_queries": 2,
    "failed_queries": 0,
    "total_records": 156,
    "graph_objects": [
      {
        "nodes": [
          {"labels": ["UtilityBill"], "count": 156},
          {"labels": ["Facility"], "count": 12},
          {"labels": ["Emission"], "count": 156}
        ],
        "relationships": [
          {"type": "BILLED_TO", "count": 156},
          {"type": "RESULTED_IN", "count": 156}
        ],
        "total_nodes": 324,
        "total_relationships": 312
      }
    ]
  },
  "queries_executed": [
    {
      "query": "MATCH (d:Document:UtilityBill)-[:EXTRACTED_TO]->(b:UtilityBill) RETURN d, b ORDER BY b.billing_period_end DESC",
      "parameters": {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
      }
    }
  ],
  "query_results": [
    {
      "query": "...",
      "parameters": {...},
      "results": [...],
      "record_count": 156,
      "status": "success"
    }
  ],
  "analysis": {
    "key_findings": [
      "Total electrical consumption: 2,450,000 kWh",
      "Average monthly cost: $18,750",
      "Peak consumption period: July-August 2024"
    ],
    "patterns": [
      "Consumption increases by 35% during summer months",
      "Weekend usage 20% lower than weekday usage"
    ],
    "recommendations": [
      "Consider energy efficiency upgrades",
      "Implement demand response programs"
    ]
  }
}
```

### Query Types Enum

```python
class QueryType(str, Enum):
    FACILITY_EMISSIONS = "facility_emissions"      # Facility-level emissions
    UTILITY_CONSUMPTION = "utility_consumption"    # Electrical consumption
    WATER_CONSUMPTION = "water_consumption"        # Water usage data
    WASTE_GENERATION = "waste_generation"          # Waste manifest data
    COMPLIANCE_STATUS = "compliance_status"        # Permit compliance
    TREND_ANALYSIS = "trend_analysis"             # Trend analysis
    CUSTOM = "custom"                             # Custom queries
```

## Error Handling

### HTTP Status Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| `200` | Success | Request processed successfully |
| `400` | Bad Request | Invalid request parameters or date ranges |
| `422` | Validation Error | Pydantic model validation failed |
| `500` | Internal Server Error | Database connection issues, LLM API errors |

### Error Response Format

```json
{
  "status": "Failed",
  "message": "Error description",
  "error": "Detailed error information"
}
```

### Common Errors

#### Database Connection Error

```json
{
  "detail": "Missing required Neo4j connection configuration"
}
```

**Solution**: Verify Neo4j environment variables are properly set.

#### Date Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "date_range", "end_date"],
      "msg": "end_date must be after start_date",
      "type": "value_error"
    }
  ]
}
```

**Solution**: Ensure `end_date` is later than `start_date`.

#### Query Execution Error

```json
{
  "status": "success",
  "message": "Extraction completed with errors",
  "data": {...},
  "errors": [
    "Query 2 failed: Invalid syntax at line 1"
  ]
}
```

**Solution**: Check query syntax or contact support for custom queries.

#### LLM API Error

```json
{
  "detail": "Analysis failed: OpenAI API rate limit exceeded"
}
```

**Solution**: Check API key quotas or implement retry logic.

## Usage Examples

### Example 1: Extract All Electrical Data for a Facility

```python
import requests
import json

# API endpoint
url = "http://localhost:8001/api/v1/extract/electrical-consumption"

# Request payload
payload = {
    "facility_filter": {
        "facility_name": "Manufacturing Plant A"
    },
    "date_range": {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    },
    "include_emissions": True,
    "include_cost_analysis": True,
    "output_format": "json"
}

# Make request
response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print(f"Extracted {result['metadata']['total_records']} records")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    
    # Access specific data
    total_consumption = 0
    for query_result in result['data']['report_data']['query_results']:
        if query_result['status'] == 'success':
            for record in query_result['results']:
                if 'total_kwh' in record:
                    total_consumption += record['total_kwh']
    
    print(f"Total consumption: {total_consumption:,} kWh")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Example 2: Get Water Consumption for a Date Range

```python
import requests
from datetime import datetime, timedelta

# Calculate date range (last 30 days)
end_date = datetime.now().date()
start_date = end_date - timedelta(days=30)

url = "http://localhost:8001/api/v1/extract/water-consumption"
payload = {
    "date_range": {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    },
    "include_meter_details": True,
    "include_emissions": True,
    "output_format": "json"
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    
    # Extract water usage summary
    analysis = result['data']['report_data'].get('analysis', {})
    if 'key_findings' in analysis:
        print("Water Usage Summary:")
        for finding in analysis['key_findings']:
            print(f"  • {finding}")
else:
    print(f"Request failed: {response.text}")
```

### Example 3: Retrieve Waste Generation with Emissions Data

```python
import requests
import pandas as pd

url = "http://localhost:8001/api/v1/extract/waste-generation"
payload = {
    "date_range": {
        "start_date": "2024-01-01",
        "end_date": "2024-03-31"
    },
    "include_disposal_details": True,
    "include_transport_details": True,
    "include_emissions": True,
    "hazardous_only": False,  # Include all waste types
    "output_format": "json"
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    
    # Extract waste data for analysis
    waste_data = []
    for query_result in result['data']['report_data']['query_results']:
        if query_result['status'] == 'success':
            for record in query_result['results']:
                # Extract waste items
                if 'waste_items' in record:
                    for item in record['waste_items']:
                        if isinstance(item, dict) and 'properties' in item:
                            props = item['properties']
                            waste_data.append({
                                'quantity': props.get('quantity', 0),
                                'unit': props.get('unit', ''),
                                'description': props.get('description', ''),
                                'hazardous': props.get('hazardous', False),
                                'disposal_method': props.get('disposal_method', '')
                            })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(waste_data)
    
    print(f"Total waste records: {len(df)}")
    print(f"Hazardous waste records: {len(df[df['hazardous'] == True])}")
    print(f"Total quantity: {df['quantity'].sum():.2f}")
    
    # Group by disposal method
    disposal_summary = df.groupby('disposal_method')['quantity'].sum()
    print("\nDisposal Methods:")
    for method, quantity in disposal_summary.items():
        print(f"  {method}: {quantity:.2f}")
        
else:
    print(f"Request failed: {response.text}")
```

### Example 4: Health Check and System Status

```python
import requests

def check_api_health():
    """Check API health and return status information."""
    try:
        response = requests.get("http://localhost:8001/health")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"API Status: {health_data['status']}")
            print(f"Neo4j Connection: {'✓' if health_data['neo4j_connection'] else '✗'}")
            print(f"Version: {health_data['version']}")
            print(f"Timestamp: {health_data['timestamp']}")
            return health_data['status'] == 'healthy'
        else:
            print(f"Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Cannot connect to API: {e}")
        return False

def get_available_query_types():
    """Get list of available query types."""
    try:
        response = requests.get("http://localhost:8001/api/v1/query-types")
        
        if response.status_code == 200:
            types_data = response.json()
            print("Available Query Types:")
            for query_type in types_data['query_types']:
                print(f"  • {query_type['name']} ({query_type['value']})")
                print(f"    {query_type['description']}")
            return types_data['query_types']
        else:
            print(f"Failed to get query types: {response.status_code}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Cannot connect to API: {e}")
        return []

# Usage
if check_api_health():
    query_types = get_available_query_types()
```

## Integration Guide

### Integrating with Existing Workflows

#### 1. ETL Pipeline Integration

```python
class EHSDataExtractor:
    def __init__(self, api_base_url="http://localhost:8001"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
    
    def extract_monthly_data(self, year, month):
        """Extract all EHS data for a specific month."""
        start_date = f"{year}-{month:02d}-01"
        
        # Calculate last day of month
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"
        
        # Extract all data types
        results = {}
        
        # Electrical data
        results['electrical'] = self._extract_data(
            'electrical-consumption',
            start_date, end_date
        )
        
        # Water data
        results['water'] = self._extract_data(
            'water-consumption',
            start_date, end_date
        )
        
        # Waste data
        results['waste'] = self._extract_data(
            'waste-generation',
            start_date, end_date
        )
        
        return results
    
    def _extract_data(self, endpoint, start_date, end_date):
        """Generic data extraction method."""
        url = f"{self.api_base_url}/api/v1/extract/{endpoint}"
        payload = {
            "date_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "output_format": "json"
        }
        
        response = self.session.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}

# Usage in ETL pipeline
extractor = EHSDataExtractor()
monthly_data = extractor.extract_monthly_data(2024, 3)
```

#### 2. Reporting Dashboard Integration

```python
class EHSReportingService:
    def __init__(self, api_base_url="http://localhost:8001"):
        self.api_base_url = api_base_url
    
    def generate_dashboard_data(self, facility_id=None, days_back=30):
        """Generate data for EHS dashboard."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Base filters
        filters = {
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        }
        
        if facility_id:
            filters["facility_filter"] = {"facility_id": facility_id}
        
        dashboard_data = {}
        
        # Get electrical consumption trends
        electrical_response = requests.post(
            f"{self.api_base_url}/api/v1/extract/electrical-consumption",
            json={**filters, "include_emissions": True}
        )
        
        if electrical_response.status_code == 200:
            electrical_data = electrical_response.json()
            dashboard_data['electrical'] = self._process_electrical_data(electrical_data)
        
        # Get waste generation summary
        waste_response = requests.post(
            f"{self.api_base_url}/api/v1/extract/waste-generation",
            json={**filters, "include_emissions": True}
        )
        
        if waste_response.status_code == 200:
            waste_data = waste_response.json()
            dashboard_data['waste'] = self._process_waste_data(waste_data)
        
        return dashboard_data
    
    def _process_electrical_data(self, data):
        """Process electrical data for dashboard display."""
        summary = data.get('metadata', {})
        analysis = data.get('data', {}).get('report_data', {}).get('analysis', {})
        
        return {
            'total_records': summary.get('total_records', 0),
            'processing_time': data.get('processing_time', 0),
            'key_findings': analysis.get('key_findings', []),
            'trends': analysis.get('patterns', [])
        }
    
    def _process_waste_data(self, data):
        """Process waste data for dashboard display."""
        # Similar processing for waste data
        return {
            'total_manifests': data.get('metadata', {}).get('total_records', 0),
            'analysis': data.get('data', {}).get('report_data', {}).get('analysis', {}),
        }
```

#### 3. Automated Compliance Reporting

```python
class ComplianceReporter:
    def __init__(self, api_base_url="http://localhost:8001"):
        self.api_base_url = api_base_url
    
    def generate_quarterly_report(self, year, quarter):
        """Generate quarterly compliance report."""
        # Calculate quarter date ranges
        quarter_starts = {
            1: f"{year}-01-01",
            2: f"{year}-04-01", 
            3: f"{year}-07-01",
            4: f"{year}-10-01"
        }
        
        quarter_ends = {
            1: f"{year}-03-31",
            2: f"{year}-06-30",
            3: f"{year}-09-30", 
            4: f"{year}-12-31"
        }
        
        start_date = quarter_starts[quarter]
        end_date = quarter_ends[quarter]
        
        # Extract compliance-related data
        compliance_data = {}
        
        # Get facility emissions for reporting
        emissions_response = requests.post(
            f"{self.api_base_url}/api/v1/extract/custom",
            json={
                "query_type": "facility_emissions",
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "output_format": "json"
            }
        )
        
        if emissions_response.status_code == 200:
            compliance_data['emissions'] = emissions_response.json()
        
        # Get waste generation for RCRA reporting
        waste_response = requests.post(
            f"{self.api_base_url}/api/v1/extract/waste-generation",
            json={
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "include_disposal_details": True,
                "include_emissions": True,
                "hazardous_only": True  # Focus on hazardous waste
            }
        )
        
        if waste_response.status_code == 200:
            compliance_data['hazardous_waste'] = waste_response.json()
        
        return self._format_compliance_report(compliance_data, year, quarter)
    
    def _format_compliance_report(self, data, year, quarter):
        """Format compliance data into structured report."""
        report = {
            'report_period': f"Q{quarter} {year}",
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        # Process emissions data
        if 'emissions' in data:
            emissions_data = data['emissions']
            report['sections']['emissions'] = {
                'total_facilities': emissions_data.get('metadata', {}).get('total_records', 0),
                'summary': emissions_data.get('data', {}).get('report_data', {}).get('analysis', {})
            }
        
        # Process hazardous waste data
        if 'hazardous_waste' in data:
            waste_data = data['hazardous_waste']
            report['sections']['hazardous_waste'] = {
                'total_manifests': waste_data.get('metadata', {}).get('total_records', 0),
                'summary': waste_data.get('data', {}).get('report_data', {}).get('analysis', {})
            }
        
        return report
```

## Performance Considerations

### Request Optimization

1. **Date Range Filtering**: Use specific date ranges to limit query scope
2. **Facility Filtering**: Filter by specific facilities when possible
3. **Output Format**: Use JSON for programmatic access, TXT for human review
4. **Selective Inclusion**: Only request additional details (emissions, costs) when needed

### Recommended Practices

```python
# Good: Specific date range
payload = {
    "date_range": {
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    }
}

# Avoid: No date filtering (queries entire database)
payload = {}

# Good: Facility-specific queries
payload = {
    "facility_filter": {
        "facility_id": "FAC001"
    }
}

# Good: Only include needed data
payload = {
    "include_emissions": False,      # Skip if not needed
    "include_cost_analysis": False   # Skip if not needed
}
```

### Performance Monitoring

```python
import time

class PerformanceMonitor:
    def __init__(self, api_base_url="http://localhost:8001"):
        self.api_base_url = api_base_url
    
    def benchmark_query_types(self, date_range):
        """Benchmark performance of different query types."""
        query_types = [
            'electrical-consumption',
            'water-consumption', 
            'waste-generation'
        ]
        
        results = {}
        
        for query_type in query_types:
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base_url}/api/v1/extract/{query_type}",
                json={"date_range": date_range}
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                results[query_type] = {
                    'response_time': end_time - start_time,
                    'processing_time': data.get('processing_time', 0),
                    'total_records': data.get('metadata', {}).get('total_records', 0),
                    'records_per_second': data.get('metadata', {}).get('total_records', 0) / (end_time - start_time)
                }
            else:
                results[query_type] = {'error': response.text}
        
        return results

# Usage
monitor = PerformanceMonitor()
benchmark = monitor.benchmark_query_types({
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
})

for query_type, metrics in benchmark.items():
    if 'error' not in metrics:
        print(f"{query_type}:")
        print(f"  Response time: {metrics['response_time']:.2f}s")
        print(f"  Processing time: {metrics['processing_time']:.2f}s") 
        print(f"  Records: {metrics['total_records']}")
        print(f"  Records/sec: {metrics['records_per_second']:.1f}")
```

### Scaling Recommendations

1. **Connection Pooling**: Use persistent connections for multiple requests
2. **Parallel Processing**: Process multiple facilities or time periods in parallel
3. **Caching**: Cache frequently accessed data at the application level
4. **Database Optimization**: Ensure Neo4j indexes are properly configured

```python
# Connection pooling example
session = requests.Session()
session.headers.update({'Content-Type': 'application/json'})

# Parallel processing example
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

def extract_facility_data(facility_id, date_range):
    """Extract data for a specific facility."""
    payload = {
        "facility_filter": {"facility_id": facility_id},
        "date_range": date_range
    }
    
    response = session.post(
        "http://localhost:8001/api/v1/extract/electrical-consumption",
        json=payload
    )
    
    return facility_id, response.json()

# Process multiple facilities in parallel
facilities = ["FAC001", "FAC002", "FAC003", "FAC004"]
date_range = {"start_date": "2024-01-01", "end_date": "2024-01-31"}

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(extract_facility_data, fac_id, date_range)
        for fac_id in facilities
    ]
    
    results = {}
    for future in concurrent.futures.as_completed(futures):
        facility_id, data = future.result()
        results[facility_id] = data
```

## API Versioning Strategy

### Current Version: v1

The API follows semantic versioning principles:

- **Major version** (`v1`): Breaking changes to API contract
- **Minor updates**: New features, backward compatible
- **Patch updates**: Bug fixes, no API changes

### Version Headers

```bash
# Request specific API version
curl -H "Accept: application/vnd.ehs.v1+json" \
  http://localhost:8001/api/v1/extract/electrical-consumption
```

### Upgrade Path

When new API versions are released:

1. **v1 Support**: Current v1 endpoints will remain supported
2. **Migration Period**: Minimum 6 months before deprecation
3. **Documentation**: Migration guides provided for breaking changes
4. **Feature Flags**: New features may be released behind feature flags

### Future Versions

Planned improvements for future versions:

#### v2 (Planned)
- **Async Processing**: Long-running queries with webhook callbacks
- **Batch Operations**: Multiple extractions in single request
- **Enhanced Filtering**: More sophisticated query capabilities
- **GraphQL Support**: Alternative query interface

#### Example v2 Async Request
```json
{
  "requests": [
    {
      "type": "electrical-consumption",
      "date_range": {"start_date": "2024-01-01", "end_date": "2024-12-31"}
    },
    {
      "type": "water-consumption", 
      "date_range": {"start_date": "2024-01-01", "end_date": "2024-12-31"}
    }
  ],
  "callback_url": "https://your-app.com/webhook/ehs-data",
  "notification_email": "admin@company.com"
}
```

### Deprecation Policy

1. **Advance Notice**: 90 days minimum before deprecation
2. **Migration Support**: Dedicated support during transition
3. **Gradual Rollout**: New versions released alongside existing ones
4. **Documentation**: Clear migration paths and compatibility matrices

---

## Support and Resources

### Getting Help

- **Documentation**: This guide and inline API documentation
- **API Explorer**: Interactive documentation at `/api/docs`
- **Health Check**: Monitor API status at `/health`
- **Query Types**: Available operations at `/api/v1/query-types`

### Best Practices Summary

1. **Always check health status** before making extraction requests
2. **Use specific date ranges** to optimize query performance
3. **Include facility filters** when working with specific locations
4. **Monitor processing times** and adjust request frequency accordingly
5. **Handle errors gracefully** with proper retry logic
6. **Cache frequently accessed data** to reduce API calls
7. **Use appropriate output formats** (JSON for integration, TXT for review)

### Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| Connection refused | API not running | Check if server is started on correct port |
| 500 Internal Server Error | Database connection | Verify Neo4j credentials and connectivity |
| Empty results | Date range filter | Check if data exists for specified time period |
| Slow response times | Large dataset | Use more specific filters or smaller date ranges |
| Analysis errors | LLM API issues | Check API keys and rate limits |

This comprehensive API documentation provides everything needed to successfully integrate with and use the EHS Extraction API for environmental data analysis and reporting.