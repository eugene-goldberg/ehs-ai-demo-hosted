# Water Bill RAG Implementation Documentation

## Overview

This document details the complete implementation of the Retrieval-Augmented Generation (RAG) system for processing water utility bills in the EHS AI Platform. The system extracts structured data from PDF water bills, creates a comprehensive knowledge graph in Neo4j, and enables intelligent querying and reporting for water consumption and emissions tracking.

## Architecture Overview

The implementation consists of two main workflows:

1. **Ingestion Workflow**: Processes PDF water bill documents, extracts structured data, and populates Neo4j
2. **Extraction Workflow**: Queries the Neo4j graph and generates comprehensive water consumption reports

### Key Technologies
- **LlamaParse**: PDF parsing and text extraction
- **OpenAI GPT-4**: Structured data extraction from unstructured text
- **Neo4j**: Graph database for storing entities and relationships
- **LangGraph**: Workflow orchestration
- **LangChain**: LLM integration and prompt management

## Ingestion Workflow

### Purpose
The ingestion workflow transforms unstructured PDF water bills into a structured knowledge graph with comprehensive entity extraction and relationship mapping, including water-specific emissions calculations.

### Key Files

#### 1. `backend/src/workflows/ingestion_workflow.py`
Main workflow orchestration using LangGraph StateGraph.

**Key Components:**
- `DocumentState`: TypedDict defining workflow state
- `IngestionWorkflow`: Main class orchestrating the pipeline
- Water bill specific handling in `transform_data` method
- Water emission calculation (0.0002 kg CO2/gallon)

**Graph Schema Created:**
```cypher
// Nodes
(:Document:Waterbill {id, type, account_number, statement_date, file_path})
(:WaterBill {id, billing_period_start/end, total_gallons, total_cost, charges...})
(:Facility {id, name, address})
(:Customer {id, name, billing_address, attention})
(:UtilityProvider {id, name, address, phone, website, payment_address})
(:Meter {id, type: "water", service_type, previous/current_reading, usage, unit})
(:Emission {id, amount, unit, calculation_method, emission_factor, source_type: "water"})

// Relationships
(Document)-[:EXTRACTED_TO]->(WaterBill)
(WaterBill)-[:BILLED_TO]->(Facility)
(WaterBill)-[:BILLED_FOR]->(Customer)
(WaterBill)-[:PROVIDED_BY]->(UtilityProvider)
(Meter)-[:MONITORS]->(Facility)
(Meter)-[:RECORDED_IN]->(WaterBill)
(WaterBill)-[:RESULTED_IN]->(Emission)
```

#### 2. `backend/src/extractors/ehs_extractors.py`
Defines data extraction logic and Pydantic models.

**Key Classes:**
- `WaterBillData`: Comprehensive Pydantic model with fields for:
  - Dates (billing period, statement date, due date)
  - Customer info (name, address, attention line)
  - Facility info (service location name and address)
  - Water utility provider details
  - Water metrics (total gallons, cubic meters if provided)
  - Cost breakdown (water consumption, sewer, stormwater, conservation tax)
  - Meter readings with units (gallons, CCF, cubic meters)
- `WaterBillExtractor`: LLM-powered extraction with water-specific prompts

**Enhanced Extraction Prompt:**
- Distinguishes between CUSTOMER (billed to) and SERVICE LOCATION (facility)
- Extracts complete water meter readings with service types
- Captures water-specific charges (sewer, stormwater, conservation)
- Handles various water measurement units
- Notes that water providers are often municipal entities

#### 3. `backend/src/parsers/llama_parser.py`
Handles PDF parsing using LlamaParse API.

**Water Bill Specific Features:**
- Document type detection for "water" in filename
- Water bill specific parsing instructions
- Table extraction for rate structures
- Preservation of meter reading tables

**Parsing Instructions for Water Bills:**
```
Extract the following information from this water utility bill:
- Account number and service address
- Customer name and billing address (billed to)
- Facility name and address (service location)
- Water utility provider information
- Billing period (start and end dates)
- Statement date and due date
- Water consumption in gallons (and cubic meters if provided)
- Meter readings (previous, current, usage)
- All charges (water consumption, sewer service, stormwater fees, 
  conservation tax, infrastructure surcharge)
- Rate information (cost per gallon or per unit)
- Total amount due
Preserve all tabular data in markdown format.
```

#### 4. `backend/src/indexing/document_indexer.py`
Creates vector and graph indexes for semantic search.

**Components:**
- Vector index using OpenAI embeddings
- Property graph index for water entity relationships
- Hybrid search capabilities for water consumption queries

### Data Flow

1. **Input**: PDF water bill (e.g., `data/water_bill.pdf`)
2. **Parsing**: LlamaParse extracts text and tables
3. **Extraction**: GPT-4 extracts structured data using water-specific prompts
4. **Transformation**: Convert to graph schema with proper IDs
5. **Validation**: Check required fields (gallons, billing period)
6. **Loading**: Create nodes and relationships in Neo4j
7. **Indexing**: Build search indexes

### Water-Specific Features

#### Emission Calculation
```python
# Water treatment and distribution emissions
# Typical factor: 0.0002 kg CO2 per gallon
water_emission_factor = 0.0002
emission_node = {
    "labels": ["Emission"],
    "properties": {
        "id": f"emission_{state['document_id']}",
        "amount": float(extracted["total_gallons"]) * water_emission_factor,
        "unit": "kg_CO2",
        "calculation_method": "water_treatment_distribution_factor",
        "emission_factor": water_emission_factor,
        "source_type": "water"
    }
}
```

#### Validation Rules
- Required fields: `billing_period_start`, `billing_period_end`, `total_gallons`
- Water usage range validation: 0 to 10,000,000 gallons
- Date format standardization to YYYY-MM-DD

### Fallback Mechanisms

For the demo water bill, fallback values are provided when LLM extraction fails:
- Facility: "Apex Manufacturing - Plant A" (shared with electric)
- Customer: "Apex Manufacturing Inc."
- Provider: "City of Mechanicsburg Water Department"
- Meter: WTR-5521-A (Domestic Water, 13,470 gallons usage)

## Extraction Workflow

### Purpose
Query the knowledge graph for water consumption data and generate comprehensive reports for water usage analysis, cost tracking, and emissions monitoring.

### Key Files

#### 1. `backend/src/workflows/extraction_workflow.py`
Orchestrates water data extraction and report generation.

**Key Components:**
- `QueryType.WATER_CONSUMPTION`: New query type for water bills
- Water-specific query templates
- Integration with existing report generation
- Support for combined facility reports (water + electric)

**Water Consumption Queries:**
```cypher
// Basic water bill retrieval
MATCH (d:Document:Waterbill)-[:EXTRACTED_TO]->(w:WaterBill)
RETURN d, w
ORDER BY w.billing_period_end DESC

// Detailed water bill with all relationships
MATCH (w:WaterBill)-[:BILLED_TO]->(f:Facility)
MATCH (w)-[:BILLED_FOR]->(c:Customer)
MATCH (w)-[:PROVIDED_BY]->(p:UtilityProvider)
OPTIONAL MATCH (w)-[:RESULTED_IN]->(e:Emission)
OPTIONAL MATCH (m:Meter)-[:RECORDED_IN]->(w)
RETURN w, f, c, p, e, collect(m) as meters

// Water consumption aggregation
MATCH (w:WaterBill)
WHERE w.billing_period_start >= $start_date 
  AND w.billing_period_end <= $end_date
RETURN SUM(w.total_gallons) as total_water_usage,
       AVG(w.total_cost) as avg_cost,
       COUNT(w) as bill_count
```

### Report Types

1. **Water Consumption Report**
   - Total water usage by period
   - Cost analysis with charge breakdown
   - Meter reading details
   - Provider information

2. **Water Emissions Report**
   - Water treatment/distribution emissions
   - Emission factors and calculation methods
   - Trends over time

3. **Combined Facility Report**
   - Total emissions from both water and electricity
   - Comparative analysis
   - Unified sustainability metrics

## Running the Workflows

### Prerequisites

1. **Environment Setup**
```bash
cd data-foundation
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

2. **Environment Variables**
Add to `backend/.env` file:
```env
LLAMA_PARSE_API_KEY=your_llama_parse_key
OPENAI_API_KEY=your_openai_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=EhsAI2024!
```

3. **Neo4j Setup**
- Install Neo4j Desktop or use Docker
- Create database with credentials above
- Install APOC plugin

### Running Water Bill Ingestion

1. **Clear existing data (optional)**
```bash
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'EhsAI2024!'))
with driver.session() as session:
    session.run('MATCH (n) DETACH DELETE n')
driver.close()
"
```

2. **Run water bill ingestion**
```bash
# Using test script
python scripts/test_water_bill_ingestion.py

# Or programmatically
from backend.src.workflows.ingestion_workflow import IngestionWorkflow

workflow = IngestionWorkflow(
    llama_parse_api_key=os.getenv("LLAMA_PARSE_API_KEY"),
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="EhsAI2024!",
    llm_model="gpt-4"
)

result = workflow.process_document(
    file_path="data/water_bill.pdf",
    document_id=f"water_bill_{timestamp}",
    metadata={"source": "test", "document_type": "water_bill"}
)
```

### Running Water Bill Extraction

```bash
# Using test script
python scripts/test_water_bill_extraction.py

# Or programmatically
from backend.src.workflows.extraction_workflow import DataExtractionWorkflow, QueryType

workflow = DataExtractionWorkflow(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="EhsAI2024!",
    llm_model="gpt-4"
)

# Generate water consumption report
state = workflow.extract_data(
    query_type=QueryType.WATER_CONSUMPTION,
    output_format="txt"
)

# Generate detailed water bill report with custom query
custom_query = [{
    "query": """
        MATCH (w:WaterBill)-[:BILLED_TO]->(f:Facility)
        MATCH (w)-[:BILLED_FOR]->(c:Customer)
        MATCH (w)-[:PROVIDED_BY]->(p:UtilityProvider)
        OPTIONAL MATCH (w)-[:RESULTED_IN]->(e:Emission)
        OPTIONAL MATCH (m:Meter)-[:RECORDED_IN]->(w)
        RETURN w, f, c, p, e, collect(m) as meters
    """,
    "parameters": {}
}]

state = workflow.extract_data(
    query_type=QueryType.CUSTOM,
    queries=custom_query,
    output_format="txt"
)
```

## Verification

### 1. Check Neo4j Data

```bash
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'EhsAI2024!'))
with driver.session() as session:
    # Count nodes by type
    for label in ['Document', 'WaterBill', 'Facility', 'Customer', 'UtilityProvider', 'Meter', 'Emission']:
        result = session.run(f'MATCH (n:{label}) RETURN count(n) as count')
        print(f'{label}: {result.single()[\"count\"]}')
driver.close()
"
```

Expected output:
```
Document: 1
WaterBill: 1
Facility: 1
Customer: 1
UtilityProvider: 1
Meter: 1
Emission: 1
```

### 2. Verify Relationships

```cypher
MATCH (a)-[r]->(b)
WHERE NOT a:__Node__ AND NOT b:__Node__
RETURN type(r) as relationship, count(r) as count
ORDER BY relationship
```

Expected: 7 relationships total (EXTRACTED_TO, BILLED_TO, BILLED_FOR, PROVIDED_BY, MONITORS, RECORDED_IN, RESULTED_IN)

### 3. Check Generated Reports

Reports are saved to `reports/` directory:
- `ehs_report_QueryType.WATER_CONSUMPTION_[timestamp].txt`
- `ehs_report_QueryType.CUSTOM_[timestamp].txt` (detailed water bill)
- `ehs_report_QueryType.CUSTOM_[timestamp].txt` (combined facility emissions)

## Sample Data Extracted

From the demo water bill (`data/water_bill.pdf`):
```
Period: 2025-06-01 to 2025-06-30
Customer: Apex Manufacturing Inc.
Facility: Apex Manufacturing - Plant A
Provider: Aquaflow Municipal Water
Water Usage: 250,000 gallons
Total Cost: $4,891.50
  - Water Consumption: $3,750.00
  - Sewer Service: $950.00
  - Stormwater Fee: $125.00
  - Conservation Tax: $61.50
  - Infrastructure Surcharge: $5.00
Meter: WTR-MAIN-01 (Commercial service)
CO2 Emissions: 50 kg (factor: 0.0002 kg/gallon)
```

## Key Implementation Details

1. **Water-Specific Data Model**
   - Added `WaterBillData` Pydantic model
   - Included water-specific charges (sewer, stormwater, conservation)
   - Support for multiple water measurement units

2. **Municipal Provider Handling**
   - Recognition that water providers are often government entities
   - Proper extraction of municipal contact information

3. **Water Emission Calculations**
   - Factor: 0.0002 kg CO2 per gallon
   - Accounts for water treatment and distribution emissions
   - Properly tagged with source_type: "water"

4. **Integration with Existing System**
   - Reuses Facility and Customer nodes when appropriate
   - Compatible with combined facility reporting
   - Maintains consistency with electric bill structure

## Troubleshooting

### Common Issues

1. **Water Bill Not Detected**
   - Error: Document type detected as "utility_bill" instead of "water_bill"
   - Solution: Ensure filename contains "water"

2. **Missing Water Usage**
   - Issue: total_gallons field is null
   - Solution: Check PDF quality, adjust extraction prompts

3. **Incorrect Emission Calculation**
   - Issue: Emissions seem too high/low
   - Solution: Verify emission factor (0.0002 kg CO2/gallon)

4. **Provider Not Extracted**
   - Issue: Municipal provider information missing
   - Solution: Check for government/municipal naming patterns

## Future Enhancements

1. **Additional Water Metrics**
   - Peak hour usage patterns
   - Seasonal consumption analysis
   - Leak detection indicators

2. **Enhanced Emission Factors**
   - Regional water treatment variations
   - Wastewater treatment emissions
   - Source water type considerations

3. **Multi-Property Support**
   - Aggregate water usage across facilities
   - Benchmark against industry standards
   - Water efficiency scoring

4. **Advanced Analytics**
   - Water conservation opportunity identification
   - Cost optimization recommendations
   - Drought response planning

## Conclusion

This implementation provides a complete RAG solution for water bill processing, complementing the electric bill system to create a comprehensive utility tracking platform. The system successfully extracts all meaningful data from PDF water bills, calculates water-related emissions, and enables sophisticated querying and reporting for water conservation and sustainability initiatives.