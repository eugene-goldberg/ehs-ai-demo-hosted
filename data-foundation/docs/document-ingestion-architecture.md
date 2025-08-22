# EHS AI Platform - Document Ingestion Pipeline Architecture

## Overview

This document outlines the architecture for Sprint 3-4: Document Processing Pipeline, which implements automated data ingestion and extraction using LlamaParse, LlamaIndex, and LangGraph for the EHS AI Platform.

## Architecture Goals

1. **Automated Data Ingestion**: Extract structured ESG data from unstructured documents (PDF utility bills, invoices, permits)
2. **High Accuracy**: Achieve 95%+ extraction accuracy for critical data points
3. **Scalability**: Process documents in under 2 minutes each
4. **Flexibility**: Support multiple document types and formats
5. **Reliability**: Include error handling and retry mechanisms

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Document Ingestion Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐        │
│  │   Document  │    │  LlamaParse  │    │  LlamaIndex   │        │
│  │   Upload    │───▶│   Parser     │───▶│   Indexing    │        │
│  │   Service   │    │              │    │               │        │
│  └─────────────┘    └──────────────┘    └───────────────┘        │
│         │                   │                    │                  │
│         │                   │                    │                  │
│         ▼                   ▼                    ▼                  │
│  ┌─────────────────────────────────────────────────────┐          │
│  │              LangGraph Orchestration                 │          │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐│          │
│  │  │Validation│  │Extraction│ │Transform│  │ Load   ││          │
│  │  │  Agent  │─▶│  Agent  │─▶│  Agent  │─▶│ Agent  ││          │
│  │  └─────────┘  └─────────┘  └─────────┘  └────────┘│          │
│  └─────────────────────────────────────────────────────┘          │
│                                │                                    │
│                                ▼                                    │
│                        ┌──────────────┐                            │
│                        │   Neo4j      │                            │
│                        │Knowledge Graph│                            │
│                        └──────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Document Upload Service
- **Purpose**: Handle document uploads and queue management
- **Technology**: FastAPI endpoint with async processing
- **Features**:
  - File validation (PDF, images, text)
  - Document metadata capture
  - Queue management for batch processing
  - Progress tracking

### 2. LlamaParse Integration
- **Purpose**: Parse complex documents with high accuracy
- **Key Features**:
  - Table extraction from utility bills
  - Multi-column layout understanding
  - Image/chart recognition
  - Structured markdown output
- **Configuration**:
  ```python
  parser = LlamaParse(
      api_key=os.getenv("LLAMA_PARSE_API_KEY"),
      result_type="markdown",
      parsing_instruction="Extract all tables, especially utility consumption data, costs, and dates",
      use_vendor_extraction=True,
      vendor_extraction_config={
          "utility_bills": True,
          "invoices": True,
          "permits": True
      }
  )
  ```

### 3. LlamaIndex Document Processing
- **Purpose**: Create searchable indexes and enable RAG
- **Components**:
  - Document chunking with context preservation
  - Vector embeddings generation
  - Hybrid search capabilities (vector + keyword)
- **Implementation**:
  ```python
  # Document processing pipeline
  documents = parser.load_data(file_path)
  
  # Configure chunking for EHS documents
  text_splitter = TokenTextSplitter(
      chunk_size=512,  # Optimized for table data
      chunk_overlap=50
  )
  
  # Create vector index
  vector_index = VectorStoreIndex.from_documents(
      documents,
      service_context=ServiceContext.from_defaults(
          embed_model=embed_model,
          text_splitter=text_splitter
      )
  )
  ```

### 4. LangGraph Orchestration Workflow
- **Purpose**: Coordinate multi-step processing with error handling
- **Workflow Steps**:

#### Step 1: Validation Agent
- Validates document type (utility bill, permit, invoice)
- Checks document quality and readability
- Routes to appropriate extraction pipeline

#### Step 2: Extraction Agent
- Uses LLM to extract structured data based on document type
- Handles different extraction schemas:
  - Utility Bills: kWh usage, billing period, costs
  - Permits: permit numbers, dates, compliance requirements
  - Invoices: line items, quantities, environmental impact

#### Step 3: Transform Agent
- Standardizes units (e.g., converting all energy to kWh)
- Validates extracted data against business rules
- Calculates derived metrics (e.g., CO2 emissions)

#### Step 4: Load Agent
- Maps extracted data to Neo4j schema
- Creates nodes and relationships
- Updates existing records if needed

### 5. Neo4j Knowledge Graph Integration
- **Schema Design**:
  ```cypher
  // Core EHS Entities
  (d:Document {id, type, uploadDate, sourceUrl})
  (f:Facility {id, name, location})
  (u:UtilityBill {id, billDate, totalKwh, totalCost})
  (p:Permit {id, permitNumber, issueDate, expiryDate})
  (e:Emission {id, amount, unit, calculationMethod})
  
  // Relationships
  (d)-[:EXTRACTED_TO]->(u)
  (u)-[:BILLED_TO]->(f)
  (u)-[:RESULTED_IN]->(e)
  ```

## Implementation Plan

### Phase 1: Core Pipeline Setup (Week 1)
1. Set up LlamaParse API integration
2. Create basic document upload endpoint
3. Implement simple extraction workflow
4. Test with sample utility bills

### Phase 2: Advanced Processing (Week 2)
1. Build LangGraph orchestration workflow
2. Implement document-type specific extractors
3. Add error handling and retry logic
4. Create Neo4j schema and load procedures

### Phase 3: Production Readiness (Week 3)
1. Add monitoring and logging
2. Implement batch processing
3. Create API documentation
4. Performance optimization

### Phase 4: Testing & Refinement (Week 4)
1. Test with 100+ real documents
2. Measure and improve accuracy
3. Handle edge cases
4. Deploy to staging environment

## Error Handling Strategy

1. **Parse Failures**: Retry with different LlamaParse settings
2. **Extraction Errors**: Fallback to simpler extraction patterns
3. **Validation Failures**: Queue for human review
4. **Load Failures**: Implement idempotent operations

## Performance Targets

- Document parsing: < 30 seconds per document
- Data extraction: < 60 seconds per document
- Total processing: < 2 minutes per document
- Accuracy: > 95% for key data points

## Security Considerations

1. Encrypt documents at rest and in transit
2. Implement access controls for sensitive data
3. Audit trail for all document processing
4. PII detection and masking

## Next Steps

1. Create development environment with all dependencies
2. Implement proof-of-concept with single document type
3. Iterate based on accuracy results
4. Scale to full production pipeline