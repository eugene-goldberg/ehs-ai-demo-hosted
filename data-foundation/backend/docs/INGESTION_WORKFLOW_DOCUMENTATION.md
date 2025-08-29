# EHS AI Demo Ingestion Workflow Documentation

This document provides comprehensive documentation for the EHS AI Demo document ingestion workflow, detailing the complete architecture, processing stages, LangGraph integration, and Phase 1 enhancements.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step Workflow Description](#step-by-step-workflow-description)
4. [Complete Workflow ASCII Diagram](#complete-workflow-ascii-diagram)
5. [LangGraph Integration](#langgraph-integration)
6. [External Service Integrations](#external-service-integrations)
7. [State Management and Data Flow](#state-management-and-data-flow)
8. [Error Handling and Retry Mechanisms](#error-handling-and-retry-mechanisms)
9. [Phase 1 Enhancements](#phase-1-enhancements)
10. [Document Processing Flow](#document-processing-flow)

## Overview

The EHS AI Demo ingestion workflow is a sophisticated document processing pipeline built on LangGraph that handles Environmental, Health, and Safety (EHS) documents from initial upload through final storage in Neo4j. The system processes utility bills, water bills, waste manifests, and permits, extracting structured data and creating comprehensive knowledge graph representations.

### Key Features

- **LangGraph-based State Management**: Reliable workflow orchestration with state persistence
- **Multi-Document Type Support**: Specialized processing for different EHS document types
- **Comprehensive Validation**: Multi-stage quality checks and validation
- **Phase 1 Enhancements**: Audit trail, prorating, rejection handling, and duplicate detection
- **Neo4j Knowledge Graph**: Rich entity relationship modeling
- **Automatic Emission Calculations**: Environmental impact tracking
- **Robust Error Handling**: Retry mechanisms and graceful failure handling
- **External API Integration**: LlamaParse, OpenAI, Neo4j connectivity

## Architecture Overview

The ingestion workflow follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   LangGraph      │    │   Neo4j         │
│   Upload        ├───►│   Workflow       ├───►│   Knowledge     │
│   (File Input)  │    │   Engine         │    │   Graph         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Phase 1         │
                       │  Enhancements    │
                       │  (Audit/Reject)  │
                       └──────────────────┘
```

### Core Components

1. **IngestionWorkflow**: Main LangGraph orchestrator
2. **DocumentRecognitionService**: Document type identification
3. **EHSDocumentParser**: LlamaParse-based PDF processing
4. **Data Extractors**: Type-specific data extraction engines
5. **Phase 1 Services**: Audit trail, rejection, prorating services
6. **Neo4j Loader**: Knowledge graph population
7. **Document Indexer**: Search index creation

## Step-by-Step Workflow Description

### Stage 1: Document Recognition and Validation
1. **Document Upload**: File received via API or batch processing
2. **Document Recognition**: AI-powered document type identification
3. **Quality Assessment**: Document readability and content quality checks
4. **Acceptance Decision**: Accept or reject based on confidence thresholds

### Stage 2: File Storage and Audit Trail
1. **Source File Storage**: Secure file storage with UUID-based organization
2. **Audit Trail Creation**: Comprehensive processing history tracking
3. **Metadata Extraction**: File properties and upload context capture

### Stage 3: Document Parsing and Processing
1. **LlamaParse Integration**: Advanced PDF text and table extraction
2. **Content Structuring**: Organized text and tabular data preparation
3. **Table Recognition**: Specialized table extraction for billing data

### Stage 4: Duplicate Detection
1. **Content Fingerprinting**: Document similarity analysis
2. **Duplicate Comparison**: Cross-reference with existing documents
3. **Rejection Workflow**: Automated duplicate handling

### Stage 5: Data Extraction
1. **Extractor Selection**: Document type-specific processor selection
2. **LLM-Powered Extraction**: Structured data extraction using GPT/Claude
3. **Field Normalization**: Data standardization and type conversion

### Stage 6: Data Transformation
1. **Neo4j Schema Mapping**: Entity and relationship modeling
2. **Node Creation**: Document, facility, customer, provider entities
3. **Relationship Establishment**: Complex inter-entity connections
4. **Emission Calculations**: Automatic environmental impact computation

### Stage 7: Validation and Quality Assurance
1. **Data Validation**: Required field and range checks
2. **Business Rule Enforcement**: Domain-specific validation rules
3. **Quality Scoring**: Processing quality assessment

### Stage 8: Prorating and Financial Processing
1. **Billing Period Analysis**: Time-based cost allocation
2. **Facility Attribution**: Multi-facility cost distribution
3. **Allocation Tracking**: Detailed financial record keeping

### Stage 9: Neo4j Loading
1. **Transaction Management**: Atomic database operations
2. **Node Creation**: Structured entity creation with properties
3. **Relationship Building**: Complex graph structure establishment
4. **Index Management**: Query performance optimization

### Stage 10: Document Indexing and Completion
1. **Search Index Creation**: Vector and text-based search preparation
2. **Audit Trail Finalization**: Complete processing history recording
3. **Status Updates**: Final state transition and notification

## Complete Workflow ASCII Diagram

### Original Ingestion Workflow
```
                    ┌─── Document Upload ───┐
                    │                       │
                    ▼                       ▼
            ┌─────────────────┐    ┌─────────────────┐
            │   validate      │    │  handle_error   │
            │   document      │◄───┤  (retry logic)  │
            └─────────────────┘    └─────────────────┘
                    │                       ▲
                    ▼                       │
            ┌─────────────────┐             │
            │   parse         │─────────────┘
            │   document      │    (on error)
            └─────────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │   extract       │
            │   data          │
            └─────────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │   transform     │
            │   data          │
            └─────────────────┘
                    │
                    ▼
            ┌─────────────────┐    ┌─────────────────┐
            │   validate      │    │     retry       │
            │   data          ├───►│   extract       │
            └─────────────────┘    └─────────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │   load to       │
            │   Neo4j         │
            └─────────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │   index         │
            │   document      │
            └─────────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │   complete      │
            │   processing    │
            └─────────────────┘
                    │
                    ▼
                  [END]
```

### Enhanced Workflow with Phase 1 Features
```
                    ┌─── Document Upload ───┐
                    │                       │
                    ▼                       ▼
            ┌─────────────────┐    ┌─────────────────┐
            │  recognize      │    │  handle_error   │
            │  document       │◄───┤  (retry logic)  │
            │  [AI-powered]   │    │  + audit trail  │
            └─────────────────┘    └─────────────────┘
                    │                       ▲
                    ▼                       │
          ┌─────────────────────┐           │
          │   Decision Point    │           │
          │   Accept/Reject?    │           │
          └─────────────────────┘           │
                    │                       │
        ┌───────────┼───────────┐           │
        │accept     │           │reject     │
        ▼           ▼           ▼           │
┌─────────────┐┌─────────────┐┌─────────────┐│
│store_source ││   handle    ││             ││
│file + audit ││  rejection  ││             ││
│    trail    ││   workflow  ││             ││
└─────────────┘└─────────────┘│             ││
        │           │         │             ││
        ▼           ▼         │             ││
┌─────────────┐    [END]      │             ││
│  validate   │               │             ││
│ document    │───────────────┘             ││
└─────────────┘                             ││
        │                                   ││
        ▼                                   ││
┌─────────────────┐                         ││
│validate_document│                         ││
│    _quality     │─────────────────────────┘│
│ [Phase 1 check] │          (on error)      │
└─────────────────┘                         │
        │                                   │
        ▼                                   │
┌─────────────────┐                         │
│   Decision      │                         │
│Quality Pass/Fail│                         │
└─────────────────┘                         │
        │                                   │
┌───────┼───────┐                           │
│pass   │       │fail                       │
▼       ▼       ▼                           │
│   ┌─────────────┐                         │
│   │handle_reject│                         │
│   │   + audit   │                         │
│   └─────────────┘                         │
│           │                               │
│           ▼                               │
│         [END]                             │
│                                           │
▼                                           │
┌─────────────────┐                         │
│     parse       │─────────────────────────┘
│   document      │        (on error)
│  + audit log    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│check_for        │
│  duplicates     │
│ [Phase 1 check] │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   Decision      │
│ Unique/Duplicate│
└─────────────────┘
        │
┌───────┼───────┐
│unique │       │duplicate
▼       ▼       ▼
│   ┌─────────────┐
│   │handle_reject│
│   │(duplicate)  │
│   └─────────────┘
│           │
│           ▼
│         [END]
│
▼
┌─────────────────┐
│    extract      │
│     data        │
│  + audit log    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   transform     │
│     data        │
│ + Phase 1 meta  │
└─────────────────┘
        │
        ▼
┌─────────────────┐    ┌─────────────────┐
│  validate_data  │    │     retry       │
│ + quality score ├───►│   extract       │
└─────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐    ┌─────────────────┐
│ process_        │    │      skip       │
│ prorating       ├───►│   prorating     │
│ [Phase 1]       │    │                 │
└─────────────────┘    └─────────────────┘
        │                       │
        ▼                       │
┌─────────────────┐              │
│   load to       │◄─────────────┘
│   Neo4j         │
│ + audit trail   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│     index       │
│   document      │
│ + search index  │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   complete      │
│  processing     │
│ + finalize      │
│ audit trail     │
└─────────────────┘
        │
        ▼
      [END]
```

### Decision Points and Conditional Edges

#### 1. Document Acceptance Decision
```
┌─────────────────────────────────────┐
│        check_document_acceptance    │
├─────────────────────────────────────┤
│ if state.is_accepted:               │
│     return "accept"                 │
│ else:                               │
│     return "reject"                 │
└─────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   "accept"               "reject"
        │                     │
        ▼                     ▼
┌─────────────┐    ┌─────────────────┐
│  validate   │    │ handle_rejection│
└─────────────┘    └─────────────────┘
```

#### 2. Quality Validation Decision
```
┌─────────────────────────────────────┐
│       check_quality_validation      │
├─────────────────────────────────────┤
│ if state.get("errors"):             │
│     return "error"                  │
│ if state.get("rejection_id"):       │
│     return "rejected"               │
│ return "passed"                     │
└─────────────────────────────────────┘
```

#### 3. Duplicate Detection Decision
```
┌─────────────────────────────────────┐
│       check_duplicate_status        │
├─────────────────────────────────────┤
│ if state.get("errors"):             │
│     return "error"                  │
│ if state.get("is_duplicate"):       │
│     return "duplicate"              │
│ return "unique"                     │
└─────────────────────────────────────┘
```

#### 4. Prorating Processing Decision
```
┌─────────────────────────────────────┐
│      check_prorating_needed         │
├─────────────────────────────────────┤
│ if (enable_phase1 and               │
│     doc_type in utility_bills and   │
│     extracted_data):                │
│     return "prorating_needed"       │
│ return "skip_prorating"             │
└─────────────────────────────────────┘
```

## LangGraph Integration

### LangGraph Workflow Architecture

LangGraph provides the foundation for the ingestion workflow with several key benefits:

1. **State Management**: Persistent state across workflow nodes
2. **Conditional Routing**: Dynamic workflow paths based on processing results
3. **Error Handling**: Built-in retry and error recovery mechanisms
4. **Parallel Processing**: Concurrent execution of independent operations
5. **Workflow Visualization**: Clear representation of processing flows

### State Definition

```python
class DocumentState(TypedDict):
    # Input
    file_path: str
    document_id: str
    upload_metadata: Dict[str, Any]
    
    # Recognition and Validation
    recognition_result: Optional[Dict[str, Any]]
    is_accepted: bool
    rejection_reason: Optional[str]
    
    # Processing State
    document_type: Optional[str]
    parsed_content: Optional[List[Dict[str, Any]]]
    extracted_data: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    
    # Phase 1 Enhancement Fields
    source_file_path: Optional[str]
    audit_trail_id: Optional[str]
    rejection_id: Optional[str]
    validation_score: Optional[float]
    is_duplicate: bool
    prorating_allocation_id: Optional[str]
    
    # Error Handling
    errors: List[str]
    retry_count: int
    status: str
```

### Workflow Construction

```python
def _build_workflow(self) -> StateGraph:
    workflow = StateGraph(DocumentState)
    
    # Add processing nodes
    workflow.add_node("recognize_document", self.recognize_document)
    workflow.add_node("store_source_file", self.store_source_file)
    workflow.add_node("validate_document_quality", self.validate_document_quality)
    workflow.add_node("check_for_duplicates", self.check_for_duplicates)
    workflow.add_node("extract", self.extract_data)
    workflow.add_node("process_prorating", self.process_prorating)
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "recognize_document",
        self.check_document_acceptance,
        {"accept": "store_source_file", "reject": "handle_rejection"}
    )
    
    return workflow.compile()
```

### Node Processing Pattern

Each workflow node follows a consistent pattern:

```python
def process_node(self, state: DocumentState) -> DocumentState:
    logger.info(f"Processing node: {node_name}")
    
    try:
        # Update processing status
        state["status"] = ProcessingStatus.PROCESSING
        
        # Perform node-specific processing
        result = perform_processing_logic(state)
        
        # Update state with results
        state.update(result)
        
        # Log success
        logger.info(f"Node {node_name} completed successfully")
        
    except Exception as e:
        # Handle errors
        state["errors"].append(f"{node_name} error: {str(e)}")
        logger.error(f"Node {node_name} failed: {str(e)}")
    
    return state
```

## External Service Integrations

### 1. Neo4j Knowledge Graph

**Connection Management**:
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_username, neo4j_password)
)
```

**Node Creation Pattern**:
```python
def create_nodes_and_relationships(self, nodes, relationships):
    with driver.session() as session:
        # Create nodes
        for node in nodes:
            labels = ":".join(node["labels"])
            props = node["properties"]
            
            prop_strings = [f"{k}: ${k}" for k in props.keys()]
            props_str = "{" + ", ".join(prop_strings) + "}"
            
            query = f"CREATE (n:{labels} {props_str}) RETURN n"
            session.run(query, **props)
        
        # Create relationships
        for rel in relationships:
            query = """
            MATCH (a {id: $source_id})
            MATCH (b {id: $target_id})
            CREATE (a)-[r:%s]->(b)
            SET r = $properties
            """ % rel["type"]
            
            session.run(query, 
                source_id=rel["source"],
                target_id=rel["target"],
                properties=rel.get("properties", {})
            )
```

**Entity Relationship Mapping**:
- **Documents**: Base document nodes with metadata
- **UtilityBill/WaterBill**: Billing information nodes
- **Facility**: Physical location entities
- **Customer**: Billing customer entities
- **UtilityProvider**: Service provider entities
- **Meter**: Measurement device entities
- **Emission**: Environmental impact calculations
- **WasteManifest**: Waste tracking documents
- **WasteShipment**: Waste transportation records

### 2. LlamaParse API Integration

**Document Parsing**:
```python
class EHSDocumentParser:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.parser = LlamaParse(api_key=api_key)
    
    def parse_document(self, file_path: str, document_type: str):
        # Configure parser for document type
        self.parser.result_type = "markdown"
        self.parser.verbose = True
        
        # Parse document
        documents = self.parser.load_data(file_path)
        
        # Extract tables separately
        tables = self.extract_tables(documents)
        
        return documents, tables
```

**Table Extraction**:
```python
def extract_tables(self, documents):
    tables = []
    for doc in documents:
        # Use table extraction instructions
        table_parser = LlamaParse(
            api_key=self.api_key,
            parsing_instruction="Focus on extracting tabular data"
        )
        
        doc_tables = table_parser.get_tables(doc)
        tables.extend(doc_tables)
    
    return tables
```

### 3. OpenAI/Anthropic LLM Integration

**Data Extraction Configuration**:
```python
def configure_llm(self, model_name: str):
    if "claude" in model_name.lower():
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            max_tokens=4000
        )
    else:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=4000
        )
```

**Extraction Chain Setup**:
```python
def create_extraction_chain(self, document_type: str):
    extraction_prompt = ChatPromptTemplate.from_template(
        """
        Extract structured data from this {doc_type} document:
        
        Document content: {content}
        
        Extract the following fields:
        {field_definitions}
        
        Return as valid JSON.
        """
    )
    
    chain = extraction_prompt | self.llm | JsonOutputParser()
    return chain
```

### 4. Document Recognition Service

**AI-Powered Document Classification**:
```python
class DocumentRecognitionService:
    def analyze_document_type(self, file_path: str):
        # Extract document features
        features = self.extract_features(file_path)
        
        # Classify document type
        classification = self.classify_document(features)
        
        # Calculate confidence score
        confidence = self.calculate_confidence(classification, features)
        
        return {
            "document_type": classification,
            "confidence": confidence,
            "features": features
        }
```

## State Management and Data Flow

### State Persistence

The workflow maintains persistent state across all processing nodes:

```python
# Initial state creation
initial_state: DocumentState = {
    "file_path": file_path,
    "document_id": document_id,
    "upload_metadata": metadata or {},
    "status": ProcessingStatus.PENDING,
    "start_time": datetime.utcnow().timestamp()
}

# State updates flow through each node
final_state = self.workflow.invoke(initial_state)
```

### Data Flow Patterns

#### 1. Sequential Processing
```
State → Node A → Updated State → Node B → Final State
```

#### 2. Conditional Branching
```
State → Decision Node → Branch A or Branch B → Merged State
```

#### 3. Error Recovery
```
State → Node (error) → Error Handler → Retry or Fail
```

### State Tracking Example

```python
def track_processing_progress(self, state: DocumentState):
    progress = {
        "document_id": state["document_id"],
        "current_status": state["status"],
        "nodes_completed": [],
        "errors": state.get("errors", []),
        "processing_time": time.time() - state.get("start_time", 0)
    }
    
    # Track completion of major milestones
    if state.get("recognition_result"):
        progress["nodes_completed"].append("document_recognition")
    
    if state.get("parsed_content"):
        progress["nodes_completed"].append("document_parsing")
    
    if state.get("extracted_data"):
        progress["nodes_completed"].append("data_extraction")
    
    if state.get("neo4j_nodes"):
        progress["nodes_completed"].append("data_transformation")
    
    return progress
```

## Error Handling and Retry Mechanisms

### Error Handling Strategy

The workflow implements comprehensive error handling at multiple levels:

#### 1. Node-Level Error Handling
```python
def safe_node_execution(self, node_function, state: DocumentState):
    try:
        return node_function(state)
    except Exception as e:
        state["errors"].append(f"Node error: {str(e)}")
        state["retry_count"] += 1
        logger.error(f"Node execution failed: {str(e)}")
        return state
```

#### 2. Workflow-Level Error Recovery
```python
def handle_error(self, state: DocumentState) -> DocumentState:
    state["retry_count"] += 1
    
    if state["retry_count"] >= self.max_retries:
        state["status"] = ProcessingStatus.FAILED
        logger.error(f"Max retries exceeded for document: {state['document_id']}")
    else:
        state["status"] = ProcessingStatus.RETRY
        state["errors"] = []  # Clear errors for retry
        logger.info(f"Retrying processing. Attempt {state['retry_count']}")
    
    return state
```

#### 3. Service-Level Error Handling
```python
def safe_api_call(self, api_function, *args, **kwargs):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            return api_function(*args, **kwargs)
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
```

### Retry Logic Implementation

```python
def check_retry(self, state: DocumentState) -> str:
    """Determine if processing should be retried."""
    if state["status"] == ProcessingStatus.RETRY:
        return "retry"
    else:
        return "fail"

# Conditional edge configuration
workflow.add_conditional_edges(
    "handle_error",
    self.check_retry,
    {
        "retry": "validate",  # Restart from validation
        "fail": END
    }
)
```

### Error Recovery Strategies

#### 1. Graceful Degradation
```python
def extract_with_fallback(self, content: str, document_type: str):
    try:
        # Primary extraction method
        return self.advanced_extraction(content, document_type)
    except Exception:
        logger.warning("Advanced extraction failed, using fallback")
        # Fallback to simpler extraction
        return self.basic_extraction(content, document_type)
```

#### 2. Partial Success Handling
```python
def handle_partial_extraction(self, extraction_results: dict):
    required_fields = ["total_cost", "billing_period"]
    optional_fields = ["meter_readings", "facility_info"]
    
    missing_required = [f for f in required_fields 
                       if f not in extraction_results]
    
    if missing_required:
        raise ValueError(f"Missing required fields: {missing_required}")
    
    # Continue with available data
    return self.create_partial_entities(extraction_results)
```

## Phase 1 Enhancements

Phase 1 enhancements add comprehensive audit trail, rejection handling, duplicate detection, and prorating capabilities to the base ingestion workflow.

### 1. Audit Trail Service

**Purpose**: Comprehensive tracking of document processing history

**Features**:
- **File Storage Management**: UUID-based secure file organization
- **Processing History**: Complete workflow step tracking
- **Metadata Preservation**: Original filename and upload context
- **Secure Access**: Controlled file retrieval and serving

**Implementation**:
```python
class AuditTrailService:
    def store_source_file(self, uploaded_file_path: str, 
                         original_filename: str, 
                         document_id: str = None) -> Tuple[str, str]:
        # Generate UUID if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        # Create UUID-based directory structure
        storage_dir = self.base_storage_path / document_id[:2] / document_id[2:4]
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Store file with metadata
        stored_path = storage_dir / f"{document_id}_{original_filename}"
        shutil.copy2(uploaded_file_path, stored_path)
        
        # Create audit entry
        self.create_audit_entry(document_id, original_filename, stored_path)
        
        return document_id, str(stored_path)
```

**Audit Trail Integration**:
```python
def store_source_file(self, state: DocumentState) -> DocumentState:
    try:
        if self.enable_phase1:
            # Store with audit trail
            document_id, stored_path = self.audit_trail_service.store_source_file(
                uploaded_file_path=state["file_path"],
                original_filename=os.path.basename(state["file_path"]),
                document_id=state["document_id"]
            )
            
            state["source_file_path"] = stored_path
            state["phase1_processing"]["file_stored"] = True
            
        else:
            # Direct file usage
            state["source_file_path"] = state["file_path"]
            
    except Exception as e:
        state["errors"].append(f"File storage error: {str(e)}")
    
    return state
```

### 2. Document Rejection Workflow

**Purpose**: Intelligent document quality assessment and rejection handling

**Rejection Reasons**:
- **Poor Quality**: Unreadable or damaged documents
- **Unsupported Type**: Documents outside supported categories
- **Duplicate**: Previously processed documents
- **Invalid Format**: Incorrect file formats or corruption
- **Insufficient Data**: Missing critical information

**Rejection Process**:
```python
class RejectionWorkflowService:
    def validate_document_quality(self, document_id: str) -> ValidationResult:
        # Perform quality checks
        quality_score = self.calculate_quality_score(document_id)
        rule_violations = self.check_business_rules(document_id)
        
        is_valid = (quality_score >= self.quality_threshold and 
                   len(rule_violations) == 0)
        
        rejection_reasons = []
        if quality_score < self.quality_threshold:
            rejection_reasons.append(RejectionReason.POOR_QUALITY)
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            rule_violations=rule_violations,
            rejection_reasons=rejection_reasons
        )
    
    def initiate_rejection_review(self, document_id: str, 
                                 reason: RejectionReason,
                                 reviewer: str,
                                 notes: str,
                                 auto_approve: bool = False) -> str:
        rejection_id = str(uuid.uuid4())
        
        # Create rejection record in Neo4j
        self.create_rejection_record(
            rejection_id, document_id, reason, reviewer, notes
        )
        
        if auto_approve:
            self.approve_rejection(rejection_id)
        
        return rejection_id
```

**Workflow Integration**:
```python
def validate_document_quality(self, state: DocumentState) -> DocumentState:
    try:
        if self.enable_phase1:
            validation_result = self.rejection_service.validate_document_quality(
                state["document_id"]
            )
            
            if not validation_result.is_valid:
                rejection_id = self.rejection_service.initiate_rejection_review(
                    state["document_id"],
                    validation_result.rejection_reasons[0],
                    "system_validation",
                    f"Quality validation failed: {validation_result.rule_violations}",
                    auto_approve=True
                )
                
                state["rejection_id"] = rejection_id
                state["rejection_reason"] = validation_result.rejection_reasons[0].value
                state["status"] = ProcessingStatus.REJECTED
                
        else:
            # Phase 1 disabled - assume valid
            state["validation_score"] = 100.0
            
    except Exception as e:
        state["errors"].append(f"Quality validation error: {str(e)}")
    
    return state
```

### 3. Duplicate Detection

**Purpose**: Prevent processing of duplicate documents

**Detection Methods**:
- **Content Fingerprinting**: SHA-256 hash comparison
- **Semantic Similarity**: Vector-based content similarity
- **Metadata Matching**: Filename and upload context comparison
- **Business Logic**: Document-specific duplicate criteria

**Implementation**:
```python
def check_for_duplicates(self, state: DocumentState) -> DocumentState:
    try:
        if self.enable_phase1:
            duplicate_result = self.rejection_service.check_for_duplicates(
                state["document_id"],
                similarity_threshold=0.85
            )
            
            state["is_duplicate"] = not duplicate_result.is_valid
            
            if not duplicate_result.is_valid:
                rejection_id = self.rejection_service.initiate_rejection_review(
                    state["document_id"],
                    RejectionReason.DUPLICATE,
                    "system_validation",
                    f"Duplicate detected: {duplicate_result.rule_violations}",
                    auto_approve=True
                )
                
                state["rejection_id"] = rejection_id
                state["status"] = ProcessingStatus.REJECTED
                
        else:
            state["is_duplicate"] = False
            
    except Exception as e:
        state["errors"].append(f"Duplicate check error: {str(e)}")
    
    return state
```

### 4. Prorating Service

**Purpose**: Intelligent cost allocation for utility bills across multiple facilities or time periods

**Prorating Features**:
- **Multi-Facility Attribution**: Distribute costs across multiple locations
- **Time-Based Allocation**: Prorate costs across billing periods
- **Usage-Based Distribution**: Allocate based on actual consumption
- **Custom Allocation Rules**: Business-specific distribution logic

**Prorating Logic**:
```python
class ProRatingService:
    def process_utility_bill(self, 
                           billing_period: BillingPeriod,
                           facility_info: FacilityInfo,
                           allocation_rules: List[AllocationRule]) -> ProRatingResult:
        
        # Calculate base allocation
        total_allocation = sum(rule.percentage for rule in allocation_rules)
        
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError("Allocation percentages must sum to 100%")
        
        # Create allocation entries
        allocations = []
        for rule in allocation_rules:
            allocated_amount = billing_period.total_cost * (rule.percentage / 100.0)
            
            allocation = AllocationEntry(
                facility_id=rule.facility_id,
                allocation_percentage=rule.percentage,
                allocated_amount=allocated_amount,
                allocation_method=rule.method,
                billing_period_start=billing_period.start_date,
                billing_period_end=billing_period.end_date
            )
            
            allocations.append(allocation)
        
        # Store in Neo4j
        allocation_id = self.store_prorating_allocation(
            billing_period, facility_info, allocations
        )
        
        return ProRatingResult(
            allocation_id=allocation_id,
            allocations=allocations,
            total_allocated=sum(a.allocated_amount for a in allocations)
        )
```

**Workflow Integration**:
```python
def process_prorating(self, state: DocumentState) -> DocumentState:
    try:
        if (state["document_type"] in ["utility_bill", "water_bill"] and 
            self.enable_phase1):
            
            extracted_data = state["extracted_data"]
            if extracted_data:
                # In real implementation:
                # 1. Create BillingPeriod from extracted data
                # 2. Get FacilityInfo from database
                # 3. Call prorating_service.process_utility_bill()
                
                state["phase1_processing"]["prorating"] = {
                    "attempted": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "document_type": state["document_type"]
                }
        else:
            state["phase1_processing"]["prorating"] = {
                "attempted": False,
                "reason": "Document type not eligible or Phase 1 disabled"
            }
            
    except Exception as e:
        state["errors"].append(f"Pro-rating error: {str(e)}")
    
    return state
```

## Document Processing Flow

### Complete Processing Example

Here's how a utility bill moves through the complete enhanced workflow:

#### 1. Initial Upload
```json
{
  "file_path": "/tmp/electric_bill.pdf",
  "document_id": "doc_12345",
  "upload_metadata": {
    "source": "web_upload",
    "timestamp": "2025-08-28T10:00:00Z",
    "uploader": "system"
  }
}
```

#### 2. Document Recognition
```json
{
  "recognition_result": {
    "document_type": "electricity_bill",
    "confidence": 0.92,
    "features": {
      "key_terms": ["kWh", "billing period", "utility"],
      "metadata": {"page_count": 3, "table_count": 2}
    }
  },
  "is_accepted": true
}
```

#### 3. File Storage with Audit Trail
```json
{
  "source_file_path": "/app/storage/do/c1/doc_12345_electric_bill.pdf",
  "original_filename": "electric_bill.pdf",
  "phase1_processing": {
    "file_stored": true,
    "validation_started": "2025-08-28T10:00:01Z"
  }
}
```

#### 4. Document Parsing
```json
{
  "parsed_content": [
    {
      "content": "VOLTSTREAM ENERGY\nAccount: 85-23459-88-1...",
      "metadata": {"page": 1, "content_type": "text"}
    },
    {
      "tables": [
        {
          "headers": ["Service Type", "Previous", "Current", "Usage"],
          "rows": [["Peak", "543210", "613210", "70000"]]
        }
      ],
      "metadata": {"type": "extracted_tables"}
    }
  ]
}
```

#### 5. Data Extraction
```json
{
  "extracted_data": {
    "account_number": "85-23459-88-1",
    "billing_period_start": "2025-07-01",
    "billing_period_end": "2025-07-31",
    "total_kwh": 130000,
    "total_cost": 18245.67,
    "facility_name": "Apex Manufacturing - Plant A",
    "customer_name": "Apex Manufacturing Inc.",
    "provider_name": "Voltstream Energy"
  }
}
```

#### 6. Data Transformation (Neo4j Schema)
```json
{
  "neo4j_nodes": [
    {
      "labels": ["Document", "UtilityBill"],
      "properties": {
        "id": "doc_12345",
        "document_type": "electricity_bill",
        "account_number": "85-23459-88-1",
        "validation_score": 92.0
      }
    },
    {
      "labels": ["UtilityBill"],
      "properties": {
        "id": "bill_doc_12345",
        "total_kwh": 130000,
        "total_cost": 18245.67
      }
    },
    {
      "labels": ["Facility"],
      "properties": {
        "id": "facility_apex_manufacturing_plant_a",
        "name": "Apex Manufacturing - Plant A"
      }
    }
  ],
  "neo4j_relationships": [
    {
      "source": "doc_12345",
      "target": "bill_doc_12345",
      "type": "EXTRACTED_TO"
    },
    {
      "source": "bill_doc_12345",
      "target": "facility_apex_manufacturing_plant_a",
      "type": "BILLED_TO"
    }
  ]
}
```

#### 7. Final State
```json
{
  "status": "completed",
  "processing_time": 45.23,
  "phase1_processing": {
    "file_stored": true,
    "quality_validation": {"score": 92.0, "passed": true},
    "duplicate_check": {"is_duplicate": false, "passed": true},
    "prorating": {"attempted": true, "completed": true},
    "completed_at": "2025-08-28T10:00:45Z"
  },
  "indexed": true,
  "errors": []
}
```

This comprehensive workflow documentation provides a complete understanding of the EHS AI Demo ingestion system, from initial document upload through final Neo4j storage, including all Phase 1 enhancements for audit trail, rejection handling, duplicate detection, and prorating capabilities.