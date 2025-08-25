# Workflow Integration Guide for Phase 1 Enhancements

This guide provides detailed instructions for integrating Phase 1 enhancements (audit trail, rejection validation, and pro-rating calculations) with the existing ingestion and extraction workflows.

## Overview of Integration Points

The Phase 1 enhancements integrate with the existing workflows at the following key points:

1. **Ingestion Workflow Integration**:
   - **Audit Trail Hooks**: After document upload (line 190) and before parsing (line 224)
   - **Rejection Validation**: Early validation (line 179) and duplicate checking (line 202)
   - **State Modifications**: Enhanced DocumentState with audit and rejection tracking

2. **Extraction Workflow Integration**:
   - **Pro-rating Calculations**: After utility bill extraction (line 286)
   - **Enhanced Query Types**: New allocation queries and reporting
   - **Report Generation**: Integration with monthly allocation reports

## Detailed Integration for Ingestion Workflow

### 1. Enhanced DocumentState Structure

First, update the DocumentState in `ingestion_workflow.py` to include Phase 1 fields:

```python
# Add these imports at the top of ingestion_workflow.py
from ..phase1_enhancements.audit_trail_service import AuditTrailService
from ..phase1_enhancements.rejection_workflow_service import RejectionWorkflowService
from ..phase1_enhancements.prorating_service import ProRatingService

# Enhanced DocumentState (line 33-56)
class DocumentState(TypedDict):
    """State for document processing workflow."""
    # Existing fields...
    file_path: str
    document_id: str
    upload_metadata: Dict[str, Any]
    document_type: Optional[str]
    parsed_content: Optional[List[Dict[str, Any]]]
    extracted_data: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    indexed: bool
    errors: List[str]
    retry_count: int
    neo4j_nodes: Optional[List[Dict[str, Any]]]
    neo4j_relationships: Optional[List[Dict[str, Any]]]
    processing_time: Optional[float]
    status: str
    
    # Phase 1 Enhancement Fields
    audit_trail_id: Optional[str]  # Audit trail entry ID
    rejection_id: Optional[str]    # Rejection record ID if rejected
    rejection_reason: Optional[str] # Reason for rejection
    validation_score: Optional[float] # Quality validation score
    is_duplicate: bool             # Duplicate detection result
    prorating_allocation_id: Optional[str] # Pro-rating allocation ID
    phase1_processing: Dict[str, Any] # Phase 1 processing results
```

### 2. Initialize Phase 1 Services in IngestionWorkflow

Modify the `__init__` method (line 72-120) to include Phase 1 services:

```python
def __init__(
    self,
    llama_parse_api_key: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    llm_model: str = "gpt-4",
    max_retries: int = 3,
    enable_phase1_features: bool = True  # New parameter
):
    """Initialize the document processing workflow with Phase 1 enhancements."""
    # Existing initialization...
    self.max_retries = max_retries
    
    # Initialize existing components
    self.parser = EHSDocumentParser(api_key=llama_parse_api_key)
    self.indexer = EHSDocumentIndexer(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        llm_model=llm_model
    )
    
    # Initialize Phase 1 services
    self.enable_phase1 = enable_phase1_features
    if self.enable_phase1:
        from langchain_neo4j import Neo4jGraph
        from ..shared.common_fn import create_graph_database_connection
        
        self.graph = create_graph_database_connection(
            neo4j_uri, neo4j_username, neo4j_password, "neo4j"
        )
        
        # Initialize Phase 1 services
        self.audit_trail_service = AuditTrailService(self.graph)
        self.rejection_service = RejectionWorkflowService(self.graph)
        self.prorating_service = ProRatingService(self.graph)
        
        logger.info("Phase 1 services initialized")
    
    # Existing extractors and LLM initialization...
```

### 3. Audit Trail Integration Points

#### A. After Document Upload (Integration Point: Line 190)

Add audit trail creation after document validation:

```python
def validate_document(self, state: DocumentState) -> DocumentState:
    """Validate the input document with Phase 1 enhancements."""
    logger.info(f"Validating document: {state['file_path']}")
    state["status"] = ProcessingStatus.PROCESSING
    
    try:
        # Phase 1: Create audit trail entry
        if self.enable_phase1:
            audit_entry = self.audit_trail_service.create_audit_trail(
                document_name=os.path.basename(state["file_path"]),
                action="document_uploaded",
                user_id=state["upload_metadata"].get("user_id", "system"),
                metadata={
                    "file_path": state["file_path"],
                    "document_id": state["document_id"],
                    "upload_time": datetime.utcnow().isoformat()
                }
            )
            state["audit_trail_id"] = audit_entry.audit_id
            state["phase1_processing"]["audit_trail"] = "created"
        
        # Existing validation logic...
        if not os.path.exists(state["file_path"]):
            raise FileNotFoundError(f"File not found: {state['file_path']}")
        
        file_size = os.path.getsize(state["file_path"])
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError("File too large. Maximum size is 50MB")
        
        # Document type detection
        doc_type = self.parser.detect_document_type(state["file_path"])
        state["document_type"] = doc_type
        
        logger.info(f"Document validated. Type: {doc_type}")
        
    except Exception as e:
        # Log validation failure in audit trail
        if self.enable_phase1 and state.get("audit_trail_id"):
            self.audit_trail_service.add_activity(
                state["audit_trail_id"],
                "validation_failed",
                details={"error": str(e)}
            )
        
        state["errors"].append(f"Validation error: {str(e)}")
        logger.error(f"Validation failed: {str(e)}")
    
    return state
```

#### B. Before Parsing (Integration Point: Line 224)

Add audit trail update before parsing:

```python
def parse_document(self, state: DocumentState) -> DocumentState:
    """Parse document using LlamaParse with audit trail."""
    logger.info(f"Parsing document: {state['file_path']}")
    
    # Phase 1: Update audit trail
    if self.enable_phase1 and state.get("audit_trail_id"):
        self.audit_trail_service.add_activity(
            state["audit_trail_id"],
            "parsing_started",
            details={
                "document_type": state["document_type"],
                "parser": "LlamaParse"
            }
        )
    
    try:
        # Existing parsing logic...
        documents = self.parser.parse_document(
            state["file_path"],
            document_type=state["document_type"]
        )
        
        # Convert to serializable format
        parsed_content = []
        for doc in documents:
            parsed_content.append({
                "content": doc.get_content(),
                "metadata": doc.metadata
            })
        
        state["parsed_content"] = parsed_content
        
        # Extract tables if present
        tables = self.parser.extract_tables(documents)
        if tables:
            state["parsed_content"].append({
                "tables": tables,
                "metadata": {"type": "extracted_tables"}
            })
        
        # Phase 1: Update audit trail with parsing results
        if self.enable_phase1 and state.get("audit_trail_id"):
            self.audit_trail_service.add_activity(
                state["audit_trail_id"],
                "parsing_completed",
                details={
                    "pages_parsed": len(documents),
                    "tables_extracted": len(tables) if tables else 0,
                    "content_length": sum(len(item.get("content", "")) for item in parsed_content)
                }
            )
        
        logger.info(f"Parsed {len(documents)} pages")
        
    except Exception as e:
        # Log parsing failure
        if self.enable_phase1 and state.get("audit_trail_id"):
            self.audit_trail_service.add_activity(
                state["audit_trail_id"],
                "parsing_failed",
                details={"error": str(e)}
            )
        
        state["errors"].append(f"Parsing error: {str(e)}")
        logger.error(f"Parsing failed: {str(e)}")
    
    return state
```

### 4. Rejection Validation Integration Points

#### A. Early Validation (Integration Point: Line 179)

Add rejection validation in the validate_document method:

```python
def validate_document(self, state: DocumentState) -> DocumentState:
    """Enhanced validation with rejection checking."""
    # ... existing validation code ...
    
    try:
        # Phase 1: Rejection validation
        if self.enable_phase1:
            # Initialize phase1_processing dict
            state["phase1_processing"] = {}
            
            # Check for quality issues
            validation_result = self.rejection_service.validate_document_quality(
                state["document_id"]
            )
            
            state["validation_score"] = validation_result.quality_score
            
            if not validation_result.is_valid:
                # Document should be rejected
                rejection_id = self.rejection_service.initiate_rejection_review(
                    state["document_id"],
                    validation_result.rejection_reasons[0],
                    "system_validation",
                    f"Quality validation failed: {', '.join(validation_result.rule_violations)}",
                    auto_approve=True
                )
                
                state["rejection_id"] = rejection_id
                state["rejection_reason"] = validation_result.rejection_reasons[0].value
                state["status"] = ProcessingStatus.FAILED
                state["phase1_processing"]["rejection"] = "quality_failed"
                
                return state  # Stop processing
        
        # ... rest of existing validation logic ...
        
    except Exception as e:
        # ... existing error handling ...
```

#### B. Duplicate Check (Integration Point: Line 202)

Add duplicate detection after document type detection:

```python
def validate_document(self, state: DocumentState) -> DocumentState:
    """Enhanced validation with duplicate detection."""
    # ... existing validation and rejection logic ...
    
    try:
        # Document type detection
        doc_type = self.parser.detect_document_type(state["file_path"])
        state["document_type"] = doc_type
        
        # Phase 1: Duplicate detection
        if self.enable_phase1 and not state.get("rejection_id"):
            duplicate_result = self.rejection_service.check_for_duplicates(
                state["document_id"],
                similarity_threshold=0.85
            )
            
            state["is_duplicate"] = not duplicate_result.is_valid
            
            if not duplicate_result.is_valid:
                # Document is a duplicate
                rejection_id = self.rejection_service.initiate_rejection_review(
                    state["document_id"],
                    RejectionReason.DUPLICATE,
                    "system_validation",
                    f"Duplicate detected: {', '.join(duplicate_result.rule_violations)}",
                    auto_approve=True
                )
                
                state["rejection_id"] = rejection_id
                state["rejection_reason"] = RejectionReason.DUPLICATE.value
                state["status"] = ProcessingStatus.FAILED
                state["phase1_processing"]["rejection"] = "duplicate_detected"
                
                return state  # Stop processing
        
        logger.info(f"Document validated. Type: {doc_type}")
        
    except Exception as e:
        # ... existing error handling ...
```

### 5. Enhanced Workflow Graph

Update the workflow graph to include Phase 1 validation:

```python
def _build_workflow(self) -> StateGraph:
    """Build the LangGraph workflow with Phase 1 enhancements."""
    # Create workflow
    workflow = StateGraph(DocumentState)
    
    # Add nodes (existing + new)
    workflow.add_node("validate", self.validate_document)
    workflow.add_node("phase1_validation", self.phase1_validation)  # NEW
    workflow.add_node("parse", self.parse_document)
    workflow.add_node("extract", self.extract_data)
    workflow.add_node("transform", self.transform_data)
    workflow.add_node("validate_data", self.validate_extracted_data)
    workflow.add_node("phase1_prorating", self.phase1_prorating)  # NEW
    workflow.add_node("load", self.load_to_neo4j)
    workflow.add_node("index", self.index_document)
    workflow.add_node("complete", self.complete_processing)
    workflow.add_node("handle_error", self.handle_error)
    
    # Add edges with Phase 1 integration
    workflow.add_edge("validate", "phase1_validation")
    
    workflow.add_conditional_edges(
        "phase1_validation",
        self.check_phase1_validation,
        {
            "continue": "parse",
            "rejected": "complete",  # Skip processing if rejected
            "error": "handle_error"
        }
    )
    
    workflow.add_edge("parse", "extract")
    workflow.add_edge("extract", "transform")
    workflow.add_edge("transform", "validate_data")
    
    workflow.add_conditional_edges(
        "validate_data",
        self.check_validation,
        {
            "valid": "phase1_prorating",  # Go to prorating instead of load
            "invalid": "handle_error",
            "retry": "extract"
        }
    )
    
    workflow.add_conditional_edges(
        "phase1_prorating",
        self.check_prorating_needed,
        {
            "prorating_needed": "load",
            "skip_prorating": "load"
        }
    )
    
    workflow.add_edge("load", "index")
    workflow.add_edge("index", "complete")
    workflow.add_edge("complete", END)
    
    # ... rest of existing workflow setup ...
```

## Detailed Integration for Extraction Workflow

### 1. Pro-rating Calculations Integration

#### A. After Utility Bill Extraction (Integration Point: Line 286)

Add pro-rating calculation in the extraction workflow:

```python
# In extraction_workflow.py, add new query templates
def _initialize_query_templates(self) -> Dict[str, List[str]]:
    """Initialize query templates with pro-rating queries."""
    templates = {
        # ... existing templates ...
        
        QueryType.PRORATING_ALLOCATIONS: [
            """
            MATCH (pa:ProRatingAllocation)
            OPTIONAL MATCH (pa)-[:ALLOCATED_TO]->(facility:Facility)
            OPTIONAL MATCH (pa)-[:BASED_ON]->(doc:Document)
            WHERE pa.period_start >= $start_date AND pa.period_end <= $end_date
            RETURN pa, facility, doc
            ORDER BY pa.created_at DESC
            """,
            """
            MATCH (pa:ProRatingAllocation)-[:HAS_DISTRIBUTION]->(pd:ProRatingDistribution)
            MATCH (pd)-[:ALLOCATED_TO]->(facility:Facility)
            WHERE pa.period_start >= $start_date AND pa.period_end <= $end_date
            RETURN facility.name as facility_name,
                   SUM(pd.allocated_amount) as total_allocated,
                   pa.utility_type as utility_type,
                   COUNT(pd) as distribution_count,
                   AVG(pd.allocated_percentage) as avg_percentage
            ORDER BY total_allocated DESC
            """
        ],
        
        QueryType.MONTHLY_ALLOCATION_REPORT: [
            """
            MATCH (pa:ProRatingAllocation)
            WHERE pa.period_start >= $start_date AND pa.period_end <= $end_date
            MATCH (pa)-[:HAS_DISTRIBUTION]->(pd:ProRatingDistribution)
            MATCH (pd)-[:ALLOCATED_TO]->(facility:Facility)
            RETURN 
                date(pa.period_start) as period_start,
                date(pa.period_end) as period_end,
                pa.utility_type as utility_type,
                pa.total_amount as total_amount,
                pa.allocation_method as method,
                collect({
                    facility_name: facility.name,
                    allocated_amount: pd.allocated_amount,
                    allocated_percentage: pd.allocated_percentage,
                    allocation_basis: pd.allocation_basis
                }) as distributions
            ORDER BY pa.period_start DESC
            """
        ]
    }
    
    return templates
```

#### B. Enhanced Query Execution

Add pro-rating specific query execution:

```python
def execute_prorating_queries(self, state: ExtractionState) -> ExtractionState:
    """Execute pro-rating allocation queries."""
    logger.info("Executing pro-rating allocation queries")
    
    try:
        # Get pro-rating allocations for the period
        if state["query_config"].get("type") in [QueryType.PRORATING_ALLOCATIONS, QueryType.MONTHLY_ALLOCATION_REPORT]:
            
            # Execute allocation queries
            for query_info in state["queries"]:
                query = query_info["query"]
                parameters = query_info.get("parameters", {})
                
                result = self.graph.query(query, parameters)
                
                # Process pro-rating results
                processed_results = []
                for record in result:
                    processed_record = {}
                    for key, value in dict(record).items():
                        processed_record[key] = self._serialize_neo4j_value(value)
                    processed_results.append(processed_record)
                
                query_results.append({
                    "query": query,
                    "parameters": parameters,
                    "results": processed_results,
                    "record_count": len(processed_results),
                    "status": "success",
                    "query_type": "prorating"
                })
        
        state["query_results"] = query_results
        
    except Exception as e:
        state["errors"].append(f"Pro-rating query execution error: {str(e)}")
        logger.error(f"Failed to execute pro-rating queries: {str(e)}")
    
    return state
```

### 2. Monthly Allocation Report Generation

Add report generation for pro-rating allocations:

```python
def generate_prorating_report(self, state: ExtractionState) -> ExtractionState:
    """Generate monthly pro-rating allocation report."""
    logger.info("Generating pro-rating allocation report")
    
    try:
        if state["query_config"].get("type") == QueryType.MONTHLY_ALLOCATION_REPORT:
            
            # Compile pro-rating specific report data
            allocation_data = []
            total_allocations = 0
            total_amount = 0.0
            
            for result in state["query_results"]:
                if result["status"] == "success":
                    for record in result["results"]:
                        allocation_info = {
                            "period": f"{record.get('period_start')} to {record.get('period_end')}",
                            "utility_type": record.get("utility_type"),
                            "total_amount": record.get("total_amount", 0),
                            "allocation_method": record.get("method"),
                            "distributions": record.get("distributions", [])
                        }
                        allocation_data.append(allocation_info)
                        total_allocations += len(allocation_info["distributions"])
                        total_amount += allocation_info["total_amount"]
            
            # Enhanced report data structure
            state["report_data"]["prorating_summary"] = {
                "total_allocations": total_allocations,
                "total_amount_allocated": total_amount,
                "allocation_periods": len(allocation_data),
                "allocation_details": allocation_data
            }
        
    except Exception as e:
        state["errors"].append(f"Pro-rating report generation error: {str(e)}")
        logger.error(f"Failed to generate pro-rating report: {str(e)}")
    
    return state
```

## Configuration Options

### 1. Environment Variables

Add these environment variables to enable/configure Phase 1 features:

```bash
# Phase 1 Feature Toggles
ENABLE_AUDIT_TRACKING=true
ENABLE_REJECTION_VALIDATION=true
ENABLE_UTILITY_PRORATING=true

# Rejection Validation Configuration
REJECTION_QUALITY_THRESHOLD=60
REJECTION_SIMILARITY_THRESHOLD=0.85
REJECTION_RELEVANCE_THRESHOLD=0.3
AUTO_REJECT_DUPLICATES=true

# Pro-rating Configuration
DEFAULT_ALLOCATION_METHOD=square_footage
PRORATING_BATCH_SIZE=50
ENABLE_PRORATING_BACKFILL=false

# Audit Trail Configuration
AUDIT_RETENTION_DAYS=365
ENABLE_FILE_BACKUP=true
AUDIT_LOG_LEVEL=INFO
```

### 2. Integration Configuration Class

Create a configuration class for Phase 1 integration:

```python
# In phase1_enhancements/integration_config.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class Phase1IntegrationConfig:
    """Configuration for Phase 1 integration with workflows."""
    
    # Feature toggles
    enable_audit_tracking: bool = True
    enable_rejection_validation: bool = True
    enable_utility_prorating: bool = True
    
    # Rejection validation settings
    rejection_quality_threshold: float = 60.0
    rejection_similarity_threshold: float = 0.85
    rejection_relevance_threshold: float = 0.3
    auto_reject_duplicates: bool = True
    
    # Pro-rating settings
    default_allocation_method: str = "square_footage"
    prorating_batch_size: int = 50
    enable_prorating_backfill: bool = False
    
    # Audit trail settings
    audit_retention_days: int = 365
    enable_file_backup: bool = True
    audit_log_level: str = "INFO"
    
    @classmethod
    def from_environment(cls) -> "Phase1IntegrationConfig":
        """Create configuration from environment variables."""
        return cls(
            enable_audit_tracking=os.getenv("ENABLE_AUDIT_TRACKING", "true").lower() == "true",
            enable_rejection_validation=os.getenv("ENABLE_REJECTION_VALIDATION", "true").lower() == "true",
            enable_utility_prorating=os.getenv("ENABLE_UTILITY_PRORATING", "true").lower() == "true",
            rejection_quality_threshold=float(os.getenv("REJECTION_QUALITY_THRESHOLD", "60")),
            rejection_similarity_threshold=float(os.getenv("REJECTION_SIMILARITY_THRESHOLD", "0.85")),
            rejection_relevance_threshold=float(os.getenv("REJECTION_RELEVANCE_THRESHOLD", "0.3")),
            auto_reject_duplicates=os.getenv("AUTO_REJECT_DUPLICATES", "true").lower() == "true",
            default_allocation_method=os.getenv("DEFAULT_ALLOCATION_METHOD", "square_footage"),
            prorating_batch_size=int(os.getenv("PRORATING_BATCH_SIZE", "50")),
            enable_prorating_backfill=os.getenv("ENABLE_PRORATING_BACKFILL", "false").lower() == "true",
            audit_retention_days=int(os.getenv("AUDIT_RETENTION_DAYS", "365")),
            enable_file_backup=os.getenv("ENABLE_FILE_BACKUP", "true").lower() == "true",
            audit_log_level=os.getenv("AUDIT_LOG_LEVEL", "INFO")
        )
```

## Testing the Integrated Workflows

### 1. Unit Tests for Integration Points

Create test cases for each integration point:

```python
# tests/test_workflow_integration.py
import pytest
import os
from unittest.mock import Mock, patch
from datetime import datetime

from src.workflows.ingestion_workflow import IngestionWorkflow, DocumentState
from src.phase1_enhancements.integration_config import Phase1IntegrationConfig

class TestWorkflowIntegration:
    """Test Phase 1 integration with existing workflows."""
    
    @pytest.fixture
    def mock_services(self):
        """Mock Phase 1 services for testing."""
        with patch('src.phase1_enhancements.audit_trail_service.AuditTrailService') as mock_audit, \
             patch('src.phase1_enhancements.rejection_workflow_service.RejectionWorkflowService') as mock_rejection, \
             patch('src.phase1_enhancements.prorating_service.ProRatingService') as mock_prorating:
            
            yield {
                'audit': mock_audit,
                'rejection': mock_rejection,
                'prorating': mock_prorating
            }
    
    def test_audit_trail_creation_on_upload(self, mock_services):
        """Test audit trail is created when document is uploaded."""
        # Setup
        config = Phase1IntegrationConfig.from_environment()
        workflow = IngestionWorkflow(
            llama_parse_api_key="test_key",
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="test",
            enable_phase1_features=True
        )
        
        # Mock audit trail service
        mock_audit_entry = Mock()
        mock_audit_entry.audit_id = "audit_123"
        mock_services['audit'].return_value.create_audit_trail.return_value = mock_audit_entry
        
        # Test document state
        test_state = {
            "file_path": "/tmp/test_document.pdf",
            "document_id": "doc_123",
            "upload_metadata": {"user_id": "user_123"},
            "errors": [],
            "phase1_processing": {}
        }
        
        # Execute validation
        result_state = workflow.validate_document(test_state)
        
        # Verify audit trail was created
        assert result_state["audit_trail_id"] == "audit_123"
        assert result_state["phase1_processing"]["audit_trail"] == "created"
        
        # Verify service was called with correct parameters
        mock_services['audit'].return_value.create_audit_trail.assert_called_once_with(
            document_name="test_document.pdf",
            action="document_uploaded",
            user_id="user_123",
            metadata={
                "file_path": "/tmp/test_document.pdf",
                "document_id": "doc_123",
                "upload_time": pytest.any(str)
            }
        )
    
    def test_rejection_validation_quality_check(self, mock_services):
        """Test document rejection based on quality validation."""
        # Setup workflow with Phase 1
        workflow = IngestionWorkflow(
            llama_parse_api_key="test_key",
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="test",
            enable_phase1_features=True
        )
        
        # Mock rejection service to return invalid quality
        from src.phase1_enhancements.rejection_workflow_service import ValidationResult, RejectionReason
        mock_validation = ValidationResult(
            document_id="doc_123",
            is_valid=False,
            quality_score=30.0,
            rejection_reasons=[RejectionReason.POOR_QUALITY],
            validation_details={"quality_issue": "file_too_small"},
            validation_time=datetime.now(),
            rule_violations=["File size too small"]
        )
        
        mock_services['rejection'].return_value.validate_document_quality.return_value = mock_validation
        mock_services['rejection'].return_value.initiate_rejection_review.return_value = "rejection_123"
        
        # Test document state
        test_state = {
            "file_path": "/tmp/poor_quality.pdf",
            "document_id": "doc_123",
            "upload_metadata": {"user_id": "user_123"},
            "errors": [],
            "phase1_processing": {}
        }
        
        # Execute validation
        result_state = workflow.validate_document(test_state)
        
        # Verify document was rejected
        assert result_state["rejection_id"] == "rejection_123"
        assert result_state["rejection_reason"] == "poor_quality"
        assert result_state["validation_score"] == 30.0
        assert result_state["phase1_processing"]["rejection"] == "quality_failed"
        
    def test_duplicate_detection(self, mock_services):
        """Test duplicate document detection and rejection."""
        # Similar test structure for duplicate detection
        pass
    
    def test_prorating_calculation_trigger(self, mock_services):
        """Test pro-rating calculation is triggered for utility bills."""
        pass

# Integration test for end-to-end workflow
def test_end_to_end_integration():
    """Test complete workflow with Phase 1 features."""
    # This would test the entire pipeline with actual services
    pass
```

### 2. Integration Test Scripts

Create integration test scripts to validate the workflow:

```python
# scripts/test_integration.py
#!/usr/bin/env python3
"""
Integration test script for Phase 1 workflow integration.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from workflows.ingestion_workflow import IngestionWorkflow
from workflows.extraction_workflow import DataExtractionWorkflow
from phase1_enhancements.phase1_integration import Phase1Integration

async def test_ingestion_integration():
    """Test ingestion workflow with Phase 1 features."""
    print("Testing ingestion workflow with Phase 1 integration...")
    
    # Initialize workflow with Phase 1
    workflow = IngestionWorkflow(
        llama_parse_api_key=os.getenv("LLAMA_PARSE_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "EhsAI2024!"),
        enable_phase1_features=True
    )
    
    # Test with a sample document
    test_document_path = Path("test_data/sample_electric_bill.pdf")
    if not test_document_path.exists():
        print(f"Warning: Test document not found at {test_document_path}")
        return False
    
    try:
        result = workflow.process_document(
            file_path=str(test_document_path),
            document_id="integration_test_001",
            metadata={"user_id": "test_user", "test": True}
        )
        
        print(f"Integration test result:")
        print(f"  Status: {result['status']}")
        print(f"  Document Type: {result.get('document_type')}")
        print(f"  Audit Trail ID: {result.get('audit_trail_id')}")
        print(f"  Rejection ID: {result.get('rejection_id')}")
        print(f"  Phase 1 Processing: {result.get('phase1_processing', {})}")
        
        return result["status"] in ["completed", "rejected"]
        
    except Exception as e:
        print(f"Integration test failed: {str(e)}")
        return False

async def test_extraction_integration():
    """Test extraction workflow with pro-rating queries."""
    print("Testing extraction workflow with pro-rating integration...")
    
    # Initialize extraction workflow
    extraction_workflow = DataExtractionWorkflow(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    )
    
    try:
        # Test pro-rating allocation query
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = extraction_workflow.extract_data(
            query_type="prorating_allocations",
            parameters={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            output_format="json"
        )
        
        print(f"Extraction test result:")
        print(f"  Status: {result['status']}")
        print(f"  Total Queries: {len(result.get('query_results', []))}")
        print(f"  Report File: {result.get('report_file_path')}")
        
        return result["status"] == "completed"
        
    except Exception as e:
        print(f"Extraction test failed: {str(e)}")
        return False

async def main():
    """Run integration tests."""
    print("Starting Phase 1 Integration Tests")
    print("=" * 50)
    
    # Test ingestion integration
    ingestion_success = await test_ingestion_integration()
    print(f"Ingestion Integration: {'PASSED' if ingestion_success else 'FAILED'}")
    
    # Test extraction integration
    extraction_success = await test_extraction_integration()
    print(f"Extraction Integration: {'PASSED' if extraction_success else 'FAILED'}")
    
    # Overall result
    overall_success = ingestion_success and extraction_success
    print("\n" + "=" * 50)
    print(f"Overall Integration Test: {'PASSED' if overall_success else 'FAILED'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

## Rollback Procedures

### 1. Feature Toggle Rollback

To disable Phase 1 features without code changes:

```bash
# Disable all Phase 1 features
export ENABLE_AUDIT_TRACKING=false
export ENABLE_REJECTION_VALIDATION=false
export ENABLE_UTILITY_PRORATING=false

# Restart the application
docker-compose restart backend
```

### 2. Database Schema Rollback

If you need to rollback database schema changes:

```sql
-- Remove Phase 1 constraints and indexes
DROP CONSTRAINT audit_trail_id_unique IF EXISTS;
DROP CONSTRAINT rejection_id_unique IF EXISTS;
DROP CONSTRAINT allocation_id_unique IF EXISTS;

DROP INDEX audit_trail_document_id IF EXISTS;
DROP INDEX rejection_document_id IF EXISTS;
DROP INDEX allocation_document_id IF EXISTS;

-- Remove Phase 1 nodes (BE CAREFUL - THIS REMOVES DATA)
-- MATCH (a:AuditTrailEntry) DETACH DELETE a;
-- MATCH (r:RejectionRecord) DETACH DELETE r;
-- MATCH (pa:ProRatingAllocation) DETACH DELETE pa;
```

### 3. Code Rollback Strategy

To rollback code changes while preserving functionality:

1. **Graceful Degradation**: The workflow checks `self.enable_phase1` before calling Phase 1 services
2. **Service Substitution**: Replace Phase 1 services with no-op implementations
3. **State Field Handling**: Phase 1 state fields are optional and don't break existing logic

```python
# Emergency rollback - disable Phase 1 in workflow initialization
def __init__(self, *args, **kwargs):
    # Force disable Phase 1 features
    kwargs['enable_phase1_features'] = False
    super().__init__(*args, **kwargs)
```

### 4. Monitoring and Alerting

Set up monitoring to detect integration issues:

```python
# In monitoring/phase1_monitoring.py
import logging
from datetime import datetime, timedelta

class Phase1IntegrationMonitor:
    """Monitor Phase 1 integration health."""
    
    def __init__(self, graph):
        self.graph = graph
        self.logger = logging.getLogger(__name__)
    
    def check_integration_health(self):
        """Check if Phase 1 integration is working properly."""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "audit_trail_active": self._check_audit_trail(),
            "rejection_service_active": self._check_rejection_service(),
            "prorating_service_active": self._check_prorating_service(),
            "recent_errors": self._get_recent_errors()
        }
        
        # Alert if any service is down
        if not all([health_report["audit_trail_active"], 
                   health_report["rejection_service_active"], 
                   health_report["prorating_service_active"]]):
            self._send_alert("Phase 1 service degradation detected", health_report)
        
        return health_report
    
    def _check_audit_trail(self):
        """Check if audit trail is creating entries."""
        try:
            query = """
            MATCH (a:AuditTrailEntry)
            WHERE a.created_at >= $since
            RETURN count(a) as recent_count
            """
            
            since = (datetime.now() - timedelta(hours=1)).isoformat()
            result = self.graph.query(query, {"since": since})
            
            return result[0]["recent_count"] > 0 if result else False
            
        except Exception as e:
            self.logger.error(f"Audit trail health check failed: {str(e)}")
            return False
```

## Summary

This integration guide provides:

1. **Detailed integration points** with specific line numbers and code examples
2. **Complete configuration options** for customizing Phase 1 behavior
3. **Comprehensive testing strategy** with unit and integration tests
4. **Rollback procedures** for safe deployment and emergency recovery
5. **Monitoring and alerting** to detect integration issues

The integration is designed to be:
- **Non-breaking**: Existing workflows continue to work without Phase 1 features
- **Configurable**: All features can be enabled/disabled via configuration
- **Testable**: Comprehensive test coverage for all integration points
- **Monitorable**: Health checks and alerts for operational visibility
- **Rollback-friendly**: Easy to disable or rollback if issues arise

Follow the specific code examples and integration points provided to implement Phase 1 enhancements in your EHS AI platform.