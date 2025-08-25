# Document Rejection Tracking Integration Plan

> Created: 2024-08-24
> Version: 1.0.0
> Status: Implementation Ready

## Executive Summary

This document outlines the detailed integration plan for adding Document Rejection Tracking to the existing, fully functional ingestion workflow in the EHS AI Demo backend. The plan ensures zero disruption to current operations while adding comprehensive rejection capabilities that identify and track documents that don't match the three supported types: electricity bills, water bills, and waste manifests.

## 1. Current Workflow Analysis

### Existing Ingestion Pipeline Stages

The current workflow processes documents through these stages:

1. **Validation** (`validate_document` - Line 182)
   - File existence check
   - File size limits (50MB)
   - Document type detection via filename patterns

2. **Document Type Detection** (`parser.detect_document_type` - Line 206)
   - Based on filename patterns in LlamaParse
   - Returns: `utility_bill`, `water_bill`, `waste_manifest`, `permit`, `invoice`

3. **Parsing** (`parse_document` - Line 217)
   - LlamaParse document parsing
   - Text extraction with metadata

4. **Data Extraction** (`extract_data` - Line 262)
   - LLM-based structured data extraction
   - Uses specialized extractors per document type

5. **Transformation** (`transform_data` - Line 302)
   - Converts extracted data to Neo4j graph structure
   - Creates nodes and relationships

6. **Data Validation** (`validate_extracted_data` - Line 1014)
   - Validates required fields per document type
   - Quality checks and data range validation

7. **Loading** (`load_to_neo4j` - Line 1114)
   - Persists data to Neo4j database
   - Creates nodes and relationships

### Existing Phase 1 Rejection Components

The system already includes:
- `rejection_workflow_service.py` - Comprehensive rejection workflow management
- `rejection_tracking_schema.py` - Database schema for rejection tracking
- `rejection_tracking_api.py` - REST API endpoints for rejection operations
- Integration hooks in existing workflows via `WORKFLOW_INTEGRATION_GUIDE.md`

## 2. Integration Points in Existing Workflow

### 2.1 Pre-Ingestion Document Recognition (NEW)

**Integration Point**: Before `validate_document` (Line 182)
**Purpose**: Positive identification of supported document types

**New Node**: `recognize_document_type`
**Location**: Insert between workflow entry point and validation

### 2.2 Enhanced Document Type Detection (MODIFIED)

**Integration Point**: Within `validate_document` (Line 206)
**Purpose**: Replace basic filename detection with comprehensive recognition
**Modification**: Enhance `parser.detect_document_type` with content-based analysis

### 2.3 Rejection Decision Point (NEW)

**Integration Point**: After document recognition, before parsing
**Purpose**: Decide whether to proceed with ingestion or reject
**New Conditional Edge**: `check_document_acceptance`

### 2.4 Rejection Handling Workflow (NEW)

**Integration Point**: Parallel to main ingestion workflow
**Purpose**: Handle rejected documents separately
**New Nodes**: `handle_rejection`, `store_rejection`

## 3. New Components to be Created

### 3.1 Document Recognition Service

**File**: `src/recognition/document_recognition_service.py`

**Purpose**: Advanced document type recognition using multiple analysis methods

**Key Methods**:
- `analyze_document_type(file_path: str) -> DocumentTypeResult`
- `extract_document_features(content: str) -> DocumentFeatures`
- `classify_with_confidence(features: DocumentFeatures) -> ClassificationResult`
- `validate_document_structure(content: str, doc_type: str) -> bool`

**Recognition Methods**:
1. **Content-based Analysis**: LLM analysis of document structure and keywords
2. **Template Matching**: Compare against known document templates
3. **Metadata Analysis**: Extract and analyze document metadata
4. **Feature Extraction**: Identify key document characteristics

### 3.2 Enhanced Document Type Detector

**File**: `src/parsers/enhanced_document_detector.py`

**Purpose**: Replace basic filename-based detection with comprehensive analysis

**Integration**: Extends existing `EHSDocumentParser.detect_document_type`

**Key Methods**:
- `detect_with_confidence(file_path: str) -> TypeDetectionResult`
- `analyze_content_structure(parsed_content: List[str]) -> StructureAnalysis`
- `validate_document_compatibility(doc_type: str, content: str) -> bool`

### 3.3 Rejection Decision Engine

**File**: `src/rejection/rejection_decision_engine.py`

**Purpose**: Centralized decision making for document acceptance/rejection

**Key Methods**:
- `evaluate_document(recognition_result: DocumentTypeResult) -> RejectionDecision`
- `apply_rejection_rules(doc_type: str, confidence: float) -> bool`
- `generate_rejection_reason(analysis: DocumentAnalysis) -> str`

**Decision Criteria**:
- Document type confidence threshold (e.g., >85%)
- Supported document type validation
- Content quality assessment
- Duplicate detection

### 3.4 Rejected Document Storage Service

**File**: `src/storage/rejected_document_service.py`

**Purpose**: Separate storage system for rejected documents

**Key Methods**:
- `store_rejected_document(file_path: str, metadata: Dict) -> str`
- `create_rejection_record(document_id: str, reason: str) -> RejectionRecord`
- `link_rejection_to_audit_trail(rejection_id: str, audit_id: str)`

**Storage Strategy**:
- Physical storage in separate directory (`/tmp/rejected_documents/`)
- Metadata storage in rejection tracking database
- No Neo4j storage for rejected documents

### 3.5 Enhanced API Endpoints

**File**: `src/api/rejection_endpoints.py`

**Purpose**: REST API endpoints for rejection management

**New Endpoints**:
- `GET /api/v1/rejected-documents` - List rejected documents
- `GET /api/v1/rejected-documents/{id}` - Get rejection details
- `POST /api/v1/rejected-documents/{id}/re-evaluate` - Re-evaluate rejection
- `GET /api/v1/rejection-statistics` - Get rejection statistics

## 4. Files That Need Modification

### 4.1 Core Workflow Files

#### `src/workflows/ingestion_workflow.py`

**Modifications**:

**Line 33-57**: Enhance `DocumentState` TypedDict
```python
class DocumentState(TypedDict):
    # Existing fields...
    file_path: str
    document_id: str
    # ... existing fields ...
    
    # NEW: Rejection tracking fields
    recognition_result: Optional[Dict[str, Any]]
    rejection_decision: Optional[Dict[str, Any]]
    is_rejected: bool
    rejection_id: Optional[str]
    rejection_reason: Optional[str]
    rejection_confidence: Optional[float]
```

**Line 73-123**: Modify `__init__` method
```python
def __init__(self, ...):
    # Existing initialization...
    
    # NEW: Initialize rejection services
    self.document_recognition_service = DocumentRecognitionService(llm_model)
    self.rejection_decision_engine = RejectionDecisionEngine()
    self.rejected_document_service = RejectedDocumentService(
        storage_path="/tmp/rejected_documents"
    )
```

**Line 124-181**: Update `_build_workflow` method
```python
def _build_workflow(self) -> StateGraph:
    # Add new nodes
    workflow.add_node("recognize_document", self.recognize_document_type)
    workflow.add_node("decide_acceptance", self.decide_document_acceptance)
    workflow.add_node("handle_rejection", self.handle_document_rejection)
    
    # Modify edges
    workflow.add_edge("recognize_document", "decide_acceptance")
    workflow.add_conditional_edges(
        "decide_acceptance",
        self.check_document_acceptance,
        {
            "accept": "validate",
            "reject": "handle_rejection"
        }
    )
    workflow.add_edge("handle_rejection", END)
    
    # Update entry point
    workflow.set_entry_point("recognize_document")
```

**NEW Methods**:
```python
def recognize_document_type(self, state: DocumentState) -> DocumentState:
    """Comprehensive document type recognition."""
    
def decide_document_acceptance(self, state: DocumentState) -> DocumentState:
    """Decide whether to accept or reject the document."""
    
def handle_document_rejection(self, state: DocumentState) -> DocumentState:
    """Handle rejected documents."""
    
def check_document_acceptance(self, state: DocumentState) -> str:
    """Conditional edge function for acceptance decision."""
```

#### `src/parsers/llama_parser.py`

**Modifications**:

**Line 142-180**: Enhance `detect_document_type` method
```python
def detect_document_type(self, file_path: str) -> str:
    """Enhanced document type detection with content analysis."""
    
    # Keep existing filename-based detection as fallback
    filename_type = self._detect_by_filename(file_path)
    
    # NEW: Add content-based detection
    try:
        content = self._extract_sample_content(file_path)
        content_type = self._detect_by_content(content)
        
        # Combine results with confidence scoring
        return self._reconcile_detection_results(filename_type, content_type)
    except Exception as e:
        self.logger.warning(f"Content-based detection failed: {e}")
        return filename_type
```

### 4.2 API Files

#### `src/ehs_extraction_api.py`

**Modifications**:

**Line 24**: Add rejection endpoint imports
```python
from rejection.rejection_endpoints import rejection_router
```

**Line 48-60**: Add rejection router
```python
app.include_router(rejection_router, prefix="/api/v1")
```

**NEW**: Health check enhancement to include rejection service status

## 5. Database Schema for Rejection Tracking

### 5.1 New Node Types

#### RejectedDocument
```cypher
CREATE CONSTRAINT rejected_document_id IF NOT EXISTS 
FOR (rd:RejectedDocument) REQUIRE rd.id IS UNIQUE;

// Properties:
// - id: Unique document identifier
// - original_filename: Original file name
// - file_path: Path to rejected document storage
// - rejection_reason: Reason for rejection
// - rejection_confidence: Confidence score (0.0-1.0)
// - rejected_at: Timestamp of rejection
// - document_size: File size in bytes
// - attempted_type: What type was attempted to detect
// - content_preview: First 500 characters of content
```

#### RejectionRule
```cypher
CREATE CONSTRAINT rejection_rule_id IF NOT EXISTS 
FOR (rr:RejectionRule) REQUIRE rr.id IS UNIQUE;

// Properties:
// - id: Rule identifier
// - rule_type: Type of rejection rule
// - threshold: Confidence threshold
// - enabled: Whether rule is active
// - created_at: Rule creation timestamp
```

#### DocumentRecognition
```cypher
CREATE CONSTRAINT recognition_id IF NOT EXISTS 
FOR (dr:DocumentRecognition) REQUIRE dr.id IS UNIQUE;

// Properties:
// - id: Recognition process identifier
// - document_id: Related document ID
// - recognition_method: Method used for recognition
// - confidence_score: Recognition confidence (0.0-1.0)
// - detected_type: Detected document type
// - processing_time: Time taken for recognition
// - created_at: Recognition timestamp
```

### 5.2 New Relationship Types

```cypher
// RejectedDocument -> RejectionRule
(:RejectedDocument)-[:REJECTED_BY]->(:RejectionRule)

// RejectedDocument -> DocumentRecognition  
(:RejectedDocument)-[:ANALYZED_BY]->(:DocumentRecognition)

// RejectedDocument -> AuditTrailEntry (if Phase 1 enabled)
(:RejectedDocument)-[:TRACKED_BY]->(:AuditTrailEntry)
```

### 5.3 Indexes for Performance

```cypher
CREATE INDEX rejected_document_timestamp IF NOT EXISTS 
FOR (rd:RejectedDocument) ON (rd.rejected_at);

CREATE INDEX rejected_document_reason IF NOT EXISTS 
FOR (rd:RejectedDocument) ON (rd.rejection_reason);

CREATE INDEX recognition_confidence IF NOT EXISTS 
FOR (dr:DocumentRecognition) ON (dr.confidence_score);
```

## 6. Step-by-Step Implementation Approach

### Phase 1: Foundation Components (Days 1-2)

1. **Create Document Recognition Service**
   - Implement `DocumentRecognitionService` class
   - Add LLM-based content analysis
   - Create confidence scoring system
   - Add template matching capabilities

2. **Create Rejection Decision Engine**
   - Implement decision logic
   - Configure rejection thresholds
   - Add rule-based rejection criteria
   - Create rejection reason generation

3. **Set up Rejected Document Storage**
   - Create storage directory structure
   - Implement `RejectedDocumentService`
   - Add metadata storage capabilities
   - Create file organization system

### Phase 2: Database Integration (Days 3-4)

4. **Update Database Schema**
   - Run migration to add new node types
   - Create constraints and indexes
   - Update existing schema if needed
   - Test schema changes

5. **Integrate with Existing Rejection Tracking**
   - Connect new components with existing `rejection_workflow_service`
   - Update `rejection_tracking_schema`
   - Enhance `rejection_tracking_api`
   - Test integration points

### Phase 3: Workflow Integration (Days 5-6)

6. **Modify Ingestion Workflow**
   - Update `DocumentState` structure
   - Add new workflow nodes
   - Modify workflow edges and conditions
   - Update initialization methods

7. **Enhance Document Parser**
   - Update `detect_document_type` method
   - Add content-based detection
   - Implement confidence scoring
   - Add fallback mechanisms

### Phase 4: API Enhancement (Day 7)

8. **Add New API Endpoints**
   - Create rejection management endpoints
   - Add rejection statistics endpoints
   - Update health check endpoints
   - Add rejection re-evaluation capabilities

9. **Update Existing Endpoints**
   - Modify ingestion endpoints to handle rejections
   - Add rejection information to responses
   - Update error handling for rejections
   - Enhance response models

### Phase 5: Testing and Validation (Days 8-9)

10. **Comprehensive Testing**
    - Unit tests for all new components
    - Integration tests for workflow changes
    - End-to-end tests for complete flow
    - Performance testing for rejection detection

11. **Documentation and Deployment**
    - Update API documentation
    - Create integration guide updates
    - Prepare deployment scripts
    - Create monitoring dashboards

## 7. Zero Disruption Strategy

### 7.1 Backward Compatibility

**Existing Workflow Preservation**:
- All existing functionality remains unchanged
- New rejection logic is optional and configurable
- Existing document processing continues normally
- No breaking changes to existing APIs

**Configuration-Based Activation**:
```python
class IngestionWorkflow:
    def __init__(self, ..., enable_rejection_tracking: bool = True):
        self.rejection_enabled = enable_rejection_tracking
        if self.rejection_enabled:
            # Initialize rejection components
        else:
            # Use existing workflow without changes
```

### 7.2 Gradual Rollout Strategy

**Phase 1**: Shadow Mode
- Run rejection detection in parallel
- Log results without affecting workflow
- Collect performance metrics
- Validate accuracy

**Phase 2**: Selective Activation
- Enable rejection for specific document types
- Monitor impact on processing time
- Gradually expand coverage
- Fine-tune thresholds

**Phase 3**: Full Activation
- Enable for all document types
- Monitor rejection rates
- Adjust rules based on feedback
- Full production deployment

### 7.3 Fallback Mechanisms

**Detection Failure Handling**:
- If recognition service fails, use existing filename detection
- If rejection service fails, default to acceptance
- Log all failures for investigation
- Maintain processing speed

**Database Failure Handling**:
- If rejection database fails, store in temporary files
- Queue rejections for later processing
- Continue main workflow without interruption
- Alert administrators of issues

## 8. Testing Strategy

### 8.1 Unit Testing

**New Component Tests**:
- `test_document_recognition_service.py`
- `test_rejection_decision_engine.py`
- `test_rejected_document_service.py`
- `test_enhanced_document_detector.py`

**Test Coverage Requirements**:
- Minimum 90% code coverage for new components
- Edge case testing for all rejection scenarios
- Error handling and recovery testing
- Performance benchmarking

### 8.2 Integration Testing

**Workflow Integration Tests**:
- `test_ingestion_workflow_with_rejection.py`
- `test_rejection_workflow_integration.py`
- `test_api_endpoints_rejection.py`

**Test Scenarios**:
- Supported document type acceptance
- Unsupported document type rejection
- Edge cases (corrupted files, empty documents)
- Bulk processing with mixed document types
- Rejection re-evaluation workflows

### 8.3 End-to-End Testing

**Complete Workflow Tests**:
- Upload supported documents → successful processing
- Upload unsupported documents → proper rejection
- Mixed batch processing → correct handling
- API interactions → proper responses
- Database consistency → data integrity

**Performance Testing**:
- Processing time impact measurement
- Memory usage analysis
- Database query performance
- API response time validation
- Concurrent processing testing

### 8.4 Regression Testing

**Existing Functionality Validation**:
- All existing tests must continue to pass
- No performance degradation for existing workflows
- Existing API contracts maintained
- Database schema backward compatibility
- Integration with Phase 1 features preserved

## 9. Monitoring and Alerting

### 9.1 Key Metrics

**Rejection Metrics**:
- Rejection rate by document type
- Recognition confidence distribution
- Processing time impact
- False positive/negative rates
- Storage utilization for rejected documents

**Performance Metrics**:
- Average recognition time
- Peak processing time impact
- Database query performance
- API response times
- Error rates and recovery times

### 9.2 Alerting Rules

**High Priority Alerts**:
- Rejection rate suddenly increases >20%
- Recognition service failures >5 in 10 minutes
- Database connection failures
- Storage capacity approaching limits
- Processing time increases >50%

**Medium Priority Alerts**:
- Unusual document types detected
- Confidence scores dropping below thresholds
- Rejected document storage growing rapidly
- API error rates increasing
- Integration test failures

### 9.3 Dashboard Components

**Rejection Dashboard**:
- Real-time rejection statistics
- Document type distribution
- Recognition confidence trends
- Storage utilization graphs
- Processing time comparisons

**Health Dashboard**:
- Service availability status
- Database connection health
- API endpoint response times
- Error rate trends
- System resource utilization

## 10. Risk Assessment and Mitigation

### 10.1 Technical Risks

**Risk**: Document Recognition Accuracy
- **Impact**: High false rejections or false acceptances
- **Mitigation**: Multiple recognition methods, confidence thresholds, human review process

**Risk**: Performance Impact
- **Impact**: Slower document processing
- **Mitigation**: Efficient algorithms, caching, parallel processing, performance monitoring

**Risk**: Storage Growth
- **Impact**: Rejected documents consuming excessive disk space
- **Mitigation**: Automated cleanup policies, compression, archive strategies

### 10.2 Integration Risks

**Risk**: Workflow Disruption
- **Impact**: Breaking existing document processing
- **Mitigation**: Comprehensive testing, gradual rollout, fallback mechanisms

**Risk**: Database Schema Conflicts
- **Impact**: Data integrity issues
- **Mitigation**: Careful schema design, migration testing, backup strategies

**Risk**: API Breaking Changes
- **Impact**: Client applications failing
- **Mitigation**: Backward compatibility, API versioning, deprecation notices

### 10.3 Operational Risks

**Risk**: False Rejection of Valid Documents
- **Impact**: Loss of important data
- **Mitigation**: Review processes, re-evaluation capabilities, audit trails

**Risk**: Security Vulnerabilities
- **Impact**: Data exposure or system compromise
- **Mitigation**: Security reviews, access controls, encryption, monitoring

**Risk**: Capacity Planning
- **Impact**: System overload during high volume
- **Mitigation**: Load testing, scalability planning, resource monitoring

## 11. Success Metrics

### 11.1 Functional Metrics

- **Rejection Accuracy**: >95% correct rejection decisions
- **Processing Speed**: <10% impact on existing workflow performance
- **Zero False Negatives**: No valid documents incorrectly rejected
- **Complete Coverage**: All unsupported document types properly rejected

### 11.2 Technical Metrics

- **System Availability**: >99.9% uptime
- **Response Time**: <2 seconds for rejection decisions
- **Database Performance**: <100ms average query time
- **Storage Efficiency**: <1GB growth per 1000 rejections

### 11.3 Business Metrics

- **Data Quality**: Improved data quality by filtering out irrelevant documents
- **Processing Efficiency**: Reduced manual review time
- **Compliance**: Full audit trail for all document decisions
- **User Satisfaction**: Positive feedback on rejection handling

## 12. Implementation Timeline

### Week 1: Foundation
- Days 1-2: Document Recognition Service
- Days 3-4: Rejection Decision Engine
- Day 5: Rejected Document Storage

### Week 2: Integration
- Days 6-7: Database Schema Updates
- Days 8-9: Workflow Integration
- Day 10: API Enhancements

### Week 3: Testing & Deployment
- Days 11-12: Comprehensive Testing
- Days 13-14: Performance Optimization
- Day 15: Production Deployment

## 13. Conclusion

This integration plan provides a comprehensive roadmap for adding Document Rejection Tracking to the existing ingestion workflow while ensuring zero disruption to current operations. The approach emphasizes backward compatibility, gradual rollout, and comprehensive testing to minimize risks while maximizing the benefits of automated document rejection.

The plan leverages existing Phase 1 rejection tracking infrastructure and enhances it with advanced document recognition capabilities. By following this systematic approach, the system will be able to accurately identify and track documents that don't match the three supported types (electricity bills, water bills, waste manifests) while maintaining full audit trails and providing management visibility into rejection patterns.

Key success factors include thorough testing, performance monitoring, and maintaining the ability to fall back to existing behavior if issues arise. The implementation timeline allows for careful integration and validation at each step, ensuring a smooth transition to the enhanced system.

## 14. Comprehensive Testing Strategy

### 14.1 Test-Driven Development Approach

Following the strict requirements in CLAUDE.md:
- **Create tests BEFORE implementing each increment**
- **Run ALL tests after EVERY change** 
- **100% test coverage requirement - no exceptions**
- **NO mocks or hardcoded values allowed**
- **Every feature must be fully tested before marking as complete**

### 14.2 Test Files to Create

All tests must be placed in the ../test subdirectory as per CLAUDE.md rules:

#### Core Service Tests
- `/backend/test/test_document_recognition_service.py`
  - Test electricity bill recognition
  - Test water bill recognition  
  - Test waste manifest recognition
  - Test unrecognized document rejection
  - Test confidence scoring accuracy
  - Test edge cases (corrupted files, empty documents)

- `/backend/test/test_rejection_decision_engine.py`
  - Test rejection rules application
  - Test confidence threshold validation
  - Test rejection reason generation
  - Test duplicate detection
  - Test decision consistency

- `/backend/test/test_rejected_document_service.py`
  - Test document storage
  - Test metadata persistence
  - Test audit trail integration
  - Test storage organization
  - Test cleanup policies

#### Integration Tests
- `/backend/test/test_rejection_tracking_api.py`
  - Test all rejection endpoints
  - Test error handling
  - Test response validation
  - Test pagination
  - Test statistics generation

- `/backend/test/test_ingestion_workflow_with_rejection.py`
  - Test complete workflow with rejection
  - Test conditional routing
  - Test state management
  - Test error recovery
  - Test performance impact

- `/backend/test/test_rejection_integration_e2e.py`
  - Test full end-to-end scenarios
  - Test batch processing
  - Test concurrent processing
  - Test system resilience
  - Test data integrity

### 14.3 Regression Testing Strategy

**Critical Existing Tests to Run After Each Change:**

1. **API Tests**
   - `backend/tests/test_ehs_extraction_api.py` - All 831 lines must pass
   - `backend/src/test_api/comprehensive_test_api.py`
   - All shell scripts in `backend/`: 
     - `test_phase1_all_features.sh`
     - `final_comprehensive_test.sh`

2. **Pipeline Tests**
   - `scripts/test_document_pipeline.py`
   - `scripts/test_complete_pipeline.py`
   - `scripts/ingest_all_documents.py`

3. **Document-Specific Tests**
   - `scripts/test_waste_manifest_ingestion.py`
   - `scripts/test_water_bill_ingestion.py`
   - `scripts/test_extraction_workflow.py`

### 14.4 Test Data Requirements

**Real PDF Samples Required:**
- 5 valid electricity bills (different formats/providers)
- 5 valid water bills (different formats/providers)
- 5 valid waste manifests (different formats)
- 10 unrecognized documents:
  - Gas bills
  - Phone bills
  - General invoices
  - Permits
  - Random PDFs
- 5 edge case documents:
  - Corrupted PDFs
  - Password-protected PDFs
  - Scanned images
  - Empty files
  - Extremely large files (>40MB)

**No Hardcoded Test Data:**
- All test data must be real documents
- Store test documents in `/backend/test/test_documents/`
- Use environment variables for test data paths
- No synthetic or mocked data generation

### 14.5 Test Execution Plan

#### Background Execution with Logging
```bash
# Run tests in background with detailed logging
cd /backend
python -m pytest test/test_document_recognition_service.py -v --log-cli-level=DEBUG > test_logs/recognition_$(date +%Y%m%d_%H%M%S).log 2>&1 &
TEST_PID=$!

# Monitor test progress
tail -f test_logs/recognition_*.log

# Check test status
ps -p $TEST_PID
```

#### Continuous Test Execution
```bash
# Create test runner script
cat > run_all_rejection_tests.sh << 'EOF'
#!/bin/bash
# Run all rejection tests with logging

TEST_DIR="/tmp/rejection_test_logs"
mkdir -p $TEST_DIR

# Run each test in background
python -m pytest test/test_document_recognition_service.py -v > $TEST_DIR/recognition.log 2>&1 &
python -m pytest test/test_rejection_decision_engine.py -v > $TEST_DIR/decision.log 2>&1 &
python -m pytest test/test_rejected_document_service.py -v > $TEST_DIR/storage.log 2>&1 &
python -m pytest test/test_rejection_tracking_api.py -v > $TEST_DIR/api.log 2>&1 &
python -m pytest test/test_ingestion_workflow_with_rejection.py -v > $TEST_DIR/workflow.log 2>&1 &
python -m pytest test/test_rejection_integration_e2e.py -v > $TEST_DIR/e2e.log 2>&1 &

# Monitor all logs
tail -f $TEST_DIR/*.log
EOF

chmod +x run_all_rejection_tests.sh
```

#### Regression Test Execution
```bash
# Run all existing tests to ensure no breakage
./run_tests.py --test-type all --coverage

# Run specific critical tests
python -m pytest tests/test_ehs_extraction_api.py -v
python scripts/ingest_all_documents.py
./test_phase1_all_features.sh
```

### 14.6 Success Criteria

**All Tests Must Meet These Criteria:**

1. **Existing Test Integrity**
   - ALL existing tests continue to pass (100% success rate)
   - No performance degradation (processing time within 5% of baseline)
   - No memory leaks or resource issues

2. **New Test Coverage**
   - 100% code coverage for all new components
   - All edge cases covered with specific tests
   - All error scenarios tested and handled
   - Performance benchmarks established and maintained

3. **Data Integrity**
   - ZERO rejected documents in Neo4j database
   - All rejected documents properly stored separately
   - Complete audit trail for every rejection
   - No data loss or corruption

4. **Performance Requirements**
   - Document recognition < 2 seconds per document
   - Rejection decision < 100ms
   - No impact on existing ingestion performance
   - API response times < 500ms

### 14.7 Test Development Timeline

**Week 1 - Test Infrastructure**
- Day 1: Create test directory structure and test data collection
- Day 2: Develop test_document_recognition_service.py
- Day 3: Develop test_rejection_decision_engine.py
- Day 4: Develop test_rejected_document_service.py
- Day 5: Run initial test suite, fix any issues

**Week 2 - Integration Testing**
- Day 6: Develop test_rejection_tracking_api.py
- Day 7: Develop test_ingestion_workflow_with_rejection.py
- Day 8: Develop test_rejection_integration_e2e.py
- Day 9: Full regression testing
- Day 10: Performance testing and optimization

**Week 3 - Validation and Deployment**
- Day 11: Run complete test suite continuously for 24 hours
- Day 12: Fix any issues found during extended testing
- Day 13: Final regression testing
- Day 14: Documentation of test results
- Day 15: Production deployment with monitoring

### 14.8 Test Maintenance

**Continuous Testing Requirements:**
- Run full test suite after EVERY code change
- Update tests when requirements change
- Add new tests for any bugs found in production
- Maintain test documentation in TEST_INVENTORY.md
- Regular test performance optimization

**Test Review Checklist:**
- [ ] All tests pass with 100% success rate
- [ ] No hardcoded values or mocks used
- [ ] Real test data properly managed
- [ ] Tests run in background with logging
- [ ] Regression tests confirm no breakage
- [ ] Performance benchmarks maintained
- [ ] Test coverage reports generated
- [ ] All edge cases covered

This comprehensive testing strategy ensures that the Document Rejection Tracking feature is thoroughly tested at every stage, maintaining the integrity of the existing system while adding new capabilities with complete confidence.