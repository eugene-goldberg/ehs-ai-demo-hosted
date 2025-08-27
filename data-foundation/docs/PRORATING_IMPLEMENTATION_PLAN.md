# Phase 1 Prorating Feature - Implementation Plan

## Overview
This document outlines a detailed plan to complete the Phase 1 Prorating feature implementation. Each task is designed to be self-contained, testable, and deliverable in small increments.

## Current Status Summary
- ✅ Core prorating service logic implemented
- ✅ API endpoints defined
- ❌ API routes not registered (404 errors)
- ❌ Workflow integration incomplete
- ❌ Testing blocked by API issues
- ❌ Documentation incomplete

## Implementation Tasks

### Task 1: Fix API Router Registration (Priority: CRITICAL)
**Goal**: Make prorating endpoints accessible via REST API

#### 1.1 Register Prorating Router in Main API
- **File**: `/data-foundation/backend/src/ehs_extraction_api.py`
- **Action**: Import and include the prorating router
- **Code Changes**:
  ```python
  from phase1_enhancements.phase1_integration import setup_phase1_features
  
  # In app initialization:
  setup_phase1_features(app, graph)
  ```
- **Test**: `curl http://localhost:8000/api/v1/prorating/health`
- **Success Criteria**: Returns 200 OK with health status

#### 1.2 Fix Double-Prefix Issue
- **File**: `/data-foundation/backend/src/phase1_enhancements/prorating_api.py`
- **Action**: Ensure router prefix doesn't duplicate `/api/v1`
- **Test**: Update test scripts to use correct URLs
- **Success Criteria**: Single prefix in URLs works correctly

#### 1.3 Verify Service Initialization
- **Action**: Ensure prorating service is initialized before first request
- **Test**: Check logs for initialization messages
- **Success Criteria**: No 503 Service Unavailable errors

**Deliverable**: Working health check endpoint
**Time Estimate**: 2 hours

---

### Task 2: Create Minimal Working Example (MWE)
**Goal**: Demonstrate end-to-end prorating functionality

#### 2.1 Create Test Document in Neo4j
- **Action**: Script to insert a test ProcessedDocument node
- **File**: `/data-foundation/backend/scripts/create_test_document.py`
- **Test Data**: Simple utility bill with known values
- **Success Criteria**: Document queryable in Neo4j

#### 2.2 Process Test Document via API
- **Action**: Call prorating endpoint with test document
- **Test**: Verify allocations created in Neo4j
- **Success Criteria**: MonthlyUsageAllocation nodes created

#### 2.3 Query Results
- **Action**: Retrieve allocations via API
- **Test**: Verify calculation correctness
- **Success Criteria**: Correct prorated values returned

**Deliverable**: Working prorating for single document
**Time Estimate**: 3 hours

---

### Task 3: Complete Workflow Integration
**Goal**: Automatic prorating during document processing

#### 3.1 Implement process_prorating Method
- **File**: `/data-foundation/backend/src/workflows/ingestion_workflow_enhanced.py`
- **Action**: Add actual prorating logic to workflow node
- **Code**:
  ```python
  async def process_prorating(self, state: WorkflowState) -> WorkflowState:
      if state.get("document_type") in ["utility_bill", "electricity_bill", "water_bill"]:
          # Extract billing period from state
          # Call prorating service
          # Update state with results
  ```
- **Test**: Process document through workflow
- **Success Criteria**: Prorating happens automatically

#### 3.2 Add Prorating Configuration
- **Action**: Add prorating settings to workflow config
- **Options**: Enable/disable, default method, facility mappings
- **Test**: Toggle prorating on/off
- **Success Criteria**: Configurable behavior

#### 3.3 Handle Edge Cases
- **Action**: Add error handling for missing data
- **Cases**: No billing period, no facility info, invalid amounts
- **Test**: Process documents with missing fields
- **Success Criteria**: Graceful degradation

**Deliverable**: Automatic prorating in ingestion workflow
**Time Estimate**: 4 hours

---

### Task 4: Implement Unit Tests
**Goal**: Comprehensive test coverage

#### 4.1 Test Prorating Calculator
- **File**: `/data-foundation/backend/tests/test_prorating_calculator.py`
- **Tests**:
  - Daily rate calculations
  - Monthly allocations across billing periods
  - Different allocation methods
  - Edge cases (partial months, leap years)
- **Success Criteria**: 100% calculator coverage

#### 4.2 Test Prorating Service
- **File**: `/data-foundation/backend/tests/test_prorating_service.py`
- **Tests**:
  - Document processing
  - Batch operations
  - Neo4j interactions (mocked)
  - Error scenarios
- **Success Criteria**: 90%+ service coverage

#### 4.3 Test API Endpoints
- **File**: `/data-foundation/backend/tests/test_prorating_api.py`
- **Tests**:
  - All endpoints with valid/invalid data
  - Authentication/authorization
  - Error responses
  - Pagination
- **Success Criteria**: All endpoints tested

**Deliverable**: Test suite with >85% coverage
**Time Estimate**: 4 hours

---

### Task 5: Integration Testing
**Goal**: End-to-end validation

#### 5.1 Create Integration Test Suite
- **File**: `/data-foundation/backend/tests/integration/test_prorating_integration.py`
- **Tests**:
  - Document upload → Processing → Prorating → Query
  - Multiple document types
  - Batch processing scenarios
- **Success Criteria**: Full pipeline tested

#### 5.2 Performance Testing
- **Action**: Test with large batches
- **Metrics**: Processing time, memory usage
- **Test**: 100, 1000, 10000 documents
- **Success Criteria**: Linear scaling

#### 5.3 Create Docker Test Environment
- **File**: `/data-foundation/backend/docker-compose.test.yml`
- **Components**: Neo4j, API, test runner
- **Test**: Run full suite in isolation
- **Success Criteria**: Tests pass in container

**Deliverable**: Automated integration tests
**Time Estimate**: 3 hours

---

### Task 6: Update Frontend Integration
**Goal**: Display prorating data in UI

#### 6.1 Add Prorating Display to Processed Documents
- **File**: `/data-foundation/web-app/frontend/src/components/DataManagement.js`
- **Action**: Show allocation summary in table
- **Fields**: Original amount, allocated amounts, method
- **Success Criteria**: Prorating visible in UI

#### 6.2 Create Prorating Details View
- **Action**: Modal/page showing all allocations
- **Features**: Monthly breakdown, facility allocation
- **Test**: Click through from main table
- **Success Criteria**: Detailed view works

#### 6.3 Add Prorating Reports
- **Action**: Monthly summary report component
- **Features**: Charts, export to CSV
- **Test**: Generate report for various periods
- **Success Criteria**: Accurate reports generated

**Deliverable**: Frontend prorating features
**Time Estimate**: 5 hours

---

### Task 7: Documentation
**Goal**: Complete user and developer documentation

#### 7.1 API Documentation
- **File**: `/data-foundation/backend/docs/PRORATING_API.md`
- **Content**:
  - All endpoints with examples
  - Request/response schemas
  - Error codes
  - Usage scenarios

#### 7.2 User Guide
- **File**: `/data-foundation/docs/PRORATING_USER_GUIDE.md`
- **Content**:
  - What is prorating and why use it
  - How to enable/configure
  - Understanding results
  - Troubleshooting

#### 7.3 Developer Guide
- **File**: `/data-foundation/backend/docs/PRORATING_DEVELOPER_GUIDE.md`
- **Content**:
  - Architecture overview
  - Adding new allocation methods
  - Extending functionality
  - Database schema

**Deliverable**: Complete documentation
**Time Estimate**: 3 hours

---

### Task 8: Migration and Deployment
**Goal**: Production readiness

#### 8.1 Create Migration Script
- **File**: `/data-foundation/backend/scripts/migrate_existing_documents.py`
- **Action**: Backfill prorating for existing documents
- **Features**: Progress tracking, rollback capability
- **Test**: Run on copy of production data
- **Success Criteria**: All documents processed

#### 8.2 Add Feature Flags
- **Action**: Runtime enable/disable of prorating
- **Config**: Environment variables, database flags
- **Test**: Toggle without restart
- **Success Criteria**: Dynamic control

#### 8.3 Monitoring and Alerts
- **Action**: Add metrics and logging
- **Metrics**: Processing time, success rate, errors
- **Alerts**: Failed allocations, performance degradation
- **Success Criteria**: Operational visibility

**Deliverable**: Production-ready feature
**Time Estimate**: 4 hours

---

## Testing Strategy

### For Each Task:
1. **Unit Test First**: Write tests before implementation
2. **Manual Verification**: Test via curl/Postman
3. **Automated Test**: Add to test suite
4. **Documentation**: Update as you go

### Test Data Sets:
- Single facility, full month
- Multiple facilities, partial month
- Cross-month billing period
- Cross-year billing period
- Zero amounts
- Missing data scenarios

## Success Metrics

### Technical Metrics:
- All API endpoints return correct responses
- 85%+ test coverage
- No performance regression
- Zero critical bugs

### Business Metrics:
- Accurate monthly allocations
- Audit trail maintained
- Reports match manual calculations
- User acceptance achieved

## Risk Mitigation

### Risks:
1. **Neo4j Schema Changes**: May break existing queries
   - Mitigation: Comprehensive testing, gradual rollout

2. **Performance Impact**: Prorating adds processing time
   - Mitigation: Async processing, caching

3. **Data Quality**: Bad input data causes incorrect allocations
   - Mitigation: Validation, error handling, audit logs

## Timeline

### Week 1:
- Task 1: Fix API Registration (Day 1)
- Task 2: Create MWE (Day 1-2)
- Task 3: Workflow Integration (Day 2-3)
- Task 4: Unit Tests (Day 3-4)

### Week 2:
- Task 5: Integration Testing (Day 5)
- Task 6: Frontend Integration (Day 6-7)
- Task 7: Documentation (Day 8)
- Task 8: Migration/Deployment (Day 9-10)

### Buffer: 2 days for unexpected issues

## Definition of Done

Each task is complete when:
1. Code is implemented and working
2. Tests are written and passing
3. Documentation is updated
4. Code is reviewed and merged
5. Feature is verified in staging environment

## Next Steps

1. Start with Task 1.1 - Fix API Router Registration
2. Create tracking issue in GitHub
3. Set up daily progress check-ins
4. Maintain implementation log

---

This plan ensures systematic completion of the prorating feature with minimal risk and maximum visibility into progress.