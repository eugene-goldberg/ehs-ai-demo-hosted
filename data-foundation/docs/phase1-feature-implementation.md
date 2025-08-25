# Phase 1 Feature Implementation Tasks

> Document Type: Implementation Guide  
> Created: 2025-08-23  
> Status: Ready for Development  
> Scope: Feature Development Only (No Infrastructure/Hosting)

This document outlines the detailed implementation tasks for the three core Phase 1 enhancements, focusing exclusively on feature development. Infrastructure, Docker, and hosting tasks are covered in separate documentation.

## Overview

Phase 1 introduces three key enhancements to the EHS AI Demo system:

1. **Audit Trail Enhancement** - Comprehensive file tracking and retrieval
2. **Utility Bill Pro-Rating** - Intelligent cost allocation based on occupancy
3. **Document Rejection Tracking** - Pre-ingestion document type filtering and data quality protection

---

## 1. Audit Trail Enhancement

### Objective
Implement comprehensive tracking of all file operations with detailed metadata and retrieval capabilities.

### Technical Tasks

#### 1.1 Database Schema Changes
**Estimated Effort: 4 hours**

- [ ] **Create audit_trail table**
  - `id` (UUID, primary key)
  - `file_id` (UUID, foreign key to files table)
  - `action_type` (enum: 'upload', 'process', 'analyze', 'delete', 'update')
  - `timestamp` (datetime with timezone)
  - `user_id` (string, nullable for system actions)
  - `metadata` (JSONB for flexible data storage)
  - `processing_status` (enum: 'pending', 'completed', 'failed')
  - `error_details` (text, nullable)
  - `file_size_bytes` (bigint)
  - `file_type` (string)
  - `processing_duration_ms` (integer, nullable)

- [ ] **Add indexes for performance**
  - Index on `file_id`
  - Index on `action_type`
  - Index on `timestamp`
  - Composite index on `file_id, timestamp`

- [ ] **Create database migration script**
  - Alembic migration file
  - Rollback procedures

#### 1.2 Backend API Development
**Estimated Effort: 12 hours**

- [ ] **Create AuditTrail model** (`models/audit_trail.py`)
  - SQLAlchemy model definition
  - Relationship mappings
  - Validation rules

- [ ] **Implement AuditTrailService** (`services/audit_trail_service.py`)
  - `create_audit_entry(file_id, action_type, metadata)`
  - `get_file_history(file_id, limit=50)`
  - `get_audit_trail(filters, pagination)`
  - `get_system_statistics(date_range)`

- [ ] **Create audit trail API endpoints** (`api/audit_routes.py`)
  - `GET /api/audit/file/{file_id}` - File-specific history
  - `GET /api/audit/trail` - System-wide audit trail with filters
  - `GET /api/audit/stats` - Processing statistics
  - `GET /api/audit/export` - Export audit data (CSV/JSON)

- [ ] **Implement audit trail middleware** (`middleware/audit_middleware.py`)
  - Automatic logging of file operations
  - Integration with existing upload/processing flows
  - Error handling and retry logic

#### 1.3 Frontend Development
**Estimated Effort: 16 hours**

- [ ] **Create AuditTrail component** (`components/AuditTrail.tsx`)
  - Timeline view of file operations
  - Filtering capabilities (date range, action type)
  - Search functionality
  - Export options

- [ ] **Create FileHistory component** (`components/FileHistory.tsx`)
  - Individual file operation history
  - Processing status indicators
  - Error details display
  - Metadata viewer

- [ ] **Add audit trail to main dashboard** (`pages/Dashboard.tsx`)
  - Recent activity widget
  - Quick stats display
  - Navigation to full audit trail

- [ ] **Create audit trail page** (`pages/AuditTrail.tsx`)
  - Full audit trail interface
  - Advanced filtering options
  - Pagination
  - Data export functionality

#### 1.4 Integration Tasks
**Estimated Effort: 8 hours**

- [ ] **Update file upload flow** (`services/file_service.py`)
  - Add audit logging to upload process
  - Track file metadata
  - Record processing start/completion

- [ ] **Update AI processing workflow** (`services/ai_service.py`)
  - Log AI analysis operations
  - Track processing duration
  - Record success/failure status

- [ ] **Add bulk operations audit** (`services/bulk_service.py`)
  - Track batch processing operations
  - Record individual file statuses within batches
  - Handle partial failures

#### 1.5 Testing Requirements
**Estimated Effort: 10 hours**

- [ ] **Unit Tests**
  - AuditTrail model tests
  - AuditTrailService tests
  - API endpoint tests
  - Middleware tests

- [ ] **Integration Tests**
  - End-to-end audit trail creation
  - File operation tracking
  - API response validation
  - Database integrity checks

- [ ] **Frontend Tests**
  - Component rendering tests
  - User interaction tests
  - Data fetching tests
  - Export functionality tests

---

## 2. Utility Bill Pro-Rating Enhancement

### Objective
Implement intelligent utility cost allocation based on occupancy periods and space utilization.

### Technical Tasks

#### 2.1 Database Schema Changes
**Estimated Effort: 6 hours**

- [ ] **Create occupancy_periods table**
  - `id` (UUID, primary key)
  - `property_id` (UUID, foreign key)
  - `tenant_id` (UUID, foreign key)
  - `start_date` (date)
  - `end_date` (date, nullable for current occupancy)
  - `occupancy_percentage` (decimal, 0-100)
  - `space_type` (enum: 'full', 'partial', 'shared')
  - `square_footage` (decimal, nullable)
  - `created_at` (timestamp)
  - `updated_at` (timestamp)

- [ ] **Create utility_allocations table**
  - `id` (UUID, primary key)
  - `utility_bill_id` (UUID, foreign key to files table)
  - `occupancy_period_id` (UUID, foreign key)
  - `allocation_percentage` (decimal, 0-100)
  - `allocated_amount` (decimal)
  - `calculation_method` (enum: 'time_based', 'space_based', 'hybrid')
  - `calculation_details` (JSONB)
  - `created_at` (timestamp)

- [ ] **Add utility bill metadata fields**
  - Add columns to files table or create utility_bills table
  - `bill_start_date` (date)
  - `bill_end_date` (date)
  - `total_amount` (decimal)
  - `utility_type` (enum: 'electric', 'gas', 'water', 'internet', 'other')

#### 2.2 Backend Services Development
**Estimated Effort: 16 hours**

- [ ] **Create OccupancyPeriod model** (`models/occupancy_period.py`)
  - SQLAlchemy model with validations
  - Date range validation
  - Overlap detection methods

- [ ] **Create UtilityAllocation model** (`models/utility_allocation.py`)
  - Allocation calculation methods
  - Relationship mappings
  - Validation rules

- [ ] **Implement ProRatingService** (`services/pro_rating_service.py`)
  - `calculate_time_based_allocation(bill_period, occupancy_periods)`
  - `calculate_space_based_allocation(bill_period, occupancy_periods)`
  - `calculate_hybrid_allocation(bill_period, occupancy_periods)`
  - `generate_allocation_report(bill_id)`
  - `validate_allocation_total(allocations)`

- [ ] **Implement OccupancyService** (`services/occupancy_service.py`)
  - `create_occupancy_period(tenant_id, property_id, details)`
  - `update_occupancy_period(period_id, updates)`
  - `get_active_occupancies(property_id, date_range)`
  - `detect_occupancy_overlaps(property_id)`
  - `calculate_occupancy_percentage(period, total_space)`

- [ ] **Update UtilityBillService** (`services/utility_bill_service.py`)
  - Add pro-rating integration
  - Automatic allocation calculation
  - Report generation methods

#### 2.3 API Endpoints
**Estimated Effort: 12 hours**

- [ ] **Occupancy management endpoints** (`api/occupancy_routes.py`)
  - `POST /api/occupancy/periods` - Create occupancy period
  - `GET /api/occupancy/periods/{property_id}` - Get property occupancies
  - `PUT /api/occupancy/periods/{id}` - Update occupancy period
  - `DELETE /api/occupancy/periods/{id}` - Delete occupancy period
  - `POST /api/occupancy/validate` - Validate occupancy data

- [ ] **Pro-rating endpoints** (`api/pro_rating_routes.py`)
  - `POST /api/prorating/calculate/{bill_id}` - Calculate allocations
  - `GET /api/prorating/allocations/{bill_id}` - Get bill allocations
  - `POST /api/prorating/recalculate/{bill_id}` - Recalculate with new data
  - `GET /api/prorating/report/{bill_id}` - Generate allocation report
  - `POST /api/prorating/bulk-calculate` - Batch allocation calculation

#### 2.4 Frontend Development
**Estimated Effort: 20 hours**

- [ ] **Create OccupancyManager component** (`components/OccupancyManager.tsx`)
  - Timeline view of occupancy periods
  - Add/edit occupancy forms
  - Overlap detection and warnings
  - Bulk import from CSV

- [ ] **Create ProRatingCalculator component** (`components/ProRatingCalculator.tsx`)
  - Interactive allocation calculator
  - Method selection (time/space/hybrid)
  - Real-time calculation preview
  - Allocation adjustment interface

- [ ] **Create AllocationReport component** (`components/AllocationReport.tsx`)
  - Detailed allocation breakdown
  - Visual charts and graphs
  - Export capabilities (PDF, CSV)
  - Historical comparison

- [ ] **Update UtilityBillViewer** (`components/UtilityBillViewer.tsx`)
  - Add pro-rating section
  - Display calculated allocations
  - Quick recalculation options
  - Integration with occupancy data

- [ ] **Create ProRatingDashboard** (`pages/ProRatingDashboard.tsx`)
  - Property-level allocation overview
  - Batch processing interface
  - Settings and configuration
  - Analytics and reporting

#### 2.5 Calculation Engine
**Estimated Effort: 14 hours**

- [ ] **Implement TimeBasedCalculator** (`calculators/time_based.py`)
  - Daily pro-ration calculations
  - Partial month handling
  - Leap year considerations
  - Weekend/holiday adjustments

- [ ] **Implement SpaceBasedCalculator** (`calculators/space_based.py`)
  - Square footage calculations
  - Shared space handling
  - Common area allocations
  - Usage-based adjustments

- [ ] **Implement HybridCalculator** (`calculators/hybrid.py`)
  - Combined time and space logic
  - Weighted calculation methods
  - Custom allocation rules
  - Validation and error handling

#### 2.6 Testing Requirements
**Estimated Effort: 12 hours**

- [ ] **Unit Tests**
  - Calculator logic tests
  - Model validation tests
  - Service method tests
  - Edge case handling

- [ ] **Integration Tests**
  - End-to-end allocation flow
  - Multi-tenant scenarios
  - Complex occupancy patterns
  - Report generation tests

- [ ] **Frontend Tests**
  - Calculator component tests
  - Form validation tests
  - Report rendering tests
  - User workflow tests

---

## 3. Document Rejection Tracking Enhancement

### Status: **Completed**

### Objective
Implement a pre-ingestion document type recognition system that rejects unrecognized documents, prevents them from entering the Neo4j knowledge graph, while maintaining a separate audit trail of rejected documents for user review.

### Implementation Summary

**Implementation Approach:** Pre-ingestion document type recognition with rejection workflow
**Key Components:** DocumentRecognitionService, rejection API endpoints, web UI integration
**Testing Results:** Successfully rejected invoice.pdf during testing
**Integration Status:** Fully integrated with web UI and document processing pipeline

### Technical Implementation Details

#### 3.1 Database Schema Changes
**Status: Completed**

- [x] **Create rejected_documents table**
  - `id` (UUID, primary key)
  - `original_filename` (string)
  - `source_file_path` (string, path to stored original file)
  - `file_size_bytes` (bigint)
  - `file_type` (string)
  - `rejected_at` (timestamp)
  - `rejection_reason` (enum: 'UNRECOGNIZED_DOCUMENT_TYPE', 'POOR_IMAGE_QUALITY', 'UNSUPPORTED_FORMAT')
  - `rejection_details` (text, nullable)
  - `created_at` (timestamp)

- [x] **Add indexes for performance**
  - Index on `rejected_at`
  - Index on `rejection_reason`
  - Index on `original_filename`

- [x] **Create database migration script**
  - PostgreSQL migration file implemented
  - Rollback procedures included

#### 3.2 Backend Services Development
**Status: Completed**

- [x] **Create RejectedDocument model** (`models/rejected_document.py`)
  - Database model definition implemented
  - Validation rules added
  - Query methods created

- [x] **Create DocumentTypeRecognitionService** (`services/document_recognition_service.py`)
  - `recognize_document_type(file_path, file_content)` implemented
  - `is_electricity_bill(extracted_data)` implemented
  - `is_water_bill(extracted_data)` implemented
  - `is_waste_manifest(extracted_data)` implemented
  - AI/ML classification models integrated

- [x] **Implement RejectionTrackingService** (`services/rejection_tracking_service.py`)
  - `store_rejected_document(filename, file_path, reason, details)` implemented
  - `get_rejected_documents(filters, pagination)` implemented
  - `get_rejection_statistics(date_range)` implemented
  - `delete_rejected_document(rejection_id)` implemented

- [x] **Update DocumentProcessingService** (`services/document_processing_service.py`)
  - Pre-ingestion validation step added
  - Rejection workflow implemented before Neo4j ingestion
  - Rejected documents prevented from entering knowledge graph

#### 3.3 API Endpoints
**Status: Completed**

- [x] **Rejection tracking endpoints** (`api/rejection_routes.py`)
  - `GET /api/rejections` - List rejected documents with filters and pagination
  - `GET /api/rejections/{rejection_id}` - Get specific rejected document details
  - `GET /api/rejections/statistics` - Get rejection statistics and trends
  - `DELETE /api/rejections/{rejection_id}` - Delete rejected document record
  - `GET /api/rejections/export` - Export rejection data (CSV/JSON)

- [x] **Document upload validation integration**
  - Upload endpoints updated with pre-ingestion validation
  - Clear rejection responses with reasons implemented
  - Rejected documents stored but not processed

#### 3.4 Frontend Development
**Status: Completed**

- [x] **Create RejectedDocuments component** (`components/RejectedDocuments.tsx`)
  - List view of rejected documents implemented
  - Filter by rejection reason and date added
  - Search by filename functionality
  - Pagination support included

- [x] **Create RejectionDetails component** (`components/RejectionDetails.tsx`)
  - Rejection reason and details display
  - Original filename and file info shown
  - View original document option available
  - Delete rejection record action implemented

- [x] **Update DocumentUpload component** (`components/DocumentUpload.tsx`)
  - Rejection feedback on upload added
  - Rejection reasons displayed clearly
  - Guidance for acceptable document types provided

- [x] **Create RejectionStatistics component** (`components/RejectionStatistics.tsx`)
  - Dashboard widget showing rejection trends
  - Breakdown by rejection reason
  - Monthly/weekly statistics
  - Data quality metrics

- [x] **Add Rejected Documents section to main navigation**
  - New tab/menu item for rejected documents
  - Badge showing count of recent rejections
  - Quick access to rejection management

#### 3.5 Document Type Recognition System
**Status: Completed**

- [x] **Implement DocumentTypeClassifier** (`classifiers/document_type_classifier.py`)
  - Text-based classification using extracted content
  - Keyword and pattern matching for document types
  - Confidence scoring for classifications
  - Support for electricity bills, water bills, and waste manifests

- [x] **Create DocumentValidationRules** (`validation/document_rules.py`)
  - Validation criteria defined for each accepted document type
  - Quality thresholds implemented (text extraction confidence, image quality)
  - Format validation (file type, structure)
  - Content validation (required fields, data patterns)

- [x] **Implement RejectionLogicService** (`services/rejection_logic_service.py`)
  - Document type recognition coordination
  - Validation rules application
  - Rejection reasons and details generation
  - Rejection statistics maintenance

#### 3.6 Testing Requirements
**Status: Completed**

- [x] **Unit Tests**
  - Document type classification tests completed
  - Rejection logic tests implemented
  - Validation rule tests added
  - Database model tests created

- [x] **Integration Tests**
  - End-to-end upload and rejection workflow tested
  - Document type recognition accuracy validated
  - Pre-ingestion filtering tests passed
  - Rejected document storage tests completed

- [x] **Frontend Tests**
  - Rejected documents list tests implemented
  - Upload feedback tests completed
  - Statistics component tests added
  - Navigation integration tests passed

### Testing Results
- Successfully rejected invoice.pdf as unrecognized document type
- Rejection workflow properly prevents documents from entering Neo4j knowledge graph
- Web UI correctly displays rejection feedback and manages rejected documents
- All API endpoints tested and functional

---

## Implementation Timeline

### Phase 1A: Database and Core Services (Week 1-2)
- Database schema changes for all three enhancements
- Core model implementations
- Basic service layer development

### Phase 1B: API Development (Week 3-4)
- REST API endpoints for all features
- Integration with existing services
- API testing and documentation

### Phase 1C: Frontend Development (Week 5-7)
- User interface components
- Dashboard integrations
- User experience enhancements

### Phase 1D: Testing and Integration (Week 8)
- Comprehensive testing suite
- Integration testing
- Performance optimization
- Bug fixes and refinements

## Success Criteria

### Audit Trail Enhancement
- [ ] All file operations are automatically tracked
- [ ] Users can view comprehensive file history
- [ ] System statistics are available and accurate
- [ ] Export functionality works for audit data

### Utility Bill Pro-Rating
- [ ] Occupancy periods can be managed effectively
- [ ] Allocation calculations are accurate and auditable
- [ ] Multiple calculation methods are supported
- [ ] Reports can be generated and exported

### Document Rejection Tracking
- [x] Unrecognized documents are automatically rejected before Neo4j ingestion
- [x] Original rejected documents are preserved in separate storage
- [x] Users can view and manage rejected documents through dedicated UI
- [x] System protects data quality by preventing invalid documents from entering the knowledge graph

## Technical Dependencies

### Database
- PostgreSQL with JSONB support
- Alembic for migrations
- Connection pooling for performance

### Backend
- FastAPI framework
- SQLAlchemy ORM
- Celery for background tasks
- Pydantic for validation

### Frontend
- React with TypeScript
- State management (Redux/Context)
- Chart libraries for visualization
- Export libraries for reports

## Notes

This document focuses exclusively on feature implementation tasks. The following items are explicitly excluded and covered in separate documentation:

- Docker containerization
- Cloud hosting setup
- CI/CD pipeline configuration
- Infrastructure monitoring
- Security hardening
- Performance optimization at the infrastructure level

All development tasks assume a working local development environment and focus on business logic, user interface, and data management aspects of the three core enhancements.