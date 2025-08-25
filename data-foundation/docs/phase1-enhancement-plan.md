# EHS AI Demo - Phase 1 Enhancement Plan

**Document Version:** 1.2
**Date:** 2025-08-22
**Status:** Docker-Based Self-Sufficient Deployment

## 1. Introduction

This document outlines the technical specification and implementation plan for Phase 1 enhancements to the EHS AI Demo project. The goal of this phase is to improve data auditability, financial accuracy, and data quality management within a self-contained Docker-based environment. This plan serves as a guide for the development team, covering database modifications, API changes, UI/UX considerations, and deployment architecture.

**Important Note:** This implementation uses a completely self-sufficient Docker-based architecture that requires no external cloud services. All file storage, database operations, and application services run within Docker containers orchestrated by docker-compose.

The three core enhancements for Phase 1 are:
1.  **Audit Trail Enhancement:** Tracking the source file for each processed document.
2.  **Utility Bill Pro-Rating:** Allocating costs and usage from billing periods to their corresponding calendar months.
3.  **Document Rejection Tracking:** Implementing a pre-ingestion filter system that rejects unrecognized documents (those not positively identified as electricity bills, water bills, or waste manifests) before they enter the Neo4j knowledge graph, while maintaining a separate audit trail of rejected documents.

---

## 1.1. Docker-Based Architecture Overview

### 1.1.1. Deployment Strategy

This EHS AI Demo is designed as a completely self-contained system that runs entirely within Docker containers. The architecture eliminates dependencies on external cloud services, making it ideal for demonstrations, development, and environments with restricted internet access.

### 1.1.2. Container Architecture

The system consists of the following Docker services orchestrated via docker-compose:

- **Application Container**: Houses the main EHS AI application with web interface
- **Neo4j Database Container**: Provides graph database functionality for document relationships
- **File Storage Container**: Manages local file storage using Docker volumes
- **Optional Services**: Additional containers for supporting services (e.g., background processing)

### 1.1.3. Local Storage Structure

All file storage utilizes Docker volumes with the following structure:

```
/app/storage/
├── uploads/           # Original uploaded documents
│   └── {uuid}/       # UUID-based directory structure
│       └── {sanitized_filename}
├── processed/         # Processed document artifacts
├── temp/             # Temporary processing files
└── backups/          # Database and file backups
```

### 1.1.4. Deployment Commands

The entire system can be deployed using standard Docker Compose commands:

```bash
# Initial deployment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the system
docker-compose down

# Update and restart
docker-compose down && docker-compose up -d --build
```

### 1.1.5. Data Persistence

All critical data persists through Docker volumes:
- **neo4j-data**: Database files and logs
- **app-storage**: Uploaded and processed documents
- **app-config**: Application configuration files

---

## 2. Enhancement 1: Audit Trail

### 2.1. Current State

Currently, once a document is uploaded and processed, the system extracts the relevant data but does not maintain a link back to the original source file. This makes it difficult to perform audits or manually verify the data extraction against the original document.

### 2.2. Requirements

-   The system must store a reference to the original uploaded file for every processed document.
-   The original filename must be preserved and displayed to the user.
-   Users with appropriate permissions must be able to view or download the original source file from the application.
-   The file storage solution uses Docker volumes for persistence, preventing filename collisions through UUID-based directory structure.
-   All file operations must be container-aware and work within the Docker environment.

### 2.3. Database Schema Changes

We will add two new properties to the `ProcessedDocument` nodes to store the original filename for display and a unique path for retrieval from local Docker volume storage.

**Node:** `ProcessedDocument`

```cypher
// Add new properties to existing ProcessedDocument nodes
MATCH (doc:ProcessedDocument)
SET doc.original_filename = null,
    doc.source_file_path = null;
```

-   `original_filename`: Stores the original name of the uploaded file (e.g., `invoice_july_2023.pdf`).
-   `source_file_path`: Stores the unique path within the Docker volume (e.g., `/app/storage/uploads/{uuid}/{sanitized_filename}`). The recommended format is `uploads/{uuid}/{sanitized_filename}` to prevent collisions. The filename component of the path must be sanitized to remove characters that could be problematic for file systems or URLs.

### 2.4. API Endpoint Modifications

A new API endpoint will be created to serve the original source file directly from the Docker volume storage.

**Endpoint:** `GET /api/v1/documents/{document_id}/source_file`

-   **Description:** Serves the original source file directly from local storage with appropriate headers.
-   **Authorization:** Requires user to be authenticated and have permission to view the specified document.
-   **Success Response (200 OK):** Returns the file content with appropriate headers:
    ```
    Content-Type: application/pdf (or appropriate MIME type)
    Content-Disposition: attachment; filename="invoice_july_2023.pdf"
    Content-Length: <file_size>
    ```
-   **Alternative Endpoint:** `GET /api/v1/documents/{document_id}/source_url` for compatibility
    ```json
    {
      "download_url": "/api/v1/documents/{document_id}/source_file",
      "original_filename": "invoice_july_2023.pdf",
      "file_size": 1024000
    }
    ```
-   **Failure Response (404 Not Found):** If the document or source file does not exist.
-   **Failure Response (403 Forbidden):** If the user is not authorized.

### 2.5. UI/UX Design Considerations

-   The list view for processed documents should include a new column titled "Source File".
-   The value in this column will be a clickable link displaying the `original_filename`.
-   Clicking the link will open a modal window that displays the PDF contents directly in the browser using a PDF viewer component (such as react-pdf or similar).
-   In the detailed view for a single document, a prominent "View Original" button or link should be present that opens the PDF in the same modal viewer.

### 2.6. Implementation Steps

1.  Configure Docker volumes for file storage in docker-compose.yml.
2.  Update the Neo4j database schema by adding the new properties to `ProcessedDocument` nodes.
3.  Modify the document upload service to ensure an **atomic operation**:
    a. Sanitize the original filename for safe storage.
    b. Generate a unique path (e.g., `uploads/{uuid}/{sanitized_filename}`).
    c. Save the file to the Docker volume at `/app/storage/uploads/{uuid}/{sanitized_filename}`.
    d. **Only upon successful file save**, update the `ProcessedDocument` node with the `original_filename` and `source_file_path` properties. This prevents orphaned node data.
4.  Implement the new backend API endpoint (`GET /api/v1/documents/{document_id}/source_file`) including logic for serving files directly from local storage.
5.  Update the frontend to display the source file link in the UI and handle the PDF viewer modal action.
6.  Ensure proper file permissions and security within the Docker container environment.

### 2.7. Testing Strategy

-   **Unit Tests:** Verify file upload logic, path generation, **filename sanitization**, and file serving functionality.
-   **Integration Tests:** Test the end-to-end flow from file upload to Neo4j node property updates. Test the API endpoint authorization and response. **Verify that a failed file save does not update the ProcessedDocument node properties.**
-   **Manual Tests:** Verify that files can be successfully uploaded and downloaded through the UI. Test with filenames containing special characters.
-   **Docker Tests:** Verify file persistence across container restarts and proper volume mounting.

### 2.8. Estimated Effort

-   **Effort:** Medium (M)
-   **Timeline:** ~1 week

---

## 3. Enhancement 2: Utility Bill Pro-Rating

### 3.1. Current State

The system currently stores utility usage with a `start_date` and `end_date` corresponding to the billing period. When viewing monthly reports, usage is attributed to the month in which the billing period ends, which is inaccurate for bills that span two calendar months.

### 3.2. Requirements

-   Usage and cost from a single bill must be proportionally allocated to the calendar months they cover.
-   The allocation logic should be based on the number of days in the billing period that fall within each month.
-   The UI must be able to display both the original billing period data and the allocated monthly data.
-   Reporting queries for monthly totals must be fast and efficient.

### 3.3. Database Schema Changes

To support efficient reporting, we will pre-calculate and store the allocations as new nodes connected to the processed documents. This avoids complex on-the-fly calculations and simplifies analytics.

**New Node Type:** `MonthlyUsageAllocation`

```cypher
// Create new MonthlyUsageAllocation nodes and relationships
CREATE (allocation:MonthlyUsageAllocation {
    usage_year: $year,
    usage_month: $month,
    allocated_usage: $usage,
    allocated_cost: $cost,
    created_at: datetime()
})

// Create relationship between ProcessedDocument and MonthlyUsageAllocation
MATCH (doc:ProcessedDocument {id: $document_id})
CREATE (doc)-[:HAS_ALLOCATION]->(allocation)

// Create indexes for efficient querying
CREATE INDEX allocation_year_month IF NOT EXISTS FOR (a:MonthlyUsageAllocation) ON (a.usage_year, a.usage_month);
```

### 3.4. API Endpoint Modifications

Existing reporting endpoints will be modified to query the new `MonthlyUsageAllocation` nodes instead of aggregating from `ProcessedDocument` nodes for monthly summaries.

**Modified Endpoint:** `GET /api/v1/reports/monthly_summary?year=<year>&month=<month>`

-   **Description:** Retrieves a summary of all usage and costs for a given calendar month.
-   **Backend Logic:** The endpoint will now execute a Cypher query to sum allocated usage and costs from `MonthlyUsageAllocation` nodes, filtering by `usage_year` and `usage_month`.

A new endpoint may be needed to view the allocation breakdown for a single source document.

**New Endpoint:** `GET /api/v1/documents/{document_id}/allocations`

-   **Success Response (200 OK):**
    ```json
    {
      "allocations": [
        { "year": 2023, "month": 8, "allocated_usage": 50.5, "allocated_cost": 10.25 },
        { "year": 2023, "month": 9, "allocated_usage": 150.0, "allocated_cost": 30.75 }
      ]
    }
    ```

### 3.5. UI/UX Design Considerations

-   Monthly summary dashboards will now display the accurately allocated totals.
-   In the detailed view for a processed document, a new section should display the pro-rated breakdown, showing how much usage/cost was allocated to each calendar month.
-   UI tooltips or info icons can be used to explain how the pro-rating is calculated.

### 3.6. Implementation Steps

1.  Create the new `MonthlyUsageAllocation` node type and relationship structure in Neo4j.
2.  Develop a robust pro-rating calculation service/module. This must correctly handle leap years and varying month lengths.
3.  Integrate this service into the document processing pipeline. After a document's data is extracted, the service will calculate and create the allocation nodes with appropriate relationships.
4.  Update all relevant reporting APIs and services to query the new allocation nodes using Cypher queries.
5.  Update the frontend to display the allocation breakdown on the document detail page.
6.  **(Mandatory)** Create a one-time backfill script to process all existing `ProcessedDocument` nodes and create corresponding `MonthlyUsageAllocation` nodes to ensure historical data consistency.
7.  Implement logic to handle updates to `ProcessedDocument` nodes. If a document's billing period or financial data is modified, its related `MonthlyUsageAllocation` nodes must be deleted and recalculated to maintain data integrity.

### 3.7. Testing Strategy

-   **Unit Tests:** Extensively test the pro-rating logic with edge cases:
    -   Billing periods fully within one month.
    -   Periods spanning two months.
    -   Periods spanning a year-end (e.g., Dec 15 - Jan 15).
    -   Periods spanning a leap day.
-   **Integration Tests:** Verify that processing a document correctly creates both `ProcessedDocument` nodes and related `MonthlyUsageAllocation` nodes with proper relationships. **Test that updating a document correctly triggers recalculation of allocations.**
-   **Manual Tests:** Verify that monthly reports show correct totals and that the allocation breakdown in the UI is accurate.

### 3.8. Estimated Effort

-   **Effort:** Large (L)
-   **Timeline:** ~2 weeks (due to complexity of logic, update handling, and mandatory data backfilling)

---

## 4. Enhancement 3: Document Rejection Tracking

### 4.1. Current State

The system currently processes all uploaded documents without validating whether they are actually relevant EHS documents (electricity bills, water bills, or waste manifests). This can lead to data quality issues where unrelated documents (e.g., marketing materials, personal invoices, contracts) are processed and potentially ingested into the Neo4j knowledge graph, cluttering the dataset with irrelevant data.

### 4.2. Requirements

-   The system must implement a pre-ingestion document type recognition system that validates documents before they reach the Neo4j processing pipeline.
-   Only documents positively identified as electricity bills, water bills, or waste manifests should be allowed to proceed to Neo4j ingestion.
-   Unrecognized or invalid documents must be rejected and stored in a separate rejected documents table.
-   The original file must be preserved in its original form for rejected documents to maintain audit capability.
-   A reason for rejection must be stored (e.g., "UNRECOGNIZED_DOCUMENT_TYPE", "POOR_IMAGE_QUALITY", "UNSUPPORTED_FORMAT").
-   The web application UI must provide a separate section for users to view and manage rejected documents.
-   The rejection system must act as a data quality protection mechanism, ensuring the Neo4j knowledge graph only contains validated, relevant EHS data.

### 4.3. Database Schema Changes

Since rejected documents should never reach the Neo4j knowledge graph, we will create a separate PostgreSQL table to store rejected documents. This maintains complete separation between valid EHS data in Neo4j and rejected documents in the relational database.

**New Table:** `rejected_documents` (PostgreSQL)

```sql
CREATE TABLE rejected_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_filename VARCHAR(500) NOT NULL,
    source_file_path VARCHAR(1000) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    rejected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    rejection_reason VARCHAR(50) NOT NULL CHECK (
        rejection_reason IN (
            'UNRECOGNIZED_DOCUMENT_TYPE',
            'POOR_IMAGE_QUALITY', 
            'UNSUPPORTED_FORMAT'
        )
    ),
    rejection_details TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_rejected_documents_rejected_at ON rejected_documents(rejected_at);
CREATE INDEX idx_rejected_documents_rejection_reason ON rejected_documents(rejection_reason);
CREATE INDEX idx_rejected_documents_filename ON rejected_documents(original_filename);
```

**Important:** No changes are needed to Neo4j schema since rejected documents will never reach the Neo4j database. Only successfully validated documents (electricity bills, water bills, waste manifests) will be processed into `ProcessedDocument` nodes.

### 4.4. API Endpoint Modifications

New endpoints will be created to manage rejected documents, while the existing document processing pipeline will be enhanced with pre-ingestion validation.

**Enhanced Upload Processing:** The existing document upload endpoints will be modified to include document type recognition before processing.

**New Endpoints for Rejected Documents:**

**Endpoint:** `GET /api/v1/rejected-documents`
-   **Description:** List all rejected documents with filtering and pagination.
-   **Query Parameters:**
    - `reason`: Filter by rejection reason
    - `date_from`, `date_to`: Filter by rejection date range
    - `page`, `page_size`: Pagination
-   **Response:**
    ```json
    {
      "documents": [
        {
          "id": "uuid",
          "original_filename": "marketing_brochure.pdf",
          "file_size_bytes": 1024000,
          "file_type": "application/pdf",
          "rejected_at": "2024-01-15T10:30:00Z",
          "rejection_reason": "UNRECOGNIZED_DOCUMENT_TYPE",
          "rejection_details": "Document content does not match any accepted EHS document types"
        }
      ],
      "total_count": 42,
      "page": 1,
      "page_size": 10
    }
    ```

**Endpoint:** `GET /api/v1/rejected-documents/{id}`
-   **Description:** Get details of a specific rejected document.

**Endpoint:** `GET /api/v1/rejected-documents/{id}/download`
-   **Description:** Download the original rejected document file.

**Endpoint:** `DELETE /api/v1/rejected-documents/{id}`
-   **Description:** Delete a rejected document record and its associated file.

### 4.5. UI/UX Design Considerations

-   A new section/tab in the UI labeled "Rejected Documents" will be created, completely separate from the main document management interface.
-   This view will list all rejected documents, showing the original filename, upload date, rejection reason, file size, and rejection details.
-   Users will be able to download the original rejected documents for manual review.
-   The document upload interface will provide clear feedback when documents are rejected, explaining why they were not accepted and what document types are supported.
-   A dashboard widget will show rejection statistics (total rejections, breakdown by reason, recent trends) to help monitor data quality.
-   The main document list will only show successfully processed documents (those in Neo4j), keeping the primary interface clean and focused on valid EHS data.

### 4.6. Implementation Steps

1.  Create the PostgreSQL `rejected_documents` table with appropriate indexes and constraints.
2.  Implement document type recognition service that can identify electricity bills, water bills, and waste manifests using text content analysis and keyword matching.
3.  Modify the document upload/processing pipeline to include pre-ingestion validation:
    a. Extract basic text content from uploaded documents
    b. Run document type recognition analysis
    c. If document is recognized as valid EHS type, proceed with normal Neo4j processing
    d. If document is not recognized, store in rejected_documents table and preserve original file
4.  Implement the new rejected documents API endpoints (`GET /api/v1/rejected-documents`, etc.).
5.  Develop the new "Rejected Documents" section in the frontend UI.
6.  Update the document upload interface to provide clear rejection feedback to users.
7.  Add rejection statistics dashboard widget for monitoring data quality trends.

### 4.7. Testing Strategy

-   **Unit Tests:** Test document type recognition logic with sample electricity bills, water bills, waste manifests, and unrecognized documents.
-   **Integration Tests:** Verify end-to-end rejection flow - ensure unrecognized documents are stored in rejected_documents table, original files are preserved, and no data reaches Neo4j. Verify recognized documents continue to process normally into Neo4j.
-   **Manual Tests:** Test document upload with various file types to verify correct rejection feedback. Test rejected documents UI for listing, filtering, and downloading capabilities.

### 4.8. Future Considerations

Phase 1 focuses on basic document type recognition using text analysis and keyword matching. Future enhancements could include:
-   Machine learning models for more sophisticated document classification
-   OCR quality assessment for better handling of scanned documents
-   User-trainable classification rules for edge cases
-   Bulk re-processing of rejected documents when classification rules improve

### 4.9. Estimated Effort

-   **Effort:** Medium (M)
-   **Timeline:** ~1.5 weeks

---

## 5. Overall Timeline & Dependencies

-   **Total Estimated Timeline:** ~5 weeks
-   **Dependencies:** The Audit Trail enhancement should ideally be implemented first, as the other features will build upon the same `ProcessedDocument` nodes. The Pro-Rating and Rejection features can be developed in parallel if resources allow.

## 6. Risk Assessment & Mitigation

### 6.1. Technical Risks

-   **Docker Volume Management:** The audit trail feature relies on Docker volume persistence. Risk mitigation includes implementing proper error handling, backup strategies, and volume validation checks.
-   **Container Resource Limits:** File storage within containers may face disk space limitations. Implement monitoring and cleanup strategies for temporary files.
-   **Pro-Rating Calculation Complexity:** The date-based allocation logic, especially with updates, must handle all edge cases correctly. Extensive unit testing and validation against known scenarios will mitigate this risk.
-   **Database Performance:** Adding new node properties, relationships, and indexes could impact query performance. Neo4j indexing and Cypher query optimization will be essential.
-   **Cross-Container File Access:** Ensure proper file sharing between application containers and any background processing containers.

### 6.2. Business Risks

-   **Data Migration & Backfill:** The mandatory backfill for pro-rating must be carefully managed on production data. The script should be idempotent and thoroughly tested in a staging environment.
-   **User Training:** New UI features will require user training and documentation updates.

## 7. Success Criteria

Phase 1 will be considered successful when:

1. **Audit Trail:** Users can successfully download original source files for any processed document, and the system is resilient to upload failures.
2. **Pro-Rating:** Monthly reports accurately reflect usage allocated to calendar months. Data remains consistent even when original documents are corrected.
3. **Rejection Tracking:** The system automatically rejects unrecognized documents before they reach Neo4j, maintains complete separation between valid EHS data and rejected documents, and provides a dedicated UI for managing rejected documents.
4. **Performance:** All new features maintain response times under 2 seconds for typical operations.
5. **Reliability:** The system maintains 99.9% uptime with the new features enabled.
6. **Docker Deployment:** The entire system can be deployed and run using only `docker-compose up -d` without external dependencies.

## 8. Docker-Specific Implementation Notes

### 8.1. Volume Configuration

The docker-compose.yml must include the following volume configurations:

```yaml
volumes:
  neo4j-data:
    driver: local
  app-storage:
    driver: local
  app-config:
    driver: local

services:
  app:
    volumes:
      - app-storage:/app/storage
      - app-config:/app/config
  
  neo4j:
    volumes:
      - neo4j-data:/data
```

### 8.2. Environment Variables

All configuration should use environment variables that can be set in docker-compose.yml:

```yaml
environment:
  - STORAGE_ROOT=/app/storage
  - NEO4J_URI=bolt://neo4j:7687
  - UPLOAD_MAX_SIZE=50MB
  - DEBUG_MODE=false
```

### 8.3. File Permissions

Ensure the application runs with appropriate user permissions within the container to read/write to the storage volumes.

### 8.4. Backup Strategy

Implement backup procedures for both Neo4j data and file storage:
- Database backups via Neo4j dump commands
- File storage backups via volume snapshots or rsync
- Backup scripts should be container-aware and executable via docker-compose

This plan provides a clear path forward for Phase 1 with a completely self-contained Docker-based architecture. All technical decisions should be reviewed and confirmed before beginning implementation.