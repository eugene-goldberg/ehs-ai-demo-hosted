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
3.  **Document Rejection Tracking:** Implementing a system to manage and track documents that are identified as invalid or irrelevant.

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

The system has no formal process for handling uploaded documents that are not relevant to EHS (e.g., marketing materials, other invoices). These files may be processed with errors or simply ignored, but there is no visibility into why a document was not included in the dataset.

### 4.2. Requirements

-   The system must allow documents to be marked as "rejected".
-   A reason for rejection must be stored for each rejected document.
-   An audit trail of who rejected a document and when must be maintained.
-   Rejection reasons should be standardized for consistent reporting.
-   A user interface must be provided to view the list of rejected documents and their rejection reasons.
-   The primary dataset of processed documents should not be cluttered with rejected items.

### 4.3. Database Schema Changes

We will add status, reason, and audit properties to the `ProcessedDocument` nodes. This is the simplest approach for Phase 1 and avoids creating separate node types. An index on the `status` property will ensure efficient querying.

**Node:** `ProcessedDocument`

```cypher
// Add new properties to existing ProcessedDocument nodes
MATCH (doc:ProcessedDocument)
SET doc.status = 'PROCESSING',
    doc.rejection_reason = null,
    doc.rejection_notes = null,
    doc.rejected_at = null,
    doc.rejected_by_user_id = null;

// Create index for efficient filtering by status
CREATE INDEX document_status IF NOT EXISTS FOR (d:ProcessedDocument) ON (d.status);

// Create relationship to User node for rejection audit trail
// (Assuming User nodes exist)
MATCH (doc:ProcessedDocument), (user:User)
WHERE doc.rejected_by_user_id = user.id
CREATE (user)-[:REJECTED]->(doc);
```

**Allowed values:**
- `status`: 'PROCESSING', 'PROCESSED', 'REJECTED', 'REVIEW_REQUIRED'
- `rejection_reason`: 'NOT_EHS_DOCUMENT', 'UNREADABLE_SCAN', 'DUPLICATE_DOCUMENT', 'MISSING_DATA', 'OTHER'

### 4.4. API Endpoint Modifications

The primary endpoint for listing documents will need a filter for status.

**Modified Endpoint:** `GET /api/v1/documents?status=<status>`

-   By default, this endpoint should return only `PROCESSED` documents (`status=PROCESSED`).
-   To view rejected documents, the client will call `GET /api/v1/documents?status=REJECTED`.

A new endpoint is required to mark a document as rejected.

**New Endpoint:** `POST /api/v1/documents/{document_id}/reject`

-   **Authorization:** Requires user with review/admin privileges.
-   **Request Body:**
    ```json
    {
      "reason": "NOT_EHS_DOCUMENT",
      "notes": "This is an invoice for marketing services."
    }
    ```
-   **Backend Logic:** Updates the document's `status` property to `REJECTED` and populates the `rejection_reason`, `rejection_notes`, `rejected_at` (with the current timestamp), and `rejected_by_user_id` (with the authenticated user's ID) properties. Also creates a `REJECTED` relationship between the User and ProcessedDocument nodes.

### 4.5. UI/UX Design Considerations

-   A new section/tab in the UI labeled "Rejected Documents" will be created.
-   This view will list all documents with a `REJECTED` status, showing the filename, upload date, rejection reason, **who rejected it, and when**.
-   In the document review queue (if one exists), a "Reject" button should be added. Clicking it would open a modal for the user to select a reason and add optional notes.
-   The main dashboard/document list should, by default, only show `PROCESSED` documents to keep the view clean.

### 4.6. Implementation Steps

1.  Update the Neo4j database schema to add the new properties to `ProcessedDocument` nodes and create necessary indexes.
2.  Update the document processing logic to set the initial `status` property to `PROCESSING` and the final status to `PROCESSED` upon success.
3.  Implement the `POST /api/v1/documents/{document_id}/reject` endpoint with proper Cypher queries to update node properties and create relationships.
4.  Modify the `GET /api/v1/documents` endpoint to filter by the `status` property using Cypher queries and default to `PROCESSED`.
5.  Develop the new "Rejected Documents" view in the frontend.
6.  Integrate the rejection functionality into the document review workflow UI.

### 4.7. Testing Strategy

-   **Unit Tests:** Test the logic in the rejection API endpoint.
-   **Integration Tests:** Ensure that rejecting a document correctly updates all relevant properties (`status`, `rejection_reason`, `rejected_at`, `rejected_by_user_id`) in the Neo4j database, creates the appropriate `REJECTED` relationship, and that it no longer appears in the default document list.
-   **Manual Tests:** Test the full rejection workflow from the UI. Verify that all rejection reasons can be selected and saved correctly, and the audit data is displayed.

### 4.8. Future Considerations

A flow to "un-reject" or revert a rejection is out of scope for Phase 1. Future enhancements should consider this functionality, which would involve changing the document's status back to a processable state (e.g., `REVIEW_REQUIRED`), clearing the rejection-related properties, and removing the `REJECTED` relationship.

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
3. **Rejection Tracking:** Users can mark documents as rejected with appropriate reasons, and these rejections are fully audited and visible in a dedicated UI section.
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