# EHS AI Demo - Phase 1 Enhancement Plan

**Document Version:** 1.1
**Date:** 2025-08-22
**Status:** Revised Proposal

## 1. Introduction

This document outlines the technical specification and implementation plan for Phase 1 enhancements to the EHS AI Demo project. The goal of this phase is to improve data auditability, financial accuracy, and data quality management. This plan serves as a guide for the development team, covering database modifications, API changes, and UI/UX considerations.

The three core enhancements for Phase 1 are:
1.  **Audit Trail Enhancement:** Tracking the source file for each processed document.
2.  **Utility Bill Pro-Rating:** Allocating costs and usage from billing periods to their corresponding calendar months.
3.  **Document Rejection Tracking:** Implementing a system to manage and track documents that are identified as invalid or irrelevant.

---

## 2. Enhancement 1: Audit Trail

### 2.1. Current State

Currently, once a document is uploaded and processed, the system extracts the relevant data but does not maintain a link back to the original source file. This makes it difficult to perform audits or manually verify the data extraction against the original document.

### 2.2. Requirements

-   The system must store a reference to the original uploaded file for every processed document.
-   The original filename must be preserved and displayed to the user.
-   Users with appropriate permissions must be able to view or download the original source file from the application.
-   The file storage solution should be scalable and secure, preventing unauthorized access and filename collisions.

### 2.3. Database Schema Changes

We will add two new columns to the `processed_documents` table to store the original filename for display and a unique path for retrieval from object storage.

**Table:** `processed_documents`

```sql
ALTER TABLE processed_documents
ADD COLUMN original_filename VARCHAR(255) NULL,
ADD COLUMN source_file_path VARCHAR(1024) NULL;
```

-   `original_filename`: Stores the original name of the uploaded file (e.g., `invoice_july_2023.pdf`).
-   `source_file_path`: Stores the unique path in the object store (e.g., S3, GCS, Azure Blob). The recommended format is `uploads/{uuid}/{sanitized_filename}` to prevent collisions. The filename component of the path must be sanitized to remove characters that could be problematic for object stores or URLs.

### 2.4. API Endpoint Modifications

A new API endpoint will be created to provide a secure, time-limited download URL for a given document's source file.

**Endpoint:** `GET /api/v1/documents/{document_id}/source_url`

-   **Description:** Retrieves a pre-signed URL for downloading the original source file.
-   **Authorization:** Requires user to be authenticated and have permission to view the specified document.
-   **Success Response (200 OK):**
    ```json
    {
      "download_url": "https://s3.amazonaws.com/bucket-name/uploads/uuid/invoice.pdf?AWSAccessKeyId=...",
      "original_filename": "invoice_july_2023.pdf",
      "expires_in": 300
    }
    ```
-   **Failure Response (404 Not Found):** If the document or source file does not exist.
-   **Failure Response (403 Forbidden):** If the user is not authorized.

### 2.5. UI/UX Design Considerations

-   The list view for processed documents should include a new column titled "Source File".
-   The value in this column will be a clickable link displaying the `original_filename`.
-   Clicking the link will trigger a call to the new API endpoint and initiate a download of the file in the user's browser.
-   In the detailed view for a single document, a prominent "Download Original" button or link should be present.

### 2.6. Implementation Steps

1.  Configure and provision a cloud object storage bucket (e.g., AWS S3).
2.  Update the database schema by running the migration script for `processed_documents`.
3.  Modify the document upload service to ensure an **atomic operation**:
    a. Sanitize the original filename for safe storage.
    b. Generate a unique path (e.g., `uploads/{uuid}/{sanitized_filename}`).
    c. Upload the file to the object storage bucket.
    d. **Only upon successful upload**, create the database record in `processed_documents` with the `original_filename` and `source_file_path`. This prevents orphaned database records.
4.  Implement the new backend API endpoint (`GET /api/v1/documents/{document_id}/source_url`) including logic for generating pre-signed URLs.
5.  Update the frontend to display the source file link in the UI and handle the download action.

### 2.7. Testing Strategy

-   **Unit Tests:** Verify file upload logic, path generation, **filename sanitization**, and pre-signed URL generation.
-   **Integration Tests:** Test the end-to-end flow from file upload to database persistence. Test the API endpoint authorization and response. **Verify that a failed upload does not create a database record.**
-   **Manual Tests:** Verify that files can be successfully uploaded and downloaded through the UI. Test with filenames containing special characters.

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

To support efficient reporting, we will pre-calculate and store the allocations in a new table. This avoids complex on-the-fly calculations and simplifies analytics.

**New Table:** `monthly_usage_allocations`

```sql
CREATE TABLE monthly_usage_allocations (
    id SERIAL PRIMARY KEY,
    processed_document_id INT NOT NULL REFERENCES processed_documents(id) ON DELETE CASCADE,
    usage_year INT NOT NULL,
    usage_month INT NOT NULL,
    allocated_usage DECIMAL(12, 4) NOT NULL,
    allocated_cost DECIMAL(12, 4) NOT NULL,
    -- Add other metrics to be allocated (e.g., water, gas)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_monthly_allocation UNIQUE (processed_document_id, usage_year, usage_month)
);

CREATE INDEX idx_monthly_allocations_year_month ON monthly_usage_allocations (usage_year, usage_month);
```

### 3.4. API Endpoint Modifications

Existing reporting endpoints will be modified to query the new `monthly_usage_allocations` table instead of the `processed_documents` table for monthly summaries.

**Modified Endpoint:** `GET /api/v1/reports/monthly_summary?year=<year>&month=<month>`

-   **Description:** Retrieves a summary of all usage and costs for a given calendar month.
-   **Backend Logic:** The endpoint will now perform a `SUM` and `GROUP BY` on the `monthly_usage_allocations` table, filtering by `usage_year` and `usage_month`.

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

1.  Create and run the database migration for the `monthly_usage_allocations` table.
2.  Develop a robust pro-rating calculation service/module. This must correctly handle leap years and varying month lengths.
3.  Integrate this service into the document processing pipeline. After a document's data is extracted, the service will calculate and insert the allocation records.
4.  Update all relevant reporting APIs and services to query the new allocations table.
5.  Update the frontend to display the allocation breakdown on the document detail page.
6.  **(Mandatory)** Create a one-time backfill script to process all existing documents and populate the `monthly_usage_allocations` table to ensure historical data consistency.
7.  Implement logic to handle updates to `processed_documents`. If a document's billing period or financial data is modified, its corresponding records in `monthly_usage_allocations` must be deleted and recalculated to maintain data integrity.

### 3.7. Testing Strategy

-   **Unit Tests:** Extensively test the pro-rating logic with edge cases:
    -   Billing periods fully within one month.
    -   Periods spanning two months.
    -   Periods spanning a year-end (e.g., Dec 15 - Jan 15).
    -   Periods spanning a leap day.
-   **Integration Tests:** Verify that processing a document correctly creates records in both `processed_documents` and `monthly_usage_allocations`. **Test that updating a document correctly triggers recalculation of allocations.**
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

We will add status, reason, and audit columns to the `processed_documents` table. This is the simplest approach for Phase 1 and avoids creating a separate table. An index on the `status` column will ensure efficient querying.

**Table:** `processed_documents`

```sql
-- Define an ENUM type for standardized statuses and reasons
CREATE TYPE document_status AS ENUM ('PROCESSING', 'PROCESSED', 'REJECTED', 'REVIEW_REQUIRED');
CREATE TYPE rejection_reason AS ENUM ('NOT_EHS_DOCUMENT', 'UNREADABLE_SCAN', 'DUPLICATE_DOCUMENT', 'MISSING_DATA', 'OTHER');

ALTER TABLE processed_documents
ADD COLUMN status document_status NOT NULL DEFAULT 'PROCESSING',
ADD COLUMN rejection_reason rejection_reason NULL,
ADD COLUMN rejection_notes TEXT NULL, -- For 'OTHER' reason
ADD COLUMN rejected_at TIMESTAMP WITH TIME ZONE NULL,
ADD COLUMN rejected_by_user_id INT NULL REFERENCES users(id); -- Assuming a 'users' table exists

-- Add an index for efficient filtering
CREATE INDEX idx_documents_status ON processed_documents (status);
```

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
-   **Backend Logic:** Updates the document's `status` to `REJECTED` and populates the `rejection_reason`, `rejection_notes`, `rejected_at` (with the current timestamp), and `rejected_by_user_id` (with the authenticated user's ID) fields.

### 4.5. UI/UX Design Considerations

-   A new section/tab in the UI labeled "Rejected Documents" will be created.
-   This view will list all documents with a `REJECTED` status, showing the filename, upload date, rejection reason, **who rejected it, and when**.
-   In the document review queue (if one exists), a "Reject" button should be added. Clicking it would open a modal for the user to select a reason and add optional notes.
-   The main dashboard/document list should, by default, only show `PROCESSED` documents to keep the view clean.

### 4.6. Implementation Steps

1.  Create and run the database migration to add the new columns and types.
2.  Update the document processing logic to set the initial `status` to `PROCESSING` and the final status to `PROCESSED` upon success.
3.  Implement the `POST /api/v1/documents/{document_id}/reject` endpoint.
4.  Modify the `GET /api/v1/documents` endpoint to filter by the `status` query parameter and default to `PROCESSED`.
5.  Develop the new "Rejected Documents" view in the frontend.
6.  Integrate the rejection functionality into the document review workflow UI.

### 4.7. Testing Strategy

-   **Unit Tests:** Test the logic in the rejection API endpoint.
-   **Integration Tests:** Ensure that rejecting a document correctly updates all relevant fields (`status`, `rejection_reason`, `rejected_at`, `rejected_by_user_id`) in the database and that it no longer appears in the default document list.
-   **Manual Tests:** Test the full rejection workflow from the UI. Verify that all rejection reasons can be selected and saved correctly, and the audit data is displayed.

### 4.8. Future Considerations

A flow to "un-reject" or revert a rejection is out of scope for Phase 1. Future enhancements should consider this functionality, which would involve changing the document's status back to a processable state (e.g., `REVIEW_REQUIRED`) and clearing the rejection-related fields.

### 4.9. Estimated Effort

-   **Effort:** Medium (M)
-   **Timeline:** ~1.5 weeks

---

## 5. Overall Timeline & Dependencies

-   **Total Estimated Timeline:** ~5 weeks
-   **Dependencies:** The Audit Trail enhancement should ideally be implemented first, as the other features will build upon the same `processed_documents` table. The Pro-Rating and Rejection features can be developed in parallel if resources allow.

## 6. Risk Assessment & Mitigation

### 6.1. Technical Risks

-   **Object Storage Dependencies:** The audit trail feature introduces a dependency on cloud object storage. Risk mitigation includes implementing proper error handling, retry logic, and fallback mechanisms.
-   **Pro-Rating Calculation Complexity:** The date-based allocation logic, especially with updates, must handle all edge cases correctly. Extensive unit testing and validation against known scenarios will mitigate this risk.
-   **Database Performance:** Adding new columns, tables, and indexes could impact query performance. Database indexing and query optimization will be essential.

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

This plan provides a clear path forward for Phase 1. All technical decisions should be reviewed and confirmed before beginning implementation.