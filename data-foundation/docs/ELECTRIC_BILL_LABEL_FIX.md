# Electric Bill Document Type Label Fix

## Issue Description
Electric bills were being displayed as "Unknown" document type in the Processed Documents table in the web UI.

## Root Cause
There was a mismatch between the Neo4j node labels and the label checking logic in the API:

1. The document recognition service (`document_recognition_service.py`) correctly identifies electric bills as `electricity_bill` type
2. This gets stored in Neo4j with the label `Electricitybill`
3. However, the API endpoint in `data_management.py` was checking for `Utilitybill` or `UtilityBill` labels
4. This mismatch caused electric bills to fall through to the "Unknown" category

## Solution
Updated the label checking logic in `/data-foundation/web-app/backend/routers/data_management.py`:

```python
# Before:
if 'Utilitybill' in labels or 'UtilityBill' in labels:
    doc_type = 'Electric Bill'

# After:
if 'Electricitybill' in labels or 'ElectricityBill' in labels:
    doc_type = 'Electric Bill'
```

## Technical Details
- **File Modified**: `/data-foundation/web-app/backend/routers/data_management.py`
- **Functions Affected**: `get_processed_documents()` and `get_document_details()`
- **Neo4j Label**: Documents are labeled as `Electricitybill` (not `Utilitybill`)
- **Document Recognition Type**: `electricity_bill` (as defined in `DOCUMENT_PATTERNS`)

## Testing
After applying the fix and restarting the backend server, electric bills now correctly display as "Electric Bill" instead of "Unknown" in the Processed Documents table.

## Related Components
- Document Recognition Service: `/data-foundation/backend/src/recognition/document_recognition_service.py`
- Ingestion Workflow: `/data-foundation/backend/src/workflows/ingestion_workflow.py`
- Data Management API: `/data-foundation/web-app/backend/routers/data_management.py`