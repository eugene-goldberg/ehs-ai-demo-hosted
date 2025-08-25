# Simple Rejection API Implementation

## Overview

This document describes the implementation of a new API endpoint that wraps RejectedDocument nodes from Neo4j and returns them in a format compatible with the Phase 1 rejection tracking API.

## Implementation Details

### New Endpoint

**URL:** `/api/v1/simple-rejected-documents`  
**Method:** GET  
**Description:** Returns rejected documents in Phase 1 compatible format

### Response Format

The endpoint returns data in the exact format expected by the Phase 1 rejection tracking API:

```json
{
  "documents": [
    {
      "document_id": "invoice_20250824_134323_901",
      "file_name": "invoice.pdf", 
      "rejection_reason": "Document type could not be determined (confidence: 0.863)",
      "rejection_status": "rejected",
      "created_at": "2025-08-24T18:43:51.673803",
      "notes": "Attempted type: unknown; Confidence: 0.863; File size: 381709 bytes; Pages: 1; Source: comprehensive_batch_ingestion"
    }
  ],
  "pagination": {
    "total": 1,
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

### Property Mapping

The API maps RejectedDocument node properties to Phase 1 format as follows:

| Phase 1 Field | RejectedDocument Property | Description |
|---------------|--------------------------|-------------|
| `document_id` | `r.id` | Unique document identifier |
| `file_name` | `r.original_filename` | Original file name |
| `rejection_reason` | `r.rejection_reason` | Reason for rejection |
| `rejection_status` | `"rejected"` (hardcoded) | Always "rejected" for RejectedDocument nodes |
| `created_at` | `r.rejected_at` or `r.upload_timestamp` | Document timestamp |
| `notes` | Constructed from metadata | Combines attempted_type, confidence, file_size, page_count, upload_source |

### Query Parameters

- `limit` (optional): Maximum number of results (1-1000, default: 50)
- `offset` (optional): Pagination offset (default: 0)

### Health Check Endpoint

**URL:** `/api/v1/simple-rejected-documents/health`  
**Method:** GET  
**Description:** Health check for the simple rejection API

Returns:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-24T13:58:45.385198",
  "service": "simple_rejection_api", 
  "version": "1.0.0",
  "database_connection": "ok",
  "rejected_documents_count": 1
}
```

## Files Created

1. `/src/api/simple_rejection_api.py` - Main API implementation
2. `/src/api/__init__.py` - Package initialization
3. Updated `/src/ehs_extraction_api.py` - Added router integration

## Neo4j Query

The API uses this Cypher query to retrieve RejectedDocument nodes:

```cypher
MATCH (r:RejectedDocument)
RETURN r.id as document_id,
       r.original_filename as file_name,
       r.rejection_reason as rejection_reason,
       r.rejected_at as rejected_at,
       r.upload_timestamp as upload_timestamp,
       r.attempted_type as attempted_type,
       r.confidence as confidence,
       r.file_size as file_size,
       r.page_count as page_count,
       r.content_length as content_length,
       r.upload_source as upload_source
ORDER BY COALESCE(r.rejected_at, r.upload_timestamp) DESC
SKIP $offset
LIMIT $limit
```

## Integration

The new API endpoint is automatically included when the EHS Extraction API starts. It integrates with the existing FastAPI application and uses the same Neo4j database connection configuration.

## Testing

The implementation includes comprehensive tests:
- Health check endpoint testing
- Main endpoint functionality testing  
- Pagination testing
- Database connectivity verification

All tests pass successfully, confirming the API works as expected.

## Usage Examples

### Basic Usage
```bash
curl "http://localhost:8000/api/v1/simple-rejected-documents"
```

### With Pagination
```bash
curl "http://localhost:8000/api/v1/simple-rejected-documents?limit=10&offset=0"
```

### Health Check
```bash
curl "http://localhost:8000/api/v1/simple-rejected-documents/health"
```

## Notes

- The API automatically maps Neo4j RejectedDocument properties to the Phase 1 expected format
- Comprehensive error handling and logging is included
- The API follows FastAPI best practices with proper response models and validation
- Database connections are properly managed and closed after each request
- The endpoint supports pagination for handling large numbers of rejected documents