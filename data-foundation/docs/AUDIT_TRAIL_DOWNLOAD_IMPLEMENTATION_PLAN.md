# Audit Trail Document Download Implementation Plan

## Overview
Add a download icon column to the Processed Documents table that allows users to download the original input files for verification purposes.

## Current State Analysis
- Documents in Neo4j have `file_path` property with original file location
- Web UI shows processed documents but no download capability
- Backend API doesn't expose file_path in responses
- Files are stored locally at paths like `/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/`

## Implementation Plan

### Phase 1: Backend Updates

#### 1.1 Update Processed Documents API Response
- Modify `/api/data/processed-documents` endpoint to include file_path
- Add file_path to the query in `data_management.py` line 163
- Include file_path in the response object

#### 1.2 Create File Download Endpoint
- Add new endpoint: `GET /api/data/documents/{document_id}/download`
- Retrieve file_path from Neo4j for the given document_id
- Stream the file back to the client with proper headers
- Handle file not found errors gracefully

### Phase 2: Frontend Updates

#### 2.1 Add Download Column to Table
- Add new table header "Download" after "Document Type"
- Add icon column with document download icon
- Make icon clickable with hover effects

#### 2.2 Implement Download Functionality
- Add click handler to download icon
- Call download endpoint with document_id
- Trigger browser file download
- Handle errors with user-friendly messages

### Phase 3: Security Considerations

#### 3.1 Access Control
- Validate user has permission to download documents
- Prevent directory traversal attacks
- Sanitize file paths

#### 3.2 Audit Logging
- Log all download attempts
- Track user, timestamp, and document downloaded

## Technical Details

### Backend Changes

1. **Update data_management.py query (line 163)**:
```python
result = session.run("""
    MATCH (d:Document)
    RETURN d.id as id, 
           d.type as document_type,
           d.created_at as date_received,
           d.file_path as file_path,  // ADD THIS
           labels(d) as labels,
           d.account_number as account_number,
           d.service_address as site
    ORDER BY d.created_at DESC
""")
```

2. **Add to response object (line 185)**:
```python
doc = {
    'id': record['id'],
    'document_type': doc_type,
    'date_received': record['date_received'] or datetime.now().isoformat(),
    'site': record['site'] or record['account_number'] or 'Main Facility',
    'file_path': record['file_path']  // ADD THIS
}
```

3. **Create download endpoint**:
```python
@router.get("/documents/{document_id}/download")
async def download_document(document_id: str):
    # Get file_path from Neo4j
    # Validate file exists and is accessible
    # Return FileResponse with proper headers
```

### Frontend Changes

1. **Update table headers**:
```javascript
<thead>
  <tr>
    <th>#</th>
    <th>Date Received</th>
    <th>Site</th>
    <th>Document Type</th>
    <th>Download</th>  // ADD THIS
  </tr>
</thead>
```

2. **Add download icon column**:
```javascript
<td>
  <button 
    className="download-icon-btn"
    onClick={() => handleDownload(doc.id, doc.file_path)}
    title="Download original document"
  >
    📄
  </button>
</td>
```

3. **Implement download handler**:
```javascript
const handleDownload = async (documentId, filePath) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/data/documents/${documentId}/download`);
    if (!response.ok) throw new Error('Download failed');
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filePath.split('/').pop();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Download error:', error);
    alert('Failed to download document');
  }
};
```

## Testing Plan

1. Verify file_path is returned in API response
2. Test download endpoint with valid document IDs
3. Test error handling for missing files
4. Verify UI shows download icons
5. Test actual file downloads in browser
6. Verify security measures prevent unauthorized access

## Future Enhancements

1. Add download progress indicator for large files
2. Support batch downloads
3. Add file preview capability
4. Implement download history tracking
5. Add file type icons based on extension