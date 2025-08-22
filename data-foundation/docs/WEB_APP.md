# EHS AI Assistant Web Application

## Overview
A modern web application for Environmental, Health, and Safety (EHS) data management and analytics, built on top of the EHS Data Foundation system. This application provides a user-friendly interface for ingesting EHS documents and viewing processed data.

## Architecture

### Technology Stack
- **Frontend**: React 18 with React Router
- **Backend**: FastAPI with Python 3.11+
- **Database**: Neo4j (via existing Data Foundation)
- **Styling**: Custom CSS with corporate EHS theme
- **Integration**: RESTful APIs connecting to existing ingestion/extraction services

### System Architecture
```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   React Frontend    │────▶│   FastAPI Backend   │────▶│   Neo4j Database    │
│    (Port 3001)      │     │    (Port 8006)      │     │    (Port 7687)      │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Ingestion API      │
                            │    (Port 8005)      │
                            └─────────────────────┘
```

## Features

### Data Management
- **Batch Document Ingestion**: One-click ingestion of all EHS documents
- **Real-time Progress Tracking**: Animated 8-stage progress indicator
- **Processed Documents View**: Table showing ingested documents with metadata
- **Automatic Data Refresh**: Table updates automatically after ingestion
- **Document Details Popup**: Hover over table rows to see extracted document data
- **Reset Functionality**: Clear processed documents table with one click

### User Interface
- **Professional Design**: Corporate EHS theme with dark blue header (#2c4a5e)
- **Sidebar Navigation**: Easy access to Data Management and Analytics sections
- **Responsive Layout**: Mobile-friendly design with proper table formatting
- **Clean UX**: No popup alerts, smooth animations, inline feedback
- **Interactive Hover Popups**: Document details appear on row hover with 0.6s fade-in

### Progress Indicator Stages
1. **Validate**: Initial document validation
2. **Parse**: Document parsing with LlamaParse
3. **Extract**: Entity and relationship extraction
4. **Transform**: Data transformation and structuring
5. **Validate Data**: Data quality validation
6. **Load**: Loading into Neo4j
7. **Index**: Creating search indexes
8. **Complete**: Finalization

## Project Structure

```
web-app/
├── backend/
│   ├── main.py                     # FastAPI application entry
│   ├── requirements.txt            # Python dependencies
│   ├── models.py                   # Pydantic models
│   ├── .env.example               # Environment template
│   └── routers/
│       ├── data_management.py      # Data endpoints
│       └── analytics.py            # Analytics endpoints
└── frontend/
    ├── package.json               # React dependencies
    ├── public/
    │   └── index.html            # HTML template
    └── src/
        ├── index.js              # React entry point
        ├── App.js                # Main app component
        ├── App.css               # App styles
        └── components/
            ├── Sidebar.js         # Navigation sidebar
            ├── DataManagement.js  # Data management view with hover popup
            ├── Analytics.js       # Analytics dashboard
            ├── IngestionProgress.js    # Progress indicator
            └── IngestionProgress.css   # Progress styles
```

## API Endpoints

### Backend API (Port 8006)

#### Data Management
- `GET /api/data/processed-documents` - Retrieve all processed documents from Neo4j
- `GET /api/data/processed-documents/{document_id}` - Get detailed document data with extracted fields
- `GET /api/data/incidents` - Get incident reports (mock data)
- `POST /api/data/upload` - Upload data file

#### Analytics
- `GET /api/analytics/dashboard` - Get dashboard metrics
- `POST /api/analytics/query` - Execute analytics query

#### Health Check
- `GET /` - Root endpoint
- `GET /health` - API health status

### Integration with Ingestion API (Port 8005)
- `POST /api/v1/ingest/batch` - Trigger batch ingestion of all documents

## Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 16+
- Neo4j database running
- EHS Data Foundation system set up

### Backend Setup

1. Navigate to backend directory:
```bash
cd web-app/backend
```

2. Use the data-foundation virtual environment:
```bash
source /path/to/data-foundation/.venv/bin/activate
```

3. Install additional dependencies:
```bash
uv pip install fastapi uvicorn python-multipart neo4j
```

4. Run the backend:
```bash
python main.py
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd web-app/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

## Configuration

### Environment Variables
Backend uses environment variables from the data-foundation system:
- `NEO4J_URI`: Neo4j connection URI (default: bolt://localhost:7687)
- `NEO4J_USERNAME`: Neo4j username (default: neo4j)
- `NEO4J_PASSWORD`: Neo4j password (default: EhsAI2024!)

### CORS Configuration
The backend is configured to accept requests from:
- `http://localhost:3000`
- `http://localhost:3001`

## Data Flow

### Ingestion Process
1. User clicks "Ingest Invoices" button
2. Frontend calls `/api/v1/ingest/batch` on port 8005
3. Progress indicator shows 8 stages of processing
4. Backend ingestion script processes 3 documents:
   - Electric Bill (PDF)
   - Water Bill (PDF)
   - Waste Manifest (PDF)
5. Documents are parsed, extracted, and loaded into Neo4j
6. Frontend automatically refreshes to show processed documents

### Document Display
The Processed Documents table shows:
- **#**: Sequential row number
- **Date Received**: Formatted timestamp (e.g., "Aug 18, 2025, 10:30 AM")
- **Site**: Facility or location name
- **Document Type**: Electric Bill, Water Bill, or Waste Manifest

## UI Components

### Sidebar Navigation
- **Data Management**: Shows document count badge
- **Analytics**: Dashboard and reporting features

### Data Management View
- **Data Upload Section**: Contains "Ingest Invoices" button with inline progress
- **Processed Documents Table**: Displays ingested documents with sorting
- **Reset Button**: Positioned below the table to clear all processed documents

### Document Details Popup
- **Hover Activation**: Appears when hovering over any table row
- **Fade-in Animation**: 0.6s smooth transition with 0.1s delay
- **Dynamic Content**: Shows document-specific extracted data:
  - **Electric Bills**: Account number, kWh usage, costs, billing period
  - **Water Bills**: Gallons used, costs, various charges
  - **Waste Manifests**: Tracking number, quantity, issue date
- **Smart Positioning**: Follows mouse cursor with offset
- **Auto-hide**: Disappears when mouse leaves the row

### Progress Indicator
- **Inline Display**: Appears to the right of the button during ingestion
- **Visual States**: 
  - ○ Pending (gray)
  - ⟳ Active (blue with pulse animation)
  - ✓ Complete (green)
- **Progress Bar**: Gradient fill with shimmer effect

## Styling

### Color Scheme
- **Header**: #2c4a5e (dark blue)
- **Sidebar**: #f5f5f5 (light gray)
- **Primary Button**: #007bff (blue)
- **Success**: #27ae60 (green)
- **Active/Hover**: #3498db (light blue)

### CSS Classes
- `.header`: Top navigation bar
- `.sidebar`: Left navigation panel
- `.data-table`: Main data display tables
- `.btn-primary`: Primary action buttons
- `.ingestion-progress`: Progress indicator container

## Development

### Running in Development Mode
Both frontend and backend support hot-reloading:
- Frontend: Changes auto-refresh at http://localhost:3001
- Backend: Use `--reload` flag with uvicorn for auto-restart

### Debugging
- Frontend: Use browser DevTools, React DevTools
- Backend: FastAPI automatic docs at http://localhost:8006/docs
- Check console for detailed logging

## Production Deployment

### Backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8006 --workers 4
```

### Frontend
```bash
npm run build
# Serve the build directory with any static file server
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend allows your frontend port
2. **Neo4j Connection**: Verify credentials and connection string
3. **Missing Dependencies**: Install `neo4j` package in backend
4. **Port Conflicts**: Check if ports 3001, 8005, 8006 are available

### Log Locations
- Frontend: Browser console
- Backend: Terminal output
- Ingestion: Check data-foundation logs

## Future Enhancements

1. **Authentication**: Add user login and role-based access
2. **Real-time Updates**: WebSocket integration for live data
3. **Export Functionality**: Download processed data as CSV/Excel
4. **Advanced Analytics**: Integration with extraction workflows
5. **Multi-tenant Support**: Facility-specific data isolation
6. **Audit Trail**: Track all user actions and data changes

## Integration Points

This web application integrates with:
- **EHS Data Foundation**: Core ingestion and extraction system
- **Neo4j Knowledge Graph**: Document and entity storage
- **Extraction API**: For analytics and reporting features

## License

Part of the EHS AI Demo system.

## Support

For issues or questions related to:
- Web app: Check this documentation
- Ingestion: See INGESTION_WORKFLOW.md
- Extraction: See EXTRACTION_WORKFLOW.md