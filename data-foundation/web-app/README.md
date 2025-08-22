# EHS Compliance Platform Web Application

A modern web application for Environmental, Health, and Safety (EHS) compliance management built with FastAPI and React.

## Overview

This web application provides a user-friendly interface for managing EHS data and analytics. It features a professional corporate design with two main sections:
- **Data Management**: Upload, view, and manage incident reports and compliance records
- **Analytics**: View dashboard metrics and perform data analysis

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework for building APIs
- **Python 3.11+**: Backend programming language
- **Pydantic**: Data validation using Python type annotations
- **Pandas**: Data manipulation and analysis
- **CORS**: Cross-Origin Resource Sharing enabled

### Frontend
- **React 18**: JavaScript library for building user interfaces
- **React Router**: Client-side routing
- **Axios**: HTTP client for API requests
- **CSS3**: Modern styling with corporate theme

## Project Structure

```
web-app/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── requirements.txt     # Python dependencies
│   ├── models.py           # Pydantic data models
│   ├── .env.example        # Environment variables template
│   └── routers/
│       ├── data_management.py  # Data management endpoints
│       └── analytics.py        # Analytics endpoints
└── frontend/
    ├── package.json        # Node.js dependencies
    ├── public/
    │   └── index.html     # HTML template
    └── src/
        ├── index.js       # React entry point
        ├── App.js         # Main application component
        ├── App.css        # Application styles
        ├── index.css      # Global styles
        └── components/
            ├── Sidebar.js          # Navigation sidebar
            ├── Sidebar.css         # Sidebar styles
            ├── DataManagement.js   # Data management view
            └── Analytics.js        # Analytics dashboard

```

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Node.js 16 or higher
- npm or yarn package manager

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create environment file:
   ```bash
   cp .env.example .env
   ```

5. Run the backend server:
   ```bash
   python main.py
   ```

   The API will be available at http://localhost:8006

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   The application will open at http://localhost:3000

## API Endpoints

### Data Management
- `GET /api/data/incidents` - Retrieve all incidents
- `POST /api/data/incidents` - Create new incident
- `GET /api/data/compliance` - Retrieve compliance records
- `POST /api/data/compliance` - Create compliance record
- `POST /api/data/upload` - Upload data file (CSV/Excel)

### Analytics
- `GET /api/analytics/dashboard` - Get dashboard metrics
- `POST /api/analytics/query` - Execute analytics query

### Health Check
- `GET /health` - API health status

## Features

### Data Management
- View and manage incident reports
- Track compliance records
- Upload data files (CSV/Excel format)
- Filter and sort data tables
- Status tracking (Open/Closed/In Progress)

### Analytics Dashboard
- Key performance metrics
- Total incidents and compliance rate
- Average resolution time
- Active issues tracking
- Custom analytics queries

## UI Design

The application features a professional corporate design:
- **Header**: Dark blue (#2c4a5e) with white text
- **Sidebar**: Light gray (#f5f5f5) with folder icons
- **Content**: White background with clean typography
- **Accent**: Blue (#3498db) for active states and links
- **Status Colors**: 
  - Success: Green (#27ae60)
  - Warning: Orange (#f39c12)
  - Error: Red (#e74c3c)

## Development

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Style
- Backend: Follow PEP 8 Python style guide
- Frontend: ESLint configuration included

## Deployment

### Production Build

1. Backend:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8006 --workers 4
   ```

2. Frontend:
   ```bash
   npm run build
   ```

## Integration with EHS Data Foundation

This web application is designed to integrate with the EHS Data Foundation system:
- Connect to Neo4j knowledge graph for data retrieval
- Use extraction workflows for analytics
- Display ingested document data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is part of the EHS AI Demo system.

## Support

For issues or questions, please contact the EHS AI team.