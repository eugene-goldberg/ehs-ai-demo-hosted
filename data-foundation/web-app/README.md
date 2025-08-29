# EHS Compliance Platform Web Application

A comprehensive web application for Environmental, Health, and Safety (EHS) compliance management and analytics built with FastAPI and React. This application serves as the user interface layer for the EHS Data Foundation system, providing real-time access to processed documents, compliance metrics, and AI-powered insights.

## Overview

This web application provides a multi-faceted interface for EHS data management and analytics with the following main capabilities:

- **Data Management**: Upload, view, and manage incident reports, compliance records, and processed documents from Neo4j database
- **Analytics Dashboard**: View comprehensive dashboard metrics with executive-level insights
- **Hierarchical Analytics**: Multi-level data analysis with drill-down capabilities
- **Document Processing**: Integration with Neo4j knowledge graph for document retrieval and analysis
- **LangSmith Integration**: Trace monitoring and AI conversation tracking
- **Real-time Data**: Live integration with EHS Data Foundation backend services

## Technology Stack

### Backend
- **FastAPI 0.104.1**: Modern Python web framework for building APIs with automatic documentation
- **Python 3.11+**: Backend programming language with type hints
- **Pydantic 2.5.0**: Data validation and serialization using Python type annotations
- **SQLAlchemy 2.0.23**: Database ORM for PostgreSQL integration
- **Neo4j Python Driver**: Graph database integration for document management
- **Uvicorn 0.24.0**: ASGI server for production deployment
- **Python-JOSE 3.3.0**: JWT token handling for authentication
- **Pandas 2.1.4**: Data manipulation and analysis
- **NumPy 1.24.3**: Numerical computing support
- **Passlib & BCrypt**: Password hashing and security
- **Python-multipart**: File upload handling
- **Python-dotenv**: Environment configuration management

### Frontend
- **React 18.2.0**: Modern JavaScript library for building user interfaces
- **React Router DOM 6.8.1**: Client-side routing and navigation
- **Axios 1.3.4**: HTTP client for API requests with interceptors
- **Chart.js 4.2.1 & React-ChartJS-2 5.2.0**: Data visualization and charting
- **Material-UI 5.11.10**: Component library for modern UI design
- **Emotion React/Styled**: CSS-in-JS styling solution
- **Date-fns 2.29.3**: Date manipulation and formatting utilities
- **CSS3**: Custom styling with corporate theme and responsive design

### Database & Storage
- **PostgreSQL**: Primary relational database for structured data
- **Neo4j Graph Database**: Knowledge graph for document relationships and analytics
- **File System**: Local file storage for uploads and document management
- **JSON**: Transcript and configuration data storage

## Project Structure

```
web-app/
├── README.md                    # This comprehensive documentation
├── test_neo4j_documents.py      # Neo4j database testing script
├── venv/                        # Python virtual environment
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entry point
│   ├── requirements.txt         # Python dependencies with specific versions
│   ├── models.py               # Pydantic data models and validation schemas
│   ├── .env.example            # Environment variables template
│   ├── backend.log             # Application logs
│   ├── transcript_data.json    # LLM conversation transcript storage
│   ├── venv/                   # Backend virtual environment
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── data_management.py  # Data management, Neo4j integration, document handling
│   │   └── analytics.py        # Analytics endpoints and dashboard metrics
│   └── src/
│       ├── langsmith_fetcher.py        # LangSmith client integration
│       └── api/
│           └── langsmith_traces_api.py # LangSmith API endpoints
└── frontend/
    ├── package.json            # Node.js dependencies with specific versions
    ├── package-lock.json       # Dependency lock file
    ├── build/                  # Production build output
    ├── node_modules/           # Node.js dependencies
    ├── public/
    │   └── index.html         # HTML template with Meta tags
    └── src/
        ├── index.js           # React application entry point
        ├── index.css          # Global styles and CSS variables
        ├── App.js             # Main application component with routing
        ├── App.css            # Application-wide styles
        ├── config/
        │   └── api.js         # API configuration and endpoints
        └── components/
            ├── Analytics.js              # Standard analytics dashboard
            ├── AnalyticsExecutive.js     # Executive-level analytics view
            ├── AnalyticsHierarchical.js  # Hierarchical data analysis
            ├── Analytics_metrics_backup.js # Analytics backup implementation
            ├── DataManagement.js         # Document management interface
            ├── IngestionProgress.js      # Document ingestion progress tracking
            ├── IngestionProgress.css     # Ingestion progress styles
            ├── Sidebar.js               # Navigation sidebar with icons
            └── Sidebar.css              # Sidebar styling and animations
```

## Getting Started

### Prerequisites
- **Python 3.11+**: Backend runtime
- **Node.js 18+**: Frontend runtime and build tools
- **PostgreSQL 13+**: Primary database server
- **Neo4j 5.0+**: Graph database for document management
- **npm or yarn**: Package manager for frontend dependencies

### Environment Setup

#### 1. Database Configuration

**PostgreSQL Setup:**
```bash
# Create database
createdb ehs_compliance

# Verify connection
psql -d ehs_compliance -c "SELECT version();"
```

**Neo4j Setup:**
```bash
# Start Neo4j service
neo4j start

# Access browser interface at http://localhost:7474
# Default credentials: neo4j/neo4j
# Change password to: EhsAI2024!
```

#### 2. Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your configuration:
   ```bash
   # Database Configuration
   DATABASE_URL=postgresql://username:password@localhost:5432/ehs_compliance
   
   # Neo4j Configuration (add these to your .env)
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=EhsAI2024!
   
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8001
   DEBUG=True
   
   # Security
   SECRET_KEY=your-256-bit-secret-key-here
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   
   # CORS Settings
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
   
   # LangSmith Integration (optional)
   LANGSMITH_API_KEY=your-langsmith-api-key
   LANGSMITH_PROJECT=ehs-ai-demo
   ```

5. **Initialize database schemas (if using PostgreSQL):**
   ```bash
   # Run any database migration scripts
   python -c "from models import *; print('Models imported successfully')"
   ```

6. **Test Neo4j connection:**
   ```bash
   python test_neo4j_documents.py
   ```

7. **Run the backend server:**
   ```bash
   python main.py
   ```
   
   The API will be available at:
   - Main API: http://localhost:8001
   - Interactive Docs: http://localhost:8001/docs
   - ReDoc: http://localhost:8001/redoc

#### 3. Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure environment (optional):**
   ```bash
   # Create .env file for custom API URL
   echo "REACT_APP_API_URL=http://localhost:8001" > .env
   ```

4. **Start development server:**
   ```bash
   npm start
   ```
   
   The application will open at http://localhost:3000

### Quick Start Verification

1. **Check backend health:**
   ```bash
   curl http://localhost:8001/health
   ```

2. **Test API endpoints:**
   ```bash
   # Test data endpoints
   curl http://localhost:8001/api/data/processed-documents
   
   # Test analytics endpoints
   curl http://localhost:8001/api/analytics/dashboard
   
   # Test LangSmith integration
   curl http://localhost:8001/api/langsmith/health
   ```

3. **Access frontend:** Open http://localhost:3000 in your browser

4. **Verify Neo4j integration:** Use the Data Management section to view processed documents

## API Documentation

### Core Endpoints

#### Health & Status
- `GET /health` - API health status
- `GET /` - API root information

#### Data Management
**Incidents:**
- `GET /api/data/incidents` - Retrieve all incident reports with mock data
- `POST /api/data/incidents` - Create new incident report
- `GET /api/data/incidents/{incident_id}` - Get specific incident by ID
- `PUT /api/data/incidents/{incident_id}` - Update existing incident

**Compliance Records:**
- `GET /api/data/compliance` - Retrieve compliance records with mock data
- `POST /api/data/compliance` - Create new compliance record

**File Operations:**
- `POST /api/data/upload` - Upload data files (CSV/Excel) with validation

**Document Management (Neo4j Integration):**
- `GET /api/data/processed-documents` - Retrieve all processed documents from Neo4j
- `GET /api/data/processed-documents/{document_id}` - Get detailed document information
- `GET /api/data/documents/{document_id}/download` - Download document file

**Transcript Management:**
- `GET /api/data/transcript` - Get LLM conversation transcript
- `POST /api/data/transcript` - Add entry to transcript (deprecated)
- `POST /api/data/transcript/add` - Add structured transcript entry
- `DELETE /api/data/transcript` - Clear all transcript entries

#### Analytics
**Dashboard:**
- `GET /api/analytics/dashboard` - Get comprehensive dashboard metrics
- `POST /api/analytics/query` - Execute custom analytics queries

**Reports:**
- `GET /api/analytics/reports/incidents-by-location` - Incident distribution by location
- `GET /api/analytics/reports/compliance-by-category` - Compliance status by category
- `GET /api/analytics/trends/monthly-incidents` - Monthly incident trends

#### LangSmith Integration
**Projects:**
- `GET /api/langsmith/projects` - List available LangSmith projects
- `GET /api/langsmith/health` - LangSmith service health check

**Traces:**
- `GET /api/langsmith/traces/{project_name}` - Get traces for specific project
- `GET /api/langsmith/trace/{run_id}` - Get detailed trace information

### API Response Models

#### Data Models
```python
# Incident Report
{
  "id": "INC-001",
  "title": "Chemical Spill in Laboratory",
  "description": "Minor chemical spill during routine testing",
  "severity": "medium|low|high|critical",
  "location": "Lab Building A, Room 205",
  "reporter_name": "John Smith",
  "reporter_email": "john.smith@company.com",
  "incident_date": "2024-01-15T14:30:00",
  "created_at": "2024-01-15T15:00:00",
  "status": "open|closed|under_investigation|resolved"
}

# Compliance Record
{
  "id": "COMP-001",
  "regulation_name": "OSHA 29 CFR 1910.1200",
  "compliance_status": "compliant|non_compliant|pending|under_review",
  "last_audit_date": "2023-12-01T00:00:00",
  "next_audit_date": "2024-06-01T00:00:00",
  "responsible_person": "Mike Wilson",
  "notes": "Hazard communication standard - all requirements met"
}

# Dashboard Metrics
{
  "total_incidents": 24,
  "open_incidents": 3,
  "compliance_rate": 92.5,
  "overdue_audits": 2,
  "recent_incidents": [...],
  "compliance_overview": [...]
}
```

### Error Handling

All endpoints follow consistent error response format:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "additional_info": "value"
    }
  }
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (validation errors)
- `404`: Resource Not Found
- `500`: Internal Server Error
- `502`: External Service Error (Neo4j, LangSmith)
- `503`: Service Unavailable

## Features

### Data Management
**Incident Management:**
- Create, view, and update incident reports with severity classification
- Track incident status (Open/Closed/Under Investigation/Resolved)
- Location-based incident tracking and reporting
- Reporter information and timestamp management
- Severity levels: Low, Medium, High, Critical

**Compliance Tracking:**
- Monitor regulatory compliance across multiple standards (OSHA, EPA, DOT)
- Track audit schedules and compliance status
- Responsible person assignment and notes management
- Compliance status: Compliant, Non-Compliant, Pending, Under Review

**Document Processing:**
- Neo4j graph database integration for document management
- Support for multiple document types: Electric Bills, Water Bills, Waste Manifests
- Real-time document processing status and metadata extraction
- File download capabilities with security validation
- Document type classification using AI labeling

**File Operations:**
- Multi-format file upload support (CSV, Excel)
- File validation and processing feedback
- Bulk data import capabilities
- Upload history and status tracking

### Analytics & Reporting
**Executive Dashboard:**
- Real-time KPI monitoring and executive-level insights
- Compliance rate calculations and trending
- Incident statistics and severity breakdowns
- Overdue audit tracking and notifications

**Hierarchical Analytics:**
- Multi-level data drill-down capabilities
- Department and location-based analysis
- Trend analysis with historical comparisons
- Custom query execution and filtering

**Visualization & Charts:**
- Chart.js integration for interactive data visualization
- Monthly trend analysis with severity breakdowns
- Location-based incident distribution
- Compliance status by regulation category
- Real-time dashboard updates

**Report Generation:**
- Incidents by location reporting
- Compliance status by category
- Monthly incident trends with severity analysis
- Safety performance metrics and benchmarking
- Custom analytics query execution

### AI & Integration Features
**LangSmith Integration:**
- AI conversation trace monitoring
- Project-based trace organization
- Detailed trace analysis with token usage tracking
- Real-time AI interaction logging
- Performance metrics for AI operations

**Transcript Management:**
- LLM conversation transcript storage and retrieval
- Structured conversation logging with context
- Thread-safe transcript operations
- Conversation history with timestamps and metadata
- Transcript search and analysis capabilities

### User Interface
**Modern Web Design:**
- Responsive Material-UI component library
- Professional corporate theme with consistent branding
- Intuitive navigation with icon-based sidebar
- Multi-view support (Standard, Executive, Hierarchical analytics)
- Real-time data updates and notifications

**Interactive Components:**
- Dynamic data tables with sorting and filtering
- Interactive charts and visualizations
- Progress tracking for document ingestion
- Modal dialogs for detailed information
- Responsive design for mobile and desktop

**Navigation & UX:**
- Single-page application with client-side routing
- Breadcrumb navigation for complex workflows
- Context-aware sidebar navigation
- Quick access to frequently used features
- Keyboard shortcuts and accessibility support

## UI Design System

### Visual Identity
The application features a professional corporate design system optimized for EHS compliance workflows:

**Color Palette:**
- **Primary Header**: Dark blue (#2c4a5e) with high contrast white text
- **Navigation**: Light gray (#f5f5f5) with subtle shadows and hover effects
- **Content Background**: Clean white (#ffffff) for optimal readability
- **Accent Colors**: Professional blue (#3498db) for interactive elements
- **Brand Consistency**: Unified color scheme across all components

**Status Indicator System:**
- **Success/Compliant**: Green (#27ae60) - compliance achieved, tasks completed
- **Warning/Pending**: Orange (#f39c12) - attention required, pending review
- **Error/Non-Compliant**: Red (#e74c3c) - issues requiring immediate action
- **Info/Neutral**: Blue (#3498db) - informational content and navigation
- **Critical**: Dark red (#c0392b) - high-priority incidents and alerts

### Component Design
**Navigation System:**
- Icon-based sidebar with tooltips for quick identification
- Hierarchical menu structure for complex workflows
- Active state indicators with smooth transitions
- Collapsible sidebar for space optimization

**Data Visualization:**
- Chart.js integration with corporate color scheme
- Interactive elements with hover states and animations
- Responsive chart sizing for different screen sizes
- Consistent styling across all visualization types

**Form & Input Design:**
- Material-UI components with custom theming
- Consistent validation styling and error messaging
- Accessibility-compliant form controls
- Clear visual hierarchy and labeling

**Typography & Layout:**
- Clean, readable font stack with proper line spacing
- Consistent heading hierarchy and text sizing
- Adequate whitespace for visual breathing room
- Grid-based layout system for alignment consistency

### Responsive Design
- Mobile-first responsive breakpoints
- Adaptive layout for tablets and mobile devices
- Touch-friendly interactive elements
- Consistent experience across device types

## Development Workflow

### Testing Strategy

**Backend Testing:**
```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
source venv/bin/activate

# Run all tests with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_data_management.py -v
pytest tests/test_analytics.py -v
pytest tests/test_langsmith_integration.py -v

# Test Neo4j integration
python test_neo4j_documents.py

# API integration tests
pytest tests/test_api_endpoints.py --integration
```

**Frontend Testing:**
```bash
# Navigate to frontend directory
cd frontend

# Run test suite
npm test

# Run tests with coverage
npm test -- --coverage

# Run specific component tests
npm test -- --testNamePattern="DataManagement"
npm test -- --testNamePattern="Analytics"

# End-to-end tests
npm run test:e2e

# Visual regression tests
npm run test:visual
```

**Database Testing:**
```bash
# Test PostgreSQL connection
psql -d ehs_compliance -c "SELECT 1;"

# Test Neo4j connection
echo "RETURN 'Connection successful' as result" | cypher-shell

# Run database migration tests
python manage.py test_migrations
```

### Code Quality & Standards

**Backend Standards:**
- **Style Guide**: Strict PEP 8 compliance with 88-character line limit
- **Type Hints**: Full type annotation coverage using Python 3.11+ features
- **Documentation**: Comprehensive docstrings following Google style
- **Linting**: Flake8, Black formatter, isort for import organization
- **Security**: Bandit security linter for vulnerability detection

```bash
# Code formatting and linting
black backend/ --line-length=88
flake8 backend/ --max-line-length=88
isort backend/ --profile=black
bandit -r backend/ -f json
```

**Frontend Standards:**
- **Style Guide**: Airbnb JavaScript style guide with custom modifications
- **Linting**: ESLint with React hooks and accessibility rules
- **Formatting**: Prettier for consistent code formatting
- **Type Safety**: PropTypes validation for component interfaces
- **Performance**: React DevTools Profiler for optimization

```bash
# Code quality checks
npm run lint
npm run lint:fix
npm run format
npm run type-check
```

**Database Standards:**
- **Migrations**: Version-controlled schema changes
- **Indexing**: Proper database indexing for performance
- **Security**: Parameterized queries to prevent SQL injection
- **Backup**: Automated backup procedures for both PostgreSQL and Neo4j

### Development Tools

**IDE Configuration:**
- VS Code workspace settings included
- Python and JavaScript debugging configurations
- Integrated terminal configurations
- Extension recommendations for optimal development

**Git Workflow:**
- Feature branch development model
- Conventional commit message format
- Pre-commit hooks for code quality
- Automated testing on pull requests

**Environment Management:**
- Separate environment configurations for development, staging, production
- Docker containerization support for consistent environments
- Environment variable validation and documentation
- Hot reloading for both frontend and backend development

### Performance Optimization

**Backend Optimization:**
- Database query optimization with SQLAlchemy
- Async/await patterns for concurrent operations
- Caching strategies for frequently accessed data
- Connection pooling for database operations

**Frontend Optimization:**
- Code splitting and lazy loading for reduced bundle size
- React.memo and useMemo for component optimization
- Optimized re-rendering patterns
- Image optimization and lazy loading

**Monitoring & Profiling:**
- Application performance monitoring integration
- Database query performance tracking
- Frontend bundle analysis tools
- Real-time error tracking and alerting

## Production Deployment

### Deployment Architecture

**Infrastructure Requirements:**
- **Application Server**: Linux-based server (Ubuntu 20.04+ recommended)
- **Database Servers**: PostgreSQL 13+ and Neo4j 5.0+ instances
- **Reverse Proxy**: Nginx for load balancing and SSL termination
- **Process Management**: Systemd or Docker for service management
- **Monitoring**: Application and infrastructure monitoring tools

### Production Build Process

#### Backend Deployment

1. **Environment Preparation:**
   ```bash
   # Create production environment
   python3 -m venv /opt/ehs-platform/venv
   source /opt/ehs-platform/venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install gunicorn  # Production WSGI server
   ```

2. **Production Configuration:**
   ```bash
   # Production environment variables
   export DATABASE_URL=postgresql://user:pass@prod-db:5432/ehs_compliance
   export NEO4J_URI=bolt://prod-neo4j:7687
   export NEO4J_USERNAME=neo4j
   export NEO4J_PASSWORD=production-password
   export DEBUG=False
   export LOG_LEVEL=WARNING
   export WORKERS=4
   ```

3. **Service Deployment:**
   ```bash
   # Using Gunicorn with multiple workers
   gunicorn main:app \
     --host 0.0.0.0 \
     --port 8001 \
     --workers 4 \
     --worker-class uvicorn.workers.UvicornWorker \
     --access-logfile /var/log/ehs-platform/access.log \
     --error-logfile /var/log/ehs-platform/error.log
   
   # Or using Uvicorn directly
   uvicorn main:app \
     --host 0.0.0.0 \
     --port 8001 \
     --workers 4 \
     --access-log \
     --log-config logging.conf
   ```

4. **Systemd Service Configuration:**
   ```ini
   # /etc/systemd/system/ehs-platform.service
   [Unit]
   Description=EHS Platform API
   After=network.target
   
   [Service]
   Type=exec
   User=ehs-platform
   Group=ehs-platform
   WorkingDirectory=/opt/ehs-platform
   Environment=PATH=/opt/ehs-platform/venv/bin
   ExecStart=/opt/ehs-platform/venv/bin/gunicorn main:app --config gunicorn.conf.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

#### Frontend Deployment

1. **Production Build:**
   ```bash
   cd frontend
   
   # Set production environment variables
   export REACT_APP_API_URL=https://api.ehs-platform.com
   export GENERATE_SOURCEMAP=false
   
   # Create optimized production build
   npm run build
   
   # Analyze bundle size
   npm run analyze
   ```

2. **Static File Serving:**
   ```bash
   # Copy build files to web server directory
   cp -r build/* /var/www/ehs-platform/
   
   # Set proper ownership and permissions
   chown -R www-data:www-data /var/www/ehs-platform/
   chmod -R 755 /var/www/ehs-platform/
   ```

3. **Nginx Configuration:**
   ```nginx
   # /etc/nginx/sites-available/ehs-platform
   server {
       listen 80;
       server_name ehs-platform.com www.ehs-platform.com;
       return 301 https://$server_name$request_uri;
   }
   
   server {
       listen 443 ssl http2;
       server_name ehs-platform.com www.ehs-platform.com;
       
       ssl_certificate /path/to/ssl/certificate.pem;
       ssl_certificate_key /path/to/ssl/private.key;
       
       # Frontend static files
       location / {
           root /var/www/ehs-platform;
           index index.html index.htm;
           try_files $uri $uri/ /index.html;
           
           # Cache static assets
           location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
               expires 1y;
               add_header Cache-Control "public, immutable";
           }
       }
       
       # API proxy
       location /api/ {
           proxy_pass http://localhost:8001;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

### Container Deployment (Docker)

#### Backend Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser
USER appuser

EXPOSE 8001

CMD ["gunicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
```

#### Frontend Dockerfile
```dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/ehs_compliance
      - NEO4J_URI=bolt://neo4j:7687
    depends_on:
      - db
      - neo4j
  
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=ehs_compliance
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  neo4j:
    image: neo4j:5.0
    environment:
      - NEO4J_AUTH=neo4j/EhsAI2024!
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

volumes:
  postgres_data:
  neo4j_data:
```

### Monitoring & Maintenance

**Application Monitoring:**
- Health check endpoints for service monitoring
- Application performance monitoring (APM)
- Error tracking and alerting systems
- Database performance monitoring

**Logging:**
- Centralized logging with structured log formats
- Log rotation and retention policies
- Real-time log monitoring and alerting
- Security audit logging

**Backup & Recovery:**
- Automated database backups with encryption
- Application configuration backups
- Disaster recovery procedures
- Backup verification and restoration testing

**Security:**
- SSL/TLS certificate management
- Security headers configuration
- Regular security vulnerability scanning
- Access control and authentication monitoring

## Integration with EHS Data Foundation

This web application serves as the primary user interface for the comprehensive EHS Data Foundation ecosystem, providing seamless integration with multiple backend services and data sources.

### Neo4j Knowledge Graph Integration

**Document Management:**
- **Real-time Document Retrieval**: Direct connection to Neo4j graph database for processed document access
- **Graph Traversal**: Utilizes Neo4j's graph traversal capabilities for complex document relationships
- **Multi-document Types**: Supports Electric Bills, Water Bills, Waste Manifests with type-specific processing
- **Metadata Extraction**: Automatically extracts and displays document metadata and extracted data
- **Relationship Mapping**: Visualizes connections between documents, facilities, and compliance records

**Advanced Query Capabilities:**
```cypher
# Example Neo4j queries used by the application:
MATCH (d:Document)-[:EXTRACTED_TO]->(ub:UtilityBill)
RETURN d, ub

MATCH (d:Document)-[:HAS_MONTHLY_ALLOCATION]->(ma:MonthlyUsageAllocation)
WHERE ma.usage_year = 2024 AND ma.usage_month = 8
RETURN d, ma.allocated_usage
```

**Connection Configuration:**
- **Secure Authentication**: Neo4j authentication with encrypted credentials
- **Connection Pooling**: Optimized connection management for high-throughput operations
- **Fallback Mechanisms**: Mock data fallback when Neo4j is unavailable
- **Error Handling**: Comprehensive error handling with detailed logging

### Backend Service Integration

**Data Processing Pipeline:**
- **Document Ingestion**: Real-time monitoring of document processing status
- **Progress Tracking**: Visual progress indicators for document ingestion workflows
- **Status Updates**: Live updates on processing completion and errors
- **Batch Processing**: Support for bulk document processing operations

**External API Integration:**
- **Multi-service Architecture**: Integration with multiple backend services on different ports
- **Service Discovery**: Automatic detection and routing to available services
- **Load Balancing**: Request distribution across multiple backend instances
- **Circuit Breaker**: Fault tolerance patterns for service failures

### LangSmith AI Integration

**AI Conversation Tracking:**
- **Trace Management**: Complete trace lifecycle management for AI interactions
- **Project Organization**: Project-based organization of AI traces and conversations
- **Performance Monitoring**: Token usage, latency, and success rate tracking
- **Conversation Context**: Full conversation history with context preservation

**Integration Features:**
- **Real-time Trace Fetching**: Live trace data from LangSmith platform
- **Structured Logging**: Standardized trace data format for analysis
- **Error Tracking**: Comprehensive error tracking for AI operations
- **Performance Analytics**: Detailed performance metrics and optimization insights

### Data Foundation Architecture

**Service Mesh:**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │───▶│   API Gateway    │───▶│  Document API   │
│  (Port 3000)    │    │   (Port 8001)    │    │  (Port 8000)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │◀───│  Analytics API   │───▶│    Neo4j DB     │
│   Database      │    │                  │    │  Knowledge      │
│                 │    │                  │    │     Graph       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   LangSmith      │
                       │   Integration    │
                       │                  │
                       └──────────────────┘
```

**Data Flow:**
1. **Document Upload**: Files uploaded through web interface
2. **Processing Pipeline**: Documents processed through AI extraction workflows
3. **Graph Storage**: Extracted data stored in Neo4j with relationships
4. **Analytics Processing**: Real-time analytics computed from graph data
5. **Frontend Display**: Processed data displayed through responsive interface

### Integration APIs

**Document Processing API:**
- `POST /api/v1/documents/upload` - Upload documents for processing
- `GET /api/v1/documents/status/{id}` - Check processing status
- `GET /api/v1/documents/extracted/{id}` - Get extracted data

**Analytics Integration:**
- `GET /api/analytics/compliance-metrics` - Real-time compliance calculations
- `POST /api/analytics/custom-queries` - Execute custom analytics queries
- `GET /api/analytics/trend-analysis` - Historical trend analysis

**AI Integration:**
- `POST /api/ai/analyze-document` - AI-powered document analysis
- `GET /api/ai/conversation-history` - Retrieve AI conversation history
- `POST /api/ai/query` - Natural language querying of data

### Data Synchronization

**Real-time Updates:**
- **WebSocket Connections**: Live updates for document processing status
- **Event-driven Architecture**: Reactive updates based on data changes
- **Caching Strategies**: Intelligent caching for frequently accessed data
- **Conflict Resolution**: Handling concurrent data updates and conflicts

**Batch Synchronization:**
- **Scheduled Updates**: Regular synchronization of large datasets
- **Incremental Sync**: Efficient updates for changed data only
- **Data Validation**: Comprehensive validation of synchronized data
- **Rollback Mechanisms**: Safe rollback procedures for failed updates

### Security & Compliance

**Data Protection:**
- **End-to-end Encryption**: Encrypted data transmission and storage
- **Access Control**: Role-based access to sensitive EHS data
- **Audit Logging**: Comprehensive audit trails for compliance
- **Data Privacy**: GDPR and industry compliance measures

**Integration Security:**
- **API Authentication**: Secure authentication across all services
- **Service-to-service Security**: Encrypted inter-service communication
- **Input Validation**: Comprehensive validation of all data inputs
- **Rate Limiting**: Protection against abuse and overload

## Security Best Practices

### Application Security

**Authentication & Authorization:**
- JWT token-based authentication with configurable expiration
- Role-based access control (RBAC) for different user types
- Secure password hashing using bcrypt with salt rounds
- Session management with secure cookie configurations
- Multi-factor authentication support for admin users

**Data Protection:**
- Input validation using Pydantic models and custom validators
- SQL injection prevention through parameterized queries
- XSS protection with Content Security Policy headers
- CORS configuration with explicit origin whitelisting
- File upload restrictions and virus scanning integration

**Infrastructure Security:**
- HTTPS enforcement with proper SSL/TLS configuration
- Security headers (HSTS, X-Frame-Options, X-Content-Type-Options)
- Database connection encryption and credential management
- Environment variable protection and secret management
- Regular security dependency updates and vulnerability scanning

### Data Privacy & Compliance

**EHS Data Handling:**
- Sensitive incident data encryption at rest and in transit
- Personal information anonymization for analytics
- Compliance with industry regulations (OSHA, EPA, DOT)
- Data retention policies with automatic cleanup
- Audit logging for all data access and modifications

**Neo4j Security:**
- Encrypted database connections with certificate validation
- User authentication and authorization within Neo4j
- Query injection prevention through parameterized queries
- Database backup encryption and secure storage
- Network isolation and firewall configuration

## Troubleshooting

### Common Issues & Solutions

**Backend Issues:**

1. **Neo4j Connection Failed:**
   ```bash
   # Check Neo4j service status
   neo4j status
   
   # Verify connection parameters
   python test_neo4j_documents.py
   
   # Check network connectivity
   telnet localhost 7687
   ```

2. **Database Migration Errors:**
   ```bash
   # Reset database schema
   dropdb ehs_compliance
   createdb ehs_compliance
   
   # Run migrations
   python manage.py migrate
   ```

3. **Import Errors:**
   ```bash
   # Verify virtual environment
   which python
   pip list | grep fastapi
   
   # Reinstall dependencies
   pip install -r requirements.txt --upgrade
   ```

4. **Performance Issues:**
   ```bash
   # Monitor resource usage
   htop
   
   # Check database connections
   netstat -an | grep 5432
   
   # Analyze slow queries
   tail -f backend.log | grep "slow"
   ```

**Frontend Issues:**

1. **Build Failures:**
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules package-lock.json
   npm install
   
   # Check for dependency conflicts
   npm ls
   
   # Update dependencies
   npm audit fix
   ```

2. **API Connection Issues:**
   ```bash
   # Verify API endpoint configuration
   echo $REACT_APP_API_URL
   
   # Test API connectivity
   curl http://localhost:8001/health
   
   # Check CORS configuration
   curl -H "Origin: http://localhost:3000" http://localhost:8001/health
   ```

3. **Performance Issues:**
   ```bash
   # Analyze bundle size
   npm run analyze
   
   # Enable source maps for debugging
   export GENERATE_SOURCEMAP=true
   npm start
   ```

**Database Issues:**

1. **Neo4j Performance:**
   ```cypher
   # Check database statistics
   CALL db.stats.retrieve('GRAPH COUNTS')
   
   # Analyze query performance
   PROFILE MATCH (d:Document) RETURN count(d)
   
   # Check indexes
   SHOW INDEXES
   ```

2. **PostgreSQL Issues:**
   ```sql
   -- Check connection limits
   SELECT * FROM pg_stat_activity;
   
   -- Analyze query performance
   SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC;
   
   -- Check table sizes
   SELECT schemaname,tablename,attname,n_distinct,correlation FROM pg_stats;
   ```

### Logging & Debugging

**Enable Debug Logging:**
```bash
# Backend debug mode
export DEBUG=True
export LOG_LEVEL=DEBUG

# Frontend debug mode
export REACT_APP_DEBUG=true
```

**Log Analysis:**
```bash
# Monitor backend logs
tail -f backend.log | jq .

# Monitor access logs
tail -f /var/log/nginx/access.log

# Monitor system resources
watch -n 1 'free -h && df -h'
```

### Performance Optimization

**Backend Optimization:**
- Enable database query caching
- Implement connection pooling
- Use async/await for I/O operations
- Optimize Neo4j queries with proper indexing
- Enable response compression

**Frontend Optimization:**
- Implement code splitting and lazy loading
- Optimize image sizes and formats
- Enable service worker for caching
- Minimize bundle size with tree shaking
- Use React.memo for expensive components

## Contributing Guidelines

### Development Process

1. **Fork and Clone:**
   ```bash
   git clone https://github.com/your-username/ehs-ai-demo.git
   cd ehs-ai-demo/data-foundation/web-app
   ```

2. **Create Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development Setup:**
   ```bash
   # Backend setup
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Frontend setup
   cd ../frontend
   npm install
   ```

4. **Code Quality Checks:**
   ```bash
   # Backend checks
   black backend/ --check
   flake8 backend/
   pytest backend/tests/ --cov
   
   # Frontend checks
   npm run lint
   npm run test -- --coverage
   ```

5. **Commit and Push:**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request:**
   - Provide clear description of changes
   - Include test coverage information
   - Add screenshots for UI changes
   - Reference related issues

### Code Standards

**Commit Message Format:**
```
type(scope): short description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Pull Request Requirements:**
- All tests must pass
- Code coverage must not decrease
- Documentation must be updated
- Security review for sensitive changes
- Performance impact assessment

## License & Support

### License Information

This project is part of the EHS AI Demo system developed for environmental, health, and safety compliance management. All rights reserved.

**Usage Rights:**
- Internal use within organizations for EHS compliance
- Educational and research purposes
- Development and testing environments

**Restrictions:**
- Commercial redistribution requires explicit permission
- Modification of core security features requires approval
- Integration with external systems requires security review

### Support Channels

**Technical Support:**
- **Email**: ehs-ai-support@company.com
- **Documentation**: Internal wiki and knowledge base
- **Issue Tracking**: GitHub Issues for bug reports and feature requests
- **Emergency Support**: 24/7 hotline for critical production issues

**Community Resources:**
- **Developer Forum**: Internal developer community
- **Training Materials**: Comprehensive onboarding documentation
- **Best Practices**: Industry-specific implementation guides
- **Regular Updates**: Monthly releases with new features and improvements

**Professional Services:**
- **Custom Implementation**: Tailored deployment for specific requirements
- **Integration Services**: Professional integration with existing EHS systems
- **Training Programs**: Comprehensive user and administrator training
- **Consulting Services**: EHS compliance strategy and optimization

For immediate assistance with critical issues, please contact the EHS AI team through the appropriate support channel based on the severity and nature of your request.