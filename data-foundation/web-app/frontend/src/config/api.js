// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

export const API_ENDPOINTS = {
  // Data Management endpoints
  processedDocuments: `${API_BASE_URL}/api/data/processed-documents`,
  processedDocumentById: (id) => `${API_BASE_URL}/api/data/processed-documents/${id}`,
  rejectedDocuments: 'http://localhost:8000/api/v1/simple-rejected-documents',
  
  // Analytics endpoints
  analyticsDashboard: `${API_BASE_URL}/api/analytics/dashboard`,
  analyticsQuery: `${API_BASE_URL}/api/analytics/query`,
};

export default API_BASE_URL;