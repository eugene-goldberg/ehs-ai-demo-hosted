// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://10.136.0.4:8000';
const DATA_API_BASE_URL = process.env.REACT_APP_DATA_API_URL || 'http://10.136.0.4:8001';

export const API_ENDPOINTS = {
  // Data Management endpoints (port 8001)
  processedDocuments: `${DATA_API_BASE_URL}/api/data/processed-documents`,
  processedDocumentById: (id) => `${DATA_API_BASE_URL}/api/data/processed-documents/${id}`,
  rejectedDocuments: `${API_BASE_URL}/api/v1/simple-rejected-documents`,
  analyticsDashboard: `${DATA_API_BASE_URL}/api/analytics/dashboard`,
  analyticsQuery: `${DATA_API_BASE_URL}/api/analytics/query`,
  
  // Main backend endpoints (port 8000)
  executiveDashboard: `${API_BASE_URL}/api/v2/executive-dashboard`,
  ingestion: `${API_BASE_URL}/api/batch-ingest`,
  
  // LangSmith/Conversations endpoints (port 8001)
  conversations: `${DATA_API_BASE_URL}/api/langsmith/`,
  
  // Environmental Dashboard endpoints (port 8000)
  dashboardElectricity: `${API_BASE_URL}/api/dashboard/electricity`,
  dashboardWater: `${API_BASE_URL}/api/dashboard/water`,
  dashboardWaste: `${API_BASE_URL}/api/dashboard/waste`,
  
  // Risk Assessment endpoints (port 8001)
  riskAssessmentTranscript: `${DATA_API_BASE_URL}/api/risk-assessment-transcript/`,
  
  // Add other endpoints as needed
};

export default {
  data: {
    processedDocuments: API_ENDPOINTS.processedDocuments,
    rejectedDocuments: API_ENDPOINTS.rejectedDocuments,
  },
  analytics: {
    dashboard: API_ENDPOINTS.analyticsDashboard,
    query: API_ENDPOINTS.analyticsQuery,
  },
  executive: {
    dashboard: API_ENDPOINTS.executiveDashboard,
  },
  environmental: {
    electricity: API_ENDPOINTS.dashboardElectricity,
    water: API_ENDPOINTS.dashboardWater,
    waste: API_ENDPOINTS.dashboardWaste,
  },
  langsmith: {
    conversations: API_ENDPOINTS.conversations,
  },
  riskAssessment: {
    transcript: API_ENDPOINTS.riskAssessmentTranscript,
  },
  ingestion: API_ENDPOINTS.ingestion,
};
