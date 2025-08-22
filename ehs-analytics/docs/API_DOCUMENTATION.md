# EHS Analytics API Documentation

## Overview

The EHS Analytics API provides a comprehensive set of endpoints for processing natural language queries about environmental, health, and safety (EHS) data. The API uses advanced AI techniques including GraphRAG, multi-strategy retrieval, and intelligent agent orchestration to deliver accurate and actionable insights.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.ehs-analytics.company.com`

## API Version

- **Current Version**: `v1`
- **Base Path**: `/api/v1`

## Authentication

### Development Mode
Currently in development mode with simplified authentication. The API requires a `user_id` for tracking and access control.

### Production Mode
Production deployments require JWT token authentication:

```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

## Request/Response Format

- **Request Format**: JSON
- **Response Format**: JSON
- **Character Encoding**: UTF-8
- **Date Format**: ISO 8601 (`YYYY-MM-DDTHH:MM:SS.sssZ`)

## Common Headers

### Request Headers
```http
Content-Type: application/json
Authorization: Bearer <token>  # Production only
X-User-ID: <user_id>          # Development mode
```

### Response Headers
```http
Content-Type: application/json
X-Request-ID: <uuid>
X-Response-Time: <time_in_ms>ms
```

## Rate Limiting

- **Default Limit**: 100 requests per minute per user
- **Burst Limit**: 20 requests per 10 seconds
- **Headers**: Rate limit information in response headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Core Endpoints

### 1. Submit Query

Submit a natural language EHS query for processing.

```http
POST /api/v1/analytics/query
```

#### Request Body

```json
{
  "query": "Show electricity usage for Apex Manufacturing in Q1 2024",
  "user_id": "user123",
  "session_id": "session456",
  "context": {
    "facility_focus": "energy_efficiency",
    "department": "environmental"
  },
  "preferences": {
    "detail_level": "high",
    "include_charts": true
  }
}
```

#### Query Processing Options

```json
{
  "include_recommendations": true,
  "max_results": 10,
  "timeout_seconds": 300,
  "retrieval_strategy": "hybrid",
  "explain_reasoning": false
}
```

#### Response (202 Accepted)

```json
{
  "success": true,
  "message": "Query processing initiated successfully",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "query_id": "query-123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "estimated_completion_time": 300
}
```

#### Supported Query Types

The API automatically classifies queries into seven intent types:

1. **consumption_analysis** - Energy, water, resource usage analysis
2. **compliance_check** - Regulatory compliance monitoring
3. **risk_assessment** - Environmental and safety risk evaluation
4. **emission_tracking** - Carbon footprint and emission monitoring
5. **equipment_efficiency** - Asset performance optimization
6. **permit_status** - Environmental permit management
7. **general_inquiry** - General EHS information requests

#### Error Responses

```json
{
  "success": false,
  "message": "Validation error",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "error": {
    "error_type": "validation_error",
    "error_code": "INVALID_QUERY",
    "message": "Query cannot be empty or whitespace only",
    "details": {
      "field": "query",
      "value": ""
    },
    "trace_id": "trace-789"
  }
}
```

### 2. Get Query Result

Retrieve the complete results of a processed query.

```http
GET /api/v1/analytics/query/{query_id}?include_trace=false
```

#### Path Parameters

- `query_id` (required): Unique query identifier

#### Query Parameters

- `include_trace` (optional): Include workflow execution trace (default: false)

#### Response (200 OK)

```json
{
  "success": true,
  "message": "Query results retrieved successfully",
  "timestamp": "2024-01-01T12:05:00.000Z",
  "query_id": "query-123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "original_query": "Show electricity usage for Apex Manufacturing in Q1 2024",
  "processing_time_ms": 2500,
  "classification": {
    "intent_type": "consumption_analysis",
    "confidence_score": 0.95,
    "entities_identified": {
      "facilities": ["Apex Manufacturing"],
      "date_ranges": ["Q1 2024"],
      "equipment": [],
      "pollutants": [],
      "regulations": [],
      "departments": [],
      "metrics": ["electricity usage"]
    },
    "suggested_retriever": "consumption_retriever",
    "reasoning": "Query requests specific consumption data for a facility and time period",
    "query_rewrite": null
  },
  "retrieval_results": {
    "documents": [
      {
        "content": "Electricity consumption data for Apex Manufacturing Q1 2024: 125,000 kWh",
        "metadata": {
          "source": "utility_bills",
          "confidence": 0.98,
          "timestamp": "2024-03-31T23:59:59.000Z",
          "query_used": "MATCH (f:Facility {name: 'Apex Manufacturing'})-[:CONSUMES]->(u:UtilityUsage) WHERE u.period = 'Q1 2024' AND u.type = 'electricity' RETURN u.amount"
        },
        "relevance_score": 0.98,
        "document_id": "util-001"
      }
    ],
    "total_count": 5,
    "retrieval_strategy": "consumption_retriever",
    "execution_time_ms": 150
  },
  "analysis_results": [
    {
      "consumption_type": "electricity",
      "current_value": 125000.0,
      "unit": "kWh",
      "trend": "decreasing",
      "trend_percentage": -8.5,
      "comparison_period": "Q4 2023",
      "efficiency_rating": "good",
      "recommendations": [
        "Continue current energy efficiency initiatives",
        "Consider LED lighting upgrades in remaining areas"
      ]
    }
  ],
  "recommendations": {
    "recommendations": [
      {
        "title": "Implement Smart Lighting Controls",
        "description": "Install occupancy sensors and daylight harvesting controls",
        "priority": "medium",
        "category": "efficiency",
        "estimated_cost": 15000.0,
        "estimated_savings": 8500.0,
        "payback_period_months": 21,
        "implementation_effort": "medium",
        "confidence": 0.85
      }
    ],
    "total_estimated_cost": 15000.0,
    "total_estimated_savings": 8500.0,
    "recommendations_count": 1,
    "generated_at": "2024-01-01T12:04:50.000Z"
  },
  "confidence_score": 0.95,
  "workflow_trace": [
    "Starting query classification",
    "Query classified as: consumption_analysis",
    "Starting data retrieval",
    "Data retrieval completed",
    "Starting analysis",
    "Analysis completed",
    "Generating recommendations",
    "Recommendations generated",
    "Query processing completed successfully"
  ]
}
```

### 3. Get Query Status

Check the processing status of a submitted query.

```http
GET /api/v1/analytics/query/{query_id}/status
```

#### Response (200 OK)

```json
{
  "success": true,
  "message": "Query status retrieved successfully",
  "timestamp": "2024-01-01T12:02:30.000Z",
  "query_id": "query-123e4567-e89b-12d3-a456-426614174000",
  "status": "in_progress",
  "progress_percentage": 70,
  "current_step": "analysis",
  "estimated_remaining_time": 45
}
```

#### Status Values

- `pending` - Query queued for processing
- `in_progress` - Currently being processed
- `completed` - Processing completed successfully
- `failed` - Processing failed with error
- `cancelled` - Processing was cancelled

### 4. Cancel Query

Cancel a pending or in-progress query.

```http
DELETE /api/v1/analytics/query/{query_id}
```

#### Response (200 OK)

```json
{
  "success": true,
  "message": "Query query-123e4567-e89b-12d3-a456-426614174000 has been cancelled successfully",
  "timestamp": "2024-01-01T12:03:00.000Z"
}
```

### 5. List User Queries

Retrieve a list of queries for the authenticated user.

```http
GET /api/v1/analytics/queries?limit=10&offset=0&status_filter=completed
```

#### Query Parameters

- `limit` (optional): Maximum queries to return (1-100, default: 10)
- `offset` (optional): Number of queries to skip (default: 0)  
- `status_filter` (optional): Filter by status (`pending`, `in_progress`, `completed`, `failed`, `cancelled`)

#### Response (200 OK)

```json
[
  {
    "query_id": "query-123e4567-e89b-12d3-a456-426614174000",
    "original_query": "Show electricity usage for Apex Manufacturing in Q1 2024",
    "status": "completed",
    "created_at": "2024-01-01T12:00:00.000Z",
    "updated_at": "2024-01-01T12:05:00.000Z",
    "intent_type": "consumption_analysis"
  },
  {
    "query_id": "query-456e7890-e89b-12d3-a456-426614174001",
    "original_query": "What permits are expiring in the next 90 days for all facilities?",
    "status": "completed", 
    "created_at": "2024-01-01T11:30:00.000Z",
    "updated_at": "2024-01-01T11:33:00.000Z",
    "intent_type": "permit_status"
  }
]
```

### 6. Analytics Health Check

Check the health status of the analytics service components.

```http
GET /api/v1/analytics/health
```

#### Response (200 OK)

```json
{
  "success": true,
  "message": "Health check completed",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "overall_status": "healthy",
  "services": [
    {
      "service_name": "Neo4j Database",
      "status": "healthy",
      "response_time_ms": 25,
      "error_message": null,
      "last_check": "2024-01-01T12:00:00.000Z",
      "metadata": {
        "connected": true
      }
    },
    {
      "service_name": "LangGraph Workflow",
      "status": "healthy",
      "response_time_ms": 15,
      "error_message": null,
      "last_check": "2024-01-01T12:00:00.000Z",
      "metadata": {
        "initialized": true
      }
    },
    {
      "service_name": "Query Processing",
      "status": "healthy",
      "response_time_ms": 5,
      "error_message": null,
      "last_check": "2024-01-01T12:00:00.000Z",
      "metadata": {
        "active_queries": 3,
        "total_queries": 45
      }
    },
    {
      "service_name": "OpenAI API",
      "status": "healthy",
      "response_time_ms": null,
      "error_message": null,
      "last_check": "2024-01-01T12:00:00.000Z",
      "metadata": {
        "note": "Not actively monitored"
      }
    }
  ],
  "uptime_seconds": 86400,
  "version": "0.1.0",
  "environment": "development"
}
```

## System Endpoints

### 1. Global Health Check

Check overall system health including all components.

```http
GET /health
```

#### Response (200 OK)

```json
{
  "success": true,
  "message": "Global health check completed",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "overall_status": "healthy",
  "services": [
    {
      "service_name": "FastAPI Application",
      "status": "healthy",
      "response_time_ms": 5,
      "metadata": {
        "version": "0.1.0",
        "environment": "development",
        "python_version": "3.11.5"
      }
    }
  ],
  "uptime_seconds": 86400,
  "version": "0.1.0",
  "environment": "development"
}
```

### 2. API Root Information

Get basic API information and available endpoints.

```http
GET /
```

#### Response (200 OK)

```json
{
  "name": "EHS Analytics API",
  "version": "0.1.0",
  "description": "AI-powered environmental, health, and safety analytics platform",
  "status": "operational",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "endpoints": {
    "docs": "/docs",
    "redoc": "/redoc",
    "openapi": "/openapi.json",
    "health": "/health",
    "analytics": "/api/v1/analytics"
  },
  "features": [
    "Natural language EHS query processing",
    "AI-powered risk assessment",
    "Regulatory compliance monitoring",
    "Equipment efficiency analysis",
    "Emission tracking and reporting"
  ]
}
```

### 3. OpenAPI Specification

Get the complete OpenAPI specification for the API.

```http
GET /openapi.json
```

### 4. Interactive Documentation

Access interactive API documentation:

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

## Data Models

### Query Request Model

```json
{
  "query": "string (1-2000 characters, required)",
  "user_id": "string (optional)",
  "session_id": "string (optional)",
  "context": {
    "facility_focus": "string",
    "department": "string",
    "urgency": "string",
    "time_period": "string",
    "benchmark": "boolean"
  },
  "preferences": {
    "detail_level": "low | medium | high",
    "include_charts": "boolean",
    "format": "summary | detailed | technical"
  }
}
```

### Query Processing Options

```json
{
  "include_recommendations": "boolean (default: true)",
  "max_results": "integer (1-100, default: 10)", 
  "timeout_seconds": "integer (30-600, default: 300)",
  "retrieval_strategy": "string (optional override)",
  "explain_reasoning": "boolean (default: false)"
}
```

### Analysis Result Types

#### Risk Assessment
```json
{
  "risk_level": "low | medium | high | critical",
  "risk_score": "number (0.0-1.0)",
  "risk_factors": ["string"],
  "mitigation_suggestions": ["string"],
  "confidence": "number (0.0-1.0)"
}
```

#### Compliance Status
```json
{
  "compliant": "boolean",
  "compliance_score": "number (0.0-1.0)",
  "violations": ["string"],
  "requirements_met": ["string"],
  "next_review_date": "datetime (optional)"
}
```

#### Consumption Analysis
```json
{
  "consumption_type": "energy | water | gas | waste",
  "current_value": "number",
  "unit": "string",
  "trend": "increasing | decreasing | stable",
  "trend_percentage": "number",
  "comparison_period": "string",
  "efficiency_rating": "poor | fair | good | excellent",
  "recommendations": ["string"]
}
```

#### Equipment Efficiency
```json
{
  "equipment_id": "string",
  "equipment_type": "string", 
  "efficiency_score": "number (0.0-1.0)",
  "uptime_percentage": "number (0.0-100.0)",
  "maintenance_status": "string",
  "performance_issues": ["string"],
  "optimization_suggestions": ["string"]
}
```

### Recommendation Model

```json
{
  "title": "string",
  "description": "string",
  "priority": "low | medium | high | critical",
  "category": "cost_reduction | compliance | efficiency | risk_mitigation",
  "estimated_cost": "number (optional)",
  "estimated_savings": "number (optional)",
  "payback_period_months": "integer (optional)",
  "implementation_effort": "low | medium | high",
  "confidence": "number (0.0-1.0)"
}
```

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "message": "string",
  "timestamp": "datetime",
  "error": {
    "error_type": "error_category",
    "error_code": "SPECIFIC_ERROR_CODE",
    "message": "string",
    "details": {
      "field": "string",
      "value": "any",
      "constraint": "string"
    },
    "trace_id": "string"
  }
}
```

### Error Types

| Error Type | Description | HTTP Status |
|------------|-------------|-------------|
| `validation_error` | Request validation failed | 400 |
| `processing_error` | Query processing failed | 500 |
| `database_error` | Database connectivity issues | 503 |
| `timeout_error` | Processing timeout exceeded | 504 |
| `authorization_error` | Access denied | 403 |
| `not_found_error` | Resource not found | 404 |
| `rate_limit_error` | Rate limit exceeded | 429 |

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `INVALID_QUERY` | Query validation failed |
| `QUERY_NOT_FOUND` | Query ID does not exist |
| `PROCESSING_TIMEOUT` | Query processing timeout |
| `DATABASE_UNAVAILABLE` | Database connection failed |
| `INSUFFICIENT_PERMISSIONS` | Access denied to resource |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `WORKFLOW_ERROR` | Internal processing error |

### Error Response Examples

#### 400 Bad Request - Validation Error
```json
{
  "success": false,
  "message": "Validation error",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "error": {
    "error_type": "validation_error",
    "error_code": "INVALID_QUERY",
    "message": "Query cannot be empty or whitespace only",
    "details": {
      "field": "query",
      "value": "",
      "constraint": "min_length=1"
    }
  }
}
```

#### 404 Not Found - Query Not Found
```json
{
  "success": false,
  "message": "Resource not found",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "error": {
    "error_type": "not_found_error",
    "error_code": "QUERY_NOT_FOUND",
    "message": "Query with ID query-invalid-id not found",
    "details": {
      "query_id": "query-invalid-id"
    }
  }
}
```

#### 429 Too Many Requests - Rate Limiting
```json
{
  "success": false,
  "message": "Rate limit exceeded",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "error": {
    "error_type": "rate_limit_error",
    "error_code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit of 100 requests per minute exceeded",
    "details": {
      "limit": 100,
      "period": "minute",
      "retry_after": 45
    }
  }
}
```

## Usage Examples

### Python Example using `requests`

```python
import requests
import time
import json

# Configuration
API_BASE_URL = "http://localhost:8000"
USER_ID = "user123"

def submit_query(query_text, context=None):
    """Submit a query and return the query ID."""
    url = f"{API_BASE_URL}/api/v1/analytics/query"
    payload = {
        "query": query_text,
        "user_id": USER_ID,
        "context": context or {}
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["query_id"]

def wait_for_completion(query_id, timeout=300):
    """Poll for query completion."""
    url = f"{API_BASE_URL}/api/v1/analytics/query/{query_id}/status"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        status = data["status"]
        if status == "completed":
            return True
        elif status == "failed":
            raise Exception("Query processing failed")
        
        time.sleep(2)
    
    raise TimeoutError("Query processing timeout")

def get_results(query_id):
    """Retrieve query results."""
    url = f"{API_BASE_URL}/api/v1/analytics/query/{query_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Example usage
if __name__ == "__main__":
    # Submit query
    query_id = submit_query(
        "Show water consumption trends for all facilities in 2024",
        context={"department": "environmental", "urgency": "medium"}
    )
    print(f"Query submitted: {query_id}")
    
    # Wait for completion
    wait_for_completion(query_id)
    print("Query completed")
    
    # Get results
    results = get_results(query_id)
    print(json.dumps(results, indent=2))
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

class EHSAnalyticsClient {
    constructor(baseUrl = 'http://localhost:8000', userId = 'user123') {
        this.baseUrl = baseUrl;
        this.userId = userId;
        this.client = axios.create({
            baseURL: baseUrl,
            headers: {
                'Content-Type': 'application/json',
                'X-User-ID': userId
            }
        });
    }
    
    async submitQuery(query, context = {}, options = {}) {
        try {
            const response = await this.client.post('/api/v1/analytics/query', {
                query,
                user_id: this.userId,
                context,
                ...options
            });
            return response.data.query_id;
        } catch (error) {
            throw new Error(`Failed to submit query: ${error.response?.data?.error?.message || error.message}`);
        }
    }
    
    async waitForCompletion(queryId, timeout = 300000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            try {
                const response = await this.client.get(`/api/v1/analytics/query/${queryId}/status`);
                const status = response.data.status;
                
                if (status === 'completed') {
                    return true;
                } else if (status === 'failed') {
                    throw new Error('Query processing failed');
                }
                
                await new Promise(resolve => setTimeout(resolve, 2000));
            } catch (error) {
                throw new Error(`Failed to check status: ${error.response?.data?.error?.message || error.message}`);
            }
        }
        
        throw new Error('Query processing timeout');
    }
    
    async getResults(queryId, includeTrace = false) {
        try {
            const response = await this.client.get(`/api/v1/analytics/query/${queryId}`, {
                params: { include_trace: includeTrace }
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get results: ${error.response?.data?.error?.message || error.message}`);
        }
    }
    
    async processQuery(query, context = {}, options = {}) {
        const queryId = await this.submitQuery(query, context, options);
        await this.waitForCompletion(queryId);
        return this.getResults(queryId);
    }
}

// Example usage
(async () => {
    const client = new EHSAnalyticsClient();
    
    try {
        const results = await client.processQuery(
            "Identify high-risk equipment at Apex Manufacturing",
            { department: "safety", risk_types: ["environmental", "operational"] }
        );
        
        console.log('Query Results:', JSON.stringify(results, null, 2));
    } catch (error) {
        console.error('Error:', error.message);
    }
})();
```

### cURL Examples

#### Submit Query
```bash
curl -X POST "http://localhost:8000/api/v1/analytics/query" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user123" \
  -d '{
    "query": "Show electricity usage trends for all facilities in 2024",
    "context": {
      "department": "environmental",
      "time_period": "yearly"
    }
  }'
```

#### Check Status
```bash
curl -X GET "http://localhost:8000/api/v1/analytics/query/{query_id}/status" \
  -H "X-User-ID: user123"
```

#### Get Results
```bash
curl -X GET "http://localhost:8000/api/v1/analytics/query/{query_id}?include_trace=true" \
  -H "X-User-ID: user123"
```

## Best Practices

### Query Writing Tips

1. **Be Specific**: Include facility names, time periods, and specific metrics
2. **Use Context**: Provide relevant context for better classification
3. **Natural Language**: Write queries as you would ask a human expert
4. **Avoid Ambiguity**: Be clear about what you want to analyze

#### Good Examples:
- "Show water consumption for Apex Manufacturing in Q1 2024"
- "What environmental permits expire in the next 60 days?"
- "Identify equipment with maintenance issues at Plant A"

#### Poor Examples:
- "Show usage" (too vague)
- "Get data" (no specific request)
- "Problems" (unclear intent)

### Performance Optimization

1. **Use Appropriate Timeouts**: Set realistic timeout values based on query complexity
2. **Limit Results**: Use `max_results` to limit data retrieval
3. **Cache Results**: Store frequently accessed query results
4. **Monitor Usage**: Track query patterns for optimization

### Error Handling

1. **Implement Retry Logic**: Retry failed requests with exponential backoff
2. **Handle Rate Limits**: Respect rate limits and implement backoff strategies
3. **Validate Requests**: Validate input before sending to API
4. **Log Errors**: Log detailed error information for debugging

## Support

For API support and questions:

- **Documentation**: This document and `/docs` endpoint
- **Interactive Testing**: Use `/docs` for live API testing
- **Health Monitoring**: Check `/health` endpoints regularly
- **Error Reporting**: Include request ID in error reports

---

**Last Updated**: 2024-08-20  
**API Version**: v1.0.0  
**Documentation Version**: 1.0