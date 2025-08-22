"""
API Models for EHS Analytics

This module defines Pydantic models for request/response validation
and data serialization for the EHS Analytics API endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

from ..agents.query_router import IntentType, RetrieverType, EntityExtraction


class QueryStatus(str, Enum):
    """Status of a query processing request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorType(str, Enum):
    """Types of API errors."""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    DATABASE_ERROR = "database_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorDetail(BaseModel):
    """Detailed error information."""
    error_type: ErrorType
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error: ErrorDetail


# Entity Models (matching workflow state)
class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction results."""
    facilities: List[str] = Field(default_factory=list, description="Extracted facility names")
    date_ranges: List[str] = Field(default_factory=list, description="Extracted date ranges")
    equipment: List[str] = Field(default_factory=list, description="Extracted equipment names")
    pollutants: List[str] = Field(default_factory=list, description="Extracted pollutant types")
    regulations: List[str] = Field(default_factory=list, description="Extracted regulation references")
    departments: List[str] = Field(default_factory=list, description="Extracted department names")
    metrics: List[str] = Field(default_factory=list, description="Extracted metric types")

    @classmethod
    def from_entity_extraction(cls, extraction: EntityExtraction) -> "EntityExtractionResponse":
        """Convert from internal EntityExtraction to response model."""
        return cls(
            facilities=extraction.facilities,
            date_ranges=extraction.date_ranges,
            equipment=extraction.equipment,
            pollutants=extraction.pollutants,
            regulations=extraction.regulations,
            departments=extraction.departments,
            metrics=extraction.metrics
        )


# Query Classification Models
class QueryClassificationResponse(BaseModel):
    """Response model for query classification results."""
    intent_type: IntentType
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    entities_identified: EntityExtractionResponse
    suggested_retriever: RetrieverType
    reasoning: str
    query_rewrite: Optional[str] = None


# Data Retrieval Models
class RetrievalMetadata(BaseModel):
    """Metadata for retrieved data."""
    source: str
    confidence: float
    timestamp: datetime
    query_used: str
    node_count: Optional[int] = None
    relationship_count: Optional[int] = None


class RetrievedDocument(BaseModel):
    """A single retrieved document or data point."""
    content: str
    metadata: RetrievalMetadata
    relevance_score: float = Field(ge=0.0, le=1.0)
    document_id: Optional[str] = None


class RetrievalResults(BaseModel):
    """Results from data retrieval operations."""
    documents: List[RetrievedDocument]
    total_count: int
    retrieval_strategy: str
    execution_time_ms: int
    query_embedding: Optional[List[float]] = None


# Analysis Results Models
class RiskAssessment(BaseModel):
    """Risk assessment analysis results."""
    risk_level: str = Field(..., description="Overall risk level (low, medium, high, critical)")
    risk_score: float = Field(ge=0.0, le=1.0, description="Numerical risk score")
    risk_factors: List[str] = Field(description="Identified risk factors")
    mitigation_suggestions: List[str] = Field(description="Suggested mitigation actions")
    confidence: float = Field(ge=0.0, le=1.0)


class ComplianceStatus(BaseModel):
    """Compliance assessment results."""
    compliant: bool
    compliance_score: float = Field(ge=0.0, le=1.0)
    violations: List[str] = Field(default_factory=list)
    requirements_met: List[str] = Field(default_factory=list)
    next_review_date: Optional[datetime] = None


class ConsumptionAnalysis(BaseModel):
    """Consumption analysis results."""
    consumption_type: str  # energy, water, gas, etc.
    current_value: float
    unit: str
    trend: str  # increasing, decreasing, stable
    trend_percentage: float
    comparison_period: str
    efficiency_rating: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)


class EmissionTracking(BaseModel):
    """Emission tracking results."""
    emission_type: str
    total_emissions: float
    unit: str
    scope: str  # Scope 1, 2, or 3
    reduction_target: Optional[float] = None
    progress_to_target: Optional[float] = None
    key_sources: List[str] = Field(default_factory=list)


class EquipmentEfficiency(BaseModel):
    """Equipment efficiency analysis."""
    equipment_id: str
    equipment_type: str
    efficiency_score: float = Field(ge=0.0, le=1.0)
    uptime_percentage: float = Field(ge=0.0, le=100.0)
    maintenance_status: str
    performance_issues: List[str] = Field(default_factory=list)
    optimization_suggestions: List[str] = Field(default_factory=list)


class PermitStatus(BaseModel):
    """Permit status information."""
    permit_id: str
    permit_type: str
    status: str  # active, expired, pending_renewal, suspended
    issue_date: datetime
    expiration_date: datetime
    renewal_required: bool
    compliance_status: str
    action_required: Optional[str] = None


# Analysis Response Union Type
AnalysisResult = Union[
    RiskAssessment,
    ComplianceStatus,
    ConsumptionAnalysis,
    EmissionTracking,
    EquipmentEfficiency,
    PermitStatus,
    Dict[str, Any]  # Fallback for general inquiries
]


# Recommendation Models
class Recommendation(BaseModel):
    """A single recommendation."""
    title: str
    description: str
    priority: str  # low, medium, high, critical
    category: str  # cost_reduction, compliance, efficiency, risk_mitigation
    estimated_cost: Optional[float] = None
    estimated_savings: Optional[float] = None
    payback_period_months: Optional[int] = None
    implementation_effort: str  # low, medium, high
    confidence: float = Field(ge=0.0, le=1.0)


class RecommendationEngine(BaseModel):
    """Recommendation engine results."""
    recommendations: List[Recommendation]
    total_estimated_cost: Optional[float] = None
    total_estimated_savings: Optional[float] = None
    recommendations_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# Workflow State Models
class WorkflowState(BaseModel):
    """Current state of the LangGraph workflow."""
    query_id: str
    original_query: str
    current_step: str
    classification: Optional[QueryClassificationResponse] = None
    retrieval_results: Optional[RetrievalResults] = None
    analysis_results: Optional[List[AnalysisResult]] = None
    recommendations: Optional[RecommendationEngine] = None
    error: Optional[ErrorDetail] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Request Models
class QueryRequest(BaseModel):
    """Request model for natural language EHS queries."""
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    user_id: Optional[str] = Field(None, description="User identifier for tracking")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query content."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class QueryProcessingOptions(BaseModel):
    """Options for query processing."""
    include_recommendations: bool = Field(default=True, description="Include recommendation generation")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    timeout_seconds: int = Field(default=300, ge=30, le=600, description="Processing timeout")
    retrieval_strategy: Optional[str] = Field(None, description="Override retrieval strategy")
    explain_reasoning: bool = Field(default=False, description="Include detailed reasoning")


# Response Models
class QueryResponse(BaseResponse):
    """Response for immediate query submission."""
    query_id: str = Field(..., description="Unique identifier for the query")
    status: QueryStatus
    estimated_completion_time: Optional[int] = Field(None, description="Estimated completion time in seconds")


class QueryResultResponse(BaseResponse):
    """Response with complete query results."""
    query_id: str
    status: QueryStatus
    original_query: str
    processing_time_ms: int
    classification: Optional[QueryClassificationResponse] = None
    retrieval_results: Optional[RetrievalResults] = None
    analysis_results: Optional[List[AnalysisResult]] = None
    recommendations: Optional[RecommendationEngine] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    workflow_trace: Optional[List[str]] = Field(None, description="Workflow execution trace")


class QueryStatusResponse(BaseResponse):
    """Response for query status check."""
    query_id: str
    status: QueryStatus
    progress_percentage: Optional[int] = Field(None, ge=0, le=100)
    current_step: Optional[str] = None
    estimated_remaining_time: Optional[int] = Field(None, description="Estimated remaining time in seconds")


# Health Check Models
class ServiceHealth(BaseModel):
    """Health status for a service component."""
    service_name: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseResponse):
    """Health check response."""
    overall_status: str  # healthy, degraded, unhealthy
    services: List[ServiceHealth]
    uptime_seconds: int
    version: str
    environment: str


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    size: int = Field(default=20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size


class PaginatedResponse(BaseModel):
    """Base paginated response."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(cls, items: List[Any], total: int, pagination: PaginationParams) -> "PaginatedResponse":
        """Create paginated response from items and pagination params."""
        pages = (total + pagination.size - 1) // pagination.size
        return cls(
            items=items,
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=pages,
            has_next=pagination.page < pages,
            has_previous=pagination.page > 1
        )