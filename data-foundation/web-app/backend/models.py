from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"

class DataUploadRequest(BaseModel):
    file_name: str
    file_type: str
    data_source: str
    description: Optional[str] = None

class DataUploadResponse(BaseModel):
    upload_id: str
    status: str
    message: str
    records_processed: int

class IncidentReport(BaseModel):
    id: Optional[str] = None
    title: str
    description: str
    severity: IncidentSeverity
    location: str
    reporter_name: str
    reporter_email: str
    incident_date: datetime
    created_at: Optional[datetime] = None
    status: str = "open"

class ComplianceRecord(BaseModel):
    id: Optional[str] = None
    regulation_name: str
    compliance_status: ComplianceStatus
    last_audit_date: Optional[datetime] = None
    next_audit_date: Optional[datetime] = None
    responsible_person: str
    notes: Optional[str] = None

class AnalyticsQuery(BaseModel):
    metric_type: str
    date_range: Dict[str, str]
    filters: Optional[Dict[str, Any]] = None

class AnalyticsResponse(BaseModel):
    metric_type: str
    data: List[Dict[str, Any]]
    summary: Dict[str, Any]
    generated_at: datetime

class DashboardMetrics(BaseModel):
    total_incidents: int
    open_incidents: int
    compliance_rate: float
    overdue_audits: int
    recent_incidents: List[IncidentReport]
    compliance_overview: List[ComplianceRecord]