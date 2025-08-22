from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timedelta
from models import AnalyticsQuery, AnalyticsResponse, DashboardMetrics, IncidentReport, ComplianceRecord
import random

router = APIRouter()

@router.get("/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """Get dashboard overview metrics"""
    
    # Mock recent incidents
    recent_incidents = [
        IncidentReport(
            id="INC-003",
            title="Near Miss - Forklift Operation",
            description="Pedestrian nearly struck by forklift in warehouse",
            severity="low",
            location="Warehouse C",
            reporter_name="Tom Anderson",
            reporter_email="tom.anderson@company.com",
            incident_date=datetime.now() - timedelta(days=2),
            created_at=datetime.now() - timedelta(days=2),
            status="closed"
        ),
        IncidentReport(
            id="INC-004",
            title="PPE Compliance Issue",
            description="Worker found without required safety glasses",
            severity="medium",
            location="Production Line 3",
            reporter_name="Alice Brown",
            reporter_email="alice.brown@company.com",
            incident_date=datetime.now() - timedelta(days=5),
            created_at=datetime.now() - timedelta(days=5),
            status="resolved"
        )
    ]
    
    # Mock compliance overview
    compliance_overview = [
        ComplianceRecord(
            id="COMP-003",
            regulation_name="OSHA Machine Guarding",
            compliance_status="compliant",
            last_audit_date=datetime.now() - timedelta(days=30),
            next_audit_date=datetime.now() + timedelta(days=90),
            responsible_person="David Lee",
            notes="All machinery properly guarded"
        )
    ]
    
    return DashboardMetrics(
        total_incidents=24,
        open_incidents=3,
        compliance_rate=92.5,
        overdue_audits=2,
        recent_incidents=recent_incidents,
        compliance_overview=compliance_overview
    )

@router.post("/query", response_model=AnalyticsResponse)
async def run_analytics_query(query: AnalyticsQuery):
    """Run custom analytics query"""
    
    # Mock analytics data based on metric type
    if query.metric_type == "incident_trends":
        data = [
            {"month": "Jan 2024", "incidents": 8, "severity_breakdown": {"low": 4, "medium": 3, "high": 1}},
            {"month": "Feb 2024", "incidents": 6, "severity_breakdown": {"low": 3, "medium": 2, "high": 1}},
            {"month": "Mar 2024", "incidents": 10, "severity_breakdown": {"low": 5, "medium": 4, "high": 1}},
        ]
        summary = {"total_incidents": 24, "avg_per_month": 8, "trend": "decreasing"}
        
    elif query.metric_type == "compliance_status":
        data = [
            {"regulation": "OSHA Standards", "compliant": 15, "non_compliant": 2, "pending": 1},
            {"regulation": "EPA Regulations", "compliant": 8, "non_compliant": 1, "pending": 2},
            {"regulation": "DOT Requirements", "compliant": 5, "non_compliant": 0, "pending": 1},
        ]
        summary = {"overall_compliance_rate": 92.5, "total_regulations": 34}
        
    elif query.metric_type == "safety_performance":
        data = [
            {"metric": "Days Since Last Incident", "value": 15},
            {"metric": "Total Recordable Incident Rate", "value": 2.3},
            {"metric": "Lost Time Incident Rate", "value": 0.8},
            {"metric": "Near Miss Reports", "value": 45},
        ]
        summary = {"safety_score": 87.2, "industry_benchmark": 82.1}
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported metric type")
    
    return AnalyticsResponse(
        metric_type=query.metric_type,
        data=data,
        summary=summary,
        generated_at=datetime.now()
    )

@router.get("/reports/incidents-by-location")
async def get_incidents_by_location():
    """Get incident distribution by location"""
    return {
        "data": [
            {"location": "Production Floor A", "count": 8},
            {"location": "Production Floor B", "count": 6},
            {"location": "Warehouse", "count": 4},
            {"location": "Laboratory", "count": 3},
            {"location": "Office Areas", "count": 2},
            {"location": "Maintenance Shop", "count": 1}
        ],
        "total": 24
    }

@router.get("/reports/compliance-by-category")
async def get_compliance_by_category():
    """Get compliance status by regulation category"""
    return {
        "data": [
            {"category": "Occupational Safety", "compliant": 12, "non_compliant": 1, "rate": 92.3},
            {"category": "Environmental", "compliant": 8, "non_compliant": 2, "rate": 80.0},
            {"category": "Transportation", "compliant": 5, "non_compliant": 0, "rate": 100.0},
            {"category": "Chemical Management", "compliant": 7, "non_compliant": 1, "rate": 87.5}
        ],
        "overall_rate": 89.6
    }

@router.get("/trends/monthly-incidents")
async def get_monthly_incident_trends():
    """Get monthly incident trends for the past year"""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    data = []
    
    for month in months:
        # Generate mock data with some randomness
        incident_count = random.randint(3, 12)
        data.append({
            "month": f"{month} 2024",
            "incidents": incident_count,
            "severity": {
                "low": random.randint(1, incident_count - 2),
                "medium": random.randint(1, max(1, incident_count - 4)),
                "high": random.randint(0, 2),
                "critical": random.randint(0, 1)
            }
        })
    
    return {"data": data, "year": "2024"}