"""
Executive Dashboard API Router

This module provides FastAPI endpoints for the executive dashboard functionality,
integrating with the ExecutiveDashboardService to deliver comprehensive EHS data
including environmental goals, real-time metrics, and site-specific dashboards.

Features:
- Executive dashboard with environmental goals integration
- Site-specific dashboard endpoints
- Real-time KPI monitoring and tracking
- Environmental goals progress tracking
- Location-based filtering
- Comprehensive error handling and fallback support

Created: 2025-08-31
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Query, Path as FastAPIPath, Depends
from pydantic import BaseModel, Field, validator
import logging

from src.services.executive_dashboard.dashboard_service import (
    ExecutiveDashboardService, LocationFilter, DateRangeFilter,
    AggregationPeriod, DashboardStatus, AlertLevel, create_dashboard_service
)
from src.config.ehs_goals_config import EHSGoalsConfig, ehs_goals_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["executive-dashboard"])

# Global dashboard service instance (lazy initialization)
_dashboard_service: Optional[ExecutiveDashboardService] = None


# Pydantic Models

class DashboardResponse(BaseModel):
    """Response model for executive dashboard data"""
    summary: Dict[str, Any] = Field(..., description="Dashboard summary metrics")
    kpis: Dict[str, Any] = Field(..., description="Key performance indicators")
    environmental_goals: Dict[str, Any] = Field(..., description="Environmental goals and progress")
    trends: Dict[str, Any] = Field(..., description="Trend analysis data")
    alerts: Dict[str, Any] = Field(..., description="Current alerts and notifications")
    charts: Dict[str, Any] = Field(..., description="Chart data for visualization")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")


class SiteDashboardResponse(BaseModel):
    """Response model for site-specific dashboard data"""
    site_id: str = Field(..., description="Site identifier")
    site_name: str = Field(..., description="Site display name")
    summary: Dict[str, Any] = Field(..., description="Site summary metrics")
    environmental_goals: Dict[str, Any] = Field(..., description="Site environmental goals and progress")
    performance_metrics: Dict[str, Any] = Field(..., description="Site performance data")
    trends: Dict[str, Any] = Field(..., description="Site trend analysis")
    alerts: Dict[str, Any] = Field(..., description="Site-specific alerts")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")


class HealthResponse(BaseModel):
    """Response model for service health check"""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, Any] = Field(..., description="Component health status")
    environmental_goals_status: Dict[str, Any] = Field(..., description="Environmental goals configuration status")


# Dependency Functions

def get_dashboard_service() -> ExecutiveDashboardService:
    """
    Dependency to get dashboard service instance.
    Uses lazy initialization to avoid connection overhead.
    """
    global _dashboard_service
    
    if _dashboard_service is None:
        try:
            _dashboard_service = create_dashboard_service()
            logger.info("Executive dashboard service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard service: {e}")
            raise HTTPException(
                status_code=500,
                detail="Dashboard service initialization failed"
            )
    
    return _dashboard_service


def parse_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "30d"
) -> DateRangeFilter:
    """
    Parse date range parameters into DateRangeFilter object.
    """
    now = datetime.now()
    
    if start_date and end_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            
            # Determine aggregation period based on range
            days_diff = (end_dt - start_dt).days
            if days_diff <= 90:
                aggregation_period = AggregationPeriod.DAILY
            elif days_diff <= 365:
                aggregation_period = AggregationPeriod.WEEKLY
            else:
                aggregation_period = AggregationPeriod.MONTHLY
            
            return DateRangeFilter(
                start_date=start_dt,
                end_date=end_dt,
                period=aggregation_period
            )
        except Exception as e:
            logger.warning(f"Failed to parse custom date range: {e}")
    
    # Parse period parameter
    try:
        if period.endswith('d'):
            days = int(period[:-1])
            return DateRangeFilter(
                end_date=now,
                start_date=now - timedelta(days=days),
                period=AggregationPeriod.DAILY if days <= 90 else AggregationPeriod.WEEKLY
            )
        elif period.endswith('w'):
            weeks = int(period[:-1])
            return DateRangeFilter(
                end_date=now,
                start_date=now - timedelta(weeks=weeks),
                period=AggregationPeriod.WEEKLY
            )
        elif period.endswith('m'):
            months = int(period[:-1])
            return DateRangeFilter(
                end_date=now,
                start_date=now - timedelta(days=months*30),
                period=AggregationPeriod.MONTHLY
            )
    except Exception as e:
        logger.warning(f"Failed to parse period '{period}': {e}")
    
    # Default: last 30 days
    return DateRangeFilter(
        end_date=now,
        start_date=now - timedelta(days=30),
        period=AggregationPeriod.DAILY
    )


def parse_location_filter(location: Optional[str] = None) -> Optional[LocationFilter]:
    """
    Parse location parameter into LocationFilter object.
    """
    if not location or location.lower() == 'all':
        return None
    
    # Parse comma-separated facility IDs or site names
    facility_ids = [f.strip() for f in location.split(',') if f.strip()]
    
    if not facility_ids:
        return None
    
    return LocationFilter(facility_ids=facility_ids)


def transform_alerts_to_dict(alerts_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Transform alerts list from service into expected dict format.
    
    Args:
        alerts_list: List of alert dictionaries from the dashboard service
        
    Returns:
        Dictionary with alerts categorized by severity
    """
    alerts_dict = {
        "critical": [],
        "warning": [],
        "info": [],
        "total_count": 0,
        "active_count": 0,
        "last_updated": datetime.now().isoformat()
    }
    
    if not alerts_list:
        return alerts_dict
    
    # Categorize alerts by severity
    for alert in alerts_list:
        severity = alert.get("severity", "info").lower()
        
        # Map severity levels to categories
        if severity in ["critical", "high"]:
            alerts_dict["critical"].append(alert)
        elif severity in ["medium", "warning"]:
            alerts_dict["warning"].append(alert)
        else:
            alerts_dict["info"].append(alert)
        
        # Count active alerts
        if alert.get("status") == "active":
            alerts_dict["active_count"] += 1
        
        alerts_dict["total_count"] += 1
    
    return alerts_dict


def get_fallback_dashboard_data(site_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get fallback dashboard data when service is unavailable
    """
    return {
        "summary": {
            "status": "service_unavailable",
            "message": "Dashboard service is temporarily unavailable",
            "total_facilities": 0,
            "active_alerts": 0,
            "last_updated": datetime.now().isoformat()
        },
        "environmental_goals": {
            "total_goals": 0,
            "on_track": 0,
            "behind": 0,
            "ahead": 0,
            "goals": []
        },
        "kpis": {
            "safety_incidents": {"value": 0, "trend": "stable", "status": "unknown"},
            "environmental_compliance": {"value": 0, "trend": "stable", "status": "unknown"},
            "energy_consumption": {"value": 0, "trend": "stable", "status": "unknown"}
        },
        "trends": {
            "safety": [],
            "environmental": [],
            "operational": []
        },
        "alerts": {
            "critical": [],
            "warning": [],
            "info": [],
            "total_count": 0,
            "active_count": 0,
            "last_updated": datetime.now().isoformat()
        },
        "charts": {
            "safety_trends": [],
            "environmental_trends": [],
            "goals_progress": []
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "fallback",
            "site_id": site_id,
            "version": "1.0.0"
        }
    }


# API Endpoints

@router.get("/executive", 
           response_model=DashboardResponse,
           summary="Get Executive Dashboard",
           description="Retrieve comprehensive executive dashboard with environmental goals")
async def get_executive_dashboard(
    location: Optional[str] = Query(None, description="Comma-separated facility IDs or site names"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    period: str = Query("30d", description="Time period: '30d', '90d', '1y', etc."),
    include_goals: bool = Query(True, description="Include environmental goals data"),
    include_trends: bool = Query(True, description="Include trend analysis"),
    include_forecasts: bool = Query(False, description="Include forecasting data"),
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Get comprehensive executive dashboard data including environmental goals,
    real-time KPIs, trend analysis, and site performance metrics.
    
    This endpoint provides:
    - Environmental goals and progress tracking
    - Real-time KPI metrics and status indicators  
    - Historical trend analysis
    - Risk assessment and recommendations
    - Compliance monitoring
    - Alert management
    """
    try:
        logger.info(f"Executive dashboard request: location={location}, period={period}, include_goals={include_goals}")
        
        # Parse filters
        location_filter = parse_location_filter(location)
        date_filter = parse_date_range(start_date, end_date, period)
        
        try:
            # Generate dashboard data using the service
            dashboard_data = dashboard_service.generate_executive_dashboard(
                location_filter=location_filter,
                date_filter=date_filter,
                include_trends=include_trends,
                include_recommendations=True,
                include_forecasts=include_forecasts
            )
            
            # Add environmental goals data
            if include_goals:
                try:
                    # Get environmental goals data from the service
                    goals_data = dashboard_service.get_environmental_goals_data(
                        location=location
                    )
                    dashboard_data["environmental_goals"] = goals_data
                except Exception as e:
                    logger.warning(f"Failed to get environmental goals data: {e}")
                    dashboard_data["environmental_goals"] = {
                        "status": "unavailable",
                        "message": str(e)
                    }
            
            # Ensure required fields are present
            if "summary" not in dashboard_data:
                dashboard_data["summary"] = {}
            if "kpis" not in dashboard_data:
                dashboard_data["kpis"] = {}
            if "trends" not in dashboard_data:
                dashboard_data["trends"] = {}
            if "charts" not in dashboard_data:
                dashboard_data["charts"] = {}
            if "metadata" not in dashboard_data:
                dashboard_data["metadata"] = {}
            
            # Transform alerts from list to dict format if needed
            if "alerts" in dashboard_data and isinstance(dashboard_data["alerts"], list):
                dashboard_data["alerts"] = transform_alerts_to_dict(dashboard_data["alerts"])
            elif "alerts" not in dashboard_data:
                dashboard_data["alerts"] = {
                    "critical": [],
                    "warning": [],
                    "info": [],
                    "total_count": 0,
                    "active_count": 0,
                    "last_updated": datetime.now().isoformat()
                }
            
            # Add API metadata
            dashboard_data["metadata"].update({
                "api_version": "1.0.0",
                "endpoint": "/api/dashboard/executive",
                "generated_at": datetime.now().isoformat(),
                "location_filter": location,
                "date_range": {
                    "start": date_filter.start_date.isoformat() if date_filter.start_date else None,
                    "end": date_filter.end_date.isoformat() if date_filter.end_date else None,
                    "period": period
                },
                "includes": {
                    "goals": include_goals,
                    "trends": include_trends,
                    "forecasts": include_forecasts
                }
            })
            
            logger.info("Executive dashboard request completed successfully")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard service failed: {e}")
            # Return fallback data
            fallback_data = get_fallback_dashboard_data()
            fallback_data["metadata"]["error"] = str(e)
            fallback_data["metadata"]["fallback_used"] = True
            return fallback_data
        
    except Exception as e:
        logger.error(f"Unexpected error in executive dashboard endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve executive dashboard: {str(e)}"
        )


@router.get("/executive/{site_id}",
           response_model=SiteDashboardResponse, 
           summary="Get Site-Specific Executive Dashboard",
           description="Retrieve executive dashboard data for a specific site")
async def get_site_executive_dashboard(
    site_id: str = FastAPIPath(..., description="Site identifier (e.g., 'algonquin_illinois', 'houston_texas')"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    period: str = Query("30d", description="Time period: '30d', '90d', '1y', etc."),
    include_goals: bool = Query(True, description="Include environmental goals data"),
    include_trends: bool = Query(True, description="Include trend analysis"),
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Get executive dashboard data for a specific site including site-specific
    environmental goals, performance metrics, and trend analysis.
    
    This endpoint provides:
    - Site-specific environmental goals and progress
    - Site performance metrics and KPIs
    - Historical trend analysis for the site
    - Site-specific alerts and recommendations
    - Compliance status for the site
    """
    try:
        logger.info(f"Site dashboard request: site_id={site_id}, period={period}")
        
        # Validate site_id against available sites
        available_sites = ehs_goals_config.get_site_names()
        if site_id not in available_sites:
            raise HTTPException(
                status_code=404,
                detail=f"Site '{site_id}' not found. Available sites: {', '.join(available_sites)}"
            )
        
        # Parse filters
        location_filter = LocationFilter(facility_ids=[site_id])
        date_filter = parse_date_range(start_date, end_date, period)
        
        try:
            # Generate dashboard data for the specific site
            dashboard_data = dashboard_service.generate_executive_dashboard(
                location_filter=location_filter,
                date_filter=date_filter,
                include_trends=include_trends,
                include_recommendations=True,
                include_forecasts=False
            )
            
            # Add site-specific environmental goals data
            if include_goals:
                try:
                    goals_data = dashboard_service.get_environmental_goals_data(
                        location=site_id
                    )
                    dashboard_data["environmental_goals"] = goals_data
                except Exception as e:
                    logger.warning(f"Failed to get environmental goals data for site {site_id}: {e}")
                    dashboard_data["environmental_goals"] = {
                        "status": "unavailable",
                        "message": str(e)
                    }
            
            # Get site display name
            site_display_name = site_id.replace('_', ' ').title()
            
            # Transform alerts from list to dict format if needed
            alerts_data = dashboard_data.get("alerts", [])
            if isinstance(alerts_data, list):
                alerts_data = transform_alerts_to_dict(alerts_data)
            elif not isinstance(alerts_data, dict):
                alerts_data = {
                    "critical": [],
                    "warning": [],
                    "info": [],
                    "total_count": 0,
                    "active_count": 0,
                    "last_updated": datetime.now().isoformat()
                }
            
            # Structure response for site-specific dashboard
            response_data = {
                "site_id": site_id,
                "site_name": site_display_name,
                "summary": dashboard_data.get("summary", {}),
                "environmental_goals": dashboard_data.get("environmental_goals", {}),
                "performance_metrics": dashboard_data.get("kpis", {}),
                "trends": dashboard_data.get("trends", {}),
                "alerts": alerts_data,
                "metadata": {
                    "api_version": "1.0.0",
                    "endpoint": f"/api/dashboard/executive/{site_id}",
                    "generated_at": datetime.now().isoformat(),
                    "site_id": site_id,
                    "site_name": site_display_name,
                    "date_range": {
                        "start": date_filter.start_date.isoformat() if date_filter.start_date else None,
                        "end": date_filter.end_date.isoformat() if date_filter.end_date else None,
                        "period": period
                    },
                    "includes": {
                        "goals": include_goals,
                        "trends": include_trends
                    }
                }
            }
            
            logger.info(f"Site dashboard request for {site_id} completed successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Dashboard service failed for site {site_id}: {e}")
            # Return fallback data
            fallback_data = get_fallback_dashboard_data(site_id)
            response_data = {
                "site_id": site_id,
                "site_name": site_id.replace('_', ' ').title(),
                "summary": fallback_data["summary"],
                "environmental_goals": fallback_data["environmental_goals"],
                "performance_metrics": fallback_data["kpis"],
                "trends": fallback_data["trends"],
                "alerts": fallback_data["alerts"],
                "metadata": fallback_data["metadata"]
            }
            response_data["metadata"]["error"] = str(e)
            response_data["metadata"]["fallback_used"] = True
            return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in site dashboard endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve site dashboard for {site_id}: {str(e)}"
        )


@router.get("/health",
           response_model=HealthResponse,
           summary="Service Health Check", 
           description="Check health status of dashboard service and environmental goals configuration")
async def health_check(
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Perform comprehensive health check of the dashboard service,
    including environmental goals configuration status.
    """
    try:
        # Get service health data
        health_data = dashboard_service.health_check()
        
        # Check environmental goals configuration
        try:
            goals_valid = ehs_goals_config.validate_configuration()
            total_goals = len(ehs_goals_config.get_all_goals()) if hasattr(ehs_goals_config, 'get_all_goals') else 0
            available_sites = ehs_goals_config.get_site_names()
            available_categories = ehs_goals_config.get_category_names()
            
            environmental_goals_status = {
                "status": "healthy" if goals_valid else "unhealthy",
                "configuration_valid": goals_valid,
                "total_goals": total_goals,
                "available_sites": available_sites,
                "available_categories": available_categories
            }
        except Exception as e:
            logger.error(f"Environmental goals health check failed: {e}")
            environmental_goals_status = {
                "status": "unhealthy",
                "error": str(e),
                "configuration_valid": False
            }
        
        # Determine overall status
        overall_status = "healthy"
        if (any(component.get("status") == "unhealthy" for component in health_data.values() if isinstance(component, dict)) or
            environmental_goals_status.get("status") == "unhealthy"):
            overall_status = "unhealthy"
        elif (any(component.get("status") == "degraded" for component in health_data.values() if isinstance(component, dict)) or
              environmental_goals_status.get("status") == "degraded"):
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            components=health_data,
            environmental_goals_status=environmental_goals_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0", 
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)},
            environmental_goals_status={
                "status": "unhealthy",
                "error": "Health check failed"
            }
        )


# Cleanup function
@router.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global _dashboard_service
    if _dashboard_service:
        try:
            _dashboard_service.close()
            logger.info("Executive dashboard service closed during shutdown")
        except Exception as e:
            logger.error(f"Error closing dashboard service: {e}")
    _dashboard_service = None