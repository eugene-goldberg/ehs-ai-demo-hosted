"""
Executive Dashboard API v2

This module provides a comprehensive FastAPI router for the executive dashboard functionality,
integrating with the ExecutiveDashboardService to deliver real-time EHS data, trend analysis,
risk assessment, and dynamic dashboard generation capabilities.

Features:
- FastAPI router pattern with comprehensive error handling
- Integration with the new ExecutiveDashboardService
- Backward compatibility with existing API parameters (location, dateRange)
- Support for both dynamic data and fallback to static files
- Enhanced API parameters for advanced features
- Comprehensive API documentation and validation
- Real-time metrics and KPI endpoints
- Trend analysis and forecasting capabilities
- Risk assessment integration
- Flexible filtering and aggregation options

Created: 2025-08-28
Version: 2.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add the src directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = parent_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fastapi import APIRouter, HTTPException, Query, Path as FastAPIPath, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Import dashboard service
from services.executive_dashboard.dashboard_service import (
    ExecutiveDashboardService, LocationFilter, DateRangeFilter, 
    AggregationPeriod, DashboardStatus, AlertLevel, create_dashboard_service
)
from neo4j_enhancements.queries.analytics.aggregation_layer import KPIMetric

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI router - Remove the prefix since it will be added when including the router
router = APIRouter(
    tags=["Executive Dashboard v2"],
    responses={
        404: {"description": "Not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Static dashboard files path
STATIC_DASHBOARD_PATH = Path(__file__).parent.parent.parent / "docs" / "static_dashboards"

# Global dashboard service instance (lazy initialization)
_dashboard_service: Optional[ExecutiveDashboardService] = None


# Pydantic Models

class DashboardRequest(BaseModel):
    """Request model for executive dashboard"""
    location: Optional[str] = Field(None, description="Comma-separated facility IDs or 'all' for all facilities")
    dateRange: Optional[str] = Field(None, description="Date range in format 'YYYY-MM-DD:YYYY-MM-DD' or preset like '30d', '90d', '1y'")
    aggregationPeriod: Optional[str] = Field("daily", description="Aggregation period: 'daily', 'weekly', 'monthly', 'quarterly'")
    includeTrends: bool = Field(True, description="Include trend analysis in response")
    includeRecommendations: bool = Field(True, description="Include AI-generated recommendations")
    includeForecasts: bool = Field(False, description="Include forecasting data (premium feature)")
    includeAlerts: bool = Field(True, description="Include alerts and notifications")
    cacheTimeout: Optional[int] = Field(300, ge=60, le=3600, description="Cache timeout in seconds (60-3600)")
    format: str = Field("full", description="Response format: 'full', 'summary', 'kpis_only', 'charts_only'")

    @validator('aggregationPeriod')
    def validate_aggregation_period(cls, v):
        valid_periods = {'daily', 'weekly', 'monthly', 'quarterly'}
        if v not in valid_periods:
            raise ValueError(f"aggregationPeriod must be one of: {', '.join(valid_periods)}")
        return v

    @validator('format')
    def validate_format(cls, v):
        valid_formats = {'full', 'summary', 'kpis_only', 'charts_only', 'alerts_only'}
        if v not in valid_formats:
            raise ValueError(f"format must be one of: {', '.join(valid_formats)}")
        return v


class LocationsResponse(BaseModel):
    """Response model for available locations"""
    locations: List[Dict[str, Any]] = Field(..., description="List of available locations/facilities")
    total_count: int = Field(..., description="Total number of locations")
    generated_at: str = Field(..., description="Response generation timestamp")


class MetricsResponse(BaseModel):
    """Response model for real-time metrics"""
    metrics: Dict[str, Any] = Field(..., description="Real-time metrics data")
    alert_level: str = Field(..., description="Current alert level")
    last_updated: str = Field(..., description="Last update timestamp")


class HealthResponse(BaseModel):
    """Response model for service health check"""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, Any] = Field(..., description="Component health status")


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
            logger.info("Dashboard service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard service: {e}")
            raise HTTPException(
                status_code=500,
                detail="Dashboard service initialization failed"
            )
    
    return _dashboard_service


def parse_location_filter(location: Optional[str]) -> Optional[LocationFilter]:
    """Parse location parameter into LocationFilter object"""
    if not location or location.lower() == 'all':
        return None
    
    # Parse comma-separated facility IDs
    facility_ids = [f.strip() for f in location.split(',') if f.strip()]
    
    if not facility_ids:
        return None
    
    return LocationFilter(facility_ids=facility_ids)


def parse_date_range(date_range: Optional[str]) -> DateRangeFilter:
    """Parse dateRange parameter into DateRangeFilter object"""
    if not date_range:
        # Default: last 30 days
        return DateRangeFilter(
            end_date=datetime.now(),
            start_date=datetime.now() - timedelta(days=30),
            period=AggregationPeriod.DAILY
        )
    
    try:
        # Handle preset ranges
        if date_range.endswith('d'):
            days = int(date_range[:-1])
            return DateRangeFilter(
                end_date=datetime.now(),
                start_date=datetime.now() - timedelta(days=days),
                period=AggregationPeriod.DAILY if days <= 90 else AggregationPeriod.WEEKLY
            )
        elif date_range.endswith('w'):
            weeks = int(date_range[:-1])
            return DateRangeFilter(
                end_date=datetime.now(),
                start_date=datetime.now() - timedelta(weeks=weeks),
                period=AggregationPeriod.WEEKLY
            )
        elif date_range.endswith('m'):
            months = int(date_range[:-1])
            return DateRangeFilter(
                end_date=datetime.now(),
                start_date=datetime.now() - timedelta(days=months*30),
                period=AggregationPeriod.MONTHLY
            )
        elif date_range.endswith('y'):
            years = int(date_range[:-1])
            return DateRangeFilter(
                end_date=datetime.now(),
                start_date=datetime.now() - timedelta(days=years*365),
                period=AggregationPeriod.MONTHLY
            )
        elif ':' in date_range:
            # Handle explicit date range
            start_str, end_str = date_range.split(':', 1)
            start_date = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
            
            # Determine period based on range length
            days_diff = (end_date - start_date).days
            if days_diff <= 90:
                period = AggregationPeriod.DAILY
            elif days_diff <= 365:
                period = AggregationPeriod.WEEKLY
            else:
                period = AggregationPeriod.MONTHLY
            
            return DateRangeFilter(
                start_date=start_date,
                end_date=end_date,
                period=period
            )
        else:
            raise ValueError(f"Invalid date range format: {date_range}")
    
    except Exception as e:
        logger.warning(f"Failed to parse date range '{date_range}': {e}")
        # Fallback to default
        return DateRangeFilter(
            end_date=datetime.now(),
            start_date=datetime.now() - timedelta(days=30),
            period=AggregationPeriod.DAILY
        )


def get_static_dashboard_fallback(location: Optional[str] = None) -> Dict[str, Any]:
    """
    Get static dashboard data as fallback when service is unavailable
    """
    try:
        # Try to load static dashboard file
        static_file = STATIC_DASHBOARD_PATH / "executive_dashboard_sample.json"
        
        if static_file.exists():
            with open(static_file, 'r') as f:
                dashboard_data = json.load(f)
                
            # Add metadata indicating this is static data
            dashboard_data["metadata"] = dashboard_data.get("metadata", {})
            dashboard_data["metadata"]["source"] = "static_fallback"
            dashboard_data["metadata"]["generated_at"] = datetime.now().isoformat()
            dashboard_data["metadata"]["note"] = "Static fallback data - dashboard service unavailable"
            
            return dashboard_data
    
    except Exception as e:
        logger.error(f"Failed to load static dashboard fallback: {e}")
    
    # Generate minimal fallback response
    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "minimal_fallback",
            "note": "Dashboard service unavailable - minimal response generated",
            "version": "2.0.0"
        },
        "summary": {
            "status": "service_unavailable",
            "message": "Dashboard service is temporarily unavailable"
        },
        "error": "Dashboard service unavailable"
    }


# API Endpoints

@router.get("/executive-dashboard", 
           summary="Get Executive Dashboard Data",
           description="Retrieve comprehensive executive dashboard data with flexible filtering and format options")
async def get_executive_dashboard(
    location: Optional[str] = Query(None, description="Comma-separated facility IDs or 'all' for all facilities"),
    dateRange: Optional[str] = Query(None, description="Date range: 'YYYY-MM-DD:YYYY-MM-DD' or '30d', '90d', '1y'"),
    aggregationPeriod: Optional[str] = Query("daily", description="Aggregation period: daily, weekly, monthly, quarterly"),
    includeTrends: bool = Query(True, description="Include trend analysis"),
    includeRecommendations: bool = Query(True, description="Include AI recommendations"),
    includeForecasts: bool = Query(False, description="Include forecasts (premium feature)"),
    includeAlerts: bool = Query(True, description="Include alerts and notifications"),
    format: str = Query("full", description="Response format: full, summary, kpis_only, charts_only"),
    useCache: bool = Query(True, description="Use cached results if available"),
    cacheTimeout: Optional[int] = Query(300, description="Cache timeout in seconds"),
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Get comprehensive executive dashboard data with advanced filtering and customization options.
    
    This endpoint provides:
    - Real-time KPI metrics and status indicators
    - Historical trend analysis and anomaly detection
    - Risk assessment and safety recommendations
    - Compliance monitoring and audit results
    - Dynamic chart data for visualization
    - Alert management and notifications
    - Forecasting capabilities (when enabled)
    
    The API supports both dynamic data from Neo4j and fallback to static files for reliability.
    """
    try:
        logger.info(f"Dashboard request: location={location}, dateRange={dateRange}, format={format}")
        
        # Validate and parse request parameters
        try:
            request_data = DashboardRequest(
                location=location,
                dateRange=dateRange,
                aggregationPeriod=aggregationPeriod,
                includeTrends=includeTrends,
                includeRecommendations=includeRecommendations,
                includeForecasts=includeForecasts,
                includeAlerts=includeAlerts,
                cacheTimeout=cacheTimeout,
                format=format
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid request parameters: {e}")
        
        # Parse filters
        location_filter = parse_location_filter(location)
        date_filter = parse_date_range(dateRange)
        
        # Map aggregation period
        period_map = {
            'daily': AggregationPeriod.DAILY,
            'weekly': AggregationPeriod.WEEKLY,
            'monthly': AggregationPeriod.MONTHLY,
            'quarterly': AggregationPeriod.QUARTERLY
        }
        date_filter.period = period_map.get(aggregationPeriod, AggregationPeriod.DAILY)
        
        # Configure cache if not using cache
        if not useCache:
            dashboard_service.clear_cache()
        
        try:
            # Generate dashboard data
            dashboard_data = dashboard_service.generate_executive_dashboard(
                location_filter=location_filter,
                date_filter=date_filter,
                include_trends=includeTrends,
                include_recommendations=includeRecommendations,
                include_forecasts=includeForecasts
            )
            
            # Check for service errors
            if 'error' in dashboard_data:
                logger.warning(f"Dashboard service returned error: {dashboard_data['error']}")
                # Fall back to static data
                dashboard_data = get_static_dashboard_fallback(location)
            
        except Exception as e:
            logger.error(f"Dashboard service failed: {e}")
            # Fall back to static data
            dashboard_data = get_static_dashboard_fallback(location)
        
        # Apply format filtering
        if format == "summary":
            dashboard_data = {
                "metadata": dashboard_data.get("metadata", {}),
                "summary": dashboard_data.get("summary", {}),
                "status": dashboard_data.get("status", {}),
                "alerts": {
                    "summary": dashboard_data.get("alerts", {}).get("summary", {})
                }
            }
        elif format == "kpis_only":
            dashboard_data = {
                "metadata": dashboard_data.get("metadata", {}),
                "kpis": dashboard_data.get("kpis", {}),
                "summary": dashboard_data.get("summary", {})
            }
        elif format == "charts_only":
            dashboard_data = {
                "metadata": dashboard_data.get("metadata", {}),
                "charts": dashboard_data.get("charts", {})
            }
        elif format == "alerts_only":
            dashboard_data = {
                "metadata": dashboard_data.get("metadata", {}),
                "alerts": dashboard_data.get("alerts", {}),
                "status": dashboard_data.get("status", {})
            }
        
        # Add API metadata
        dashboard_data["api_info"] = {
            "version": "2.0.0",
            "endpoint": "/api/v2/executive-dashboard",
            "format": format,
            "cache_used": useCache,
            "backwards_compatible": True
        }
        
        logger.info(f"Dashboard request completed successfully (format: {format})")
        return dashboard_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in executive dashboard endpoint: {e}")
        # Return static fallback on any error
        try:
            fallback_data = get_static_dashboard_fallback(location)
            fallback_data["api_info"] = {
                "version": "2.0.0",
                "endpoint": "/api/v2/executive-dashboard",
                "error": str(e),
                "fallback_used": True
            }
            return fallback_data
        except:
            raise HTTPException(
                status_code=500,
                detail=f"Dashboard service failed and fallback unavailable: {str(e)}"
            )


@router.get("/dashboard-summary",
           response_model=Dict[str, Any],
           summary="Get Dashboard Summary",
           description="Get high-level dashboard summary for quick overview")
async def get_dashboard_summary(
    location: Optional[str] = Query(None, description="Comma-separated facility IDs or 'all'"),
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Get a high-level dashboard summary for quick overview.
    This is a lightweight endpoint that provides key metrics without detailed data.
    """
    try:
        location_filter = parse_location_filter(location)
        facility_ids = location_filter.facility_ids if location_filter else None
        
        summary_data = dashboard_service.get_dashboard_summary(facility_ids)
        
        return {
            "data": summary_data,
            "api_info": {
                "version": "2.0.0",
                "endpoint": "/api/v2/dashboard-summary"
            }
        }
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/real-time-metrics",
           response_model=MetricsResponse,
           summary="Get Real-time Metrics",
           description="Get current real-time metrics and alert status")
async def get_real_time_metrics(
    location: Optional[str] = Query(None, description="Comma-separated facility IDs or 'all'"),
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Get real-time metrics including current alert levels and active incidents.
    This endpoint provides the most up-to-date status information.
    """
    try:
        location_filter = parse_location_filter(location)
        facility_ids = location_filter.facility_ids if location_filter else None
        
        metrics_data = dashboard_service.get_real_time_metrics(facility_ids)
        
        return MetricsResponse(
            metrics=metrics_data.get("metrics", {}),
            alert_level=metrics_data.get("alert_level", "GREEN"),
            last_updated=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kpis",
           summary="Get KPI Details",
           description="Get detailed KPI information with historical context")
async def get_kpi_details(
    location: Optional[str] = Query(None, description="Comma-separated facility IDs or 'all'"),
    dateRange: Optional[str] = Query("30d", description="Date range for KPI calculation"),
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Get detailed KPI information including historical trends and benchmarks.
    """
    try:
        location_filter = parse_location_filter(location)
        facility_ids = location_filter.facility_ids if location_filter else None
        
        # Parse date range for days
        date_range_days = 30
        if dateRange.endswith('d'):
            try:
                date_range_days = int(dateRange[:-1])
            except:
                pass
        
        kpi_data = dashboard_service.get_kpi_details(facility_ids, date_range_days)
        
        return {
            "data": kpi_data,
            "api_info": {
                "version": "2.0.0",
                "endpoint": "/api/v2/kpis",
                "date_range_days": date_range_days
            }
        }
    except Exception as e:
        logger.error(f"Error getting KPI details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/locations",
           response_model=LocationsResponse,
           summary="Get Available Locations",
           description="Get list of available facilities and locations for filtering")
async def get_available_locations(
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Get list of available facilities and locations that can be used for filtering.
    """
    try:
        # Get facility overview
        facilities = dashboard_service.analytics.get_facility_overview()
        
        locations = []
        for facility in facilities:
            locations.append({
                "facility_id": facility.get("facility_id"),
                "facility_name": facility.get("facility_name"),
                "location": facility.get("location"),
                "facility_type": facility.get("facility_type"),
                "status": facility.get("status", "active")
            })
        
        return LocationsResponse(
            locations=locations,
            total_count=len(locations),
            generated_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting available locations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health",
           response_model=HealthResponse,
           summary="Service Health Check",
           description="Check the health status of the dashboard service and its components")
async def health_check(
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Perform comprehensive health check of the dashboard service and its dependencies.
    """
    try:
        health_data = dashboard_service.health_check()
        
        overall_status = "healthy"
        if any(component.get("status") == "unhealthy" for component in health_data.values() if isinstance(component, dict)):
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            version="2.0.0",
            timestamp=datetime.now().isoformat(),
            components=health_data
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="2.0.0",
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)}
        )


@router.post("/cache/clear",
            summary="Clear Dashboard Cache",
            description="Clear the dashboard service cache to force fresh data retrieval")
async def clear_dashboard_cache(
    background_tasks: BackgroundTasks,
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Clear the dashboard service cache to ensure fresh data on next request.
    This operation runs in the background to avoid blocking the response.
    """
    def clear_cache():
        dashboard_service.clear_cache()
        logger.info("Dashboard cache cleared")
    
    background_tasks.add_task(clear_cache)
    
    return {
        "message": "Cache clearing initiated",
        "timestamp": datetime.now().isoformat(),
        "api_info": {
            "version": "2.0.0",
            "endpoint": "/api/v2/cache/clear"
        }
    }


@router.get("/static-dashboard/{filename}",
           summary="Get Static Dashboard File",
           description="Retrieve static dashboard files for fallback scenarios")
async def get_static_dashboard_file(
    filename: str = FastAPIPath(..., description="Static dashboard filename")
):
    """
    Serve static dashboard files as fallback when dynamic service is unavailable.
    """
    try:
        file_path = STATIC_DASHBOARD_PATH / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Static dashboard file not found")
        
        if not file_path.suffix.lower() == '.json':
            raise HTTPException(status_code=400, detail="Only JSON files are supported")
        
        return FileResponse(
            path=file_path,
            media_type="application/json",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving static dashboard file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Legacy compatibility endpoints (v1 format support)

@router.get("/dashboard",
           summary="Legacy Dashboard Endpoint (v1 compatibility)",
           description="Legacy endpoint for backward compatibility with v1 API clients")
async def get_legacy_dashboard(
    location: Optional[str] = Query(None, description="Location parameter (legacy format)"),
    dateRange: Optional[str] = Query(None, description="Date range parameter (legacy format)"),
    dashboard_service: ExecutiveDashboardService = Depends(get_dashboard_service)
):
    """
    Legacy dashboard endpoint for backward compatibility with v1 API clients.
    This endpoint automatically maps v1 parameters to v2 functionality.
    """
    # Forward to the main endpoint with appropriate mapping
    return await get_executive_dashboard(
        location=location,
        dateRange=dateRange,
        aggregationPeriod="daily",
        includeTrends=True,
        includeRecommendations=True,
        includeForecasts=False,
        includeAlerts=True,
        format="full",
        useCache=True,
        cacheTimeout=300,
        dashboard_service=dashboard_service
    )


# Cleanup function
@router.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global _dashboard_service
    if _dashboard_service:
        try:
            _dashboard_service.close()
            logger.info("Dashboard service closed during shutdown")
        except Exception as e:
            logger.error(f"Error closing dashboard service: {e}")
    _dashboard_service = None


# Export the router for integration with main application
executive_dashboard_router = router

# Example usage
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Executive Dashboard API v2", version="2.0.0")
    app.include_router(router)
    
    print("Starting Executive Dashboard API v2 server...")
    print("Access API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")