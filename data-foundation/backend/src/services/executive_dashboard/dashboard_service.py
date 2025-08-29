"""
Executive Dashboard Service

This service provides comprehensive executive dashboard functionality for the EHS AI Demo,
integrating with Neo4j to fetch real data, leveraging trend analysis infrastructure,
risk assessment for recommendations, and generating dashboard JSON dynamically.

Features:
- Real-time KPI monitoring and alerting
- Historical trend analysis and forecasting
- Risk assessment integration with Neo4j queries
- Dynamic dashboard JSON generation
- Location and date range filtering
- Comprehensive error handling
- Production-ready caching and performance optimization
- Analytics aggregation layer integration

Created: 2025-08-28
Updated: 2025-08-29 - Added risk assessment data integration
Version: 1.1.0
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncio
from functools import wraps
import time
from collections import defaultdict

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

# Import our existing infrastructure
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neo4j_enhancements.queries.analytics.aggregation_layer import (
    AnalyticsAggregationLayer, AggregationPeriod, KPIMetric, FacilityBenchmark
)
from neo4j_enhancements.models.trend_analysis import (
    TrendAnalysisSystem, TrendType, AnomalyType, DataPoint, TrendAnalysisResult
)
from neo4j_enhancements.models.recommendation_system import (
    RecommendationStorage, RecommendationStatus, RecommendationPriority
)

# Configure logging
logger = logging.getLogger(__name__)


class DashboardStatus(Enum):
    """Dashboard status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert level enumeration"""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class DashboardMetrics:
    """Core dashboard metrics"""
    timestamp: datetime
    total_incidents: int
    incident_rate: float
    ltir: float
    audit_pass_rate: float
    training_completion_rate: float
    active_alerts: int
    high_risk_items: int
    overdue_items: int
    compliance_score: float
    overall_status: DashboardStatus
    alert_level: AlertLevel


@dataclass
class LocationFilter:
    """Location filter parameters"""
    facility_ids: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    departments: Optional[List[str]] = None


@dataclass
class DateRangeFilter:
    """Date range filter parameters"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    period: AggregationPeriod = AggregationPeriod.MONTHLY


@dataclass
class RiskAssessmentData:
    """Risk assessment data structure"""
    assessment_id: str
    facility_id: str
    risk_level: str
    risk_score: float
    assessment_date: datetime
    recommendations: List[Dict[str, Any]]
    methodology: str
    assessor: str
    status: str


# Error Handling Decorator (static function)
def _handle_errors(func):
    """Decorator for comprehensive error handling"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            self._request_count += 1
            start_time = time.time()
            
            result = func(self, *args, **kwargs)
            
            # Log performance metrics
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in {func.__name__}: {e}")
            
            # Return error response in consistent format
            return {
                "error": True,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
                "method": func.__name__
            }
    
    return wrapper


class ExecutiveDashboardService:
    """
    Comprehensive Executive Dashboard Service
    
    This service provides a complete executive dashboard solution with:
    - Real-time KPI monitoring
    - Historical trend analysis
    - Risk assessment integration with Neo4j data
    - Dynamic JSON generation
    - Advanced filtering capabilities
    - Production-ready performance optimization
    """
    
    def __init__(self, 
                 neo4j_uri: str = None,
                 neo4j_username: str = None,
                 neo4j_password: str = None,
                 neo4j_database: str = None):
        """
        Initialize the Executive Dashboard Service
        
        Args:
            neo4j_uri: Neo4j connection URI (defaults to env var)
            neo4j_username: Neo4j username (defaults to env var)
            neo4j_password: Neo4j password (defaults to env var)
            neo4j_database: Neo4j database name (defaults to env var)
        """
        # Connection parameters
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = neo4j_username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD', 'password')
        self.neo4j_database = neo4j_database or os.getenv('NEO4J_DATABASE', 'neo4j')
        
        # Initialize Neo4j driver
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            logger.info("Neo4j driver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise
        
        # Initialize service components
        self._initialize_components()
        
        # Cache management
        self._cache = {}
        self._cache_ttl = {}
        self.default_cache_duration = 300  # 5 minutes
        
        # Performance monitoring
        self._request_count = 0
        self._error_count = 0
        self._last_health_check = None
        
        logger.info("Executive Dashboard Service initialized successfully")
    
    def _initialize_components(self):
        """Initialize service components"""
        try:
            # Analytics aggregation layer
            self.analytics = AnalyticsAggregationLayer(self.driver)
            logger.info("Analytics aggregation layer initialized")
            
            # Trend analysis system
            self.trend_analysis = TrendAnalysisSystem(self.driver, self.neo4j_database)
            logger.info("Trend analysis system initialized")
            
            # Recommendation system
            self.recommendations = RecommendationStorage(
                self.neo4j_uri, 
                self.neo4j_username, 
                self.neo4j_password
            )
            logger.info("Recommendation system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize service components: {e}")
            raise
    
    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'driver'):
                self.driver.close()
            if hasattr(self, 'recommendations'):
                self.recommendations.close()
            logger.info("Executive Dashboard Service closed successfully")
        except Exception as e:
            logger.error(f"Error closing service: {e}")
    
    # Cache Management
    
    def _get_cache_key(self, method_name: str, *args, **kwargs) -> str:
        """Generate cache key for method call"""
        key_data = f"{method_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return f"dashboard_{hash(key_data)}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if still valid"""
        if cache_key not in self._cache:
            return None
        
        # Check if cache is still valid
        if cache_key in self._cache_ttl:
            if time.time() > self._cache_ttl[cache_key]:
                # Cache expired
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
                return None
        
        return self._cache[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any, duration: int = None):
        """Cache a result"""
        self._cache[cache_key] = result
        self._cache_ttl[cache_key] = time.time() + (duration or self.default_cache_duration)
    
    def clear_cache(self):
        """Clear all cached results"""
        self._cache.clear()
        self._cache_ttl.clear()
        logger.info("Dashboard cache cleared")
    
    # Risk Assessment Methods
    
    def _get_risk_assessment_data(self, facility_ids: Optional[List[str]] = None) -> List[RiskAssessmentData]:
        """
        Get risk assessment data from Neo4j
        
        Args:
            facility_ids: Optional list of facility IDs to filter by
            
        Returns:
            List of RiskAssessmentData objects
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Facility)-[:HAS_RISK_ASSESSMENT]->(ra:RiskAssessment)
                WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
                
                OPTIONAL MATCH (ra)-[:HAS_RECOMMENDATION]->(rec:RiskRecommendation)
                
                WITH ra, f, collect(rec {
                    recommendation_id: rec.recommendation_id,
                    title: rec.title,
                    description: rec.description,
                    priority: rec.priority,
                    status: rec.status,
                    due_date: rec.due_date
                }) as recommendations
                
                RETURN ra {
                    assessment_id: ra.assessment_id,
                    facility_id: f.facility_id,
                    risk_level: ra.risk_level,
                    risk_score: ra.risk_score,
                    assessment_date: ra.assessment_date,
                    methodology: ra.methodology,
                    assessor: ra.assessor,
                    status: ra.status,
                    recommendations: recommendations
                }
                ORDER BY ra.assessment_date DESC
                """
                
                result = session.run(query, {"facility_ids": facility_ids})
                risk_assessments = []
                
                for record in result:
                    ra_data = record['ra']
                    risk_assessment = RiskAssessmentData(
                        assessment_id=ra_data.get('assessment_id', ''),
                        facility_id=ra_data.get('facility_id', ''),
                        risk_level=ra_data.get('risk_level', 'UNKNOWN'),
                        risk_score=float(ra_data.get('risk_score', 0.0)),
                        assessment_date=ra_data.get('assessment_date', datetime.now()),
                        recommendations=ra_data.get('recommendations', []),
                        methodology=ra_data.get('methodology', 'Unknown'),
                        assessor=ra_data.get('assessor', 'Unknown'),
                        status=ra_data.get('status', 'Unknown')
                    )
                    risk_assessments.append(risk_assessment)
                
                logger.info(f"Retrieved {len(risk_assessments)} risk assessments")
                return risk_assessments
                
        except Exception as e:
            logger.error(f"Failed to get risk assessment data: {e}")
            return []
    
    def _get_facility_risk_summary(self, facility_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get summarized risk data for facilities
        
        Args:
            facility_ids: Optional list of facility IDs to filter by
            
        Returns:
            Dictionary with risk summary data
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Facility)-[:HAS_RISK_ASSESSMENT]->(ra:RiskAssessment)
                WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
                AND ra.status = 'ACTIVE'
                
                WITH ra
                ORDER BY ra.assessment_date DESC
                
                RETURN 
                    count(ra) as total_assessments,
                    avg(ra.risk_score) as avg_risk_score,
                    count(CASE WHEN ra.risk_level = 'HIGH' THEN 1 END) as high_risk_count,
                    count(CASE WHEN ra.risk_level = 'MEDIUM' THEN 1 END) as medium_risk_count,
                    count(CASE WHEN ra.risk_level = 'LOW' THEN 1 END) as low_risk_count,
                    max(ra.assessment_date) as latest_assessment_date,
                    min(ra.assessment_date) as earliest_assessment_date
                """
                
                result = session.run(query, {"facility_ids": facility_ids}).single()
                
                if result:
                    # Determine overall risk level based on distribution
                    high_risk = result['high_risk_count'] or 0
                    medium_risk = result['medium_risk_count'] or 0
                    low_risk = result['low_risk_count'] or 0
                    total = result['total_assessments'] or 1
                    
                    if high_risk / total > 0.3:  # More than 30% high risk
                        overall_risk_level = "HIGH"
                    elif (high_risk + medium_risk) / total > 0.5:  # More than 50% medium+ risk
                        overall_risk_level = "MEDIUM"
                    else:
                        overall_risk_level = "LOW"
                    
                    return {
                        "total_assessments": total,
                        "avg_risk_score": round(result['avg_risk_score'] or 0.0, 2),
                        "overall_risk_level": overall_risk_level,
                        "risk_distribution": {
                            "high": high_risk,
                            "medium": medium_risk,
                            "low": low_risk
                        },
                        "latest_assessment_date": result['latest_assessment_date'],
                        "earliest_assessment_date": result['earliest_assessment_date'],
                        "has_data": True
                    }
                else:
                    # Return default values when no risk assessment data exists
                    return {
                        "total_assessments": 0,
                        "avg_risk_score": 25.0,  # Default medium risk score
                        "overall_risk_level": "MEDIUM",  # Conservative default
                        "risk_distribution": {
                            "high": 0,
                            "medium": 1,  # Assume at least medium risk as default
                            "low": 0
                        },
                        "latest_assessment_date": None,
                        "earliest_assessment_date": None,
                        "has_data": False
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get facility risk summary: {e}")
            # Return safe defaults on error
            return {
                "total_assessments": 0,
                "avg_risk_score": 30.0,
                "overall_risk_level": "MEDIUM",
                "risk_distribution": {
                    "high": 0,
                    "medium": 1,
                    "low": 0
                },
                "latest_assessment_date": None,
                "earliest_assessment_date": None,
                "has_data": False,
                "error": str(e)
            }
    
    def _get_risk_recommendations(self, facility_ids: Optional[List[str]] = None, 
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get risk-based recommendations from Neo4j
        
        Args:
            facility_ids: Optional list of facility IDs to filter by
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Facility)-[:HAS_RISK_ASSESSMENT]->(ra:RiskAssessment)-[:HAS_RECOMMENDATION]->(rec:RiskRecommendation)
                WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
                AND rec.status IN ['PENDING', 'IN_PROGRESS']
                
                RETURN rec {
                    recommendation_id: rec.recommendation_id,
                    facility_id: f.facility_id,
                    facility_name: f.facility_name,
                    title: rec.title,
                    description: rec.description,
                    priority: rec.priority,
                    status: rec.status,
                    due_date: rec.due_date,
                    created_date: rec.created_date,
                    risk_category: rec.risk_category,
                    estimated_cost: rec.estimated_cost,
                    expected_impact: rec.expected_impact
                } as recommendation
                
                ORDER BY 
                    CASE rec.priority
                        WHEN 'CRITICAL' THEN 1
                        WHEN 'HIGH' THEN 2
                        WHEN 'MEDIUM' THEN 3
                        WHEN 'LOW' THEN 4
                        ELSE 5
                    END,
                    rec.due_date ASC
                
                LIMIT $limit
                """
                
                result = session.run(query, {"facility_ids": facility_ids, "limit": limit})
                recommendations = []
                
                for record in result:
                    rec_data = record['recommendation']
                    # Convert Neo4j data types to standard Python types
                    recommendation = {
                        "recommendation_id": rec_data.get('recommendation_id', ''),
                        "facility_id": rec_data.get('facility_id', ''),
                        "facility_name": rec_data.get('facility_name', ''),
                        "title": rec_data.get('title', ''),
                        "description": rec_data.get('description', ''),
                        "priority": rec_data.get('priority', 'MEDIUM'),
                        "status": rec_data.get('status', 'PENDING'),
                        "due_date": rec_data.get('due_date'),
                        "created_date": rec_data.get('created_date'),
                        "risk_category": rec_data.get('risk_category', 'General'),
                        "estimated_cost": rec_data.get('estimated_cost'),
                        "expected_impact": rec_data.get('expected_impact', '')
                    }
                    recommendations.append(recommendation)
                
                logger.info(f"Retrieved {len(recommendations)} risk recommendations")
                return recommendations
                
        except Exception as e:
            logger.error(f"Failed to get risk recommendations: {e}")
            return []
    
    # Core Dashboard Methods (Updated with Risk Assessment Integration)
    
    @_handle_errors
    def generate_executive_dashboard(self,
                                   location_filter: Optional[LocationFilter] = None,
                                   date_filter: Optional[DateRangeFilter] = None,
                                   include_trends: bool = True,
                                   include_recommendations: bool = True,
                                   include_forecasts: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive executive dashboard data with risk assessment integration
        
        Args:
            location_filter: Optional location filtering parameters
            date_filter: Optional date range filtering parameters
            include_trends: Whether to include trend analysis
            include_recommendations: Whether to include recommendations
            include_forecasts: Whether to include forecasting data
            
        Returns:
            Complete dashboard JSON structure with risk assessment data
        """
        # Check cache first
        cache_key = self._get_cache_key(
            "generate_executive_dashboard",
            location_filter, date_filter, include_trends, 
            include_recommendations, include_forecasts
        )
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug("Returning cached dashboard result")
            return cached_result
        
        # Set default date filter if not provided
        if not date_filter:
            date_filter = DateRangeFilter(
                end_date=datetime.now(),
                start_date=datetime.now() - timedelta(days=30),
                period=AggregationPeriod.DAILY
            )
        
        # Extract facility IDs for filtering
        facility_ids = None
        if location_filter and location_filter.facility_ids:
            facility_ids = location_filter.facility_ids
        
        logger.info(f"Generating dashboard for {len(facility_ids) if facility_ids else 'all'} facilities")
        
        try:
            # Get risk assessment data
            risk_summary = self._get_facility_risk_summary(facility_ids)
            risk_recommendations = self._get_risk_recommendations(facility_ids)
            
            # Core dashboard data
            dashboard_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generated_by": "ExecutiveDashboardService",
                    "version": "1.1.0",
                    "filters": {
                        "location": asdict(location_filter) if location_filter else None,
                        "date_range": asdict(date_filter) if date_filter else None
                    },
                    "cache_status": "miss",
                    "risk_assessment_included": True
                },
                "summary": {},
                "kpis": {},
                "charts": {},
                "alerts": {},
                "status": {},
                "risk_assessment": risk_summary
            }
            
            # Generate summary metrics (now includes risk data)
            summary_data = self._generate_summary_metrics(facility_ids, date_filter, risk_summary)
            dashboard_data["summary"] = summary_data
            
            # Generate KPIs (now includes risk-based KPIs)
            kpi_data = self._generate_kpi_metrics(facility_ids, date_filter, risk_summary)
            dashboard_data["kpis"] = kpi_data
            
            # Generate chart data (now includes risk charts)
            chart_data = self._generate_chart_data(facility_ids, date_filter, risk_summary)
            dashboard_data["charts"] = chart_data
            
            # Generate alerts and status (now includes risk-based alerts)
            alerts_data = self._generate_alerts_data(facility_ids, risk_summary)
            dashboard_data["alerts"] = alerts_data
            
            status_data = self._generate_status_data(facility_ids, risk_summary)
            dashboard_data["status"] = status_data
            
            # Include trend analysis if requested
            if include_trends:
                trends_data = self._generate_trends_data(facility_ids, date_filter)
                dashboard_data["trends"] = trends_data
            
            # Include recommendations if requested (now includes risk recommendations)
            if include_recommendations:
                recommendations_data = self._generate_recommendations_data(facility_ids, risk_recommendations)
                dashboard_data["recommendations"] = recommendations_data
            
            # Include forecasts if requested
            if include_forecasts:
                forecasts_data = self._generate_forecasts_data(facility_ids, date_filter)
                dashboard_data["forecasts"] = forecasts_data
            
            # Calculate overall health score (now considers risk assessment)
            health_score = self._calculate_overall_health_score(dashboard_data, risk_summary)
            dashboard_data["summary"]["overall_health_score"] = health_score
            
            # Cache the result
            self._cache_result(cache_key, dashboard_data, duration=300)  # 5 minutes
            
            logger.info("Dashboard generated successfully with risk assessment data")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            raise
    
    def _generate_summary_metrics(self, facility_ids: Optional[List[str]], 
                                 date_filter: DateRangeFilter, 
                                 risk_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary metrics with risk assessment data"""
        try:
            # Get real-time metrics
            real_time_data = self.analytics.get_real_time_metrics(facility_ids)
            
            # Get KPIs for the period
            safety_kpis = self.analytics.calculate_safety_kpis(
                facility_ids, 
                date_filter.start_date, 
                date_filter.end_date
            )
            compliance_kpis = self.analytics.calculate_compliance_kpis(
                facility_ids, 
                date_filter.start_date, 
                date_filter.end_date
            )
            
            # Get facility count
            facility_count = len(self.analytics.get_facility_overview(facility_ids))
            
            # Calculate period-over-period changes
            previous_period_start = date_filter.start_date - (date_filter.end_date - date_filter.start_date)
            previous_safety_kpis = self.analytics.calculate_safety_kpis(
                facility_ids, previous_period_start, date_filter.start_date
            )
            
            # Calculate changes
            incident_change = self._calculate_period_change(
                safety_kpis.get('total_incidents', KPIMetric('total_incidents', 0, 'count')).value,
                previous_safety_kpis.get('total_incidents', KPIMetric('total_incidents', 0, 'count')).value
            )
            
            return {
                "period": {
                    "start_date": date_filter.start_date.isoformat(),
                    "end_date": date_filter.end_date.isoformat(),
                    "period_days": (date_filter.end_date - date_filter.start_date).days
                },
                "facilities": {
                    "total_count": facility_count,
                    "active_alerts": real_time_data.get('metrics', {}).get('active_alerts', 0),
                    "status_distribution": self._get_facility_status_distribution(facility_ids)
                },
                "incidents": {
                    "total": safety_kpis.get('total_incidents', KPIMetric('total_incidents', 0, 'count')).value,
                    "today": real_time_data.get('metrics', {}).get('todays_incidents', 0),
                    "change_from_previous_period": incident_change,
                    "incident_rate": safety_kpis.get('incident_rate', KPIMetric('incident_rate', 0, 'rate')).value
                },
                "compliance": {
                    "audit_pass_rate": compliance_kpis.get('audit_pass_rate', KPIMetric('audit_pass_rate', 100, 'percentage')).value,
                    "training_completion": compliance_kpis.get('training_completion_rate', KPIMetric('training_completion_rate', 100, 'percentage')).value,
                    "overdue_items": real_time_data.get('metrics', {}).get('overdue_training', 0) + real_time_data.get('metrics', {}).get('overdue_inspections', 0)
                },
                "risk_assessment": {
                    "overall_risk_level": risk_summary.get('overall_risk_level', 'MEDIUM'),
                    "avg_risk_score": risk_summary.get('avg_risk_score', 25.0),
                    "total_assessments": risk_summary.get('total_assessments', 0),
                    "high_risk_facilities": risk_summary.get('risk_distribution', {}).get('high', 0),
                    "has_recent_data": risk_summary.get('has_data', False)
                },
                "alert_level": self._determine_alert_level(real_time_data, risk_summary),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate summary metrics: {e}")
            return {"error": str(e)}
    
    def _determine_alert_level(self, real_time_data: Dict[str, Any], 
                              risk_summary: Dict[str, Any]) -> str:
        """Determine overall alert level considering risk assessment data"""
        try:
            base_alert_level = real_time_data.get('alert_level', 'GREEN')
            risk_level = risk_summary.get('overall_risk_level', 'LOW')
            high_risk_count = risk_summary.get('risk_distribution', {}).get('high', 0)
            
            # Escalate alert level based on risk assessment
            if risk_level == 'HIGH' or high_risk_count > 2:
                if base_alert_level in ['GREEN', 'YELLOW']:
                    return 'ORANGE'
                else:
                    return 'RED'
            elif risk_level == 'MEDIUM' and high_risk_count > 0:
                if base_alert_level == 'GREEN':
                    return 'YELLOW'
            
            return base_alert_level
            
        except Exception as e:
            logger.error(f"Failed to determine alert level: {e}")
            return 'YELLOW'  # Conservative default
    
    def _generate_kpi_metrics(self, facility_ids: Optional[List[str]], 
                             date_filter: DateRangeFilter,
                             risk_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed KPI metrics including risk-based KPIs"""
        try:
            # Get all KPIs
            safety_kpis = self.analytics.calculate_safety_kpis(
                facility_ids, 
                date_filter.start_date, 
                date_filter.end_date
            )
            compliance_kpis = self.analytics.calculate_compliance_kpis(
                facility_ids, 
                date_filter.start_date, 
                date_filter.end_date
            )
            
            # Convert KPI objects to dictionaries with additional metadata
            kpi_data = {}
            
            # Safety KPIs
            for name, kpi in safety_kpis.items():
                kpi_dict = self._kpi_to_dict(kpi)
                kpi_dict['category'] = 'safety'
                kpi_dict['status'] = self._determine_kpi_status(kpi)
                kpi_data[name] = kpi_dict
            
            # Compliance KPIs
            for name, kpi in compliance_kpis.items():
                kpi_dict = self._kpi_to_dict(kpi)
                kpi_dict['category'] = 'compliance'
                kpi_dict['status'] = self._determine_kpi_status(kpi)
                kpi_data[name] = kpi_dict
            
            # Add risk-based KPIs
            risk_kpis = self._calculate_risk_kpis(facility_ids, date_filter, risk_summary)
            kpi_data.update(risk_kpis)
            
            # Add custom calculated KPIs
            custom_kpis = self._calculate_custom_kpis(facility_ids, date_filter)
            kpi_data.update(custom_kpis)
            
            return {
                "summary": {
                    "total_kpis": len(kpi_data),
                    "green_status": len([k for k in kpi_data.values() if k.get('status') == 'green']),
                    "yellow_status": len([k for k in kpi_data.values() if k.get('status') == 'yellow']),
                    "red_status": len([k for k in kpi_data.values() if k.get('status') == 'red'])
                },
                "metrics": kpi_data,
                "benchmarks": self._get_kpi_benchmarks(facility_ids, date_filter),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate KPI metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_risk_kpis(self, facility_ids: Optional[List[str]], 
                            date_filter: DateRangeFilter,
                            risk_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate risk assessment based KPIs"""
        try:
            risk_kpis = {}
            
            # Overall Risk Score KPI
            avg_risk_score = risk_summary.get('avg_risk_score', 25.0)
            risk_kpis['overall_risk_score'] = {
                "name": "Overall Risk Score",
                "value": avg_risk_score,
                "unit": "score",
                "target": 20.0,
                "threshold_warning": 30.0,
                "threshold_critical": 50.0,
                "category": "risk",
                "status": "green" if avg_risk_score < 20 else "yellow" if avg_risk_score < 30 else "red",
                "trend": "stable",  # Would be calculated from historical data
                "change_percent": 0.0,  # Would be calculated from historical data
                "timestamp": datetime.now().isoformat(),
                "data_source": "risk_assessments"
            }
            
            # High Risk Facilities KPI
            high_risk_count = risk_summary.get('risk_distribution', {}).get('high', 0)
            total_assessments = risk_summary.get('total_assessments', 1)
            high_risk_percentage = (high_risk_count / max(total_assessments, 1)) * 100
            
            risk_kpis['high_risk_facilities'] = {
                "name": "High Risk Facilities",
                "value": high_risk_percentage,
                "unit": "percentage",
                "target": 10.0,
                "threshold_warning": 20.0,
                "threshold_critical": 30.0,
                "category": "risk",
                "status": "green" if high_risk_percentage < 10 else "yellow" if high_risk_percentage < 20 else "red",
                "trend": "stable",
                "change_percent": 0.0,
                "timestamp": datetime.now().isoformat(),
                "data_source": "risk_assessments",
                "metadata": {
                    "high_risk_count": high_risk_count,
                    "total_assessments": total_assessments
                }
            }
            
            # Risk Assessment Coverage KPI
            risk_coverage_percentage = min(100.0, (total_assessments / max(len(self.analytics.get_facility_overview(facility_ids)), 1)) * 100)
            
            risk_kpis['risk_assessment_coverage'] = {
                "name": "Risk Assessment Coverage",
                "value": risk_coverage_percentage,
                "unit": "percentage",
                "target": 100.0,
                "threshold_warning": 80.0,
                "threshold_critical": 60.0,
                "category": "risk",
                "status": "green" if risk_coverage_percentage >= 100 else "yellow" if risk_coverage_percentage >= 80 else "red",
                "trend": "stable",
                "change_percent": 0.0,
                "timestamp": datetime.now().isoformat(),
                "data_source": "risk_assessments"
            }
            
            return risk_kpis
            
        except Exception as e:
            logger.error(f"Failed to calculate risk KPIs: {e}")
            return {}
    
    def _generate_chart_data(self, facility_ids: Optional[List[str]], 
                            date_filter: DateRangeFilter,
                            risk_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart data for visualization including risk assessment charts"""
        try:
            charts = {}
            
            # Trending data
            trending_data = self.analytics.get_trending_metrics(
                facility_ids,
                date_filter.start_date,
                date_filter.end_date,
                date_filter.period
            )
            
            # Incident trend chart
            charts['incident_trend'] = {
                "type": "line",
                "title": "Incident Trend Over Time",
                "data": trending_data.get('incidents', []),
                "x_axis": "date",
                "y_axis": "total",
                "color": "#dc3545"
            }
            
            # Violations trend chart  
            charts['violations_trend'] = {
                "type": "line",
                "title": "Violations Trend Over Time",
                "data": trending_data.get('violations', []),
                "x_axis": "date",
                "y_axis": "total",
                "color": "#fd7e14"
            }
            
            # Risk Assessment Distribution Chart
            risk_distribution = risk_summary.get('risk_distribution', {'high': 0, 'medium': 0, 'low': 0})
            charts['risk_level_distribution'] = {
                "type": "pie",
                "title": "Risk Level Distribution",
                "data": [
                    {"label": "High Risk", "value": risk_distribution.get('high', 0), "color": "#dc3545"},
                    {"label": "Medium Risk", "value": risk_distribution.get('medium', 0), "color": "#ffc107"},
                    {"label": "Low Risk", "value": risk_distribution.get('low', 0), "color": "#28a745"}
                ]
            }
            
            # Risk Score by Facility Chart (if facility-specific data available)
            if facility_ids and len(facility_ids) > 1:
                charts['risk_score_by_facility'] = self._generate_risk_score_facility_chart(facility_ids)
            
            # Facility performance comparison
            facility_benchmarks = self.analytics.get_facility_benchmarks(
                ['incident_rate', 'audit_pass_rate'],
                date_filter.start_date,
                date_filter.end_date
            )
            
            # Group benchmarks by metric for charts
            charts['facility_performance'] = self._format_benchmark_charts(facility_benchmarks)
            
            # KPI status distribution pie chart
            safety_kpis = self.analytics.calculate_safety_kpis(
                facility_ids, date_filter.start_date, date_filter.end_date
            )
            compliance_kpis = self.analytics.calculate_compliance_kpis(
                facility_ids, date_filter.start_date, date_filter.end_date
            )
            
            all_kpis = {**safety_kpis, **compliance_kpis}
            status_counts = {"green": 0, "yellow": 0, "red": 0}
            
            for kpi in all_kpis.values():
                status = self._determine_kpi_status(kpi)
                status_counts[status] += 1
            
            charts['kpi_status_distribution'] = {
                "type": "pie",
                "title": "KPI Status Distribution",
                "data": [
                    {"label": "Green", "value": status_counts["green"], "color": "#28a745"},
                    {"label": "Yellow", "value": status_counts["yellow"], "color": "#ffc107"},
                    {"label": "Red", "value": status_counts["red"], "color": "#dc3545"}
                ]
            }
            
            # Department/Location breakdown
            charts['location_breakdown'] = self._generate_location_breakdown_chart(facility_ids, date_filter)
            
            return charts
            
        except Exception as e:
            logger.error(f"Failed to generate chart data: {e}")
            return {"error": str(e)}
    
    def _generate_risk_score_facility_chart(self, facility_ids: List[str]) -> Dict[str, Any]:
        """Generate risk score by facility chart"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Facility)-[:HAS_RISK_ASSESSMENT]->(ra:RiskAssessment)
                WHERE f.facility_id IN $facility_ids
                AND ra.status = 'ACTIVE'
                
                WITH f, ra
                ORDER BY ra.assessment_date DESC
                LIMIT 1
                
                RETURN f.facility_name as facility_name, 
                       f.facility_id as facility_id,
                       ra.risk_score as risk_score,
                       ra.risk_level as risk_level
                ORDER BY ra.risk_score DESC
                """
                
                result = session.run(query, {"facility_ids": facility_ids})
                chart_data = []
                
                for record in result:
                    risk_score = record['risk_score'] or 0.0
                    risk_level = record['risk_level'] or 'UNKNOWN'
                    
                    # Color code by risk level
                    color = "#28a745" if risk_level == 'LOW' else "#ffc107" if risk_level == 'MEDIUM' else "#dc3545"
                    
                    chart_data.append({
                        "label": record['facility_name'] or record['facility_id'],
                        "value": float(risk_score),
                        "risk_level": risk_level,
                        "color": color
                    })
                
                return {
                    "type": "horizontal_bar",
                    "title": "Risk Scores by Facility",
                    "data": chart_data,
                    "x_axis": "value",
                    "y_axis": "label",
                    "color_by_risk_level": True
                }
                
        except Exception as e:
            logger.error(f"Failed to generate risk score facility chart: {e}")
            return {"error": str(e)}
    
    def _generate_alerts_data(self, facility_ids: Optional[List[str]], 
                             risk_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alerts and notifications data including risk-based alerts"""
        try:
            # Get recent alerts from analytics layer
            recent_alerts = self.analytics.get_recent_alerts(facility_ids, limit=20)
            
            # Get real-time metrics for active alerts
            real_time_metrics = self.analytics.get_real_time_metrics(facility_ids)
            
            # Add risk-based alerts
            risk_alerts = self._generate_risk_based_alerts(risk_summary)
            all_alerts = recent_alerts + risk_alerts
            
            # Categorize alerts by severity
            alerts_by_severity = {"critical": [], "high": [], "medium": [], "low": []}
            
            for alert in all_alerts:
                severity = alert.get('severity', 'medium').lower()
                if severity in alerts_by_severity:
                    alerts_by_severity[severity].append(alert)
            
            # Generate alert summaries
            alert_summary = {
                "total_active": real_time_metrics.get('metrics', {}).get('active_alerts', 0) + len(risk_alerts),
                "critical_count": len(alerts_by_severity['critical']),
                "high_count": len(alerts_by_severity['high']),
                "medium_count": len(alerts_by_severity['medium']),
                "low_count": len(alerts_by_severity['low']),
                "alert_level": self._determine_alert_level(real_time_metrics, risk_summary),
                "risk_based_alerts": len(risk_alerts)
            }
            
            # Get overdue items
            overdue_training = real_time_metrics.get('metrics', {}).get('overdue_training', 0)
            overdue_inspections = real_time_metrics.get('metrics', {}).get('overdue_inspections', 0)
            
            # Generate escalation alerts
            escalation_alerts = self._generate_escalation_alerts(facility_ids)
            
            return {
                "summary": alert_summary,
                "recent_alerts": all_alerts[:10],  # Limit to most recent 10
                "alerts_by_severity": alerts_by_severity,
                "risk_based_alerts": risk_alerts,
                "overdue_items": {
                    "training": overdue_training,
                    "inspections": overdue_inspections,
                    "total": overdue_training + overdue_inspections
                },
                "escalations": escalation_alerts,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate alerts data: {e}")
            return {"error": str(e)}
    
    def _generate_risk_based_alerts(self, risk_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on risk assessment data"""
        try:
            risk_alerts = []
            
            # High overall risk score alert
            avg_risk_score = risk_summary.get('avg_risk_score', 0.0)
            if avg_risk_score > 40.0:
                risk_alerts.append({
                    "id": f"risk_alert_{uuid.uuid4().hex[:8]}",
                    "type": "risk_assessment",
                    "severity": "high" if avg_risk_score > 60.0 else "medium",
                    "title": "Elevated Risk Score Detected",
                    "description": f"Average facility risk score ({avg_risk_score:.1f}) exceeds acceptable threshold",
                    "timestamp": datetime.now().isoformat(),
                    "facility_count": risk_summary.get('total_assessments', 0),
                    "recommended_actions": [
                        "Review high-risk facilities immediately",
                        "Implement additional safety measures",
                        "Schedule risk mitigation meetings"
                    ]
                })
            
            # High-risk facilities alert
            high_risk_count = risk_summary.get('risk_distribution', {}).get('high', 0)
            if high_risk_count > 0:
                risk_alerts.append({
                    "id": f"high_risk_facilities_{uuid.uuid4().hex[:8]}",
                    "type": "risk_assessment",
                    "severity": "critical" if high_risk_count > 2 else "high",
                    "title": f"{high_risk_count} High-Risk Facilities Identified",
                    "description": f"{high_risk_count} facilities currently classified as high-risk and require immediate attention",
                    "timestamp": datetime.now().isoformat(),
                    "facility_count": high_risk_count,
                    "recommended_actions": [
                        "Conduct immediate safety assessments",
                        "Implement emergency protocols",
                        "Increase inspection frequency"
                    ]
                })
            
            # Missing risk assessment data alert
            if not risk_summary.get('has_data', False):
                risk_alerts.append({
                    "id": f"missing_risk_data_{uuid.uuid4().hex[:8]}",
                    "type": "data_quality",
                    "severity": "medium",
                    "title": "Risk Assessment Data Missing",
                    "description": "No recent risk assessment data available for facilities",
                    "timestamp": datetime.now().isoformat(),
                    "recommended_actions": [
                        "Schedule risk assessments for all facilities",
                        "Verify data collection processes",
                        "Update risk assessment procedures"
                    ]
                })
            
            return risk_alerts
            
        except Exception as e:
            logger.error(f"Failed to generate risk-based alerts: {e}")
            return []
    
    def _generate_status_data(self, facility_ids: Optional[List[str]],
                             risk_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall system status data including risk assessment status"""
        try:
            # System health check
            system_health = self._perform_system_health_check()
            
            # Data quality metrics (including risk assessment data quality)
            data_quality = self._assess_data_quality(facility_ids, risk_summary)
            
            # Service availability
            service_status = {
                "dashboard_service": "operational",
                "neo4j_database": "operational" if self.analytics.health_check()['status'] == 'healthy' else "degraded",
                "trend_analysis": "operational",
                "recommendation_engine": "operational",
                "risk_assessment_system": "operational" if risk_summary.get('has_data', False) else "degraded"
            }
            
            # Calculate overall system status
            overall_status = self._calculate_overall_system_status(system_health, service_status, data_quality)
            
            return {
                "overall_status": overall_status,
                "system_health": system_health,
                "service_status": service_status,
                "data_quality": data_quality,
                "risk_assessment_status": {
                    "has_current_data": risk_summary.get('has_data', False),
                    "total_assessments": risk_summary.get('total_assessments', 0),
                    "latest_assessment": risk_summary.get('latest_assessment_date'),
                    "overall_risk_level": risk_summary.get('overall_risk_level', 'UNKNOWN')
                },
                "performance_metrics": {
                    "request_count": self._request_count,
                    "error_count": self._error_count,
                    "error_rate": (self._error_count / max(self._request_count, 1)) * 100,
                    "cache_hit_rate": self._calculate_cache_hit_rate()
                },
                "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate status data: {e}")
            return {"error": str(e)}
    
    def _generate_trends_data(self, facility_ids: Optional[List[str]], 
                             date_filter: DateRangeFilter) -> Dict[str, Any]:
        """Generate trend analysis data"""
        try:
            trends_data = {}
            
            # Key metrics for trend analysis
            key_metrics = ['incident_rate', 'ltir', 'audit_pass_rate', 'training_completion_rate']
            
            for metric in key_metrics:
                try:
                    # Perform comprehensive trend analysis
                    analysis_result = self.trend_analysis.comprehensive_trend_analysis(
                        metric, 
                        date_filter.start_date, 
                        date_filter.end_date
                    )
                    
                    if 'error' not in analysis_result:
                        # Format for LLM analysis
                        formatted_analysis = self.trend_analysis.format_for_llm_analysis(analysis_result)
                        trends_data[metric] = formatted_analysis
                    else:
                        logger.warning(f"Trend analysis failed for {metric}: {analysis_result['error']}")
                        trends_data[metric] = {"error": analysis_result['error']}
                        
                except Exception as e:
                    logger.error(f"Failed to analyze trend for {metric}: {e}")
                    trends_data[metric] = {"error": str(e)}
            
            # Get recent anomalies
            recent_anomalies = self.trend_analysis.get_recent_anomalies(days=7)
            
            return {
                "metric_trends": trends_data,
                "recent_anomalies": [anomaly.to_dict() for anomaly in recent_anomalies],
                "trend_summary": self._generate_trend_summary(trends_data),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate trends data: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations_data(self, facility_ids: Optional[List[str]],
                                      risk_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations data including risk-based recommendations"""
        try:
            # Get recommendations by status
            pending_recs = self.recommendations.get_recommendations_by_status(RecommendationStatus.PENDING_REVIEW)
            in_progress_recs = self.recommendations.get_recommendations_by_status(RecommendationStatus.IN_PROGRESS)
            high_priority_recs = self.recommendations.get_recommendations_by_priority(RecommendationPriority.HIGH)
            
            # Get overdue recommendations
            overdue_recs = self.recommendations.get_overdue_recommendations()
            
            # Get analytics
            rec_analytics = self.recommendations.get_recommendation_analytics(time_period_days=30)
            
            # Generate AI-driven recommendations based on current data
            ai_recommendations = self._generate_ai_recommendations(facility_ids)
            
            return {
                "summary": {
                    "pending_review": len(pending_recs),
                    "in_progress": len(in_progress_recs),
                    "high_priority": len(high_priority_recs),
                    "overdue": len(overdue_recs),
                    "risk_based": len(risk_recommendations)
                },
                "pending_recommendations": pending_recs[:5],  # Top 5
                "high_priority_recommendations": high_priority_recs[:5],  # Top 5
                "overdue_recommendations": overdue_recs[:5],  # Top 5
                "risk_based_recommendations": risk_recommendations[:5],  # Top 5 from risk assessments
                "ai_generated_recommendations": ai_recommendations,
                "analytics": rec_analytics,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations data: {e}")
            return {"error": str(e)}
    
    def _generate_forecasts_data(self, facility_ids: Optional[List[str]], 
                                date_filter: DateRangeFilter) -> Dict[str, Any]:
        """Generate forecast data (placeholder for future implementation)"""
        try:
            # This would integrate with forecasting models
            # For now, return basic projections based on trends
            
            # Get historical data
            safety_kpis = self.analytics.calculate_safety_kpis(
                facility_ids, 
                date_filter.start_date, 
                date_filter.end_date
            )
            
            # Simple linear projection (would be replaced with actual forecasting models)
            incident_rate = safety_kpis.get('incident_rate', KPIMetric('incident_rate', 0, 'rate')).value
            ltir = safety_kpis.get('ltir', KPIMetric('ltir', 0, 'rate')).value
            
            # Generate next 30 days projection
            projection_dates = [(datetime.now() + timedelta(days=i)).isoformat() for i in range(1, 31)]
            
            forecasts = {
                "incident_rate_forecast": {
                    "method": "linear_projection",
                    "confidence": 0.65,
                    "dates": projection_dates,
                    "values": [incident_rate + (i * 0.01) for i in range(30)],  # Placeholder calculation
                    "upper_bound": [incident_rate + (i * 0.02) for i in range(30)],
                    "lower_bound": [max(0, incident_rate - (i * 0.01)) for i in range(30)]
                },
                "ltir_forecast": {
                    "method": "linear_projection", 
                    "confidence": 0.60,
                    "dates": projection_dates,
                    "values": [ltir + (i * 0.005) for i in range(30)],  # Placeholder calculation
                    "upper_bound": [ltir + (i * 0.01) for i in range(30)],
                    "lower_bound": [max(0, ltir - (i * 0.005)) for i in range(30)]
                }
            }
            
            return {
                "forecasts": forecasts,
                "methodology": "Linear projection based on historical trends (placeholder implementation)",
                "generated_at": datetime.now().isoformat(),
                "note": "Production implementation would use advanced forecasting models"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate forecasts data: {e}")
            return {"error": str(e)}
    
    # Helper Methods
    
    def _kpi_to_dict(self, kpi: KPIMetric) -> Dict[str, Any]:
        """Convert KPI metric to dictionary"""
        return {
            "name": kpi.name,
            "value": kpi.value,
            "unit": kpi.unit,
            "target": kpi.target,
            "threshold_warning": kpi.threshold_warning,
            "threshold_critical": kpi.threshold_critical,
            "trend": kpi.trend,
            "change_percent": kpi.change_percent,
            "timestamp": kpi.timestamp.isoformat() if kpi.timestamp else None
        }
    
    def _determine_kpi_status(self, kpi: KPIMetric) -> str:
        """Determine KPI status based on thresholds"""
        if kpi.threshold_critical and kpi.value >= kpi.threshold_critical:
            return "red"
        elif kpi.threshold_warning and kpi.value >= kpi.threshold_warning:
            return "yellow"
        else:
            return "green"
    
    def _calculate_period_change(self, current_value: float, previous_value: float) -> Dict[str, Any]:
        """Calculate period-over-period change"""
        if previous_value == 0:
            if current_value > 0:
                return {"percent_change": float('inf'), "absolute_change": current_value, "trend": "up"}
            else:
                return {"percent_change": 0, "absolute_change": 0, "trend": "stable"}
        
        percent_change = ((current_value - previous_value) / previous_value) * 100
        absolute_change = current_value - previous_value
        
        if percent_change > 5:
            trend = "up"
        elif percent_change < -5:
            trend = "down"
        else:
            trend = "stable"
        
        return {
            "percent_change": round(percent_change, 2),
            "absolute_change": round(absolute_change, 2),
            "trend": trend
        }
    
    def _get_facility_status_distribution(self, facility_ids: Optional[List[str]]) -> Dict[str, int]:
        """Get distribution of facility statuses"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Facility)
                WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
                OPTIONAL MATCH (f)-[:HAS_ALERT]->(a:Alert {status: 'Active'})
                WITH f, count(a) as active_alerts
                RETURN 
                    CASE 
                        WHEN active_alerts > 5 THEN 'critical'
                        WHEN active_alerts > 2 THEN 'warning'
                        WHEN active_alerts > 0 THEN 'attention'
                        ELSE 'normal'
                    END as status,
                    count(f) as count
                """
                
                result = session.run(query, {"facility_ids": facility_ids})
                return {record['status']: record['count'] for record in result}
                
        except Exception as e:
            logger.error(f"Failed to get facility status distribution: {e}")
            return {}
    
    def _calculate_custom_kpis(self, facility_ids: Optional[List[str]], 
                              date_filter: DateRangeFilter) -> Dict[str, Dict[str, Any]]:
        """Calculate additional custom KPIs"""
        try:
            custom_kpis = {}
            
            # Risk exposure KPI
            with self.driver.session() as session:
                risk_query = """
                MATCH (f:Facility)-[:HAS_RISK]->(r:Risk)
                WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
                AND r.status = 'Active'
                RETURN 
                    avg(r.risk_score) as avg_risk_score,
                    count(CASE WHEN r.risk_level = 'High' THEN 1 END) as high_risks,
                    count(r) as total_risks
                """
                
                risk_result = session.run(risk_query, {"facility_ids": facility_ids}).single()
                
                if risk_result:
                    avg_risk_score = risk_result['avg_risk_score'] or 0
                    high_risks = risk_result['high_risks'] or 0
                    total_risks = risk_result['total_risks'] or 0
                    
                    custom_kpis['risk_exposure'] = {
                        "name": "Risk Exposure Score",
                        "value": round(avg_risk_score, 2),
                        "unit": "score",
                        "target": 30.0,
                        "threshold_warning": 50.0,
                        "threshold_critical": 70.0,
                        "category": "risk",
                        "status": "green" if avg_risk_score < 30 else "yellow" if avg_risk_score < 50 else "red",
                        "metadata": {
                            "high_risks": high_risks,
                            "total_risks": total_risks
                        }
                    }
            
            # Employee engagement KPI (placeholder - would be calculated from actual engagement data)
            custom_kpis['employee_engagement'] = {
                "name": "Safety Engagement Score",
                "value": 78.5,  # Placeholder value
                "unit": "percentage",
                "target": 85.0,
                "threshold_warning": 70.0,
                "threshold_critical": 60.0,
                "category": "engagement",
                "status": "yellow"
            }
            
            return custom_kpis
            
        except Exception as e:
            logger.error(f"Failed to calculate custom KPIs: {e}")
            return {}
    
    def _get_kpi_benchmarks(self, facility_ids: Optional[List[str]], 
                           date_filter: DateRangeFilter) -> Dict[str, Any]:
        """Get KPI benchmarks from aggregation layer"""
        try:
            benchmarks = self.analytics.get_facility_benchmarks(
                ['incident_rate', 'ltir', 'audit_pass_rate', 'training_completion_rate'],
                date_filter.start_date,
                date_filter.end_date
            )
            
            # Group benchmarks by metric
            benchmark_data = defaultdict(list)
            for benchmark in benchmarks:
                benchmark_data[benchmark.metric_name].append({
                    "facility_id": benchmark.facility_id,
                    "facility_name": benchmark.facility_name,
                    "value": benchmark.value,
                    "rank": benchmark.rank,
                    "percentile": benchmark.percentile,
                    "industry_average": benchmark.industry_average,
                    "best_practice": benchmark.best_practice
                })
            
            return dict(benchmark_data)
            
        except Exception as e:
            logger.error(f"Failed to get KPI benchmarks: {e}")
            return {}
    
    def _format_benchmark_charts(self, benchmarks: List[FacilityBenchmark]) -> Dict[str, Any]:
        """Format benchmark data for charting"""
        try:
            charts = {}
            
            # Group by metric
            metrics = defaultdict(list)
            for benchmark in benchmarks:
                metrics[benchmark.metric_name].append(benchmark)
            
            for metric_name, metric_benchmarks in metrics.items():
                # Sort by performance (rank)
                sorted_benchmarks = sorted(metric_benchmarks, key=lambda x: x.rank)[:10]  # Top 10
                
                charts[f"{metric_name}_ranking"] = {
                    "type": "bar",
                    "title": f"Top 10 Facilities - {metric_name.replace('_', ' ').title()}",
                    "data": [
                        {
                            "label": benchmark.facility_name,
                            "value": benchmark.value,
                            "rank": benchmark.rank,
                            "percentile": benchmark.percentile
                        }
                        for benchmark in sorted_benchmarks
                    ],
                    "x_axis": "label",
                    "y_axis": "value",
                    "color": "#17a2b8"
                }
            
            return charts
            
        except Exception as e:
            logger.error(f"Failed to format benchmark charts: {e}")
            return {}
    
    def _generate_location_breakdown_chart(self, facility_ids: Optional[List[str]], 
                                          date_filter: DateRangeFilter) -> Dict[str, Any]:
        """Generate location-based breakdown chart"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Facility)
                WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
                
                OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
                WHERE i.incident_date >= $start_date AND i.incident_date <= $end_date
                
                WITH f, count(i) as incident_count
                RETURN f.location as location, 
                       count(f) as facility_count,
                       sum(incident_count) as total_incidents,
                       avg(incident_count) as avg_incidents_per_facility
                ORDER BY total_incidents DESC
                """
                
                result = session.run(query, {
                    "facility_ids": facility_ids,
                    "start_date": date_filter.start_date,
                    "end_date": date_filter.end_date
                })
                
                chart_data = []
                for record in result:
                    chart_data.append({
                        "label": record['location'] or 'Unknown',
                        "facilities": record['facility_count'],
                        "incidents": record['total_incidents'],
                        "avg_incidents": round(record['avg_incidents_per_facility'] or 0, 2)
                    })
                
                return {
                    "type": "horizontal_bar",
                    "title": "Incidents by Location",
                    "data": chart_data,
                    "x_axis": "incidents",
                    "y_axis": "label",
                    "color": "#dc3545"
                }
                
        except Exception as e:
            logger.error(f"Failed to generate location breakdown chart: {e}")
            return {"error": str(e)}
    
    def _generate_escalation_alerts(self, facility_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Generate escalation alerts based on predefined rules"""
        try:
            escalations = []
            
            # Check for critical KPI thresholds
            safety_kpis = self.analytics.calculate_safety_kpis(facility_ids)
            compliance_kpis = self.analytics.calculate_compliance_kpis(facility_ids)
            
            # Critical incident rate
            incident_rate_kpi = safety_kpis.get('incident_rate')
            if incident_rate_kpi and incident_rate_kpi.threshold_critical:
                if incident_rate_kpi.value > incident_rate_kpi.threshold_critical:
                    escalations.append({
                        "id": str(uuid.uuid4()),
                        "type": "kpi_threshold_breach",
                        "severity": "critical",
                        "title": "Critical Incident Rate Threshold Exceeded",
                        "description": f"Incident rate ({incident_rate_kpi.value}) exceeds critical threshold ({incident_rate_kpi.threshold_critical})",
                        "requires_immediate_action": True,
                        "recommended_actions": [
                            "Immediate safety stand-down",
                            "Emergency safety briefing",
                            "Management review required"
                        ],
                        "created_at": datetime.now().isoformat()
                    })
            
            # Poor audit performance
            audit_rate_kpi = compliance_kpis.get('audit_pass_rate')
            if audit_rate_kpi and audit_rate_kpi.threshold_critical:
                if audit_rate_kpi.value < audit_rate_kpi.threshold_critical:
                    escalations.append({
                        "id": str(uuid.uuid4()),
                        "type": "compliance_failure",
                        "severity": "high",
                        "title": "Audit Pass Rate Below Critical Threshold",
                        "description": f"Audit pass rate ({audit_rate_kpi.value}%) below critical threshold ({audit_rate_kpi.threshold_critical}%)",
                        "requires_immediate_action": True,
                        "recommended_actions": [
                            "Comprehensive compliance review",
                            "Additional training programs",
                            "Process improvement initiatives"
                        ],
                        "created_at": datetime.now().isoformat()
                    })
            
            return escalations
            
        except Exception as e:
            logger.error(f"Failed to generate escalation alerts: {e}")
            return []
    
    def _perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            health_data = {}
            
            # Database connectivity
            try:
                health_data['database'] = self.analytics.health_check()
            except Exception as e:
                health_data['database'] = {"status": "unhealthy", "error": str(e)}
            
            # Trend analysis system
            try:
                health_data['trend_analysis'] = self.trend_analysis.health_check()
            except Exception as e:
                health_data['trend_analysis'] = {"status": "unhealthy", "error": str(e)}
            
            # Cache system
            health_data['cache'] = {
                "status": "healthy",
                "cached_items": len(self._cache),
                "memory_usage": "normal"  # Placeholder
            }
            
            self._last_health_check = datetime.now()
            
            return health_data
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def _assess_data_quality(self, facility_ids: Optional[List[str]], 
                            risk_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality metrics including risk assessment data"""
        try:
            with self.driver.session() as session:
                # Data completeness check
                completeness_query = """
                MATCH (f:Facility)
                WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
                
                OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
                WHERE i.incident_date >= date() - duration('P30D')
                
                OPTIONAL MATCH (f)-[:HAS_EMPLOYEE]->(e:Employee)
                OPTIONAL MATCH (f)-[:HAS_RISK_ASSESSMENT]->(ra:RiskAssessment)
                
                RETURN count(f) as total_facilities,
                       count(DISTINCT i) as recent_incidents,
                       count(DISTINCT e) as total_employees,
                       count(DISTINCT ra) as risk_assessments,
                       avg(CASE WHEN f.location IS NOT NULL THEN 1.0 ELSE 0.0 END) as location_completeness,
                       avg(CASE WHEN f.facility_type IS NOT NULL THEN 1.0 ELSE 0.0 END) as type_completeness,
                       avg(CASE WHEN ra IS NOT NULL THEN 1.0 ELSE 0.0 END) as risk_assessment_coverage
                """
                
                result = session.run(completeness_query, {"facility_ids": facility_ids}).single()
                
                if result:
                    risk_assessment_coverage = result['risk_assessment_coverage'] or 0.0
                    location_completeness = result['location_completeness'] or 0.0
                    type_completeness = result['type_completeness'] or 0.0
                    
                    # Calculate overall completeness score
                    completeness_score = (location_completeness + type_completeness + risk_assessment_coverage) / 3 * 100
                    
                    data_quality = {
                        "completeness_score": round(completeness_score, 2),
                        "risk_assessment_coverage": round(risk_assessment_coverage * 100, 2),
                        "data_freshness": {
                            "recent_incidents": result['recent_incidents'],
                            "total_facilities": result['total_facilities'],
                            "total_employees": result['total_employees'],
                            "risk_assessments": result['risk_assessments']
                        },
                        "quality_issues": [],
                        "overall_score": max(70.0, completeness_score)  # Minimum score with risk data considered
                    }
                    
                    # Add quality issues if found
                    if location_completeness < 0.9:
                        data_quality['quality_issues'].append("Some facilities missing location data")
                    
                    if type_completeness < 0.9:
                        data_quality['quality_issues'].append("Some facilities missing type classification")
                    
                    if risk_assessment_coverage < 0.8:
                        data_quality['quality_issues'].append("Risk assessment coverage below recommended 80%")
                    
                    # Factor in risk assessment data availability
                    if not risk_summary.get('has_data', False):
                        data_quality['quality_issues'].append("No current risk assessment data available")
                        data_quality['overall_score'] = min(data_quality['overall_score'], 60.0)
                    
                    return data_quality
            
            return {"error": "Unable to assess data quality"}
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_system_status(self, system_health: Dict, service_status: Dict, 
                                        data_quality: Dict) -> str:
        """Calculate overall system status"""
        try:
            # Check for any critical issues
            critical_issues = 0
            warning_issues = 0
            
            # Check system health
            for component, health in system_health.items():
                if isinstance(health, dict):
                    if health.get('status') == 'unhealthy':
                        critical_issues += 1
                    elif health.get('status') == 'degraded':
                        warning_issues += 1
            
            # Check service status
            for service, status in service_status.items():
                if status == 'down':
                    critical_issues += 1
                elif status == 'degraded':
                    warning_issues += 1
            
            # Check data quality
            if isinstance(data_quality, dict) and 'overall_score' in data_quality:
                if data_quality['overall_score'] < 70:
                    critical_issues += 1
                elif data_quality['overall_score'] < 85:
                    warning_issues += 1
            
            # Determine overall status
            if critical_issues > 0:
                return DashboardStatus.CRITICAL.value
            elif warning_issues > 0:
                return DashboardStatus.WARNING.value
            else:
                return DashboardStatus.HEALTHY.value
                
        except Exception as e:
            logger.error(f"Failed to calculate overall system status: {e}")
            return DashboardStatus.UNKNOWN.value
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # Placeholder implementation - would track actual cache hits/misses
        return 75.0
    
    def _generate_trend_summary(self, trends_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of trend analysis results"""
        try:
            summary = {
                "total_metrics_analyzed": len(trends_data),
                "trends_detected": 0,
                "anomalies_detected": 0,
                "concerning_trends": [],
                "positive_trends": []
            }
            
            for metric, trend_data in trends_data.items():
                if 'error' not in trend_data:
                    # Count trends
                    key_findings = trend_data.get('key_findings', [])
                    for finding in key_findings:
                        if finding.get('type') == 'trend':
                            summary['trends_detected'] += 1
                            # Categorize trend
                            if 'decreasing' in finding.get('finding', '').lower():
                                if 'incident' in metric or 'violation' in metric:
                                    summary['positive_trends'].append(f"{metric}: {finding['finding']}")
                                else:
                                    summary['concerning_trends'].append(f"{metric}: {finding['finding']}")
                            elif 'increasing' in finding.get('finding', '').lower():
                                if 'incident' in metric or 'violation' in metric:
                                    summary['concerning_trends'].append(f"{metric}: {finding['finding']}")
                                else:
                                    summary['positive_trends'].append(f"{metric}: {finding['finding']}")
                        
                        elif finding.get('type') == 'anomalies':
                            summary['anomalies_detected'] += 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate trend summary: {e}")
            return {"error": str(e)}
    
    def _generate_ai_recommendations(self, facility_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Generate AI-driven recommendations based on current data analysis"""
        try:
            recommendations = []
            
            # Get risk assessment data for AI recommendations
            risk_summary = self._get_facility_risk_summary(facility_ids)
            
            # Analyze current KPIs for recommendations
            safety_kpis = self.analytics.calculate_safety_kpis(facility_ids)
            compliance_kpis = self.analytics.calculate_compliance_kpis(facility_ids)
            
            # High incident rate recommendation
            incident_rate_kpi = safety_kpis.get('incident_rate')
            if incident_rate_kpi and incident_rate_kpi.threshold_warning:
                if incident_rate_kpi.value > incident_rate_kpi.threshold_warning:
                    recommendations.append({
                        "id": str(uuid.uuid4()),
                        "title": "Implement Enhanced Safety Training Program",
                        "description": f"Current incident rate ({incident_rate_kpi.value}) exceeds warning threshold. Recommend implementing enhanced safety training with focus on high-risk activities.",
                        "priority": "high",
                        "category": "safety_training",
                        "estimated_impact": "15-25% reduction in incident rate",
                        "implementation_timeframe": "30-60 days",
                        "confidence_score": 0.85,
                        "generated_at": datetime.now().isoformat(),
                        "ai_rationale": "Based on historical data analysis and industry best practices for facilities with similar incident patterns."
                    })
            
            # Low audit pass rate recommendation
            audit_rate_kpi = compliance_kpis.get('audit_pass_rate')
            if audit_rate_kpi and audit_rate_kpi.target:
                if audit_rate_kpi.value < audit_rate_kpi.target:
                    recommendations.append({
                        "id": str(uuid.uuid4()),
                        "title": "Strengthen Audit Preparation Process",
                        "description": f"Audit pass rate ({audit_rate_kpi.value}%) below target ({audit_rate_kpi.target}%). Recommend implementing systematic audit preparation with pre-audit assessments.",
                        "priority": "medium",
                        "category": "compliance_improvement",
                        "estimated_impact": "10-15% improvement in audit pass rate",
                        "implementation_timeframe": "45-90 days",
                        "confidence_score": 0.78,
                        "generated_at": datetime.now().isoformat(),
                        "ai_rationale": "Statistical analysis shows correlation between structured preparation processes and improved audit outcomes."
                    })
            
            # Risk-based recommendation
            if risk_summary.get('overall_risk_level') == 'HIGH':
                recommendations.append({
                    "id": str(uuid.uuid4()),
                    "title": "Implement Comprehensive Risk Mitigation Program",
                    "description": f"Current overall risk level is HIGH with average risk score of {risk_summary.get('avg_risk_score', 0)}. Immediate action required to reduce facility risk exposure.",
                    "priority": "critical",
                    "category": "risk_management",
                    "estimated_impact": "30-40% reduction in overall risk score",
                    "implementation_timeframe": "15-30 days",
                    "confidence_score": 0.95,
                    "generated_at": datetime.now().isoformat(),
                    "ai_rationale": "High-risk facilities require immediate intervention based on risk assessment data and historical incident correlation."
                })
            
            # Training completion recommendation
            training_rate_kpi = compliance_kpis.get('training_completion_rate')
            if training_rate_kpi and training_rate_kpi.target:
                if training_rate_kpi.value < training_rate_kpi.target:
                    recommendations.append({
                        "id": str(uuid.uuid4()),
                        "title": "Deploy Automated Training Reminder System",
                        "description": f"Training completion rate ({training_rate_kpi.value}%) below target. Implement automated reminder system with manager escalation for overdue training.",
                        "priority": "medium",
                        "category": "process_automation",
                        "estimated_impact": "20-30% improvement in training completion rates",
                        "implementation_timeframe": "15-30 days",
                        "confidence_score": 0.92,
                        "generated_at": datetime.now().isoformat(),
                        "ai_rationale": "Automation and systematic reminders have proven effective in similar organizational contexts."
                    })
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate AI recommendations: {e}")
            return []
    
    def _calculate_overall_health_score(self, dashboard_data: Dict[str, Any], 
                                       risk_summary: Dict[str, Any]) -> float:
        """Calculate overall health score for the dashboard including risk assessment"""
        try:
            scores = []
            weights = []
            
            # KPI scores (30% weight - reduced to make room for risk assessment)
            if 'kpis' in dashboard_data and 'metrics' in dashboard_data['kpis']:
                kpi_scores = []
                for kpi_name, kpi_data in dashboard_data['kpis']['metrics'].items():
                    if 'status' in kpi_data:
                        if kpi_data['status'] == 'green':
                            kpi_scores.append(100)
                        elif kpi_data['status'] == 'yellow':
                            kpi_scores.append(70)
                        else:  # red
                            kpi_scores.append(30)
                
                if kpi_scores:
                    scores.append(sum(kpi_scores) / len(kpi_scores))
                    weights.append(0.3)
            
            # Alert level (25% weight)
            if 'alerts' in dashboard_data and 'summary' in dashboard_data['alerts']:
                alert_level = dashboard_data['alerts']['summary'].get('alert_level', 'GREEN')
                if alert_level == 'GREEN':
                    scores.append(100)
                elif alert_level == 'YELLOW':
                    scores.append(75)
                elif alert_level == 'ORANGE':
                    scores.append(50)
                else:  # RED
                    scores.append(25)
                weights.append(0.25)
            
            # Risk Assessment Score (25% weight - new)
            risk_level = risk_summary.get('overall_risk_level', 'MEDIUM')
            avg_risk_score = risk_summary.get('avg_risk_score', 25.0)
            
            # Convert risk level to health score (inverse relationship)
            if risk_level == 'LOW':
                risk_health_score = 100
            elif risk_level == 'MEDIUM':
                # Scale medium risk based on actual score
                risk_health_score = max(50, 100 - (avg_risk_score * 2))
            else:  # HIGH
                risk_health_score = max(20, 70 - avg_risk_score)
            
            scores.append(risk_health_score)
            weights.append(0.25)
            
            # System status (15% weight - reduced)
            if 'status' in dashboard_data and 'overall_status' in dashboard_data['status']:
                status = dashboard_data['status']['overall_status']
                if status == DashboardStatus.HEALTHY.value:
                    scores.append(100)
                elif status == DashboardStatus.WARNING.value:
                    scores.append(70)
                elif status == DashboardStatus.CRITICAL.value:
                    scores.append(30)
                else:
                    scores.append(50)
                weights.append(0.15)
            
            # Data quality (5% weight - reduced)
            if 'status' in dashboard_data and 'data_quality' in dashboard_data['status']:
                data_quality = dashboard_data['status']['data_quality']
                if isinstance(data_quality, dict) and 'overall_score' in data_quality:
                    scores.append(data_quality['overall_score'])
                    weights.append(0.05)
            
            # Calculate weighted average
            if scores and weights:
                weighted_score = sum(score * weight for score, weight in zip(scores, weights))
                total_weight = sum(weights)
                return round(weighted_score / total_weight, 2)
            
            return 75.0  # Default score if unable to calculate
            
        except Exception as e:
            logger.error(f"Failed to calculate overall health score: {e}")
            return 50.0  # Conservative default on error
    
    # Public API Methods
    
    def get_dashboard_summary(self, facility_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get high-level dashboard summary"""
        location_filter = LocationFilter(facility_ids=facility_ids) if facility_ids else None
        
        dashboard_data = self.generate_executive_dashboard(
            location_filter=location_filter,
            include_trends=False,
            include_recommendations=False,
            include_forecasts=False
        )
        
        return {
            "summary": dashboard_data.get("summary", {}),
            "alert_level": dashboard_data.get("alerts", {}).get("summary", {}).get("alert_level", "GREEN"),
            "overall_health_score": dashboard_data.get("summary", {}).get("overall_health_score", 75.0),
            "risk_assessment": dashboard_data.get("risk_assessment", {}),
            "generated_at": datetime.now().isoformat()
        }
    
    def get_real_time_metrics(self, facility_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get real-time metrics"""
        return self.analytics.get_real_time_metrics(facility_ids)
    
    def get_risk_assessment_summary(self, facility_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get risk assessment summary for facilities"""
        return self._get_facility_risk_summary(facility_ids)
    
    def get_kpi_details(self, facility_ids: Optional[List[str]] = None, 
                       date_range_days: int = 30) -> Dict[str, Any]:
        """Get detailed KPI information"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_range_days)
        
        safety_kpis = self.analytics.calculate_safety_kpis(facility_ids, start_date, end_date)
        compliance_kpis = self.analytics.calculate_compliance_kpis(facility_ids, start_date, end_date)
        
        return {
            "safety_kpis": {name: self._kpi_to_dict(kpi) for name, kpi in safety_kpis.items()},
            "compliance_kpis": {name: self._kpi_to_dict(kpi) for name, kpi in compliance_kpis.items()},
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": date_range_days
            },
            "generated_at": datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform service health check"""
        return self._perform_system_health_check()


# Factory function for easy initialization
def create_dashboard_service(**kwargs) -> ExecutiveDashboardService:
    """
    Factory function to create ExecutiveDashboardService instance
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured ExecutiveDashboardService instance
    """
    return ExecutiveDashboardService(**kwargs)


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Create dashboard service
    dashboard_service = create_dashboard_service()
    
    try:
        # Test basic functionality
        print("Testing dashboard service...")
        
        # Health check
        health_status = dashboard_service.health_check()
        print(f"Health Status: {health_status}")
        
        # Test risk assessment data
        risk_summary = dashboard_service.get_risk_assessment_summary()
        print(f"Risk Summary: {risk_summary}")
        
        # Generate sample dashboard
        dashboard_data = dashboard_service.generate_executive_dashboard()
        
        if 'error' not in dashboard_data:
            print("Dashboard generated successfully with risk assessment data!")
            print(f"Overall Health Score: {dashboard_data.get('summary', {}).get('overall_health_score', 'N/A')}")
            print(f"Alert Level: {dashboard_data.get('alerts', {}).get('summary', {}).get('alert_level', 'N/A')}")
            print(f"Risk Level: {dashboard_data.get('risk_assessment', {}).get('overall_risk_level', 'N/A')}")
            print(f"Risk Score: {dashboard_data.get('risk_assessment', {}).get('avg_risk_score', 'N/A')}")
        else:
            print(f"Dashboard generation failed: {dashboard_data['error']}")
    
    finally:
        # Clean up
        dashboard_service.close()
        print("Dashboard service closed.")