"""
Dashboard Category API Router

This module provides FastAPI endpoints for category-specific dashboard data,
including electricity, water, and waste consumption metrics. Each endpoint
integrates with EHS goals configuration and Neo4j data to provide comprehensive
views of environmental performance across different categories.

Features:
- GET /api/dashboard/electricity - Electricity consumption dashboard
- GET /api/dashboard/water - Water consumption dashboard  
- GET /api/dashboard/waste - Waste generation dashboard
- Site-specific filtering (algonquin_il or houston_tx)
- Integration with EHS goals configuration
- 6-month consumption facts from Neo4j
- Risk assessment data (populated by Risk Agent)
- Recommendation data (populated by Risk Agent)

Created: 2025-08-31
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from pydantic import BaseModel, Field, validator
import logging

from src.config.ehs_goals_config import (
    EHSGoalsConfig, EHSGoal, SiteLocation, EHSCategory, 
    ehs_goals_config, get_goal, get_reduction_percentage
)
from src.services.environmental_assessment_service import EnvironmentalAssessmentService
from src.database.neo4j_client import Neo4jClient, ConnectionConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard-category"])

# Pydantic Models

class GoalSummary(BaseModel):
    """Model for EHS goal summary data"""
    reduction_percentage: float = Field(..., description="Target reduction percentage")
    baseline_year: int = Field(..., description="Baseline year for comparison")
    target_year: int = Field(..., description="Target year for achievement")
    unit: str = Field(..., description="Unit of measurement")
    description: str = Field(..., description="Goal description")

class ConsumptionFacts(BaseModel):
    """Model for consumption facts data"""
    total_consumption: float = Field(..., description="Total consumption for the period")
    average_consumption: float = Field(..., description="Average consumption per period")
    max_consumption: float = Field(..., description="Maximum consumption in period")
    min_consumption: float = Field(..., description="Minimum consumption in period")
    total_cost: float = Field(..., description="Total cost for the period")
    average_cost: float = Field(..., description="Average cost per period")
    data_points: int = Field(..., description="Number of data points")
    date_range: Dict[str, str] = Field(..., description="Date range of the data")
    unit: str = Field(..., description="Unit of measurement")

class WasteFacts(BaseModel):
    """Model for waste generation facts data"""
    total_generated: float = Field(..., description="Total waste generated for the period")
    total_recycled: float = Field(..., description="Total waste recycled")
    recycling_rate: float = Field(..., description="Recycling rate as percentage")
    average_generated: float = Field(..., description="Average waste generated per period")
    max_generated: float = Field(..., description="Maximum waste generated in period")
    min_generated: float = Field(..., description="Minimum waste generated in period")
    total_cost: float = Field(..., description="Total disposal cost")
    average_cost: float = Field(..., description="Average disposal cost per period")
    data_points: int = Field(..., description="Number of data points")
    date_range: Dict[str, str] = Field(..., description="Date range of the data")
    waste_types: Dict[str, float] = Field(..., description="Breakdown by waste type")
    unit: str = Field(..., description="Unit of measurement")

class RiskData(BaseModel):
    """Model for risk assessment data"""
    type: str = Field(..., description="Risk type identifier")
    severity: str = Field(..., description="Risk severity level")
    description: str = Field(..., description="Risk description")
    recommendation: str = Field(..., description="Risk mitigation recommendation")

class CategoryDashboardResponse(BaseModel):
    """Response model for category dashboard endpoints"""
    category: str = Field(..., description="Category name (electricity, water, waste)")
    site_id: str = Field(..., description="Site identifier")
    goals: GoalSummary = Field(..., description="EHS goals for this category and site")
    facts: Union[ConsumptionFacts, WasteFacts] = Field(..., description="6-month consumption/generation facts")
    risks: List[RiskData] = Field(..., description="Risk assessment data (empty until Risk Agent runs)")
    recommendations: List[str] = Field(..., description="Recommendation data (empty until Risk Agent runs)")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")

class ErrorResponse(BaseModel):
    """Response model for error cases"""
    error: str = Field(..., description="Error message")
    category: str = Field(..., description="Category name")
    site_id: str = Field(..., description="Site identifier")
    timestamp: str = Field(..., description="Error timestamp")

# Dependency Functions

async def get_environmental_service() -> Optional[EnvironmentalAssessmentService]:
    """
    Dependency to get environmental assessment service with Neo4j client.
    
    Returns:
        EnvironmentalAssessmentService instance or None if unavailable
    """
    try:
        # Create Neo4j client configuration from environment
        config = ConnectionConfig.from_env()
        
        # Create Neo4j client instance
        neo4j_client = Neo4jClient(config=config, enable_logging=True)
        
        # Test the connection
        if neo4j_client.connect():
            logger.info("Successfully connected to Neo4j database for dashboard category service")
            return EnvironmentalAssessmentService(neo4j_client)
        else:
            logger.warning("Failed to connect to Neo4j database for dashboard category service")
            return None
            
    except ImportError as e:
        logger.warning(f"Neo4j dependencies not available for dashboard category service: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing Neo4j client for dashboard category service: {e}")
        return None

async def get_neo4j_client() -> Optional[Neo4jClient]:
    """
    Dependency to get Neo4j client directly.
    
    Returns:
        Neo4jClient instance or None if unavailable
    """
    try:
        # Create Neo4j client configuration from environment
        config = ConnectionConfig.from_env()
        
        # Create Neo4j client instance
        neo4j_client = Neo4jClient(config=config, enable_logging=True)
        
        # Test the connection
        if neo4j_client.connect():
            logger.info("Successfully connected to Neo4j database for direct access")
            return neo4j_client
        else:
            logger.warning("Failed to connect to Neo4j database for direct access")
            return None
            
    except ImportError as e:
        logger.warning(f"Neo4j dependencies not available for direct access: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing Neo4j client for direct access: {e}")
        return None

# Helper Functions

def validate_site_id(site_id: str) -> str:
    """
    Validate and normalize site_id parameter.
    
    Args:
        site_id: Site identifier (algonquin_il or houston_tx)
        
    Returns:
        Normalized site identifier
        
    Raises:
        HTTPException: If site_id is invalid
    """
    # Normalize site_id mapping
    site_mapping = {
        'algonquin_il': 'algonquin_illinois',
        'algonquin_illinois': 'algonquin_illinois',
        'houston_tx': 'houston_texas',
        'houston_texas': 'houston_texas'
    }
    
    normalized_site = site_mapping.get(site_id.lower())
    if not normalized_site:
        available_sites = ', '.join(site_mapping.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid site_id '{site_id}'. Available sites: {available_sites}"
        )
    
    return normalized_site

def get_location_filter(site_id: str) -> str:
    """
    Convert site_id to location filter for Neo4j queries.
    
    Args:
        site_id: Normalized site identifier
        
    Returns:
        Location filter string for Neo4j queries
    """
    location_filters = {
        'algonquin_illinois': 'algonquin',
        'houston_texas': 'houston'
    }
    return location_filters.get(site_id, site_id)

def get_six_month_date_range() -> tuple[str, str]:
    """
    Get date range for the last 6 months.
    
    Returns:
        Tuple of (start_date, end_date) as ISO date strings
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # Approximately 6 months
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def convert_consumption_facts(facts_data: Dict[str, Any], unit: str) -> ConsumptionFacts:
    """
    Convert environmental service facts to ConsumptionFacts model.
    
    Args:
        facts_data: Raw facts data from environmental service
        unit: Unit of measurement
        
    Returns:
        ConsumptionFacts model instance
    """
    return ConsumptionFacts(
        total_consumption=facts_data.get('total_consumption', 0),
        average_consumption=facts_data.get('average_consumption', 0),
        max_consumption=facts_data.get('max_consumption', 0),
        min_consumption=facts_data.get('min_consumption', 0),
        total_cost=facts_data.get('total_cost', 0),
        average_cost=facts_data.get('average_cost', 0),
        data_points=facts_data.get('data_points', 0),
        date_range=facts_data.get('date_range', {'start': '', 'end': ''}),
        unit=unit
    )

def convert_waste_facts(facts_data: Dict[str, Any], unit: str) -> WasteFacts:
    """
    Convert environmental service facts to WasteFacts model.
    
    Args:
        facts_data: Raw facts data from environmental service
        unit: Unit of measurement
        
    Returns:
        WasteFacts model instance
    """
    return WasteFacts(
        total_generated=facts_data.get('total_generated', 0),
        total_recycled=facts_data.get('total_recycled', 0),
        recycling_rate=facts_data.get('recycling_rate', 0),
        average_generated=facts_data.get('average_generated', 0),
        max_generated=facts_data.get('max_generated', 0),
        min_generated=facts_data.get('min_generated', 0),
        total_cost=facts_data.get('total_cost', 0),
        average_cost=facts_data.get('average_cost', 0),
        data_points=facts_data.get('data_points', 0),
        date_range=facts_data.get('date_range', {'start': '', 'end': ''}),
        waste_types=facts_data.get('waste_types', {}),
        unit=unit
    )

def convert_risks(risks_data: List[Dict[str, Any]]) -> List[RiskData]:
    """
    Convert environmental service risks to RiskData models.
    
    Args:
        risks_data: Raw risks data from environmental service
        
    Returns:
        List of RiskData model instances
    """
    return [
        RiskData(
            type=risk.get('type', 'unknown'),
            severity=risk.get('severity', 'unknown'),
            description=risk.get('description', ''),
            recommendation=risk.get('recommendation', '')
        )
        for risk in risks_data
    ]

def get_risks_from_neo4j(neo4j_client: Neo4jClient, site_id: str, category: str) -> List[Dict]:
    """Retrieve risk assessments from Neo4j"""
    try:
        with neo4j_client.session_scope() as session:
            query = """
            MATCH (s:Site {id: $site_id})-[:HAS_RISK]->(r:RiskAssessment {category: $category})
            RETURN r.risk_level as level, r.description as description, 
                   r.confidence_score as confidence, r.factors as factors
            ORDER BY r.assessment_date DESC
            LIMIT 5
            """
            
            result = session.run(query, {"site_id": site_id, "category": category})
            risks = []
            for record in result:
                risks.append({
                    "type": "risk_assessment",
                    "severity": record["level"],
                    "description": record["description"],
                    "recommendation": "See recommendations below"
                })
            return risks
    except Exception as e:
        logger.error(f"Error retrieving risks: {e}")
        return []

def get_recommendations_from_neo4j(neo4j_client: Neo4jClient, site_id: str, category: str) -> List[str]:
    """Retrieve recommendations from Neo4j"""
    try:
        with neo4j_client.session_scope() as session:
            query = """
            MATCH (s:Site {id: $site_id})-[:HAS_RECOMMENDATION]->(r:Recommendation {category: $category})
            RETURN r.title as title, r.description as description, r.priority as priority
            ORDER BY r.priority DESC, r.created_at DESC
            LIMIT 5
            """
            
            result = session.run(query, {"site_id": site_id, "category": category})
            recommendations = []
            for record in result:
                rec_text = f"{record['title']}: {record['description']}"
                recommendations.append(rec_text)
            return recommendations
    except Exception as e:
        logger.error(f"Error retrieving recommendations: {e}")
        return []

# API Endpoints

@router.get("/electricity", response_model=CategoryDashboardResponse)
async def get_electricity_dashboard(
    site_id: str = Query(..., description="Site identifier (algonquin_il or houston_tx)"),
    service = Depends(get_environmental_service),
    neo4j_client = Depends(get_neo4j_client)
):
    """
    Get electricity consumption dashboard data.
    
    Returns electricity goals, facts, risks, and recommendations for the specified site.
    Includes 6 months of consumption data from Neo4j.
    
    Args:
        site_id: Site identifier (algonquin_il or houston_tx)
        
    Returns:
        CategoryDashboardResponse with electricity data
    """
    try:
        logger.info(f"Getting electricity dashboard for site: {site_id}")
        
        # Validate and normalize site_id
        normalized_site = validate_site_id(site_id)
        
        # Get EHS goal for electricity (CO2 emissions)
        goal = get_goal(normalized_site, EHSCategory.CO2)
        if not goal:
            raise HTTPException(
                status_code=404,
                detail=f"No electricity goal found for site '{site_id}'"
            )
        
        # Convert goal to response model
        goal_summary = GoalSummary(
            reduction_percentage=goal.reduction_percentage,
            baseline_year=goal.baseline_year,
            target_year=goal.target_year,
            unit=goal.unit,
            description=goal.description
        )
        
        # Initialize empty facts, risks, and recommendations
        facts = ConsumptionFacts(
            total_consumption=0,
            average_consumption=0,
            max_consumption=0,
            min_consumption=0,
            total_cost=0,
            average_cost=0,
            data_points=0,
            date_range={'start': '', 'end': ''},
            unit='kWh'
        )
        risks = []
        recommendations = []
        
        # Try to get data from environmental service if available
        if service:
            try:
                # Get 6-month date range
                start_date, end_date = get_six_month_date_range()
                location_filter = get_location_filter(normalized_site)
                
                # Query electricity data
                electricity_data = service.assess_electricity_consumption(
                    location_filter=location_filter,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if 'error' not in electricity_data:
                    # Convert facts
                    if 'facts' in electricity_data:
                        facts = convert_consumption_facts(electricity_data['facts'], 'kWh')
                    
                    # Convert risks - these will be overridden by Neo4j data if available
                    if 'risks' in electricity_data:
                        risks = convert_risks(electricity_data['risks'])
                    
                    # Convert recommendations - these will be overridden by Neo4j data if available
                    if 'recommendations' in electricity_data:
                        recommendations = electricity_data['recommendations']
                        
                logger.info(f"Successfully retrieved electricity data for {normalized_site}")
                
            except Exception as e:
                logger.warning(f"Error retrieving electricity data from service: {e}")
                # Continue with empty data - service errors shouldn't fail the endpoint
        
        # Get risks and recommendations from Neo4j using original site_id
        if neo4j_client:
            neo4j_risks = get_risks_from_neo4j(neo4j_client, site_id, "electricity")
            if neo4j_risks:
                risks = convert_risks(neo4j_risks)
            
            neo4j_recommendations = get_recommendations_from_neo4j(neo4j_client, site_id, "electricity")
            if neo4j_recommendations:
                recommendations = neo4j_recommendations
        
        response = CategoryDashboardResponse(
            category="electricity",
            site_id=site_id,
            goals=goal_summary,
            facts=facts,
            risks=risks,
            recommendations=recommendations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "data_source": "neo4j" if service else "unavailable",
                "site_normalized": normalized_site,
                "goal_category": "co2_emissions",
                "risks_source": "neo4j_direct" if neo4j_client else "service",
                "recommendations_source": "neo4j_direct" if neo4j_client else "service"
            }
        )
        
        logger.info(f"Successfully generated electricity dashboard for {site_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating electricity dashboard for {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate electricity dashboard for site {site_id}"
        )

@router.get("/water", response_model=CategoryDashboardResponse)
async def get_water_dashboard(
    site_id: str = Query(..., description="Site identifier (algonquin_il or houston_tx)"),
    service = Depends(get_environmental_service),
    neo4j_client = Depends(get_neo4j_client)
):
    """
    Get water consumption dashboard data.
    
    Returns water goals, facts, risks, and recommendations for the specified site.
    Includes 6 months of consumption data from Neo4j.
    
    Args:
        site_id: Site identifier (algonquin_il or houston_tx)
        
    Returns:
        CategoryDashboardResponse with water data
    """
    try:
        logger.info(f"Getting water dashboard for site: {site_id}")
        
        # Validate and normalize site_id
        normalized_site = validate_site_id(site_id)
        
        # Get EHS goal for water consumption
        goal = get_goal(normalized_site, EHSCategory.WATER)
        if not goal:
            raise HTTPException(
                status_code=404,
                detail=f"No water goal found for site '{site_id}'"
            )
        
        # Convert goal to response model
        goal_summary = GoalSummary(
            reduction_percentage=goal.reduction_percentage,
            baseline_year=goal.baseline_year,
            target_year=goal.target_year,
            unit=goal.unit,
            description=goal.description
        )
        
        # Initialize empty facts, risks, and recommendations
        facts = ConsumptionFacts(
            total_consumption=0,
            average_consumption=0,
            max_consumption=0,
            min_consumption=0,
            total_cost=0,
            average_cost=0,
            data_points=0,
            date_range={'start': '', 'end': ''},
            unit='gallons'
        )
        risks = []
        recommendations = []
        
        # Try to get data from environmental service if available
        if service:
            try:
                # Get 6-month date range
                start_date, end_date = get_six_month_date_range()
                location_filter = get_location_filter(normalized_site)
                
                # Query water data
                water_data = service.assess_water_consumption(
                    location_filter=location_filter,
                    start_date=start_date,
                    end_date=end_date
                )

                # Override with hardcoded values for specific sites


                
                if 'error' not in water_data:
                    # Convert facts
                    if 'facts' in water_data:
                        facts = convert_consumption_facts(water_data['facts'], 'gallons')
                    
                    # Convert risks - these will be overridden by Neo4j data if available
                    if 'risks' in water_data:
                        risks = convert_risks(water_data['risks'])
                    
                    # Convert recommendations - these will be overridden by Neo4j data if available
                    if 'recommendations' in water_data:
                        recommendations = water_data['recommendations']
                        
                logger.info(f"Successfully retrieved water data for {normalized_site}")
                
            except Exception as e:
                logger.warning(f"Error retrieving water data from service: {e}")
                # Continue with empty data - service errors shouldn't fail the endpoint
        
        # Get risks and recommendations from Neo4j using original site_id
        if neo4j_client:
            neo4j_risks = get_risks_from_neo4j(neo4j_client, site_id, "water")
            if neo4j_risks:
                risks = convert_risks(neo4j_risks)
            
            neo4j_recommendations = get_recommendations_from_neo4j(neo4j_client, site_id, "water")
            if neo4j_recommendations:
                recommendations = neo4j_recommendations
        
        response = CategoryDashboardResponse(
            category="water",
            site_id=site_id,
            goals=goal_summary,
            facts=facts,
            risks=risks,
            recommendations=recommendations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "data_source": "neo4j" if service else "unavailable",
                "site_normalized": normalized_site,
                "goal_category": "water_consumption",
                "risks_source": "neo4j_direct" if neo4j_client else "service",
                "recommendations_source": "neo4j_direct" if neo4j_client else "service"
            }
        )
        
        logger.info(f"Successfully generated water dashboard for {site_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating water dashboard for {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate water dashboard for site {site_id}"
        )

@router.get("/waste", response_model=CategoryDashboardResponse)
async def get_waste_dashboard(
    site_id: str = Query(..., description="Site identifier (algonquin_il or houston_tx)"),
    service = Depends(get_environmental_service),
    neo4j_client = Depends(get_neo4j_client)
):
    """
    Get waste generation dashboard data.
    
    Returns waste goals, facts, risks, and recommendations for the specified site.
    Includes 6 months of generation data from Neo4j.
    
    Args:
        site_id: Site identifier (algonquin_il or houston_tx)
        
    Returns:
        CategoryDashboardResponse with waste data
    """
    try:
        logger.info(f"Getting waste dashboard for site: {site_id}")
        
        # Validate and normalize site_id
        normalized_site = validate_site_id(site_id)
        
        # Get EHS goal for waste generation
        goal = get_goal(normalized_site, EHSCategory.WASTE)
        if not goal:
            raise HTTPException(
                status_code=404,
                detail=f"No waste goal found for site '{site_id}'"
            )
        
        # Convert goal to response model
        goal_summary = GoalSummary(
            reduction_percentage=goal.reduction_percentage,
            baseline_year=goal.baseline_year,
            target_year=goal.target_year,
            unit=goal.unit,
            description=goal.description
        )
        
        # Initialize empty facts, risks, and recommendations
        facts = WasteFacts(
            total_generated=0,
            total_recycled=0,
            recycling_rate=0,
            average_generated=0,
            max_generated=0,
            min_generated=0,
            total_cost=0,
            average_cost=0,
            data_points=0,
            date_range={'start': '', 'end': ''},
            waste_types={},
            unit='pounds'
        )
        risks = []
        recommendations = []
        
        # Try to get data from environmental service if available
        if service:
            try:
                # Get 6-month date range
                start_date, end_date = get_six_month_date_range()
                location_filter = get_location_filter(normalized_site)
                
                # Query waste data
                waste_data = service.assess_waste_generation(
                    location_filter=location_filter,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if 'error' not in waste_data:
                    # Convert facts
                    if 'facts' in waste_data:
                        facts = convert_waste_facts(waste_data['facts'], 'pounds')
                    
                    # Convert risks - these will be overridden by Neo4j data if available
                    if 'risks' in waste_data:
                        risks = convert_risks(waste_data['risks'])
                    
                    # Convert recommendations - these will be overridden by Neo4j data if available
                    if 'recommendations' in waste_data:
                        recommendations = waste_data['recommendations']
                        
                logger.info(f"Successfully retrieved waste data for {normalized_site}")
                
            except Exception as e:
                logger.warning(f"Error retrieving waste data from service: {e}")
                # Continue with empty data - service errors shouldn't fail the endpoint
        
        # Get risks and recommendations from Neo4j using original site_id
        if neo4j_client:
            neo4j_risks = get_risks_from_neo4j(neo4j_client, site_id, "waste")
            if neo4j_risks:
                risks = convert_risks(neo4j_risks)
            
            neo4j_recommendations = get_recommendations_from_neo4j(neo4j_client, site_id, "waste")
            if neo4j_recommendations:
                recommendations = neo4j_recommendations
        
        response = CategoryDashboardResponse(
            category="waste",
            site_id=site_id,
            goals=goal_summary,
            facts=facts,
            risks=risks,
            recommendations=recommendations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "data_source": "neo4j" if service else "unavailable",
                "site_normalized": normalized_site,
                "goal_category": "waste_generation",
                "risks_source": "neo4j_direct" if neo4j_client else "service",
                "recommendations_source": "neo4j_direct" if neo4j_client else "service"
            }
        )
        
        logger.info(f"Successfully generated waste dashboard for {site_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating waste dashboard for {site_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate waste dashboard for site {site_id}"
        )

# Health check endpoint
@router.get("/category/health")
async def dashboard_category_health():
    """
    Health check endpoint for dashboard category service.
    
    Returns:
        Health status information including service availability
    """
    try:
        # Check EHS goals configuration
        config_valid = ehs_goals_config.validate_configuration()
        
        # Check Neo4j service availability
        service = await get_environmental_service()
        neo4j_available = service is not None
        
        # Determine overall status
        if config_valid and neo4j_available:
            status = "healthy"
        elif config_valid:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ehs_goals_config": "healthy" if config_valid else "unhealthy",
                "neo4j_service": "healthy" if neo4j_available else "unavailable",
                "environmental_service": "healthy" if service else "unavailable"
            },
            "endpoints": [
                "/api/dashboard/electricity",
                "/api/dashboard/water", 
                "/api/dashboard/waste"
            ],
            "supported_sites": ["algonquin_il", "houston_tx"],
            "data_period": "6 months"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "components": {
                "ehs_goals_config": "unknown",
                "neo4j_service": "unknown", 
                "environmental_service": "unknown"
            }
        }
