"""
EHS Goals API Router

This module provides FastAPI endpoints for retrieving EHS (Environmental Health & Safety) 
annual goals and progress tracking for CO2 emissions, water consumption, and waste generation
across different sites. Integrates with the environmental assessment service to calculate 
actual progress against targets.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Query, Depends, Path
from pydantic import BaseModel, Field
import logging

from src.config.ehs_goals_config import (
    EHSGoalsConfig, EHSGoal, SiteLocation, EHSCategory, 
    ehs_goals_config, get_all_goals, get_goals_summary
)
from src.services.environmental_assessment_service import EnvironmentalAssessmentService
from src.database.neo4j_client import Neo4jClient, ConnectionConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/goals", tags=["ehs-goals"])

# Pydantic Models
class EHSGoalModel(BaseModel):
    """Model for EHS goal data"""
    site: str = Field(..., description="Site location identifier")
    category: str = Field(..., description="EHS category (co2_emissions, water_consumption, waste_generation)")
    reduction_percentage: float = Field(..., description="Target reduction percentage")
    baseline_year: int = Field(..., description="Baseline year for comparison")
    target_year: int = Field(..., description="Target year for achievement")
    unit: str = Field(..., description="Unit of measurement")
    description: str = Field(..., description="Goal description")

class GoalsResponse(BaseModel):
    """Response model for goals endpoints"""
    goals: List[EHSGoalModel] = Field(..., description="List of EHS goals")
    total_goals: int = Field(..., description="Total number of goals")
    sites: List[str] = Field(..., description="Available site locations")
    categories: List[str] = Field(..., description="Available EHS categories")

class ProgressMetrics(BaseModel):
    """Model for progress calculation metrics"""
    baseline_value: Optional[float] = Field(None, description="Baseline consumption value")
    current_value: Optional[float] = Field(None, description="Current consumption value")
    target_value: Optional[float] = Field(None, description="Target consumption value")
    actual_reduction: Optional[float] = Field(None, description="Actual reduction achieved (%)")
    target_reduction: float = Field(..., description="Target reduction percentage")
    progress_percentage: Optional[float] = Field(None, description="Progress towards goal (%)")
    status: str = Field(..., description="Goal status (on_track, behind, ahead, insufficient_data)")
    unit: str = Field(..., description="Unit of measurement")

class SiteProgressModel(BaseModel):
    """Model for site progress data"""
    site: str = Field(..., description="Site location identifier")
    category: str = Field(..., description="EHS category")
    goal: EHSGoalModel = Field(..., description="Goal details")
    progress: ProgressMetrics = Field(..., description="Progress metrics")
    last_updated: datetime = Field(..., description="Last data update timestamp")

class ProgressResponse(BaseModel):
    """Response model for progress endpoints"""
    site_progress: List[SiteProgressModel] = Field(..., description="Progress data for each category")
    overall_status: str = Field(..., description="Overall site progress status")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    calculation_date: datetime = Field(..., description="When progress was calculated")

class GoalsSummaryModel(BaseModel):
    """Model for goals summary"""
    site_summary: Dict[str, Dict[str, float]] = Field(..., description="Goals by site and category")
    category_summary: Dict[str, List[Dict[str, Any]]] = Field(..., description="Goals by category")
    total_sites: int = Field(..., description="Total number of sites")
    total_categories: int = Field(..., description="Total number of categories")

# Dependency to get environmental assessment service
async def get_environmental_service():
    """Get environmental assessment service with Neo4j client initialization"""
    try:
        # Create Neo4j client configuration from environment
        config = ConnectionConfig.from_env()
        
        # Create Neo4j client instance
        neo4j_client = Neo4jClient(config=config, enable_logging=True)
        
        # Test the connection
        if neo4j_client.connect():
            logger.info("Successfully connected to Neo4j database for EHS goals")
            return EnvironmentalAssessmentService(neo4j_client)
        else:
            logger.warning("Failed to connect to Neo4j database for EHS goals service")
            return None
            
    except ImportError as e:
        logger.warning(f"Neo4j dependencies not available for EHS goals: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing Neo4j client for EHS goals: {e}")
        return None

# Helper functions
def convert_goal_to_model(goal: EHSGoal) -> EHSGoalModel:
    """Convert EHSGoal to EHSGoalModel"""
    return EHSGoalModel(
        site=goal.site.value,
        category=goal.category.value,
        reduction_percentage=goal.reduction_percentage,
        baseline_year=goal.baseline_year,
        target_year=goal.target_year,
        unit=goal.unit,
        description=goal.description
    )

def calculate_target_value(baseline_value: float, reduction_percentage: float) -> float:
    """Calculate target value based on baseline and reduction percentage"""
    return baseline_value * (1 - reduction_percentage / 100)

def calculate_progress_metrics(
    goal: EHSGoal, 
    baseline_data: Optional[Dict[str, Any]], 
    current_data: Optional[Dict[str, Any]]
) -> ProgressMetrics:
    """Calculate progress metrics for a goal"""
    
    # Extract baseline and current values
    baseline_value = None
    current_value = None
    
    if baseline_data and 'facts' in baseline_data:
        facts = baseline_data['facts']
        if 'total_consumption' in facts:
            baseline_value = float(facts['total_consumption'])
        elif 'total_generated' in facts:
            baseline_value = float(facts['total_generated'])
    
    if current_data and 'facts' in current_data:
        facts = current_data['facts']
        if 'total_consumption' in facts:
            current_value = float(facts['total_consumption'])
        elif 'total_generated' in facts:
            current_value = float(facts['total_generated'])
    
    # Calculate metrics if we have the required data
    if baseline_value is not None and current_value is not None and baseline_value > 0:
        target_value = calculate_target_value(baseline_value, goal.reduction_percentage)
        actual_reduction = ((baseline_value - current_value) / baseline_value) * 100
        
        # Calculate progress towards goal
        required_reduction = baseline_value - target_value
        achieved_reduction = baseline_value - current_value
        
        if required_reduction > 0:
            progress_percentage = (achieved_reduction / required_reduction) * 100
        else:
            progress_percentage = 100.0  # Already at or beyond target
        
        # Determine status
        if progress_percentage >= 100:
            status = "ahead"
        elif progress_percentage >= 80:
            status = "on_track"
        else:
            status = "behind"
    else:
        target_value = None
        actual_reduction = None
        progress_percentage = None
        status = "insufficient_data"
    
    return ProgressMetrics(
        baseline_value=baseline_value,
        current_value=current_value,
        target_value=target_value,
        actual_reduction=actual_reduction,
        target_reduction=goal.reduction_percentage,
        progress_percentage=progress_percentage,
        status=status,
        unit=goal.unit
    )

def map_category_to_service_method(category: str):
    """Map EHS category to environmental service method name"""
    category_map = {
        'co2_emissions': 'assess_electricity_consumption',  # CO2 from electricity
        'water_consumption': 'assess_water_consumption',
        'waste_generation': 'assess_waste_generation'
    }
    return category_map.get(category)

def get_site_location_filter(site: str) -> Optional[str]:
    """Convert site identifier to location filter for service calls"""
    site_filters = {
        'algonquin_illinois': 'algonquin',
        'houston_texas': 'houston'
    }
    return site_filters.get(site.lower())

# API Endpoints

@router.get("/annual", response_model=GoalsResponse)
async def get_annual_goals():
    """
    Get all annual EHS goals for all sites and categories.
    
    Returns comprehensive list of reduction targets for CO2 emissions,
    water consumption, and waste generation across all sites.
    """
    try:
        logger.info("Retrieving all annual EHS goals")
        
        # Get all goals from configuration
        all_goals = get_all_goals()
        
        # Convert to API models
        goal_models = [convert_goal_to_model(goal) for goal in all_goals]
        
        # Get available sites and categories
        sites = ehs_goals_config.get_site_names()
        categories = ehs_goals_config.get_category_names()
        
        response = GoalsResponse(
            goals=goal_models,
            total_goals=len(goal_models),
            sites=sites,
            categories=categories
        )
        
        logger.info(f"Successfully retrieved {len(goal_models)} annual goals")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving annual goals: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve annual goals")

@router.get("/annual/{site_id}", response_model=GoalsResponse)
async def get_site_goals(
    site_id: str = Path(..., description="Site identifier (algonquin_illinois or houston_texas)")
):
    """
    Get annual EHS goals for a specific site.
    
    Returns all reduction targets for the specified site across
    all EHS categories (CO2, water, waste).
    """
    try:
        logger.info(f"Retrieving annual goals for site: {site_id}")
        
        # Get goals for the specific site
        site_goals = ehs_goals_config.get_goals_by_site(site_id)
        
        if not site_goals:
            raise HTTPException(
                status_code=404, 
                detail=f"No goals found for site '{site_id}'. Available sites: {ehs_goals_config.get_site_names()}"
            )
        
        # Convert to API models
        goal_models = [convert_goal_to_model(goal) for goal in site_goals]
        
        # Get available sites and categories for reference
        sites = ehs_goals_config.get_site_names()
        categories = ehs_goals_config.get_category_names()
        
        response = GoalsResponse(
            goals=goal_models,
            total_goals=len(goal_models),
            sites=[site_id],  # Only the requested site
            categories=categories
        )
        
        logger.info(f"Successfully retrieved {len(goal_models)} goals for site {site_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving goals for site {site_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve goals for site {site_id}")

@router.get("/progress/{site_id}", response_model=ProgressResponse)
async def get_site_progress(
    site_id: str = Path(..., description="Site identifier (algonquin_illinois or houston_texas)"),
    service = Depends(get_environmental_service)
):
    """
    Get progress against annual EHS goals for a specific site.
    
    Calculates actual progress by comparing current consumption data
    with baseline values and reduction targets. Requires environmental
    assessment service to retrieve actual consumption data.
    """
    try:
        logger.info(f"Calculating progress for site: {site_id}")
        
        # Get goals for the site
        site_goals = ehs_goals_config.get_goals_by_site(site_id)
        
        if not site_goals:
            raise HTTPException(
                status_code=404,
                detail=f"No goals found for site '{site_id}'. Available sites: {ehs_goals_config.get_site_names()}"
            )
        
        if service is None:
            logger.warning("Environmental assessment service not available - using mock progress data")
            # Return goals with insufficient data status
            site_progress = []
            for goal in site_goals:
                progress = ProgressMetrics(
                    baseline_value=None,
                    current_value=None,
                    target_value=None,
                    actual_reduction=None,
                    target_reduction=goal.reduction_percentage,
                    progress_percentage=None,
                    status="insufficient_data",
                    unit=goal.unit
                )
                site_progress.append(SiteProgressModel(
                    site=goal.site.value,
                    category=goal.category.value,
                    goal=convert_goal_to_model(goal),
                    progress=progress,
                    last_updated=datetime.now()
                ))
            
            return ProgressResponse(
                site_progress=site_progress,
                overall_status="insufficient_data",
                summary={"message": "Environmental assessment service unavailable"},
                calculation_date=datetime.now()
            )
        
        # Get location filter for service calls
        location_filter = get_site_location_filter(site_id)
        
        # Calculate date ranges
        current_year = datetime.now().year
        baseline_year = EHSGoalsConfig.BASELINE_YEAR
        
        baseline_start = f"{baseline_year}-01-01"
        baseline_end = f"{baseline_year}-12-31"
        current_start = f"{current_year}-01-01"
        current_end = datetime.now().strftime("%Y-%m-%d")
        
        site_progress = []
        statuses = []
        
        # Calculate progress for each goal
        for goal in site_goals:
            try:
                # Get service method for this category
                service_method_name = map_category_to_service_method(goal.category.value)
                
                if not service_method_name:
                    logger.warning(f"No service method found for category: {goal.category.value}")
                    continue
                
                service_method = getattr(service, service_method_name)
                
                # Get baseline data
                baseline_data = service_method(
                    location_filter=location_filter,
                    start_date=baseline_start,
                    end_date=baseline_end
                )
                
                # Get current data
                current_data = service_method(
                    location_filter=location_filter,
                    start_date=current_start,
                    end_date=current_end
                )
                
                # Calculate progress metrics
                progress = calculate_progress_metrics(goal, baseline_data, current_data)
                statuses.append(progress.status)
                
                site_progress.append(SiteProgressModel(
                    site=goal.site.value,
                    category=goal.category.value,
                    goal=convert_goal_to_model(goal),
                    progress=progress,
                    last_updated=datetime.now()
                ))
                
            except Exception as e:
                logger.warning(f"Error calculating progress for {goal.category.value}: {e}")
                # Add entry with error status
                progress = ProgressMetrics(
                    baseline_value=None,
                    current_value=None,
                    target_value=None,
                    actual_reduction=None,
                    target_reduction=goal.reduction_percentage,
                    progress_percentage=None,
                    status="calculation_error",
                    unit=goal.unit
                )
                site_progress.append(SiteProgressModel(
                    site=goal.site.value,
                    category=goal.category.value,
                    goal=convert_goal_to_model(goal),
                    progress=progress,
                    last_updated=datetime.now()
                ))
                statuses.append("calculation_error")
        
        # Determine overall status
        if all(status == "ahead" for status in statuses):
            overall_status = "ahead"
        elif all(status == "on_track" for status in statuses if status not in ["ahead"]):
            overall_status = "on_track"
        elif any(status == "behind" for status in statuses):
            overall_status = "behind"
        else:
            overall_status = "mixed"
        
        # Calculate summary statistics
        valid_progress = [sp for sp in site_progress if sp.progress.progress_percentage is not None]
        summary = {
            "total_goals": len(site_progress),
            "goals_with_data": len(valid_progress),
            "average_progress": sum(sp.progress.progress_percentage for sp in valid_progress) / len(valid_progress) if valid_progress else None,
            "on_track_count": len([s for s in statuses if s in ["on_track", "ahead"]]),
            "behind_count": len([s for s in statuses if s == "behind"]),
            "data_issues_count": len([s for s in statuses if s in ["insufficient_data", "calculation_error"]])
        }
        
        response = ProgressResponse(
            site_progress=site_progress,
            overall_status=overall_status,
            summary=summary,
            calculation_date=datetime.now()
        )
        
        logger.info(f"Successfully calculated progress for site {site_id}: {overall_status}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating progress for site {site_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate progress for site {site_id}")

@router.get("/summary", response_model=GoalsSummaryModel)
async def get_goals_summary():
    """
    Get summary of all EHS goals organized by site and category.
    
    Provides a high-level overview of reduction targets across
    all sites and categories for dashboard display.
    """
    try:
        logger.info("Retrieving EHS goals summary")
        
        # Get summary from configuration
        site_summary = get_goals_summary()
        
        # Organize by category for easy filtering
        category_summary = {}
        all_goals = get_all_goals()
        
        for goal in all_goals:
            category_key = goal.category.value
            if category_key not in category_summary:
                category_summary[category_key] = []
            
            category_summary[category_key].append({
                "site": goal.site.value,
                "reduction_percentage": goal.reduction_percentage,
                "unit": goal.unit,
                "description": goal.description
            })
        
        response = GoalsSummaryModel(
            site_summary=site_summary,
            category_summary=category_summary,
            total_sites=len(ehs_goals_config.get_site_names()),
            total_categories=len(ehs_goals_config.get_category_names())
        )
        
        logger.info("Successfully retrieved EHS goals summary")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving goals summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve goals summary")

# Health check endpoint
@router.get("/health")
async def goals_health_check():
    """Health check for EHS goals service"""
    try:
        # Validate configuration
        is_valid = ehs_goals_config.validate_configuration()
        
        return {
            "status": "healthy" if is_valid else "degraded",
            "configuration_valid": is_valid,
            "total_goals": len(get_all_goals()),
            "sites": ehs_goals_config.get_site_names(),
            "categories": ehs_goals_config.get_category_names(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }