"""
Environmental Assessment API Router

This module provides FastAPI endpoints for retrieving environmental assessment data
including facts, risks, and recommendations for electricity, water, and waste.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import logging
import uuid

from services.environmental_assessment_service import EnvironmentalAssessmentService
from database.neo4j_client import Neo4jClient, ConnectionConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/environmental", tags=["environmental-assessment"])

# Pydantic Models
class DateRangeModel(BaseModel):
    """Date range model for filtering data"""
    start_date: datetime
    end_date: datetime

class FactModel(BaseModel):
    """Model for environmental facts"""
    id: str = Field(..., description="Unique identifier for the fact")
    category: str = Field(..., description="Category (electricity, water, waste)")
    title: str = Field(..., description="Fact title")
    description: str = Field(..., description="Detailed fact description")
    value: Optional[float] = Field(None, description="Numeric value if applicable")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    location_path: Optional[str] = Field(None, description="Location path")
    timestamp: datetime = Field(..., description="When the fact was recorded")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class RiskModel(BaseModel):
    """Model for environmental risks"""
    id: str = Field(..., description="Unique identifier for the risk")
    category: str = Field(..., description="Category (electricity, water, waste)")
    title: str = Field(..., description="Risk title")
    description: str = Field(..., description="Detailed risk description")
    severity: str = Field(..., description="Risk severity level")
    probability: str = Field(..., description="Risk probability")
    impact: str = Field(..., description="Potential impact")
    location_path: Optional[str] = Field(None, description="Location path")
    identified_date: datetime = Field(..., description="When the risk was identified")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class RecommendationModel(BaseModel):
    """Model for environmental recommendations"""
    id: str = Field(..., description="Unique identifier for the recommendation")
    category: str = Field(..., description="Category (electricity, water, waste)")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed recommendation description")
    priority: str = Field(..., description="Priority level")
    effort_level: str = Field(..., description="Implementation effort required")
    potential_impact: str = Field(..., description="Expected positive impact")
    location_path: Optional[str] = Field(None, description="Location path")
    created_date: datetime = Field(..., description="When the recommendation was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class LLMAssessmentRequest(BaseModel):
    """Model for LLM assessment requests"""
    location_path: Optional[str] = Field(None, description="Specific location to assess")
    categories: List[str] = Field(default=["electricity", "water", "waste"], description="Categories to assess")
    date_range: Optional[DateRangeModel] = Field(None, description="Date range for assessment")
    custom_prompt: Optional[str] = Field(None, description="Custom assessment prompt")

class LLMAssessmentResponse(BaseModel):
    """Model for LLM assessment responses"""
    assessment_id: str = Field(..., description="Unique assessment identifier")
    status: str = Field(..., description="Assessment status")
    facts: List[FactModel] = Field(default_factory=list, description="Generated facts")
    risks: List[RiskModel] = Field(default_factory=list, description="Identified risks")
    recommendations: List[RecommendationModel] = Field(default_factory=list, description="Generated recommendations")
    summary: str = Field(..., description="Assessment summary")
    created_at: datetime = Field(default_factory=datetime.now, description="Assessment creation time")

# Dependency to get service instance
async def get_service():
    """Get service instance with Neo4j client initialization"""
    try:
        # Create Neo4j client configuration from environment
        config = ConnectionConfig.from_env()
        
        # Create Neo4j client instance
        neo4j_client = Neo4jClient(config=config, enable_logging=True)
        
        # Test the connection
        if neo4j_client.connect():
            logger.info("Successfully connected to Neo4j database")
            # Return the service with connected Neo4j client
            return EnvironmentalAssessmentService(neo4j_client)
        else:
            logger.warning("Failed to connect to Neo4j database - service will operate in degraded mode")
            # Return None to enable graceful degradation
            return None
            
    except ImportError as e:
        logger.warning(f"Neo4j dependencies not available: {e} - service will operate in degraded mode")
        return None
    except Exception as e:
        logger.error(f"Unexpected error initializing Neo4j client: {e} - service will operate in degraded mode")
        return None

# Data conversion functions
def convert_facts_to_api_models(facts_data: List[Dict[str, Any]], category: str) -> List[FactModel]:
    """Convert service fact dictionaries to API FactModel objects"""
    api_facts = []
    
    for fact_dict in facts_data:
        try:
            api_fact = FactModel(
                id=fact_dict.get('id', str(uuid.uuid4())),
                category=category,
                title=fact_dict.get('title', f"{category.title()} Fact"),
                description=fact_dict.get('description', 'Generated fact from environmental data'),
                value=fact_dict.get('value'),
                unit=fact_dict.get('unit'),
                location_path=fact_dict.get('location_path'),
                timestamp=datetime.fromisoformat(fact_dict['timestamp']) if isinstance(fact_dict.get('timestamp'), str) else fact_dict.get('timestamp', datetime.now()),
                metadata=fact_dict.get('metadata', {})
            )
            api_facts.append(api_fact)
        except Exception as e:
            logger.warning(f"Failed to convert fact to API model: {e}")
            
    return api_facts

def convert_risks_to_api_models(risks_data: List[Dict[str, Any]], category: str) -> List[RiskModel]:
    """Convert service risk dictionaries to API RiskModel objects"""
    api_risks = []
    
    for risk_dict in risks_data:
        try:
            api_risk = RiskModel(
                id=risk_dict.get('id', str(uuid.uuid4())),
                category=category,
                title=risk_dict.get('title', risk_dict.get('type', f"{category.title()} Risk")),
                description=risk_dict.get('description', 'Risk identified in environmental data'),
                severity=risk_dict.get('severity', 'medium'),
                probability=risk_dict.get('probability', 'unknown'),
                impact=risk_dict.get('impact', risk_dict.get('recommendation', 'Impact assessment pending')),
                location_path=risk_dict.get('location_path'),
                identified_date=datetime.fromisoformat(risk_dict['identified_date']) if isinstance(risk_dict.get('identified_date'), str) else risk_dict.get('identified_date', datetime.now()),
                metadata=risk_dict.get('metadata', {})
            )
            api_risks.append(api_risk)
        except Exception as e:
            logger.warning(f"Failed to convert risk to API model: {e}")
            
    return api_risks

def convert_recommendations_to_api_models(recommendations_data: List, category: str) -> List[RecommendationModel]:
    """Convert service recommendation data to API RecommendationModel objects"""
    api_recommendations = []
    
    for rec_item in recommendations_data:
        try:
            # Handle both string and dictionary formats
            if isinstance(rec_item, str):
                rec_dict = {
                    'title': rec_item[:50] + "..." if len(rec_item) > 50 else rec_item,
                    'description': rec_item,
                    'priority': 'medium',
                    'effort_level': 'medium'
                }
            else:
                rec_dict = rec_item
            
            api_rec = RecommendationModel(
                id=rec_dict.get('id', str(uuid.uuid4())),
                category=category,
                title=rec_dict.get('title', f"{category.title()} Recommendation"),
                description=rec_dict.get('description', 'Recommended action for environmental improvement'),
                priority=rec_dict.get('priority', 'medium'),
                effort_level=rec_dict.get('effort_level', 'medium'),
                potential_impact=rec_dict.get('potential_impact', 'Positive environmental impact expected'),
                location_path=rec_dict.get('location_path'),
                created_date=datetime.fromisoformat(rec_dict['created_date']) if isinstance(rec_dict.get('created_date'), str) else rec_dict.get('created_date', datetime.now()),
                metadata=rec_dict.get('metadata', {})
            )
            api_recommendations.append(api_rec)
        except Exception as e:
            logger.warning(f"Failed to convert recommendation to API model: {e}")
            
    return api_recommendations

def convert_service_facts_to_api_facts(facts_dict: Dict[str, Any], category: str) -> List[FactModel]:
    """Convert service facts dictionary to API fact models"""
    api_facts = []
    
    # Convert key metrics to individual facts
    for key, value in facts_dict.items():
        if key in ['total_consumption', 'average_consumption', 'total_cost', 'average_efficiency', 'total_generated', 'total_recycled', 'recycling_rate']:
            try:
                # Determine appropriate title and unit
                title_map = {
                    'total_consumption': f'Total {category.title()} Consumption',
                    'average_consumption': f'Average {category.title()} Consumption',
                    'total_cost': f'Total {category.title()} Cost',
                    'average_efficiency': f'Average {category.title()} Efficiency',
                    'total_generated': f'Total {category.title()} Generated',
                    'total_recycled': f'Total {category.title()} Recycled',
                    'recycling_rate': f'{category.title()} Recycling Rate'
                }
                
                unit_map = {
                    'total_consumption': 'kWh' if category == 'electricity' else 'gallons' if category == 'water' else 'lbs',
                    'average_consumption': 'kWh' if category == 'electricity' else 'gallons' if category == 'water' else 'lbs',
                    'total_cost': 'USD',
                    'average_efficiency': 'ratio',
                    'total_generated': 'lbs',
                    'total_recycled': 'lbs',
                    'recycling_rate': 'percentage'
                }
                
                fact = FactModel(
                    id=str(uuid.uuid4()),
                    category=category,
                    title=title_map.get(key, key.replace('_', ' ').title()),
                    description=f"Calculated {key.replace('_', ' ')} for {category}",
                    value=float(value) if isinstance(value, (int, float)) else None,
                    unit=unit_map.get(key),
                    location_path=None,
                    timestamp=datetime.now(),
                    metadata={'source': 'calculated', 'metric_type': key}
                )
                api_facts.append(fact)
            except Exception as e:
                logger.warning(f"Failed to convert fact {key}: {e}")
                
    return api_facts

# Utility function to validate category
def validate_category(category: str) -> str:
    """Validate environmental category"""
    valid_categories = ["electricity", "water", "waste"]
    if category.lower() not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Must be one of: {valid_categories}"
        )
    return category.lower()

# Helper function to convert datetime to string for service calls
def datetime_to_str(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to string format expected by service"""
    return dt.strftime("%Y-%m-%d") if dt else None

# Electricity endpoints
@router.get("/electricity/facts", response_model=List[FactModel])
async def get_electricity_facts(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get electricity consumption facts"""
    try:
        logger.info("Retrieving electricity facts")
        
        if service is None:
            # Return empty list until service is properly initialized
            logger.warning("Service not initialized - returning empty facts list")
            return []
        
        # Call service with date conversion
        result = service.assess_electricity_consumption(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        # Convert service facts to API models
        facts = result.get('facts', {})
        return convert_service_facts_to_api_facts(facts, 'electricity')
        
    except Exception as e:
        logger.error(f"Error retrieving electricity facts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve electricity facts")

@router.get("/electricity/risks", response_model=List[RiskModel])
async def get_electricity_risks(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get electricity-related risks"""
    try:
        logger.info("Retrieving electricity risks")
        
        if service is None:
            logger.warning("Service not initialized - returning empty risks list")
            return []
        
        result = service.assess_electricity_consumption(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        risks = result.get('risks', [])
        return convert_risks_to_api_models(risks, 'electricity')
        
    except Exception as e:
        logger.error(f"Error retrieving electricity risks: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve electricity risks")

@router.get("/electricity/recommendations", response_model=List[RecommendationModel])
async def get_electricity_recommendations(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get electricity recommendations"""
    try:
        logger.info("Retrieving electricity recommendations")
        
        if service is None:
            logger.warning("Service not initialized - returning empty recommendations list")
            return []
        
        result = service.assess_electricity_consumption(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        recommendations = result.get('recommendations', [])
        return convert_recommendations_to_api_models(recommendations, 'electricity')
        
    except Exception as e:
        logger.error(f"Error retrieving electricity recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve electricity recommendations")

# Water endpoints
@router.get("/water/facts", response_model=List[FactModel])
async def get_water_facts(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get water consumption facts"""
    try:
        logger.info("Retrieving water facts")
        
        if service is None:
            logger.warning("Service not initialized - returning empty facts list")
            return []
        
        result = service.assess_water_consumption(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        facts = result.get('facts', {})
        return convert_service_facts_to_api_facts(facts, 'water')
        
    except Exception as e:
        logger.error(f"Error retrieving water facts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve water facts")

@router.get("/water/risks", response_model=List[RiskModel])
async def get_water_risks(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get water-related risks"""
    try:
        logger.info("Retrieving water risks")
        
        if service is None:
            logger.warning("Service not initialized - returning empty risks list")
            return []
        
        result = service.assess_water_consumption(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        risks = result.get('risks', [])
        return convert_risks_to_api_models(risks, 'water')
        
    except Exception as e:
        logger.error(f"Error retrieving water risks: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve water risks")

@router.get("/water/recommendations", response_model=List[RecommendationModel])
async def get_water_recommendations(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get water recommendations"""
    try:
        logger.info("Retrieving water recommendations")
        
        if service is None:
            logger.warning("Service not initialized - returning empty recommendations list")
            return []
        
        result = service.assess_water_consumption(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        recommendations = result.get('recommendations', [])
        return convert_recommendations_to_api_models(recommendations, 'water')
        
    except Exception as e:
        logger.error(f"Error retrieving water recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve water recommendations")

# Waste endpoints
@router.get("/waste/facts", response_model=List[FactModel])
async def get_waste_facts(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get waste generation facts"""
    try:
        logger.info("Retrieving waste facts")
        
        if service is None:
            logger.warning("Service not initialized - returning empty facts list")
            return []
        
        result = service.assess_waste_generation(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        facts = result.get('facts', {})
        return convert_service_facts_to_api_facts(facts, 'waste')
        
    except Exception as e:
        logger.error(f"Error retrieving waste facts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve waste facts")

@router.get("/waste/risks", response_model=List[RiskModel])
async def get_waste_risks(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get waste-related risks"""
    try:
        logger.info("Retrieving waste risks")
        
        if service is None:
            logger.warning("Service not initialized - returning empty risks list")
            return []
        
        result = service.assess_waste_generation(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        risks = result.get('risks', [])
        return convert_risks_to_api_models(risks, 'waste')
        
    except Exception as e:
        logger.error(f"Error retrieving waste risks: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve waste risks")

@router.get("/waste/recommendations", response_model=List[RecommendationModel])
async def get_waste_recommendations(
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get waste recommendations"""
    try:
        logger.info("Retrieving waste recommendations")
        
        if service is None:
            logger.warning("Service not initialized - returning empty recommendations list")
            return []
        
        result = service.assess_waste_generation(
            location_filter=location_path,
            start_date=datetime_to_str(start_date),
            end_date=datetime_to_str(end_date)
        )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        recommendations = result.get('recommendations', [])
        return convert_recommendations_to_api_models(recommendations, 'waste')
        
    except Exception as e:
        logger.error(f"Error retrieving waste recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve waste recommendations")

# LLM Assessment endpoint (Phase 2 placeholder)
@router.post("/llm-assessment", response_model=LLMAssessmentResponse)
async def trigger_llm_assessment(
    request: LLMAssessmentRequest,
    service = Depends(get_service)
):
    """Trigger LLM-based environmental assessment - Phase 2 functionality"""
    try:
        # Generate unique assessment ID
        assessment_id = str(uuid.uuid4())
        
        logger.info(f"Triggering LLM assessment {assessment_id} for categories: {request.categories}")
        
        # Phase 2 placeholder - this will integrate with LLM service
        # For now, return a placeholder response indicating Phase 2 implementation pending
        response = LLMAssessmentResponse(
            assessment_id=assessment_id,
            status="pending",
            facts=[],
            risks=[],
            recommendations=[],
            summary="LLM assessment functionality is planned for Phase 2. This endpoint will provide AI-generated environmental insights once LLM integration is completed.",
            created_at=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error triggering LLM assessment: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger LLM assessment")

# Generic endpoints for all categories
@router.get("/{category}/facts", response_model=List[FactModel])
async def get_category_facts(
    category: str,
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get facts for any environmental category"""
    category = validate_category(category)
    
    try:
        logger.info(f"Retrieving {category} facts")
        
        if service is None:
            logger.warning("Service not initialized - returning empty facts list")
            return []
        
        # Call appropriate service method based on category
        if category == 'electricity':
            result = service.assess_electricity_consumption(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        elif category == 'water':
            result = service.assess_water_consumption(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        elif category == 'waste':
            result = service.assess_waste_generation(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        facts = result.get('facts', {})
        return convert_service_facts_to_api_facts(facts, category)
        
    except Exception as e:
        logger.error(f"Error retrieving {category} facts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve {category} facts")

@router.get("/{category}/risks", response_model=List[RiskModel])
async def get_category_risks(
    category: str,
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get risks for any environmental category"""
    category = validate_category(category)
    
    try:
        logger.info(f"Retrieving {category} risks")
        
        if service is None:
            logger.warning("Service not initialized - returning empty risks list")
            return []
        
        # Call appropriate service method based on category
        if category == 'electricity':
            result = service.assess_electricity_consumption(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        elif category == 'water':
            result = service.assess_water_consumption(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        elif category == 'waste':
            result = service.assess_waste_generation(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        risks = result.get('risks', [])
        return convert_risks_to_api_models(risks, category)
        
    except Exception as e:
        logger.error(f"Error retrieving {category} risks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve {category} risks")

@router.get("/{category}/recommendations", response_model=List[RecommendationModel])
async def get_category_recommendations(
    category: str,
    location_path: Optional[str] = Query(None, description="Filter by location path"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    service = Depends(get_service)
):
    """Get recommendations for any environmental category"""
    category = validate_category(category)
    
    try:
        logger.info(f"Retrieving {category} recommendations")
        
        if service is None:
            logger.warning("Service not initialized - returning empty recommendations list")
            return []
        
        # Call appropriate service method based on category
        if category == 'electricity':
            result = service.assess_electricity_consumption(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        elif category == 'water':
            result = service.assess_water_consumption(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        elif category == 'waste':
            result = service.assess_waste_generation(
                location_filter=location_path,
                start_date=datetime_to_str(start_date),
                end_date=datetime_to_str(end_date)
            )
        
        if 'error' in result:
            logger.warning(f"Service returned error: {result['error']}")
            return []
        
        recommendations = result.get('recommendations', [])
        return convert_recommendations_to_api_models(recommendations, category)
        
    except Exception as e:
        logger.error(f"Error retrieving {category} recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve {category} recommendations")