"""
Enhanced Node Type Definitions for Neo4j EHS AI Platform

This module defines enhanced node types that extend the basic data model with
AI-driven analytics, goals, targets, trends, recommendations, and forecasting
capabilities as outlined in the Neo4j Data Population Plan.

Author: AI Assistant  
Created: 2025-08-28
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import uuid


class NodeValidationError(Exception):
    """Exception raised when node validation fails."""
    pass


class MetricType(Enum):
    """Enumeration of EHS metric types."""
    INCIDENT_RATE = "incident_rate"
    INJURY_RATE = "injury_rate"  
    LOST_TIME_RATE = "lost_time_rate"
    NEAR_MISS_COUNT = "near_miss_count"
    SAFETY_TRAINING_HOURS = "safety_training_hours"
    COMPLIANCE_SCORE = "compliance_score"
    AUDIT_SCORE = "audit_score"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    EMISSIONS = "emissions"
    WASTE_GENERATION = "waste_generation"
    ENERGY_CONSUMPTION = "energy_consumption"
    WATER_USAGE = "water_usage"
    REGULATORY_VIOLATIONS = "regulatory_violations"
    CORRECTIVE_ACTIONS = "corrective_actions"


class GoalLevel(Enum):
    """Enumeration of organizational goal levels."""
    CORPORATE = "corporate"
    FACILITY = "facility"  
    DEPARTMENT = "department"
    INDIVIDUAL = "individual"


class TrendDirection(Enum):
    """Enumeration of trend directions."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


class TrendConfidence(Enum):
    """Enumeration of trend confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationPriority(Enum):
    """Enumeration of recommendation priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationStatus(Enum):
    """Enumeration of recommendation statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ForecastModel(Enum):
    """Enumeration of forecasting model types."""
    LINEAR_REGRESSION = "linear_regression"
    TIME_SERIES_ARIMA = "time_series_arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class EnhancedHistoricalMetric:
    """
    Enhanced Historical Metric node extending basic metrics with AI-driven insights.
    
    This node type extends the basic HistoricalMetric with additional analytical
    capabilities, contextual information, and quality assessments.
    """
    
    # Core identification
    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_type: MetricType = MetricType.INCIDENT_RATE
    
    # Basic metric data (inherited from base HistoricalMetric)
    facility_name: str = ""
    department: Optional[str] = None
    metric_name: str = ""
    value: Decimal = Decimal('0')
    unit: str = ""
    reporting_period: date = field(default_factory=date.today)
    
    # Enhanced analytical fields
    normalized_value: Optional[Decimal] = None  # Value normalized for comparison
    percentile_rank: Optional[float] = None  # Percentile compared to historical data
    z_score: Optional[float] = None  # Statistical z-score
    seasonal_adjustment: Optional[Decimal] = None  # Seasonally adjusted value
    benchmark_comparison: Optional[Dict[str, Any]] = None  # Industry/peer benchmarks
    
    # Quality and reliability indicators
    data_quality_score: float = 1.0  # 0.0 to 1.0 scale
    confidence_interval: Optional[Dict[str, float]] = None  # Upper/lower bounds
    data_source_reliability: str = "high"  # high, medium, low
    collection_method: str = "automated"  # automated, manual, calculated
    validation_status: str = "validated"  # validated, pending, rejected
    
    # Contextual information
    business_context: Optional[str] = None  # Business context affecting the metric
    external_factors: List[str] = field(default_factory=list)  # External influencing factors
    regulatory_requirements: List[str] = field(default_factory=list)  # Applicable regulations
    
    # Temporal and trend information
    moving_average_30d: Optional[Decimal] = None
    moving_average_90d: Optional[Decimal] = None
    year_over_year_change: Optional[Decimal] = None
    month_over_month_change: Optional[Decimal] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    created_by: str = "system"
    
    def validate(self) -> bool:
        """Validate the enhanced historical metric data."""
        if not self.metric_id:
            raise NodeValidationError("metric_id is required")
        
        if not self.facility_name:
            raise NodeValidationError("facility_name is required")
            
        if not self.metric_name:
            raise NodeValidationError("metric_name is required")
            
        if self.data_quality_score < 0.0 or self.data_quality_score > 1.0:
            raise NodeValidationError("data_quality_score must be between 0.0 and 1.0")
            
        return True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j node creation."""
        data = {
            'metric_id': self.metric_id,
            'metric_type': self.metric_type.value,
            'facility_name': self.facility_name,
            'department': self.department,
            'metric_name': self.metric_name,
            'value': float(self.value),
            'unit': self.unit,
            'reporting_period': self.reporting_period.isoformat(),
            'data_quality_score': self.data_quality_score,
            'data_source_reliability': self.data_source_reliability,
            'collection_method': self.collection_method,
            'validation_status': self.validation_status,
            'external_factors': self.external_factors,
            'regulatory_requirements': self.regulatory_requirements,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by
        }
        
        # Add optional fields if they exist
        optional_fields = [
            'normalized_value', 'percentile_rank', 'z_score', 'seasonal_adjustment',
            'benchmark_comparison', 'confidence_interval', 'business_context',
            'moving_average_30d', 'moving_average_90d', 'year_over_year_change',
            'month_over_month_change', 'updated_at'
        ]
        
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, (datetime, date)):
                    data[field_name] = value.isoformat()
                elif isinstance(value, Decimal):
                    data[field_name] = float(value)
                else:
                    data[field_name] = value
        
        return data


@dataclass  
class Goal:
    """
    Goal node representing organizational EHS goals at different levels.
    
    Goals are hierarchical and can exist at corporate, facility, department,
    or individual levels with specific targets and timelines.
    """
    
    # Core identification
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_name: str = ""
    goal_level: GoalLevel = GoalLevel.FACILITY
    
    # Organizational context
    organization_unit: str = ""  # Corporate, facility name, department name, etc.
    parent_goal_id: Optional[str] = None  # Reference to parent goal
    
    # Goal definition
    description: str = ""
    success_criteria: List[str] = field(default_factory=list)
    metric_types: List[MetricType] = field(default_factory=list)  # Associated metric types
    
    # Timeline
    start_date: date = field(default_factory=date.today)
    target_date: date = field(default_factory=lambda: date.today().replace(year=date.today().year + 1))
    review_frequency: str = "monthly"  # daily, weekly, monthly, quarterly, annually
    
    # Status and progress
    status: str = "active"  # active, completed, cancelled, on_hold
    progress_percentage: float = 0.0  # 0.0 to 100.0
    last_review_date: Optional[date] = None
    next_review_date: Optional[date] = None
    
    # Ownership and accountability
    owner: str = ""  # Person responsible for the goal
    stakeholders: List[str] = field(default_factory=list)  # Other involved parties
    
    # Strategic alignment
    strategic_priority: str = "medium"  # critical, high, medium, low
    business_impact: str = ""  # Description of expected business impact
    regulatory_alignment: List[str] = field(default_factory=list)  # Regulatory frameworks
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    created_by: str = "system"
    
    def validate(self) -> bool:
        """Validate the goal data."""
        if not self.goal_id:
            raise NodeValidationError("goal_id is required")
            
        if not self.goal_name:
            raise NodeValidationError("goal_name is required")
            
        if not self.organization_unit:
            raise NodeValidationError("organization_unit is required")
            
        if self.progress_percentage < 0.0 or self.progress_percentage > 100.0:
            raise NodeValidationError("progress_percentage must be between 0.0 and 100.0")
            
        if self.target_date < self.start_date:
            raise NodeValidationError("target_date must be after start_date")
            
        return True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j node creation."""
        return {
            'goal_id': self.goal_id,
            'goal_name': self.goal_name,
            'goal_level': self.goal_level.value,
            'organization_unit': self.organization_unit,
            'parent_goal_id': self.parent_goal_id,
            'description': self.description,
            'success_criteria': self.success_criteria,
            'metric_types': [mt.value for mt in self.metric_types],
            'start_date': self.start_date.isoformat(),
            'target_date': self.target_date.isoformat(),
            'review_frequency': self.review_frequency,
            'status': self.status,
            'progress_percentage': self.progress_percentage,
            'last_review_date': self.last_review_date.isoformat() if self.last_review_date else None,
            'next_review_date': self.next_review_date.isoformat() if self.next_review_date else None,
            'owner': self.owner,
            'stakeholders': self.stakeholders,
            'strategic_priority': self.strategic_priority,
            'business_impact': self.business_impact,
            'regulatory_alignment': self.regulatory_alignment,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by
        }


@dataclass
class Target:
    """
    Target node representing specific measurable targets within goals.
    
    Targets are concrete, measurable objectives that support goal achievement
    with specific numeric values and deadlines.
    """
    
    # Core identification
    target_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_name: str = ""
    goal_id: str = ""  # Reference to parent goal
    
    # Target specification
    metric_type: MetricType = MetricType.INCIDENT_RATE
    target_value: Decimal = Decimal('0')
    current_value: Optional[Decimal] = None
    baseline_value: Optional[Decimal] = None
    unit: str = ""
    
    # Target type and direction
    target_type: str = "absolute"  # absolute, percentage_improvement, relative
    improvement_direction: str = "decrease"  # increase, decrease, maintain
    
    # Timeline
    target_date: date = field(default_factory=lambda: date.today().replace(year=date.today().year + 1))
    milestone_dates: List[Dict[str, Any]] = field(default_factory=list)  # Intermediate milestones
    
    # Progress tracking
    achievement_percentage: float = 0.0  # 0.0 to 100.0
    last_measured_date: Optional[date] = None
    measurement_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    
    # Performance indicators
    on_track_status: str = "unknown"  # on_track, at_risk, off_track, achieved
    variance_from_target: Optional[Decimal] = None
    trend_direction: Optional[TrendDirection] = None
    
    # Contextual information
    business_justification: str = ""
    success_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    # Ownership
    owner: str = ""
    reviewers: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    created_by: str = "system"
    
    def validate(self) -> bool:
        """Validate the target data."""
        if not self.target_id:
            raise NodeValidationError("target_id is required")
            
        if not self.target_name:
            raise NodeValidationError("target_name is required")
            
        if not self.goal_id:
            raise NodeValidationError("goal_id is required")
            
        if self.achievement_percentage < 0.0 or self.achievement_percentage > 100.0:
            raise NodeValidationError("achievement_percentage must be between 0.0 and 100.0")
            
        return True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j node creation."""
        return {
            'target_id': self.target_id,
            'target_name': self.target_name,
            'goal_id': self.goal_id,
            'metric_type': self.metric_type.value,
            'target_value': float(self.target_value),
            'current_value': float(self.current_value) if self.current_value else None,
            'baseline_value': float(self.baseline_value) if self.baseline_value else None,
            'unit': self.unit,
            'target_type': self.target_type,
            'improvement_direction': self.improvement_direction,
            'target_date': self.target_date.isoformat(),
            'milestone_dates': self.milestone_dates,
            'achievement_percentage': self.achievement_percentage,
            'last_measured_date': self.last_measured_date.isoformat() if self.last_measured_date else None,
            'measurement_frequency': self.measurement_frequency,
            'on_track_status': self.on_track_status,
            'variance_from_target': float(self.variance_from_target) if self.variance_from_target else None,
            'trend_direction': self.trend_direction.value if self.trend_direction else None,
            'business_justification': self.business_justification,
            'success_factors': self.success_factors,
            'risk_factors': self.risk_factors,
            'owner': self.owner,
            'reviewers': self.reviewers,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by
        }


@dataclass
class TrendAnalysis:
    """
    Trend Analysis node representing AI-generated insights about metric trends.
    
    Contains statistical analysis, pattern detection, and trend interpretation
    generated by AI algorithms from historical data.
    """
    
    # Core identification
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    analysis_name: str = ""
    
    # Scope of analysis
    metric_type: MetricType = MetricType.INCIDENT_RATE
    facility_name: Optional[str] = None
    department: Optional[str] = None
    analysis_period_start: date = field(default_factory=lambda: date.today() - timedelta(days=365))
    analysis_period_end: date = field(default_factory=date.today)
    
    # Trend characteristics
    trend_direction: TrendDirection = TrendDirection.STABLE
    trend_strength: float = 0.0  # -1.0 to 1.0 scale
    trend_confidence: TrendConfidence = TrendConfidence.MEDIUM
    statistical_significance: float = 0.0  # p-value
    
    # Statistical measures
    correlation_coefficient: Optional[float] = None
    r_squared: Optional[float] = None
    slope: Optional[float] = None  # Rate of change per time unit
    intercept: Optional[float] = None
    
    # Volatility and patterns
    volatility_measure: float = 0.0  # Standard deviation or coefficient of variation
    seasonality_detected: bool = False
    seasonal_patterns: List[str] = field(default_factory=list)  # monthly, quarterly, etc.
    cyclical_patterns: List[str] = field(default_factory=list)
    
    # Change points and anomalies
    change_points: List[Dict[str, Any]] = field(default_factory=list)  # Significant changes
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    outliers: List[Dict[str, Any]] = field(default_factory=list)
    
    # AI-generated insights
    key_insights: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    opportunity_indicators: List[str] = field(default_factory=list)
    
    # Model information
    analysis_model: str = "statistical_analysis"  # Model used for analysis
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    data_points_analyzed: int = 0
    
    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    created_by: str = "ai_system"
    version: str = "1.0"
    
    def validate(self) -> bool:
        """Validate the trend analysis data."""
        if not self.analysis_id:
            raise NodeValidationError("analysis_id is required")
            
        if not self.analysis_name:
            raise NodeValidationError("analysis_name is required")
            
        if self.trend_strength < -1.0 or self.trend_strength > 1.0:
            raise NodeValidationError("trend_strength must be between -1.0 and 1.0")
            
        if self.analysis_period_end < self.analysis_period_start:
            raise NodeValidationError("analysis_period_end must be after analysis_period_start")
            
        return True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j node creation."""
        return {
            'analysis_id': self.analysis_id,
            'analysis_name': self.analysis_name,
            'metric_type': self.metric_type.value,
            'facility_name': self.facility_name,
            'department': self.department,
            'analysis_period_start': self.analysis_period_start.isoformat(),
            'analysis_period_end': self.analysis_period_end.isoformat(),
            'trend_direction': self.trend_direction.value,
            'trend_strength': self.trend_strength,
            'trend_confidence': self.trend_confidence.value,
            'statistical_significance': self.statistical_significance,
            'correlation_coefficient': self.correlation_coefficient,
            'r_squared': self.r_squared,
            'slope': self.slope,
            'intercept': self.intercept,
            'volatility_measure': self.volatility_measure,
            'seasonality_detected': self.seasonality_detected,
            'seasonal_patterns': self.seasonal_patterns,
            'cyclical_patterns': self.cyclical_patterns,
            'change_points': self.change_points,
            'anomalies_detected': self.anomalies_detected,
            'outliers': self.outliers,
            'key_insights': self.key_insights,
            'contributing_factors': self.contributing_factors,
            'risk_indicators': self.risk_indicators,
            'opportunity_indicators': self.opportunity_indicators,
            'analysis_model': self.analysis_model,
            'model_parameters': self.model_parameters,
            'data_points_analyzed': self.data_points_analyzed,
            'analysis_date': self.analysis_date.isoformat(),
            'created_by': self.created_by,
            'version': self.version
        }


@dataclass
class Recommendation:
    """
    Recommendation node representing actionable AI-generated recommendations.
    
    Contains specific recommendations for improving EHS performance based on
    data analysis, trends, and best practices.
    """
    
    # Core identification
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recommendation_title: str = ""
    
    # Source and context
    source_analysis_id: Optional[str] = None  # Reference to TrendAnalysis
    metric_type: Optional[MetricType] = None
    facility_name: Optional[str] = None
    department: Optional[str] = None
    
    # Recommendation content
    description: str = ""
    detailed_action_plan: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    
    # Prioritization
    priority: RecommendationPriority = RecommendationPriority.MEDIUM
    urgency_level: str = "normal"  # critical, high, normal, low
    business_impact: str = "medium"  # high, medium, low
    implementation_difficulty: str = "medium"  # easy, medium, hard
    
    # Resource requirements
    estimated_cost: Optional[Decimal] = None
    cost_currency: str = "USD"
    estimated_effort_hours: Optional[int] = None
    required_skills: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    
    # Timeline
    recommended_start_date: Optional[date] = None
    estimated_completion_date: Optional[date] = None
    implementation_phases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status and tracking
    status: RecommendationStatus = RecommendationStatus.PENDING
    assigned_to: Optional[str] = None
    implementation_start_date: Optional[date] = None
    actual_completion_date: Optional[date] = None
    
    # Effectiveness tracking
    implementation_progress: float = 0.0  # 0.0 to 100.0
    effectiveness_rating: Optional[float] = None  # 1.0 to 5.0 scale
    lessons_learned: List[str] = field(default_factory=list)
    
    # AI model information
    generating_model: str = "recommendation_engine"
    confidence_score: float = 0.0  # 0.0 to 1.0
    supporting_evidence: List[str] = field(default_factory=list)
    
    # Stakeholder information
    primary_stakeholders: List[str] = field(default_factory=list)
    approval_required_from: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    created_by: str = "ai_system"
    
    def validate(self) -> bool:
        """Validate the recommendation data."""
        if not self.recommendation_id:
            raise NodeValidationError("recommendation_id is required")
            
        if not self.recommendation_title:
            raise NodeValidationError("recommendation_title is required")
            
        if not self.description:
            raise NodeValidationError("description is required")
            
        if self.implementation_progress < 0.0 or self.implementation_progress > 100.0:
            raise NodeValidationError("implementation_progress must be between 0.0 and 100.0")
            
        if self.confidence_score < 0.0 or self.confidence_score > 1.0:
            raise NodeValidationError("confidence_score must be between 0.0 and 1.0")
            
        return True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j node creation."""
        return {
            'recommendation_id': self.recommendation_id,
            'recommendation_title': self.recommendation_title,
            'source_analysis_id': self.source_analysis_id,
            'metric_type': self.metric_type.value if self.metric_type else None,
            'facility_name': self.facility_name,
            'department': self.department,
            'description': self.description,
            'detailed_action_plan': self.detailed_action_plan,
            'expected_outcomes': self.expected_outcomes,
            'success_metrics': self.success_metrics,
            'priority': self.priority.value,
            'urgency_level': self.urgency_level,
            'business_impact': self.business_impact,
            'implementation_difficulty': self.implementation_difficulty,
            'estimated_cost': float(self.estimated_cost) if self.estimated_cost else None,
            'cost_currency': self.cost_currency,
            'estimated_effort_hours': self.estimated_effort_hours,
            'required_skills': self.required_skills,
            'required_resources': self.required_resources,
            'recommended_start_date': self.recommended_start_date.isoformat() if self.recommended_start_date else None,
            'estimated_completion_date': self.estimated_completion_date.isoformat() if self.estimated_completion_date else None,
            'implementation_phases': self.implementation_phases,
            'status': self.status.value,
            'assigned_to': self.assigned_to,
            'implementation_start_date': self.implementation_start_date.isoformat() if self.implementation_start_date else None,
            'actual_completion_date': self.actual_completion_date.isoformat() if self.actual_completion_date else None,
            'implementation_progress': self.implementation_progress,
            'effectiveness_rating': self.effectiveness_rating,
            'lessons_learned': self.lessons_learned,
            'generating_model': self.generating_model,
            'confidence_score': self.confidence_score,
            'supporting_evidence': self.supporting_evidence,
            'primary_stakeholders': self.primary_stakeholders,
            'approval_required_from': self.approval_required_from,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by
        }


@dataclass
class Forecast:
    """
    Forecast node representing predictive analytics and future projections.
    
    Contains AI-generated forecasts for EHS metrics based on historical data,
    trends, and predictive models.
    """
    
    # Core identification
    forecast_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    forecast_name: str = ""
    
    # Forecast scope
    metric_type: MetricType = MetricType.INCIDENT_RATE
    facility_name: Optional[str] = None
    department: Optional[str] = None
    
    # Time horizon
    forecast_start_date: date = field(default_factory=date.today)
    forecast_end_date: date = field(default_factory=lambda: date.today() + timedelta(days=365))
    forecast_horizon_days: int = 365
    
    # Model information
    model_type: ForecastModel = ForecastModel.LINEAR_REGRESSION
    model_version: str = "1.0"
    training_data_start: date = field(default_factory=lambda: date.today() - timedelta(days=730))
    training_data_end: date = field(default_factory=date.today)
    training_data_points: int = 0
    
    # Model performance metrics
    model_accuracy: Optional[float] = None  # 0.0 to 1.0
    mean_absolute_error: Optional[float] = None
    root_mean_square_error: Optional[float] = None
    r_squared_score: Optional[float] = None
    cross_validation_score: Optional[float] = None
    
    # Forecast results
    predicted_values: List[Dict[str, Any]] = field(default_factory=list)  # Time series predictions
    confidence_intervals: List[Dict[str, Any]] = field(default_factory=list)  # Upper/lower bounds
    prediction_confidence: float = 0.0  # 0.0 to 1.0 overall confidence
    
    # Statistical measures
    forecast_trend: TrendDirection = TrendDirection.STABLE
    expected_volatility: float = 0.0
    seasonal_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Risk and scenario analysis
    best_case_scenario: Optional[Dict[str, Any]] = None
    worst_case_scenario: Optional[Dict[str, Any]] = None
    most_likely_scenario: Optional[Dict[str, Any]] = None
    risk_factors: List[str] = field(default_factory=list)
    
    # Business implications
    key_findings: List[str] = field(default_factory=list)
    business_implications: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    # Model parameters and assumptions
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Validation and monitoring
    last_validation_date: Optional[date] = None
    validation_accuracy: Optional[float] = None
    next_update_due: Optional[date] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    created_by: str = "ai_system"
    
    def validate(self) -> bool:
        """Validate the forecast data."""
        if not self.forecast_id:
            raise NodeValidationError("forecast_id is required")
            
        if not self.forecast_name:
            raise NodeValidationError("forecast_name is required")
            
        if self.forecast_end_date <= self.forecast_start_date:
            raise NodeValidationError("forecast_end_date must be after forecast_start_date")
            
        if self.prediction_confidence < 0.0 or self.prediction_confidence > 1.0:
            raise NodeValidationError("prediction_confidence must be between 0.0 and 1.0")
            
        if self.training_data_end < self.training_data_start:
            raise NodeValidationError("training_data_end must be after training_data_start")
            
        return True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j node creation."""
        return {
            'forecast_id': self.forecast_id,
            'forecast_name': self.forecast_name,
            'metric_type': self.metric_type.value,
            'facility_name': self.facility_name,
            'department': self.department,
            'forecast_start_date': self.forecast_start_date.isoformat(),
            'forecast_end_date': self.forecast_end_date.isoformat(),
            'forecast_horizon_days': self.forecast_horizon_days,
            'model_type': self.model_type.value,
            'model_version': self.model_version,
            'training_data_start': self.training_data_start.isoformat(),
            'training_data_end': self.training_data_end.isoformat(),
            'training_data_points': self.training_data_points,
            'model_accuracy': self.model_accuracy,
            'mean_absolute_error': self.mean_absolute_error,
            'root_mean_square_error': self.root_mean_square_error,
            'r_squared_score': self.r_squared_score,
            'cross_validation_score': self.cross_validation_score,
            'predicted_values': self.predicted_values,
            'confidence_intervals': self.confidence_intervals,
            'prediction_confidence': self.prediction_confidence,
            'forecast_trend': self.forecast_trend.value,
            'expected_volatility': self.expected_volatility,
            'seasonal_adjustments': self.seasonal_adjustments,
            'best_case_scenario': self.best_case_scenario,
            'worst_case_scenario': self.worst_case_scenario,
            'most_likely_scenario': self.most_likely_scenario,
            'risk_factors': self.risk_factors,
            'key_findings': self.key_findings,
            'business_implications': self.business_implications,
            'recommended_actions': self.recommended_actions,
            'model_parameters': self.model_parameters,
            'assumptions': self.assumptions,
            'limitations': self.limitations,
            'last_validation_date': self.last_validation_date.isoformat() if self.last_validation_date else None,
            'validation_accuracy': self.validation_accuracy,
            'next_update_due': self.next_update_due.isoformat() if self.next_update_due else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by
        }


@dataclass
class TrendPeriod:
    """
    Trend Period node representing time period definitions for trend analysis.
    
    Defines standard time periods used for trend analysis, reporting,
    and comparative analytics across the EHS system.
    """
    
    # Core identification
    period_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    period_name: str = ""
    period_type: str = ""  # daily, weekly, monthly, quarterly, annually, custom
    
    # Time boundaries
    start_date: date = field(default_factory=date.today)
    end_date: date = field(default_factory=date.today)
    duration_days: int = 0
    
    # Period characteristics
    is_standard_period: bool = True  # Standard vs custom period
    fiscal_period: bool = False  # Calendar vs fiscal period
    business_period: bool = False  # Standard business reporting period
    
    # Hierarchical relationships
    parent_period_id: Optional[str] = None  # Reference to parent period
    child_periods: List[str] = field(default_factory=list)  # Child period references
    
    # Period context
    description: str = ""
    business_context: str = ""  # Contextual information about the period
    key_events: List[str] = field(default_factory=list)  # Significant events in period
    
    # Usage tracking
    usage_count: int = 0  # How often this period is referenced
    associated_analyses: List[str] = field(default_factory=list)  # Analysis IDs using this period
    
    # Period status
    status: str = "active"  # active, archived, planned
    archived_date: Optional[date] = None
    
    # Comparison metrics
    comparable_periods: List[str] = field(default_factory=list)  # Similar periods for comparison
    baseline_period: bool = False  # Used as baseline for comparisons
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    created_by: str = "system"
    
    def validate(self) -> bool:
        """Validate the trend period data."""
        if not self.period_id:
            raise NodeValidationError("period_id is required")
            
        if not self.period_name:
            raise NodeValidationError("period_name is required")
            
        if not self.period_type:
            raise NodeValidationError("period_type is required")
            
        if self.end_date < self.start_date:
            raise NodeValidationError("end_date must be after start_date")
            
        # Calculate duration_days automatically if not set
        if self.duration_days == 0:
            self.duration_days = (self.end_date - self.start_date).days + 1
            
        return True
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j node creation."""
        return {
            'period_id': self.period_id,
            'period_name': self.period_name,
            'period_type': self.period_type,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'duration_days': self.duration_days,
            'is_standard_period': self.is_standard_period,
            'fiscal_period': self.fiscal_period,
            'business_period': self.business_period,
            'parent_period_id': self.parent_period_id,
            'child_periods': self.child_periods,
            'description': self.description,
            'business_context': self.business_context,
            'key_events': self.key_events,
            'usage_count': self.usage_count,
            'associated_analyses': self.associated_analyses,
            'status': self.status,
            'archived_date': self.archived_date.isoformat() if self.archived_date else None,
            'comparable_periods': self.comparable_periods,
            'baseline_period': self.baseline_period,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by
        }


# Factory functions for creating nodes with defaults

def create_enhanced_historical_metric(
    facility_name: str,
    metric_name: str,
    value: Union[float, Decimal],
    metric_type: MetricType = MetricType.INCIDENT_RATE,
    **kwargs
) -> EnhancedHistoricalMetric:
    """Factory function to create an EnhancedHistoricalMetric with sensible defaults."""
    return EnhancedHistoricalMetric(
        facility_name=facility_name,
        metric_name=metric_name,
        value=Decimal(str(value)),
        metric_type=metric_type,
        **kwargs
    )


def create_goal(
    goal_name: str,
    organization_unit: str,
    goal_level: GoalLevel = GoalLevel.FACILITY,
    **kwargs
) -> Goal:
    """Factory function to create a Goal with sensible defaults."""
    return Goal(
        goal_name=goal_name,
        organization_unit=organization_unit,
        goal_level=goal_level,
        **kwargs
    )


def create_target(
    target_name: str,
    goal_id: str,
    target_value: Union[float, Decimal],
    metric_type: MetricType = MetricType.INCIDENT_RATE,
    **kwargs
) -> Target:
    """Factory function to create a Target with sensible defaults."""
    return Target(
        target_name=target_name,
        goal_id=goal_id,
        target_value=Decimal(str(target_value)),
        metric_type=metric_type,
        **kwargs
    )


def create_trend_analysis(
    analysis_name: str,
    metric_type: MetricType = MetricType.INCIDENT_RATE,
    **kwargs
) -> TrendAnalysis:
    """Factory function to create a TrendAnalysis with sensible defaults."""
    return TrendAnalysis(
        analysis_name=analysis_name,
        metric_type=metric_type,
        **kwargs
    )


def create_recommendation(
    recommendation_title: str,
    description: str,
    **kwargs
) -> Recommendation:
    """Factory function to create a Recommendation with sensible defaults."""
    return Recommendation(
        recommendation_title=recommendation_title,
        description=description,
        **kwargs
    )


def create_forecast(
    forecast_name: str,
    metric_type: MetricType = MetricType.INCIDENT_RATE,
    **kwargs
) -> Forecast:
    """Factory function to create a Forecast with sensible defaults."""
    return Forecast(
        forecast_name=forecast_name,
        metric_type=metric_type,
        **kwargs
    )


def create_trend_period(
    period_name: str,
    period_type: str,
    start_date: date,
    end_date: date,
    **kwargs
) -> TrendPeriod:
    """Factory function to create a TrendPeriod with sensible defaults."""
    return TrendPeriod(
        period_name=period_name,
        period_type=period_type,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


# Node type registry for dynamic access
NODE_TYPES = {
    'EnhancedHistoricalMetric': EnhancedHistoricalMetric,
    'Goal': Goal,
    'Target': Target,
    'TrendAnalysis': TrendAnalysis,
    'Recommendation': Recommendation,
    'Forecast': Forecast,
    'TrendPeriod': TrendPeriod
}

# Factory function registry
NODE_FACTORIES = {
    'EnhancedHistoricalMetric': create_enhanced_historical_metric,
    'Goal': create_goal,
    'Target': create_target,
    'TrendAnalysis': create_trend_analysis,
    'Recommendation': create_recommendation,
    'Forecast': create_forecast,
    'TrendPeriod': create_trend_period
}


def get_node_type(node_type_name: str):
    """Get a node type class by name."""
    return NODE_TYPES.get(node_type_name)


def create_node(node_type_name: str, **kwargs):
    """Create a node instance using the appropriate factory function."""
    factory = NODE_FACTORIES.get(node_type_name)
    if factory:
        return factory(**kwargs)
    else:
        node_class = NODE_TYPES.get(node_type_name)
        if node_class:
            return node_class(**kwargs)
        else:
            raise ValueError(f"Unknown node type: {node_type_name}")


# Validation utilities

def validate_all_nodes(*nodes) -> List[str]:
    """Validate multiple nodes and return list of validation errors."""
    errors = []
    for i, node in enumerate(nodes):
        try:
            node.validate()
        except NodeValidationError as e:
            errors.append(f"Node {i} ({type(node).__name__}): {str(e)}")
    return errors


def batch_validate_nodes(nodes: List[Any]) -> Dict[str, List[str]]:
    """Validate a batch of nodes and return organized error report."""
    errors_by_type = {}
    for node in nodes:
        node_type = type(node).__name__
        if node_type not in errors_by_type:
            errors_by_type[node_type] = []
        
        try:
            node.validate()
        except NodeValidationError as e:
            errors_by_type[node_type].append(str(e))
    
    # Remove empty error lists
    return {k: v for k, v in errors_by_type.items() if v}