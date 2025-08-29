"""
Risk Assessment Agent State Model

This module defines comprehensive state management for the EHS risk assessment agent,
including risk categories, severity levels, assessment context, and integration
with document and facility data.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, validator


# Risk Categories and Classifications
class RiskCategory(str, Enum):
    """Primary risk categories for EHS assessment."""
    ENVIRONMENTAL = "environmental"
    HEALTH_SAFETY = "health_safety"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    REPUTATIONAL = "reputational"
    STRATEGIC = "strategic"


class EnvironmentalRiskType(str, Enum):
    """Environmental-specific risk types."""
    AIR_EMISSIONS = "air_emissions"
    WATER_POLLUTION = "water_pollution"
    SOIL_CONTAMINATION = "soil_contamination"
    WASTE_MANAGEMENT = "waste_management"
    RESOURCE_DEPLETION = "resource_depletion"
    BIODIVERSITY_IMPACT = "biodiversity_impact"
    CLIMATE_CHANGE = "climate_change"
    NOISE_POLLUTION = "noise_pollution"
    CHEMICAL_SPILL = "chemical_spill"
    ECOSYSTEM_DISRUPTION = "ecosystem_disruption"


class HealthSafetyRiskType(str, Enum):
    """Health and safety specific risk types."""
    WORKPLACE_INJURY = "workplace_injury"
    OCCUPATIONAL_ILLNESS = "occupational_illness"
    FIRE_EXPLOSION = "fire_explosion"
    CHEMICAL_EXPOSURE = "chemical_exposure"
    ERGONOMIC_HAZARD = "ergonomic_hazard"
    BIOLOGICAL_HAZARD = "biological_hazard"
    PHYSICAL_HAZARD = "physical_hazard"
    PSYCHOLOGICAL_HAZARD = "psychological_hazard"
    EMERGENCY_RESPONSE = "emergency_response"
    EQUIPMENT_FAILURE = "equipment_failure"


class ComplianceRiskType(str, Enum):
    """Regulatory compliance risk types."""
    PERMIT_VIOLATION = "permit_violation"
    REPORTING_FAILURE = "reporting_failure"
    AUDIT_FINDINGS = "audit_findings"
    REGULATORY_CHANGE = "regulatory_change"
    ENFORCEMENT_ACTION = "enforcement_action"
    CERTIFICATION_LAPSE = "certification_lapse"
    INSPECTION_FAILURE = "inspection_failure"
    DOCUMENTATION_GAP = "documentation_gap"


class RiskSeverity(str, Enum):
    """Risk severity levels based on potential impact."""
    CRITICAL = "critical"      # Immediate threat to life, environment, or operations
    HIGH = "high"             # Significant impact with serious consequences
    MEDIUM = "medium"         # Moderate impact with manageable consequences
    LOW = "low"              # Minor impact with minimal consequences
    NEGLIGIBLE = "negligible" # Minimal or no significant impact


class RiskProbability(str, Enum):
    """Risk probability levels."""
    VERY_HIGH = "very_high"   # >80% likelihood
    HIGH = "high"             # 60-80% likelihood
    MEDIUM = "medium"         # 30-60% likelihood
    LOW = "low"              # 10-30% likelihood
    VERY_LOW = "very_low"    # <10% likelihood


class RiskTimeframe(str, Enum):
    """Timeframe for risk materialization."""
    IMMEDIATE = "immediate"    # 0-30 days
    SHORT_TERM = "short_term" # 1-6 months
    MEDIUM_TERM = "medium_term" # 6-24 months
    LONG_TERM = "long_term"   # 2+ years


class AssessmentStatus(str, Enum):
    """Risk assessment status."""
    INITIATED = "initiated"
    ANALYZING = "analyzing"
    REVIEWING = "reviewing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    REQUIRES_UPDATE = "requires_update"
    ESCALATED = "escalated"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    IMMEDIATE = "immediate"    # Implement within 24 hours
    URGENT = "urgent"         # Implement within 1 week
    HIGH = "high"            # Implement within 1 month
    MEDIUM = "medium"        # Implement within 3 months
    LOW = "low"             # Implement within 6 months
    DEFERRED = "deferred"    # Beyond 6 months or conditional


# Risk Assessment Context Models
class DocumentContext(BaseModel):
    """Context from analyzed documents."""
    document_id: str
    document_type: str  # permit, report, manifest, etc.
    source_file: str
    extraction_timestamp: datetime
    relevant_sections: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    regulatory_references: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)


class FacilityContext(BaseModel):
    """Context from facility data."""
    facility_id: str
    facility_name: str
    location: Dict[str, Any]  # Address, coordinates, etc.
    facility_type: str
    operational_status: str
    permits: List[Dict[str, Any]] = Field(default_factory=list)
    historical_incidents: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_history: List[Dict[str, Any]] = Field(default_factory=list)
    environmental_conditions: Dict[str, Any] = Field(default_factory=dict)


class RiskFactor(BaseModel):
    """Individual risk factor identification."""
    factor_id: str = Field(default_factory=lambda: str(uuid4()))
    factor_type: str
    description: str
    source_evidence: List[str] = Field(default_factory=list)
    quantitative_data: Optional[Dict[str, Union[float, int]]] = None
    confidence_level: float = Field(ge=0.0, le=1.0)
    contributing_documents: List[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Complete risk assessment result."""
    risk_id: str = Field(default_factory=lambda: str(uuid4()))
    category: RiskCategory
    subcategory: Union[EnvironmentalRiskType, HealthSafetyRiskType, ComplianceRiskType, str]
    title: str
    description: str
    
    # Risk scoring
    severity: RiskSeverity
    probability: RiskProbability
    risk_score: float = Field(ge=0.0, le=100.0)
    timeframe: RiskTimeframe
    
    # Evidence and analysis
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    
    # Context
    affected_facilities: List[str] = Field(default_factory=list)
    regulatory_context: List[str] = Field(default_factory=list)
    business_impact: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis metadata
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    analyst_id: Optional[str] = None
    review_required: bool = True
    confidence_score: float = Field(ge=0.0, le=1.0)


class RiskMitigationRecommendation(BaseModel):
    """Risk mitigation recommendation."""
    recommendation_id: str = Field(default_factory=lambda: str(uuid4()))
    risk_id: str  # Links to associated risk
    title: str
    description: str
    
    # Implementation details
    priority: RecommendationPriority
    implementation_timeframe: str
    estimated_cost: Optional[float] = None
    resource_requirements: List[str] = Field(default_factory=list)
    responsible_parties: List[str] = Field(default_factory=list)
    
    # Effectiveness
    risk_reduction_potential: float = Field(ge=0.0, le=100.0)  # Percentage
    implementation_complexity: str = Field(default="medium")  # low, medium, high
    regulatory_compliance_impact: List[str] = Field(default_factory=list)
    
    # Dependencies
    prerequisite_actions: List[str] = Field(default_factory=list)
    related_recommendations: List[str] = Field(default_factory=list)


class LangSmithTrace(BaseModel):
    """LangSmith tracing metadata."""
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_cost: Optional[float] = None


# Main State Definition
class RiskAssessmentState(TypedDict):
    """
    Main state for the Risk Assessment Agent.
    
    This TypedDict defines the complete state structure for managing
    risk assessment workflows, including input processing, analysis,
    and recommendation generation.
    """
    
    # Request Configuration
    assessment_id: str  # Unique identifier for this assessment
    request_type: Literal["facility_assessment", "document_assessment", "incident_assessment", "comprehensive_assessment"]
    assessment_scope: Dict[str, Any]  # Defines what to assess
    priority_level: str  # urgent, high, medium, low
    
    # Input Data
    document_contexts: List[DocumentContext]
    facility_contexts: List[FacilityContext]
    additional_data: Dict[str, Any]  # Flexible for various data types
    assessment_parameters: Dict[str, Any]  # Analysis parameters and thresholds
    
    # Analysis State
    identified_risks: List[RiskAssessment]
    risk_correlations: Dict[str, List[str]]  # Risk ID -> List of correlated risk IDs
    risk_trends: Dict[str, Dict[str, Any]]  # Historical trend analysis
    aggregate_risk_profile: Dict[str, Any]  # Overall risk summary
    
    # Recommendations
    recommendations: List[RiskMitigationRecommendation]
    prioritized_action_plan: List[Dict[str, Any]]
    implementation_roadmap: Dict[str, Any]
    cost_benefit_analysis: Dict[str, Any]
    
    # Processing State
    current_step: str  # Which processing step we're in
    processing_status: AssessmentStatus
    step_progress: Dict[str, float]  # Progress per step (0.0-1.0)
    validation_results: Dict[str, Any]
    quality_checks: Dict[str, bool]
    
    # Integration Data
    neo4j_queries: List[Dict[str, Any]]  # Cypher queries executed
    neo4j_results: List[Dict[str, Any]]  # Results from Neo4j queries
    external_api_calls: List[Dict[str, Any]]  # External data fetches
    regulatory_database_results: Dict[str, Any]
    
    # LLM Analysis
    llm_analysis_steps: List[Dict[str, Any]]  # Each LLM call and result
    reasoning_chains: List[Dict[str, Any]]  # Chain of thought records
    confidence_assessments: Dict[str, float]  # Confidence per analysis component
    model_uncertainties: List[str]  # Areas where model was uncertain
    
    # Output Generation
    report_sections: Dict[str, Any]  # Structured report content
    visualization_data: Dict[str, Any]  # Data for charts and graphs
    executive_summary: str
    technical_details: Dict[str, Any]
    regulatory_compliance_summary: Dict[str, Any]
    
    # Metadata
    created_timestamp: datetime
    last_updated: datetime
    processing_duration: Optional[float]
    analyst_notes: List[str]
    review_history: List[Dict[str, Any]]
    
    # LangSmith Integration
    langsmith_trace: LangSmithTrace
    llm_call_logs: List[Dict[str, Any]]  # Detailed LLM interaction logs
    
    # Error Handling
    errors: List[str]
    warnings: List[str]
    validation_issues: List[str]
    recovery_actions: List[str]
    
    # Workflow Control
    next_steps: List[str]
    human_review_required: bool
    escalation_triggers: List[str]
    approval_status: str
    stakeholder_notifications: List[Dict[str, Any]]


# Utility Classes for State Management
class StateValidator:
    """Validates state integrity and completeness."""
    
    @staticmethod
    def validate_assessment_state(state: RiskAssessmentState) -> Dict[str, Any]:
        """Validate the current state and return validation results."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0
        }
        
        # Check required fields
        required_fields = ["assessment_id", "request_type", "processing_status"]
        for field in required_fields:
            if not state.get(field):
                validation_results["errors"].append(f"Missing required field: {field}")
                validation_results["is_valid"] = False
        
        # Check data consistency
        if state.get("identified_risks") and state.get("recommendations"):
            risk_ids = {risk.risk_id for risk in state["identified_risks"]}
            rec_risk_ids = {rec.risk_id for rec in state["recommendations"]}
            orphaned_recs = rec_risk_ids - risk_ids
            if orphaned_recs:
                validation_results["warnings"].append(f"Recommendations without associated risks: {orphaned_recs}")
        
        # Calculate completeness
        total_fields = len(RiskAssessmentState.__annotations__)
        populated_fields = sum(1 for key in RiskAssessmentState.__annotations__ if state.get(key))
        validation_results["completeness_score"] = populated_fields / total_fields
        
        return validation_results
    
    @staticmethod
    def check_assessment_quality(state: RiskAssessmentState) -> Dict[str, Any]:
        """Check the quality of the risk assessment."""
        quality_results = {
            "overall_quality": "pending",
            "risk_coverage": 0.0,
            "evidence_strength": 0.0,
            "recommendation_completeness": 0.0,
            "issues": []
        }
        
        if not state.get("identified_risks"):
            quality_results["issues"].append("No risks identified")
            return quality_results
        
        # Check risk coverage
        identified_categories = {risk.category for risk in state["identified_risks"]}
        total_categories = len(RiskCategory)
        quality_results["risk_coverage"] = len(identified_categories) / total_categories
        
        # Check evidence strength
        risks_with_evidence = sum(1 for risk in state["identified_risks"] if risk.supporting_evidence)
        quality_results["evidence_strength"] = risks_with_evidence / len(state["identified_risks"])
        
        # Check recommendation completeness
        if state.get("recommendations"):
            quality_results["recommendation_completeness"] = len(state["recommendations"]) / len(state["identified_risks"])
        
        # Overall quality assessment
        avg_quality = (quality_results["risk_coverage"] + quality_results["evidence_strength"] + quality_results["recommendation_completeness"]) / 3
        
        if avg_quality >= 0.8:
            quality_results["overall_quality"] = "excellent"
        elif avg_quality >= 0.6:
            quality_results["overall_quality"] = "good"
        elif avg_quality >= 0.4:
            quality_results["overall_quality"] = "fair"
        else:
            quality_results["overall_quality"] = "poor"
        
        return quality_results


class StateUpdater:
    """Utility for updating state in a controlled manner."""
    
    @staticmethod
    def update_processing_status(state: RiskAssessmentState, new_status: AssessmentStatus, notes: Optional[str] = None):
        """Update processing status with timestamp and notes."""
        state["processing_status"] = new_status
        state["last_updated"] = datetime.utcnow()
        
        if notes:
            if "analyst_notes" not in state:
                state["analyst_notes"] = []
            state["analyst_notes"].append(f"[{datetime.utcnow().isoformat()}] {notes}")
    
    @staticmethod
    def add_risk_assessment(state: RiskAssessmentState, risk: RiskAssessment):
        """Add a new risk assessment to the state."""
        if "identified_risks" not in state:
            state["identified_risks"] = []
        state["identified_risks"].append(risk)
        state["last_updated"] = datetime.utcnow()
    
    @staticmethod
    def add_recommendation(state: RiskAssessmentState, recommendation: RiskMitigationRecommendation):
        """Add a new recommendation to the state."""
        if "recommendations" not in state:
            state["recommendations"] = []
        state["recommendations"].append(recommendation)
        state["last_updated"] = datetime.utcnow()
    
    @staticmethod
    def log_llm_interaction(state: RiskAssessmentState, model: str, prompt: str, response: str, metadata: Dict[str, Any]):
        """Log LLM interaction details."""
        if "llm_call_logs" not in state:
            state["llm_call_logs"] = []
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "prompt": prompt,
            "response": response,
            "metadata": metadata
        }
        state["llm_call_logs"].append(log_entry)
        
        # Update LangSmith trace
        if "langsmith_trace" in state and state["langsmith_trace"]:
            if metadata.get("total_tokens"):
                current_tokens = state["langsmith_trace"].total_tokens or 0
                state["langsmith_trace"].total_tokens = current_tokens + metadata["total_tokens"]


# Factory Functions
def create_initial_state(
    assessment_id: str,
    request_type: Literal["facility_assessment", "document_assessment", "incident_assessment", "comprehensive_assessment"],
    assessment_scope: Dict[str, Any],
    priority_level: str = "medium"
) -> RiskAssessmentState:
    """Create an initial state for a new risk assessment."""
    
    now = datetime.utcnow()
    
    return RiskAssessmentState(
        assessment_id=assessment_id,
        request_type=request_type,
        assessment_scope=assessment_scope,
        priority_level=priority_level,
        
        # Initialize empty collections
        document_contexts=[],
        facility_contexts=[],
        additional_data={},
        assessment_parameters={},
        
        identified_risks=[],
        risk_correlations={},
        risk_trends={},
        aggregate_risk_profile={},
        
        recommendations=[],
        prioritized_action_plan=[],
        implementation_roadmap={},
        cost_benefit_analysis={},
        
        # Processing state
        current_step="initialization",
        processing_status=AssessmentStatus.INITIATED,
        step_progress={},
        validation_results={},
        quality_checks={},
        
        # Integration data
        neo4j_queries=[],
        neo4j_results=[],
        external_api_calls=[],
        regulatory_database_results={},
        
        # Analysis
        llm_analysis_steps=[],
        reasoning_chains=[],
        confidence_assessments={},
        model_uncertainties=[],
        
        # Output
        report_sections={},
        visualization_data={},
        executive_summary="",
        technical_details={},
        regulatory_compliance_summary={},
        
        # Metadata
        created_timestamp=now,
        last_updated=now,
        processing_duration=None,
        analyst_notes=[],
        review_history=[],
        
        # LangSmith
        langsmith_trace=LangSmithTrace(),
        llm_call_logs=[],
        
        # Error handling
        errors=[],
        warnings=[],
        validation_issues=[],
        recovery_actions=[],
        
        # Workflow control
        next_steps=["data_collection"],
        human_review_required=False,
        escalation_triggers=[],
        approval_status="pending",
        stakeholder_notifications=[]
    )


def create_risk_matrix() -> Dict[str, Dict[str, int]]:
    """Create a standard risk scoring matrix."""
    probability_scores = {
        RiskProbability.VERY_LOW: 1,
        RiskProbability.LOW: 2,
        RiskProbability.MEDIUM: 3,
        RiskProbability.HIGH: 4,
        RiskProbability.VERY_HIGH: 5
    }
    
    severity_scores = {
        RiskSeverity.NEGLIGIBLE: 1,
        RiskSeverity.LOW: 2,
        RiskSeverity.MEDIUM: 3,
        RiskSeverity.HIGH: 4,
        RiskSeverity.CRITICAL: 5
    }
    
    return {
        "probability": probability_scores,
        "severity": severity_scores
    }


def calculate_risk_score(severity: RiskSeverity, probability: RiskProbability) -> float:
    """Calculate numerical risk score from severity and probability."""
    matrix = create_risk_matrix()
    severity_score = matrix["severity"][severity]
    probability_score = matrix["probability"][probability]
    
    # Risk score = probability × severity × 4 (to get 0-100 scale)
    return float(probability_score * severity_score * 4)


# Example usage and testing
if __name__ == "__main__":
    # Create initial state
    state = create_initial_state(
        assessment_id="test_assessment_001",
        request_type="facility_assessment",
        assessment_scope={"facility_ids": ["FAC_001"], "assessment_depth": "comprehensive"}
    )
    
    # Add sample document context
    doc_context = DocumentContext(
        document_id="DOC_001",
        document_type="environmental_permit",
        source_file="permit_2024.pdf",
        extraction_timestamp=datetime.utcnow(),
        confidence_score=0.95
    )
    state["document_contexts"].append(doc_context)
    
    # Add sample risk assessment
    risk = RiskAssessment(
        category=RiskCategory.ENVIRONMENTAL,
        subcategory=EnvironmentalRiskType.AIR_EMISSIONS,
        title="Elevated NOx Emissions",
        description="Facility emissions exceed permitted levels",
        severity=RiskSeverity.HIGH,
        probability=RiskProbability.MEDIUM,
        risk_score=calculate_risk_score(RiskSeverity.HIGH, RiskProbability.MEDIUM),
        timeframe=RiskTimeframe.SHORT_TERM,
        confidence_score=0.85
    )
    StateUpdater.add_risk_assessment(state, risk)
    
    # Add recommendation
    recommendation = RiskMitigationRecommendation(
        risk_id=risk.risk_id,
        title="Install Selective Catalytic Reduction System",
        description="Implement SCR technology to reduce NOx emissions",
        priority=RecommendationPriority.HIGH,
        implementation_timeframe="3-6 months",
        risk_reduction_potential=80.0
    )
    StateUpdater.add_recommendation(state, recommendation)
    
    # Validate state
    validation = StateValidator.validate_assessment_state(state)
    quality = StateValidator.check_assessment_quality(state)
    
    print(f"State validation: {validation['is_valid']}")
    print(f"Quality assessment: {quality['overall_quality']}")
    print(f"Risk coverage: {quality['risk_coverage']:.2f}")
    print(f"Assessment ID: {state['assessment_id']}")
    print(f"Number of risks identified: {len(state['identified_risks'])}")
    print(f"Number of recommendations: {len(state['recommendations'])}")