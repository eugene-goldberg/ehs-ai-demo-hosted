"""
EHS Risk Assessment Recommendation Generation System

This module provides comprehensive recommendation generation for Environmental, Health,
and Safety (EHS) risk assessments. It includes context-aware templates, industry-specific
best practices, cost-benefit analysis, implementation roadmaps, and regulatory compliance
strategies.

Features:
- Context-aware recommendation templates
- Industry-specific best practices database
- Cost-benefit analysis for recommendations
- Implementation roadmap generation
- Priority scoring based on risk levels
- Resource requirement estimation
- Regulatory compliance strategies
- Success metrics definition

Author: AI Assistant
Created: 2025-08-28
Version: 1.0.0
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import math

from ..models.agent_state import (
    RiskAssessment, RiskMitigationRecommendation, RecommendationPriority,
    RiskCategory, RiskSeverity, RiskProbability, RiskTimeframe,
    EnvironmentalRiskType, HealthSafetyRiskType, ComplianceRiskType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndustryType(str, Enum):
    """Industry classification for recommendations"""
    MANUFACTURING = "manufacturing"
    CHEMICAL = "chemical"
    OIL_GAS = "oil_gas"
    CONSTRUCTION = "construction"
    HEALTHCARE = "healthcare"
    TRANSPORTATION = "transportation"
    MINING = "mining"
    UTILITIES = "utilities"
    FOOD_BEVERAGE = "food_beverage"
    PHARMACEUTICALS = "pharmaceuticals"
    AGRICULTURE = "agriculture"
    TECHNOLOGY = "technology"
    AEROSPACE = "aerospace"
    AUTOMOTIVE = "automotive"


class ImplementationComplexity(str, Enum):
    """Implementation complexity levels"""
    LOW = "low"           # Simple operational changes
    MEDIUM = "medium"     # Requires some capital investment
    HIGH = "high"         # Major system changes required
    VERY_HIGH = "very_high"  # Complex multi-year initiatives


class CostCategory(str, Enum):
    """Cost categories for recommendations"""
    OPERATIONAL = "operational"      # Ongoing operational costs
    CAPITAL = "capital"             # One-time capital expenditure
    COMPLIANCE = "compliance"       # Regulatory compliance costs
    TRAINING = "training"           # Personnel training costs
    TECHNOLOGY = "technology"       # Technology implementation costs
    CONSULTING = "consulting"       # External consulting costs


class RegulatoryFramework(str, Enum):
    """Regulatory frameworks for compliance"""
    OSHA = "osha"
    EPA = "epa"
    ISO_14001 = "iso_14001"
    ISO_45001 = "iso_45001"
    ISO_31000 = "iso_31000"
    RCRA = "rcra"
    CERCLA = "cercla"
    CLEAN_AIR_ACT = "clean_air_act"
    CLEAN_WATER_ACT = "clean_water_act"
    TSCA = "tsca"
    DOT_HAZMAT = "dot_hazmat"
    NFPA = "nfpa"
    API = "api"
    ASTM = "astm"
    EU_REACH = "eu_reach"
    GHS = "ghs"


@dataclass
class CostEstimate:
    """Cost estimation for recommendations"""
    category: CostCategory
    low_estimate: float
    high_estimate: float
    currency: str = "USD"
    confidence_level: float = 0.7  # 0-1 scale
    assumptions: List[str] = field(default_factory=list)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    
    @property
    def mid_estimate(self) -> float:
        return (self.low_estimate + self.high_estimate) / 2
    
    @property
    def range_percent(self) -> float:
        if self.mid_estimate > 0:
            return ((self.high_estimate - self.low_estimate) / self.mid_estimate) * 100
        return 0


@dataclass
class BenefitEstimate:
    """Benefit estimation for recommendations"""
    category: str  # risk_reduction, cost_savings, compliance, reputation, etc.
    quantitative_benefit: Optional[float] = None
    qualitative_description: str = ""
    probability_of_realization: float = 0.8  # 0-1 scale
    timeframe_months: int = 12
    supporting_evidence: List[str] = field(default_factory=list)
    measurement_method: str = ""


@dataclass
class ImplementationMilestone:
    """Implementation milestone definition"""
    milestone_id: str
    title: str
    description: str
    target_date: datetime
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    responsible_party: str = ""
    success_criteria: List[str] = field(default_factory=list)
    estimated_effort_hours: Optional[float] = None
    budget_allocation: Optional[float] = None


@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    resource_type: str  # personnel, equipment, budget, time, expertise
    description: str
    quantity: Union[int, float]
    unit: str  # hours, FTE, dollars, pieces, etc.
    duration_months: Optional[int] = None
    specialized_skills_required: List[str] = field(default_factory=list)
    availability_constraints: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ComplianceRequirement:
    """Regulatory compliance requirement"""
    framework: RegulatoryFramework
    requirement_id: str
    title: str
    description: str
    compliance_deadline: Optional[datetime] = None
    current_compliance_status: str = "unknown"  # compliant, non_compliant, partial, unknown
    gap_analysis: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    enforcement_risk: str = "medium"  # low, medium, high, critical


@dataclass
class SuccessMetric:
    """Success measurement definition"""
    metric_id: str
    name: str
    description: str
    measurement_method: str
    target_value: Union[float, str]
    current_baseline: Optional[Union[float, str]] = None
    measurement_frequency: str = "monthly"  # daily, weekly, monthly, quarterly, annually
    data_source: str = ""
    responsible_party: str = ""
    threshold_for_success: str = ""
    leading_or_lagging: str = "lagging"  # leading, lagging, concurrent


@dataclass
class RecommendationTemplate:
    """Template for generating recommendations"""
    template_id: str
    name: str
    description: str
    applicable_risk_categories: List[RiskCategory]
    applicable_risk_types: List[str]
    industry_applicability: List[IndustryType]
    severity_threshold: RiskSeverity
    template_content: str
    default_priority: RecommendationPriority
    typical_implementation_time: str
    typical_cost_range: Tuple[float, float]
    effectiveness_rating: float  # 0-1 scale
    implementation_complexity: ImplementationComplexity
    regulatory_frameworks: List[RegulatoryFramework] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    common_challenges: List[str] = field(default_factory=list)


@dataclass
class EnhancedRecommendation:
    """Enhanced recommendation with comprehensive implementation details"""
    base_recommendation: RiskMitigationRecommendation
    
    # Cost-benefit analysis
    cost_estimates: List[CostEstimate] = field(default_factory=list)
    benefit_estimates: List[BenefitEstimate] = field(default_factory=list)
    roi_calculation: Optional[Dict[str, float]] = None
    payback_period_months: Optional[float] = None
    
    # Implementation details
    implementation_phases: List[str] = field(default_factory=list)
    milestones: List[ImplementationMilestone] = field(default_factory=list)
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    implementation_risks: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    
    # Compliance and regulatory
    compliance_requirements: List[ComplianceRequirement] = field(default_factory=list)
    regulatory_approval_needed: bool = False
    permit_requirements: List[str] = field(default_factory=list)
    
    # Monitoring and measurement
    success_metrics: List[SuccessMetric] = field(default_factory=list)
    monitoring_frequency: str = "monthly"
    review_schedule: List[datetime] = field(default_factory=list)
    
    # Stakeholder management
    key_stakeholders: List[str] = field(default_factory=list)
    communication_plan: Dict[str, Any] = field(default_factory=dict)
    training_requirements: List[str] = field(default_factory=list)
    
    # Risk mitigation
    implementation_contingencies: List[str] = field(default_factory=list)
    fallback_options: List[str] = field(default_factory=list)
    
    @property
    def total_cost_estimate(self) -> CostEstimate:
        """Calculate total cost across all categories"""
        if not self.cost_estimates:
            return CostEstimate(CostCategory.OPERATIONAL, 0, 0)
        
        total_low = sum(ce.low_estimate for ce in self.cost_estimates)
        total_high = sum(ce.high_estimate for ce in self.cost_estimates)
        
        return CostEstimate(
            category=CostCategory.OPERATIONAL,  # Mixed category
            low_estimate=total_low,
            high_estimate=total_high,
            confidence_level=min(ce.confidence_level for ce in self.cost_estimates)
        )


class RecommendationTemplateManager:
    """Manages recommendation templates and best practices"""
    
    def __init__(self):
        self.templates: Dict[str, RecommendationTemplate] = {}
        self.industry_best_practices: Dict[IndustryType, Dict[str, Any]] = {}
        self._initialize_templates()
        self._initialize_industry_practices()
    
    def _initialize_templates(self):
        """Initialize recommendation templates"""
        
        # Environmental risk templates
        self.templates["env_emissions_control"] = RecommendationTemplate(
            template_id="env_emissions_control",
            name="Air Emissions Control System",
            description="Install or upgrade air emissions control systems",
            applicable_risk_categories=[RiskCategory.ENVIRONMENTAL],
            applicable_risk_types=[EnvironmentalRiskType.AIR_EMISSIONS.value],
            industry_applicability=[IndustryType.MANUFACTURING, IndustryType.CHEMICAL, IndustryType.OIL_GAS],
            severity_threshold=RiskSeverity.MEDIUM,
            template_content="Implement {control_technology} to reduce {pollutant_type} emissions by {target_reduction}% within {timeframe}",
            default_priority=RecommendationPriority.HIGH,
            typical_implementation_time="6-18 months",
            typical_cost_range=(50000, 500000),
            effectiveness_rating=0.85,
            implementation_complexity=ImplementationComplexity.HIGH,
            regulatory_frameworks=[RegulatoryFramework.EPA, RegulatoryFramework.CLEAN_AIR_ACT],
            success_factors=[
                "Proper technology selection",
                "Adequate maintenance planning",
                "Operator training",
                "Compliance monitoring"
            ],
            common_challenges=[
                "Capital cost justification",
                "Technology integration",
                "Maintenance complexity",
                "Regulatory approval timeline"
            ]
        )
        
        self.templates["water_treatment"] = RecommendationTemplate(
            template_id="water_treatment",
            name="Wastewater Treatment System",
            description="Upgrade wastewater treatment capabilities",
            applicable_risk_categories=[RiskCategory.ENVIRONMENTAL],
            applicable_risk_types=[EnvironmentalRiskType.WATER_POLLUTION.value],
            industry_applicability=[IndustryType.MANUFACTURING, IndustryType.CHEMICAL, IndustryType.FOOD_BEVERAGE],
            severity_threshold=RiskSeverity.MEDIUM,
            template_content="Implement {treatment_type} system to achieve {discharge_standard} compliance",
            default_priority=RecommendationPriority.HIGH,
            typical_implementation_time="4-12 months",
            typical_cost_range=(25000, 250000),
            effectiveness_rating=0.9,
            implementation_complexity=ImplementationComplexity.MEDIUM,
            regulatory_frameworks=[RegulatoryFramework.EPA, RegulatoryFramework.CLEAN_WATER_ACT],
            success_factors=[
                "Proper system sizing",
                "Regular maintenance",
                "Operator certification",
                "Monitoring compliance"
            ]
        )
        
        # Health and safety templates
        self.templates["ppe_program"] = RecommendationTemplate(
            template_id="ppe_program",
            name="Personal Protective Equipment Program",
            description="Enhance PPE selection, training, and compliance",
            applicable_risk_categories=[RiskCategory.HEALTH_SAFETY],
            applicable_risk_types=[HealthSafetyRiskType.CHEMICAL_EXPOSURE.value, HealthSafetyRiskType.PHYSICAL_HAZARD.value],
            industry_applicability=list(IndustryType),  # Applicable to all industries
            severity_threshold=RiskSeverity.LOW,
            template_content="Implement comprehensive PPE program including {ppe_types} with training and compliance monitoring",
            default_priority=RecommendationPriority.MEDIUM,
            typical_implementation_time="2-6 months",
            typical_cost_range=(5000, 50000),
            effectiveness_rating=0.8,
            implementation_complexity=ImplementationComplexity.LOW,
            regulatory_frameworks=[RegulatoryFramework.OSHA],
            success_factors=[
                "Proper hazard assessment",
                "Correct PPE selection",
                "Employee training",
                "Regular inspections"
            ]
        )
        
        self.templates["safety_training"] = RecommendationTemplate(
            template_id="safety_training",
            name="Enhanced Safety Training Program",
            description="Develop comprehensive safety training and competency program",
            applicable_risk_categories=[RiskCategory.HEALTH_SAFETY],
            applicable_risk_types=list(HealthSafetyRiskType.__members__.values()),
            industry_applicability=list(IndustryType),
            severity_threshold=RiskSeverity.LOW,
            template_content="Develop {training_type} program covering {hazard_types} with competency assessment",
            default_priority=RecommendationPriority.MEDIUM,
            typical_implementation_time="3-9 months",
            typical_cost_range=(10000, 100000),
            effectiveness_rating=0.75,
            implementation_complexity=ImplementationComplexity.MEDIUM,
            regulatory_frameworks=[RegulatoryFramework.OSHA, RegulatoryFramework.ISO_45001]
        )
        
        # Compliance templates
        self.templates["management_system"] = RecommendationTemplate(
            template_id="management_system",
            name="Environmental Management System",
            description="Implement ISO 14001 Environmental Management System",
            applicable_risk_categories=[RiskCategory.ENVIRONMENTAL, RiskCategory.REGULATORY_COMPLIANCE],
            applicable_risk_types=list(EnvironmentalRiskType.__members__.values()),
            industry_applicability=list(IndustryType),
            severity_threshold=RiskSeverity.MEDIUM,
            template_content="Implement ISO 14001 EMS with {scope} covering {environmental_aspects}",
            default_priority=RecommendationPriority.MEDIUM,
            typical_implementation_time="12-24 months",
            typical_cost_range=(25000, 150000),
            effectiveness_rating=0.8,
            implementation_complexity=ImplementationComplexity.HIGH,
            regulatory_frameworks=[RegulatoryFramework.ISO_14001]
        )
        
        # Add more templates as needed...
        logger.info(f"Initialized {len(self.templates)} recommendation templates")
    
    def _initialize_industry_practices(self):
        """Initialize industry-specific best practices"""
        
        self.industry_best_practices[IndustryType.CHEMICAL] = {
            "risk_tolerance": "low",
            "regulatory_focus": ["EPA", "OSHA", "DOT"],
            "common_controls": [
                "Process safety management",
                "Chemical inventory management",
                "Emergency response planning",
                "Vapor control systems",
                "Secondary containment"
            ],
            "technology_preferences": [
                "Automated monitoring systems",
                "Leak detection and repair",
                "Process optimization software"
            ],
            "cost_multipliers": {
                "environmental": 1.3,
                "safety": 1.4,
                "compliance": 1.2
            }
        }
        
        self.industry_best_practices[IndustryType.MANUFACTURING] = {
            "risk_tolerance": "medium",
            "regulatory_focus": ["OSHA", "EPA"],
            "common_controls": [
                "Machine guarding",
                "Lockout/tagout procedures",
                "Ergonomic assessments",
                "Noise control measures",
                "Waste minimization"
            ],
            "technology_preferences": [
                "Automated safety systems",
                "Environmental monitoring",
                "Predictive maintenance"
            ],
            "cost_multipliers": {
                "operational": 1.1,
                "technology": 1.2,
                "training": 1.0
            }
        }
        
        # Add more industry practices...
    
    def get_applicable_templates(self, risk: RiskAssessment, industry: IndustryType) -> List[RecommendationTemplate]:
        """Get applicable templates for a risk and industry"""
        applicable = []
        
        for template in self.templates.values():
            # Check risk category match
            if risk.category not in template.applicable_risk_categories:
                continue
            
            # Check risk type match
            if str(risk.subcategory) not in template.applicable_risk_types:
                continue
            
            # Check industry applicability
            if industry not in template.industry_applicability:
                continue
            
            # Check severity threshold
            severity_values = {
                RiskSeverity.NEGLIGIBLE: 1,
                RiskSeverity.LOW: 2,
                RiskSeverity.MEDIUM: 3,
                RiskSeverity.HIGH: 4,
                RiskSeverity.CRITICAL: 5
            }
            
            if severity_values[risk.severity] < severity_values[template.severity_threshold]:
                continue
            
            applicable.append(template)
        
        # Sort by effectiveness rating
        applicable.sort(key=lambda t: t.effectiveness_rating, reverse=True)
        return applicable


class CostBenefitAnalyzer:
    """Analyzes costs and benefits of recommendations"""
    
    def __init__(self, industry: IndustryType = IndustryType.MANUFACTURING):
        self.industry = industry
        self.cost_databases = self._initialize_cost_databases()
        self.benefit_models = self._initialize_benefit_models()
    
    def _initialize_cost_databases(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cost estimation databases"""
        return {
            "equipment": {
                "air_scrubber": {"low": 50000, "high": 200000, "unit": "system"},
                "water_treatment": {"low": 25000, "high": 150000, "unit": "system"},
                "monitoring_system": {"low": 10000, "high": 50000, "unit": "system"},
                "safety_equipment": {"low": 1000, "high": 25000, "unit": "employee"}
            },
            "labor": {
                "engineering": {"rate": 150, "unit": "hour"},
                "technician": {"rate": 75, "unit": "hour"},
                "contractor": {"rate": 100, "unit": "hour"},
                "training": {"rate": 50, "unit": "person-hour"}
            },
            "compliance": {
                "permit_application": {"low": 5000, "high": 25000, "unit": "permit"},
                "audit": {"low": 15000, "high": 50000, "unit": "audit"},
                "consultant": {"low": 150, "high": 300, "unit": "hour"}
            }
        }
    
    def _initialize_benefit_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize benefit estimation models"""
        return {
            "risk_reduction": {
                "formula": "avoided_cost * probability_reduction * annual_exposure",
                "factors": ["incident_cost", "probability", "frequency"]
            },
            "compliance": {
                "formula": "penalty_avoidance + reputation_value",
                "factors": ["penalty_amount", "enforcement_probability", "reputation_impact"]
            },
            "operational": {
                "formula": "efficiency_gain * annual_volume",
                "factors": ["process_improvement", "cost_savings", "productivity_gain"]
            }
        }
    
    def estimate_costs(self, recommendation: RiskMitigationRecommendation, 
                      implementation_scope: Dict[str, Any]) -> List[CostEstimate]:
        """Estimate costs for a recommendation"""
        costs = []
        
        # Capital costs
        if "equipment" in implementation_scope:
            equipment_costs = self._estimate_equipment_costs(implementation_scope["equipment"])
            costs.extend(equipment_costs)
        
        # Labor costs
        if "labor_hours" in implementation_scope:
            labor_costs = self._estimate_labor_costs(implementation_scope["labor_hours"])
            costs.extend(labor_costs)
        
        # Compliance costs
        if "regulatory_requirements" in implementation_scope:
            compliance_costs = self._estimate_compliance_costs(implementation_scope["regulatory_requirements"])
            costs.extend(compliance_costs)
        
        # Operational costs
        if "ongoing_operations" in implementation_scope:
            operational_costs = self._estimate_operational_costs(implementation_scope["ongoing_operations"])
            costs.extend(operational_costs)
        
        return costs
    
    def _estimate_equipment_costs(self, equipment_list: List[str]) -> List[CostEstimate]:
        """Estimate equipment costs"""
        costs = []
        
        for equipment in equipment_list:
            if equipment in self.cost_databases["equipment"]:
                cost_data = self.cost_databases["equipment"][equipment]
                
                costs.append(CostEstimate(
                    category=CostCategory.CAPITAL,
                    low_estimate=cost_data["low"],
                    high_estimate=cost_data["high"],
                    confidence_level=0.7,
                    assumptions=[f"Based on typical {equipment} costs"],
                    cost_breakdown={equipment: cost_data["high"]}
                ))
        
        return costs
    
    def _estimate_labor_costs(self, labor_requirements: Dict[str, int]) -> List[CostEstimate]:
        """Estimate labor costs"""
        costs = []
        
        for labor_type, hours in labor_requirements.items():
            if labor_type in self.cost_databases["labor"]:
                rate = self.cost_databases["labor"][labor_type]["rate"]
                
                costs.append(CostEstimate(
                    category=CostCategory.OPERATIONAL,
                    low_estimate=hours * rate * 0.8,  # 20% buffer
                    high_estimate=hours * rate * 1.3,  # 30% overrun
                    confidence_level=0.8,
                    assumptions=[f"Based on {hours} hours at ${rate}/hour"],
                    cost_breakdown={f"{labor_type}_labor": hours * rate}
                ))
        
        return costs
    
    def _estimate_compliance_costs(self, requirements: List[str]) -> List[CostEstimate]:
        """Estimate regulatory compliance costs"""
        costs = []
        
        for requirement in requirements:
            if requirement in self.cost_databases["compliance"]:
                cost_data = self.cost_databases["compliance"][requirement]
                
                costs.append(CostEstimate(
                    category=CostCategory.COMPLIANCE,
                    low_estimate=cost_data["low"],
                    high_estimate=cost_data["high"],
                    confidence_level=0.6,
                    assumptions=[f"Based on typical {requirement} costs"],
                    cost_breakdown={requirement: cost_data["high"]}
                ))
        
        return costs
    
    def _estimate_operational_costs(self, operational_items: Dict[str, Any]) -> List[CostEstimate]:
        """Estimate ongoing operational costs"""
        costs = []
        
        # Calculate annual operational costs
        annual_cost_low = operational_items.get("annual_low", 5000)
        annual_cost_high = operational_items.get("annual_high", 25000)
        
        costs.append(CostEstimate(
            category=CostCategory.OPERATIONAL,
            low_estimate=annual_cost_low,
            high_estimate=annual_cost_high,
            confidence_level=0.7,
            assumptions=["Annual operational costs including maintenance, utilities, supplies"],
            cost_breakdown={"annual_operations": annual_cost_high}
        ))
        
        return costs
    
    def estimate_benefits(self, recommendation: RiskMitigationRecommendation,
                         risk: RiskAssessment, context: Dict[str, Any]) -> List[BenefitEstimate]:
        """Estimate benefits for a recommendation"""
        benefits = []
        
        # Risk reduction benefits
        risk_reduction_benefit = self._calculate_risk_reduction_benefit(
            recommendation, risk, context
        )
        if risk_reduction_benefit:
            benefits.append(risk_reduction_benefit)
        
        # Compliance benefits
        compliance_benefit = self._calculate_compliance_benefit(
            recommendation, risk, context
        )
        if compliance_benefit:
            benefits.append(compliance_benefit)
        
        # Operational benefits
        operational_benefit = self._calculate_operational_benefit(
            recommendation, risk, context
        )
        if operational_benefit:
            benefits.append(operational_benefit)
        
        # Reputation benefits
        reputation_benefit = self._calculate_reputation_benefit(
            recommendation, risk, context
        )
        if reputation_benefit:
            benefits.append(reputation_benefit)
        
        return benefits
    
    def _calculate_risk_reduction_benefit(self, recommendation: RiskMitigationRecommendation,
                                        risk: RiskAssessment, context: Dict[str, Any]) -> Optional[BenefitEstimate]:
        """Calculate risk reduction benefits"""
        
        # Estimate potential incident cost
        severity_costs = {
            RiskSeverity.NEGLIGIBLE: 1000,
            RiskSeverity.LOW: 10000,
            RiskSeverity.MEDIUM: 100000,
            RiskSeverity.HIGH: 1000000,
            RiskSeverity.CRITICAL: 10000000
        }
        
        probability_factors = {
            RiskProbability.VERY_LOW: 0.05,
            RiskProbability.LOW: 0.15,
            RiskProbability.MEDIUM: 0.35,
            RiskProbability.HIGH: 0.65,
            RiskProbability.VERY_HIGH: 0.85
        }
        
        incident_cost = severity_costs.get(risk.severity, 100000)
        annual_probability = probability_factors.get(risk.probability, 0.35)
        
        # Calculate annual expected loss
        expected_annual_loss = incident_cost * annual_probability
        
        # Apply risk reduction potential
        risk_reduction = recommendation.risk_reduction_potential / 100
        annual_benefit = expected_annual_loss * risk_reduction
        
        return BenefitEstimate(
            category="risk_reduction",
            quantitative_benefit=annual_benefit,
            qualitative_description=f"Reduced expected annual loss from {risk.title}",
            probability_of_realization=0.8,
            timeframe_months=12,
            supporting_evidence=[
                f"Risk reduction potential: {recommendation.risk_reduction_potential}%",
                f"Estimated incident cost: ${incident_cost:,}",
                f"Annual probability: {annual_probability:.1%}"
            ],
            measurement_method="Expected value calculation based on risk reduction"
        )
    
    def _calculate_compliance_benefit(self, recommendation: RiskMitigationRecommendation,
                                    risk: RiskAssessment, context: Dict[str, Any]) -> Optional[BenefitEstimate]:
        """Calculate compliance-related benefits"""
        
        if not risk.regulatory_context:
            return None
        
        # Estimate penalty avoidance
        penalty_estimates = {
            RiskSeverity.LOW: 25000,
            RiskSeverity.MEDIUM: 100000,
            RiskSeverity.HIGH: 500000,
            RiskSeverity.CRITICAL: 2000000
        }
        
        potential_penalty = penalty_estimates.get(risk.severity, 100000)
        enforcement_probability = 0.3  # Assume 30% chance of enforcement action
        
        compliance_benefit = potential_penalty * enforcement_probability
        
        return BenefitEstimate(
            category="compliance",
            quantitative_benefit=compliance_benefit,
            qualitative_description="Avoided regulatory penalties and enforcement actions",
            probability_of_realization=0.7,
            timeframe_months=12,
            supporting_evidence=[
                f"Potential penalty: ${potential_penalty:,}",
                f"Enforcement probability: {enforcement_probability:.1%}",
                "Regulatory compliance improvement"
            ],
            measurement_method="Penalty avoidance calculation"
        )
    
    def _calculate_operational_benefit(self, recommendation: RiskMitigationRecommendation,
                                     risk: RiskAssessment, context: Dict[str, Any]) -> Optional[BenefitEstimate]:
        """Calculate operational efficiency benefits"""
        
        # Estimate operational improvements
        if risk.category in [RiskCategory.OPERATIONAL, RiskCategory.ENVIRONMENTAL]:
            efficiency_improvement = 0.05  # Assume 5% efficiency gain
            annual_operations_cost = context.get("annual_operations_cost", 1000000)
            
            operational_savings = annual_operations_cost * efficiency_improvement
            
            return BenefitEstimate(
                category="operational",
                quantitative_benefit=operational_savings,
                qualitative_description="Operational efficiency improvements and cost savings",
                probability_of_realization=0.6,
                timeframe_months=18,
                supporting_evidence=[
                    f"Estimated efficiency improvement: {efficiency_improvement:.1%}",
                    f"Annual operations cost: ${annual_operations_cost:,}"
                ],
                measurement_method="Efficiency improvement calculation"
            )
        
        return None
    
    def _calculate_reputation_benefit(self, recommendation: RiskMitigationRecommendation,
                                    risk: RiskAssessment, context: Dict[str, Any]) -> Optional[BenefitEstimate]:
        """Calculate reputation and brand benefits"""
        
        if risk.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]:
            # Qualitative reputation benefit
            return BenefitEstimate(
                category="reputation",
                quantitative_benefit=None,
                qualitative_description="Enhanced corporate reputation and stakeholder confidence",
                probability_of_realization=0.8,
                timeframe_months=24,
                supporting_evidence=[
                    "Proactive risk management demonstrates responsibility",
                    "Reduced likelihood of negative media attention",
                    "Improved stakeholder trust and confidence"
                ],
                measurement_method="Stakeholder feedback and media sentiment analysis"
            )
        
        return None
    
    def calculate_roi(self, costs: List[CostEstimate], benefits: List[BenefitEstimate],
                     analysis_period_years: int = 5) -> Dict[str, float]:
        """Calculate return on investment metrics"""
        
        # Calculate total costs
        total_cost_low = sum(c.low_estimate for c in costs)
        total_cost_high = sum(c.high_estimate for c in costs)
        total_cost_mid = (total_cost_low + total_cost_high) / 2
        
        # Calculate total benefits (quantitative only)
        quantitative_benefits = [b for b in benefits if b.quantitative_benefit is not None]
        
        if not quantitative_benefits:
            return {"roi": 0, "payback_period": float('inf'), "npv": -total_cost_mid}
        
        annual_benefits = sum(
            (b.quantitative_benefit * b.probability_of_realization) 
            for b in quantitative_benefits
        )
        
        # Calculate metrics
        total_benefits = annual_benefits * analysis_period_years
        roi = ((total_benefits - total_cost_mid) / total_cost_mid) * 100 if total_cost_mid > 0 else 0
        payback_period = total_cost_mid / annual_benefits if annual_benefits > 0 else float('inf')
        
        # Simple NPV calculation (assuming 10% discount rate)
        discount_rate = 0.10
        npv = -total_cost_mid
        for year in range(1, analysis_period_years + 1):
            npv += annual_benefits / ((1 + discount_rate) ** year)
        
        return {
            "roi_percent": roi,
            "payback_period_years": payback_period,
            "npv": npv,
            "total_cost": total_cost_mid,
            "annual_benefits": annual_benefits,
            "total_benefits": total_benefits
        }


class ImplementationPlanner:
    """Creates detailed implementation plans and roadmaps"""
    
    def __init__(self):
        self.phase_templates = self._initialize_phase_templates()
        self.resource_databases = self._initialize_resource_databases()
    
    def _initialize_phase_templates(self) -> Dict[str, List[str]]:
        """Initialize implementation phase templates"""
        return {
            "technology_implementation": [
                "Design and Engineering",
                "Procurement and Contracting",
                "Installation and Construction",
                "Testing and Commissioning",
                "Training and Documentation",
                "Operations and Maintenance"
            ],
            "process_improvement": [
                "Current State Assessment",
                "Future State Design",
                "Gap Analysis and Planning",
                "Pilot Implementation",
                "Full-Scale Rollout",
                "Monitoring and Optimization"
            ],
            "training_program": [
                "Training Needs Assessment",
                "Curriculum Development",
                "Material Preparation",
                "Pilot Training",
                "Full Training Rollout",
                "Competency Assessment and Certification"
            ],
            "management_system": [
                "Gap Assessment",
                "Documentation Development",
                "Implementation Planning",
                "System Implementation",
                "Internal Audit",
                "Management Review and Certification"
            ]
        }
    
    def _initialize_resource_databases(self) -> Dict[str, Dict[str, Any]]:
        """Initialize resource requirement databases"""
        return {
            "personnel": {
                "project_manager": {"fte": 0.5, "duration_months": 12, "skills": ["Project Management", "EHS Experience"]},
                "engineer": {"fte": 1.0, "duration_months": 6, "skills": ["Technical Design", "Regulatory Knowledge"]},
                "technician": {"fte": 0.25, "duration_months": 18, "skills": ["Equipment Operation", "Maintenance"]},
                "trainer": {"fte": 0.1, "duration_months": 3, "skills": ["Adult Learning", "Subject Matter Expertise"]}
            },
            "equipment": {
                "monitoring_system": {"lead_time_months": 3, "installation_months": 2},
                "treatment_system": {"lead_time_months": 6, "installation_months": 4},
                "safety_equipment": {"lead_time_months": 1, "installation_months": 1}
            },
            "external_services": {
                "consultant": {"typical_duration_months": 6, "expertise_areas": ["Regulatory", "Technical", "Training"]},
                "contractor": {"typical_duration_months": 4, "services": ["Installation", "Construction", "Testing"]}
            }
        }
    
    def create_implementation_plan(self, recommendation: RiskMitigationRecommendation,
                                 complexity: ImplementationComplexity,
                                 constraints: Dict[str, Any]) -> Tuple[List[str], List[ImplementationMilestone]]:
        """Create detailed implementation plan"""
        
        # Determine appropriate phase template
        phase_template = self._select_phase_template(recommendation, complexity)
        phases = self.phase_templates.get(phase_template, self.phase_templates["process_improvement"])
        
        # Create milestones for each phase
        milestones = []
        start_date = datetime.now()
        
        for i, phase in enumerate(phases):
            milestone = self._create_milestone(
                phase, i, start_date, recommendation, constraints
            )
            milestones.append(milestone)
            
            # Next milestone starts after this one
            start_date = milestone.target_date + timedelta(days=7)  # 1 week buffer
        
        return phases, milestones
    
    def _select_phase_template(self, recommendation: RiskMitigationRecommendation,
                             complexity: ImplementationComplexity) -> str:
        """Select appropriate phase template based on recommendation characteristics"""
        
        # Simple heuristics for template selection
        if "system" in recommendation.title.lower() or "equipment" in recommendation.title.lower():
            return "technology_implementation"
        elif "training" in recommendation.title.lower() or "program" in recommendation.title.lower():
            return "training_program"
        elif "management" in recommendation.title.lower() or "iso" in recommendation.title.lower():
            return "management_system"
        else:
            return "process_improvement"
    
    def _create_milestone(self, phase_name: str, phase_index: int, start_date: datetime,
                         recommendation: RiskMitigationRecommendation,
                         constraints: Dict[str, Any]) -> ImplementationMilestone:
        """Create a milestone for a specific phase"""
        
        # Estimate phase duration based on complexity
        phase_durations = {
            0: 30,   # First phase: 1 month
            1: 45,   # Second phase: 1.5 months
            2: 60,   # Third phase: 2 months
            3: 30,   # Fourth phase: 1 month
            4: 45,   # Fifth phase: 1.5 months
            5: 30    # Sixth phase: 1 month
        }
        
        base_duration = phase_durations.get(phase_index, 30)
        
        # Adjust for complexity
        complexity_multipliers = {
            ImplementationComplexity.LOW: 0.7,
            ImplementationComplexity.MEDIUM: 1.0,
            ImplementationComplexity.HIGH: 1.4,
            ImplementationComplexity.VERY_HIGH: 2.0
        }
        
        complexity_factor = complexity_multipliers.get(
            recommendation.implementation_complexity, 1.0
        )
        
        adjusted_duration = int(base_duration * complexity_factor)
        target_date = start_date + timedelta(days=adjusted_duration)
        
        # Create milestone
        milestone = ImplementationMilestone(
            milestone_id=f"{recommendation.recommendation_id}_{phase_index}",
            title=f"{phase_name}",
            description=f"Complete {phase_name.lower()} for {recommendation.title}",
            target_date=target_date,
            dependencies=[f"{recommendation.recommendation_id}_{phase_index-1}"] if phase_index > 0 else [],
            deliverables=self._get_phase_deliverables(phase_name),
            responsible_party=self._get_responsible_party(phase_name),
            success_criteria=self._get_success_criteria(phase_name),
            estimated_effort_hours=self._estimate_effort_hours(phase_name, complexity_factor),
            budget_allocation=None  # Will be calculated separately
        )
        
        return milestone
    
    def _get_phase_deliverables(self, phase_name: str) -> List[str]:
        """Get typical deliverables for a phase"""
        deliverables_map = {
            "Design and Engineering": [
                "Technical specifications",
                "Engineering drawings",
                "Equipment selection",
                "Installation plans"
            ],
            "Current State Assessment": [
                "Assessment report",
                "Gap analysis",
                "Risk evaluation",
                "Baseline measurements"
            ],
            "Training Needs Assessment": [
                "Training needs analysis",
                "Competency requirements",
                "Learning objectives",
                "Training strategy"
            ],
            "Gap Assessment": [
                "Current state evaluation",
                "Requirements analysis",
                "Implementation plan",
                "Resource requirements"
            ]
        }
        
        return deliverables_map.get(phase_name, [
            f"{phase_name} completion report",
            "Phase deliverables",
            "Quality checkpoints",
            "Next phase readiness"
        ])
    
    def _get_responsible_party(self, phase_name: str) -> str:
        """Get typical responsible party for a phase"""
        responsibility_map = {
            "Design and Engineering": "Engineering Team",
            "Procurement and Contracting": "Procurement Department",
            "Installation and Construction": "Maintenance/Construction Team",
            "Training and Documentation": "Training Department",
            "Current State Assessment": "EHS Team",
            "Gap Assessment": "Quality/Compliance Team"
        }
        
        return responsibility_map.get(phase_name, "Project Manager")
    
    def _get_success_criteria(self, phase_name: str) -> List[str]:
        """Get success criteria for a phase"""
        criteria_map = {
            "Design and Engineering": [
                "Technical specifications approved",
                "Regulatory compliance verified",
                "Budget estimates within range",
                "Timeline feasible"
            ],
            "Testing and Commissioning": [
                "All tests passed",
                "Performance specifications met",
                "Safety systems functional",
                "Documentation complete"
            ],
            "Training and Documentation": [
                "All personnel trained",
                "Competency assessments passed",
                "Documentation updated",
                "Procedures implemented"
            ]
        }
        
        return criteria_map.get(phase_name, [
            f"{phase_name} objectives met",
            "Quality standards achieved",
            "Deliverables completed",
            "Stakeholder approval obtained"
        ])
    
    def _estimate_effort_hours(self, phase_name: str, complexity_factor: float) -> float:
        """Estimate effort hours for a phase"""
        base_hours = {
            "Design and Engineering": 120,
            "Procurement and Contracting": 40,
            "Installation and Construction": 80,
            "Testing and Commissioning": 60,
            "Training and Documentation": 80,
            "Current State Assessment": 60,
            "Gap Analysis and Planning": 100
        }
        
        return base_hours.get(phase_name, 80) * complexity_factor
    
    def estimate_resource_requirements(self, milestones: List[ImplementationMilestone],
                                     recommendation_type: str) -> List[ResourceRequirement]:
        """Estimate resource requirements for implementation"""
        
        resources = []
        
        # Personnel requirements
        total_effort_hours = sum(m.estimated_effort_hours or 80 for m in milestones)
        
        resources.append(ResourceRequirement(
            resource_type="personnel",
            description="Project Manager",
            quantity=0.5,
            unit="FTE",
            duration_months=len(milestones),
            specialized_skills_required=["Project Management", "EHS Experience"],
            availability_constraints=["Must be available for project duration"],
            alternatives=["External consultant", "Part-time internal resource"]
        ))
        
        if total_effort_hours > 200:
            resources.append(ResourceRequirement(
                resource_type="personnel",
                description="Technical Specialist",
                quantity=1.0,
                unit="FTE",
                duration_months=max(3, len(milestones) // 2),
                specialized_skills_required=["Technical expertise", "Implementation experience"],
                availability_constraints=["Subject matter expertise required"],
                alternatives=["Contract specialist", "Multi-skilled technician"]
            ))
        
        # Budget requirements
        resources.append(ResourceRequirement(
            resource_type="budget",
            description="Implementation Budget",
            quantity=total_effort_hours * 100,  # $100/hour average
            unit="USD",
            specialized_skills_required=[],
            availability_constraints=["Budget approval required"],
            alternatives=["Phased funding", "External financing"]
        ))
        
        # Equipment/Technology requirements
        if "system" in recommendation_type.lower() or "equipment" in recommendation_type.lower():
            resources.append(ResourceRequirement(
                resource_type="equipment",
                description="Hardware/Software Systems",
                quantity=1,
                unit="system",
                duration_months=None,
                specialized_skills_required=["Technical installation", "System integration"],
                availability_constraints=["Vendor lead times", "Installation scheduling"],
                alternatives=["Lease vs purchase", "Phased implementation"]
            ))
        
        return resources


class EHSRecommendationGenerator:
    """
    Comprehensive EHS Recommendation Generation System
    
    Generates detailed, actionable recommendations with implementation plans,
    cost-benefit analysis, and regulatory compliance strategies.
    """
    
    def __init__(self, industry: IndustryType = IndustryType.MANUFACTURING):
        """
        Initialize the recommendation generator
        
        Args:
            industry: Industry type for context-specific recommendations
        """
        self.industry = industry
        self.template_manager = RecommendationTemplateManager()
        self.cost_benefit_analyzer = CostBenefitAnalyzer(industry)
        self.implementation_planner = ImplementationPlanner()
        
        logger.info(f"EHS Recommendation Generator initialized for {industry.value} industry")
    
    def generate_recommendations(self, risks: List[RiskAssessment],
                               facility_context: Optional[Dict[str, Any]] = None,
                               budget_constraints: Optional[Dict[str, Any]] = None,
                               timeline_constraints: Optional[Dict[str, Any]] = None) -> List[EnhancedRecommendation]:
        """
        Generate comprehensive recommendations for identified risks
        
        Args:
            risks: List of identified risks requiring recommendations
            facility_context: Facility-specific context information
            budget_constraints: Budget limitations and preferences
            timeline_constraints: Timeline requirements and constraints
            
        Returns:
            List of enhanced recommendations with implementation details
        """
        
        enhanced_recommendations = []
        
        for risk in risks:
            logger.info(f"Generating recommendations for risk: {risk.title}")
            
            # Get applicable templates
            templates = self.template_manager.get_applicable_templates(risk, self.industry)
            
            if not templates:
                # Generate generic recommendation if no templates match
                generic_recommendation = self._create_generic_recommendation(risk)
                enhanced_recommendations.append(generic_recommendation)
                continue
            
            # Generate recommendations from templates
            for template in templates[:3]:  # Limit to top 3 templates
                try:
                    enhanced_rec = self._generate_from_template(
                        risk, template, facility_context, budget_constraints, timeline_constraints
                    )
                    enhanced_recommendations.append(enhanced_rec)
                except Exception as e:
                    logger.error(f"Error generating recommendation from template {template.template_id}: {e}")
                    continue
        
        # Prioritize and optimize recommendations
        prioritized_recommendations = self._prioritize_recommendations(enhanced_recommendations)
        
        # Check for synergies and consolidation opportunities
        optimized_recommendations = self._optimize_recommendations(prioritized_recommendations)
        
        logger.info(f"Generated {len(optimized_recommendations)} enhanced recommendations")
        return optimized_recommendations
    
    def _generate_from_template(self, risk: RiskAssessment, template: RecommendationTemplate,
                               facility_context: Optional[Dict[str, Any]],
                               budget_constraints: Optional[Dict[str, Any]],
                               timeline_constraints: Optional[Dict[str, Any]]) -> EnhancedRecommendation:
        """Generate enhanced recommendation from template"""
        
        # Create base recommendation
        base_recommendation = RiskMitigationRecommendation(
            risk_id=risk.risk_id,
            title=template.name,
            description=self._customize_description(template, risk, facility_context),
            priority=self._determine_priority(template.default_priority, risk),
            implementation_timeframe=template.typical_implementation_time,
            estimated_cost=template.typical_cost_range[1],  # Use high estimate
            resource_requirements=self._generate_resource_list(template),
            responsible_parties=self._determine_responsible_parties(template, facility_context),
            risk_reduction_potential=template.effectiveness_rating * 100,
            implementation_complexity=template.implementation_complexity.value,
            regulatory_compliance_impact=self._get_regulatory_impact(template)
        )
        
        # Create implementation scope for cost analysis
        implementation_scope = self._create_implementation_scope(template, risk, facility_context)
        
        # Perform cost-benefit analysis
        costs = self.cost_benefit_analyzer.estimate_costs(base_recommendation, implementation_scope)
        benefits = self.cost_benefit_analyzer.estimate_benefits(
            base_recommendation, risk, facility_context or {}
        )
        roi_metrics = self.cost_benefit_analyzer.calculate_roi(costs, benefits)
        
        # Create implementation plan
        phases, milestones = self.implementation_planner.create_implementation_plan(
            base_recommendation, template.implementation_complexity, timeline_constraints or {}
        )
        
        # Estimate resource requirements
        resource_requirements = self.implementation_planner.estimate_resource_requirements(
            milestones, template.name
        )
        
        # Create compliance requirements
        compliance_requirements = self._create_compliance_requirements(template, risk)
        
        # Create success metrics
        success_metrics = self._create_success_metrics(template, risk, base_recommendation)
        
        # Create enhanced recommendation
        enhanced_rec = EnhancedRecommendation(
            base_recommendation=base_recommendation,
            cost_estimates=costs,
            benefit_estimates=benefits,
            roi_calculation=roi_metrics,
            payback_period_months=roi_metrics.get("payback_period_years", 0) * 12,
            implementation_phases=phases,
            milestones=milestones,
            resource_requirements=resource_requirements,
            implementation_risks=template.common_challenges,
            success_factors=template.success_factors,
            compliance_requirements=compliance_requirements,
            regulatory_approval_needed=len(template.regulatory_frameworks) > 0,
            permit_requirements=self._get_permit_requirements(template),
            success_metrics=success_metrics,
            monitoring_frequency=self._determine_monitoring_frequency(template),
            key_stakeholders=self._identify_stakeholders(template, facility_context),
            communication_plan=self._create_communication_plan(template),
            training_requirements=self._get_training_requirements(template),
            implementation_contingencies=self._create_contingencies(template),
            fallback_options=self._create_fallback_options(template, risk)
        )
        
        return enhanced_rec
    
    def _customize_description(self, template: RecommendationTemplate, risk: RiskAssessment,
                             facility_context: Optional[Dict[str, Any]]) -> str:
        """Customize template description for specific risk and context"""
        
        description = template.description
        
        # Replace generic placeholders with specific information
        replacements = {
            "{risk_type}": str(risk.subcategory).replace("_", " ").title(),
            "{facility_type}": facility_context.get("facility_type", "facility") if facility_context else "facility",
            "{severity_level}": risk.severity.value,
            "{target_reduction}": str(int(template.effectiveness_rating * 100)),
            "{timeframe}": template.typical_implementation_time,
            "{industry}": self.industry.value.replace("_", " ").title()
        }
        
        for placeholder, replacement in replacements.items():
            description = description.replace(placeholder, replacement)
        
        # Add risk-specific context
        if risk.supporting_evidence:
            description += f" This recommendation addresses the following evidence: {'; '.join(risk.supporting_evidence[:2])}"
        
        return description
    
    def _determine_priority(self, template_priority: RecommendationPriority, risk: RiskAssessment) -> RecommendationPriority:
        """Determine final priority based on template and risk characteristics"""
        
        # Priority escalation based on risk severity
        if risk.severity == RiskSeverity.CRITICAL:
            return RecommendationPriority.IMMEDIATE
        elif risk.severity == RiskSeverity.HIGH and template_priority in [RecommendationPriority.MEDIUM, RecommendationPriority.LOW]:
            return RecommendationPriority.HIGH
        elif risk.timeframe == RiskTimeframe.IMMEDIATE:
            return min(RecommendationPriority.URGENT, template_priority, key=lambda x: x.value)
        
        return template_priority
    
    def _generate_resource_list(self, template: RecommendationTemplate) -> List[str]:
        """Generate resource requirements list from template"""
        
        resources = ["Project management", "Technical expertise"]
        
        if template.implementation_complexity in [ImplementationComplexity.HIGH, ImplementationComplexity.VERY_HIGH]:
            resources.extend([
                "Engineering design",
                "Construction/installation",
                "Specialized contractors"
            ])
        
        if template.regulatory_frameworks:
            resources.append("Regulatory compliance support")
        
        resources.append("Training and documentation")
        
        return resources
    
    def _determine_responsible_parties(self, template: RecommendationTemplate,
                                     facility_context: Optional[Dict[str, Any]]) -> List[str]:
        """Determine responsible parties for implementation"""
        
        parties = ["EHS Manager", "Facility Manager"]
        
        if template.implementation_complexity in [ImplementationComplexity.HIGH, ImplementationComplexity.VERY_HIGH]:
            parties.extend(["Engineering Manager", "Maintenance Manager"])
        
        if template.regulatory_frameworks:
            parties.append("Compliance Officer")
        
        if "training" in template.name.lower():
            parties.append("Training Coordinator")
        
        return parties
    
    def _get_regulatory_impact(self, template: RecommendationTemplate) -> List[str]:
        """Get regulatory compliance impact"""
        
        impacts = []
        
        for framework in template.regulatory_frameworks:
            impacts.append(f"Improves compliance with {framework.value.upper().replace('_', ' ')}")
        
        if not impacts:
            impacts.append("Supports overall regulatory compliance strategy")
        
        return impacts
    
    def _create_implementation_scope(self, template: RecommendationTemplate, risk: RiskAssessment,
                                   facility_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create implementation scope for cost analysis"""
        
        scope = {
            "template_type": template.template_id,
            "risk_severity": risk.severity.value,
            "complexity": template.implementation_complexity.value
        }
        
        # Add equipment requirements
        if "system" in template.name.lower() or "equipment" in template.name.lower():
            scope["equipment"] = [template.name.lower().replace(" ", "_")]
        
        # Add labor requirements
        complexity_hours = {
            ImplementationComplexity.LOW: 40,
            ImplementationComplexity.MEDIUM: 120,
            ImplementationComplexity.HIGH: 300,
            ImplementationComplexity.VERY_HIGH: 600
        }
        
        scope["labor_hours"] = {
            "engineering": complexity_hours.get(template.implementation_complexity, 120),
            "technician": complexity_hours.get(template.implementation_complexity, 120) // 2
        }
        
        # Add regulatory requirements
        if template.regulatory_frameworks:
            scope["regulatory_requirements"] = [f.value for f in template.regulatory_frameworks]
        
        # Add operational requirements
        scope["ongoing_operations"] = {
            "annual_low": 2000,
            "annual_high": 15000
        }
        
        return scope
    
    def _create_compliance_requirements(self, template: RecommendationTemplate,
                                      risk: RiskAssessment) -> List[ComplianceRequirement]:
        """Create compliance requirements for the recommendation"""
        
        requirements = []
        
        for framework in template.regulatory_frameworks:
            requirement = ComplianceRequirement(
                framework=framework,
                requirement_id=f"{framework.value}_{risk.risk_id}",
                title=f"{framework.value.upper().replace('_', ' ')} Compliance",
                description=f"Ensure compliance with {framework.value.replace('_', ' ').title()} requirements",
                compliance_deadline=datetime.now() + timedelta(days=365),
                current_compliance_status="partial",
                gap_analysis=[
                    "Current practices assessment needed",
                    "Documentation review required",
                    "Training gap identification"
                ],
                remediation_steps=[
                    "Conduct compliance audit",
                    "Develop corrective action plan",
                    "Implement required changes",
                    "Verify compliance through testing"
                ],
                enforcement_risk="medium"
            )
            requirements.append(requirement)
        
        return requirements
    
    def _create_success_metrics(self, template: RecommendationTemplate, risk: RiskAssessment,
                              recommendation: RiskMitigationRecommendation) -> List[SuccessMetric]:
        """Create success metrics for monitoring recommendation effectiveness"""
        
        metrics = []
        
        # Risk reduction metric
        metrics.append(SuccessMetric(
            metric_id=f"risk_reduction_{recommendation.recommendation_id}",
            name="Risk Score Reduction",
            description=f"Reduction in risk score for {risk.title}",
            measurement_method="Risk assessment scoring",
            target_value=f"{recommendation.risk_reduction_potential}% reduction",
            current_baseline=str(risk.risk_score),
            measurement_frequency="quarterly",
            data_source="Risk assessment database",
            responsible_party="EHS Manager",
            threshold_for_success=f"Achieve {recommendation.risk_reduction_potential * 0.8}% reduction minimum",
            leading_or_lagging="lagging"
        ))
        
        # Implementation progress metric
        metrics.append(SuccessMetric(
            metric_id=f"implementation_progress_{recommendation.recommendation_id}",
            name="Implementation Progress",
            description="Percentage of implementation milestones completed on time",
            measurement_method="Project tracking system",
            target_value="100% on-time completion",
            current_baseline="0%",
            measurement_frequency="monthly",
            data_source="Project management system",
            responsible_party="Project Manager",
            threshold_for_success="90% on-time milestone completion",
            leading_or_lagging="leading"
        ))
        
        # Cost effectiveness metric
        metrics.append(SuccessMetric(
            metric_id=f"cost_effectiveness_{recommendation.recommendation_id}",
            name="Cost Effectiveness",
            description="Actual costs vs. budgeted costs",
            measurement_method="Financial tracking",
            target_value="Within 110% of budget",
            current_baseline="0% (not started)",
            measurement_frequency="monthly",
            data_source="Financial management system",
            responsible_party="Finance Manager",
            threshold_for_success="Complete within 110% of approved budget",
            leading_or_lagging="concurrent"
        ))
        
        # Add category-specific metrics
        if risk.category == RiskCategory.ENVIRONMENTAL:
            metrics.append(SuccessMetric(
                metric_id=f"environmental_impact_{recommendation.recommendation_id}",
                name="Environmental Impact Reduction",
                description="Measurable reduction in environmental impact",
                measurement_method="Environmental monitoring",
                target_value="Achieve permit compliance limits",
                measurement_frequency="monthly",
                data_source="Environmental monitoring system",
                responsible_party="Environmental Manager",
                threshold_for_success="Maintain compliance for 12 consecutive months",
                leading_or_lagging="lagging"
            ))
        
        return metrics
    
    def _determine_monitoring_frequency(self, template: RecommendationTemplate) -> str:
        """Determine appropriate monitoring frequency"""
        
        if template.implementation_complexity == ImplementationComplexity.VERY_HIGH:
            return "weekly"
        elif template.implementation_complexity == ImplementationComplexity.HIGH:
            return "bi-weekly"
        else:
            return "monthly"
    
    def _identify_stakeholders(self, template: RecommendationTemplate,
                             facility_context: Optional[Dict[str, Any]]) -> List[str]:
        """Identify key stakeholders for the recommendation"""
        
        stakeholders = [
            "EHS Manager",
            "Facility Manager",
            "Operations Manager",
            "Finance Manager"
        ]
        
        if template.implementation_complexity in [ImplementationComplexity.HIGH, ImplementationComplexity.VERY_HIGH]:
            stakeholders.extend([
                "Engineering Manager",
                "Maintenance Manager",
                "Senior Management"
            ])
        
        if template.regulatory_frameworks:
            stakeholders.extend([
                "Compliance Officer",
                "Legal Department"
            ])
        
        if "training" in template.name.lower():
            stakeholders.append("HR Manager")
        
        return stakeholders
    
    def _create_communication_plan(self, template: RecommendationTemplate) -> Dict[str, Any]:
        """Create communication plan for recommendation implementation"""
        
        return {
            "kick_off_meeting": {
                "participants": ["All stakeholders"],
                "agenda": ["Project overview", "Roles and responsibilities", "Timeline review"],
                "frequency": "One-time"
            },
            "progress_updates": {
                "participants": ["Project team", "Management"],
                "format": "Status report",
                "frequency": "Monthly"
            },
            "milestone_reviews": {
                "participants": ["Key stakeholders"],
                "format": "Review meeting",
                "frequency": "At each milestone"
            },
            "completion_report": {
                "participants": ["All stakeholders"],
                "format": "Final report and presentation",
                "frequency": "Project completion"
            }
        }
    
    def _get_training_requirements(self, template: RecommendationTemplate) -> List[str]:
        """Get training requirements for recommendation implementation"""
        
        training = ["General project awareness"]
        
        if "equipment" in template.name.lower() or "system" in template.name.lower():
            training.extend([
                "Equipment operation training",
                "Maintenance procedures",
                "Troubleshooting"
            ])
        
        if template.regulatory_frameworks:
            training.extend([
                "Regulatory requirements",
                "Compliance procedures",
                "Documentation requirements"
            ])
        
        if "safety" in template.name.lower():
            training.extend([
                "Safety procedures",
                "Emergency response",
                "Incident reporting"
            ])
        
        return training
    
    def _create_contingencies(self, template: RecommendationTemplate) -> List[str]:
        """Create implementation contingency plans"""
        
        contingencies = [
            "Budget overrun mitigation plan",
            "Schedule delay recovery plan",
            "Resource availability backup plan"
        ]
        
        if template.implementation_complexity in [ImplementationComplexity.HIGH, ImplementationComplexity.VERY_HIGH]:
            contingencies.extend([
                "Technical implementation failure plan",
                "Vendor/contractor performance issues plan",
                "Regulatory approval delay plan"
            ])
        
        return contingencies
    
    def _create_fallback_options(self, template: RecommendationTemplate, risk: RiskAssessment) -> List[str]:
        """Create fallback options if primary recommendation fails"""
        
        fallbacks = [
            "Interim risk mitigation measures",
            "Alternative technology solutions",
            "Enhanced monitoring and controls"
        ]
        
        if risk.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]:
            fallbacks.extend([
                "Operational restrictions",
                "Additional safety measures",
                "Emergency response enhancements"
            ])
        
        return fallbacks
    
    def _get_permit_requirements(self, template: RecommendationTemplate) -> List[str]:
        """Get permit requirements for recommendation"""
        
        permits = []
        
        for framework in template.regulatory_frameworks:
            if framework in [RegulatoryFramework.EPA, RegulatoryFramework.CLEAN_AIR_ACT]:
                permits.append("Air quality permit modification")
            elif framework in [RegulatoryFramework.CLEAN_WATER_ACT]:
                permits.append("Water discharge permit update")
            elif framework == RegulatoryFramework.RCRA:
                permits.append("Waste management permit")
        
        return permits
    
    def _create_generic_recommendation(self, risk: RiskAssessment) -> EnhancedRecommendation:
        """Create generic recommendation when no templates match"""
        
        base_recommendation = RiskMitigationRecommendation(
            risk_id=risk.risk_id,
            title=f"Risk Mitigation for {risk.title}",
            description=f"Implement appropriate risk mitigation measures for {risk.title}",
            priority=RecommendationPriority.MEDIUM,
            implementation_timeframe="6-12 months",
            estimated_cost=50000.0,
            resource_requirements=["Risk assessment", "Technical evaluation", "Implementation planning"],
            responsible_parties=["EHS Manager", "Facility Manager"],
            risk_reduction_potential=60.0,
            implementation_complexity="medium"
        )
        
        return EnhancedRecommendation(
            base_recommendation=base_recommendation,
            cost_estimates=[CostEstimate(CostCategory.OPERATIONAL, 25000, 75000)],
            benefit_estimates=[],
            implementation_phases=["Assessment", "Planning", "Implementation", "Verification"],
            success_factors=["Proper planning", "Stakeholder engagement", "Regular monitoring"],
            key_stakeholders=["EHS Manager", "Operations Manager"]
        )
    
    def _prioritize_recommendations(self, recommendations: List[EnhancedRecommendation]) -> List[EnhancedRecommendation]:
        """Prioritize recommendations based on multiple criteria"""
        
        def priority_score(rec: EnhancedRecommendation) -> float:
            """Calculate priority score for a recommendation"""
            
            # Priority level scoring
            priority_scores = {
                RecommendationPriority.IMMEDIATE: 100,
                RecommendationPriority.URGENT: 80,
                RecommendationPriority.HIGH: 60,
                RecommendationPriority.MEDIUM: 40,
                RecommendationPriority.LOW: 20,
                RecommendationPriority.DEFERRED: 10
            }
            
            base_score = priority_scores.get(rec.base_recommendation.priority, 40)
            
            # ROI adjustment
            if rec.roi_calculation and rec.roi_calculation.get("roi_percent", 0) > 0:
                roi_bonus = min(20, rec.roi_calculation["roi_percent"] / 10)
                base_score += roi_bonus
            
            # Payback period adjustment
            if rec.payback_period_months and rec.payback_period_months < 24:
                payback_bonus = max(0, 20 - rec.payback_period_months / 2)
                base_score += payback_bonus
            
            # Risk reduction potential
            risk_reduction_bonus = (rec.base_recommendation.risk_reduction_potential / 100) * 15
            base_score += risk_reduction_bonus
            
            return base_score
        
        # Sort by priority score (descending)
        return sorted(recommendations, key=priority_score, reverse=True)
    
    def _optimize_recommendations(self, recommendations: List[EnhancedRecommendation]) -> List[EnhancedRecommendation]:
        """Optimize recommendations by identifying synergies and consolidation opportunities"""
        
        optimized = recommendations.copy()
        
        # Look for consolidation opportunities
        # (This is a simplified implementation - in practice, would use more sophisticated algorithms)
        
        # Group recommendations by category or type
        grouped = {}
        for rec in recommendations:
            category = rec.base_recommendation.risk_id[:3]  # Simple grouping by risk ID prefix
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(rec)
        
        # For each group with multiple recommendations, check for consolidation
        for category, recs in grouped.items():
            if len(recs) > 1:
                # Check if recommendations can be combined
                combined_rec = self._try_combine_recommendations(recs)
                if combined_rec:
                    # Remove individual recommendations and add combined one
                    for rec in recs:
                        if rec in optimized:
                            optimized.remove(rec)
                    optimized.append(combined_rec)
        
        return self._prioritize_recommendations(optimized)
    
    def _try_combine_recommendations(self, recommendations: List[EnhancedRecommendation]) -> Optional[EnhancedRecommendation]:
        """Attempt to combine similar recommendations"""
        
        if len(recommendations) < 2:
            return None
        
        # Simple heuristic: if recommendations have similar titles or implementation phases, they might be combinable
        first_rec = recommendations[0]
        
        # Check if all recommendations have similar characteristics
        similar_phases = all(
            set(rec.implementation_phases) & set(first_rec.implementation_phases)
            for rec in recommendations[1:]
        )
        
        if not similar_phases:
            return None
        
        # Create combined recommendation
        combined_title = f"Integrated Risk Mitigation Program ({len(recommendations)} initiatives)"
        combined_description = f"Combined implementation of {len(recommendations)} related risk mitigation measures"
        
        # Combine costs
        all_costs = []
        for rec in recommendations:
            all_costs.extend(rec.cost_estimates)
        
        # Combine benefits
        all_benefits = []
        for rec in recommendations:
            all_benefits.extend(rec.benefit_estimates)
        
        # Create new base recommendation
        base_rec = RiskMitigationRecommendation(
            risk_id="COMBINED_" + str(uuid4())[:8],
            title=combined_title,
            description=combined_description,
            priority=max(rec.base_recommendation.priority for rec in recommendations),
            implementation_timeframe=max(rec.base_recommendation.implementation_timeframe for rec in recommendations),
            estimated_cost=sum(rec.base_recommendation.estimated_cost or 0 for rec in recommendations),
            risk_reduction_potential=min(100, sum(rec.base_recommendation.risk_reduction_potential for rec in recommendations) / len(recommendations))
        )
        
        # Create combined enhanced recommendation
        combined_rec = EnhancedRecommendation(
            base_recommendation=base_rec,
            cost_estimates=all_costs,
            benefit_estimates=all_benefits,
            roi_calculation=self.cost_benefit_analyzer.calculate_roi(all_costs, all_benefits),
            implementation_phases=list(set().union(*(rec.implementation_phases for rec in recommendations))),
            success_factors=list(set().union(*(rec.success_factors for rec in recommendations))),
            key_stakeholders=list(set().union(*(rec.key_stakeholders for rec in recommendations)))
        )
        
        return combined_rec
    
    def generate_implementation_roadmap(self, recommendations: List[EnhancedRecommendation],
                                      total_budget: Optional[float] = None,
                                      implementation_horizon_months: int = 24) -> Dict[str, Any]:
        """Generate comprehensive implementation roadmap for all recommendations"""
        
        roadmap = {
            "summary": {
                "total_recommendations": len(recommendations),
                "implementation_period_months": implementation_horizon_months,
                "total_budget_estimate": total_budget,
                "created_date": datetime.now().isoformat()
            },
            "phases": [],
            "resource_allocation": {},
            "timeline": {},
            "budget_allocation": {},
            "success_metrics": [],
            "risk_mitigation_timeline": {}
        }
        
        # Group recommendations by priority and implementation timeframe
        immediate = [r for r in recommendations if r.base_recommendation.priority == RecommendationPriority.IMMEDIATE]
        urgent = [r for r in recommendations if r.base_recommendation.priority == RecommendationPriority.URGENT]
        high = [r for r in recommendations if r.base_recommendation.priority == RecommendationPriority.HIGH]
        medium = [r for r in recommendations if r.base_recommendation.priority == RecommendationPriority.MEDIUM]
        low = [r for r in recommendations if r.base_recommendation.priority == RecommendationPriority.LOW]
        
        # Create implementation phases
        if immediate:
            roadmap["phases"].append({
                "phase": "Immediate Actions (0-1 months)",
                "recommendations": [r.base_recommendation.title for r in immediate],
                "total_cost": sum(r.total_cost_estimate.mid_estimate for r in immediate),
                "expected_risk_reduction": sum(r.base_recommendation.risk_reduction_potential for r in immediate) / len(immediate)
            })
        
        if urgent:
            roadmap["phases"].append({
                "phase": "Urgent Actions (1-3 months)",
                "recommendations": [r.base_recommendation.title for r in urgent],
                "total_cost": sum(r.total_cost_estimate.mid_estimate for r in urgent),
                "expected_risk_reduction": sum(r.base_recommendation.risk_reduction_potential for r in urgent) / len(urgent)
            })
        
        if high:
            roadmap["phases"].append({
                "phase": "High Priority (3-12 months)",
                "recommendations": [r.base_recommendation.title for r in high],
                "total_cost": sum(r.total_cost_estimate.mid_estimate for r in high),
                "expected_risk_reduction": sum(r.base_recommendation.risk_reduction_potential for r in high) / len(high)
            })
        
        if medium:
            roadmap["phases"].append({
                "phase": "Medium Priority (12-18 months)",
                "recommendations": [r.base_recommendation.title for r in medium],
                "total_cost": sum(r.total_cost_estimate.mid_estimate for r in medium),
                "expected_risk_reduction": sum(r.base_recommendation.risk_reduction_potential for r in medium) / len(medium)
            })
        
        if low:
            roadmap["phases"].append({
                "phase": "Low Priority (18-24+ months)",
                "recommendations": [r.base_recommendation.title for r in low],
                "total_cost": sum(r.total_cost_estimate.mid_estimate for r in low),
                "expected_risk_reduction": sum(r.base_recommendation.risk_reduction_potential for r in low) / len(low)
            })
        
        # Calculate resource allocation
        all_resources = []
        for rec in recommendations:
            all_resources.extend(rec.resource_requirements)
        
        resource_summary = {}
        for resource in all_resources:
            if resource.resource_type not in resource_summary:
                resource_summary[resource.resource_type] = {
                    "total_quantity": 0,
                    "total_duration_months": 0,
                    "specialized_skills": set()
                }
            
            resource_summary[resource.resource_type]["total_quantity"] += resource.quantity
            if resource.duration_months:
                resource_summary[resource.resource_type]["total_duration_months"] += resource.duration_months
            resource_summary[resource.resource_type]["specialized_skills"].update(resource.specialized_skills_required)
        
        # Convert sets to lists for JSON serialization
        for resource_type, data in resource_summary.items():
            data["specialized_skills"] = list(data["specialized_skills"])
        
        roadmap["resource_allocation"] = resource_summary
        
        # Calculate total budget
        total_cost = sum(r.total_cost_estimate.mid_estimate for r in recommendations)
        roadmap["summary"]["total_cost_estimate"] = total_cost
        
        # Budget allocation by phase
        for phase in roadmap["phases"]:
            roadmap["budget_allocation"][phase["phase"]] = phase["total_cost"]
        
        # Aggregate success metrics
        all_metrics = []
        for rec in recommendations:
            all_metrics.extend(rec.success_metrics)
        
        roadmap["success_metrics"] = [
            {
                "metric_name": metric.name,
                "target_value": metric.target_value,
                "measurement_frequency": metric.measurement_frequency,
                "responsible_party": metric.responsible_party
            }
            for metric in all_metrics[:10]  # Limit to top 10 metrics
        ]
        
        return roadmap


# Example usage and testing
if __name__ == "__main__":
    
    # Create sample risk assessment
    sample_risk = RiskAssessment(
        category=RiskCategory.ENVIRONMENTAL,
        subcategory=EnvironmentalRiskType.AIR_EMISSIONS,
        title="Elevated NOx Emissions from Boiler Stack",
        description="Facility boiler stack emissions exceed permitted NOx levels during peak operations",
        severity=RiskSeverity.HIGH,
        probability=RiskProbability.HIGH,
        risk_score=16.0,
        timeframe=RiskTimeframe.SHORT_TERM,
        supporting_evidence=[
            "Quarterly emissions monitoring report shows 150 ppm NOx (permit limit: 120 ppm)",
            "Stack testing conducted by certified third party",
            "Exceedance occurred during 3 of last 4 peak demand periods"
        ],
        regulatory_context=[
            "EPA Clean Air Act permit requirements",
            "State DEQ air quality regulations",
            "NOx RACT requirements"
        ],
        confidence_score=0.9
    )
    
    # Initialize recommendation generator
    generator = EHSRecommendationGenerator(industry=IndustryType.MANUFACTURING)
    
    # Generate recommendations
    recommendations = generator.generate_recommendations(
        risks=[sample_risk],
        facility_context={
            "facility_type": "Manufacturing Plant",
            "annual_operations_cost": 5000000,
            "employee_count": 200,
            "location": "Industrial Zone"
        },
        budget_constraints={
            "annual_budget": 500000,
            "project_budget_limit": 250000
        },
        timeline_constraints={
            "urgent_deadline": datetime.now() + timedelta(days=90),
            "compliance_deadline": datetime.now() + timedelta(days=180)
        }
    )
    
    # Generate implementation roadmap
    roadmap = generator.generate_implementation_roadmap(recommendations, total_budget=500000)
    
    # Print results
    print(f"\nGenerated {len(recommendations)} recommendations for NOx emissions risk:")
    print("="*80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.base_recommendation.title}")
        print(f"   Priority: {rec.base_recommendation.priority.value}")
        print(f"   Cost Estimate: ${rec.total_cost_estimate.low_estimate:,.0f} - ${rec.total_cost_estimate.high_estimate:,.0f}")
        print(f"   Risk Reduction: {rec.base_recommendation.risk_reduction_potential:.0f}%")
        print(f"   Implementation Time: {rec.base_recommendation.implementation_timeframe}")
        
        if rec.roi_calculation:
            print(f"   ROI: {rec.roi_calculation.get('roi_percent', 0):.1f}%")
            print(f"   Payback: {rec.roi_calculation.get('payback_period_years', 0):.1f} years")
        
        print(f"   Phases: {', '.join(rec.implementation_phases[:3])}...")
        print(f"   Key Stakeholders: {', '.join(rec.key_stakeholders[:3])}")
    
    print(f"\nImplementation Roadmap Summary:")
    print(f"Total Cost Estimate: ${roadmap['summary']['total_cost_estimate']:,.0f}")
    print(f"Implementation Period: {roadmap['summary']['implementation_period_months']} months")
    print(f"Number of Phases: {len(roadmap['phases'])}")
    
    for phase in roadmap["phases"]:
        print(f"\n{phase['phase']}: ${phase['total_cost']:,.0f} ({len(phase['recommendations'])} recommendations)")
