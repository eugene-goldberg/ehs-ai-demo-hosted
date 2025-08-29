"""
EHS Risk Assessment Engine

A comprehensive risk assessment engine for Environmental, Health, and Safety (EHS) applications.
Implements industry-standard methodologies including ISO 31000, OHSAS 18001, and regulatory frameworks.

Author: AI Assistant
Created: 2025-08-28
Version: 1.0.0
"""

import logging
import math
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    """EHS Risk Categories"""
    ENVIRONMENTAL = "environmental"
    HEALTH = "health"
    SAFETY = "safety"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"


class RiskLevel(Enum):
    """Risk Level Classifications"""
    VERY_LOW = (1, "Very Low", "#00FF00")
    LOW = (2, "Low", "#90EE90")
    MODERATE = (3, "Moderate", "#FFFF00")
    HIGH = (4, "High", "#FFA500")
    VERY_HIGH = (5, "Very High", "#FF0000")
    CRITICAL = (6, "Critical", "#8B0000")

    def __init__(self, value, label, color):
        self.value = value
        self.label = label
        self.color = color


class ProbabilityLevel(Enum):
    """Probability Classifications"""
    RARE = (1, "Rare", 0.05, "Once in 10+ years")
    UNLIKELY = (2, "Unlikely", 0.15, "Once in 5-10 years")
    POSSIBLE = (3, "Possible", 0.35, "Once in 2-5 years")
    LIKELY = (4, "Likely", 0.65, "Once per year")
    ALMOST_CERTAIN = (5, "Almost Certain", 0.85, "Multiple times per year")

    def __init__(self, value, label, probability, description):
        self.value = value
        self.label = label
        self.probability = probability
        self.description = description


class SeverityLevel(Enum):
    """Severity Classifications"""
    NEGLIGIBLE = (1, "Negligible", "Minor inconvenience")
    MINOR = (2, "Minor", "Some disruption, minor injury")
    MODERATE = (3, "Moderate", "Significant disruption, medical treatment")
    MAJOR = (4, "Major", "Severe disruption, hospitalization")
    CATASTROPHIC = (5, "Catastrophic", "Fatality, major environmental damage")

    def __init__(self, value, label, description):
        self.value = value
        self.label = label
        self.description = description


class RegulatoryFramework(Enum):
    """Regulatory Framework Types"""
    OSHA = "osha"
    EPA = "epa"
    ISO_14001 = "iso_14001"
    ISO_45001 = "iso_45001"
    ISO_31000 = "iso_31000"
    OHSAS_18001 = "ohsas_18001"
    DOT = "dot"
    FDA = "fda"
    EU_REACH = "eu_reach"


class IndustryType(Enum):
    """Industry Classification"""
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


@dataclass
class RiskFactor:
    """Individual risk factor data structure"""
    id: str
    name: str
    category: RiskCategory
    probability: ProbabilityLevel
    severity: SeverityLevel
    current_controls: List[str] = field(default_factory=list)
    control_effectiveness: float = 0.0  # 0-1 scale
    regulatory_requirements: List[RegulatoryFramework] = field(default_factory=list)
    environmental_impact_score: float = 0.0
    health_impact_score: float = 0.0
    safety_impact_score: float = 0.0
    financial_impact: float = 0.0
    last_incident_date: Optional[datetime] = None
    trend_direction: str = "stable"  # increasing, decreasing, stable
    confidence_level: float = 0.8  # 0-1 scale
    data_quality: str = "good"  # poor, fair, good, excellent
    stakeholder_concern: float = 0.0  # 0-1 scale
    media_attention: float = 0.0  # 0-1 scale


@dataclass
class RiskAssessmentResult:
    """Risk assessment calculation results"""
    risk_factor: RiskFactor
    inherent_risk_score: float
    residual_risk_score: float
    risk_level: RiskLevel
    risk_matrix_position: Tuple[int, int]
    regulatory_compliance_score: float
    environmental_score: float
    health_safety_score: float
    trend_adjusted_score: float
    aggregated_score: float
    recommendations: List[str]
    next_review_date: datetime
    assessment_confidence: float
    monte_carlo_results: Optional[Dict] = None


@dataclass
class IndustryBenchmark:
    """Industry benchmark data"""
    industry: IndustryType
    risk_category: RiskCategory
    average_score: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    best_practice_score: float
    sample_size: int
    last_updated: datetime


class RiskAssessmentFramework(ABC):
    """Abstract base class for risk assessment frameworks"""
    
    @abstractmethod
    def calculate_risk_score(self, risk_factor: RiskFactor) -> float:
        pass
    
    @abstractmethod
    def get_risk_matrix(self) -> np.ndarray:
        pass


class ISO31000Framework(RiskAssessmentFramework):
    """ISO 31000 Risk Management Framework Implementation"""
    
    def __init__(self):
        self.risk_matrix = np.array([
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10],
            [3, 6, 9, 12, 15],
            [4, 8, 12, 16, 20],
            [5, 10, 15, 20, 25]
        ])
    
    def calculate_risk_score(self, risk_factor: RiskFactor) -> float:
        """Calculate risk score using ISO 31000 methodology"""
        base_score = risk_factor.probability.value * risk_factor.severity.value
        
        # Apply control effectiveness reduction
        controlled_score = base_score * (1 - risk_factor.control_effectiveness)
        
        # Apply confidence adjustment
        confidence_factor = 1 + (1 - risk_factor.confidence_level) * 0.2
        
        return controlled_score * confidence_factor
    
    def get_risk_matrix(self) -> np.ndarray:
        return self.risk_matrix


class OHSAS18001Framework(RiskAssessmentFramework):
    """OHSAS 18001 Occupational Health and Safety Framework"""
    
    def __init__(self):
        self.risk_matrix = np.array([
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9]
        ])
    
    def calculate_risk_score(self, risk_factor: RiskFactor) -> float:
        """Calculate risk score with OHSAS 18001 emphasis on health and safety"""
        base_score = risk_factor.probability.value * risk_factor.severity.value
        
        # Higher weighting for health and safety categories
        if risk_factor.category in [RiskCategory.HEALTH, RiskCategory.SAFETY]:
            base_score *= 1.2
        
        # Apply control effectiveness
        controlled_score = base_score * (1 - risk_factor.control_effectiveness)
        
        return min(controlled_score, 25)  # Cap at maximum matrix value
    
    def get_risk_matrix(self) -> np.ndarray:
        return self.risk_matrix


class EHSRiskEngine:
    """
    Comprehensive EHS Risk Assessment Engine
    
    Implements multiple risk assessment methodologies with industry-specific
    calculations and benchmarking capabilities.
    """
    
    def __init__(self, 
                 framework: RiskAssessmentFramework = None,
                 industry: IndustryType = IndustryType.MANUFACTURING):
        """
        Initialize the EHS Risk Engine
        
        Args:
            framework: Risk assessment framework to use
            industry: Industry type for benchmarking
        """
        self.framework = framework or ISO31000Framework()
        self.industry = industry
        self.benchmarks: Dict[str, IndustryBenchmark] = {}
        self.risk_weights = self._get_default_weights()
        self.monte_carlo_iterations = 10000
        
        # Load default industry benchmarks
        self._load_industry_benchmarks()
        
        logger.info(f"EHS Risk Engine initialized with {framework.__class__.__name__} framework")
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default weights for multi-dimensional risk aggregation"""
        return {
            'inherent_risk': 0.25,
            'regulatory_compliance': 0.20,
            'environmental_impact': 0.20,
            'health_safety': 0.20,
            'trend_adjustment': 0.10,
            'stakeholder_concern': 0.05
        }
    
    def _load_industry_benchmarks(self):
        """Load industry benchmark data (placeholder for actual data loading)"""
        # In production, this would load from database or external data source
        sample_benchmarks = {
            f"{self.industry.value}_{RiskCategory.ENVIRONMENTAL.value}": IndustryBenchmark(
                industry=self.industry,
                risk_category=RiskCategory.ENVIRONMENTAL,
                average_score=12.5,
                percentile_25=8.0,
                percentile_75=16.0,
                percentile_90=20.0,
                best_practice_score=5.0,
                sample_size=1000,
                last_updated=datetime.now()
            )
        }
        self.benchmarks.update(sample_benchmarks)
    
    def assess_risk(self, risk_factor: RiskFactor) -> RiskAssessmentResult:
        """
        Perform comprehensive risk assessment
        
        Args:
            risk_factor: Risk factor to assess
            
        Returns:
            Complete risk assessment results
        """
        logger.info(f"Assessing risk for: {risk_factor.name}")
        
        # 1. Calculate inherent risk score
        inherent_score = self._calculate_inherent_risk(risk_factor)
        
        # 2. Calculate residual risk score (after controls)
        residual_score = self._calculate_residual_risk(risk_factor, inherent_score)
        
        # 3. Determine risk level and matrix position
        risk_level, matrix_position = self._determine_risk_level(residual_score)
        
        # 4. Calculate regulatory compliance score
        compliance_score = self._calculate_regulatory_compliance(risk_factor)
        
        # 5. Calculate environmental impact score
        environmental_score = self._calculate_environmental_impact(risk_factor)
        
        # 6. Calculate health and safety score
        health_safety_score = self._calculate_health_safety_score(risk_factor)
        
        # 7. Apply trend-based adjustments
        trend_adjusted_score = self._apply_trend_adjustments(residual_score, risk_factor)
        
        # 8. Calculate aggregated multi-dimensional score
        aggregated_score = self._calculate_aggregated_score(
            inherent_score, compliance_score, environmental_score,
            health_safety_score, trend_adjusted_score, risk_factor
        )
        
        # 9. Generate recommendations
        recommendations = self._generate_recommendations(risk_factor, aggregated_score)
        
        # 10. Calculate assessment confidence
        assessment_confidence = self._calculate_assessment_confidence(risk_factor)
        
        # 11. Run Monte Carlo simulation if requested
        monte_carlo_results = self._run_monte_carlo_simulation(risk_factor)
        
        # 12. Set next review date
        next_review = self._calculate_next_review_date(risk_level, risk_factor)
        
        result = RiskAssessmentResult(
            risk_factor=risk_factor,
            inherent_risk_score=inherent_score,
            residual_risk_score=residual_score,
            risk_level=risk_level,
            risk_matrix_position=matrix_position,
            regulatory_compliance_score=compliance_score,
            environmental_score=environmental_score,
            health_safety_score=health_safety_score,
            trend_adjusted_score=trend_adjusted_score,
            aggregated_score=aggregated_score,
            recommendations=recommendations,
            next_review_date=next_review,
            assessment_confidence=assessment_confidence,
            monte_carlo_results=monte_carlo_results
        )
        
        logger.info(f"Risk assessment completed. Aggregated score: {aggregated_score:.2f}")
        return result
    
    def _calculate_inherent_risk(self, risk_factor: RiskFactor) -> float:
        """Calculate inherent risk score (before controls)"""
        return self.framework.calculate_risk_score(risk_factor)
    
    def _calculate_residual_risk(self, risk_factor: RiskFactor, inherent_score: float) -> float:
        """Calculate residual risk score (after applying controls)"""
        # Base residual calculation
        residual = inherent_score * (1 - risk_factor.control_effectiveness)
        
        # Apply data quality adjustment
        quality_multipliers = {
            'poor': 1.3,
            'fair': 1.1,
            'good': 1.0,
            'excellent': 0.95
        }
        residual *= quality_multipliers.get(risk_factor.data_quality, 1.0)
        
        return residual
    
    def _determine_risk_level(self, risk_score: float) -> Tuple[RiskLevel, Tuple[int, int]]:
        """Determine risk level and matrix position based on score"""
        # Risk level thresholds
        if risk_score <= 2:
            level = RiskLevel.VERY_LOW
        elif risk_score <= 4:
            level = RiskLevel.LOW
        elif risk_score <= 9:
            level = RiskLevel.MODERATE
        elif risk_score <= 16:
            level = RiskLevel.HIGH
        elif risk_score <= 20:
            level = RiskLevel.VERY_HIGH
        else:
            level = RiskLevel.CRITICAL
        
        # Calculate matrix position (simplified)
        probability_pos = min(int(math.sqrt(risk_score)), 4)
        severity_pos = min(int(risk_score / (probability_pos + 1)), 4)
        
        return level, (probability_pos, severity_pos)
    
    def _calculate_regulatory_compliance(self, risk_factor: RiskFactor) -> float:
        """Calculate regulatory compliance score"""
        if not risk_factor.regulatory_requirements:
            return 100.0  # No specific requirements
        
        # Base compliance calculation
        base_compliance = 85.0  # Assume reasonable baseline
        
        # Adjust based on number of regulations
        regulation_penalty = len(risk_factor.regulatory_requirements) * 2
        
        # Adjust based on control effectiveness
        control_bonus = risk_factor.control_effectiveness * 15
        
        compliance_score = base_compliance - regulation_penalty + control_bonus
        
        return max(0, min(100, compliance_score))
    
    def _calculate_environmental_impact(self, risk_factor: RiskFactor) -> float:
        """Calculate environmental impact score using lifecycle assessment principles"""
        base_score = risk_factor.environmental_impact_score
        
        # Apply category-specific multipliers
        if risk_factor.category == RiskCategory.ENVIRONMENTAL:
            base_score *= 1.5
        
        # Consider ecosystem impact
        ecosystem_factors = [
            'air_quality',
            'water_quality',
            'soil_contamination',
            'biodiversity',
            'climate_change',
            'resource_depletion'
        ]
        
        # Weighted environmental calculation (simplified)
        environmental_score = base_score * (1 + risk_factor.stakeholder_concern * 0.2)
        
        return min(25, environmental_score)
    
    def _calculate_health_safety_score(self, risk_factor: RiskFactor) -> float:
        """Calculate health and safety metrics based on OSHA/ISO standards"""
        health_score = risk_factor.health_impact_score
        safety_score = risk_factor.safety_impact_score
        
        # OSHA recordable incident weighting
        if risk_factor.last_incident_date:
            days_since_incident = (datetime.now() - risk_factor.last_incident_date).days
            incident_factor = max(0.1, 1 - (days_since_incident / 365))
            health_score *= (1 + incident_factor)
            safety_score *= (1 + incident_factor)
        
        # Combine health and safety scores
        combined_score = (health_score + safety_score) / 2
        
        # Apply severity weighting for health/safety categories
        if risk_factor.category in [RiskCategory.HEALTH, RiskCategory.SAFETY]:
            combined_score *= 1.25
        
        return min(25, combined_score)
    
    def _apply_trend_adjustments(self, base_score: float, risk_factor: RiskFactor) -> float:
        """Apply trend-based risk adjustments with temporal analysis"""
        trend_multipliers = {
            'increasing': 1.3,
            'stable': 1.0,
            'decreasing': 0.8
        }
        
        trend_factor = trend_multipliers.get(risk_factor.trend_direction, 1.0)
        
        # Consider media attention as trend amplifier
        media_amplifier = 1 + (risk_factor.media_attention * 0.1)
        
        # Apply stakeholder concern
        stakeholder_amplifier = 1 + (risk_factor.stakeholder_concern * 0.15)
        
        adjusted_score = base_score * trend_factor * media_amplifier * stakeholder_amplifier
        
        return adjusted_score
    
    def _calculate_aggregated_score(self, inherent: float, compliance: float, 
                                   environmental: float, health_safety: float,
                                   trend_adjusted: float, risk_factor: RiskFactor) -> float:
        """Calculate multi-dimensional risk aggregation with weighted scoring"""
        
        # Normalize scores to same scale (0-25)
        normalized_compliance = (compliance / 100) * 25
        
        # Apply weights
        weighted_score = (
            self.risk_weights['inherent_risk'] * inherent +
            self.risk_weights['regulatory_compliance'] * (25 - normalized_compliance) +  # Invert compliance
            self.risk_weights['environmental_impact'] * environmental +
            self.risk_weights['health_safety'] * health_safety +
            self.risk_weights['trend_adjustment'] * (trend_adjusted - inherent) +
            self.risk_weights['stakeholder_concern'] * (risk_factor.stakeholder_concern * 25)
        )
        
        return max(0, weighted_score)
    
    def _generate_recommendations(self, risk_factor: RiskFactor, aggregated_score: float) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        # Risk level specific recommendations
        if aggregated_score > 20:
            recommendations.append("CRITICAL: Implement immediate risk mitigation measures")
            recommendations.append("Consider temporary shutdown/suspension of activities")
        elif aggregated_score > 16:
            recommendations.append("HIGH PRIORITY: Develop comprehensive mitigation plan within 30 days")
            recommendations.append("Increase monitoring and inspection frequency")
        elif aggregated_score > 9:
            recommendations.append("MODERATE: Review and enhance existing controls")
            recommendations.append("Conduct quarterly risk assessment reviews")
        
        # Control effectiveness recommendations
        if risk_factor.control_effectiveness < 0.5:
            recommendations.append("Improve control effectiveness through training and procedures")
        
        # Regulatory compliance recommendations
        if risk_factor.regulatory_requirements:
            recommendations.append("Ensure full compliance with applicable regulations")
            recommendations.append("Consider third-party compliance audit")
        
        # Category-specific recommendations
        if risk_factor.category == RiskCategory.ENVIRONMENTAL:
            recommendations.append("Conduct environmental impact assessment")
            recommendations.append("Implement environmental management system")
        elif risk_factor.category == RiskCategory.HEALTH:
            recommendations.append("Enhance health monitoring programs")
            recommendations.append("Provide health and safety training")
        elif risk_factor.category == RiskCategory.SAFETY:
            recommendations.append("Review and update safety procedures")
            recommendations.append("Conduct safety culture assessment")
        
        # Trend-based recommendations
        if risk_factor.trend_direction == 'increasing':
            recommendations.append("Investigate root causes of increasing risk trend")
            recommendations.append("Implement additional preventive measures")
        
        return recommendations
    
    def _calculate_assessment_confidence(self, risk_factor: RiskFactor) -> float:
        """Calculate overall assessment confidence based on data quality and completeness"""
        
        # Base confidence from risk factor
        base_confidence = risk_factor.confidence_level
        
        # Data quality adjustment
        quality_adjustments = {
            'poor': -0.3,
            'fair': -0.1,
            'good': 0.0,
            'excellent': 0.1
        }
        
        confidence = base_confidence + quality_adjustments.get(risk_factor.data_quality, 0)
        
        # Control data availability
        if risk_factor.current_controls:
            confidence += 0.05
        
        # Historical data availability
        if risk_factor.last_incident_date:
            confidence += 0.05
        
        # Regulatory clarity
        if risk_factor.regulatory_requirements:
            confidence += 0.05
        
        return max(0, min(1, confidence))
    
    def _run_monte_carlo_simulation(self, risk_factor: RiskFactor) -> Dict:
        """Run Monte Carlo simulation for uncertainty analysis"""
        
        # Generate probability distributions
        prob_samples = np.random.triangular(
            risk_factor.probability.value - 1,
            risk_factor.probability.value,
            risk_factor.probability.value + 1,
            self.monte_carlo_iterations
        )
        
        sev_samples = np.random.triangular(
            risk_factor.severity.value - 1,
            risk_factor.severity.value,
            risk_factor.severity.value + 1,
            self.monte_carlo_iterations
        )
        
        control_samples = np.random.beta(
            risk_factor.control_effectiveness * 10 + 1,
            (1 - risk_factor.control_effectiveness) * 10 + 1,
            self.monte_carlo_iterations
        )
        
        # Calculate risk scores for each iteration
        risk_scores = []
        for i in range(self.monte_carlo_iterations):
            prob = max(1, min(5, prob_samples[i]))
            sev = max(1, min(5, sev_samples[i]))
            control = max(0, min(1, control_samples[i]))
            
            score = prob * sev * (1 - control)
            risk_scores.append(score)
        
        risk_scores = np.array(risk_scores)
        
        return {
            'mean': float(np.mean(risk_scores)),
            'std': float(np.std(risk_scores)),
            'percentile_5': float(np.percentile(risk_scores, 5)),
            'percentile_95': float(np.percentile(risk_scores, 95)),
            'min': float(np.min(risk_scores)),
            'max': float(np.max(risk_scores)),
            'confidence_interval_95': (
                float(np.percentile(risk_scores, 2.5)),
                float(np.percentile(risk_scores, 97.5))
            )
        }
    
    def _calculate_next_review_date(self, risk_level: RiskLevel, risk_factor: RiskFactor) -> datetime:
        """Calculate next review date based on risk level and trend"""
        
        # Base review intervals (days)
        review_intervals = {
            RiskLevel.CRITICAL: 30,
            RiskLevel.VERY_HIGH: 60,
            RiskLevel.HIGH: 90,
            RiskLevel.MODERATE: 180,
            RiskLevel.LOW: 365,
            RiskLevel.VERY_LOW: 730
        }
        
        base_interval = review_intervals[risk_level]
        
        # Adjust based on trend
        if risk_factor.trend_direction == 'increasing':
            base_interval = int(base_interval * 0.75)
        elif risk_factor.trend_direction == 'decreasing':
            base_interval = int(base_interval * 1.25)
        
        return datetime.now() + timedelta(days=base_interval)
    
    def benchmark_against_industry(self, risk_assessment: RiskAssessmentResult) -> Dict[str, float]:
        """Compare risk assessment results against industry benchmarks"""
        
        benchmark_key = f"{self.industry.value}_{risk_assessment.risk_factor.category.value}"
        
        if benchmark_key not in self.benchmarks:
            logger.warning(f"No benchmark available for {benchmark_key}")
            return {}
        
        benchmark = self.benchmarks[benchmark_key]
        score = risk_assessment.aggregated_score
        
        # Calculate percentile ranking
        if score <= benchmark.percentile_25:
            percentile_rank = 25
        elif score <= benchmark.average_score:
            percentile_rank = 50
        elif score <= benchmark.percentile_75:
            percentile_rank = 75
        elif score <= benchmark.percentile_90:
            percentile_rank = 90
        else:
            percentile_rank = 95
        
        return {
            'industry_average': benchmark.average_score,
            'your_score': score,
            'percentile_rank': percentile_rank,
            'best_practice_gap': score - benchmark.best_practice_score,
            'improvement_potential': max(0, score - benchmark.percentile_25),
            'benchmark_sample_size': benchmark.sample_size
        }
    
    def analyze_risk_interdependencies(self, risk_factors: List[RiskFactor]) -> Dict[str, Any]:
        """Analyze interdependencies between multiple risk factors"""
        
        if len(risk_factors) < 2:
            return {}
        
        # Calculate correlation matrix
        scores = []
        for rf in risk_factors:
            assessment = self.assess_risk(rf)
            scores.append(assessment.aggregated_score)
        
        # Simple correlation analysis (in production, use more sophisticated methods)
        correlations = {}
        for i, rf1 in enumerate(risk_factors):
            for j, rf2 in enumerate(risk_factors[i+1:], i+1):
                corr_key = f"{rf1.name}_vs_{rf2.name}"
                # Simplified correlation based on category and controls overlap
                correlation = self._calculate_risk_correlation(rf1, rf2)
                correlations[corr_key] = correlation
        
        # Identify high-risk combinations
        high_risk_combinations = []
        for i, rf1 in enumerate(risk_factors):
            for j, rf2 in enumerate(risk_factors[i+1:], i+1):
                combined_score = (scores[i] + scores[j]) / 2
                if combined_score > 15:  # Threshold for high combined risk
                    high_risk_combinations.append({
                        'risk_1': rf1.name,
                        'risk_2': rf2.name,
                        'combined_score': combined_score,
                        'synergy_factor': self._calculate_synergy_factor(rf1, rf2)
                    })
        
        return {
            'correlations': correlations,
            'high_risk_combinations': high_risk_combinations,
            'overall_risk_concentration': max(scores) if scores else 0,
            'risk_diversity_index': len(set(rf.category for rf in risk_factors)) / len(RiskCategory)
        }
    
    def _calculate_risk_correlation(self, rf1: RiskFactor, rf2: RiskFactor) -> float:
        """Calculate correlation between two risk factors"""
        
        correlation = 0.0
        
        # Same category increases correlation
        if rf1.category == rf2.category:
            correlation += 0.3
        
        # Shared controls increase correlation
        shared_controls = set(rf1.current_controls) & set(rf2.current_controls)
        correlation += len(shared_controls) * 0.1
        
        # Same regulatory requirements increase correlation
        shared_regs = set(rf1.regulatory_requirements) & set(rf2.regulatory_requirements)
        correlation += len(shared_regs) * 0.1
        
        # Similar trend directions increase correlation
        if rf1.trend_direction == rf2.trend_direction:
            correlation += 0.1
        
        return min(1.0, correlation)
    
    def _calculate_synergy_factor(self, rf1: RiskFactor, rf2: RiskFactor) -> float:
        """Calculate synergy factor for combined risks"""
        
        # Base synergy from risk scores
        score1 = rf1.probability.value * rf1.severity.value
        score2 = rf2.probability.value * rf2.severity.value
        
        base_synergy = (score1 + score2) / 25  # Normalize to 0-1
        
        # Category-specific synergies
        category_synergies = {
            (RiskCategory.ENVIRONMENTAL, RiskCategory.HEALTH): 1.2,
            (RiskCategory.HEALTH, RiskCategory.SAFETY): 1.3,
            (RiskCategory.SAFETY, RiskCategory.OPERATIONAL): 1.1,
            (RiskCategory.REGULATORY, RiskCategory.ENVIRONMENTAL): 1.25
        }
        
        synergy_key = tuple(sorted([rf1.category, rf2.category]))
        category_multiplier = category_synergies.get(synergy_key, 1.0)
        
        return base_synergy * category_multiplier
    
    def generate_risk_report(self, risk_assessments: List[RiskAssessmentResult]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment report"""
        
        if not risk_assessments:
            return {}
        
        # Overall statistics
        scores = [ra.aggregated_score for ra in risk_assessments]
        risk_levels = [ra.risk_level for ra in risk_assessments]
        
        # Risk distribution
        risk_distribution = {}
        for level in RiskLevel:
            count = sum(1 for rl in risk_levels if rl == level)
            risk_distribution[level.label] = count
        
        # Category analysis
        category_scores = {}
        for category in RiskCategory:
            cat_scores = [ra.aggregated_score for ra in risk_assessments 
                         if ra.risk_factor.category == category]
            if cat_scores:
                category_scores[category.value] = {
                    'average': statistics.mean(cat_scores),
                    'max': max(cat_scores),
                    'count': len(cat_scores)
                }
        
        # Top risks
        top_risks = sorted(risk_assessments, key=lambda x: x.aggregated_score, reverse=True)[:10]
        
        # Key recommendations
        all_recommendations = []
        for ra in risk_assessments:
            all_recommendations.extend(ra.recommendations)
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        top_recommendations = sorted(recommendation_counts.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'summary': {
                'total_risks': len(risk_assessments),
                'average_score': statistics.mean(scores),
                'highest_score': max(scores),
                'lowest_score': min(scores),
                'assessment_date': datetime.now().isoformat()
            },
            'risk_distribution': risk_distribution,
            'category_analysis': category_scores,
            'top_risks': [
                {
                    'name': ra.risk_factor.name,
                    'category': ra.risk_factor.category.value,
                    'score': ra.aggregated_score,
                    'level': ra.risk_level.label
                }
                for ra in top_risks
            ],
            'top_recommendations': [
                {'recommendation': rec, 'frequency': count}
                for rec, count in top_recommendations
            ],
            'next_review_dates': [
                {
                    'risk_name': ra.risk_factor.name,
                    'next_review': ra.next_review_date.isoformat()
                }
                for ra in sorted(risk_assessments, key=lambda x: x.next_review_date)[:10]
            ]
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sample risk factor
    sample_risk = RiskFactor(
        id="ENV-001",
        name="Chemical Spill Risk",
        category=RiskCategory.ENVIRONMENTAL,
        probability=ProbabilityLevel.POSSIBLE,
        severity=SeverityLevel.MAJOR,
        current_controls=["Containment barriers", "Emergency response plan", "Regular inspections"],
        control_effectiveness=0.7,
        regulatory_requirements=[RegulatoryFramework.EPA, RegulatoryFramework.OSHA],
        environmental_impact_score=8.5,
        health_impact_score=6.0,
        safety_impact_score=7.5,
        financial_impact=500000.0,
        trend_direction="stable",
        confidence_level=0.85,
        data_quality="good",
        stakeholder_concern=0.6,
        media_attention=0.2
    )
    
    # Initialize risk engine
    engine = EHSRiskEngine(
        framework=ISO31000Framework(),
        industry=IndustryType.CHEMICAL
    )
    
    # Perform risk assessment
    result = engine.assess_risk(sample_risk)
    
    # Print results
    print(f"Risk Assessment Results for: {result.risk_factor.name}")
    print(f"Aggregated Score: {result.aggregated_score:.2f}")
    print(f"Risk Level: {result.risk_level.label}")
    print(f"Assessment Confidence: {result.assessment_confidence:.2f}")
    print(f"Next Review Date: {result.next_review_date.strftime('%Y-%m-%d')}")
    print("\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. {rec}")