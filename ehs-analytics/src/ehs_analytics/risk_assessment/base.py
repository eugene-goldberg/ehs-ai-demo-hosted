"""
Risk Assessment Framework Foundation

This module provides the core components for risk assessment in the EHS Analytics system,
including risk enums, data structures, and base analyzer classes for consistent risk evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from datetime import datetime


class RiskSeverity(Enum):
    """Risk severity levels for EHS assessments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other):
        """Enable comparison between risk severity levels."""
        if not isinstance(other, RiskSeverity):
            return NotImplemented
        severity_order = [self.LOW, self.MEDIUM, self.HIGH, self.CRITICAL]
        return severity_order.index(self) < severity_order.index(other)

    def __le__(self, other):
        """Enable comparison between risk severity levels."""
        return self < other or self == other

    def __gt__(self, other):
        """Enable comparison between risk severity levels."""
        if not isinstance(other, RiskSeverity):
            return NotImplemented
        return not self <= other

    def __ge__(self, other):
        """Enable comparison between risk severity levels."""
        return not self < other

    @property
    def numeric_value(self) -> int:
        """Get numeric value for calculations."""
        return {
            self.LOW: 1,
            self.MEDIUM: 2,
            self.HIGH: 3,
            self.CRITICAL: 4
        }[self]


@dataclass
class RiskThresholds:
    """Thresholds for determining risk severity levels."""
    low_threshold: float = 0.25
    medium_threshold: float = 0.50
    high_threshold: float = 0.75
    critical_threshold: float = 0.9

    def get_severity(self, risk_score: float) -> RiskSeverity:
        """
        Determine risk severity based on score and thresholds.
        
        Args:
            risk_score: Normalized risk score (0.0 to 1.0)
            
        Returns:
            RiskSeverity: The appropriate severity level
        """
        if risk_score >= self.critical_threshold:
            return RiskSeverity.CRITICAL
        elif risk_score >= self.high_threshold:
            return RiskSeverity.HIGH
        elif risk_score >= self.medium_threshold:
            return RiskSeverity.MEDIUM
        else:
            return RiskSeverity.LOW


@dataclass
class RiskFactor:
    """
    Individual risk factor with value, weight, and severity information.
    
    Attributes:
        name: Human-readable name of the risk factor
        value: Current value of the risk factor
        weight: Importance weight (0.0 to 1.0) in overall risk calculation
        severity: Current severity level of this factor
        thresholds: Thresholds for determining severity levels
        description: Optional description of what this factor measures
        unit: Optional unit of measurement
        metadata: Additional metadata for the risk factor
    """
    name: str
    value: float
    weight: float
    severity: RiskSeverity
    thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    description: Optional[str] = None
    unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate risk factor data after initialization."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")
        
        # Update severity based on value and thresholds if not explicitly set
        if self.value is not None:
            calculated_severity = self.thresholds.get_severity(self.value)
            if self.severity != calculated_severity:
                self.severity = calculated_severity

    @property
    def weighted_score(self) -> float:
        """Calculate the weighted score for this risk factor."""
        return self.value * self.weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert risk factor to dictionary representation."""
        return {
            'name': self.name,
            'value': self.value,
            'weight': self.weight,
            'severity': self.severity.value,
            'weighted_score': self.weighted_score,
            'thresholds': {
                'low': self.thresholds.low_threshold,
                'medium': self.thresholds.medium_threshold,
                'high': self.thresholds.high_threshold,
                'critical': self.thresholds.critical_threshold
            },
            'description': self.description,
            'unit': self.unit,
            'metadata': self.metadata
        }


@dataclass
class RiskAssessment:
    """
    Complete risk assessment with overall score, severity, and recommendations.
    
    Attributes:
        overall_score: Weighted average risk score (0.0 to 1.0)
        severity: Overall risk severity level
        factors: List of individual risk factors
        recommendations: List of recommended actions
        assessment_id: Unique identifier for this assessment
        timestamp: When the assessment was created
        assessment_type: Type of risk assessment performed
        confidence_score: Confidence in the assessment (0.0 to 1.0)
        metadata: Additional assessment metadata
    """
    overall_score: float
    severity: RiskSeverity
    factors: List[RiskFactor]
    recommendations: List[str] = field(default_factory=list)
    assessment_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    assessment_type: str = "general"
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate assessment data after initialization."""
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError(f"Overall score must be between 0.0 and 1.0, got {self.overall_score}")
        
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")

        # Validate that factor weights sum to reasonable total
        total_weight = sum(factor.weight for factor in self.factors)
        if total_weight > 1.1:  # Allow slight tolerance
            raise ValueError(f"Total factor weights exceed 1.0: {total_weight}")

    @classmethod
    def from_factors(
        cls,
        factors: List[RiskFactor],
        recommendations: Optional[List[str]] = None,
        assessment_type: str = "general",
        **kwargs
    ) -> "RiskAssessment":
        """
        Create risk assessment from list of risk factors.
        
        Args:
            factors: List of risk factors
            recommendations: Optional list of recommendations
            assessment_type: Type of assessment
            **kwargs: Additional arguments for RiskAssessment
            
        Returns:
            RiskAssessment: New assessment instance
        """
        if not factors:
            raise ValueError("At least one risk factor is required")

        # Calculate overall score as weighted average
        total_weighted_score = sum(factor.weighted_score for factor in factors)
        total_weight = sum(factor.weight for factor in factors)
        
        if total_weight > 0:
            overall_score = total_weighted_score / total_weight
        else:
            overall_score = 0.0

        # Determine overall severity
        thresholds = RiskThresholds()
        severity = thresholds.get_severity(overall_score)

        return cls(
            overall_score=overall_score,
            severity=severity,
            factors=factors,
            recommendations=recommendations or [],
            assessment_type=assessment_type,
            **kwargs
        )

    def get_critical_factors(self) -> List[RiskFactor]:
        """Get all factors with critical severity."""
        return [factor for factor in self.factors if factor.severity == RiskSeverity.CRITICAL]

    def get_high_risk_factors(self) -> List[RiskFactor]:
        """Get all factors with high or critical severity."""
        return [factor for factor in self.factors 
                if factor.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert risk assessment to dictionary representation."""
        return {
            'assessment_id': self.assessment_id,
            'timestamp': self.timestamp.isoformat(),
            'assessment_type': self.assessment_type,
            'overall_score': self.overall_score,
            'severity': self.severity.value,
            'confidence_score': self.confidence_score,
            'factors': [factor.to_dict() for factor in self.factors],
            'recommendations': self.recommendations,
            'critical_factors_count': len(self.get_critical_factors()),
            'high_risk_factors_count': len(self.get_high_risk_factors()),
            'metadata': self.metadata
        }


class BaseRiskAnalyzer(ABC):
    """
    Abstract base class for risk analyzers.
    
    All risk analyzers should inherit from this class and implement
    the analyze method to provide consistent risk assessment capabilities.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        """
        Initialize the risk analyzer.
        
        Args:
            name: Name of the analyzer
            description: Optional description of what the analyzer does
        """
        self.name = name
        self.description = description

    @abstractmethod
    def analyze(self, data: Dict[str, Any], **kwargs) -> RiskAssessment:
        """
        Analyze risk based on input data.
        
        Args:
            data: Input data for risk analysis
            **kwargs: Additional analysis parameters
            
        Returns:
            RiskAssessment: Complete risk assessment
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the analyze method")

    def validate_input_data(self, data: Dict[str, Any]) -> None:
        """
        Validate input data for analysis.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if not data:
            raise ValueError("Input data cannot be empty")

    def get_analyzer_info(self) -> Dict[str, Any]:
        """
        Get information about this analyzer.
        
        Returns:
            Dict containing analyzer metadata
        """
        return {
            'name': self.name,
            'description': self.description,
            'class': self.__class__.__name__,
            'module': self.__class__.__module__
        }