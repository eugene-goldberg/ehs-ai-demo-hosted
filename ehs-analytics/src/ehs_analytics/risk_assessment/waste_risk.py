"""
Waste Generation Risk Analyzer

This module implements comprehensive waste generation risk analysis for EHS Analytics,
providing regulatory compliance monitoring, disposal cost analysis, diversion rate tracking,
and circular economy metrics following ISO 31000 risk management guidelines.
"""

import asyncio
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import uuid

from .base import BaseRiskAnalyzer, RiskAssessment, RiskFactor, RiskSeverity, RiskThresholds


# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class WasteRegulationData:
    """Waste regulation compliance information."""
    regulation_id: str
    facility_id: str
    waste_category: str  # hazardous, recyclable, general, electronic, etc.
    storage_limit_tons: float
    disposal_frequency_days: int
    reporting_threshold_tons: float
    regulatory_body: str  # EPA, state, local
    effective_date: datetime
    compliance_deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WasteGenerationRecord:
    """Individual waste generation data point."""
    timestamp: datetime
    facility_id: str
    waste_category: str
    amount_tons: float
    disposal_method: str  # landfill, recycling, incineration, treatment, etc.
    disposal_cost_per_ton: float
    storage_location_id: Optional[str] = None
    waste_stream_id: Optional[str] = None
    hazardous_level: str = "non-hazardous"  # hazardous, non-hazardous
    contamination_risk: float = 0.0  # 0.0 to 1.0 scale
    quality_flag: bool = True  # True if data is reliable
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WasteStreamData:
    """Waste stream efficiency and diversion metrics."""
    stream_id: str
    stream_type: str  # production, office, construction, etc.
    baseline_generation_rate: float  # tons per unit production
    current_generation_rate: float
    diversion_rate: float  # percentage recycled/recovered
    target_diversion_rate: float
    contamination_incidents: int
    cost_per_ton_trend: float  # percentage change over time
    last_audit_date: datetime
    operational_status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageFacilityData:
    """Waste storage facility capacity and utilization."""
    facility_id: str
    facility_type: str  # temporary, permanent, treatment
    total_capacity_tons: float
    current_utilization_tons: float
    waste_categories: List[str]
    last_inspection_date: datetime
    compliance_status: str = "compliant"
    safety_incidents: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WasteGenerationRiskAnalyzer(BaseRiskAnalyzer):
    """
    Comprehensive waste generation risk analyzer implementing ISO 31000 guidelines.
    
    This analyzer evaluates multiple risk factors including regulatory compliance,
    disposal cost optimization, diversion rate performance, storage capacity utilization,
    and contamination risks to provide actionable risk assessments and recommendations.
    """

    def __init__(
        self,
        name: str = "Waste Generation Risk Analyzer",
        description: str = "Analyzes waste generation risks across compliance, costs, diversion, and storage",
        compliance_buffer_percentage: float = 0.10,  # 10% buffer before regulatory limits
        cost_trend_analysis_days: int = 120,  # Days for cost trend analysis
        diversion_target_threshold: float = 0.15,  # 15% below target triggers risk
        storage_capacity_threshold: float = 0.85,  # 85% utilization threshold
    ):
        """
        Initialize the waste generation risk analyzer.
        
        Args:
            name: Analyzer name
            description: Analyzer description
            compliance_buffer_percentage: Safety buffer percentage before regulatory limits
            cost_trend_analysis_days: Number of days for cost trend analysis
            diversion_target_threshold: Threshold below target diversion rate
            storage_capacity_threshold: Storage utilization threshold
        """
        super().__init__(name, description)
        self.compliance_buffer_percentage = compliance_buffer_percentage
        self.cost_trend_analysis_days = cost_trend_analysis_days
        self.diversion_target_threshold = diversion_target_threshold
        self.storage_capacity_threshold = storage_capacity_threshold
        
        # Risk factor weights (must sum to 1.0)
        self.risk_weights = {
            'regulatory_compliance': 0.30,
            'disposal_cost_trends': 0.25,
            'diversion_performance': 0.20,
            'storage_utilization': 0.15,
            'contamination_risk': 0.10
        }
        
        # ISO 31000 aligned risk thresholds
        self.risk_thresholds = RiskThresholds(
            low_threshold=0.25,     # Acceptable risk level
            medium_threshold=0.50,  # Tolerable risk requiring monitoring
            high_threshold=0.75,    # Unacceptable risk requiring immediate action
            critical_threshold=0.90  # Critical risk requiring emergency response
        )

    async def analyze(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> RiskAssessment:
        """
        Perform comprehensive waste generation risk analysis.
        
        Args:
            data: Dictionary containing:
                - waste_records: List[WasteGenerationRecord]
                - regulation_data: List[WasteRegulationData]
                - stream_data: List[WasteStreamData] (optional)
                - storage_data: List[StorageFacilityData] (optional)
                - facility_id: str
            **kwargs: Additional analysis parameters
                
        Returns:
            RiskAssessment: Complete risk assessment with factors and recommendations
            
        Raises:
            ValueError: If required data is missing or invalid
        """
        try:
            logger.info(f"Starting waste generation risk analysis for {data.get('facility_id', 'unknown')}")
            
            # Validate input data
            self._validate_waste_data(data)
            
            # Extract data components
            waste_records = data['waste_records']
            regulation_data = data['regulation_data']
            stream_data = data.get('stream_data', [])
            storage_data = data.get('storage_data', [])
            facility_id = data['facility_id']
            
            # Analyze each risk factor concurrently
            risk_factors = []
            
            # Run risk factor analyses concurrently
            compliance_task = self._analyze_regulatory_compliance(waste_records, regulation_data)
            cost_task = self._analyze_disposal_cost_trends(waste_records)
            diversion_task = self._analyze_diversion_performance(waste_records, stream_data)
            storage_task = self._analyze_storage_utilization(storage_data, waste_records)
            contamination_task = self._analyze_contamination_risk(waste_records, stream_data)
            
            # Wait for all analyses to complete
            compliance_factor, cost_factor, diversion_factor, storage_factor, contamination_factor = await asyncio.gather(
                compliance_task, cost_task, diversion_task, storage_task, contamination_task
            )
            
            risk_factors.extend([
                compliance_factor, cost_factor, diversion_factor, 
                storage_factor, contamination_factor
            ])
            
            # Generate assessment
            assessment = RiskAssessment.from_factors(
                factors=risk_factors,
                assessment_type="waste_generation_risk",
                assessment_id=str(uuid.uuid4()),
                metadata={
                    'facility_id': facility_id,
                    'analysis_period_days': self.cost_trend_analysis_days,
                    'compliance_buffer_percentage': self.compliance_buffer_percentage,
                    'total_waste_records': len(waste_records),
                    'regulation_count': len(regulation_data),
                    'waste_stream_count': len(stream_data),
                    'storage_facility_count': len(storage_data)
                }
            )
            
            # Generate specific recommendations
            assessment.recommendations = await self._generate_recommendations(
                assessment, regulation_data, waste_records, stream_data, storage_data
            )
            
            logger.info(f"Completed waste risk analysis: {assessment.severity.value} risk level")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in waste generation risk analysis: {str(e)}")
            raise

    async def _analyze_regulatory_compliance(
        self,
        waste_records: List[WasteGenerationRecord],
        regulation_data: List[WasteRegulationData]
    ) -> RiskFactor:
        """
        Analyze regulatory compliance risk for waste management.
        
        Evaluates current waste generation against regulatory limits for different
        waste categories (hazardous vs non-hazardous) and storage requirements.
        """
        try:
            logger.debug("Analyzing regulatory compliance for waste management")
            
            if not regulation_data:
                logger.warning("No regulation data provided for compliance analysis")
                return RiskFactor(
                    name="Regulatory Compliance",
                    value=0.5,  # Medium risk when no regulation data
                    weight=self.risk_weights['regulatory_compliance'],
                    severity=RiskSeverity.MEDIUM,
                    thresholds=self.risk_thresholds,
                    description="Limited compliance monitoring due to missing regulation data",
                    unit="compliance_score"
                )
            
            compliance_violations = 0
            total_checks = 0
            hazardous_overage_risk = 0.0
            storage_violations = 0
            
            # Analyze compliance by waste category
            for regulation in regulation_data:
                total_checks += 1
                
                # Get relevant waste records for this regulation
                relevant_records = [
                    r for r in waste_records 
                    if r.waste_category == regulation.waste_category 
                    and r.facility_id == regulation.facility_id
                ]
                
                if not relevant_records:
                    continue
                
                # Check storage limit compliance
                current_storage = sum(r.amount_tons for r in relevant_records 
                                    if r.timestamp >= datetime.now() - timedelta(days=30))
                
                storage_limit_with_buffer = regulation.storage_limit_tons * (1 - self.compliance_buffer_percentage)
                
                if current_storage > storage_limit_with_buffer:
                    storage_violations += 1
                    if current_storage > regulation.storage_limit_tons:
                        compliance_violations += 1
                
                # Special handling for hazardous waste
                if regulation.waste_category == "hazardous":
                    hazardous_overage = max(0, current_storage - storage_limit_with_buffer)
                    hazardous_overage_risk += hazardous_overage / regulation.storage_limit_tons
                
                # Check reporting threshold compliance
                monthly_generation = sum(r.amount_tons for r in relevant_records 
                                       if r.timestamp >= datetime.now() - timedelta(days=30))
                
                if monthly_generation > regulation.reporting_threshold_tons:
                    # This should trigger reporting requirements
                    logger.info(f"Reporting threshold exceeded for {regulation.waste_category}: {monthly_generation} tons")
            
            # Calculate compliance risk score
            if total_checks == 0:
                compliance_score = 0.5
            else:
                base_violation_rate = compliance_violations / total_checks
                storage_risk = min(1.0, storage_violations / total_checks)
                hazardous_multiplier = 1.0 + min(2.0, hazardous_overage_risk)  # Amplify hazardous risk
                
                compliance_score = min(1.0, (base_violation_rate + storage_risk * 0.5) * hazardous_multiplier)
            
            return RiskFactor(
                name="Regulatory Compliance",
                value=compliance_score,
                weight=self.risk_weights['regulatory_compliance'],
                severity=self.risk_thresholds.get_severity(compliance_score),
                thresholds=self.risk_thresholds,
                description=f"Compliance analysis across {len(regulation_data)} regulations",
                unit="compliance_score",
                metadata={
                    'total_regulations': len(regulation_data),
                    'compliance_violations': compliance_violations,
                    'storage_violations': storage_violations,
                    'hazardous_overage_risk': hazardous_overage_risk,
                    'total_checks': total_checks
                }
            )
            
        except Exception as e:
            logger.error(f"Error in regulatory compliance analysis: {str(e)}")
            raise

    async def _analyze_disposal_cost_trends(
        self,
        waste_records: List[WasteGenerationRecord]
    ) -> RiskFactor:
        """
        Analyze disposal cost trends and optimization opportunities.
        
        Evaluates cost trends across different disposal methods and identifies
        cost escalation risks and optimization opportunities.
        """
        try:
            logger.debug("Analyzing disposal cost trends")
            
            if not waste_records:
                return self._create_default_risk_factor(
                    "Disposal Cost Trends", 
                    self.risk_weights['disposal_cost_trends'],
                    "No waste records available for cost trend analysis"
                )
            
            # Filter recent records for trend analysis
            cutoff_date = datetime.now() - timedelta(days=self.cost_trend_analysis_days)
            recent_records = [r for r in waste_records if r.timestamp >= cutoff_date]
            
            if len(recent_records) < 10:  # Minimum data points for trend analysis
                return self._create_default_risk_factor(
                    "Disposal Cost Trends",
                    self.risk_weights['disposal_cost_trends'],
                    "Insufficient data for reliable cost trend analysis"
                )
            
            # Analyze cost trends by disposal method
            disposal_methods = {}
            for record in recent_records:
                method = record.disposal_method
                if method not in disposal_methods:
                    disposal_methods[method] = []
                disposal_methods[method].append({
                    'timestamp': record.timestamp,
                    'cost_per_ton': record.disposal_cost_per_ton,
                    'amount': record.amount_tons
                })
            
            cost_risk_score = 0.0
            total_weight = 0.0
            cost_escalation_risk = 0.0
            optimization_potential = 0.0
            
            for method, records in disposal_methods.items():
                if len(records) < 5:  # Skip methods with insufficient data
                    continue
                
                # Sort by timestamp for trend analysis
                records.sort(key=lambda x: x['timestamp'])
                
                # Calculate weighted average costs over time periods
                costs = [r['cost_per_ton'] for r in records]
                amounts = [r['amount'] for r in records]
                
                # Calculate cost trend (linear regression-like approach)
                if len(costs) >= 5:
                    early_costs = costs[:len(costs)//2]
                    late_costs = costs[len(costs)//2:]
                    
                    early_avg = statistics.mean(early_costs)
                    late_avg = statistics.mean(late_costs)
                    
                    cost_change_rate = (late_avg - early_avg) / early_avg if early_avg > 0 else 0
                    cost_escalation_risk += max(0, cost_change_rate) * sum(amounts)
                
                # Calculate method-specific risk based on cost variability
                cost_volatility = statistics.stdev(costs) / statistics.mean(costs) if statistics.mean(costs) > 0 else 0
                method_weight = sum(amounts)
                
                method_risk = min(1.0, cost_volatility + max(0, cost_change_rate))
                cost_risk_score += method_risk * method_weight
                total_weight += method_weight
                
                # Identify optimization potential (expensive methods with alternatives)
                if method in ['landfill', 'incineration'] and statistics.mean(costs) > 100:  # $100/ton threshold
                    optimization_potential += method_weight
            
            # Normalize risk score
            if total_weight > 0:
                cost_risk_score /= total_weight
            else:
                cost_risk_score = 0.5
            
            # Factor in overall cost escalation across facility
            total_amount = sum(r.amount_tons for r in recent_records)
            if total_amount > 0:
                escalation_factor = min(1.0, cost_escalation_risk / total_amount)
                cost_risk_score = min(1.0, cost_risk_score + escalation_factor * 0.3)
            
            return RiskFactor(
                name="Disposal Cost Trends",
                value=cost_risk_score,
                weight=self.risk_weights['disposal_cost_trends'],
                severity=self.risk_thresholds.get_severity(cost_risk_score),
                thresholds=self.risk_thresholds,
                description=f"Cost trend analysis across {len(disposal_methods)} disposal methods",
                unit="cost_risk_score",
                metadata={
                    'disposal_methods': list(disposal_methods.keys()),
                    'analysis_period_days': self.cost_trend_analysis_days,
                    'total_waste_amount': total_amount,
                    'cost_escalation_risk': cost_escalation_risk,
                    'optimization_potential_tons': optimization_potential,
                    'records_analyzed': len(recent_records)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in disposal cost trend analysis: {str(e)}")
            raise

    async def _analyze_diversion_performance(
        self,
        waste_records: List[WasteGenerationRecord],
        stream_data: List[WasteStreamData]
    ) -> RiskFactor:
        """
        Analyze waste diversion rate performance against targets.
        
        Evaluates recycling and recovery performance across waste streams
        and identifies opportunities for circular economy improvements.
        """
        try:
            logger.debug("Analyzing waste diversion performance")
            
            if not waste_records:
                return self._create_default_risk_factor(
                    "Diversion Performance",
                    self.risk_weights['diversion_performance'],
                    "No waste records available for diversion analysis"
                )
            
            # Calculate overall facility diversion rate
            recyclable_methods = ['recycling', 'composting', 'recovery', 'reuse']
            total_waste = sum(r.amount_tons for r in waste_records)
            diverted_waste = sum(r.amount_tons for r in waste_records 
                               if r.disposal_method in recyclable_methods)
            
            if total_waste == 0:
                overall_diversion_rate = 0.0
            else:
                overall_diversion_rate = diverted_waste / total_waste
            
            # Analyze stream-specific performance
            stream_performance = {}
            total_stream_risk = 0.0
            stream_count = 0
            
            for stream in stream_data:
                stream_count += 1
                current_rate = stream.diversion_rate
                target_rate = stream.target_diversion_rate
                
                # Calculate performance gap
                if target_rate > 0:
                    performance_gap = max(0, target_rate - current_rate)
                    gap_severity = performance_gap / target_rate
                    
                    # Apply threshold for concerning performance
                    if gap_severity > self.diversion_target_threshold:
                        stream_risk = min(1.0, gap_severity / self.diversion_target_threshold)
                    else:
                        stream_risk = gap_severity * 0.5  # Lower risk if within threshold
                    
                    total_stream_risk += stream_risk
                    
                    stream_performance[stream.stream_id] = {
                        'current_rate': current_rate,
                        'target_rate': target_rate,
                        'gap': performance_gap,
                        'risk_score': stream_risk
                    }
            
            # Calculate overall diversion risk
            if stream_count > 0:
                avg_stream_risk = total_stream_risk / stream_count
            else:
                # Fallback to facility-level analysis
                industry_benchmark = 0.50  # Assume 50% industry benchmark
                if overall_diversion_rate < industry_benchmark * (1 - self.diversion_target_threshold):
                    avg_stream_risk = min(1.0, (industry_benchmark - overall_diversion_rate) / industry_benchmark)
                else:
                    avg_stream_risk = max(0.0, 1 - overall_diversion_rate / industry_benchmark) * 0.5
            
            # Factor in circular economy opportunities
            circular_economy_score = self._calculate_circular_economy_score(waste_records)
            diversion_risk_score = avg_stream_risk * (1 - circular_economy_score * 0.2)  # Reduce risk for good circular economy practices
            
            return RiskFactor(
                name="Diversion Performance",
                value=diversion_risk_score,
                weight=self.risk_weights['diversion_performance'],
                severity=self.risk_thresholds.get_severity(diversion_risk_score),
                thresholds=self.risk_thresholds,
                description=f"Diversion rate analysis across {stream_count} waste streams",
                unit="diversion_risk_score",
                metadata={
                    'overall_diversion_rate': overall_diversion_rate,
                    'total_waste_tons': total_waste,
                    'diverted_waste_tons': diverted_waste,
                    'stream_performance': stream_performance,
                    'circular_economy_score': circular_economy_score,
                    'streams_analyzed': stream_count
                }
            )
            
        except Exception as e:
            logger.error(f"Error in diversion performance analysis: {str(e)}")
            raise

    async def _analyze_storage_utilization(
        self,
        storage_data: List[StorageFacilityData],
        waste_records: List[WasteGenerationRecord]
    ) -> RiskFactor:
        """
        Analyze waste storage capacity utilization and safety.
        
        Evaluates storage facility utilization rates, capacity constraints,
        and compliance with storage regulations.
        """
        try:
            logger.debug("Analyzing storage facility utilization")
            
            if not storage_data:
                logger.warning("No storage facility data provided")
                return RiskFactor(
                    name="Storage Utilization",
                    value=0.3,  # Low-medium risk when no storage data
                    weight=self.risk_weights['storage_utilization'],
                    severity=RiskSeverity.MEDIUM,
                    thresholds=self.risk_thresholds,
                    description="Limited storage monitoring due to missing facility data",
                    unit="utilization_risk_score"
                )
            
            total_utilization_risk = 0.0
            safety_incident_risk = 0.0
            compliance_risk = 0.0
            capacity_shortage_risk = 0.0
            
            for facility in storage_data:
                # Calculate utilization rate
                utilization_rate = facility.current_utilization_tons / facility.total_capacity_tons
                
                # Assess utilization risk
                if utilization_rate > self.storage_capacity_threshold:
                    utilization_risk = min(1.0, (utilization_rate - self.storage_capacity_threshold) / (1 - self.storage_capacity_threshold))
                else:
                    utilization_risk = utilization_rate * 0.2  # Low risk for reasonable utilization
                
                total_utilization_risk += utilization_risk
                
                # Assess safety incident risk
                if facility.safety_incidents > 0:
                    safety_incident_risk += min(1.0, facility.safety_incidents / 10.0)  # Scale incidents
                
                # Assess compliance risk
                if facility.compliance_status != "compliant":
                    compliance_risk += 1.0
                
                # Check inspection currency
                days_since_inspection = (datetime.now() - facility.last_inspection_date).days
                if days_since_inspection > 365:  # Annual inspection expectation
                    compliance_risk += 0.5
                
                # Assess capacity shortage risk for different waste categories
                for category in facility.waste_categories:
                    category_records = [r for r in waste_records 
                                      if r.waste_category == category 
                                      and r.timestamp >= datetime.now() - timedelta(days=30)]
                    
                    if category_records:
                        monthly_generation = sum(r.amount_tons for r in category_records)
                        # Estimate capacity needed for monthly storage
                        estimated_monthly_capacity = monthly_generation * 0.5  # Assume 50% storage period
                        available_capacity = facility.total_capacity_tons - facility.current_utilization_tons
                        
                        if estimated_monthly_capacity > available_capacity:
                            shortage_ratio = estimated_monthly_capacity / available_capacity
                            capacity_shortage_risk += min(1.0, shortage_ratio - 1.0)
            
            # Calculate overall storage risk
            facility_count = len(storage_data)
            avg_utilization_risk = total_utilization_risk / facility_count
            avg_safety_risk = safety_incident_risk / facility_count
            avg_compliance_risk = compliance_risk / facility_count
            avg_shortage_risk = capacity_shortage_risk / facility_count
            
            # Weighted combination of risk factors
            storage_risk_score = (
                avg_utilization_risk * 0.4 +
                avg_safety_risk * 0.2 +
                avg_compliance_risk * 0.3 +
                avg_shortage_risk * 0.1
            )
            
            storage_risk_score = min(1.0, storage_risk_score)
            
            return RiskFactor(
                name="Storage Utilization",
                value=storage_risk_score,
                weight=self.risk_weights['storage_utilization'],
                severity=self.risk_thresholds.get_severity(storage_risk_score),
                thresholds=self.risk_thresholds,
                description=f"Storage analysis across {facility_count} facilities",
                unit="utilization_risk_score",
                metadata={
                    'facilities_analyzed': facility_count,
                    'avg_utilization_risk': avg_utilization_risk,
                    'safety_incident_risk': avg_safety_risk,
                    'compliance_risk': avg_compliance_risk,
                    'capacity_shortage_risk': avg_shortage_risk,
                    'storage_capacity_threshold': self.storage_capacity_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error in storage utilization analysis: {str(e)}")
            raise

    async def _analyze_contamination_risk(
        self,
        waste_records: List[WasteGenerationRecord],
        stream_data: List[WasteStreamData]
    ) -> RiskFactor:
        """
        Analyze waste stream contamination risk.
        
        Evaluates contamination incidents and risk levels across waste streams
        that could impact recycling efficiency and compliance.
        """
        try:
            logger.debug("Analyzing waste stream contamination risk")
            
            if not waste_records:
                return self._create_default_risk_factor(
                    "Contamination Risk",
                    self.risk_weights['contamination_risk'],
                    "No waste records available for contamination analysis"
                )
            
            # Analyze contamination risk from waste records
            total_contamination_risk = 0.0
            total_amount = 0.0
            high_risk_streams = 0
            
            for record in waste_records:
                total_contamination_risk += record.contamination_risk * record.amount_tons
                total_amount += record.amount_tons
                
                if record.contamination_risk > 0.7:  # High contamination risk threshold
                    high_risk_streams += 1
            
            # Calculate weighted average contamination risk
            if total_amount > 0:
                avg_contamination_risk = total_contamination_risk / total_amount
            else:
                avg_contamination_risk = 0.0
            
            # Analyze stream-specific contamination incidents
            stream_incident_risk = 0.0
            if stream_data:
                total_incidents = sum(stream.contamination_incidents for stream in stream_data)
                stream_count = len(stream_data)
                
                if stream_count > 0:
                    avg_incidents_per_stream = total_incidents / stream_count
                    # Normalize incident risk (assume >5 incidents per stream is high risk)
                    stream_incident_risk = min(1.0, avg_incidents_per_stream / 5.0)
            
            # Combine contamination risks
            contamination_risk_score = min(1.0, avg_contamination_risk * 0.7 + stream_incident_risk * 0.3)
            
            # Factor in high-risk stream count
            if len(waste_records) > 0:
                high_risk_factor = high_risk_streams / len(waste_records)
                contamination_risk_score = min(1.0, contamination_risk_score + high_risk_factor * 0.2)
            
            return RiskFactor(
                name="Contamination Risk",
                value=contamination_risk_score,
                weight=self.risk_weights['contamination_risk'],
                severity=self.risk_thresholds.get_severity(contamination_risk_score),
                thresholds=self.risk_thresholds,
                description=f"Contamination analysis across {len(waste_records)} waste records",
                unit="contamination_risk_score",
                metadata={
                    'avg_contamination_risk': avg_contamination_risk,
                    'high_risk_streams': high_risk_streams,
                    'total_contamination_incidents': sum(stream.contamination_incidents for stream in stream_data) if stream_data else 0,
                    'stream_incident_risk': stream_incident_risk,
                    'records_analyzed': len(waste_records),
                    'streams_analyzed': len(stream_data)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in contamination risk analysis: {str(e)}")
            raise

    def _calculate_circular_economy_score(self, waste_records: List[WasteGenerationRecord]) -> float:
        """
        Calculate circular economy performance score.
        
        Evaluates waste management practices against circular economy principles
        including waste reduction, reuse, recycling, and recovery.
        """
        if not waste_records:
            return 0.0
        
        # Assign scores to disposal methods based on circular economy hierarchy
        method_scores = {
            'reuse': 1.0,
            'recycling': 0.8,
            'recovery': 0.6,
            'composting': 0.7,
            'treatment': 0.4,
            'incineration': 0.2,
            'landfill': 0.0
        }
        
        total_score = 0.0
        total_amount = 0.0
        
        for record in waste_records:
            method = record.disposal_method.lower()
            score = method_scores.get(method, 0.3)  # Default score for unknown methods
            
            total_score += score * record.amount_tons
            total_amount += record.amount_tons
        
        if total_amount > 0:
            return total_score / total_amount
        else:
            return 0.0

    def _create_default_risk_factor(self, name: str, weight: float, description: str) -> RiskFactor:
        """Create a default risk factor when data is insufficient."""
        return RiskFactor(
            name=name,
            value=0.4,  # Medium-low default risk
            weight=weight,
            severity=RiskSeverity.MEDIUM,
            thresholds=self.risk_thresholds,
            description=description,
            unit="risk_score"
        )

    async def _generate_recommendations(
        self,
        assessment: RiskAssessment,
        regulation_data: List[WasteRegulationData],
        waste_records: List[WasteGenerationRecord],
        stream_data: List[WasteStreamData],
        storage_data: List[StorageFacilityData]
    ) -> List[str]:
        """
        Generate specific waste management recommendations based on risk assessment.
        
        Provides actionable recommendations for reducing waste generation risks
        and improving compliance, cost efficiency, and environmental performance.
        """
        recommendations = []
        critical_factors = assessment.get_critical_factors()
        high_risk_factors = assessment.get_high_risk_factors()
        
        # Critical risk recommendations (immediate action required)
        for factor in critical_factors:
            if factor.name == "Regulatory Compliance":
                recommendations.extend([
                    "IMMEDIATE: Conduct emergency waste audit and implement compliance corrective actions",
                    "IMMEDIATE: Contact regulatory affairs team and legal counsel for compliance review",
                    "IMMEDIATE: Implement temporary waste generation reduction measures"
                ])
            
            elif factor.name == "Storage Utilization":
                recommendations.extend([
                    "IMMEDIATE: Arrange emergency waste disposal to reduce storage capacity utilization",
                    "IMMEDIATE: Implement temporary storage restrictions and generation controls",
                    "URGENT: Source additional storage capacity or accelerate disposal schedules"
                ])
            
            elif factor.name == "Contamination Risk":
                recommendations.extend([
                    "IMMEDIATE: Implement enhanced waste segregation protocols and training",
                    "URGENT: Conduct contamination source investigation and remediation",
                    "IMMEDIATE: Suspend high-risk waste streams until contamination is controlled"
                ])
        
        # High risk recommendations (action required within 30 days)
        for factor in high_risk_factors:
            if factor.name == "Disposal Cost Trends":
                recommendations.extend([
                    "Conduct disposal cost optimization analysis and vendor negotiations",
                    "Evaluate alternative disposal methods and circular economy opportunities",
                    "Implement waste reduction initiatives to control disposal costs"
                ])
            
            elif factor.name == "Diversion Performance":
                recommendations.extend([
                    "Develop waste diversion improvement plan with specific targets",
                    "Implement enhanced recycling and recovery programs",
                    "Evaluate circular economy partnerships and waste-to-resource opportunities"
                ])
            
            elif factor.name == "Regulatory Compliance":
                recommendations.extend([
                    "Schedule comprehensive regulatory compliance audit within 30 days",
                    "Implement enhanced waste tracking and reporting systems",
                    "Establish proactive regulatory monitoring and horizon scanning"
                ])
        
        # General improvement recommendations
        if assessment.overall_score > 0.5:  # Medium or higher risk
            recommendations.extend([
                "Establish waste management key performance indicators (KPIs) and dashboard",
                "Implement predictive analytics for waste generation forecasting",
                "Develop waste reduction roadmap aligned with corporate sustainability goals"
            ])
        
        # Circular economy recommendations
        circular_economy_score = self._calculate_circular_economy_score(waste_records)
        if circular_economy_score < 0.6:
            recommendations.extend([
                "Evaluate circular economy opportunities and partnerships",
                "Implement waste-to-resource pilot programs",
                "Develop supplier engagement program for packaging reduction"
            ])
        
        # Cost optimization recommendations
        if any(f.name == "Disposal Cost Trends" and f.severity.value in ['high', 'critical'] 
               for f in assessment.factors):
            recommendations.extend([
                "Conduct waste disposal procurement optimization and vendor consolidation",
                "Implement waste segregation improvements to access lower-cost disposal options",
                "Evaluate on-site treatment technologies for high-volume waste streams"
            ])
        
        # Storage management recommendations
        if storage_data and any(f.current_utilization_tons / f.total_capacity_tons > 0.8 for f in storage_data):
            recommendations.extend([
                "Develop storage capacity expansion or optimization plan",
                "Implement just-in-time disposal scheduling to minimize storage requirements",
                "Evaluate off-site storage partnerships for peak capacity management"
            ])
        
        # Ensure we have actionable recommendations
        if not recommendations:
            recommendations.append(
                "Continue monitoring waste generation trends and maintain current management practices"
            )
        
        return recommendations

    def _validate_waste_data(self, data: Dict[str, Any]) -> None:
        """
        Validate input data for waste risk analysis.
        
        Args:
            data: Input data dictionary
            
        Raises:
            ValueError: If required data is missing or invalid
        """
        # Call parent validation
        self.validate_input_data(data)
        
        # Check required fields
        required_fields = ['waste_records', 'regulation_data', 'facility_id']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from input data")
        
        # Validate data types
        if not isinstance(data['waste_records'], list):
            raise ValueError("waste_records must be a list")
        
        # Handle regulation_data - can be either list or dict
        if isinstance(data['regulation_data'], dict):
            # Convert dict to list for compatibility
            if 'regulations' in data['regulation_data']:
                # Handle nested structure
                data['regulation_data'] = data['regulation_data']['regulations']
            else:
                # Convert dict values to list
                data['regulation_data'] = list(data['regulation_data'].values())
        
        if not isinstance(data['regulation_data'], list):
            raise ValueError("regulation_data must be a list or dict")
        
        if not isinstance(data['facility_id'], str) or not data['facility_id'].strip():
            raise ValueError("facility_id must be a non-empty string")
        
        # Validate optional fields
        if 'stream_data' in data and not isinstance(data['stream_data'], list):
            raise ValueError("stream_data must be a list")
        
        if 'storage_data' in data and not isinstance(data['storage_data'], list):
            raise ValueError("storage_data must be a list")
        
        logger.debug(f"Validated waste data: {len(data['waste_records'])} records, "
                    f"{len(data['regulation_data'])} regulations for facility {data['facility_id']}")

# Create aliases for backward compatibility
WasteRiskAnalyzer = WasteGenerationRiskAnalyzer