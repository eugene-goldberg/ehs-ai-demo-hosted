"""
Water Consumption Risk Analyzer

This module implements comprehensive water consumption risk analysis for EHS Analytics,
providing permit compliance monitoring, trend analysis, seasonal patterns, and equipment
efficiency assessment following ISO 31000 risk management guidelines.
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
class WaterPermitData:
    """Water permit information for compliance analysis."""
    permit_id: str
    facility_id: str
    daily_limit: float  # gallons per day
    monthly_limit: float  # gallons per month
    annual_limit: float  # gallons per year
    issue_date: datetime
    expiry_date: datetime
    permit_type: str = "water_withdrawal"
    regulatory_body: str = "local"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaterConsumptionRecord:
    """Individual water consumption data point."""
    timestamp: datetime
    facility_id: str
    consumption_gallons: float
    meter_id: Optional[str] = None
    equipment_id: Optional[str] = None
    consumption_type: str = "operational"  # operational, cooling, process, etc.
    quality_flag: bool = True  # True if data is reliable
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EquipmentEfficiencyData:
    """Equipment efficiency metrics for water risk assessment."""
    equipment_id: str
    equipment_type: str
    baseline_efficiency: float  # gallons per unit output
    current_efficiency: float
    last_maintenance: datetime
    efficiency_trend: float  # percentage change over time
    operational_status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


class WaterConsumptionRiskAnalyzer(BaseRiskAnalyzer):
    """
    Comprehensive water consumption risk analyzer implementing ISO 31000 guidelines.
    
    This analyzer evaluates multiple risk factors including permit compliance,
    consumption trends, seasonal patterns, and equipment efficiency to provide
    actionable risk assessments and recommendations.
    """

    def __init__(
        self,
        name: str = "Water Consumption Risk Analyzer",
        description: str = "Analyzes water consumption risks across permit compliance, trends, and equipment efficiency",
        permit_buffer_percentage: float = 0.15,  # 15% buffer before permit limits
        trend_analysis_days: int = 90,  # Days for trend analysis
        seasonal_comparison_years: int = 3,  # Years for seasonal comparison
    ):
        """
        Initialize the water consumption risk analyzer.
        
        Args:
            name: Analyzer name
            description: Analyzer description
            permit_buffer_percentage: Safety buffer percentage before permit limits
            trend_analysis_days: Number of days for trend analysis
            seasonal_comparison_years: Years of historical data for seasonal analysis
        """
        super().__init__(name, description)
        self.permit_buffer_percentage = permit_buffer_percentage
        self.trend_analysis_days = trend_analysis_days
        self.seasonal_comparison_years = seasonal_comparison_years
        
        # Risk factor weights (must sum to 1.0)
        self.risk_weights = {
            'permit_compliance': 0.35,
            'consumption_trend': 0.25,
            'seasonal_deviation': 0.20,
            'equipment_efficiency': 0.20
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
        Perform comprehensive water consumption risk analysis.
        
        Args:
            data: Dictionary containing:
                - consumption_records: List[WaterConsumptionRecord]
                - permit_data: WaterPermitData
                - equipment_data: List[EquipmentEfficiencyData] (optional)
                - facility_id: str
            **kwargs: Additional analysis parameters
                
        Returns:
            RiskAssessment: Complete risk assessment with factors and recommendations
            
        Raises:
            ValueError: If required data is missing or invalid
        """
        try:
            logger.info(f"Starting water consumption risk analysis for {data.get('facility_id', 'unknown')}")
            
            # Validate input data
            self._validate_water_data(data)
            
            # Extract data components
            consumption_records = data['consumption_records']
            permit_data = data['permit_data']
            equipment_data = data.get('equipment_data', [])
            facility_id = data['facility_id']
            
            # Analyze each risk factor
            risk_factors = []
            
            # 1. Permit compliance analysis
            compliance_factor = await self._analyze_permit_compliance(
                consumption_records, permit_data
            )
            risk_factors.append(compliance_factor)
            
            # 2. Consumption trend analysis
            trend_factor = await self._analyze_consumption_trend(consumption_records)
            risk_factors.append(trend_factor)
            
            # 3. Seasonal pattern analysis
            seasonal_factor = await self._analyze_seasonal_patterns(consumption_records)
            risk_factors.append(seasonal_factor)
            
            # 4. Equipment efficiency analysis
            equipment_factor = await self._analyze_equipment_efficiency(equipment_data)
            risk_factors.append(equipment_factor)
            
            # Generate assessment
            assessment = RiskAssessment.from_factors(
                factors=risk_factors,
                assessment_type="water_consumption_risk",
                assessment_id=str(uuid.uuid4()),
                metadata={
                    'facility_id': facility_id,
                    'analysis_period_days': self.trend_analysis_days,
                    'permit_buffer_percentage': self.permit_buffer_percentage,
                    'total_consumption_records': len(consumption_records),
                    'equipment_count': len(equipment_data)
                }
            )
            
            # Generate specific recommendations
            assessment.recommendations = await self._generate_recommendations(
                assessment, permit_data, consumption_records, equipment_data
            )
            
            logger.info(f"Completed water risk analysis: {assessment.severity.value} risk level")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in water consumption risk analysis: {str(e)}")
            raise

    async def _analyze_permit_compliance(
        self,
        consumption_records: List[WaterConsumptionRecord],
        permit_data: WaterPermitData
    ) -> RiskFactor:
        """
        Analyze permit compliance risk with buffer zones.
        
        Args:
            consumption_records: Water consumption data
            permit_data: Permit information
            
        Returns:
            RiskFactor: Permit compliance risk factor
        """
        try:
            # Calculate current consumption rates
            now = datetime.now()
            
            # Daily compliance check (last 24 hours)
            daily_consumption = self._calculate_period_consumption(
                consumption_records, now - timedelta(days=1), now
            )
            daily_limit_with_buffer = permit_data.daily_limit * (1 - self.permit_buffer_percentage)
            daily_utilization = daily_consumption / permit_data.daily_limit if permit_data.daily_limit > 0 else 0
            
            # Monthly compliance check (last 30 days)
            monthly_consumption = self._calculate_period_consumption(
                consumption_records, now - timedelta(days=30), now
            )
            monthly_limit_with_buffer = permit_data.monthly_limit * (1 - self.permit_buffer_percentage)
            monthly_utilization = monthly_consumption / permit_data.monthly_limit if permit_data.monthly_limit > 0 else 0
            
            # Annual compliance check (last 365 days)
            annual_consumption = self._calculate_period_consumption(
                consumption_records, now - timedelta(days=365), now
            )
            annual_limit_with_buffer = permit_data.annual_limit * (1 - self.permit_buffer_percentage)
            annual_utilization = annual_consumption / permit_data.annual_limit if permit_data.annual_limit > 0 else 0
            
            # Risk score based on highest utilization
            max_utilization = max(daily_utilization, monthly_utilization, annual_utilization)
            
            # Apply buffer zone logic
            if daily_consumption > daily_limit_with_buffer or \
               monthly_consumption > monthly_limit_with_buffer or \
               annual_consumption > annual_limit_with_buffer:
                risk_score = min(0.75 + (max_utilization - (1 - self.permit_buffer_percentage)) * 2, 1.0)
            else:
                risk_score = max_utilization
            
            return RiskFactor(
                name="Permit Compliance",
                value=risk_score,
                weight=self.risk_weights['permit_compliance'],
                severity=self.risk_thresholds.get_severity(risk_score),
                thresholds=self.risk_thresholds,
                description=f"Water permit compliance analysis with {self.permit_buffer_percentage*100}% safety buffer",
                unit="utilization_ratio",
                metadata={
                    'daily_utilization': daily_utilization,
                    'monthly_utilization': monthly_utilization,
                    'annual_utilization': annual_utilization,
                    'daily_consumption_gallons': daily_consumption,
                    'monthly_consumption_gallons': monthly_consumption,
                    'annual_consumption_gallons': annual_consumption,
                    'permit_id': permit_data.permit_id,
                    'buffer_exceeded': max_utilization > (1 - self.permit_buffer_percentage)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in permit compliance analysis: {str(e)}")
            # Return safe default
            return RiskFactor(
                name="Permit Compliance",
                value=0.0,
                weight=self.risk_weights['permit_compliance'],
                severity=RiskSeverity.LOW,
                description="Error in permit compliance analysis",
                metadata={'error': str(e)}
            )

    async def _analyze_consumption_trend(
        self,
        consumption_records: List[WaterConsumptionRecord]
    ) -> RiskFactor:
        """
        Analyze consumption trend using statistical methods.
        
        Args:
            consumption_records: Water consumption data
            
        Returns:
            RiskFactor: Consumption trend risk factor
        """
        try:
            # Sort records by timestamp
            sorted_records = sorted(consumption_records, key=lambda x: x.timestamp)
            
            # Filter to analysis period
            cutoff_date = datetime.now() - timedelta(days=self.trend_analysis_days)
            recent_records = [r for r in sorted_records if r.timestamp >= cutoff_date]
            
            if len(recent_records) < 7:  # Need at least a week of data
                return RiskFactor(
                    name="Consumption Trend",
                    value=0.0,
                    weight=self.risk_weights['consumption_trend'],
                    severity=RiskSeverity.LOW,
                    description="Insufficient data for trend analysis",
                    metadata={'record_count': len(recent_records)}
                )
            
            # Calculate rolling averages and statistics
            daily_consumption = self._aggregate_daily_consumption(recent_records)
            values = list(daily_consumption.values())
            
            # Calculate trend metrics
            rolling_mean = statistics.mean(values)
            rolling_std = statistics.stdev(values) if len(values) > 1 else 0.0
            
            # Z-score analysis for anomalies
            recent_avg = statistics.mean(values[-7:]) if len(values) >= 7 else rolling_mean
            z_score = abs((recent_avg - rolling_mean) / rolling_std) if rolling_std > 0 else 0.0
            
            # Linear trend calculation
            trend_slope = self._calculate_trend_slope(daily_consumption)
            
            # Risk scoring
            # High positive trend = higher risk
            # High z-score = higher risk (anomalous consumption)
            trend_risk = min(abs(trend_slope) / rolling_mean, 1.0) if rolling_mean > 0 else 0.0
            anomaly_risk = min(z_score / 3.0, 1.0)  # Z-score > 3 is highly anomalous
            
            risk_score = max(trend_risk, anomaly_risk)
            
            return RiskFactor(
                name="Consumption Trend",
                value=risk_score,
                weight=self.risk_weights['consumption_trend'],
                severity=self.risk_thresholds.get_severity(risk_score),
                thresholds=self.risk_thresholds,
                description=f"Consumption trend analysis over {self.trend_analysis_days} days",
                unit="trend_risk_score",
                metadata={
                    'rolling_mean_gallons': rolling_mean,
                    'rolling_std_gallons': rolling_std,
                    'z_score': z_score,
                    'trend_slope': trend_slope,
                    'recent_7day_avg': recent_avg,
                    'analysis_days': len(values),
                    'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing'
                }
            )
            
        except Exception as e:
            logger.error(f"Error in consumption trend analysis: {str(e)}")
            return RiskFactor(
                name="Consumption Trend",
                value=0.0,
                weight=self.risk_weights['consumption_trend'],
                severity=RiskSeverity.LOW,
                description="Error in trend analysis",
                metadata={'error': str(e)}
            )

    async def _analyze_seasonal_patterns(
        self,
        consumption_records: List[WaterConsumptionRecord]
    ) -> RiskFactor:
        """
        Analyze seasonal consumption patterns and deviations.
        
        Args:
            consumption_records: Water consumption data
            
        Returns:
            RiskFactor: Seasonal pattern risk factor
        """
        try:
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            # Get historical data for the same month over previous years
            historical_monthly_consumption = []
            current_monthly_consumption = 0.0
            
            for year_offset in range(1, self.seasonal_comparison_years + 1):
                target_year = current_year - year_offset
                month_consumption = self._calculate_month_consumption(
                    consumption_records, target_year, current_month
                )
                if month_consumption > 0:
                    historical_monthly_consumption.append(month_consumption)
            
            # Get current month consumption (so far)
            current_monthly_consumption = self._calculate_month_consumption(
                consumption_records, current_year, current_month
            )
            
            if not historical_monthly_consumption:
                return RiskFactor(
                    name="Seasonal Deviation",
                    value=0.0,
                    weight=self.risk_weights['seasonal_deviation'],
                    severity=RiskSeverity.LOW,
                    description="Insufficient historical data for seasonal analysis",
                    metadata={'historical_years': 0}
                )
            
            # Calculate seasonal baseline
            historical_avg = statistics.mean(historical_monthly_consumption)
            historical_std = statistics.stdev(historical_monthly_consumption) if len(historical_monthly_consumption) > 1 else 0.0
            
            # Project current month consumption based on days elapsed
            days_in_month = (datetime(current_year, current_month + 1, 1) - datetime(current_year, current_month, 1)).days if current_month < 12 else 31
            days_elapsed = datetime.now().day
            projected_monthly = (current_monthly_consumption / days_elapsed) * days_in_month if days_elapsed > 0 else 0.0
            
            # Calculate seasonal deviation
            if historical_avg > 0:
                deviation_ratio = abs(projected_monthly - historical_avg) / historical_avg
                # Z-score for seasonal anomaly
                seasonal_z_score = abs(projected_monthly - historical_avg) / historical_std if historical_std > 0 else 0.0
            else:
                deviation_ratio = 0.0
                seasonal_z_score = 0.0
            
            # Risk scoring based on deviation from historical patterns
            risk_score = min(max(deviation_ratio, seasonal_z_score / 3.0), 1.0)
            
            return RiskFactor(
                name="Seasonal Deviation",
                value=risk_score,
                weight=self.risk_weights['seasonal_deviation'],
                severity=self.risk_thresholds.get_severity(risk_score),
                thresholds=self.risk_thresholds,
                description=f"Seasonal consumption pattern analysis ({self.seasonal_comparison_years} year comparison)",
                unit="seasonal_deviation_ratio",
                metadata={
                    'current_month': current_month,
                    'historical_avg_gallons': historical_avg,
                    'historical_std_gallons': historical_std,
                    'projected_monthly_gallons': projected_monthly,
                    'current_monthly_gallons': current_monthly_consumption,
                    'deviation_ratio': deviation_ratio,
                    'seasonal_z_score': seasonal_z_score,
                    'historical_years_count': len(historical_monthly_consumption),
                    'days_elapsed_in_month': days_elapsed
                }
            )
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern analysis: {str(e)}")
            return RiskFactor(
                name="Seasonal Deviation",
                value=0.0,
                weight=self.risk_weights['seasonal_deviation'],
                severity=RiskSeverity.LOW,
                description="Error in seasonal analysis",
                metadata={'error': str(e)}
            )

    async def _analyze_equipment_efficiency(
        self,
        equipment_data: List[EquipmentEfficiencyData]
    ) -> RiskFactor:
        """
        Analyze equipment efficiency impact on water consumption risk.
        
        Args:
            equipment_data: Equipment efficiency data
            
        Returns:
            RiskFactor: Equipment efficiency risk factor
        """
        try:
            if not equipment_data:
                return RiskFactor(
                    name="Equipment Efficiency",
                    value=0.0,
                    weight=self.risk_weights['equipment_efficiency'],
                    severity=RiskSeverity.LOW,
                    description="No equipment data available for efficiency analysis",
                    metadata={'equipment_count': 0}
                )
            
            # Analyze equipment efficiency trends and risks
            efficiency_risks = []
            maintenance_risks = []
            
            for equipment in equipment_data:
                if equipment.operational_status != 'active':
                    continue
                
                # Efficiency degradation risk
                if equipment.baseline_efficiency > 0:
                    efficiency_ratio = equipment.current_efficiency / equipment.baseline_efficiency
                    # Risk increases as efficiency degrades (ratio < 1.0 means worse efficiency)
                    efficiency_risk = max(0.0, 1.0 - efficiency_ratio)
                    efficiency_risks.append(efficiency_risk)
                
                # Maintenance schedule risk
                days_since_maintenance = (datetime.now() - equipment.last_maintenance).days
                # Risk increases with time since maintenance (assume 90 days is high risk)
                maintenance_risk = min(days_since_maintenance / 90.0, 1.0)
                maintenance_risks.append(maintenance_risk)
            
            if not efficiency_risks and not maintenance_risks:
                overall_risk = 0.0
            else:
                # Calculate weighted average of all equipment risks
                avg_efficiency_risk = statistics.mean(efficiency_risks) if efficiency_risks else 0.0
                avg_maintenance_risk = statistics.mean(maintenance_risks) if maintenance_risks else 0.0
                
                # Combined risk (weighted toward efficiency)
                overall_risk = (avg_efficiency_risk * 0.7) + (avg_maintenance_risk * 0.3)
            
            # Count critical equipment
            critical_equipment = [
                eq for eq in equipment_data 
                if eq.operational_status == 'active' and 
                (eq.current_efficiency / eq.baseline_efficiency < 0.8 if eq.baseline_efficiency > 0 else False)
            ]
            
            return RiskFactor(
                name="Equipment Efficiency",
                value=overall_risk,
                weight=self.risk_weights['equipment_efficiency'],
                severity=self.risk_thresholds.get_severity(overall_risk),
                thresholds=self.risk_thresholds,
                description="Equipment efficiency and maintenance risk analysis",
                unit="efficiency_risk_score",
                metadata={
                    'total_equipment': len(equipment_data),
                    'active_equipment': len([eq for eq in equipment_data if eq.operational_status == 'active']),
                    'critical_equipment': len(critical_equipment),
                    'avg_efficiency_risk': statistics.mean(efficiency_risks) if efficiency_risks else 0.0,
                    'avg_maintenance_risk': statistics.mean(maintenance_risks) if maintenance_risks else 0.0,
                    'equipment_needing_maintenance': len([eq for eq in equipment_data if (datetime.now() - eq.last_maintenance).days > 60])
                }
            )
            
        except Exception as e:
            logger.error(f"Error in equipment efficiency analysis: {str(e)}")
            return RiskFactor(
                name="Equipment Efficiency",
                value=0.0,
                weight=self.risk_weights['equipment_efficiency'],
                severity=RiskSeverity.LOW,
                description="Error in equipment efficiency analysis",
                metadata={'error': str(e)}
            )

    async def _generate_recommendations(
        self,
        assessment: RiskAssessment,
        permit_data: WaterPermitData,
        consumption_records: List[WaterConsumptionRecord],
        equipment_data: List[EquipmentEfficiencyData]
    ) -> List[str]:
        """
        Generate specific recommendations based on risk assessment.
        
        Args:
            assessment: Risk assessment results
            permit_data: Permit information
            consumption_records: Consumption data
            equipment_data: Equipment data
            
        Returns:
            List[str]: Actionable recommendations
        """
        recommendations = []
        
        try:
            # Get high-risk factors
            high_risk_factors = assessment.get_high_risk_factors()
            
            for factor in high_risk_factors:
                if factor.name == "Permit Compliance":
                    if factor.metadata.get('buffer_exceeded', False):
                        recommendations.append(
                            f"CRITICAL: Water consumption has exceeded the {self.permit_buffer_percentage*100}% "
                            f"safety buffer. Implement immediate conservation measures to avoid permit violations."
                        )
                    
                    max_utilization = max(
                        factor.metadata.get('daily_utilization', 0),
                        factor.metadata.get('monthly_utilization', 0),
                        factor.metadata.get('annual_utilization', 0)
                    )
                    
                    if max_utilization > 0.8:
                        recommendations.append(
                            "Implement water conservation protocols and consider permit renewal or modification "
                            "to increase allowable limits before reaching current permit thresholds."
                        )
                
                elif factor.name == "Consumption Trend":
                    if factor.metadata.get('trend_direction') == 'increasing':
                        recommendations.append(
                            f"Rising water consumption trend detected (slope: {factor.metadata.get('trend_slope', 0):.2f}). "
                            "Conduct facility audit to identify sources of increased usage and implement targeted conservation."
                        )
                    
                    if factor.metadata.get('z_score', 0) > 2.0:
                        recommendations.append(
                            "Anomalous consumption patterns detected. Check for equipment malfunctions, "
                            "leaks, or changes in operational procedures that may be driving unusual usage."
                        )
                
                elif factor.name == "Seasonal Deviation":
                    recommendations.append(
                        f"Current consumption is significantly different from historical patterns for this month. "
                        f"Investigate operational changes and consider adjusting seasonal planning strategies."
                    )
                
                elif factor.name == "Equipment Efficiency":
                    critical_count = factor.metadata.get('critical_equipment', 0)
                    if critical_count > 0:
                        recommendations.append(
                            f"Equipment efficiency analysis identifies {critical_count} critical unit(s) "
                            "operating below 80% baseline efficiency. Schedule immediate maintenance or replacement."
                        )
                    
                    maintenance_needed = factor.metadata.get('equipment_needing_maintenance', 0)
                    if maintenance_needed > 0:
                        recommendations.append(
                            f"{maintenance_needed} equipment unit(s) are overdue for maintenance (>60 days). "
                            "Implement preventive maintenance schedule to optimize water efficiency."
                        )
            
            # Overall risk-level recommendations
            if assessment.severity == RiskSeverity.CRITICAL:
                recommendations.insert(0, 
                    "EMERGENCY RESPONSE REQUIRED: Critical water consumption risk detected. "
                    "Activate emergency water conservation protocols and notify regulatory authorities if necessary."
                )
            elif assessment.severity == RiskSeverity.HIGH:
                recommendations.insert(0,
                    "HIGH PRIORITY: Immediate action required to mitigate water consumption risks. "
                    "Implement short-term conservation measures while developing long-term solutions."
                )
            elif assessment.severity == RiskSeverity.MEDIUM:
                recommendations.insert(0,
                    "Monitor water consumption closely and implement preventive measures to avoid risk escalation."
                )
            
            # Add general recommendations if no specific ones generated
            if not recommendations:
                recommendations.append("Continue monitoring water consumption patterns and maintain current conservation practices.")
                recommendations.append("Consider implementing additional water efficiency measures as part of continuous improvement.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating specific recommendations. Please review risk factors manually."]

    def _validate_water_data(self, data: Dict[str, Any]) -> None:
        """Validate input data for water risk analysis."""
        self.validate_input_data(data)
        
        required_fields = ['consumption_records', 'permit_data', 'facility_id']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from input data")
        
        if not isinstance(data['consumption_records'], list):
            raise ValueError("consumption_records must be a list")
        
        if not data['consumption_records']:
            raise ValueError("At least one consumption record is required")

    def _calculate_period_consumption(
        self,
        records: List[WaterConsumptionRecord],
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """Calculate total consumption for a specific time period."""
        period_records = [
            r for r in records 
            if start_date <= r.timestamp <= end_date and r.quality_flag
        ]
        return sum(r.consumption_gallons for r in period_records)

    def _aggregate_daily_consumption(
        self,
        records: List[WaterConsumptionRecord]
    ) -> Dict[str, float]:
        """Aggregate consumption records by day."""
        daily_consumption = {}
        
        for record in records:
            if not record.quality_flag:
                continue
            
            date_key = record.timestamp.date().isoformat()
            daily_consumption[date_key] = daily_consumption.get(date_key, 0.0) + record.consumption_gallons
        
        return daily_consumption

    def _calculate_trend_slope(self, daily_consumption: Dict[str, float]) -> float:
        """Calculate linear trend slope for daily consumption."""
        if len(daily_consumption) < 2:
            return 0.0
        
        # Convert to sorted list of (day_number, consumption) pairs
        sorted_items = sorted(daily_consumption.items())
        x_values = list(range(len(sorted_items)))
        y_values = [item[1] for item in sorted_items]
        
        # Calculate linear regression slope
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Slope formula: (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def _calculate_month_consumption(
        self,
        records: List[WaterConsumptionRecord],
        year: int,
        month: int
    ) -> float:
        """Calculate total consumption for a specific month and year."""
        month_records = [
            r for r in records 
            if r.timestamp.year == year and r.timestamp.month == month and r.quality_flag
        ]
        return sum(r.consumption_gallons for r in month_records)

# Create aliases for backward compatibility  
WaterRiskAnalyzer = WaterConsumptionRiskAnalyzer
