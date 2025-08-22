"""
Electricity Consumption Risk Analyzer

This module implements comprehensive electricity consumption risk analysis for EHS Analytics,
providing demand management, cost optimization, power quality monitoring, carbon footprint
assessment, and grid reliability analysis following ISO 31000 risk management guidelines.
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
class ElectricalContractData:
    """Electrical service contract information for demand and cost analysis."""
    contract_id: str
    facility_id: str
    contracted_demand_kw: float  # Contracted peak demand in kW
    demand_rate_per_kw: float  # Demand charge per kW
    energy_rate_peak: float  # Peak energy rate per kWh
    energy_rate_offpeak: float  # Off-peak energy rate per kWh
    power_factor_threshold: float = 0.95  # Minimum acceptable power factor
    power_factor_penalty_rate: float = 0.10  # Additional charge for poor power factor
    contract_start: datetime = field(default_factory=datetime.now)
    contract_end: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=365))
    utility_provider: str = "local_utility"
    time_of_use_schedule: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElectricityConsumptionRecord:
    """Individual electricity consumption and quality data point."""
    timestamp: datetime
    facility_id: str
    energy_kwh: float  # Energy consumption in kWh
    demand_kw: float  # Instantaneous demand in kW
    voltage_l1: Optional[float] = None  # Line 1 voltage
    voltage_l2: Optional[float] = None  # Line 2 voltage
    voltage_l3: Optional[float] = None  # Line 3 voltage
    current_l1: Optional[float] = None  # Line 1 current
    current_l2: Optional[float] = None  # Line 2 current
    current_l3: Optional[float] = None  # Line 3 current
    power_factor: Optional[float] = None  # Overall power factor
    frequency_hz: Optional[float] = 60.0  # System frequency
    total_harmonic_distortion: Optional[float] = None  # THD percentage
    meter_id: Optional[str] = None
    circuit_id: Optional[str] = None
    time_of_use_period: str = "standard"  # peak, offpeak, standard
    quality_flag: bool = True  # True if data is reliable
    carbon_intensity_factor: Optional[float] = None  # kg CO2e per kWh
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerQualityThresholds:
    """Power quality thresholds for risk assessment."""
    voltage_tolerance_percent: float = 5.0  # ±5% voltage tolerance
    frequency_tolerance_hz: float = 0.5  # ±0.5 Hz frequency tolerance
    power_factor_minimum: float = 0.95  # Minimum acceptable power factor
    thd_maximum_percent: float = 8.0  # Maximum acceptable THD
    voltage_unbalance_max_percent: float = 2.0  # Maximum voltage unbalance


@dataclass
class CarbonEmissionsData:
    """Carbon emissions and compliance data."""
    facility_id: str
    reporting_period: str  # e.g., "2024-Q1"
    annual_emissions_target_kg_co2e: float
    current_emissions_kg_co2e: float
    renewable_energy_percentage: float = 0.0
    emission_factor_grid_kg_co2e_per_kwh: float = 0.4  # Grid average
    compliance_framework: str = "GHG_Protocol"  # GHG_Protocol, CSRD, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class ElectricityConsumptionRiskAnalyzer(BaseRiskAnalyzer):
    """
    Comprehensive electricity consumption risk analyzer implementing ISO 31000 guidelines.
    
    This analyzer evaluates multiple risk factors including demand management, energy costs,
    power quality, carbon compliance, and grid reliability to provide actionable risk
    assessments and optimization recommendations.
    """

    def __init__(
        self,
        name: str = "Electricity Consumption Risk Analyzer",
        description: str = "Analyzes electricity consumption risks across demand, costs, quality, and emissions",
        demand_safety_margin: float = 0.10,  # 10% safety margin below contracted demand
        cost_volatility_threshold: float = 0.15,  # 15% cost increase threshold
        trend_analysis_days: int = 90,  # Days for trend analysis
        power_quality_thresholds: Optional[PowerQualityThresholds] = None,
    ):
        """
        Initialize the electricity consumption risk analyzer.
        
        Args:
            name: Analyzer name
            description: Analyzer description
            demand_safety_margin: Safety margin percentage below contracted demand
            cost_volatility_threshold: Threshold for cost volatility alerts
            trend_analysis_days: Number of days for trend analysis
            power_quality_thresholds: Power quality thresholds for assessment
        """
        super().__init__(name, description)
        self.demand_safety_margin = demand_safety_margin
        self.cost_volatility_threshold = cost_volatility_threshold
        self.trend_analysis_days = trend_analysis_days
        self.power_quality_thresholds = power_quality_thresholds or PowerQualityThresholds()
        
        # Risk factor weights (must sum to 1.0)
        self.risk_weights = {
            'demand_management': 0.25,      # Peak demand vs contract risk
            'energy_cost_trends': 0.20,     # Cost volatility and budget impact
            'power_quality': 0.20,          # Voltage, frequency, power factor issues
            'carbon_compliance': 0.20,      # Emissions and regulatory compliance
            'grid_reliability': 0.15        # Supply reliability and outage risk
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
        Perform comprehensive electricity consumption risk analysis.
        
        Args:
            data: Dictionary containing:
                - consumption_records: List[ElectricityConsumptionRecord]
                - contract_data: ElectricalContractData
                - emissions_data: CarbonEmissionsData (optional)
                - facility_id: str
                - grid_reliability_data: Dict (optional)
            **kwargs: Additional analysis parameters
                
        Returns:
            RiskAssessment: Complete risk assessment with factors and recommendations
            
        Raises:
            ValueError: If required data is missing or invalid
        """
        try:
            logger.info(f"Starting electricity consumption risk analysis for {data.get('facility_id', 'unknown')}")
            
            # Validate input data
            self._validate_electricity_data(data)
            
            # Extract data components
            consumption_records = data['consumption_records']
            contract_data = data['contract_data']
            emissions_data = data.get('emissions_data')
            facility_id = data['facility_id']
            grid_reliability_data = data.get('grid_reliability_data', {})
            
            # Analyze each risk factor
            risk_factors = []
            
            # 1. Demand management analysis (peak demand vs contracted capacity)
            demand_factor = await self._analyze_demand_management(
                consumption_records, contract_data
            )
            risk_factors.append(demand_factor)
            
            # 2. Energy cost trends analysis
            cost_factor = await self._analyze_energy_cost_trends(
                consumption_records, contract_data
            )
            risk_factors.append(cost_factor)
            
            # 3. Power quality analysis
            quality_factor = await self._analyze_power_quality(consumption_records)
            risk_factors.append(quality_factor)
            
            # 4. Carbon compliance analysis
            carbon_factor = await self._analyze_carbon_compliance(
                consumption_records, emissions_data
            )
            risk_factors.append(carbon_factor)
            
            # 5. Grid reliability assessment
            reliability_factor = await self._analyze_grid_reliability(
                consumption_records, grid_reliability_data
            )
            risk_factors.append(reliability_factor)
            
            # Generate assessment
            assessment = RiskAssessment.from_factors(
                factors=risk_factors,
                assessment_type="electricity_consumption_risk",
                assessment_id=str(uuid.uuid4()),
                metadata={
                    'facility_id': facility_id,
                    'analysis_period_days': self.trend_analysis_days,
                    'demand_safety_margin': self.demand_safety_margin,
                    'total_consumption_records': len(consumption_records),
                    'contracted_demand_kw': contract_data.contracted_demand_kw,
                    'contract_id': contract_data.contract_id
                }
            )
            
            # Generate specific recommendations
            assessment.recommendations = await self._generate_recommendations(
                assessment, contract_data, consumption_records, emissions_data
            )
            
            logger.info(f"Completed electricity risk analysis: {assessment.severity.value} risk level")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in electricity consumption risk analysis: {str(e)}")
            raise

    async def _analyze_demand_management(
        self,
        consumption_records: List[ElectricityConsumptionRecord],
        contract_data: ElectricalContractData
    ) -> RiskFactor:
        """
        Analyze demand management risk with safety margins.
        
        Args:
            consumption_records: Electricity consumption data
            contract_data: Contract information
            
        Returns:
            RiskFactor: Demand management risk factor
        """
        try:
            # Calculate peak demand statistics
            demands = [record.demand_kw for record in consumption_records if record.demand_kw > 0]
            
            if not demands:
                return RiskFactor(
                    name="Demand Management",
                    value=0.0,
                    weight=self.risk_weights['demand_management'],
                    severity=RiskSeverity.LOW,
                    description="No demand data available for analysis",
                    metadata={'demand_records': 0}
                )
            
            current_peak_demand = max(demands)
            avg_demand = statistics.mean(demands)
            demand_std = statistics.stdev(demands) if len(demands) > 1 else 0.0
            
            # Calculate demand utilization
            contracted_demand = contract_data.contracted_demand_kw
            safe_demand_threshold = contracted_demand * (1 - self.demand_safety_margin)
            
            demand_utilization = current_peak_demand / contracted_demand if contracted_demand > 0 else 0.0
            safety_margin_utilization = current_peak_demand / safe_demand_threshold if safe_demand_threshold > 0 else 0.0
            
            # Calculate demand variability (higher variability = higher risk)
            demand_coefficient_variation = demand_std / avg_demand if avg_demand > 0 else 0.0
            
            # Risk scoring
            # Base risk from demand utilization
            utilization_risk = min(demand_utilization, 1.0)
            
            # Additional risk if safety margin exceeded
            if current_peak_demand > safe_demand_threshold:
                margin_risk = min((safety_margin_utilization - (1 - self.demand_safety_margin)) * 2, 0.5)
            else:
                margin_risk = 0.0
            
            # Variability risk (high variability makes demand unpredictable)
            variability_risk = min(demand_coefficient_variation, 0.3)
            
            risk_score = min(utilization_risk + margin_risk + variability_risk, 1.0)
            
            # Time-of-use analysis
            peak_period_demands = [
                record.demand_kw for record in consumption_records
                if record.time_of_use_period == "peak" and record.demand_kw > 0
            ]
            peak_period_avg = statistics.mean(peak_period_demands) if peak_period_demands else 0.0
            
            return RiskFactor(
                name="Demand Management",
                value=risk_score,
                weight=self.risk_weights['demand_management'],
                severity=self.risk_thresholds.get_severity(risk_score),
                thresholds=self.risk_thresholds,
                description=f"Demand management analysis with {self.demand_safety_margin*100}% safety margin",
                unit="utilization_risk_score",
                metadata={
                    'current_peak_demand_kw': current_peak_demand,
                    'contracted_demand_kw': contracted_demand,
                    'demand_utilization': demand_utilization,
                    'safety_margin_utilization': safety_margin_utilization,
                    'avg_demand_kw': avg_demand,
                    'demand_std_kw': demand_std,
                    'demand_coefficient_variation': demand_coefficient_variation,
                    'safe_demand_threshold_kw': safe_demand_threshold,
                    'margin_exceeded': current_peak_demand > safe_demand_threshold,
                    'peak_period_avg_demand_kw': peak_period_avg,
                    'demand_records_count': len(demands)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in demand management analysis: {str(e)}")
            return RiskFactor(
                name="Demand Management",
                value=0.0,
                weight=self.risk_weights['demand_management'],
                severity=RiskSeverity.LOW,
                description="Error in demand management analysis",
                metadata={'error': str(e)}
            )

    async def _analyze_energy_cost_trends(
        self,
        consumption_records: List[ElectricityConsumptionRecord],
        contract_data: ElectricalContractData
    ) -> RiskFactor:
        """
        Analyze energy cost trends and volatility risk.
        
        Args:
            consumption_records: Electricity consumption data
            contract_data: Contract information
            
        Returns:
            RiskFactor: Energy cost trends risk factor
        """
        try:
            # Calculate daily costs based on time-of-use rates
            daily_costs = {}
            daily_consumption = {}
            
            for record in consumption_records:
                date_key = record.timestamp.date().isoformat()
                
                # Calculate cost based on time-of-use period
                if record.time_of_use_period == "peak":
                    energy_cost = record.energy_kwh * contract_data.energy_rate_peak
                else:
                    energy_cost = record.energy_kwh * contract_data.energy_rate_offpeak
                
                # Add demand charges (allocated daily)
                demand_cost = record.demand_kw * contract_data.demand_rate_per_kw / 30  # Monthly demand charge
                
                # Power factor penalties
                power_factor_penalty = 0.0
                if record.power_factor and record.power_factor < contract_data.power_factor_threshold:
                    penalty_factor = (contract_data.power_factor_threshold - record.power_factor) / contract_data.power_factor_threshold
                    power_factor_penalty = energy_cost * contract_data.power_factor_penalty_rate * penalty_factor
                
                total_cost = energy_cost + demand_cost + power_factor_penalty
                
                daily_costs[date_key] = daily_costs.get(date_key, 0.0) + total_cost
                daily_consumption[date_key] = daily_consumption.get(date_key, 0.0) + record.energy_kwh
            
            if len(daily_costs) < 7:
                return RiskFactor(
                    name="Energy Cost Trends",
                    value=0.0,
                    weight=self.risk_weights['energy_cost_trends'],
                    severity=RiskSeverity.LOW,
                    description="Insufficient data for cost trend analysis",
                    metadata={'cost_records': len(daily_costs)}
                )
            
            # Calculate cost metrics
            costs = list(daily_costs.values())
            avg_daily_cost = statistics.mean(costs)
            cost_std = statistics.stdev(costs) if len(costs) > 1 else 0.0
            cost_coefficient_variation = cost_std / avg_daily_cost if avg_daily_cost > 0 else 0.0
            
            # Calculate cost per kWh trend
            cost_per_kwh = []
            for date_key in daily_costs:
                if daily_consumption[date_key] > 0:
                    cost_per_kwh.append(daily_costs[date_key] / daily_consumption[date_key])
            
            # Trend analysis
            recent_costs = costs[-7:] if len(costs) >= 7 else costs
            recent_avg_cost = statistics.mean(recent_costs)
            cost_trend = (recent_avg_cost - avg_daily_cost) / avg_daily_cost if avg_daily_cost > 0 else 0.0
            
            # Risk scoring
            # Volatility risk (coefficient of variation)
            volatility_risk = min(cost_coefficient_variation / self.cost_volatility_threshold, 1.0)
            
            # Cost increase trend risk
            trend_risk = max(cost_trend, 0.0)  # Only consider increasing costs as risk
            
            # Power factor penalty risk
            power_factor_records = [r for r in consumption_records if r.power_factor is not None]
            poor_power_factor_ratio = len([r for r in power_factor_records 
                                         if r.power_factor < contract_data.power_factor_threshold]) / len(power_factor_records) if power_factor_records else 0.0
            
            risk_score = min(volatility_risk + trend_risk + (poor_power_factor_ratio * 0.3), 1.0)
            
            # Time-of-use optimization analysis
            peak_consumption_ratio = len([r for r in consumption_records if r.time_of_use_period == "peak"]) / len(consumption_records)
            
            return RiskFactor(
                name="Energy Cost Trends",
                value=risk_score,
                weight=self.risk_weights['energy_cost_trends'],
                severity=self.risk_thresholds.get_severity(risk_score),
                thresholds=self.risk_thresholds,
                description=f"Energy cost trends and volatility analysis ({self.cost_volatility_threshold*100}% volatility threshold)",
                unit="cost_risk_score",
                metadata={
                    'avg_daily_cost_usd': avg_daily_cost,
                    'cost_std_usd': cost_std,
                    'cost_coefficient_variation': cost_coefficient_variation,
                    'cost_trend_percentage': cost_trend * 100,
                    'recent_avg_cost_usd': recent_avg_cost,
                    'avg_cost_per_kwh_usd': statistics.mean(cost_per_kwh) if cost_per_kwh else 0.0,
                    'poor_power_factor_ratio': poor_power_factor_ratio,
                    'peak_consumption_ratio': peak_consumption_ratio,
                    'total_cost_days': len(daily_costs)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in energy cost trends analysis: {str(e)}")
            return RiskFactor(
                name="Energy Cost Trends",
                value=0.0,
                weight=self.risk_weights['energy_cost_trends'],
                severity=RiskSeverity.LOW,
                description="Error in cost trends analysis",
                metadata={'error': str(e)}
            )

    async def _analyze_power_quality(
        self,
        consumption_records: List[ElectricityConsumptionRecord]
    ) -> RiskFactor:
        """
        Analyze power quality issues including voltage, frequency, and harmonics.
        
        Args:
            consumption_records: Electricity consumption data
            
        Returns:
            RiskFactor: Power quality risk factor
        """
        try:
            # Filter records with power quality data
            quality_records = [r for r in consumption_records if any([
                r.voltage_l1, r.voltage_l2, r.voltage_l3,
                r.frequency_hz, r.total_harmonic_distortion
            ])]
            
            if not quality_records:
                return RiskFactor(
                    name="Power Quality",
                    value=0.0,
                    weight=self.risk_weights['power_quality'],
                    severity=RiskSeverity.LOW,
                    description="No power quality data available for analysis",
                    metadata={'quality_records': 0}
                )
            
            quality_issues = {
                'voltage_deviations': 0,
                'frequency_deviations': 0,
                'poor_power_factor': 0,
                'high_thd': 0,
                'voltage_unbalance': 0
            }
            
            total_records = len(quality_records)
            
            # Analyze each record for quality issues
            for record in quality_records:
                # Voltage analysis (assuming 480V nominal for commercial)
                nominal_voltage = 480.0
                voltages = [v for v in [record.voltage_l1, record.voltage_l2, record.voltage_l3] if v is not None]
                
                if voltages:
                    avg_voltage = statistics.mean(voltages)
                    voltage_deviation = abs(avg_voltage - nominal_voltage) / nominal_voltage * 100
                    
                    if voltage_deviation > self.power_quality_thresholds.voltage_tolerance_percent:
                        quality_issues['voltage_deviations'] += 1
                    
                    # Voltage unbalance
                    if len(voltages) == 3:
                        max_voltage = max(voltages)
                        min_voltage = min(voltages)
                        unbalance_percent = ((max_voltage - min_voltage) / avg_voltage) * 100
                        
                        if unbalance_percent > self.power_quality_thresholds.voltage_unbalance_max_percent:
                            quality_issues['voltage_unbalance'] += 1
                
                # Frequency analysis
                if record.frequency_hz:
                    frequency_deviation = abs(record.frequency_hz - 60.0)
                    if frequency_deviation > self.power_quality_thresholds.frequency_tolerance_hz:
                        quality_issues['frequency_deviations'] += 1
                
                # Power factor analysis
                if record.power_factor:
                    if record.power_factor < self.power_quality_thresholds.power_factor_minimum:
                        quality_issues['poor_power_factor'] += 1
                
                # THD analysis
                if record.total_harmonic_distortion:
                    if record.total_harmonic_distortion > self.power_quality_thresholds.thd_maximum_percent:
                        quality_issues['high_thd'] += 1
            
            # Calculate risk scores for each quality issue
            voltage_risk = quality_issues['voltage_deviations'] / total_records
            frequency_risk = quality_issues['frequency_deviations'] / total_records
            power_factor_risk = quality_issues['poor_power_factor'] / total_records
            thd_risk = quality_issues['high_thd'] / total_records
            unbalance_risk = quality_issues['voltage_unbalance'] / total_records
            
            # Overall power quality risk (weighted combination)
            risk_score = min(
                voltage_risk * 0.3 +
                frequency_risk * 0.2 +
                power_factor_risk * 0.2 +
                thd_risk * 0.15 +
                unbalance_risk * 0.15,
                1.0
            )
            
            # Calculate average power quality metrics
            avg_power_factor = statistics.mean([r.power_factor for r in quality_records if r.power_factor]) if any(r.power_factor for r in quality_records) else None
            avg_thd = statistics.mean([r.total_harmonic_distortion for r in quality_records if r.total_harmonic_distortion]) if any(r.total_harmonic_distortion for r in quality_records) else None
            
            return RiskFactor(
                name="Power Quality",
                value=risk_score,
                weight=self.risk_weights['power_quality'],
                severity=self.risk_thresholds.get_severity(risk_score),
                thresholds=self.risk_thresholds,
                description="Power quality analysis including voltage, frequency, power factor, and harmonics",
                unit="quality_risk_score",
                metadata={
                    'total_quality_records': total_records,
                    'voltage_deviations_count': quality_issues['voltage_deviations'],
                    'frequency_deviations_count': quality_issues['frequency_deviations'],
                    'poor_power_factor_count': quality_issues['poor_power_factor'],
                    'high_thd_count': quality_issues['high_thd'],
                    'voltage_unbalance_count': quality_issues['voltage_unbalance'],
                    'voltage_deviation_rate': voltage_risk,
                    'frequency_deviation_rate': frequency_risk,
                    'poor_power_factor_rate': power_factor_risk,
                    'high_thd_rate': thd_risk,
                    'voltage_unbalance_rate': unbalance_risk,
                    'avg_power_factor': avg_power_factor,
                    'avg_thd_percent': avg_thd
                }
            )
            
        except Exception as e:
            logger.error(f"Error in power quality analysis: {str(e)}")
            return RiskFactor(
                name="Power Quality",
                value=0.0,
                weight=self.risk_weights['power_quality'],
                severity=RiskSeverity.LOW,
                description="Error in power quality analysis",
                metadata={'error': str(e)}
            )

    async def _analyze_carbon_compliance(
        self,
        consumption_records: List[ElectricityConsumptionRecord],
        emissions_data: Optional[CarbonEmissionsData]
    ) -> RiskFactor:
        """
        Analyze carbon footprint and emissions compliance risk.
        
        Args:
            consumption_records: Electricity consumption data
            emissions_data: Carbon emissions data (optional)
            
        Returns:
            RiskFactor: Carbon compliance risk factor
        """
        try:
            # Calculate total energy consumption
            total_energy_kwh = sum(record.energy_kwh for record in consumption_records)
            
            if total_energy_kwh == 0:
                return RiskFactor(
                    name="Carbon Compliance",
                    value=0.0,
                    weight=self.risk_weights['carbon_compliance'],
                    severity=RiskSeverity.LOW,
                    description="No energy consumption data for carbon analysis",
                    metadata={'total_energy_kwh': 0}
                )
            
            # Use emissions data if available, otherwise use grid average
            if emissions_data:
                emission_factor = emissions_data.emission_factor_grid_kg_co2e_per_kwh
                annual_target = emissions_data.annual_emissions_target_kg_co2e
                current_emissions = emissions_data.current_emissions_kg_co2e
                renewable_percentage = emissions_data.renewable_energy_percentage
            else:
                emission_factor = 0.4  # Default grid average kg CO2e/kWh
                annual_target = total_energy_kwh * emission_factor * 0.8  # Assume 20% reduction target
                current_emissions = total_energy_kwh * emission_factor
                renewable_percentage = 0.0
            
            # Calculate emissions from electricity consumption
            period_days = (max(r.timestamp for r in consumption_records) - 
                          min(r.timestamp for r in consumption_records)).days + 1
            
            # Project annual emissions based on current consumption
            projected_annual_emissions = (total_energy_kwh / period_days) * 365 * emission_factor
            
            # Adjust for renewable energy
            adjusted_emissions = projected_annual_emissions * (1 - renewable_percentage / 100)
            
            # Risk scoring based on compliance with targets
            if annual_target > 0:
                emissions_ratio = adjusted_emissions / annual_target
                compliance_risk = max(0.0, min((emissions_ratio - 0.8) / 0.4, 1.0))  # Risk starts at 80% of target
            else:
                compliance_risk = 0.0
            
            # Additional risk factors
            # Carbon intensity trend (using individual record factors if available)
            carbon_intensity_records = [r.carbon_intensity_factor for r in consumption_records if r.carbon_intensity_factor]
            intensity_trend_risk = 0.0
            
            if len(carbon_intensity_records) > 7:
                recent_intensity = statistics.mean(carbon_intensity_records[-7:])
                historical_intensity = statistics.mean(carbon_intensity_records[:-7])
                if historical_intensity > 0:
                    intensity_trend = (recent_intensity - historical_intensity) / historical_intensity
                    intensity_trend_risk = max(0.0, min(intensity_trend, 0.3))  # Cap at 30% weight
            
            # Renewable energy risk (lower renewable % = higher risk)
            renewable_risk = max(0.0, (50 - renewable_percentage) / 50 * 0.3)  # Target 50% renewable
            
            risk_score = min(compliance_risk + intensity_trend_risk + renewable_risk, 1.0)
            
            return RiskFactor(
                name="Carbon Compliance",
                value=risk_score,
                weight=self.risk_weights['carbon_compliance'],
                severity=self.risk_thresholds.get_severity(risk_score),
                thresholds=self.risk_thresholds,
                description="Carbon footprint and emissions compliance risk analysis",
                unit="compliance_risk_score",
                metadata={
                    'total_energy_kwh': total_energy_kwh,
                    'period_days': period_days,
                    'emission_factor_kg_co2e_per_kwh': emission_factor,
                    'projected_annual_emissions_kg_co2e': projected_annual_emissions,
                    'adjusted_annual_emissions_kg_co2e': adjusted_emissions,
                    'annual_target_kg_co2e': annual_target,
                    'emissions_ratio': adjusted_emissions / annual_target if annual_target > 0 else 0,
                    'renewable_percentage': renewable_percentage,
                    'compliance_framework': emissions_data.compliance_framework if emissions_data else "default",
                    'carbon_intensity_records': len(carbon_intensity_records),
                    'compliance_risk': compliance_risk,
                    'intensity_trend_risk': intensity_trend_risk,
                    'renewable_risk': renewable_risk
                }
            )
            
        except Exception as e:
            logger.error(f"Error in carbon compliance analysis: {str(e)}")
            return RiskFactor(
                name="Carbon Compliance",
                value=0.0,
                weight=self.risk_weights['carbon_compliance'],
                severity=RiskSeverity.LOW,
                description="Error in carbon compliance analysis",
                metadata={'error': str(e)}
            )

    async def _analyze_grid_reliability(
        self,
        consumption_records: List[ElectricityConsumptionRecord],
        grid_reliability_data: Dict[str, Any]
    ) -> RiskFactor:
        """
        Analyze grid reliability and supply continuity risk.
        
        Args:
            consumption_records: Electricity consumption data
            grid_reliability_data: Grid reliability information
            
        Returns:
            RiskFactor: Grid reliability risk factor
        """
        try:
            # Extract grid reliability metrics
            outage_frequency = grid_reliability_data.get('annual_outage_frequency', 0.0)  # outages per year
            average_outage_duration = grid_reliability_data.get('average_outage_duration_hours', 0.0)
            voltage_stability_score = grid_reliability_data.get('voltage_stability_score', 1.0)  # 0-1, higher is better
            grid_congestion_level = grid_reliability_data.get('congestion_level', 0.0)  # 0-1, higher is worse
            renewable_integration_volatility = grid_reliability_data.get('renewable_volatility', 0.0)  # 0-1
            
            # Analyze consumption pattern consistency (more consistent = lower grid stress)
            demands = [record.demand_kw for record in consumption_records if record.demand_kw > 0]
            
            demand_variability = 0.0
            if len(demands) > 1:
                demand_std = statistics.stdev(demands)
                demand_mean = statistics.mean(demands)
                demand_variability = demand_std / demand_mean if demand_mean > 0 else 0.0
            
            # Peak demand timing analysis (consumption during peak grid hours)
            peak_hour_records = [
                r for r in consumption_records
                if 16 <= r.timestamp.hour <= 20  # Typical peak hours 4-8 PM
            ]
            peak_demand_ratio = len(peak_hour_records) / len(consumption_records) if consumption_records else 0.0
            
            # Risk scoring
            # Outage frequency risk (more outages = higher risk)
            outage_risk = min(outage_frequency / 5.0, 1.0)  # 5+ outages per year = high risk
            
            # Duration risk (longer outages = higher risk)
            duration_risk = min(average_outage_duration / 24.0, 1.0)  # 24+ hours = high risk
            
            # Voltage stability risk (lower stability = higher risk)
            stability_risk = max(0.0, 1.0 - voltage_stability_score)
            
            # Grid congestion risk
            congestion_risk = grid_congestion_level
            
            # Demand variability risk (high variability stresses grid)
            variability_risk = min(demand_variability, 0.5)
            
            # Peak timing risk (high demand during peak hours)
            peak_timing_risk = peak_demand_ratio * 0.3
            
            # Renewable volatility risk
            renewable_risk = renewable_integration_volatility * 0.2
            
            # Combined risk score
            risk_score = min(
                outage_risk * 0.25 +
                duration_risk * 0.20 +
                stability_risk * 0.20 +
                congestion_risk * 0.15 +
                variability_risk * 0.10 +
                peak_timing_risk * 0.05 +
                renewable_risk * 0.05,
                1.0
            )
            
            return RiskFactor(
                name="Grid Reliability",
                value=risk_score,
                weight=self.risk_weights['grid_reliability'],
                severity=self.risk_thresholds.get_severity(risk_score),
                thresholds=self.risk_thresholds,
                description="Grid reliability and supply continuity risk analysis",
                unit="reliability_risk_score",
                metadata={
                    'annual_outage_frequency': outage_frequency,
                    'average_outage_duration_hours': average_outage_duration,
                    'voltage_stability_score': voltage_stability_score,
                    'grid_congestion_level': grid_congestion_level,
                    'renewable_volatility': renewable_integration_volatility,
                    'demand_variability': demand_variability,
                    'peak_demand_ratio': peak_demand_ratio,
                    'outage_risk': outage_risk,
                    'duration_risk': duration_risk,
                    'stability_risk': stability_risk,
                    'congestion_risk': congestion_risk,
                    'peak_hour_records': len(peak_hour_records),
                    'total_records': len(consumption_records)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in grid reliability analysis: {str(e)}")
            return RiskFactor(
                name="Grid Reliability",
                value=0.0,
                weight=self.risk_weights['grid_reliability'],
                severity=RiskSeverity.LOW,
                description="Error in grid reliability analysis",
                metadata={'error': str(e)}
            )

    async def _generate_recommendations(
        self,
        assessment: RiskAssessment,
        contract_data: ElectricalContractData,
        consumption_records: List[ElectricityConsumptionRecord],
        emissions_data: Optional[CarbonEmissionsData]
    ) -> List[str]:
        """
        Generate specific recommendations based on risk assessment.
        
        Args:
            assessment: Risk assessment results
            contract_data: Contract information
            consumption_records: Consumption data
            emissions_data: Emissions data
            
        Returns:
            List[str]: Actionable recommendations
        """
        recommendations = []
        
        try:
            # Get high-risk factors
            high_risk_factors = assessment.get_high_risk_factors()
            
            for factor in high_risk_factors:
                if factor.name == "Demand Management":
                    if factor.metadata.get('margin_exceeded', False):
                        recommendations.append(
                            f"CRITICAL: Peak demand has exceeded the {self.demand_safety_margin*100}% safety margin. "
                            f"Current peak: {factor.metadata.get('current_peak_demand_kw', 0):.1f} kW, "
                            f"Safe threshold: {factor.metadata.get('safe_demand_threshold_kw', 0):.1f} kW. "
                            "Implement immediate load shedding protocols."
                        )
                    
                    utilization = factor.metadata.get('demand_utilization', 0)
                    if utilization > 0.8:
                        recommendations.append(
                            f"High demand utilization detected ({utilization*100:.1f}%). Consider load balancing, "
                            "demand response programs, or energy storage to manage peak demand."
                        )
                    
                    if factor.metadata.get('demand_coefficient_variation', 0) > 0.3:
                        recommendations.append(
                            "High demand variability detected. Implement demand forecasting and "
                            "automated load management to reduce peak demand charges."
                        )
                
                elif factor.name == "Energy Cost Trends":
                    cost_trend = factor.metadata.get('cost_trend_percentage', 0)
                    if cost_trend > 15:
                        recommendations.append(
                            f"Energy costs increasing by {cost_trend:.1f}%. Investigate time-of-use "
                            "optimization, energy efficiency upgrades, or alternative supply agreements."
                        )
                    
                    poor_pf_ratio = factor.metadata.get('poor_power_factor_ratio', 0)
                    if poor_pf_ratio > 0.1:
                        recommendations.append(
                            f"Poor power factor detected in {poor_pf_ratio*100:.1f}% of readings. "
                            "Install power factor correction equipment to reduce utility penalties."
                        )
                    
                    peak_ratio = factor.metadata.get('peak_consumption_ratio', 0)
                    if peak_ratio > 0.4:
                        recommendations.append(
                            f"High peak-period consumption ({peak_ratio*100:.1f}% of total). "
                            "Implement load shifting strategies to reduce peak-hour energy costs."
                        )
                
                elif factor.name == "Power Quality":
                    voltage_issues = factor.metadata.get('voltage_deviations_count', 0)
                    if voltage_issues > 0:
                        recommendations.append(
                            f"Voltage deviations detected in {voltage_issues} readings. "
                            "Install voltage regulation equipment to protect sensitive equipment."
                        )
                    
                    thd_issues = factor.metadata.get('high_thd_count', 0)
                    if thd_issues > 0:
                        recommendations.append(
                            f"High total harmonic distortion detected in {thd_issues} readings. "
                            "Install harmonic filters to improve power quality and equipment efficiency."
                        )
                    
                    unbalance_issues = factor.metadata.get('voltage_unbalance_count', 0)
                    if unbalance_issues > 0:
                        recommendations.append(
                            f"Voltage unbalance detected in {unbalance_issues} readings. "
                            "Balance loads across phases and check for equipment issues."
                        )
                
                elif factor.name == "Carbon Compliance":
                    emissions_ratio = factor.metadata.get('emissions_ratio', 0)
                    if emissions_ratio > 1.0:
                        recommendations.append(
                            f"Projected emissions exceed annual target by {(emissions_ratio-1)*100:.1f}%. "
                            "Implement energy efficiency measures and increase renewable energy procurement."
                        )
                    elif emissions_ratio > 0.8:
                        recommendations.append(
                            "Approaching annual emissions target. Accelerate energy conservation "
                            "and renewable energy initiatives to ensure compliance."
                        )
                    
                    renewable_pct = factor.metadata.get('renewable_percentage', 0)
                    if renewable_pct < 25:
                        recommendations.append(
                            f"Low renewable energy usage ({renewable_pct}%). Consider solar installation, "
                            "renewable energy certificates, or green power purchase agreements."
                        )
                
                elif factor.name == "Grid Reliability":
                    outages = factor.metadata.get('annual_outage_frequency', 0)
                    if outages > 3:
                        recommendations.append(
                            f"High outage frequency ({outages} per year). Consider backup power systems, "
                            "uninterruptible power supplies, or distributed energy resources."
                        )
                    
                    duration = factor.metadata.get('average_outage_duration_hours', 0)
                    if duration > 4:
                        recommendations.append(
                            f"Long outage durations (avg: {duration} hours). Implement energy storage "
                            "or backup generation to maintain critical operations during outages."
                        )
            
            # Overall risk-level recommendations
            if assessment.severity == RiskSeverity.CRITICAL:
                recommendations.insert(0, 
                    "EMERGENCY: Critical electricity consumption risk detected. "
                    "Activate emergency load reduction protocols and notify facility management immediately."
                )
            elif assessment.severity == RiskSeverity.HIGH:
                recommendations.insert(0,
                    "HIGH PRIORITY: Immediate action required to mitigate electricity consumption risks. "
                    "Implement emergency conservation measures and schedule urgent system assessments."
                )
            elif assessment.severity == RiskSeverity.MEDIUM:
                recommendations.insert(0,
                    "Monitor electricity consumption closely and implement preventive measures. "
                    "Schedule comprehensive energy audit within 30 days."
                )
            
            # Cost optimization recommendations
            total_cost = sum(
                (r.energy_kwh * (contract_data.energy_rate_peak if r.time_of_use_period == "peak" 
                                else contract_data.energy_rate_offpeak)) + 
                (r.demand_kw * contract_data.demand_rate_per_kw / 30)
                for r in consumption_records
            )
            
            if total_cost > 50000:  # High cost threshold
                recommendations.append(
                    f"High electricity costs detected (${total_cost:,.2f}). "
                    "Consider comprehensive energy management system implementation."
                )
            
            # Add general recommendations if no specific ones generated
            if not recommendations:
                recommendations.append("Continue monitoring electricity consumption patterns and maintain current efficiency practices.")
                recommendations.append("Consider implementing automated demand response systems for cost optimization.")
                recommendations.append("Evaluate renewable energy options to reduce costs and carbon footprint.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating specific recommendations. Please review risk factors manually."]

    def _validate_electricity_data(self, data: Dict[str, Any]) -> None:
        """Validate input data for electricity risk analysis."""
        self.validate_input_data(data)
        
        required_fields = ['consumption_records', 'contract_data', 'facility_id']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from input data")
        
        if not isinstance(data['consumption_records'], list):
            raise ValueError("consumption_records must be a list")
        
        if not data['consumption_records']:
            raise ValueError("At least one consumption record is required")

    def calculate_time_of_use_savings(
        self,
        consumption_records: List[ElectricityConsumptionRecord],
        contract_data: ElectricalContractData,
        load_shift_percentage: float = 0.20
    ) -> Dict[str, Any]:
        """
        Calculate potential savings from time-of-use optimization.
        
        Args:
            consumption_records: Electricity consumption data
            contract_data: Contract information
            load_shift_percentage: Percentage of peak load that could be shifted
            
        Returns:
            Dict containing savings analysis
        """
        try:
            peak_consumption = sum(
                r.energy_kwh for r in consumption_records 
                if r.time_of_use_period == "peak"
            )
            
            # Calculate current costs
            current_peak_cost = peak_consumption * contract_data.energy_rate_peak
            current_offpeak_cost = sum(
                r.energy_kwh * contract_data.energy_rate_offpeak 
                for r in consumption_records 
                if r.time_of_use_period != "peak"
            )
            
            # Calculate potential savings from load shifting
            shifted_load = peak_consumption * load_shift_percentage
            potential_peak_cost = (peak_consumption - shifted_load) * contract_data.energy_rate_peak
            additional_offpeak_cost = shifted_load * contract_data.energy_rate_offpeak
            
            total_current_cost = current_peak_cost + current_offpeak_cost
            total_optimized_cost = potential_peak_cost + current_offpeak_cost + additional_offpeak_cost
            potential_savings = total_current_cost - total_optimized_cost
            
            return {
                'current_total_cost': total_current_cost,
                'optimized_total_cost': total_optimized_cost,
                'potential_annual_savings': potential_savings * 12,  # Assuming monthly data
                'peak_consumption_kwh': peak_consumption,
                'shifted_load_kwh': shifted_load,
                'load_shift_percentage': load_shift_percentage,
                'savings_percentage': (potential_savings / total_current_cost * 100) if total_current_cost > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating time-of-use savings: {str(e)}")
            return {
                'error': str(e),
                'potential_annual_savings': 0,
                'savings_percentage': 0
            }

    def calculate_power_factor_optimization(
        self,
        consumption_records: List[ElectricityConsumptionRecord],
        contract_data: ElectricalContractData,
        target_power_factor: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate potential savings from power factor correction.
        
        Args:
            consumption_records: Electricity consumption data
            contract_data: Contract information
            target_power_factor: Target power factor for optimization
            
        Returns:
            Dict containing power factor analysis
        """
        try:
            pf_records = [r for r in consumption_records if r.power_factor is not None]
            
            if not pf_records:
                return {
                    'error': 'No power factor data available',
                    'potential_annual_savings': 0
                }
            
            current_avg_pf = statistics.mean([r.power_factor for r in pf_records])
            
            # Calculate current penalties
            total_energy = sum(r.energy_kwh for r in pf_records)
            poor_pf_records = [r for r in pf_records if r.power_factor < contract_data.power_factor_threshold]
            
            current_penalties = 0
            for record in poor_pf_records:
                penalty_factor = (contract_data.power_factor_threshold - record.power_factor) / contract_data.power_factor_threshold
                energy_cost = record.energy_kwh * (
                    contract_data.energy_rate_peak if record.time_of_use_period == "peak" 
                    else contract_data.energy_rate_offpeak
                )
                current_penalties += energy_cost * contract_data.power_factor_penalty_rate * penalty_factor
            
            # Calculate potential savings with target power factor
            if current_avg_pf >= target_power_factor:
                potential_savings = 0
            else:
                # Assume all penalties eliminated with target power factor
                potential_savings = current_penalties
            
            return {
                'current_avg_power_factor': current_avg_pf,
                'target_power_factor': target_power_factor,
                'total_energy_kwh': total_energy,
                'poor_pf_records_count': len(poor_pf_records),
                'current_monthly_penalties': current_penalties,
                'potential_annual_savings': potential_savings * 12,
                'improvement_needed': max(0, target_power_factor - current_avg_pf),
                'records_analyzed': len(pf_records)
            }
            
        except Exception as e:
            logger.error(f"Error calculating power factor optimization: {str(e)}")
            return {
                'error': str(e),
                'potential_annual_savings': 0
            }

# Create aliases for backward compatibility
ElectricityRiskAnalyzer = ElectricityConsumptionRiskAnalyzer
