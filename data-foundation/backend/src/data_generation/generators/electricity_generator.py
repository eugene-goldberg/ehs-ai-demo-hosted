"""
Electricity Consumption Data Generator Module

This module provides comprehensive electricity consumption pattern generation for EHS AI Demo system.
It generates realistic electricity usage patterns including peak/off-peak usage, facility-specific
consumption profiles, weather correlations, cost calculations, and CO2 emissions.

Features:
- Facility-specific consumption profiles based on operational characteristics
- Peak/off-peak time-of-use patterns with realistic pricing
- Weather correlation effects (heating/cooling demand)
- Seasonal variations and operational patterns
- Real-time cost calculations with time-of-use rates
- CO2 emissions calculations based on grid mix
- Multiple facility support with different operational profiles
- Equipment load simulation and demand response capabilities

Author: EHS AI Demo Team  
Created: 2025-08-28
Version: 1.0.0
"""

import math
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .base_generator import BaseGenerator, GeneratorConfig
from ..utils.data_utils import (
    FacilityType, 
    get_facility_profile, 
    generate_weather_data,
    apply_weather_correlation,
    calculate_co2_emissions,
    calculate_environmental_costs,
    CO2_EMISSION_FACTORS,
    EHS_CONSTANTS
)


@dataclass
class ElectricityGeneratorConfig(GeneratorConfig):
    """Configuration for electricity consumption generation"""
    
    # Time-of-use rate structure (USD per kWh)
    peak_rate: float = 0.15          # Peak hours rate
    off_peak_rate: float = 0.08      # Off-peak hours rate  
    super_off_peak_rate: float = 0.05 # Super off-peak hours rate
    
    # Peak hour definitions (24-hour format)
    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    super_off_peak_hours: List[int] = field(default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6])
    
    # Load profile parameters
    base_load_factor: float = 0.3     # Minimum load as fraction of peak
    peak_load_factor: float = 1.0     # Peak load multiplier
    load_volatility: float = 0.15     # Standard deviation of load variations
    
    # Equipment simulation parameters
    hvac_load_percentage: float = 0.4  # HVAC as percentage of total load
    lighting_load_percentage: float = 0.2
    process_load_percentage: float = 0.3
    other_load_percentage: float = 0.1
    
    # Demand response parameters
    enable_demand_response: bool = True
    dr_reduction_factor: float = 0.15  # Peak reduction during DR events
    dr_probability: float = 0.05       # Daily probability of DR event
    
    # Power factor and quality parameters
    power_factor: float = 0.85         # Typical industrial power factor
    enable_power_quality_events: bool = True
    voltage_sag_probability: float = 0.02
    harmonic_distortion_level: float = 0.05
    
    # Grid carbon intensity (kg CO2/kWh) - varies by time and season
    base_grid_carbon_intensity: float = 0.4
    renewable_mix_variation: float = 0.2  # Variation in renewable generation


class ElectricityGenerator(BaseGenerator):
    """
    Comprehensive electricity consumption data generator.
    
    Generates realistic electricity consumption patterns for industrial facilities
    including time-of-use patterns, weather correlations, costs, and emissions.
    """
    
    def __init__(self, config: Optional[ElectricityGeneratorConfig] = None):
        """
        Initialize the electricity generator.
        
        Args:
            config: Configuration object for electricity generation
        """
        self.elec_config = config or ElectricityGeneratorConfig()
        super().__init__(self.elec_config)
        
        # Initialize equipment models
        self._equipment_loads = {}
        self._demand_response_events = []
        self._power_quality_events = []
        
        self.logger = logging.getLogger(__name__)
    
    def generate(self, 
                 facility_type: FacilityType = FacilityType.MANUFACTURING,
                 facility_count: int = 1,
                 include_weather_data: bool = True,
                 frequency: str = "H") -> Dict[str, Any]:
        """
        Generate electricity consumption data for multiple facilities.
        
        Args:
            facility_type: Type of facility to generate data for
            facility_count: Number of facilities to generate
            include_weather_data: Whether to include weather correlations
            frequency: Data frequency ('H' for hourly, 'D' for daily)
            
        Returns:
            Dictionary containing generated electricity consumption data
        """
        if not self.validate_date_range():
            raise ValueError("Invalid date range configuration")
        
        self.logger.info(f"Generating electricity data for {facility_count} {facility_type.value} facilities")
        
        # Get time index based on frequency
        dates = self.get_time_index(frequency)
        
        # Initialize results structure
        results = {
            "metadata": self._get_electricity_metadata(facility_type, facility_count, frequency),
            "facilities": [],
            "summary": {},
            "time_series": {
                "dates": [d.isoformat() for d in dates],
                "total_consumption_kwh": [],
                "total_cost_usd": [],
                "total_co2_emissions_kg": [],
                "average_grid_carbon_intensity": []
            }
        }
        
        # Generate data for each facility
        facility_data_list = []
        for facility_id in range(1, facility_count + 1):
            facility_data = self._generate_facility_electricity_data(
                facility_id, facility_type, dates, include_weather_data, frequency
            )
            facility_data_list.append(facility_data)
            results["facilities"].append(facility_data)
        
        # Calculate aggregate time series
        results["time_series"] = self._calculate_aggregate_time_series(
            facility_data_list, dates
        )
        
        # Generate summary statistics
        results["summary"] = self._generate_summary_statistics(
            facility_data_list, facility_type
        )
        
        return results
    
    def _generate_facility_electricity_data(self,
                                          facility_id: int,
                                          facility_type: FacilityType,
                                          dates: List[datetime],
                                          include_weather: bool,
                                          frequency: str) -> Dict[str, Any]:
        """Generate electricity data for a single facility."""
        
        # Get facility profile
        facility_profile = get_facility_profile(facility_type)
        base_consumption = facility_profile["base_energy_consumption"]  # kWh/day
        
        # Adjust for hourly vs daily frequency
        if frequency == "H":
            base_consumption = base_consumption / 24  # kWh/hour
        
        facility_data = {
            "facility_id": facility_id,
            "facility_type": facility_type.value,
            "facility_profile": facility_profile,
            "time_series": {
                "timestamps": [d.isoformat() for d in dates],
                "consumption_kwh": [],
                "demand_kw": [],
                "cost_breakdown": [],
                "rate_type": [],
                "power_factor": [],
                "co2_emissions_kg": [],
                "equipment_loads": [],
                "weather_data": [] if include_weather else None
            },
            "daily_patterns": {},
            "monthly_statistics": {},
            "power_quality_events": []
        }
        
        # Generate consumption for each timestamp
        for i, date in enumerate(dates):
            
            # Generate weather data if requested
            weather_data = generate_weather_data(date) if include_weather else None
            if include_weather:
                facility_data["time_series"]["weather_data"].append(weather_data)
            
            # Calculate base consumption with patterns
            consumption = self._calculate_consumption_with_patterns(
                base_consumption, date, facility_profile, weather_data, frequency
            )
            
            # Apply equipment-specific loads
            equipment_loads = self._calculate_equipment_loads(consumption, date, facility_profile)
            
            # Calculate power demand (kW)
            demand = consumption if frequency == "H" else consumption / 24
            
            # Determine rate type and calculate costs
            rate_type, unit_rate = self._get_rate_type_and_price(date)
            
            # Apply demand response if enabled
            if self.elec_config.enable_demand_response:
                consumption, demand = self._apply_demand_response(
                    consumption, demand, date, rate_type
                )
            
            # Calculate costs
            costs = self._calculate_electricity_costs(consumption, demand, unit_rate, date)
            
            # Calculate CO2 emissions
            grid_carbon_intensity = self._get_grid_carbon_intensity(date)
            co2_emissions = consumption * grid_carbon_intensity
            
            # Power factor with some variation
            power_factor = self.elec_config.power_factor + np.random.normal(0, 0.05)
            power_factor = max(0.7, min(0.99, power_factor))
            
            # Store data point
            facility_data["time_series"]["consumption_kwh"].append(round(consumption, 2))
            facility_data["time_series"]["demand_kw"].append(round(demand, 2))
            facility_data["time_series"]["cost_breakdown"].append(costs)
            facility_data["time_series"]["rate_type"].append(rate_type)
            facility_data["time_series"]["power_factor"].append(round(power_factor, 3))
            facility_data["time_series"]["co2_emissions_kg"].append(round(co2_emissions, 2))
            facility_data["time_series"]["equipment_loads"].append(equipment_loads)
            
            # Generate power quality events
            if self.elec_config.enable_power_quality_events:
                pq_event = self._generate_power_quality_event(date)
                if pq_event:
                    facility_data["power_quality_events"].append(pq_event)
        
        # Calculate daily and monthly patterns
        facility_data["daily_patterns"] = self._analyze_daily_patterns(
            facility_data["time_series"], frequency
        )
        facility_data["monthly_statistics"] = self._calculate_monthly_statistics(
            facility_data["time_series"], dates
        )
        
        return facility_data
    
    def _calculate_consumption_with_patterns(self,
                                           base_consumption: float,
                                           date: datetime,
                                           facility_profile: Dict,
                                           weather_data: Optional[Dict],
                                           frequency: str) -> float:
        """Calculate consumption applying all patterns and correlations."""
        
        # Start with base consumption
        consumption = base_consumption
        
        # Apply facility-specific patterns
        consumption = self._apply_facility_operational_pattern(
            consumption, date, facility_profile, frequency
        )
        
        # Apply weather correlation if available
        if weather_data:
            weather_sensitivity = facility_profile.get("weather_sensitivity", 0.1)
            consumption = apply_weather_correlation(
                consumption, weather_data, weather_sensitivity
            )
        
        # Apply time-of-use patterns
        consumption = self._apply_time_of_use_pattern(consumption, date)
        
        # Apply seasonal variations
        seasonal_variation = facility_profile.get("seasonal_variation", 0.15)
        if seasonal_variation > 0:
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 1 + seasonal_variation * math.sin(
                2 * math.pi * (day_of_year - 81) / 365
            )
            consumption *= seasonal_factor
        
        # Add operational variance
        variance_factor = self.elec_config.load_volatility
        noise = np.random.normal(1.0, variance_factor)
        consumption *= max(0.1, noise)  # Ensure positive consumption
        
        return max(0, consumption)
    
    def _apply_facility_operational_pattern(self,
                                          consumption: float,
                                          date: datetime,
                                          facility_profile: Dict,
                                          frequency: str) -> float:
        """Apply facility-specific operational patterns."""
        
        operational_hours = facility_profile.get("operational_hours", 8)
        
        if frequency == "H" and hasattr(date, 'hour'):
            hour = date.hour
            
            if operational_hours == 24:
                # 24/7 operations (chemical, refinery, power plant)
                # Lower consumption at night, higher during day shift
                if 6 <= hour <= 18:
                    multiplier = 1.0
                elif 18 <= hour <= 22:
                    multiplier = 0.85
                else:
                    multiplier = 0.7
            else:
                # Standard business hours operations
                if 7 <= hour <= 17:
                    # Business hours - full operation
                    multiplier = 1.0
                elif 18 <= hour <= 22:
                    # Evening - reduced operations
                    multiplier = 0.4
                else:
                    # Night - minimal operations
                    multiplier = self.elec_config.base_load_factor
                    
            consumption *= multiplier
        
        # Weekend adjustments for non-24/7 operations
        if date.weekday() >= 5 and operational_hours < 24:
            consumption *= 0.3  # Significantly reduced weekend operations
        
        return consumption
    
    def _apply_time_of_use_pattern(self, consumption: float, date: datetime) -> float:
        """Apply time-of-use consumption patterns."""
        
        if not hasattr(date, 'hour'):
            return consumption
        
        hour = date.hour
        
        # Peak hours typically have higher baseline consumption due to 
        # increased economic activity
        if hour in self.elec_config.peak_hours:
            multiplier = 1.1  # Slight increase during peak hours
        elif hour in self.elec_config.super_off_peak_hours:
            multiplier = 0.9  # Slight decrease during super off-peak
        else:
            multiplier = 1.0  # Off-peak baseline
        
        return consumption * multiplier
    
    def _calculate_equipment_loads(self,
                                 total_consumption: float,
                                 date: datetime,
                                 facility_profile: Dict) -> Dict[str, float]:
        """Break down consumption into equipment categories."""
        
        # Base percentage allocations
        hvac_pct = self.elec_config.hvac_load_percentage
        lighting_pct = self.elec_config.lighting_load_percentage
        process_pct = self.elec_config.process_load_percentage
        other_pct = self.elec_config.other_load_percentage
        
        # Adjust HVAC load based on weather (if available through patterns)
        if hasattr(date, 'hour'):
            hour = date.hour
            # HVAC typically higher during daytime hours
            if 8 <= hour <= 20:
                hvac_multiplier = 1.2
            else:
                hvac_multiplier = 0.8
        else:
            hvac_multiplier = 1.0
        
        # Lighting load varies by time of day
        if hasattr(date, 'hour'):
            hour = date.hour
            if 6 <= hour <= 18:
                lighting_multiplier = 1.0  # Daytime
            elif 18 <= hour <= 22:
                lighting_multiplier = 1.3  # Evening peak
            else:
                lighting_multiplier = 0.3  # Night
        else:
            lighting_multiplier = 1.0
        
        # Calculate actual loads
        hvac_load = total_consumption * hvac_pct * hvac_multiplier
        lighting_load = total_consumption * lighting_pct * lighting_multiplier
        process_load = total_consumption * process_pct
        other_load = total_consumption * other_pct
        
        # Normalize to ensure total adds up
        total_calculated = hvac_load + lighting_load + process_load + other_load
        normalization_factor = total_consumption / total_calculated if total_calculated > 0 else 1
        
        return {
            "hvac_kwh": round(hvac_load * normalization_factor, 2),
            "lighting_kwh": round(lighting_load * normalization_factor, 2),
            "process_kwh": round(process_load * normalization_factor, 2),
            "other_kwh": round(other_load * normalization_factor, 2),
            "total_kwh": round(total_consumption, 2)
        }
    
    def _get_rate_type_and_price(self, date: datetime) -> Tuple[str, float]:
        """Determine the electricity rate type and price for given time."""
        
        if not hasattr(date, 'hour'):
            # For daily data, return average rate
            return "daily_average", self.elec_config.off_peak_rate
        
        hour = date.hour
        
        if hour in self.elec_config.peak_hours:
            return "peak", self.elec_config.peak_rate
        elif hour in self.elec_config.super_off_peak_hours:
            return "super_off_peak", self.elec_config.super_off_peak_rate
        else:
            return "off_peak", self.elec_config.off_peak_rate
    
    def _apply_demand_response(self,
                             consumption: float,
                             demand: float,
                             date: datetime,
                             rate_type: str) -> Tuple[float, float]:
        """Apply demand response reduction during peak periods."""
        
        # Demand response typically occurs during peak hours
        if (rate_type == "peak" and 
            np.random.random() < self.elec_config.dr_probability):
            
            reduction_factor = 1 - self.elec_config.dr_reduction_factor
            consumption *= reduction_factor
            demand *= reduction_factor
            
            # Log demand response event
            self._demand_response_events.append({
                "timestamp": date.isoformat(),
                "reduction_factor": self.elec_config.dr_reduction_factor,
                "estimated_savings_kwh": consumption * self.elec_config.dr_reduction_factor
            })
        
        return consumption, demand
    
    def _calculate_electricity_costs(self,
                                   consumption: float,
                                   demand: float,
                                   unit_rate: float,
                                   date: datetime) -> Dict[str, float]:
        """Calculate detailed electricity costs."""
        
        # Energy charge (consumption * rate)
        energy_charge = consumption * unit_rate
        
        # Demand charge (typically applied to peak demand)
        demand_rate = 15.0  # USD per kW of peak demand
        demand_charge = demand * demand_rate / 30  # Amortized daily
        
        # Grid service charges (fixed daily charge)
        grid_service_charge = 2.50  # USD per day
        
        # Power factor penalty (if below threshold)
        power_factor = self.elec_config.power_factor
        pf_penalty = 0.0
        if power_factor < 0.85:
            pf_penalty = energy_charge * 0.05  # 5% penalty
        
        # Total cost
        total_cost = energy_charge + demand_charge + grid_service_charge + pf_penalty
        
        return {
            "energy_charge": round(energy_charge, 2),
            "demand_charge": round(demand_charge, 2),
            "grid_service_charge": round(grid_service_charge, 2),
            "power_factor_penalty": round(pf_penalty, 2),
            "total_cost": round(total_cost, 2),
            "unit_rate": round(unit_rate, 4)
        }
    
    def _get_grid_carbon_intensity(self, date: datetime) -> float:
        """Calculate grid carbon intensity with time and seasonal variations."""
        
        base_intensity = self.elec_config.base_grid_carbon_intensity
        
        # Seasonal variation (higher in winter due to more coal/gas)
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.2 * math.sin(2 * math.pi * (day_of_year - 81) / 365 + math.pi)
        
        # Daily variation (lower during day due to solar)
        if hasattr(date, 'hour'):
            hour = date.hour
            if 10 <= hour <= 16:
                daily_factor = 0.85  # Lower during solar peak
            elif 18 <= hour <= 22:
                daily_factor = 1.15  # Higher during evening peak
            else:
                daily_factor = 1.0
        else:
            daily_factor = 1.0
        
        # Random variation in renewable generation
        renewable_factor = 1 + np.random.normal(0, self.elec_config.renewable_mix_variation)
        renewable_factor = max(0.5, min(1.5, renewable_factor))
        
        return base_intensity * seasonal_factor * daily_factor * renewable_factor
    
    def _generate_power_quality_event(self, date: datetime) -> Optional[Dict[str, Any]]:
        """Generate random power quality events."""
        
        if np.random.random() > self.elec_config.voltage_sag_probability:
            return None
        
        event_types = ["voltage_sag", "voltage_swell", "harmonic_distortion", "transient"]
        event_type = np.random.choice(event_types)
        
        if event_type == "voltage_sag":
            magnitude = np.random.uniform(0.1, 0.3)  # 10-30% sag
            duration = np.random.uniform(0.1, 2.0)   # 0.1-2 seconds
        elif event_type == "voltage_swell":
            magnitude = np.random.uniform(0.1, 0.2)  # 10-20% swell
            duration = np.random.uniform(0.1, 1.0)
        elif event_type == "harmonic_distortion":
            magnitude = np.random.uniform(0.03, 0.08)  # 3-8% THD
            duration = np.random.uniform(60, 3600)     # 1 minute to 1 hour
        else:  # transient
            magnitude = np.random.uniform(1.0, 3.0)   # 1-3x normal voltage
            duration = np.random.uniform(0.001, 0.1)  # 1ms to 100ms
        
        return {
            "timestamp": date.isoformat(),
            "event_type": event_type,
            "magnitude": round(magnitude, 3),
            "duration_seconds": round(duration, 3),
            "affected_equipment": np.random.choice([
                "HVAC System", "Process Equipment", "Lighting", "IT Equipment"
            ])
        }
    
    def _analyze_daily_patterns(self, time_series: Dict, frequency: str) -> Dict[str, Any]:
        """Analyze and extract daily consumption patterns."""
        
        if frequency != "H":
            return {"message": "Daily patterns only available for hourly data"}
        
        # Group data by hour of day
        hourly_consumption = {}
        hourly_costs = {}
        
        for i, timestamp_str in enumerate(time_series["timestamps"]):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            hour = timestamp.hour
            
            if hour not in hourly_consumption:
                hourly_consumption[hour] = []
                hourly_costs[hour] = []
            
            hourly_consumption[hour].append(time_series["consumption_kwh"][i])
            hourly_costs[hour].append(time_series["cost_breakdown"][i]["total_cost"])
        
        # Calculate statistics for each hour
        pattern_analysis = {}
        for hour in range(24):
            if hour in hourly_consumption:
                consumptions = hourly_consumption[hour]
                costs = hourly_costs[hour]
                
                pattern_analysis[hour] = {
                    "average_consumption_kwh": round(np.mean(consumptions), 2),
                    "peak_consumption_kwh": round(np.max(consumptions), 2),
                    "min_consumption_kwh": round(np.min(consumptions), 2),
                    "average_cost": round(np.mean(costs), 2),
                    "consumption_variability": round(np.std(consumptions), 2)
                }
        
        return {
            "hourly_patterns": pattern_analysis,
            "peak_hour": max(pattern_analysis.keys(), 
                           key=lambda h: pattern_analysis[h]["average_consumption_kwh"]),
            "minimum_hour": min(pattern_analysis.keys(), 
                              key=lambda h: pattern_analysis[h]["average_consumption_kwh"])
        }
    
    def _calculate_monthly_statistics(self, 
                                    time_series: Dict, 
                                    dates: List[datetime]) -> Dict[str, Any]:
        """Calculate monthly electricity statistics."""
        
        monthly_data = {}
        
        for i, date in enumerate(dates):
            month_key = f"{date.year}-{date.month:02d}"
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    "consumption_kwh": [],
                    "costs": [],
                    "co2_emissions": []
                }
            
            monthly_data[month_key]["consumption_kwh"].append(
                time_series["consumption_kwh"][i]
            )
            monthly_data[month_key]["costs"].append(
                time_series["cost_breakdown"][i]["total_cost"]
            )
            monthly_data[month_key]["co2_emissions"].append(
                time_series["co2_emissions_kg"][i]
            )
        
        # Calculate statistics for each month
        monthly_stats = {}
        for month, data in monthly_data.items():
            monthly_stats[month] = {
                "total_consumption_kwh": round(sum(data["consumption_kwh"]), 2),
                "total_cost_usd": round(sum(data["costs"]), 2),
                "total_co2_kg": round(sum(data["co2_emissions"]), 2),
                "average_daily_consumption": round(
                    np.mean(data["consumption_kwh"]), 2
                ),
                "peak_consumption_kwh": round(max(data["consumption_kwh"]), 2),
                "average_unit_cost": round(
                    sum(data["costs"]) / sum(data["consumption_kwh"]), 4
                ) if sum(data["consumption_kwh"]) > 0 else 0
            }
        
        return monthly_stats
    
    def _calculate_aggregate_time_series(self,
                                       facility_data_list: List[Dict],
                                       dates: List[datetime]) -> Dict[str, List]:
        """Calculate aggregated time series across all facilities."""
        
        n_points = len(dates)
        total_consumption = [0.0] * n_points
        total_cost = [0.0] * n_points
        total_co2 = [0.0] * n_points
        grid_intensities = []
        
        for i in range(n_points):
            point_consumption = 0
            point_cost = 0
            point_co2 = 0
            
            for facility_data in facility_data_list:
                point_consumption += facility_data["time_series"]["consumption_kwh"][i]
                point_cost += facility_data["time_series"]["cost_breakdown"][i]["total_cost"]
                point_co2 += facility_data["time_series"]["co2_emissions_kg"][i]
            
            total_consumption[i] = round(point_consumption, 2)
            total_cost[i] = round(point_cost, 2)
            total_co2[i] = round(point_co2, 2)
            
            # Calculate weighted average grid carbon intensity
            if point_consumption > 0:
                grid_intensities.append(round(point_co2 / point_consumption, 4))
            else:
                grid_intensities.append(0.0)
        
        return {
            "dates": [d.isoformat() for d in dates],
            "total_consumption_kwh": total_consumption,
            "total_cost_usd": total_cost,
            "total_co2_emissions_kg": total_co2,
            "average_grid_carbon_intensity": grid_intensities
        }
    
    def _generate_summary_statistics(self,
                                   facility_data_list: List[Dict],
                                   facility_type: FacilityType) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        
        # Aggregate all consumption and cost data
        all_consumption = []
        all_costs = []
        all_co2 = []
        all_power_factors = []
        
        for facility_data in facility_data_list:
            all_consumption.extend(facility_data["time_series"]["consumption_kwh"])
            all_costs.extend([c["total_cost"] for c in facility_data["time_series"]["cost_breakdown"]])
            all_co2.extend(facility_data["time_series"]["co2_emissions_kg"])
            all_power_factors.extend(facility_data["time_series"]["power_factor"])
        
        # Calculate statistics
        total_consumption = sum(all_consumption)
        total_cost = sum(all_costs)
        total_co2 = sum(all_co2)
        
        average_unit_cost = total_cost / total_consumption if total_consumption > 0 else 0
        
        # Power quality statistics
        power_quality_events = []
        for facility_data in facility_data_list:
            power_quality_events.extend(facility_data["power_quality_events"])
        
        return {
            "total_consumption_kwh": round(total_consumption, 2),
            "total_cost_usd": round(total_cost, 2),
            "total_co2_emissions_kg": round(total_co2, 2),
            "total_co2_emissions_tonnes": round(total_co2 / 1000, 4),
            "average_unit_cost_per_kwh": round(average_unit_cost, 4),
            "average_power_factor": round(np.mean(all_power_factors), 3),
            "consumption_statistics": {
                "mean_kwh": round(np.mean(all_consumption), 2),
                "median_kwh": round(np.median(all_consumption), 2),
                "std_dev_kwh": round(np.std(all_consumption), 2),
                "min_kwh": round(np.min(all_consumption), 2),
                "max_kwh": round(np.max(all_consumption), 2),
                "peak_to_average_ratio": round(np.max(all_consumption) / np.mean(all_consumption), 2)
            },
            "cost_breakdown": {
                "average_energy_charge_percentage": 75.0,
                "average_demand_charge_percentage": 15.0,
                "average_service_charge_percentage": 8.0,
                "average_penalties_percentage": 2.0
            },
            "power_quality": {
                "total_events": len(power_quality_events),
                "event_rate_per_day": round(len(power_quality_events) / max(1, len(self.get_date_range())), 3),
                "event_types": {
                    event_type: sum(1 for e in power_quality_events if e["event_type"] == event_type)
                    for event_type in ["voltage_sag", "voltage_swell", "harmonic_distortion", "transient"]
                }
            },
            "demand_response": {
                "total_events": len(self._demand_response_events),
                "total_savings_kwh": round(sum(e.get("estimated_savings_kwh", 0) for e in self._demand_response_events), 2)
            },
            "environmental_impact": {
                "carbon_intensity_kg_per_kwh": round(total_co2 / total_consumption, 4) if total_consumption > 0 else 0,
                "equivalent_co2_trees_required": round(total_co2 / 21.8, 0),  # kg CO2 absorbed per tree per year
                "equivalent_cars_off_road_days": round(total_co2 / 4.6, 1)  # kg CO2 per car per day
            }
        }
    
    def _get_electricity_metadata(self, 
                                facility_type: FacilityType,
                                facility_count: int,
                                frequency: str) -> Dict[str, Any]:
        """Get metadata for electricity generation."""
        
        base_metadata = self.get_generation_metadata()
        
        electricity_metadata = {
            "generator_type": "electricity_consumption",
            "facility_type": facility_type.value,
            "facility_count": facility_count,
            "data_frequency": frequency,
            "rate_structure": {
                "peak_rate_usd_per_kwh": self.elec_config.peak_rate,
                "off_peak_rate_usd_per_kwh": self.elec_config.off_peak_rate,
                "super_off_peak_rate_usd_per_kwh": self.elec_config.super_off_peak_rate,
                "peak_hours": self.elec_config.peak_hours,
                "super_off_peak_hours": self.elec_config.super_off_peak_hours
            },
            "load_profile_config": {
                "base_load_factor": self.elec_config.base_load_factor,
                "load_volatility": self.elec_config.load_volatility,
                "hvac_percentage": self.elec_config.hvac_load_percentage,
                "lighting_percentage": self.elec_config.lighting_load_percentage,
                "process_percentage": self.elec_config.process_load_percentage
            },
            "demand_response_enabled": self.elec_config.enable_demand_response,
            "power_quality_monitoring": self.elec_config.enable_power_quality_events,
            "carbon_accounting": {
                "base_grid_intensity_kg_per_kwh": self.elec_config.base_grid_carbon_intensity,
                "renewable_mix_variation": self.elec_config.renewable_mix_variation
            }
        }
        
        return {**base_metadata, **electricity_metadata}