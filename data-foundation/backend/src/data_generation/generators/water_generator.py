"""
Water Usage Data Generator Module

This module provides comprehensive water consumption pattern generation for EHS AI Demo system.
It generates realistic water usage patterns including different water types (potable, process, cooling),
facility-specific profiles, weather correlations, cost calculations with tiered pricing,
recycling/reuse tracking, and discharge quality metrics.

Features:
- Multiple water types: potable, process, cooling, fire suppression
- Facility-specific consumption profiles based on operational characteristics
- Weather correlation effects (evaporation rates, cooling tower demand, irrigation)
- Seasonal variations and operational patterns
- Tiered pricing structure with volume-based rates
- Water recycling and reuse tracking with efficiency metrics
- Discharge monitoring with quality parameters
- Water treatment costs and chemical usage
- Conservation measures and leak detection simulation
- Regulatory compliance tracking (discharge permits, quality standards)

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
from enum import Enum

from .base_generator import BaseGenerator, GeneratorConfig
from ..utils.data_utils import (
    FacilityType, 
    get_facility_profile, 
    generate_weather_data,
    apply_weather_correlation,
    calculate_environmental_costs,
    EHS_CONSTANTS
)


class WaterType(Enum):
    """Types of water used in industrial facilities."""
    POTABLE = "potable"              # Drinking water
    PROCESS = "process"              # Manufacturing processes
    COOLING = "cooling"              # Cooling towers, HVAC
    FIRE_SUPPRESSION = "fire_suppression"  # Fire safety systems
    IRRIGATION = "irrigation"        # Landscaping and grounds
    STEAM_GENERATION = "steam_generation"  # Boiler feedwater


class WaterQualityParameter(Enum):
    """Water quality parameters for monitoring."""
    PH = "ph"
    TURBIDITY = "turbidity"
    DISSOLVED_OXYGEN = "dissolved_oxygen"
    BOD = "bod"  # Biochemical Oxygen Demand
    COD = "cod"  # Chemical Oxygen Demand
    TSS = "tss"  # Total Suspended Solids
    TDS = "tds"  # Total Dissolved Solids
    TEMPERATURE = "temperature"
    CONDUCTIVITY = "conductivity"
    CHLORINE = "chlorine"


@dataclass
class WaterGeneratorConfig(GeneratorConfig):
    """Configuration for water usage generation"""
    
    # Water type distribution (percentage of total usage)
    water_type_distribution: Dict[WaterType, float] = field(default_factory=lambda: {
        WaterType.PROCESS: 0.45,
        WaterType.COOLING: 0.30,
        WaterType.POTABLE: 0.15,
        WaterType.STEAM_GENERATION: 0.07,
        WaterType.IRRIGATION: 0.02,
        WaterType.FIRE_SUPPRESSION: 0.01
    })
    
    # Tiered pricing structure (USD per 1000 gallons)
    tier_1_rate: float = 3.50          # 0-10k gallons
    tier_2_rate: float = 4.75          # 10k-50k gallons  
    tier_3_rate: float = 6.25          # 50k-200k gallons
    tier_4_rate: float = 8.50          # >200k gallons
    tier_1_threshold: float = 10000    # gallons
    tier_2_threshold: float = 50000
    tier_3_threshold: float = 200000
    
    # Usage pattern parameters
    base_usage_factor: float = 0.7     # Minimum usage as fraction of peak
    peak_usage_factor: float = 1.0     # Peak usage multiplier
    usage_volatility: float = 0.12     # Standard deviation of usage variations
    
    # Weather correlation factors
    temperature_correlation: float = 0.3   # Cooling water correlation with temperature
    humidity_correlation: float = -0.2     # Inverse correlation with humidity
    evaporation_factor: float = 0.15       # Evaporation loss factor
    
    # Recycling and reuse parameters
    enable_recycling: bool = True
    base_recycle_rate: float = 0.25        # 25% of water recycled
    recycle_efficiency: float = 0.85       # Recycling system efficiency
    recycle_cost_per_gallon: float = 0.002 # USD per gallon to recycle
    
    # Treatment and chemical costs
    treatment_chemical_cost: float = 0.15  # USD per 1000 gallons
    discharge_treatment_cost: float = 0.25 # USD per 1000 gallons discharged
    
    # Leak and conservation parameters
    enable_leak_simulation: bool = True
    leak_probability: float = 0.02         # Daily probability of leak
    leak_duration_hours: Tuple[int, int] = (4, 72)  # Min/max leak duration
    leak_rate_multiplier: float = 1.5      # Usage multiplier during leaks
    
    # Conservation measures
    enable_conservation: bool = True
    conservation_efficiency: float = 0.1   # 10% reduction from conservation
    conservation_cost: float = 500         # USD per month for conservation programs
    
    # Discharge monitoring
    enable_discharge_monitoring: bool = True
    discharge_rate: float = 0.8             # 80% of intake becomes discharge
    
    # Quality parameters ranges (for simulation)
    quality_ranges: Dict[WaterQualityParameter, Tuple[float, float]] = field(default_factory=lambda: {
        WaterQualityParameter.PH: (6.5, 8.5),
        WaterQualityParameter.TURBIDITY: (0.1, 4.0),  # NTU
        WaterQualityParameter.DISSOLVED_OXYGEN: (5.0, 14.0),  # mg/L
        WaterQualityParameter.BOD: (1.0, 30.0),  # mg/L
        WaterQualityParameter.COD: (10.0, 150.0),  # mg/L
        WaterQualityParameter.TSS: (5.0, 100.0),  # mg/L
        WaterQualityParameter.TDS: (50.0, 1500.0),  # mg/L
        WaterQualityParameter.TEMPERATURE: (10.0, 35.0),  # Celsius
        WaterQualityParameter.CONDUCTIVITY: (100.0, 2000.0),  # μS/cm
        WaterQualityParameter.CHLORINE: (0.2, 4.0)  # mg/L
    })


class WaterGenerator(BaseGenerator):
    """
    Comprehensive water usage data generator.
    
    Generates realistic water consumption patterns for industrial facilities
    including multiple water types, weather correlations, costs, recycling,
    and discharge monitoring.
    """
    
    def __init__(self, config: Optional[WaterGeneratorConfig] = None):
        """
        Initialize the water generator.
        
        Args:
            config: Configuration object for water generation
        """
        self.water_config = config or WaterGeneratorConfig()
        super().__init__(self.water_config)
        
        # Validate water type distribution
        total_distribution = sum(self.water_config.water_type_distribution.values())
        if abs(total_distribution - 1.0) > 0.01:
            self.logger.warning(f"Water type distribution sums to {total_distribution}, not 1.0")
        
        # Initialize internal state
        self._active_leaks: List[Dict] = []
        self._conservation_active = False
        
    def generate(
        self,
        facility_type: FacilityType = FacilityType.MANUFACTURING,
        facility_id: str = "FAC_001",
        include_quality_data: bool = True,
        include_cost_breakdown: bool = True,
        custom_profile: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive water usage data for a facility.
        
        Args:
            facility_type: Type of facility for usage pattern
            facility_id: Unique facility identifier
            include_quality_data: Include water quality monitoring data
            include_cost_breakdown: Include detailed cost breakdown
            custom_profile: Custom facility profile parameters
            
        Returns:
            Dictionary containing water usage data and metadata
        """
        if not self.validate_date_range():
            raise ValueError("Invalid date range configuration")
        
        # Get facility profile and dates
        facility_profile = get_facility_profile(facility_type, custom_profile)
        dates = self.get_date_range()
        
        self.logger.info(f"Generating water data for {facility_type.value} facility {facility_id}")
        self.logger.info(f"Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        
        # Generate base water usage
        water_data = self._generate_base_water_usage(dates, facility_profile)
        
        # Apply weather correlations
        water_data = self._apply_weather_effects(water_data, dates)
        
        # Generate water type breakdown
        water_data = self._generate_water_type_breakdown(water_data, dates)
        
        # Apply operational patterns and events
        water_data = self._apply_operational_events(water_data, dates)
        
        # Generate recycling and reuse data
        if self.water_config.enable_recycling:
            water_data = self._generate_recycling_data(water_data, dates)
        
        # Generate cost data
        water_data = self._generate_cost_data(water_data, dates, include_cost_breakdown)
        
        # Generate discharge data
        if self.water_config.enable_discharge_monitoring:
            water_data = self._generate_discharge_data(water_data, dates)
        
        # Generate quality data
        if include_quality_data:
            water_data = self._generate_quality_data(water_data, dates)
        
        # Apply data quality issues
        for key in ['total_usage_gallons', 'total_cost_usd']:
            if key in water_data:
                values, missing_mask = self.add_data_quality_issues(
                    np.array(water_data[key]), dates
                )
                water_data[key] = values.tolist()
                water_data[f'{key}_missing_mask'] = missing_mask.tolist()
        
        # Compile results
        result = {
            'metadata': {
                'facility_id': facility_id,
                'facility_type': facility_type.value,
                'generation_config': self.get_generation_metadata(),
                'facility_profile': facility_profile,
                'data_points': len(dates),
                'start_date': dates[0].isoformat(),
                'end_date': dates[-1].isoformat()
            },
            'water_usage': water_data,
            'summary_statistics': self._calculate_summary_statistics(water_data)
        }
        
        self.logger.info(f"Generated {len(dates)} days of water usage data")
        return result
    
    def _generate_base_water_usage(self, dates: List[datetime], facility_profile: Dict) -> Dict[str, Any]:
        """Generate base water usage patterns."""
        base_usage = facility_profile['base_water_usage']  # gallons per day
        
        # Apply facility-specific patterns
        seasonal_pattern = self.generate_seasonal_pattern(dates, base_usage)
        weekly_pattern = self.generate_weekly_pattern(dates, base_usage)
        
        # Combine patterns
        daily_usage = base_usage * seasonal_pattern * weekly_pattern
        
        # Add noise and volatility
        noise = self.generate_noise(daily_usage) * self.water_config.usage_volatility
        daily_usage += noise
        
        # Ensure non-negative usage
        daily_usage = np.maximum(daily_usage, base_usage * 0.1)
        
        return {
            'dates': [d.isoformat() for d in dates],
            'total_usage_gallons': daily_usage.tolist(),
            'base_usage_gallons': [base_usage] * len(dates)
        }
    
    def _apply_weather_effects(self, water_data: Dict, dates: List[datetime]) -> Dict:
        """Apply weather correlation effects to water usage."""
        usage_array = np.array(water_data['total_usage_gallons'])
        weather_effects = []
        evaporation_losses = []
        
        for i, date in enumerate(dates):
            weather = generate_weather_data(date)
            
            # Temperature effect (cooling water demand)
            temp_effect = 1.0 + self.water_config.temperature_correlation * (
                (weather['temperature_celsius'] - 20) / 20
            )
            
            # Humidity effect (evaporation)
            humidity_effect = 1.0 + self.water_config.humidity_correlation * (
                (weather['humidity_percent'] - 50) / 50
            )
            
            # Wind effect on evaporation
            wind_effect = 1.0 + 0.1 * (weather['wind_speed_kmh'] / 20)
            
            # Combined weather effect
            weather_multiplier = temp_effect * humidity_effect * wind_effect
            weather_effects.append(weather_multiplier)
            
            # Calculate evaporation loss
            if weather['temperature_celsius'] > 15:  # Only above 15°C
                evap_loss = usage_array[i] * self.water_config.evaporation_factor * (
                    weather['temperature_celsius'] / 30
                ) * (weather['wind_speed_kmh'] / 15)
                evaporation_losses.append(max(0, evap_loss))
            else:
                evaporation_losses.append(0)
        
        # Apply weather effects
        weather_adjusted_usage = usage_array * np.array(weather_effects)
        
        water_data.update({
            'total_usage_gallons': weather_adjusted_usage.tolist(),
            'weather_effects': weather_effects,
            'evaporation_losses_gallons': evaporation_losses
        })
        
        return water_data
    
    def _generate_water_type_breakdown(self, water_data: Dict, dates: List[datetime]) -> Dict:
        """Generate breakdown by water type."""
        total_usage = np.array(water_data['total_usage_gallons'])
        
        # Generate usage by type
        usage_by_type = {}
        for water_type, percentage in self.water_config.water_type_distribution.items():
            base_type_usage = total_usage * percentage
            
            # Add type-specific patterns
            if water_type == WaterType.COOLING:
                # Cooling water varies more with weather
                base_type_usage *= (1 + 0.2 * np.sin(
                    2 * np.pi * np.array([d.timetuple().tm_yday for d in dates]) / 365
                ))
            elif water_type == WaterType.IRRIGATION:
                # Irrigation seasonal (spring/summer)
                seasonal_factor = np.maximum(0.1, np.sin(
                    2 * np.pi * (np.array([d.timetuple().tm_yday for d in dates]) - 60) / 365
                ))
                base_type_usage *= seasonal_factor
            elif water_type == WaterType.PROCESS:
                # Process water correlates with production
                base_type_usage *= self.generate_weekly_pattern(dates, 1.0)
            
            usage_by_type[water_type.value] = base_type_usage.tolist()
        
        water_data['usage_by_type'] = usage_by_type
        return water_data
    
    def _apply_operational_events(self, water_data: Dict, dates: List[datetime]) -> Dict:
        """Apply leaks, maintenance, and conservation events."""
        usage_array = np.array(water_data['total_usage_gallons'])
        leak_events = []
        maintenance_events = []
        
        # Generate leak events
        if self.water_config.enable_leak_simulation:
            i = 0
            while i < len(dates):
                if np.random.random() < self.water_config.leak_probability:
                    # Generate leak
                    duration = np.random.randint(*self.water_config.leak_duration_hours) // 24
                    duration = max(1, duration)  # At least 1 day
                    end_idx = min(i + duration, len(dates))
                    
                    leak_multiplier = self.water_config.leak_rate_multiplier
                    for j in range(i, end_idx):
                        usage_array[j] *= leak_multiplier
                    
                    leak_events.append({
                        'start_date': dates[i].isoformat(),
                        'end_date': dates[end_idx-1].isoformat(),
                        'duration_days': duration,
                        'additional_usage_gallons': float(
                            np.sum(usage_array[i:end_idx] * (leak_multiplier - 1))
                        )
                    })
                    
                    i = end_idx
                else:
                    i += 1
        
        # Apply conservation measures
        conservation_factor = np.ones(len(dates))
        if self.water_config.enable_conservation:
            # Gradual implementation of conservation measures
            conservation_start = len(dates) // 3  # Start after 1/3 of period
            for i in range(conservation_start, len(dates)):
                progress = (i - conservation_start) / (len(dates) - conservation_start)
                conservation_factor[i] = 1.0 - (self.water_config.conservation_efficiency * progress)
        
        usage_array *= conservation_factor
        
        water_data.update({
            'total_usage_gallons': usage_array.tolist(),
            'leak_events': leak_events,
            'maintenance_events': maintenance_events,
            'conservation_factor': conservation_factor.tolist()
        })
        
        return water_data
    
    def _generate_recycling_data(self, water_data: Dict, dates: List[datetime]) -> Dict:
        """Generate water recycling and reuse data."""
        total_usage = np.array(water_data['total_usage_gallons'])
        
        # Calculate recycled water amounts
        recycled_amounts = total_usage * self.water_config.base_recycle_rate
        
        # Add efficiency variations
        efficiency_variations = np.random.normal(
            self.water_config.recycle_efficiency, 
            0.05, 
            len(dates)
        )
        efficiency_variations = np.clip(efficiency_variations, 0.7, 0.95)
        
        actual_recycled = recycled_amounts * efficiency_variations
        
        # Calculate fresh water needs
        fresh_water_needed = total_usage - actual_recycled
        
        # Recycling costs
        recycling_costs = actual_recycled * self.water_config.recycle_cost_per_gallon
        
        water_data.update({
            'recycled_water_gallons': actual_recycled.tolist(),
            'recycling_efficiency': efficiency_variations.tolist(),
            'fresh_water_intake_gallons': fresh_water_needed.tolist(),
            'recycling_cost_usd': recycling_costs.tolist(),
            'water_reuse_rate': (actual_recycled / total_usage).tolist()
        })
        
        return water_data
    
    def _generate_cost_data(self, water_data: Dict, dates: List[datetime], include_breakdown: bool) -> Dict:
        """Generate comprehensive cost data with tiered pricing."""
        usage_key = 'fresh_water_intake_gallons' if 'fresh_water_intake_gallons' in water_data else 'total_usage_gallons'
        usage = np.array(water_data[usage_key])
        
        # Calculate tiered costs
        daily_costs = []
        tier_breakdowns = []
        
        for daily_usage in usage:
            cost, breakdown = self._calculate_tiered_cost(daily_usage)
            daily_costs.append(cost)
            tier_breakdowns.append(breakdown)
        
        # Additional costs
        treatment_costs = usage * (self.water_config.treatment_chemical_cost / 1000)
        
        # Conservation program costs (monthly)
        conservation_costs = np.zeros(len(dates))
        if self.water_config.enable_conservation:
            monthly_cost = self.water_config.conservation_cost
            for i, date in enumerate(dates):
                if date.day == 1:  # First day of month
                    conservation_costs[i] = monthly_cost
        
        # Total costs
        total_costs = (
            np.array(daily_costs) + 
            treatment_costs + 
            conservation_costs
        )
        
        # Add recycling costs if available
        if 'recycling_cost_usd' in water_data:
            total_costs += np.array(water_data['recycling_cost_usd'])
        
        cost_data = {
            'total_cost_usd': total_costs.tolist(),
            'water_supply_cost_usd': daily_costs,
            'treatment_cost_usd': treatment_costs.tolist(),
            'conservation_cost_usd': conservation_costs.tolist()
        }
        
        if include_breakdown:
            cost_data['tiered_cost_breakdown'] = tier_breakdowns
        
        water_data.update(cost_data)
        return water_data
    
    def _calculate_tiered_cost(self, usage_gallons: float) -> Tuple[float, Dict]:
        """Calculate cost using tiered pricing structure."""
        remaining_usage = usage_gallons
        total_cost = 0.0
        breakdown = {}
        
        # Tier 1
        tier1_usage = min(remaining_usage, self.water_config.tier_1_threshold)
        tier1_cost = tier1_usage * (self.water_config.tier_1_rate / 1000)
        total_cost += tier1_cost
        breakdown['tier_1'] = {'usage': tier1_usage, 'cost': tier1_cost, 'rate': self.water_config.tier_1_rate}
        remaining_usage -= tier1_usage
        
        if remaining_usage > 0:
            # Tier 2
            tier2_usage = min(remaining_usage, self.water_config.tier_2_threshold - self.water_config.tier_1_threshold)
            tier2_cost = tier2_usage * (self.water_config.tier_2_rate / 1000)
            total_cost += tier2_cost
            breakdown['tier_2'] = {'usage': tier2_usage, 'cost': tier2_cost, 'rate': self.water_config.tier_2_rate}
            remaining_usage -= tier2_usage
        
        if remaining_usage > 0:
            # Tier 3
            tier3_usage = min(remaining_usage, self.water_config.tier_3_threshold - self.water_config.tier_2_threshold)
            tier3_cost = tier3_usage * (self.water_config.tier_3_rate / 1000)
            total_cost += tier3_cost
            breakdown['tier_3'] = {'usage': tier3_usage, 'cost': tier3_cost, 'rate': self.water_config.tier_3_rate}
            remaining_usage -= tier3_usage
        
        if remaining_usage > 0:
            # Tier 4
            tier4_cost = remaining_usage * (self.water_config.tier_4_rate / 1000)
            total_cost += tier4_cost
            breakdown['tier_4'] = {'usage': remaining_usage, 'cost': tier4_cost, 'rate': self.water_config.tier_4_rate}
        
        return total_cost, breakdown
    
    def _generate_discharge_data(self, water_data: Dict, dates: List[datetime]) -> Dict:
        """Generate water discharge monitoring data."""
        intake_key = 'fresh_water_intake_gallons' if 'fresh_water_intake_gallons' in water_data else 'total_usage_gallons'
        intake = np.array(water_data[intake_key])
        
        # Calculate discharge volumes
        discharge_volumes = intake * self.water_config.discharge_rate
        
        # Add discharge treatment costs
        discharge_treatment_costs = discharge_volumes * (self.water_config.discharge_treatment_cost / 1000)
        
        # Generate discharge temperature (affected by process and weather)
        discharge_temps = []
        for i, date in enumerate(dates):
            weather = generate_weather_data(date)
            base_temp = weather['temperature_celsius'] + np.random.normal(5, 2)  # Process heating
            discharge_temps.append(max(base_temp, weather['temperature_celsius']))
        
        water_data.update({
            'discharge_volume_gallons': discharge_volumes.tolist(),
            'discharge_temperature_celsius': discharge_temps,
            'discharge_treatment_cost_usd': discharge_treatment_costs.tolist()
        })
        
        # Update total costs
        if 'total_cost_usd' in water_data:
            updated_costs = np.array(water_data['total_cost_usd']) + discharge_treatment_costs
            water_data['total_cost_usd'] = updated_costs.tolist()
        
        return water_data
    
    def _generate_quality_data(self, water_data: Dict, dates: List[datetime]) -> Dict:
        """Generate water quality monitoring data."""
        quality_data = {}
        
        for parameter, (min_val, max_val) in self.water_config.quality_ranges.items():
            # Generate base values within range
            base_values = np.random.uniform(min_val, max_val, len(dates))
            
            # Add seasonal variations for some parameters
            if parameter in [WaterQualityParameter.TEMPERATURE, WaterQualityParameter.DISSOLVED_OXYGEN]:
                seasonal_effect = 0.2 * np.sin(2 * np.pi * np.array([d.timetuple().tm_yday for d in dates]) / 365)
                base_values *= (1 + seasonal_effect)
            
            # Add process-related variations
            if parameter in [WaterQualityParameter.BOD, WaterQualityParameter.COD, WaterQualityParameter.TSS]:
                # Higher values during high production periods
                production_effect = np.array(water_data['total_usage_gallons'])
                production_effect = (production_effect - np.min(production_effect)) / (np.max(production_effect) - np.min(production_effect))
                base_values *= (1 + 0.3 * production_effect)
            
            # Add random variations
            noise = np.random.normal(0, 0.1 * (max_val - min_val), len(dates))
            final_values = np.clip(base_values + noise, min_val, max_val)
            
            quality_data[f'{parameter.value}_values'] = final_values.tolist()
        
        water_data['quality_monitoring'] = quality_data
        return water_data
    
    def _calculate_summary_statistics(self, water_data: Dict) -> Dict:
        """Calculate summary statistics for the generated water data."""
        usage = np.array(water_data['total_usage_gallons'])
        costs = np.array(water_data['total_cost_usd'])
        
        stats = {
            'total_usage': {
                'mean_daily_gallons': float(np.mean(usage)),
                'median_daily_gallons': float(np.median(usage)),
                'std_daily_gallons': float(np.std(usage)),
                'min_daily_gallons': float(np.min(usage)),
                'max_daily_gallons': float(np.max(usage)),
                'total_period_gallons': float(np.sum(usage))
            },
            'total_costs': {
                'mean_daily_usd': float(np.mean(costs)),
                'median_daily_usd': float(np.median(costs)),
                'std_daily_usd': float(np.std(costs)),
                'min_daily_usd': float(np.min(costs)),
                'max_daily_usd': float(np.max(costs)),
                'total_period_usd': float(np.sum(costs))
            }
        }
        
        # Add recycling statistics if available
        if 'recycled_water_gallons' in water_data:
            recycled = np.array(water_data['recycled_water_gallons'])
            stats['recycling'] = {
                'mean_daily_recycled_gallons': float(np.mean(recycled)),
                'total_recycled_gallons': float(np.sum(recycled)),
                'recycling_percentage': float(np.sum(recycled) / np.sum(usage) * 100)
            }
        
        # Add leak event statistics if available
        if 'leak_events' in water_data and water_data['leak_events']:
            total_leak_loss = sum(event['additional_usage_gallons'] for event in water_data['leak_events'])
            stats['leak_events'] = {
                'total_events': len(water_data['leak_events']),
                'total_additional_usage_gallons': total_leak_loss,
                'leak_loss_percentage': float(total_leak_loss / np.sum(usage) * 100)
            }
        
        return stats


def main():
    """Example usage of the WaterGenerator."""
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create generator with custom config
    config = WaterGeneratorConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        enable_recycling=True,
        enable_leak_simulation=True,
        base_recycle_rate=0.3
    )
    
    generator = WaterGenerator(config)
    
    # Generate data for different facility types
    facilities = [
        (FacilityType.MANUFACTURING, "PLANT_001"),
        (FacilityType.CHEMICAL, "CHEM_001"),
        (FacilityType.POWER_PLANT, "PWR_001")
    ]
    
    for facility_type, facility_id in facilities:
        print(f"\nGenerating water data for {facility_type.value} facility {facility_id}")
        
        data = generator.generate(
            facility_type=facility_type,
            facility_id=facility_id,
            include_quality_data=True,
            include_cost_breakdown=True
        )
        
        # Print summary
        print(f"Generated {data['metadata']['data_points']} days of data")
        print(f"Mean daily usage: {data['summary_statistics']['total_usage']['mean_daily_gallons']:.0f} gallons")
        print(f"Mean daily cost: ${data['summary_statistics']['total_costs']['mean_daily_usd']:.2f}")
        
        if 'recycling' in data['summary_statistics']:
            print(f"Recycling rate: {data['summary_statistics']['recycling']['recycling_percentage']:.1f}%")


if __name__ == "__main__":
    main()