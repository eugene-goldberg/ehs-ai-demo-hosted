"""
Data utility functions for EHS AI demo data generation.

This module provides utility functions for generating realistic environmental,
health, and safety data including weather patterns, facility profiles, cost
calculations, and emission factors based on industry standards.
"""

import random
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

class FacilityType(Enum):
    """Facility types with different risk and operational profiles."""
    MANUFACTURING = "manufacturing"
    CHEMICAL = "chemical"
    OIL_REFINERY = "oil_refinery"
    POWER_PLANT = "power_plant"
    WAREHOUSE = "warehouse"
    OFFICE = "office"
    CONSTRUCTION = "construction"

class WeatherCondition(Enum):
    """Weather condition types affecting operations."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"
    FOGGY = "foggy"

# EHS Constants based on industry standards
EHS_CONSTANTS = {
    "OSHA_RECORDABLE_RATE_BENCHMARK": 3.0,  # per 100 employees
    "LTIR_BENCHMARK": 1.2,  # Lost Time Incident Rate
    "DART_BENCHMARK": 1.8,  # Days Away, Restricted, Transfer rate
    "NEAR_MISS_TO_INCIDENT_RATIO": 300,  # Near misses per incident
    "CO2_PRICE_PER_TONNE": 50,  # USD per tonne CO2
    "ENERGY_COST_INDUSTRIAL": 0.08,  # USD per kWh
    "WATER_COST_INDUSTRIAL": 0.003,  # USD per gallon
    "WASTE_DISPOSAL_COST": 120,  # USD per tonne
    "COMPLIANCE_VIOLATION_FINE": 25000,  # Average fine
}

# CO2 Emission factors (kg CO2 per unit)
CO2_EMISSION_FACTORS = {
    "electricity_kwh": 0.4,  # kg CO2 per kWh (US grid average)
    "natural_gas_m3": 1.9,  # kg CO2 per cubic meter
    "diesel_liter": 2.67,  # kg CO2 per liter
    "gasoline_liter": 2.31,  # kg CO2 per liter
    "coal_kg": 2.86,  # kg CO2 per kg coal
    "steam_kg": 0.18,  # kg CO2 per kg steam
}

# Facility profiles with operational characteristics
FACILITY_PROFILES = {
    FacilityType.MANUFACTURING: {
        "base_energy_consumption": 5000,  # kWh/day
        "base_water_usage": 2000,  # gallons/day
        "base_waste_generation": 2.5,  # tonnes/day
        "employee_count_range": (150, 800),
        "risk_multiplier": 1.2,
        "seasonal_variation": 0.15,
        "weather_sensitivity": 0.1,
    },
    FacilityType.CHEMICAL: {
        "base_energy_consumption": 8000,
        "base_water_usage": 3500,
        "base_waste_generation": 4.0,
        "employee_count_range": (100, 400),
        "risk_multiplier": 2.5,
        "seasonal_variation": 0.08,
        "weather_sensitivity": 0.25,
    },
    FacilityType.OIL_REFINERY: {
        "base_energy_consumption": 15000,
        "base_water_usage": 8000,
        "base_waste_generation": 8.5,
        "employee_count_range": (200, 600),
        "risk_multiplier": 3.0,
        "seasonal_variation": 0.12,
        "weather_sensitivity": 0.2,
    },
    FacilityType.POWER_PLANT: {
        "base_energy_consumption": 25000,
        "base_water_usage": 15000,
        "base_waste_generation": 12.0,
        "employee_count_range": (80, 250),
        "risk_multiplier": 2.2,
        "seasonal_variation": 0.3,
        "weather_sensitivity": 0.15,
    },
    FacilityType.WAREHOUSE: {
        "base_energy_consumption": 800,
        "base_water_usage": 200,
        "base_waste_generation": 0.5,
        "employee_count_range": (50, 300),
        "risk_multiplier": 0.6,
        "seasonal_variation": 0.2,
        "weather_sensitivity": 0.05,
    },
    FacilityType.OFFICE: {
        "base_energy_consumption": 300,
        "base_water_usage": 150,
        "base_waste_generation": 0.2,
        "employee_count_range": (100, 1000),
        "risk_multiplier": 0.2,
        "seasonal_variation": 0.1,
        "weather_sensitivity": 0.02,
    },
    FacilityType.CONSTRUCTION: {
        "base_energy_consumption": 2000,
        "base_water_usage": 800,
        "base_waste_generation": 3.5,
        "employee_count_range": (20, 200),
        "risk_multiplier": 4.0,
        "seasonal_variation": 0.4,
        "weather_sensitivity": 0.5,
    },
}

def generate_weather_data(date: datetime, location: str = "default") -> Dict[str, Any]:
    """
    Generate realistic weather data for a given date and location.
    
    Args:
        date: Date for weather generation
        location: Location identifier (affects regional patterns)
        
    Returns:
        Dictionary containing weather parameters
    """
    # Seasonal temperature pattern
    day_of_year = date.timetuple().tm_yday
    seasonal_temp = 60 + 30 * math.sin(2 * math.pi * (day_of_year - 81) / 365)
    
    # Add daily variation and randomness
    temp_celsius = seasonal_temp + random.normalvariate(0, 8)
    temp_celsius = max(-30, min(45, temp_celsius))  # Reasonable bounds
    
    # Humidity based on temperature and season
    base_humidity = 50 + 20 * math.sin(2 * math.pi * day_of_year / 365)
    humidity = max(20, min(90, base_humidity + random.normalvariate(0, 15)))
    
    # Wind speed (affected by season)
    wind_speed = abs(random.normalvariate(10, 5))
    
    # Pressure
    pressure = random.normalvariate(1013.25, 15)  # hPa
    
    # Precipitation
    precip_probability = 0.3 if date.month in [6, 7, 8] else 0.4  # Summer vs other
    precipitation = random.expovariate(0.2) if random.random() < precip_probability else 0
    
    # Weather condition based on precipitation and other factors
    if precipitation > 10:
        condition = WeatherCondition.STORMY
    elif precipitation > 2:
        condition = WeatherCondition.RAINY
    elif humidity > 80 and wind_speed < 5:
        condition = WeatherCondition.FOGGY
    elif temp_celsius < 0 and precipitation > 0:
        condition = WeatherCondition.SNOWY
    elif humidity > 70:
        condition = WeatherCondition.CLOUDY
    else:
        condition = WeatherCondition.CLEAR
    
    return {
        "date": date,
        "temperature_celsius": round(temp_celsius, 1),
        "humidity_percent": round(humidity, 1),
        "wind_speed_kmh": round(wind_speed, 1),
        "pressure_hpa": round(pressure, 1),
        "precipitation_mm": round(precipitation, 2),
        "condition": condition.value,
        "uv_index": max(0, min(11, int(8 * math.sin(2 * math.pi * day_of_year / 365)) + random.randint(-2, 2))),
        "visibility_km": max(0.5, min(50, random.normalvariate(20, 10)))
    }

def get_facility_profile(facility_type: FacilityType, custom_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get facility operational profile with optional customization.
    
    Args:
        facility_type: Type of facility
        custom_params: Optional custom parameters to override defaults
        
    Returns:
        Facility profile dictionary
    """
    profile = FACILITY_PROFILES[facility_type].copy()
    
    # Generate specific employee count within range
    emp_range = profile["employee_count_range"]
    profile["employee_count"] = random.randint(emp_range[0], emp_range[1])
    
    # Add facility-specific characteristics
    profile["facility_type"] = facility_type.value
    profile["operational_hours"] = 24 if facility_type in [
        FacilityType.CHEMICAL, FacilityType.OIL_REFINERY, FacilityType.POWER_PLANT
    ] else 8
    
    # Override with custom parameters if provided
    if custom_params:
        profile.update(custom_params)
    
    return profile

def calculate_environmental_costs(
    energy_consumption: float,
    water_usage: float,
    waste_generation: float,
    co2_emissions: float,
    compliance_violations: int = 0
) -> Dict[str, float]:
    """
    Calculate environmental costs based on consumption and emissions.
    
    Args:
        energy_consumption: Energy consumption in kWh
        water_usage: Water usage in gallons
        waste_generation: Waste generation in tonnes
        co2_emissions: CO2 emissions in tonnes
        compliance_violations: Number of violations
        
    Returns:
        Dictionary of cost breakdowns
    """
    energy_cost = energy_consumption * EHS_CONSTANTS["ENERGY_COST_INDUSTRIAL"]
    water_cost = water_usage * EHS_CONSTANTS["WATER_COST_INDUSTRIAL"]
    waste_cost = waste_generation * EHS_CONSTANTS["WASTE_DISPOSAL_COST"]
    carbon_cost = co2_emissions * EHS_CONSTANTS["CO2_PRICE_PER_TONNE"]
    violation_cost = compliance_violations * EHS_CONSTANTS["COMPLIANCE_VIOLATION_FINE"]
    
    total_cost = energy_cost + water_cost + waste_cost + carbon_cost + violation_cost
    
    return {
        "energy_cost": round(energy_cost, 2),
        "water_cost": round(water_cost, 2),
        "waste_cost": round(waste_cost, 2),
        "carbon_cost": round(carbon_cost, 2),
        "violation_cost": round(violation_cost, 2),
        "total_environmental_cost": round(total_cost, 2)
    }

def calculate_co2_emissions(
    electricity_kwh: float = 0,
    natural_gas_m3: float = 0,
    diesel_liters: float = 0,
    gasoline_liters: float = 0,
    coal_kg: float = 0,
    steam_kg: float = 0
) -> Dict[str, float]:
    """
    Calculate CO2 emissions from various energy sources.
    
    Args:
        electricity_kwh: Electricity consumption in kWh
        natural_gas_m3: Natural gas consumption in cubic meters
        diesel_liters: Diesel consumption in liters
        gasoline_liters: Gasoline consumption in liters
        coal_kg: Coal consumption in kg
        steam_kg: Steam consumption in kg
        
    Returns:
        Dictionary with emission breakdowns
    """
    emissions = {
        "electricity_co2": electricity_kwh * CO2_EMISSION_FACTORS["electricity_kwh"],
        "natural_gas_co2": natural_gas_m3 * CO2_EMISSION_FACTORS["natural_gas_m3"],
        "diesel_co2": diesel_liters * CO2_EMISSION_FACTORS["diesel_liter"],
        "gasoline_co2": gasoline_liters * CO2_EMISSION_FACTORS["gasoline_liter"],
        "coal_co2": coal_kg * CO2_EMISSION_FACTORS["coal_kg"],
        "steam_co2": steam_kg * CO2_EMISSION_FACTORS["steam_kg"],
    }
    
    total_co2_kg = sum(emissions.values())
    total_co2_tonnes = total_co2_kg / 1000
    
    return {
        **{k: round(v, 2) for k, v in emissions.items()},
        "total_co2_kg": round(total_co2_kg, 2),
        "total_co2_tonnes": round(total_co2_tonnes, 4)
    }

def apply_weather_correlation(
    base_value: float,
    weather_data: Dict[str, Any],
    sensitivity: float = 0.1
) -> float:
    """
    Apply weather-based correlations to operational metrics.
    
    Args:
        base_value: Base value before weather adjustment
        weather_data: Weather data dictionary
        sensitivity: Sensitivity factor (0-1)
        
    Returns:
        Weather-adjusted value
    """
    temp = weather_data["temperature_celsius"]
    humidity = weather_data["humidity_percent"]
    wind_speed = weather_data["wind_speed_kmh"]
    precipitation = weather_data["precipitation_mm"]
    
    # Temperature effect (higher energy use in extreme temperatures)
    temp_effect = 1 + sensitivity * abs(temp - 20) / 20
    
    # Humidity effect (slight increase in energy use)
    humidity_effect = 1 + sensitivity * 0.3 * (humidity - 50) / 100
    
    # Wind effect (slight reduction in energy use with wind)
    wind_effect = 1 - sensitivity * 0.1 * min(wind_speed, 30) / 30
    
    # Precipitation effect (increased operational demands)
    precip_effect = 1 + sensitivity * 0.2 * min(precipitation, 20) / 20
    
    total_effect = temp_effect * humidity_effect * wind_effect * precip_effect
    
    return base_value * total_effect

def apply_seasonal_variation(
    base_value: float,
    date: datetime,
    variation_amplitude: float = 0.2
) -> float:
    """
    Apply seasonal variation to operational metrics.
    
    Args:
        base_value: Base value before seasonal adjustment
        date: Date for seasonal calculation
        variation_amplitude: Amplitude of seasonal variation (0-1)
        
    Returns:
        Seasonally adjusted value
    """
    day_of_year = date.timetuple().tm_yday
    seasonal_factor = 1 + variation_amplitude * math.sin(2 * math.pi * (day_of_year - 81) / 365)
    
    return base_value * seasonal_factor

def calculate_safety_metrics(
    incidents: int,
    near_misses: int,
    employee_count: int,
    hours_worked: float,
    facility_risk_multiplier: float = 1.0
) -> Dict[str, float]:
    """
    Calculate standard safety metrics.
    
    Args:
        incidents: Number of recordable incidents
        near_misses: Number of near miss events
        employee_count: Total employee count
        hours_worked: Total hours worked
        facility_risk_multiplier: Risk adjustment factor
        
    Returns:
        Dictionary of safety metrics
    """
    # Standard calculations (per 100 employees or 200,000 hours)
    total_recordable_rate = (incidents / employee_count) * 100 if employee_count > 0 else 0
    incident_rate_200k = (incidents / hours_worked) * 200000 if hours_worked > 0 else 0
    
    # Apply facility risk adjustment
    adjusted_trr = total_recordable_rate * facility_risk_multiplier
    adjusted_ir = incident_rate_200k * facility_risk_multiplier
    
    # Near miss ratio
    near_miss_ratio = near_misses / max(incidents, 1)
    
    return {
        "total_recordable_rate": round(adjusted_trr, 2),
        "incident_rate_200k_hours": round(adjusted_ir, 2),
        "near_miss_ratio": round(near_miss_ratio, 1),
        "safety_performance_index": round(max(0, 100 - adjusted_trr * 10), 1),
        "benchmark_comparison": round((adjusted_trr / EHS_CONSTANTS["OSHA_RECORDABLE_RATE_BENCHMARK"]) * 100, 1)
    }

def generate_operational_variance(
    base_value: float,
    variance_type: str = "normal",
    variance_factor: float = 0.1
) -> float:
    """
    Add realistic operational variance to base values.
    
    Args:
        base_value: Base operational value
        variance_type: Type of variance ('normal', 'lognormal', 'uniform')
        variance_factor: Variance amplitude factor
        
    Returns:
        Value with applied variance
    """
    if variance_type == "normal":
        multiplier = random.normalvariate(1.0, variance_factor)
    elif variance_type == "lognormal":
        multiplier = random.lognormvariate(0, variance_factor)
    elif variance_type == "uniform":
        multiplier = random.uniform(1 - variance_factor, 1 + variance_factor)
    else:
        multiplier = 1.0
    
    return max(0, base_value * multiplier)

def calculate_correlation_matrix(data_points: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlation matrix for operational metrics.
    
    Args:
        data_points: List of data point dictionaries
        
    Returns:
        Correlation matrix as nested dictionary
    """
    if not data_points:
        return {}
    
    # Extract metric names
    metrics = list(data_points[0].keys())
    
    # Convert to numpy arrays for correlation calculation
    data_matrix = []
    for metric in metrics:
        values = [point.get(metric, 0) for point in data_points]
        data_matrix.append(values)
    
    # Calculate correlation matrix
    correlation_matrix = {}
    for i, metric1 in enumerate(metrics):
        correlation_matrix[metric1] = {}
        for j, metric2 in enumerate(metrics):
            if len(set(data_matrix[i])) > 1 and len(set(data_matrix[j])) > 1:
                corr = np.corrcoef(data_matrix[i], data_matrix[j])[0, 1]
                correlation_matrix[metric1][metric2] = round(corr if not np.isnan(corr) else 0, 3)
            else:
                correlation_matrix[metric1][metric2] = 1.0 if i == j else 0.0
    
    return correlation_matrix

def get_industry_benchmarks(facility_type: FacilityType) -> Dict[str, float]:
    """
    Get industry-specific benchmarks for comparison.
    
    Args:
        facility_type: Type of facility
        
    Returns:
        Dictionary of industry benchmarks
    """
    base_benchmarks = {
        "energy_intensity_kwh_per_employee": 50,
        "water_intensity_gal_per_employee": 20,
        "waste_intensity_kg_per_employee": 5,
        "co2_intensity_tonnes_per_employee": 8,
        "incident_rate": EHS_CONSTANTS["OSHA_RECORDABLE_RATE_BENCHMARK"],
    }
    
    # Adjust benchmarks based on facility type
    multipliers = {
        FacilityType.MANUFACTURING: 1.2,
        FacilityType.CHEMICAL: 2.5,
        FacilityType.OIL_REFINERY: 4.0,
        FacilityType.POWER_PLANT: 8.0,
        FacilityType.WAREHOUSE: 0.3,
        FacilityType.OFFICE: 0.1,
        FacilityType.CONSTRUCTION: 1.5,
    }
    
    multiplier = multipliers.get(facility_type, 1.0)
    
    return {
        key: round(value * multiplier, 2) 
        for key, value in base_benchmarks.items()
    }

def validate_data_quality(data_point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean data points for quality assurance.
    
    Args:
        data_point: Data point to validate
        
    Returns:
        Dictionary with validation results and cleaned data
    """
    issues = []
    cleaned_data = data_point.copy()
    
    # Check for negative values where they shouldn't be
    non_negative_fields = [
        "energy_consumption", "water_usage", "waste_generation",
        "co2_emissions", "employee_count", "incidents", "near_misses"
    ]
    
    for field in non_negative_fields:
        if field in cleaned_data and cleaned_data[field] < 0:
            issues.append(f"Negative value for {field}: {cleaned_data[field]}")
            cleaned_data[field] = 0
    
    # Check for unrealistic ranges
    range_checks = {
        "temperature_celsius": (-50, 60),
        "humidity_percent": (0, 100),
        "wind_speed_kmh": (0, 200),
        "pressure_hpa": (900, 1100),
    }
    
    for field, (min_val, max_val) in range_checks.items():
        if field in cleaned_data:
            if cleaned_data[field] < min_val or cleaned_data[field] > max_val:
                issues.append(f"Out of range value for {field}: {cleaned_data[field]}")
                cleaned_data[field] = max(min_val, min(max_val, cleaned_data[field]))
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "cleaned_data": cleaned_data,
        "data_quality_score": max(0, 100 - len(issues) * 10)
    }