"""
Waste Generation Data Generator Module

This module provides comprehensive waste generation pattern simulation for EHS AI Demo system.
It generates realistic waste generation patterns including different waste types, facility-specific
waste profiles, regulatory compliance tracking, disposal costs, and environmental impact metrics.

Features:
- Multiple waste stream types (hazardous, non-hazardous, recyclable, organic)
- Facility-specific waste generation profiles based on industry type
- Recycling and diversion rate tracking with realistic patterns
- Hazardous waste manifest generation with regulatory compliance
- Disposal cost calculations by waste type and disposal method
- Environmental impact metrics (landfill diversion, GHG savings)
- Seasonal waste generation variations
- Regulatory compliance tracking and reporting
- Waste treatment method selection and optimization
- Circular economy metrics and waste-to-energy calculations

Author: EHS AI Demo Team
Created: 2025-08-28
Version: 1.0.0
"""

import math
import random
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .base_generator import BaseGenerator, GeneratorConfig
from ..utils.data_utils import (
    FacilityType, 
    get_facility_profile,
    calculate_co2_emissions,
    calculate_environmental_costs,
    EHS_CONSTANTS
)


class WasteType(Enum):
    """Enumeration of waste types"""
    HAZARDOUS = "hazardous"
    NON_HAZARDOUS = "non_hazardous"
    RECYCLABLE = "recyclable"
    ORGANIC = "organic"
    E_WASTE = "e_waste"
    CONSTRUCTION = "construction"
    MEDICAL = "medical"
    CHEMICAL = "chemical"


class WasteCategory(Enum):
    """EPA waste categories"""
    RCRA_LISTED = "rcra_listed"
    RCRA_CHARACTERISTIC = "rcra_characteristic"
    UNIVERSAL = "universal"
    NON_REGULATED = "non_regulated"
    STATE_REGULATED = "state_regulated"


class DisposalMethod(Enum):
    """Waste disposal/treatment methods"""
    LANDFILL = "landfill"
    INCINERATION = "incineration"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    WASTE_TO_ENERGY = "waste_to_energy"
    TREATMENT = "treatment"
    NEUTRALIZATION = "neutralization"
    SOLIDIFICATION = "solidification"
    FUEL_BLENDING = "fuel_blending"


@dataclass
class WasteStream:
    """Definition of a waste stream"""
    name: str
    waste_type: WasteType
    category: WasteCategory
    epa_waste_code: Optional[str] = None
    hazard_class: Optional[str] = None
    physical_state: str = "solid"  # solid, liquid, gas, sludge
    disposal_method: DisposalMethod = DisposalMethod.LANDFILL
    disposal_cost_per_ton: float = 150.0  # USD per ton
    recycling_value_per_ton: float = 0.0  # USD per ton
    carbon_factor: float = 0.0  # kg CO2e per ton
    regulatory_requirements: List[str] = field(default_factory=list)


@dataclass
class WasteGeneratorConfig(GeneratorConfig):
    """Configuration for waste generation"""
    
    # Base generation rates (tons per day for different facility types)
    base_generation_rates: Dict[str, float] = field(default_factory=lambda: {
        "manufacturing": 2.5,
        "office": 0.1,
        "warehouse": 0.3,
        "laboratory": 0.8,
        "hospital": 1.2,
        "chemical_plant": 5.0
    })
    
    # Waste composition by facility type (percentage of total)
    waste_composition_profiles: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "manufacturing": {
            "non_hazardous": 0.60,
            "recyclable": 0.25,
            "hazardous": 0.10,
            "organic": 0.05
        },
        "office": {
            "non_hazardous": 0.30,
            "recyclable": 0.65,
            "organic": 0.05,
            "hazardous": 0.00
        },
        "laboratory": {
            "hazardous": 0.40,
            "non_hazardous": 0.35,
            "recyclable": 0.20,
            "chemical": 0.05
        },
        "chemical_plant": {
            "hazardous": 0.45,
            "chemical": 0.25,
            "non_hazardous": 0.20,
            "recyclable": 0.10
        }
    })
    
    # Recycling and diversion rates by waste type
    base_recycling_rates: Dict[str, float] = field(default_factory=lambda: {
        "recyclable": 0.85,
        "organic": 0.70,
        "non_hazardous": 0.15,
        "e_waste": 0.80,
        "construction": 0.60,
        "hazardous": 0.05,  # Treatment, not recycling
        "chemical": 0.10
    })
    
    # Disposal costs by waste type and method (USD per ton)
    disposal_costs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "hazardous": {
            "treatment": 800.0,
            "incineration": 1200.0,
            "solidification": 600.0,
            "neutralization": 900.0
        },
        "non_hazardous": {
            "landfill": 120.0,
            "incineration": 180.0,
            "waste_to_energy": 150.0
        },
        "recyclable": {
            "recycling": -20.0,  # Revenue from recycling
            "landfill": 120.0
        },
        "organic": {
            "composting": 80.0,
            "waste_to_energy": 120.0,
            "landfill": 150.0
        }
    })
    
    # Environmental impact factors
    carbon_factors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "landfill": {
            "non_hazardous": 1.2,  # kg CO2e per ton
            "organic": 0.8
        },
        "incineration": {
            "non_hazardous": 0.3,
            "hazardous": 0.5
        },
        "recycling": {
            "recyclable": -2.1,  # CO2 savings
            "e_waste": -1.8
        },
        "composting": {
            "organic": -0.5
        }
    })
    
    # Regulatory compliance parameters
    manifest_required_threshold: float = 0.1  # tons
    generator_id_prefix: str = "DEMO"
    enable_regulatory_tracking: bool = True
    compliance_failure_rate: float = 0.02  # 2% chance of minor compliance issues
    
    # Seasonal variations
    seasonal_waste_factors: Dict[str, Dict[int, float]] = field(default_factory=lambda: {
        "organic": {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.3, 
                   7: 1.4, 8: 1.3, 9: 1.2, 10: 1.1, 11: 1.0, 12: 0.9},
        "construction": {1: 0.6, 2: 0.7, 3: 1.0, 4: 1.3, 5: 1.5, 6: 1.4,
                        7: 1.3, 8: 1.4, 9: 1.2, 10: 1.1, 11: 0.8, 12: 0.6}
    })
    
    # Quality assurance parameters
    enable_waste_audits: bool = True
    audit_frequency_days: int = 90
    contamination_rate: float = 0.05  # 5% contamination in recyclables
    
    # Waste minimization targets
    annual_reduction_target: float = 0.03  # 3% annual reduction
    recycling_improvement_target: float = 0.02  # 2% annual improvement


class WasteGenerator(BaseGenerator):
    """
    Comprehensive waste generation data generator.
    
    Generates realistic waste generation patterns for industrial facilities
    including multiple waste streams, regulatory compliance, costs, and 
    environmental impact metrics.
    """
    
    def __init__(self, config: Optional[WasteGeneratorConfig] = None):
        """
        Initialize the waste generator.
        
        Args:
            config: Configuration object for waste generation
        """
        self.waste_config = config or WasteGeneratorConfig()
        super().__init__(self.waste_config)
        
        # Initialize waste stream definitions
        self._initialize_waste_streams()
        
        # Initialize regulatory tracking
        self._initialize_regulatory_data()
        
    def _initialize_waste_streams(self) -> None:
        """Initialize standard waste stream definitions."""
        self.waste_streams = {
            "office_paper": WasteStream(
                name="Mixed Office Paper",
                waste_type=WasteType.RECYCLABLE,
                category=WasteCategory.NON_REGULATED,
                disposal_method=DisposalMethod.RECYCLING,
                disposal_cost_per_ton=50.0,
                recycling_value_per_ton=80.0,
                carbon_factor=-2.1
            ),
            "cardboard": WasteStream(
                name="Corrugated Cardboard",
                waste_type=WasteType.RECYCLABLE,
                category=WasteCategory.NON_REGULATED,
                disposal_method=DisposalMethod.RECYCLING,
                disposal_cost_per_ton=45.0,
                recycling_value_per_ton=100.0,
                carbon_factor=-1.8
            ),
            "food_waste": WasteStream(
                name="Food Waste",
                waste_type=WasteType.ORGANIC,
                category=WasteCategory.NON_REGULATED,
                disposal_method=DisposalMethod.COMPOSTING,
                disposal_cost_per_ton=80.0,
                carbon_factor=-0.5
            ),
            "general_trash": WasteStream(
                name="General Non-Hazardous Waste",
                waste_type=WasteType.NON_HAZARDOUS,
                category=WasteCategory.NON_REGULATED,
                disposal_method=DisposalMethod.LANDFILL,
                disposal_cost_per_ton=120.0,
                carbon_factor=1.2
            ),
            "solvent_waste": WasteStream(
                name="Waste Solvents",
                waste_type=WasteType.HAZARDOUS,
                category=WasteCategory.RCRA_LISTED,
                epa_waste_code="F003",
                hazard_class="Ignitable",
                physical_state="liquid",
                disposal_method=DisposalMethod.INCINERATION,
                disposal_cost_per_ton=1200.0,
                carbon_factor=0.5,
                regulatory_requirements=["Manifest Required", "DOT Shipping", "EPA ID Required"]
            ),
            "paint_waste": WasteStream(
                name="Waste Paint",
                waste_type=WasteType.HAZARDOUS,
                category=WasteCategory.RCRA_CHARACTERISTIC,
                epa_waste_code="D001",
                hazard_class="Ignitable",
                physical_state="liquid",
                disposal_method=DisposalMethod.TREATMENT,
                disposal_cost_per_ton=800.0,
                carbon_factor=0.8,
                regulatory_requirements=["Manifest Required", "Treatment Standards"]
            ),
            "electronics": WasteStream(
                name="Electronic Waste",
                waste_type=WasteType.E_WASTE,
                category=WasteCategory.UNIVERSAL,
                disposal_method=DisposalMethod.RECYCLING,
                disposal_cost_per_ton=250.0,
                recycling_value_per_ton=150.0,
                carbon_factor=-1.8,
                regulatory_requirements=["Certified Recycler Required"]
            ),
            "batteries": WasteStream(
                name="Lead-Acid Batteries",
                waste_type=WasteType.HAZARDOUS,
                category=WasteCategory.UNIVERSAL,
                epa_waste_code="U001",
                disposal_method=DisposalMethod.RECYCLING,
                disposal_cost_per_ton=100.0,
                recycling_value_per_ton=200.0,
                carbon_factor=-0.5,
                regulatory_requirements=["Universal Waste Standards"]
            ),
            "construction_debris": WasteStream(
                name="Construction & Demolition Debris",
                waste_type=WasteType.CONSTRUCTION,
                category=WasteCategory.NON_REGULATED,
                disposal_method=DisposalMethod.RECYCLING,
                disposal_cost_per_ton=75.0,
                recycling_value_per_ton=25.0,
                carbon_factor=-0.8
            ),
            "chemical_waste": WasteStream(
                name="Lab Chemical Waste",
                waste_type=WasteType.CHEMICAL,
                category=WasteCategory.RCRA_LISTED,
                epa_waste_code="P001",
                hazard_class="Toxic",
                physical_state="liquid",
                disposal_method=DisposalMethod.TREATMENT,
                disposal_cost_per_ton=1500.0,
                carbon_factor=1.0,
                regulatory_requirements=["Manifest Required", "Lab Pack", "Compatibility Testing"]
            )
        }
    
    def _initialize_regulatory_data(self) -> None:
        """Initialize regulatory compliance tracking data."""
        self.generator_id = f"{self.waste_config.generator_id_prefix}{random.randint(1000, 9999)}"
        self.epa_id_number = f"WAR{random.randint(100000000, 999999999)}"
        self.manifest_counter = 1
        
        # Disposal facility data
        self.disposal_facilities = {
            "landfill": {
                "name": "Regional Sanitary Landfill",
                "epa_id": "WAR123456789",
                "address": "123 Disposal Lane, Landfill City, ST 12345",
                "permit_number": "LF-2024-001"
            },
            "incinerator": {
                "name": "Advanced Waste Treatment Facility",
                "epa_id": "WAR987654321",
                "address": "456 Incinerator Dr, Burn City, ST 54321",
                "permit_number": "IN-2024-002"
            },
            "recycling": {
                "name": "Metro Recycling Center",
                "epa_id": "WAR555666777",
                "address": "789 Recycling Blvd, Green City, ST 67890",
                "permit_number": "RC-2024-003"
            }
        }
    
    def generate_waste_streams_for_facility(self, facility_type: str, facility_size: float = 1.0) -> Dict[str, float]:
        """
        Generate waste stream quantities for a specific facility type.
        
        Args:
            facility_type: Type of facility (manufacturing, office, etc.)
            facility_size: Size multiplier for the facility
            
        Returns:
            Dictionary mapping waste stream names to daily quantities (tons)
        """
        base_rate = self.waste_config.base_generation_rates.get(facility_type, 1.0)
        composition = self.waste_config.waste_composition_profiles.get(facility_type, {
            "non_hazardous": 0.60,
            "recyclable": 0.30,
            "hazardous": 0.08,
            "organic": 0.02
        })
        
        total_daily_waste = base_rate * facility_size
        
        # Map waste types to specific streams
        waste_type_to_streams = {
            "non_hazardous": ["general_trash"],
            "recyclable": ["office_paper", "cardboard"],
            "hazardous": ["solvent_waste", "paint_waste", "batteries"],
            "organic": ["food_waste"],
            "e_waste": ["electronics"],
            "construction": ["construction_debris"],
            "chemical": ["chemical_waste"]
        }
        
        waste_quantities = {}
        
        for waste_type, percentage in composition.items():
            type_total = total_daily_waste * percentage
            streams = waste_type_to_streams.get(waste_type, ["general_trash"])
            
            # Distribute evenly among streams of this type
            per_stream = type_total / len(streams)
            
            for stream in streams:
                if stream in self.waste_streams:
                    waste_quantities[stream] = per_stream
        
        return waste_quantities
    
    def apply_seasonal_variations(self, waste_quantities: Dict[str, float], date: datetime) -> Dict[str, float]:
        """
        Apply seasonal variations to waste quantities.
        
        Args:
            waste_quantities: Base waste quantities
            date: Date for seasonal calculation
            
        Returns:
            Seasonally adjusted waste quantities
        """
        adjusted_quantities = waste_quantities.copy()
        month = date.month
        
        for stream_name, quantity in waste_quantities.items():
            stream = self.waste_streams.get(stream_name)
            if stream:
                waste_type_str = stream.waste_type.value
                seasonal_factors = self.waste_config.seasonal_waste_factors.get(waste_type_str, {})
                
                if seasonal_factors:
                    factor = seasonal_factors.get(month, 1.0)
                    adjusted_quantities[stream_name] = quantity * factor
        
        return adjusted_quantities
    
    def calculate_recycling_and_diversion(self, waste_quantities: Dict[str, float], date: datetime) -> Dict[str, Dict[str, float]]:
        """
        Calculate recycling and diversion rates for each waste stream.
        
        Args:
            waste_quantities: Daily waste quantities
            date: Date for calculation
            
        Returns:
            Dictionary with recycling/diversion data for each stream
        """
        results = {}
        
        for stream_name, quantity in waste_quantities.items():
            stream = self.waste_streams.get(stream_name)
            if not stream:
                continue
                
            waste_type_str = stream.waste_type.value
            base_recycling_rate = self.waste_config.base_recycling_rates.get(waste_type_str, 0.0)
            
            # Add seasonal variation to recycling rates
            seasonal_variation = 0.05 * math.sin(2 * math.pi * date.timetuple().tm_yday / 365.25)
            actual_recycling_rate = min(0.95, max(0.0, base_recycling_rate + seasonal_variation))
            
            # Account for contamination
            if stream.waste_type == WasteType.RECYCLABLE:
                contamination_impact = 1 - self.waste_config.contamination_rate
                actual_recycling_rate *= contamination_impact
            
            recycled_quantity = quantity * actual_recycling_rate
            disposed_quantity = quantity - recycled_quantity
            
            # Calculate diversion rate (percentage not going to landfill)
            diversion_rate = actual_recycling_rate
            if stream.disposal_method in [DisposalMethod.COMPOSTING, DisposalMethod.WASTE_TO_ENERGY]:
                diversion_rate = min(0.95, diversion_rate + 0.1)
            
            results[stream_name] = {
                "total_quantity": quantity,
                "recycled_quantity": recycled_quantity,
                "disposed_quantity": disposed_quantity,
                "recycling_rate": actual_recycling_rate,
                "diversion_rate": diversion_rate,
                "landfill_quantity": disposed_quantity * (1 - diversion_rate)
            }
        
        return results
    
    def generate_manifest_data(self, hazardous_waste: Dict[str, float], date: datetime) -> List[Dict[str, Any]]:
        """
        Generate hazardous waste manifest data.
        
        Args:
            hazardous_waste: Dictionary of hazardous waste streams and quantities
            date: Generation date
            
        Returns:
            List of manifest records
        """
        manifests = []
        
        for stream_name, quantity in hazardous_waste.items():
            stream = self.waste_streams.get(stream_name)
            if not stream or quantity < self.waste_config.manifest_required_threshold:
                continue
            
            # Determine disposal facility
            disposal_method = stream.disposal_method.value
            facility_key = "incinerator" if disposal_method == "incineration" else "landfill"
            if disposal_method in ["treatment", "neutralization"]:
                facility_key = "incinerator"  # Use treatment facility
                
            facility = self.disposal_facilities.get(facility_key, self.disposal_facilities["landfill"])
            
            manifest_number = f"M{date.strftime('%Y%m%d')}{self.manifest_counter:03d}"
            self.manifest_counter += 1
            
            manifest = {
                "manifest_number": manifest_number,
                "generator_id": self.generator_id,
                "epa_id": self.epa_id_number,
                "date_generated": date.strftime("%Y-%m-%d"),
                "waste_stream": stream_name,
                "waste_description": stream.name,
                "epa_waste_codes": [stream.epa_waste_code] if stream.epa_waste_code else [],
                "hazard_class": stream.hazard_class,
                "physical_state": stream.physical_state,
                "quantity": round(quantity, 3),
                "unit": "tons",
                "container_type": "drums" if stream.physical_state == "liquid" else "boxes",
                "disposal_facility": facility,
                "disposal_method": disposal_method,
                "transport_date": (date + timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                "disposal_date": (date + timedelta(days=random.randint(7, 30))).strftime("%Y-%m-%d"),
                "regulatory_requirements": stream.regulatory_requirements,
                "certification": f"I hereby declare that the contents of this shipment are fully and accurately described and properly classified, packed, marked, and labeled.",
                "generator_signature": f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            manifests.append(manifest)
        
        return manifests
    
    def calculate_disposal_costs(self, waste_data: Dict[str, Dict[str, float]], date: datetime) -> Dict[str, Any]:
        """
        Calculate disposal costs by waste stream and method.
        
        Args:
            waste_data: Waste stream data with quantities
            date: Date for cost calculation
            
        Returns:
            Dictionary with detailed cost breakdown
        """
        total_costs = 0.0
        cost_breakdown = {}
        revenue_breakdown = {}
        
        for stream_name, stream_data in waste_data.items():
            stream = self.waste_streams.get(stream_name)
            if not stream:
                continue
            
            disposed_quantity = stream_data["disposed_quantity"]
            recycled_quantity = stream_data["recycled_quantity"]
            
            # Disposal costs
            disposal_cost = disposed_quantity * stream.disposal_cost_per_ton
            total_costs += disposal_cost
            
            cost_breakdown[stream_name] = {
                "quantity": disposed_quantity,
                "cost_per_ton": stream.disposal_cost_per_ton,
                "total_cost": disposal_cost,
                "disposal_method": stream.disposal_method.value
            }
            
            # Recycling revenue
            if recycled_quantity > 0 and stream.recycling_value_per_ton > 0:
                revenue = recycled_quantity * stream.recycling_value_per_ton
                revenue_breakdown[stream_name] = {
                    "quantity": recycled_quantity,
                    "value_per_ton": stream.recycling_value_per_ton,
                    "total_revenue": revenue
                }
                total_costs -= revenue  # Revenue reduces net cost
        
        # Calculate cost per ton metrics
        total_waste = sum(data["total_quantity"] for data in waste_data.values())
        avg_cost_per_ton = total_costs / total_waste if total_waste > 0 else 0
        
        return {
            "date": date.strftime("%Y-%m-%d"),
            "total_net_cost": round(total_costs, 2),
            "total_waste_tons": round(total_waste, 3),
            "average_cost_per_ton": round(avg_cost_per_ton, 2),
            "cost_breakdown": cost_breakdown,
            "revenue_breakdown": revenue_breakdown,
            "disposal_method_summary": self._summarize_disposal_methods(waste_data)
        }
    
    def _summarize_disposal_methods(self, waste_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Summarize waste quantities by disposal method."""
        method_summary = {}
        
        for stream_name, stream_data in waste_data.items():
            stream = self.waste_streams.get(stream_name)
            if stream:
                method = stream.disposal_method.value
                if method not in method_summary:
                    method_summary[method] = 0.0
                method_summary[method] += stream_data["total_quantity"]
        
        return {method: round(quantity, 3) for method, quantity in method_summary.items()}
    
    def calculate_environmental_impact(self, waste_data: Dict[str, Dict[str, float]], date: datetime) -> Dict[str, Any]:
        """
        Calculate environmental impact metrics.
        
        Args:
            waste_data: Waste stream data with quantities
            date: Date for calculation
            
        Returns:
            Environmental impact metrics
        """
        total_co2_impact = 0.0
        impact_breakdown = {}
        
        for stream_name, stream_data in waste_data.items():
            stream = self.waste_streams.get(stream_name)
            if not stream:
                continue
            
            disposed_quantity = stream_data["disposed_quantity"]
            recycled_quantity = stream_data["recycled_quantity"]
            
            # CO2 impact from disposal
            disposal_co2 = disposed_quantity * abs(stream.carbon_factor)
            if stream.carbon_factor > 0:  # Positive = emissions
                total_co2_impact += disposal_co2
            
            # CO2 savings from recycling
            recycling_co2_savings = recycled_quantity * abs(stream.carbon_factor)
            if stream.carbon_factor < 0:  # Negative = savings
                total_co2_impact -= recycling_co2_savings
            
            impact_breakdown[stream_name] = {
                "disposal_co2_kg": round(disposal_co2, 2),
                "recycling_co2_savings_kg": round(recycling_co2_savings, 2),
                "net_co2_impact_kg": round(disposal_co2 - recycling_co2_savings, 2)
            }
        
        # Calculate diversion metrics
        total_waste = sum(data["total_quantity"] for data in waste_data.values())
        total_diverted = sum(data["total_quantity"] - data["landfill_quantity"] for data in waste_data.values())
        diversion_rate = total_diverted / total_waste if total_waste > 0 else 0
        
        return {
            "date": date.strftime("%Y-%m-%d"),
            "total_co2_impact_kg": round(total_co2_impact, 2),
            "total_co2_impact_tons": round(total_co2_impact / 1000, 4),
            "landfill_diversion_rate": round(diversion_rate, 3),
            "total_waste_diverted_tons": round(total_diverted, 3),
            "impact_breakdown": impact_breakdown
        }
    
    def check_regulatory_compliance(self, waste_data: Dict[str, Dict[str, float]], manifests: List[Dict[str, Any]], date: datetime) -> Dict[str, Any]:
        """
        Check regulatory compliance for waste generation and disposal.
        
        Args:
            waste_data: Waste stream data
            manifests: Manifest records
            date: Date for compliance check
            
        Returns:
            Compliance status and any issues
        """
        compliance_status = "COMPLIANT"
        issues = []
        
        # Check manifest requirements
        hazardous_streams = []
        for stream_name, stream_data in waste_data.items():
            stream = self.waste_streams.get(stream_name)
            if stream and stream.waste_type in [WasteType.HAZARDOUS, WasteType.CHEMICAL]:
                if stream_data["total_quantity"] >= self.waste_config.manifest_required_threshold:
                    hazardous_streams.append(stream_name)
        
        # Check if all required manifests are present
        manifest_streams = [m["waste_stream"] for m in manifests]
        for stream in hazardous_streams:
            if stream not in manifest_streams:
                issues.append(f"Missing manifest for hazardous waste stream: {stream}")
                compliance_status = "NON_COMPLIANT"
        
        # Simulate occasional compliance issues
        if random.random() < self.waste_config.compliance_failure_rate:
            minor_issues = [
                "Late manifest submission",
                "Incomplete waste characterization",
                "Minor labeling deficiency",
                "Missing training record",
                "Outdated disposal facility permit"
            ]
            issues.append(random.choice(minor_issues))
            if compliance_status == "COMPLIANT":
                compliance_status = "MINOR_ISSUES"
        
        # Calculate compliance metrics
        total_manifests = len(manifests)
        compliant_manifests = total_manifests - len([i for i in issues if "manifest" in i.lower()])
        
        return {
            "date": date.strftime("%Y-%m-%d"),
            "compliance_status": compliance_status,
            "issues": issues,
            "total_manifests": total_manifests,
            "compliant_manifests": compliant_manifests,
            "compliance_rate": compliant_manifests / total_manifests if total_manifests > 0 else 1.0,
            "next_audit_date": (date + timedelta(days=self.waste_config.audit_frequency_days)).strftime("%Y-%m-%d")
        }
    
    def generate(self, facility_type: str = "manufacturing", facility_size: float = 1.0, 
                 num_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive waste data for the specified period.
        
        Args:
            facility_type: Type of facility generating waste
            facility_size: Size multiplier for the facility
            num_days: Number of days to generate data for (uses config date range if None)
            
        Returns:
            Dictionary containing comprehensive waste generation data
        """
        if not self.validate_date_range():
            raise ValueError("Invalid date range configuration")
        
        # Determine date range
        if num_days:
            dates = [self.config.start_date + timedelta(days=i) for i in range(num_days)]
        else:
            dates = self.get_date_range()
        
        # Initialize results
        daily_waste_data = []
        all_manifests = []
        cost_summary = {"total_cost": 0.0, "total_revenue": 0.0}
        environmental_summary = {"total_co2_kg": 0.0, "total_diverted_tons": 0.0}
        compliance_summary = {"total_issues": 0, "compliant_days": 0}
        
        self.logger.info(f"Generating waste data for {len(dates)} days, facility type: {facility_type}")
        
        # Generate base waste stream quantities for this facility
        base_waste_quantities = self.generate_waste_streams_for_facility(facility_type, facility_size)
        
        for date in dates:
            # Apply seasonal and random variations
            daily_quantities = self.apply_seasonal_variations(base_waste_quantities, date)
            
            # Add noise to quantities
            for stream_name in daily_quantities:
                base_value = daily_quantities[stream_name]
                noise = self.generate_noise(base_value)[0] if isinstance(base_value, (int, float)) else self.generate_noise([base_value])[0]
                daily_quantities[stream_name] = max(0, base_value + noise)
            
            # Calculate recycling and diversion
            recycling_data = self.calculate_recycling_and_diversion(daily_quantities, date)
            
            # Generate manifests for hazardous waste
            hazardous_waste = {
                name: data["total_quantity"] 
                for name, data in recycling_data.items() 
                if self.waste_streams.get(name) and 
                   self.waste_streams[name].waste_type in [WasteType.HAZARDOUS, WasteType.CHEMICAL]
            }
            manifests = self.generate_manifest_data(hazardous_waste, date)
            all_manifests.extend(manifests)
            
            # Calculate costs
            costs = self.calculate_disposal_costs(recycling_data, date)
            cost_summary["total_cost"] += costs["total_net_cost"]
            
            # Calculate environmental impact
            env_impact = self.calculate_environmental_impact(recycling_data, date)
            environmental_summary["total_co2_kg"] += env_impact["total_co2_impact_kg"]
            environmental_summary["total_diverted_tons"] += env_impact["total_waste_diverted_tons"]
            
            # Check compliance
            compliance = self.check_regulatory_compliance(recycling_data, manifests, date)
            compliance_summary["total_issues"] += len(compliance["issues"])
            if compliance["compliance_status"] == "COMPLIANT":
                compliance_summary["compliant_days"] += 1
            
            # Compile daily data
            daily_data = {
                "date": date.strftime("%Y-%m-%d"),
                "facility_type": facility_type,
                "facility_size": facility_size,
                "waste_streams": recycling_data,
                "manifests": manifests,
                "costs": costs,
                "environmental_impact": env_impact,
                "compliance": compliance,
                "total_waste_generated": sum(data["total_quantity"] for data in recycling_data.values()),
                "total_waste_recycled": sum(data["recycled_quantity"] for data in recycling_data.values()),
                "overall_recycling_rate": sum(data["recycled_quantity"] for data in recycling_data.values()) / 
                                        sum(data["total_quantity"] for data in recycling_data.values()) 
                                        if sum(data["total_quantity"] for data in recycling_data.values()) > 0 else 0
            }
            
            daily_waste_data.append(daily_data)
        
        # Calculate summary statistics
        total_days = len(dates)
        avg_daily_waste = sum(day["total_waste_generated"] for day in daily_waste_data) / total_days
        avg_recycling_rate = sum(day["overall_recycling_rate"] for day in daily_waste_data) / total_days
        
        summary_stats = {
            "total_days": total_days,
            "average_daily_waste_tons": round(avg_daily_waste, 3),
            "total_waste_generated_tons": round(sum(day["total_waste_generated"] for day in daily_waste_data), 2),
            "average_recycling_rate": round(avg_recycling_rate, 3),
            "total_manifests_generated": len(all_manifests),
            "total_disposal_cost": round(cost_summary["total_cost"], 2),
            "total_co2_impact_kg": round(environmental_summary["total_co2_kg"], 2),
            "total_waste_diverted_tons": round(environmental_summary["total_diverted_tons"], 2),
            "compliance_rate": round(compliance_summary["compliant_days"] / total_days, 3),
            "total_compliance_issues": compliance_summary["total_issues"]
        }
        
        # Compile final results
        results = {
            "metadata": self.get_generation_metadata(),
            "summary": summary_stats,
            "facility_profile": {
                "type": facility_type,
                "size_multiplier": facility_size,
                "generator_id": self.generator_id,
                "epa_id": self.epa_id_number
            },
            "waste_streams_config": {name: {
                "type": stream.waste_type.value,
                "category": stream.category.value,
                "disposal_method": stream.disposal_method.value
            } for name, stream in self.waste_streams.items()},
            "daily_data": daily_waste_data,
            "all_manifests": all_manifests,
            "regulatory_compliance": {
                "generator_id": self.generator_id,
                "epa_id_number": self.epa_id_number,
                "disposal_facilities": self.disposal_facilities,
                "compliance_rate": summary_stats["compliance_rate"],
                "total_issues": compliance_summary["total_issues"]
            }
        }
        
        self.logger.info(f"Generated waste data: {summary_stats['total_waste_generated_tons']} tons over {total_days} days")
        
        return results

    def get_waste_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific waste stream."""
        stream = self.waste_streams.get(stream_name)
        if not stream:
            return {}
        
        return {
            "name": stream.name,
            "type": stream.waste_type.value,
            "category": stream.category.value,
            "epa_waste_code": stream.epa_waste_code,
            "hazard_class": stream.hazard_class,
            "physical_state": stream.physical_state,
            "disposal_method": stream.disposal_method.value,
            "disposal_cost_per_ton": stream.disposal_cost_per_ton,
            "recycling_value_per_ton": stream.recycling_value_per_ton,
            "carbon_factor": stream.carbon_factor,
            "regulatory_requirements": stream.regulatory_requirements
        }
    
    def list_available_waste_streams(self) -> List[str]:
        """Get list of all available waste stream names."""
        return list(self.waste_streams.keys())
    
    def get_facility_waste_profile(self, facility_type: str) -> Dict[str, Any]:
        """Get the waste generation profile for a facility type."""
        base_rate = self.waste_config.base_generation_rates.get(facility_type, 1.0)
        composition = self.waste_config.waste_composition_profiles.get(facility_type, {})
        
        return {
            "facility_type": facility_type,
            "base_generation_rate_tons_per_day": base_rate,
            "waste_composition": composition,
            "primary_waste_streams": [
                stream_name for stream_name, stream in self.waste_streams.items()
                if stream.waste_type.value in composition
            ]
        }