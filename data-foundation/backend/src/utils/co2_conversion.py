"""
CO2 Conversion Engine for Electricity and Water Consumption

This module provides utilities to convert electricity consumption (kWh) and water usage
(gallons) to CO2 emissions using EPA and recognized standards.

Conversion factors are based on:
- EPA eGRID data for electricity emissions factors
- EPA water treatment/distribution emissions factors
- Regional grid mix and state-specific factors

Created: 2025-08-31
Version: 1.0.0
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported regions for CO2 conversion"""
    US_AVERAGE = "us_average"
    ILLINOIS = "illinois"
    TEXAS = "texas"
    ALGONQUIN_IL = "algonquin_il"
    HOUSTON_TX = "houston_tx"


@dataclass
class ConversionFactors:
    """Data class to hold conversion factors for a specific region"""
    electricity_kg_co2_per_kwh: float
    water_kg_co2_per_gallon: float
    region_name: str
    data_source: str
    year: int


class CO2ConversionEngine:
    """
    CO2 Conversion Engine for calculating emissions from electricity and water consumption.
    
    Uses EPA eGRID data and recognized standards for accurate regional conversions.
    """
    
    def __init__(self):
        """Initialize the conversion engine with regional factors"""
        self._conversion_factors = self._load_conversion_factors()
    
    def _load_conversion_factors(self) -> Dict[Region, ConversionFactors]:
        """
        Load conversion factors for different regions based on EPA data.
        
        Electricity factors from EPA eGRID 2022 data (kg CO2/MWh converted to kg CO2/kWh)
        Water factors from EPA water treatment/distribution studies
        
        Returns:
            Dict mapping regions to their conversion factors
        """
        return {
            Region.US_AVERAGE: ConversionFactors(
                electricity_kg_co2_per_kwh=0.386,  # EPA eGRID 2022 US average
                water_kg_co2_per_gallon=0.0029,    # EPA water treatment average
                region_name="United States Average",
                data_source="EPA eGRID 2022",
                year=2022
            ),
            Region.ILLINOIS: ConversionFactors(
                electricity_kg_co2_per_kwh=0.340,  # Illinois state average (eGRID 2022)
                water_kg_co2_per_gallon=0.0031,    # Midwest water treatment average
                region_name="Illinois State",
                data_source="EPA eGRID 2022",
                year=2022
            ),
            Region.TEXAS: ConversionFactors(
                electricity_kg_co2_per_kwh=0.390,  # Texas state average (eGRID 2022)
                water_kg_co2_per_gallon=0.0027,    # Southwest water treatment average
                region_name="Texas State",
                data_source="EPA eGRID 2022",
                year=2022
            ),
            Region.ALGONQUIN_IL: ConversionFactors(
                electricity_kg_co2_per_kwh=0.335,  # ComEd territory (Northern Illinois)
                water_kg_co2_per_gallon=0.0030,    # Northern Illinois water systems
                region_name="Algonquin, Illinois",
                data_source="EPA eGRID 2022 - ComEd Territory",
                year=2022
            ),
            Region.HOUSTON_TX: ConversionFactors(
                electricity_kg_co2_per_kwh=0.405,  # ERCOT Houston zone
                water_kg_co2_per_gallon=0.0025,    # Houston metro water systems
                region_name="Houston, Texas",
                data_source="EPA eGRID 2022 - ERCOT",
                year=2022
            )
        }
    
    def get_available_regions(self) -> Dict[str, str]:
        """
        Get a dictionary of available regions and their names.
        
        Returns:
            Dict mapping region enum values to human-readable names
        """
        return {
            region.value: factors.region_name 
            for region, factors in self._conversion_factors.items()
        }
    
    def get_conversion_factors(self, region: Union[Region, str]) -> ConversionFactors:
        """
        Get conversion factors for a specific region.
        
        Args:
            region: Region enum or string identifier
            
        Returns:
            ConversionFactors for the specified region
            
        Raises:
            ValueError: If region is not supported
        """
        if isinstance(region, str):
            try:
                region = Region(region)
            except ValueError:
                raise ValueError(f"Unsupported region: {region}. Available: {list(self.get_available_regions().keys())}")
        
        if region not in self._conversion_factors:
            raise ValueError(f"No conversion factors available for region: {region}")
        
        return self._conversion_factors[region]
    
    def electricity_to_co2_kg(
        self, 
        kwh: float, 
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> float:
        """
        Convert electricity consumption to CO2 emissions in kilograms.
        
        Args:
            kwh: Electricity consumption in kilowatt-hours
            region: Region for conversion factor (default: US average)
            
        Returns:
            CO2 emissions in kilograms
        """
        if kwh < 0:
            raise ValueError("Electricity consumption cannot be negative")
        
        factors = self.get_conversion_factors(region)
        co2_kg = kwh * factors.electricity_kg_co2_per_kwh
        
        logger.debug(f"Converted {kwh} kWh to {co2_kg:.3f} kg CO2 using {factors.region_name} factors")
        return co2_kg
    
    def electricity_to_co2_tonnes(
        self, 
        kwh: float, 
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> float:
        """
        Convert electricity consumption to CO2 emissions in tonnes.
        
        Args:
            kwh: Electricity consumption in kilowatt-hours
            region: Region for conversion factor (default: US average)
            
        Returns:
            CO2 emissions in tonnes
        """
        co2_kg = self.electricity_to_co2_kg(kwh, region)
        return co2_kg / 1000.0
    
    def water_to_co2_kg(
        self, 
        gallons: float, 
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> float:
        """
        Convert water consumption to CO2 emissions in kilograms.
        
        This accounts for indirect emissions from water treatment, pumping, and distribution.
        
        Args:
            gallons: Water consumption in gallons
            region: Region for conversion factor (default: US average)
            
        Returns:
            CO2 emissions in kilograms
        """
        if gallons < 0:
            raise ValueError("Water consumption cannot be negative")
        
        factors = self.get_conversion_factors(region)
        co2_kg = gallons * factors.water_kg_co2_per_gallon
        
        logger.debug(f"Converted {gallons} gallons to {co2_kg:.3f} kg CO2 using {factors.region_name} factors")
        return co2_kg
    
    def water_to_co2_tonnes(
        self, 
        gallons: float, 
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> float:
        """
        Convert water consumption to CO2 emissions in tonnes.
        
        Args:
            gallons: Water consumption in gallons
            region: Region for conversion factor (default: US average)
            
        Returns:
            CO2 emissions in tonnes
        """
        co2_kg = self.water_to_co2_kg(gallons, region)
        return co2_kg / 1000.0
    
    def monthly_electricity_to_co2_kg(
        self, 
        monthly_kwh: float, 
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> float:
        """
        Calculate monthly CO2 emissions from electricity consumption.
        
        Args:
            monthly_kwh: Monthly electricity consumption in kWh
            region: Region for conversion factor
            
        Returns:
            Monthly CO2 emissions in kilograms
        """
        return self.electricity_to_co2_kg(monthly_kwh, region)
    
    def annual_electricity_to_co2_kg(
        self, 
        annual_kwh: float, 
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> float:
        """
        Calculate annual CO2 emissions from electricity consumption.
        
        Args:
            annual_kwh: Annual electricity consumption in kWh
            region: Region for conversion factor
            
        Returns:
            Annual CO2 emissions in kilograms
        """
        return self.electricity_to_co2_kg(annual_kwh, region)
    
    def monthly_water_to_co2_kg(
        self, 
        monthly_gallons: float, 
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> float:
        """
        Calculate monthly CO2 emissions from water consumption.
        
        Args:
            monthly_gallons: Monthly water consumption in gallons
            region: Region for conversion factor
            
        Returns:
            Monthly CO2 emissions in kilograms
        """
        return self.water_to_co2_kg(monthly_gallons, region)
    
    def annual_water_to_co2_kg(
        self, 
        annual_gallons: float, 
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> float:
        """
        Calculate annual CO2 emissions from water consumption.
        
        Args:
            annual_gallons: Annual water consumption in gallons
            region: Region for conversion factor
            
        Returns:
            Annual CO2 emissions in kilograms
        """
        return self.water_to_co2_kg(annual_gallons, region)
    
    def combined_utility_emissions_kg(
        self,
        electricity_kwh: float,
        water_gallons: float,
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> Dict[str, float]:
        """
        Calculate combined CO2 emissions from electricity and water consumption.
        
        Args:
            electricity_kwh: Electricity consumption in kWh
            water_gallons: Water consumption in gallons
            region: Region for conversion factors
            
        Returns:
            Dictionary containing breakdown of emissions by source and total
        """
        electricity_co2 = self.electricity_to_co2_kg(electricity_kwh, region)
        water_co2 = self.water_to_co2_kg(water_gallons, region)
        total_co2 = electricity_co2 + water_co2
        
        return {
            "electricity_co2_kg": electricity_co2,
            "water_co2_kg": water_co2,
            "total_co2_kg": total_co2,
            "electricity_kwh": electricity_kwh,
            "water_gallons": water_gallons,
            "region": self.get_conversion_factors(region).region_name
        }
    
    def combined_utility_emissions_tonnes(
        self,
        electricity_kwh: float,
        water_gallons: float,
        region: Union[Region, str] = Region.US_AVERAGE
    ) -> Dict[str, float]:
        """
        Calculate combined CO2 emissions from electricity and water consumption in tonnes.
        
        Args:
            electricity_kwh: Electricity consumption in kWh
            water_gallons: Water consumption in gallons
            region: Region for conversion factors
            
        Returns:
            Dictionary containing breakdown of emissions by source and total in tonnes
        """
        emissions_kg = self.combined_utility_emissions_kg(electricity_kwh, water_gallons, region)
        
        return {
            "electricity_co2_tonnes": emissions_kg["electricity_co2_kg"] / 1000.0,
            "water_co2_tonnes": emissions_kg["water_co2_kg"] / 1000.0,
            "total_co2_tonnes": emissions_kg["total_co2_kg"] / 1000.0,
            "electricity_kwh": electricity_kwh,
            "water_gallons": water_gallons,
            "region": emissions_kg["region"]
        }


# Convenience function for quick conversions
def quick_electricity_conversion(kwh: float, region: str = "us_average") -> float:
    """
    Quick utility function to convert electricity to CO2 in kg.
    
    Args:
        kwh: Electricity consumption in kWh
        region: Region string identifier
        
    Returns:
        CO2 emissions in kilograms
    """
    engine = CO2ConversionEngine()
    return engine.electricity_to_co2_kg(kwh, region)


def quick_water_conversion(gallons: float, region: str = "us_average") -> float:
    """
    Quick utility function to convert water consumption to CO2 in kg.
    
    Args:
        gallons: Water consumption in gallons
        region: Region string identifier
        
    Returns:
        CO2 emissions in kilograms
    """
    engine = CO2ConversionEngine()
    return engine.water_to_co2_kg(gallons, region)


# Example usage
if __name__ == "__main__":
    # Initialize the conversion engine
    converter = CO2ConversionEngine()
    
    # Example conversions for Algonquin, Illinois
    print("=== Algonquin, Illinois Example ===")
    monthly_kwh = 1200  # Typical residential monthly usage
    monthly_gallons = 3000  # Typical residential monthly water usage
    
    algonquin_emissions = converter.combined_utility_emissions_kg(
        monthly_kwh, monthly_gallons, Region.ALGONQUIN_IL
    )
    print(f"Monthly emissions for {monthly_kwh} kWh and {monthly_gallons} gallons:")
    print(f"  Electricity: {algonquin_emissions['electricity_co2_kg']:.2f} kg CO2")
    print(f"  Water: {algonquin_emissions['water_co2_kg']:.2f} kg CO2")
    print(f"  Total: {algonquin_emissions['total_co2_kg']:.2f} kg CO2")
    
    # Example conversions for Houston, Texas
    print("\n=== Houston, Texas Example ===")
    houston_emissions = converter.combined_utility_emissions_kg(
        monthly_kwh, monthly_gallons, Region.HOUSTON_TX
    )
    print(f"Monthly emissions for {monthly_kwh} kWh and {monthly_gallons} gallons:")
    print(f"  Electricity: {houston_emissions['electricity_co2_kg']:.2f} kg CO2")
    print(f"  Water: {houston_emissions['water_co2_kg']:.2f} kg CO2")
    print(f"  Total: {houston_emissions['total_co2_kg']:.2f} kg CO2")
    
    # Annual calculation
    print(f"\n=== Annual Emissions (Houston) ===")
    annual_kwh = monthly_kwh * 12
    annual_gallons = monthly_gallons * 12
    annual_emissions = converter.combined_utility_emissions_tonnes(
        annual_kwh, annual_gallons, Region.HOUSTON_TX
    )
    print(f"Annual emissions for {annual_kwh} kWh and {annual_gallons} gallons:")
    print(f"  Total: {annual_emissions['total_co2_tonnes']:.2f} tonnes CO2")