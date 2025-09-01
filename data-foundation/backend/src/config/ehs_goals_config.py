"""
EHS Annual Goals Configuration Module

This module defines the annual reduction targets for Environmental Health & Safety (EHS)
metrics including CO2 emissions (from electricity consumption), water consumption, and
waste generation for Algonquin Illinois and Houston Texas sites.

Based on specifications from EXECUTIVE_DASHBOARD_LOGIC.md
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class SiteLocation(Enum):
    """Site location enumeration"""
    ALGONQUIN = "algonquin_illinois"
    HOUSTON = "houston_texas"


class EHSCategory(Enum):
    """EHS category enumeration"""
    CO2 = "co2_emissions"
    WATER = "water_consumption"
    WASTE = "waste_generation"


@dataclass
class EHSGoal:
    """EHS goal data structure"""
    site: SiteLocation
    category: EHSCategory
    reduction_percentage: float
    baseline_year: int
    target_year: int
    unit: str
    description: str


class EHSGoalsConfig:
    """
    EHS Goals Configuration Manager
    
    Manages annual reduction targets for CO2 emissions, water consumption,
    and waste generation across different sites.
    """
    
    # Configuration constants
    BASELINE_YEAR = 2024
    TARGET_YEAR = 2025
    
    # Site-specific goals configuration
    _GOALS_CONFIG = {
        SiteLocation.ALGONQUIN: {
            EHSCategory.CO2: {
                "reduction_percentage": 15.0,
                "unit": "tonnes CO2e",
                "description": "CO2 emissions reduction from electricity consumption"
            },
            EHSCategory.WATER: {
                "reduction_percentage": 12.0,
                "unit": "gallons",
                "description": "Water consumption reduction"
            },
            EHSCategory.WASTE: {
                "reduction_percentage": 10.0,
                "unit": "pounds",
                "description": "Waste generation reduction"
            }
        },
        SiteLocation.HOUSTON: {
            EHSCategory.CO2: {
                "reduction_percentage": 18.0,
                "unit": "tonnes CO2e",
                "description": "CO2 emissions reduction from electricity consumption"
            },
            EHSCategory.WATER: {
                "reduction_percentage": 10.0,
                "unit": "gallons",
                "description": "Water consumption reduction"
            },
            EHSCategory.WASTE: {
                "reduction_percentage": 8.0,
                "unit": "pounds",
                "description": "Waste generation reduction"
            }
        }
    }
    
    @classmethod
    def get_goal(self, site: Union[SiteLocation, str], category: Union[EHSCategory, str]) -> Optional[EHSGoal]:
        """
        Get specific EHS goal for a site and category
        
        Args:
            site: Site location (SiteLocation enum or string)
            category: EHS category (EHSCategory enum or string)
            
        Returns:
            EHSGoal object or None if not found
        """
        # Convert string inputs to enums if necessary
        if isinstance(site, str):
            site = self._string_to_site(site)
        if isinstance(category, str):
            category = self._string_to_category(category)
            
        if site is None or category is None:
            return None
            
        try:
            goal_config = self._GOALS_CONFIG[site][category]
            return EHSGoal(
                site=site,
                category=category,
                reduction_percentage=goal_config["reduction_percentage"],
                baseline_year=self.BASELINE_YEAR,
                target_year=self.TARGET_YEAR,
                unit=goal_config["unit"],
                description=goal_config["description"]
            )
        except KeyError:
            return None
    
    @classmethod
    def get_goals_by_site(self, site: Union[SiteLocation, str]) -> List[EHSGoal]:
        """
        Get all EHS goals for a specific site
        
        Args:
            site: Site location (SiteLocation enum or string)
            
        Returns:
            List of EHSGoal objects for the site
        """
        if isinstance(site, str):
            site = self._string_to_site(site)
            
        if site is None:
            return []
            
        goals = []
        for category in EHSCategory:
            goal = self.get_goal(site, category)
            if goal:
                goals.append(goal)
                
        return goals
    
    @classmethod
    def get_goals_by_category(self, category: Union[EHSCategory, str]) -> List[EHSGoal]:
        """
        Get all EHS goals for a specific category across all sites
        
        Args:
            category: EHS category (EHSCategory enum or string)
            
        Returns:
            List of EHSGoal objects for the category
        """
        if isinstance(category, str):
            category = self._string_to_category(category)
            
        if category is None:
            return []
            
        goals = []
        for site in SiteLocation:
            goal = self.get_goal(site, category)
            if goal:
                goals.append(goal)
                
        return goals
    
    @classmethod
    def get_all_goals(self) -> List[EHSGoal]:
        """
        Get all EHS goals for all sites and categories
        
        Returns:
            List of all EHSGoal objects
        """
        goals = []
        for site in SiteLocation:
            goals.extend(self.get_goals_by_site(site))
            
        return goals
    
    @classmethod
    def get_reduction_percentage(self, site: Union[SiteLocation, str], category: Union[EHSCategory, str]) -> Optional[float]:
        """
        Get reduction percentage for a specific site and category
        
        Args:
            site: Site location (SiteLocation enum or string)
            category: EHS category (EHSCategory enum or string)
            
        Returns:
            Reduction percentage as float or None if not found
        """
        goal = self.get_goal(site, category)
        return goal.reduction_percentage if goal else None
    
    @classmethod
    def get_site_names(self) -> List[str]:
        """
        Get list of all site names
        
        Returns:
            List of site name strings
        """
        return [site.value for site in SiteLocation]
    
    @classmethod
    def get_category_names(self) -> List[str]:
        """
        Get list of all category names
        
        Returns:
            List of category name strings
        """
        return [category.value for category in EHSCategory]
    
    @classmethod
    def get_goals_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of all goals as nested dictionary
        
        Returns:
            Dictionary with site -> category -> reduction_percentage structure
        """
        summary = {}
        for site in SiteLocation:
            summary[site.value] = {}
            for category in EHSCategory:
                goal = self.get_goal(site, category)
                if goal:
                    summary[site.value][category.value] = goal.reduction_percentage
                    
        return summary
    
    @classmethod
    def validate_configuration(self) -> bool:
        """
        Validate that all required goals are configured
        
        Returns:
            True if all goals are properly configured, False otherwise
        """
        try:
            for site in SiteLocation:
                for category in EHSCategory:
                    goal = self.get_goal(site, category)
                    if goal is None:
                        return False
                    if goal.reduction_percentage <= 0 or goal.reduction_percentage >= 100:
                        return False
            return True
        except Exception:
            return False
    
    @staticmethod
    def _string_to_site(site_str: str) -> Optional[SiteLocation]:
        """Convert string to SiteLocation enum"""
        site_mapping = {
            "algonquin": SiteLocation.ALGONQUIN,
            "algonquin_illinois": SiteLocation.ALGONQUIN,
            "algonquin illinois": SiteLocation.ALGONQUIN,
            "houston": SiteLocation.HOUSTON,
            "houston_texas": SiteLocation.HOUSTON,
            "houston texas": SiteLocation.HOUSTON,
        }
        return site_mapping.get(site_str.lower().strip())
    
    @staticmethod
    def _string_to_category(category_str: str) -> Optional[EHSCategory]:
        """Convert string to EHSCategory enum"""
        category_mapping = {
            "co2": EHSCategory.CO2,
            "co2_emissions": EHSCategory.CO2,
            "co2 emissions": EHSCategory.CO2,
            "carbon": EHSCategory.CO2,
            "electricity": EHSCategory.CO2,  # CO2 from electricity
            "water": EHSCategory.WATER,
            "water_consumption": EHSCategory.WATER,
            "water consumption": EHSCategory.WATER,
            "waste": EHSCategory.WASTE,
            "waste_generation": EHSCategory.WASTE,
            "waste generation": EHSCategory.WASTE,
        }
        return category_mapping.get(category_str.lower().strip())


# Configuration instance for easy import
ehs_goals_config = EHSGoalsConfig()


# Convenience functions for common operations
def get_goal(site: Union[SiteLocation, str], category: Union[EHSCategory, str]) -> Optional[EHSGoal]:
    """Convenience function to get a specific EHS goal"""
    return ehs_goals_config.get_goal(site, category)


def get_reduction_percentage(site: Union[SiteLocation, str], category: Union[EHSCategory, str]) -> Optional[float]:
    """Convenience function to get reduction percentage"""
    return ehs_goals_config.get_reduction_percentage(site, category)


def get_all_goals() -> List[EHSGoal]:
    """Convenience function to get all EHS goals"""
    return ehs_goals_config.get_all_goals()


def get_goals_summary() -> Dict[str, Dict[str, float]]:
    """Convenience function to get goals summary"""
    return ehs_goals_config.get_goals_summary()


if __name__ == "__main__":
    # Example usage and validation
    print("EHS Goals Configuration")
    print("=" * 50)
    
    # Validate configuration
    if ehs_goals_config.validate_configuration():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has issues")
    
    # Print summary
    print("\nGoals Summary:")
    summary = get_goals_summary()
    for site, categories in summary.items():
        print(f"\n{site.replace('_', ' ').title()}:")
        for category, percentage in categories.items():
            print(f"  {category.replace('_', ' ').title()}: {percentage}% reduction")
    
    # Example goal retrieval
    print("\nExample Goal Retrieval:")
    algonquin_co2 = get_goal("algonquin", "co2")
    if algonquin_co2:
        print(f"Algonquin CO2 Goal: {algonquin_co2.reduction_percentage}% reduction ({algonquin_co2.description})")