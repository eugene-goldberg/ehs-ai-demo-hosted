"""
Pro-rating Calculator for EHS AI Demo
====================================

This module provides comprehensive pro-rating calculation functionality for
distributing usage and costs across billing periods with precise financial accuracy.

Features:
- Time-based pro-rating by days in billing period
- Space-based pro-rating by facility square footage
- Hybrid pro-rating combining time and space factors
- Calendar month allocation with partial month handling
- Leap year and month boundary edge case management
- Decimal precision for financial calculations

Author: EHS AI Demo Team
Created: 2025-08-23
"""

import calendar
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


class ProRatingMethod(Enum):
    """Enumeration of available pro-rating methods."""
    TIME_BASED = "time_based"
    SPACE_BASED = "space_based"
    HYBRID = "hybrid"


@dataclass
class BillingPeriod:
    """
    Represents a billing period with associated usage and cost data.
    
    Attributes:
        start_date: Beginning of the billing period
        end_date: End of the billing period
        total_usage: Total usage amount for the period (e.g., kWh, gallons)
        total_cost: Total cost for the period in currency units
        facility_id: Optional identifier for the facility
        usage_type: Optional type of usage (electricity, water, gas, etc.)
    """
    start_date: date
    end_date: date
    total_usage: Decimal
    total_cost: Decimal
    facility_id: Optional[str] = None
    usage_type: Optional[str] = None
    
    def __post_init__(self):
        """Validate billing period after initialization."""
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before or equal to end date")
        if self.total_usage < 0:
            raise ValueError("Total usage cannot be negative")
        if self.total_cost < 0:
            raise ValueError("Total cost cannot be negative")


@dataclass
class MonthlyAllocation:
    """
    Represents the allocated portion for a specific calendar month.
    
    Attributes:
        year: Calendar year
        month: Calendar month (1-12)
        days: Number of days in this month that fall within the billing period
        usage: Allocated usage amount for this month
        cost: Allocated cost for this month
        percentage: Percentage of total allocation (0.0 to 1.0)
        facility_id: Optional identifier for the facility
        usage_type: Optional type of usage
    """
    year: int
    month: int
    days: int
    usage: Decimal
    cost: Decimal
    percentage: Decimal
    facility_id: Optional[str] = None
    usage_type: Optional[str] = None
    
    @property
    def month_name(self) -> str:
        """Return the full month name."""
        return calendar.month_name[self.month]
    
    @property
    def month_year_key(self) -> str:
        """Return a string key in format 'YYYY-MM'."""
        return f"{self.year:04d}-{self.month:02d}"


@dataclass
class FacilityInfo:
    """
    Information about a facility for space-based pro-rating.
    
    Attributes:
        facility_id: Unique identifier for the facility
        square_footage: Total square footage of the facility
        occupied_percentage: Percentage of facility that is occupied (0.0 to 1.0)
        facility_name: Optional human-readable name
    """
    facility_id: str
    square_footage: Decimal
    occupied_percentage: Decimal = Decimal('1.0')
    facility_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate facility info after initialization."""
        if self.square_footage <= 0:
            raise ValueError("Square footage must be positive")
        if not 0 <= self.occupied_percentage <= 1:
            raise ValueError("Occupied percentage must be between 0 and 1")


class ProRatingCalculator:
    """
    Comprehensive pro-rating calculator for distributing usage and costs.
    
    This class provides methods for time-based, space-based, and hybrid
    pro-rating calculations with precise decimal arithmetic for financial
    accuracy.
    """
    
    def __init__(self, precision: int = 4):
        """
        Initialize the pro-rating calculator.
        
        Args:
            precision: Number of decimal places for calculations (default: 4)
        """
        self.precision = precision
        self.decimal_context = Decimal('0.1') ** precision
        
    def calculate_time_based_allocation(
        self, 
        billing_period: BillingPeriod
    ) -> List[MonthlyAllocation]:
        """
        Calculate time-based pro-rating allocation by days.
        
        Distributes usage and costs proportionally based on the number of days
        in each calendar month that fall within the billing period.
        
        Args:
            billing_period: The billing period to allocate
            
        Returns:
            List of MonthlyAllocation objects for each affected month
        """
        self.validate_billing_period(billing_period)
        
        # Calculate daily allocations
        daily_allocations = self.calculate_daily_allocations(billing_period)
        
        # Allocate to calendar months
        monthly_allocations = self.allocate_to_calendar_months(
            daily_allocations, billing_period
        )
        
        return monthly_allocations
    
    def calculate_space_based_allocation(
        self,
        billing_period: BillingPeriod,
        facility_info: FacilityInfo,
        total_facility_space: Decimal
    ) -> List[MonthlyAllocation]:
        """
        Calculate space-based pro-rating allocation by facility square footage.
        
        Distributes usage and costs based on the facility's portion of total
        space, then applies time-based allocation within that proportion.
        
        Args:
            billing_period: The billing period to allocate
            facility_info: Information about the specific facility
            total_facility_space: Total square footage across all facilities
            
        Returns:
            List of MonthlyAllocation objects adjusted for space allocation
        """
        if total_facility_space <= 0:
            raise ValueError("Total facility space must be positive")
        
        # Calculate space allocation percentage
        effective_space = facility_info.square_footage * facility_info.occupied_percentage
        space_percentage = effective_space / total_facility_space
        
        # Create adjusted billing period with space-based amounts
        adjusted_period = BillingPeriod(
            start_date=billing_period.start_date,
            end_date=billing_period.end_date,
            total_usage=billing_period.total_usage * space_percentage,
            total_cost=billing_period.total_cost * space_percentage,
            facility_id=facility_info.facility_id,
            usage_type=billing_period.usage_type
        )
        
        # Apply time-based allocation to the space-adjusted amounts
        monthly_allocations = self.calculate_time_based_allocation(adjusted_period)
        
        # Update facility_id in allocations
        for allocation in monthly_allocations:
            allocation.facility_id = facility_info.facility_id
        
        return monthly_allocations
    
    def calculate_hybrid_allocation(
        self,
        billing_period: BillingPeriod,
        facility_info: FacilityInfo,
        total_facility_space: Decimal,
        time_weight: Decimal = Decimal('0.7'),
        space_weight: Decimal = Decimal('0.3')
    ) -> List[MonthlyAllocation]:
        """
        Calculate hybrid pro-rating combining time and space factors.
        
        Creates a weighted combination of time-based and space-based allocations.
        
        Args:
            billing_period: The billing period to allocate
            facility_info: Information about the specific facility
            total_facility_space: Total square footage across all facilities
            time_weight: Weight for time-based allocation (default: 0.7)
            space_weight: Weight for space-based allocation (default: 0.3)
            
        Returns:
            List of MonthlyAllocation objects with hybrid allocation
        """
        # Validate weights
        if abs(time_weight + space_weight - Decimal('1.0')) > Decimal('0.001'):
            raise ValueError("Time weight and space weight must sum to 1.0")
        
        # Calculate both allocation types
        time_allocations = self.calculate_time_based_allocation(billing_period)
        space_allocations = self.calculate_space_based_allocation(
            billing_period, facility_info, total_facility_space
        )
        
        # Combine allocations with weights
        hybrid_allocations = []
        
        # Create lookup for space allocations
        space_lookup = {
            (alloc.year, alloc.month): alloc for alloc in space_allocations
        }
        
        for time_alloc in time_allocations:
            month_key = (time_alloc.year, time_alloc.month)
            space_alloc = space_lookup.get(month_key)
            
            if space_alloc:
                # Weighted combination
                hybrid_usage = (time_alloc.usage * time_weight + 
                              space_alloc.usage * space_weight)
                hybrid_cost = (time_alloc.cost * time_weight + 
                             space_alloc.cost * space_weight)
                hybrid_percentage = (time_alloc.percentage * time_weight + 
                                   space_alloc.percentage * space_weight)
                
                hybrid_allocation = MonthlyAllocation(
                    year=time_alloc.year,
                    month=time_alloc.month,
                    days=time_alloc.days,
                    usage=hybrid_usage.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
                    cost=hybrid_cost.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
                    percentage=hybrid_percentage.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
                    facility_id=facility_info.facility_id,
                    usage_type=billing_period.usage_type
                )
                
                hybrid_allocations.append(hybrid_allocation)
        
        return hybrid_allocations
    
    def calculate_daily_allocations(
        self, 
        billing_period: BillingPeriod
    ) -> Dict[date, Tuple[Decimal, Decimal]]:
        """
        Split usage and cost evenly across all days in the billing period.
        
        Args:
            billing_period: The billing period to split
            
        Returns:
            Dictionary mapping each date to (daily_usage, daily_cost) tuple
        """
        total_days = self.get_billing_period_days(billing_period)
        
        if total_days == 0:
            return {}
        
        daily_usage = billing_period.total_usage / Decimal(total_days)
        daily_cost = billing_period.total_cost / Decimal(total_days)
        
        daily_allocations = {}
        current_date = billing_period.start_date
        
        while current_date <= billing_period.end_date:
            daily_allocations[current_date] = (
                daily_usage.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
                daily_cost.quantize(self.decimal_context, rounding=ROUND_HALF_UP)
            )
            current_date += timedelta(days=1)
        
        return daily_allocations
    
    def allocate_to_calendar_months(
        self,
        daily_allocations: Dict[date, Tuple[Decimal, Decimal]],
        billing_period: BillingPeriod
    ) -> List[MonthlyAllocation]:
        """
        Distribute daily allocations to calendar months.
        
        Args:
            daily_allocations: Dictionary of daily usage and cost amounts
            billing_period: Original billing period for reference
            
        Returns:
            List of MonthlyAllocation objects
        """
        if not daily_allocations:
            return []
        
        # Group allocations by month
        monthly_groups: Dict[Tuple[int, int], List[Tuple[date, Decimal, Decimal]]] = {}
        
        for allocation_date, (daily_usage, daily_cost) in daily_allocations.items():
            month_key = (allocation_date.year, allocation_date.month)
            if month_key not in monthly_groups:
                monthly_groups[month_key] = []
            monthly_groups[month_key].append((allocation_date, daily_usage, daily_cost))
        
        # Create monthly allocations
        monthly_allocations = []
        total_billing_days = len(daily_allocations)
        
        for (year, month), daily_entries in monthly_groups.items():
            days_in_month = len(daily_entries)
            month_usage = sum(entry[1] for entry in daily_entries)
            month_cost = sum(entry[2] for entry in daily_entries)
            
            # Calculate percentage of total billing period
            percentage = Decimal(days_in_month) / Decimal(total_billing_days)
            
            allocation = MonthlyAllocation(
                year=year,
                month=month,
                days=days_in_month,
                usage=month_usage.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
                cost=month_cost.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
                percentage=percentage.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
                facility_id=billing_period.facility_id,
                usage_type=billing_period.usage_type
            )
            
            monthly_allocations.append(allocation)
        
        # Sort by year and month
        monthly_allocations.sort(key=lambda x: (x.year, x.month))
        
        return monthly_allocations
    
    def handle_partial_months(
        self, 
        billing_period: BillingPeriod
    ) -> Dict[str, int]:
        """
        Analyze partial month coverage at start and end of billing period.
        
        Args:
            billing_period: The billing period to analyze
            
        Returns:
            Dictionary with partial month information
        """
        result = {
            'start_month_partial_days': 0,
            'end_month_partial_days': 0,
            'start_month_total_days': 0,
            'end_month_total_days': 0,
            'has_partial_start': False,
            'has_partial_end': False
        }
        
        start_date = billing_period.start_date
        end_date = billing_period.end_date
        
        # Analyze start month
        start_month_days = self.get_days_in_month(start_date.year, start_date.month)
        if start_date.day > 1:
            result['has_partial_start'] = True
            result['start_month_partial_days'] = start_month_days - start_date.day + 1
        else:
            result['start_month_partial_days'] = start_month_days
        result['start_month_total_days'] = start_month_days
        
        # Analyze end month (only if different from start month)
        if (end_date.year, end_date.month) != (start_date.year, start_date.month):
            end_month_days = self.get_days_in_month(end_date.year, end_date.month)
            if end_date.day < end_month_days:
                result['has_partial_end'] = True
                result['end_month_partial_days'] = end_date.day
            else:
                result['end_month_partial_days'] = end_month_days
            result['end_month_total_days'] = end_month_days
        
        return result
    
    def calculate_allocation_percentage(
        self, 
        days_in_allocation: int, 
        total_billing_days: int
    ) -> Decimal:
        """
        Calculate the percentage allocation for a given number of days.
        
        Args:
            days_in_allocation: Number of days being allocated
            total_billing_days: Total days in the billing period
            
        Returns:
            Decimal percentage (0.0 to 1.0)
        """
        if total_billing_days == 0:
            return Decimal('0')
        
        percentage = Decimal(days_in_allocation) / Decimal(total_billing_days)
        return percentage.quantize(self.decimal_context, rounding=ROUND_HALF_UP)
    
    def get_days_in_month(self, year: int, month: int) -> int:
        """
        Get the number of days in a specific month, accounting for leap years.
        
        Args:
            year: Calendar year
            month: Calendar month (1-12)
            
        Returns:
            Number of days in the month
        """
        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12")
        
        return calendar.monthrange(year, month)[1]
    
    def get_billing_period_days(self, billing_period: BillingPeriod) -> int:
        """
        Calculate the total number of days in a billing period.
        
        Args:
            billing_period: The billing period to calculate
            
        Returns:
            Total number of days (inclusive of start and end dates)
        """
        return (billing_period.end_date - billing_period.start_date).days + 1
    
    def validate_billing_period(self, billing_period: BillingPeriod) -> None:
        """
        Validate that a billing period is valid for calculations.
        
        Args:
            billing_period: The billing period to validate
            
        Raises:
            ValueError: If the billing period is invalid
        """
        if billing_period.start_date > billing_period.end_date:
            raise ValueError("Start date must be before or equal to end date")
        
        if billing_period.total_usage < 0:
            raise ValueError("Total usage cannot be negative")
        
        if billing_period.total_cost < 0:
            raise ValueError("Total cost cannot be negative")
        
        # Check for reasonable date range (not more than 2 years)
        max_days = 2 * 365
        if self.get_billing_period_days(billing_period) > max_days:
            raise ValueError(f"Billing period too long (max {max_days} days)")
    
    def is_leap_year(self, year: int) -> bool:
        """
        Determine if a year is a leap year.
        
        Args:
            year: Calendar year to check
            
        Returns:
            True if the year is a leap year, False otherwise
        """
        return calendar.isleap(year)
    
    def get_month_boundaries(self, billing_period: BillingPeriod) -> List[Tuple[int, int]]:
        """
        Get all (year, month) combinations covered by the billing period.
        
        Args:
            billing_period: The billing period to analyze
            
        Returns:
            List of (year, month) tuples in chronological order
        """
        boundaries = []
        current_date = billing_period.start_date.replace(day=1)
        end_month = billing_period.end_date.replace(day=1)
        
        while current_date <= end_month:
            boundaries.append((current_date.year, current_date.month))
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return boundaries
    
    def summarize_allocation(self, allocations: List[MonthlyAllocation]) -> Dict[str, Decimal]:
        """
        Create a summary of the allocation results.
        
        Args:
            allocations: List of monthly allocations to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        if not allocations:
            return {
                'total_usage': Decimal('0'),
                'total_cost': Decimal('0'),
                'total_days': 0,
                'total_percentage': Decimal('0'),
                'number_of_months': 0
            }
        
        total_usage = sum(alloc.usage for alloc in allocations)
        total_cost = sum(alloc.cost for alloc in allocations)
        total_days = sum(alloc.days for alloc in allocations)
        total_percentage = sum(alloc.percentage for alloc in allocations)
        
        return {
            'total_usage': total_usage.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
            'total_cost': total_cost.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
            'total_days': total_days,
            'total_percentage': total_percentage.quantize(self.decimal_context, rounding=ROUND_HALF_UP),
            'number_of_months': len(allocations),
            'average_monthly_usage': (total_usage / len(allocations)).quantize(self.decimal_context, rounding=ROUND_HALF_UP),
            'average_monthly_cost': (total_cost / len(allocations)).quantize(self.decimal_context, rounding=ROUND_HALF_UP)
        }


def create_sample_billing_period() -> BillingPeriod:
    """
    Create a sample billing period for testing and demonstration.
    
    Returns:
        BillingPeriod with sample data
    """
    return BillingPeriod(
        start_date=date(2024, 1, 15),
        end_date=date(2024, 3, 10),
        total_usage=Decimal('1000.00'),
        total_cost=Decimal('250.00'),
        facility_id="FAC001",
        usage_type="electricity"
    )


def create_sample_facility_info() -> FacilityInfo:
    """
    Create sample facility information for testing and demonstration.
    
    Returns:
        FacilityInfo with sample data
    """
    return FacilityInfo(
        facility_id="FAC001",
        square_footage=Decimal('50000'),
        occupied_percentage=Decimal('0.85'),
        facility_name="Main Office Building"
    )


if __name__ == "__main__":
    """
    Example usage and demonstration of the ProRatingCalculator.
    """
    # Create calculator instance
    calculator = ProRatingCalculator(precision=4)
    
    # Create sample data
    billing_period = create_sample_billing_period()
    facility_info = create_sample_facility_info()
    
    print("Pro-rating Calculator Demonstration")
    print("=" * 50)
    print(f"Billing Period: {billing_period.start_date} to {billing_period.end_date}")
    print(f"Total Usage: {billing_period.total_usage}")
    print(f"Total Cost: {billing_period.total_cost}")
    print()
    
    # Time-based allocation
    print("Time-based Allocation:")
    print("-" * 30)
    time_allocations = calculator.calculate_time_based_allocation(billing_period)
    
    for allocation in time_allocations:
        print(f"{allocation.month_name} {allocation.year}: "
              f"{allocation.days} days, "
              f"Usage: {allocation.usage}, "
              f"Cost: {allocation.cost}, "
              f"Percentage: {allocation.percentage:.2%}")
    
    print()
    time_summary = calculator.summarize_allocation(time_allocations)
    print(f"Time-based Summary: {time_summary}")
    print()
    
    # Space-based allocation
    print("Space-based Allocation:")
    print("-" * 30)
    total_space = Decimal('200000')  # Total space across all facilities
    space_allocations = calculator.calculate_space_based_allocation(
        billing_period, facility_info, total_space
    )
    
    for allocation in space_allocations:
        print(f"{allocation.month_name} {allocation.year}: "
              f"{allocation.days} days, "
              f"Usage: {allocation.usage}, "
              f"Cost: {allocation.cost}, "
              f"Percentage: {allocation.percentage:.2%}")
    
    print()
    space_summary = calculator.summarize_allocation(space_allocations)
    print(f"Space-based Summary: {space_summary}")