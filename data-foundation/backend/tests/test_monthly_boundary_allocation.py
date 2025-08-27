"""
Test Suite for Monthly Boundary Allocation in ProRatingCalculator
==============================================================

Comprehensive tests to verify that the ProRatingCalculator properly handles
monthly boundary allocations for billing periods that span multiple months.

Test Coverage:
- Single month bills (100% allocation to that month)
- Bills spanning 2 months with different day counts
- Bills spanning 3+ months
- Bills crossing year boundaries
- Bills in leap year February
- Bills starting mid-month and ending mid-month
- Exact day count calculations
- Usage and cost proportional allocation
- Edge cases and boundary conditions

Author: EHS AI Demo Team
Created: 2025-08-26
"""

import pytest
import calendar
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

# Import the classes we're testing
from src.phase1_enhancements.prorating_calculator import (
    ProRatingCalculator,
    BillingPeriod,
    MonthlyAllocation
)


class TestMonthlyBoundaryAllocation:
    """Test class for monthly boundary allocation functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.calculator = ProRatingCalculator(precision=4)
        self.test_usage = Decimal('1000.0000')
        self.test_cost = Decimal('250.0000')
    
    def test_single_month_bill_allocation(self):
        """Test that bills within a single month allocate 100% to that month."""
        # Test case: January 15-25, 2024 (11 days)
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 25),
            total_usage=self.test_usage,
            total_cost=self.test_cost,
            facility_id="TEST_FAC",
            usage_type="electricity"
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly one allocation
        assert len(allocations) == 1
        
        allocation = allocations[0]
        assert allocation.year == 2024
        assert allocation.month == 1
        assert allocation.days == 11  # Jan 15-25 inclusive
        assert allocation.usage == self.test_usage
        assert allocation.cost == self.test_cost
        assert allocation.percentage == Decimal('1.0000')
        assert allocation.facility_id == "TEST_FAC"
        assert allocation.usage_type == "electricity"
    
    def test_two_month_bill_equal_days(self):
        """Test bills spanning exactly 2 months with equal day distribution."""
        # Test case: January 16 - February 15, 2024 (31 days total, ~15.5 days each month)
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 16),
            end_date=date(2024, 2, 15),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly two allocations
        assert len(allocations) == 2
        
        # January allocation (Jan 16-31 = 16 days)
        jan_allocation = next(alloc for alloc in allocations if alloc.month == 1)
        assert jan_allocation.year == 2024
        assert jan_allocation.days == 16
        
        # February allocation (Feb 1-15 = 15 days)  
        feb_allocation = next(alloc for alloc in allocations if alloc.month == 2)
        assert feb_allocation.year == 2024
        assert feb_allocation.days == 15
        
        # Verify total days add up correctly
        total_days = jan_allocation.days + feb_allocation.days
        assert total_days == 31
        
        # Verify proportional allocation
        jan_expected_usage = self.test_usage * Decimal('16') / Decimal('31')
        feb_expected_usage = self.test_usage * Decimal('15') / Decimal('31')
        
        assert abs(jan_allocation.usage - jan_expected_usage.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        assert abs(feb_allocation.usage - feb_expected_usage.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        
        # Verify total usage preservation
        total_usage = jan_allocation.usage + feb_allocation.usage
        assert abs(total_usage - self.test_usage) < Decimal('0.0001')
        
        # Verify total cost preservation
        total_cost = jan_allocation.cost + feb_allocation.cost
        assert abs(total_cost - self.test_cost) < Decimal('0.0001')
    
    def test_three_month_bill_allocation(self):
        """Test bills spanning 3 months with different day counts."""
        # Test case: January 20 - March 10, 2024
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 20),
            end_date=date(2024, 3, 10),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly three allocations
        assert len(allocations) == 3
        
        # January allocation (Jan 20-31 = 12 days)
        jan_allocation = next(alloc for alloc in allocations if alloc.month == 1)
        assert jan_allocation.year == 2024
        assert jan_allocation.days == 12
        
        # February allocation (Feb 1-29 = 29 days, 2024 is leap year)
        feb_allocation = next(alloc for alloc in allocations if alloc.month == 2)
        assert feb_allocation.year == 2024
        assert feb_allocation.days == 29
        
        # March allocation (Mar 1-10 = 10 days)
        mar_allocation = next(alloc for alloc in allocations if alloc.month == 3)
        assert mar_allocation.year == 2024
        assert mar_allocation.days == 10
        
        # Verify total days (12 + 29 + 10 = 51)
        total_days = sum(alloc.days for alloc in allocations)
        expected_total_days = (date(2024, 3, 10) - date(2024, 1, 20)).days + 1
        assert total_days == expected_total_days == 51
        
        # Verify proportional allocation
        for allocation in allocations:
            expected_percentage = Decimal(allocation.days) / Decimal(total_days)
            assert abs(allocation.percentage - expected_percentage.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        
        # Verify usage and cost totals match original
        total_usage = sum(alloc.usage for alloc in allocations)
        total_cost = sum(alloc.cost for alloc in allocations)
        
        assert abs(total_usage - self.test_usage) < Decimal('0.0001')
        assert abs(total_cost - self.test_cost) < Decimal('0.0001')
    
    def test_year_boundary_crossing(self):
        """Test bills that cross year boundaries."""
        # Test case: December 15, 2023 - January 15, 2024
        billing_period = BillingPeriod(
            start_date=date(2023, 12, 15),
            end_date=date(2024, 1, 15),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly two allocations
        assert len(allocations) == 2
        
        # December 2023 allocation (Dec 15-31 = 17 days)
        dec_allocation = next(alloc for alloc in allocations if alloc.year == 2023 and alloc.month == 12)
        assert dec_allocation.days == 17
        
        # January 2024 allocation (Jan 1-15 = 15 days)
        jan_allocation = next(alloc for alloc in allocations if alloc.year == 2024 and alloc.month == 1)
        assert jan_allocation.days == 15
        
        # Verify total days (17 + 15 = 32)
        total_days = dec_allocation.days + jan_allocation.days
        assert total_days == 32
        
        # Verify year boundary crossing is handled correctly
        assert dec_allocation.year == 2023
        assert jan_allocation.year == 2024
        
        # Verify usage and cost totals
        total_usage = dec_allocation.usage + jan_allocation.usage
        total_cost = dec_allocation.cost + jan_allocation.cost
        
        assert abs(total_usage - self.test_usage) < Decimal('0.0001')
        assert abs(total_cost - self.test_cost) < Decimal('0.0001')
    
    def test_leap_year_february_handling(self):
        """Test bills spanning leap year February (29 days)."""
        # Test case: February 1 - February 29, 2024 (leap year)
        billing_period = BillingPeriod(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 29),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly one allocation
        assert len(allocations) == 1
        
        allocation = allocations[0]
        assert allocation.year == 2024
        assert allocation.month == 2
        assert allocation.days == 29  # Leap year February has 29 days
        assert allocation.usage == self.test_usage
        assert allocation.cost == self.test_cost
        assert allocation.percentage == Decimal('1.0000')
        
        # Compare with non-leap year
        non_leap_period = BillingPeriod(
            start_date=date(2023, 2, 1),
            end_date=date(2023, 2, 28),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        non_leap_allocations = self.calculator.calculate_time_based_allocation(non_leap_period)
        non_leap_allocation = non_leap_allocations[0]
        
        assert non_leap_allocation.days == 28  # Non-leap year February has 28 days
        assert allocation.days == 29  # Leap year February has 29 days
    
    def test_leap_year_boundary_spanning(self):
        """Test bills that span across leap year February boundary."""
        # Test case: January 15 - March 15, 2024 (spans leap year February)
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 3, 15),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly three allocations
        assert len(allocations) == 3
        
        # Find February allocation
        feb_allocation = next(alloc for alloc in allocations if alloc.month == 2)
        assert feb_allocation.year == 2024
        assert feb_allocation.days == 29  # Full leap year February
        
        # Verify leap year is correctly identified
        assert self.calculator.is_leap_year(2024) == True
        assert self.calculator.is_leap_year(2023) == False
    
    def test_mid_month_to_mid_month_allocation(self):
        """Test bills starting mid-month and ending mid-month."""
        # Test case: January 10 - February 20, 2024
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 10),
            end_date=date(2024, 2, 20),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly two allocations
        assert len(allocations) == 2
        
        # January allocation (Jan 10-31 = 22 days)
        jan_allocation = next(alloc for alloc in allocations if alloc.month == 1)
        assert jan_allocation.days == 22
        
        # February allocation (Feb 1-20 = 20 days)
        feb_allocation = next(alloc for alloc in allocations if alloc.month == 2)
        assert feb_allocation.days == 20
        
        # Verify total days (22 + 20 = 42)
        total_days = jan_allocation.days + feb_allocation.days
        expected_total_days = (date(2024, 2, 20) - date(2024, 1, 10)).days + 1
        assert total_days == expected_total_days == 42
        
        # Verify neither month gets full month allocation
        assert jan_allocation.days < 31  # January has 31 days
        assert feb_allocation.days < 29  # February 2024 has 29 days
        
        # Verify proportional calculations
        jan_percentage = Decimal('22') / Decimal('42')
        feb_percentage = Decimal('20') / Decimal('42')
        
        assert abs(jan_allocation.percentage - jan_percentage.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        assert abs(feb_allocation.percentage - feb_percentage.quantize(Decimal('0.0001'))) < Decimal('0.0001')
    
    def test_exact_day_count_calculations(self):
        """Test that day counts are calculated exactly correctly."""
        test_cases = [
            # (start_date, end_date, expected_total_days)
            (date(2024, 1, 1), date(2024, 1, 31), 31),  # Full January
            (date(2024, 2, 1), date(2024, 2, 29), 29),  # Full leap February
            (date(2023, 2, 1), date(2023, 2, 28), 28),  # Full non-leap February
            (date(2024, 1, 15), date(2024, 1, 15), 1),  # Single day
            (date(2024, 12, 31), date(2025, 1, 1), 2),  # Year boundary
            (date(2024, 1, 31), date(2024, 2, 1), 2),   # Month boundary
        ]
        
        for start_date, end_date, expected_days in test_cases:
            billing_period = BillingPeriod(
                start_date=start_date,
                end_date=end_date,
                total_usage=self.test_usage,
                total_cost=self.test_cost
            )
            
            actual_days = self.calculator.get_billing_period_days(billing_period)
            assert actual_days == expected_days, f"Failed for {start_date} to {end_date}: expected {expected_days}, got {actual_days}"
            
            # Verify allocation also gets correct day counts
            allocations = self.calculator.calculate_time_based_allocation(billing_period)
            total_allocated_days = sum(alloc.days for alloc in allocations)
            assert total_allocated_days == expected_days
    
    def test_usage_and_cost_proportional_allocation(self):
        """Test that usage and costs are properly proportioned across months."""
        # Test case with specific amounts for easy verification
        test_usage = Decimal('3100.0000')  # Divisible by 31 for easy calculation
        test_cost = Decimal('310.0000')    # Divisible by 31 for easy calculation
        
        # January 1-31, 2024 (31 days) + February 1-28, 2024 (29 days) = 60 days total
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 29),
            total_usage=test_usage,
            total_cost=test_cost
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly two allocations
        assert len(allocations) == 2
        
        jan_allocation = next(alloc for alloc in allocations if alloc.month == 1)
        feb_allocation = next(alloc for alloc in allocations if alloc.month == 2)
        
        # January: 31 days out of 60 total = 31/60 = 0.5167
        expected_jan_percentage = Decimal('31') / Decimal('60')
        expected_jan_usage = test_usage * expected_jan_percentage
        expected_jan_cost = test_cost * expected_jan_percentage
        
        # February: 29 days out of 60 total = 29/60 = 0.4833
        expected_feb_percentage = Decimal('29') / Decimal('60')
        expected_feb_usage = test_usage * expected_feb_percentage
        expected_feb_cost = test_cost * expected_feb_percentage
        
        # Verify allocations are within precision tolerance
        assert abs(jan_allocation.usage - expected_jan_usage.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        assert abs(jan_allocation.cost - expected_jan_cost.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        assert abs(feb_allocation.usage - expected_feb_usage.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        assert abs(feb_allocation.cost - expected_feb_cost.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        
        # Verify totals are preserved
        total_usage = jan_allocation.usage + feb_allocation.usage
        total_cost = jan_allocation.cost + feb_allocation.cost
        
        assert abs(total_usage - test_usage) < Decimal('0.0001')
        assert abs(total_cost - test_cost) < Decimal('0.0001')
        
        # Verify percentages sum to 1.0000
        total_percentage = jan_allocation.percentage + feb_allocation.percentage
        assert abs(total_percentage - Decimal('1.0000')) < Decimal('0.0001')
    
    def test_edge_case_single_day_bills(self):
        """Test edge case of single-day billing periods."""
        # Test various single-day periods
        test_dates = [
            date(2024, 1, 1),    # New Year's Day
            date(2024, 2, 29),   # Leap day
            date(2024, 12, 31),  # New Year's Eve
            date(2024, 6, 15),   # Mid-month, mid-year
        ]
        
        for test_date in test_dates:
            billing_period = BillingPeriod(
                start_date=test_date,
                end_date=test_date,
                total_usage=self.test_usage,
                total_cost=self.test_cost
            )
            
            allocations = self.calculator.calculate_time_based_allocation(billing_period)
            
            # Should have exactly one allocation
            assert len(allocations) == 1
            
            allocation = allocations[0]
            assert allocation.year == test_date.year
            assert allocation.month == test_date.month
            assert allocation.days == 1
            assert allocation.usage == self.test_usage
            assert allocation.cost == self.test_cost
            assert allocation.percentage == Decimal('1.0000')
    
    def test_edge_case_full_year_bill(self):
        """Test edge case of a full year billing period."""
        # Test case: January 1 - December 31, 2024 (leap year)
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            total_usage=Decimal('36600.0000'),  # 100 units per day * 366 days
            total_cost=Decimal('3660.0000')
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should have exactly 12 allocations (one per month)
        assert len(allocations) == 12
        
        # Verify each month has correct number of days
        expected_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 2024 leap year
        
        for i, allocation in enumerate(allocations):
            expected_month = i + 1
            assert allocation.month == expected_month
            assert allocation.year == 2024
            assert allocation.days == expected_days[i]
        
        # Verify total days sum to 366 (leap year)
        total_days = sum(alloc.days for alloc in allocations)
        assert total_days == 366
        
        # Verify usage and cost totals are preserved
        total_usage = sum(alloc.usage for alloc in allocations)
        total_cost = sum(alloc.cost for alloc in allocations)
        
        assert abs(total_usage - billing_period.total_usage) < Decimal('0.0001')
        assert abs(total_cost - billing_period.total_cost) < Decimal('0.0001')
    
    def test_month_boundary_identification(self):
        """Test the get_month_boundaries method."""
        # Test case spanning multiple months
        billing_period = BillingPeriod(
            start_date=date(2024, 10, 15),
            end_date=date(2025, 2, 10),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        boundaries = self.calculator.get_month_boundaries(billing_period)
        
        expected_boundaries = [
            (2024, 10),  # October 2024
            (2024, 11),  # November 2024
            (2024, 12),  # December 2024
            (2025, 1),   # January 2025
            (2025, 2),   # February 2025
        ]
        
        assert boundaries == expected_boundaries
    
    def test_partial_month_analysis(self):
        """Test the handle_partial_months method."""
        # Test case: January 10 - March 20, 2024
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 10),
            end_date=date(2024, 3, 20),
            total_usage=self.test_usage,
            total_cost=self.test_cost
        )
        
        partial_info = self.calculator.handle_partial_months(billing_period)
        
        # January is partial (starts on 10th)
        assert partial_info['has_partial_start'] == True
        assert partial_info['start_month_partial_days'] == 22  # Jan 10-31 = 22 days
        assert partial_info['start_month_total_days'] == 31
        
        # March is partial (ends on 20th)  
        assert partial_info['has_partial_end'] == True
        assert partial_info['end_month_partial_days'] == 20  # Mar 1-20 = 20 days
        assert partial_info['end_month_total_days'] == 31
    
    def test_allocation_summary_statistics(self):
        """Test the allocation summary functionality."""
        # Test case spanning multiple months
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            total_usage=Decimal('9100.0000'),  # 100 per day * 91 days
            total_cost=Decimal('910.0000')
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        summary = self.calculator.summarize_allocation(allocations)
        
        # Verify summary statistics
        assert summary['number_of_months'] == 3
        assert summary['total_days'] == 91  # Jan(31) + Feb(29) + Mar(31) = 91 days
        assert abs(summary['total_usage'] - billing_period.total_usage) < Decimal('0.0001')
        assert abs(summary['total_cost'] - billing_period.total_cost) < Decimal('0.0001')
        assert abs(summary['total_percentage'] - Decimal('1.0000')) < Decimal('0.0001')
        
        # Verify averages
        expected_avg_usage = billing_period.total_usage / Decimal('3')
        expected_avg_cost = billing_period.total_cost / Decimal('3')
        
        assert abs(summary['average_monthly_usage'] - expected_avg_usage.quantize(Decimal('0.0001'))) < Decimal('0.0001')
        assert abs(summary['average_monthly_cost'] - expected_avg_cost.quantize(Decimal('0.0001'))) < Decimal('0.0001')
    
    def test_zero_usage_and_cost_handling(self):
        """Test handling of zero usage and cost values."""
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 29),
            total_usage=Decimal('0.0000'),
            total_cost=Decimal('0.0000')
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        # Should still create proper allocations
        assert len(allocations) == 2
        
        for allocation in allocations:
            assert allocation.usage == Decimal('0.0000')
            assert allocation.cost == Decimal('0.0000')
            assert allocation.days > 0  # Should still track days correctly
            assert allocation.percentage > Decimal('0')  # Should have valid percentages
    
    def test_high_precision_calculations(self):
        """Test calculations maintain high precision for financial accuracy."""
        # Use a precision-challenging amount
        test_usage = Decimal('1000.3333')  # Repeating decimal
        test_cost = Decimal('333.3333')
        
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),  # 3 days
            total_usage=test_usage,
            total_cost=test_cost
        )
        
        allocations = self.calculator.calculate_time_based_allocation(billing_period)
        
        assert len(allocations) == 1
        allocation = allocations[0]
        
        # With 4 decimal precision, should maintain accuracy
        assert allocation.usage == test_usage
        assert allocation.cost == test_cost
        
        # Test daily allocation precision
        daily_allocations = self.calculator.calculate_daily_allocations(billing_period)
        total_daily_usage = sum(daily_usage for daily_usage, daily_cost in daily_allocations.values())
        total_daily_cost = sum(daily_cost for daily_usage, daily_cost in daily_allocations.values())
        
        # Should be within precision tolerance
        assert abs(total_daily_usage - test_usage) < Decimal('0.0001')
        assert abs(total_daily_cost - test_cost) < Decimal('0.0001')


# Additional test utility functions

def create_test_billing_period(start_date: date, end_date: date, usage: str = "1000.0000", cost: str = "250.0000") -> BillingPeriod:
    """Helper function to create test billing periods."""
    return BillingPeriod(
        start_date=start_date,
        end_date=end_date,
        total_usage=Decimal(usage),
        total_cost=Decimal(cost),
        facility_id="TEST_FACILITY",
        usage_type="test_usage"
    )


class TestEdgeCasesAndValidation:
    """Additional test class for edge cases and validation scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = ProRatingCalculator(precision=4)
    
    def test_invalid_date_ranges(self):
        """Test validation of invalid date ranges."""
        with pytest.raises(ValueError, match="Start date must be before or equal to end date"):
            BillingPeriod(
                start_date=date(2024, 1, 15),
                end_date=date(2024, 1, 10),  # End before start
                total_usage=Decimal('100'),
                total_cost=Decimal('50')
            )
    
    def test_negative_values_validation(self):
        """Test validation of negative usage and cost values."""
        with pytest.raises(ValueError, match="Total usage cannot be negative"):
            BillingPeriod(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                total_usage=Decimal('-100'),  # Negative usage
                total_cost=Decimal('50')
            )
        
        with pytest.raises(ValueError, match="Total cost cannot be negative"):
            BillingPeriod(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                total_usage=Decimal('100'),
                total_cost=Decimal('-50')  # Negative cost
            )
    
    def test_extremely_long_billing_period(self):
        """Test handling of extremely long billing periods."""
        # Create a 3-year billing period (exceeds 2-year limit)
        billing_period = BillingPeriod(
            start_date=date(2024, 1, 1),
            end_date=date(2027, 1, 1),  # 3+ years
            total_usage=Decimal('1000'),
            total_cost=Decimal('250')
        )
        
        # Should raise validation error
        with pytest.raises(ValueError, match="Billing period too long"):
            self.calculator.validate_billing_period(billing_period)
    
    def test_month_name_property(self):
        """Test the month_name property of MonthlyAllocation."""
        allocation = MonthlyAllocation(
            year=2024,
            month=3,
            days=31,
            usage=Decimal('100'),
            cost=Decimal('25'),
            percentage=Decimal('1.0000')
        )
        
        assert allocation.month_name == "March"
        assert allocation.month_year_key == "2024-03"
    
    def test_days_in_month_validation(self):
        """Test validation of month parameter in get_days_in_month."""
        with pytest.raises(ValueError, match="Month must be between 1 and 12"):
            self.calculator.get_days_in_month(2024, 0)  # Invalid month
        
        with pytest.raises(ValueError, match="Month must be between 1 and 12"):
            self.calculator.get_days_in_month(2024, 13)  # Invalid month
        
        # Valid months should work
        assert self.calculator.get_days_in_month(2024, 1) == 31
        assert self.calculator.get_days_in_month(2024, 2) == 29  # Leap year
        assert self.calculator.get_days_in_month(2023, 2) == 28  # Non-leap year


if __name__ == "__main__":
    """
    Run tests directly if this file is executed.
    
    Usage:
        python3 -m pytest tests/test_monthly_boundary_allocation.py -v
    """
    pytest.main([__file__, "-v"])