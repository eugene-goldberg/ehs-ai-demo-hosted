"""
Base Data Generator Module

This module provides the abstract base class for all data generators in the EHS AI Demo system.
It includes common functionality for date handling, noise generation, pattern creation,
configuration management, and reproducibility through random seed management.

Author: EHS AI Demo Team
Created: 2025-08-28
Version: 1.0.0
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import math
import logging


@dataclass
class GeneratorConfig:
    """Configuration class for data generators"""
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Date range configuration
    start_date: datetime = field(default_factory=lambda: datetime(2023, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2024, 12, 31))
    
    # Noise configuration
    noise_level: float = 0.1  # Standard deviation as fraction of base value
    noise_type: str = "gaussian"  # gaussian, uniform, or exponential
    
    # Pattern configuration
    enable_seasonal_patterns: bool = True
    enable_weekly_patterns: bool = True
    enable_daily_patterns: bool = True
    
    # Seasonal pattern parameters
    seasonal_amplitude: float = 0.2  # Amplitude as fraction of base value
    seasonal_phase_offset: float = 0.0  # Phase offset in radians
    
    # Weekly pattern parameters
    weekly_amplitude: float = 0.15
    weekend_multiplier: float = 0.7  # Reduction factor for weekends
    
    # Daily pattern parameters
    daily_amplitude: float = 0.1
    peak_hour: int = 14  # Hour of day with peak activity (24-hour format)
    
    # Data quality parameters
    missing_data_rate: float = 0.02  # Fraction of data points to make missing
    outlier_rate: float = 0.01  # Fraction of data points to make outliers
    outlier_multiplier: float = 3.0  # Factor to multiply base value for outliers


class BaseGenerator(ABC):
    """
    Abstract base class for all data generators.
    
    This class provides common functionality including:
    - Date handling and time series generation
    - Noise generation with various distributions
    - Seasonal, weekly, and daily pattern creation
    - Configuration management
    - Random seed management for reproducibility
    - Data quality simulation (missing data, outliers)
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the base generator.
        
        Args:
            config: Configuration object for the generator. If None, uses default config.
        """
        self.config = config or GeneratorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Initialize internal state
        self._date_range = None
        self._time_index = None
        
    def _set_random_seeds(self) -> None:
        """Set random seeds for all random number generators."""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
    def get_date_range(self) -> List[datetime]:
        """
        Generate a list of datetime objects for the configured date range.
        
        Returns:
            List of datetime objects from start_date to end_date (daily frequency)
        """
        if self._date_range is None:
            dates = []
            current_date = self.config.start_date
            while current_date <= self.config.end_date:
                dates.append(current_date)
                current_date += timedelta(days=1)
            self._date_range = dates
        return self._date_range
    
    def get_time_index(self, frequency: str = "D") -> List[datetime]:
        """
        Generate time index with specified frequency.
        
        Args:
            frequency: Time frequency ('D' for daily, 'H' for hourly, 'T' for minute)
            
        Returns:
            List of datetime objects with specified frequency
        """
        if frequency == "D":
            return self.get_date_range()
        elif frequency == "H":
            return self._generate_hourly_index()
        elif frequency == "T":
            return self._generate_minute_index()
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
    
    def _generate_hourly_index(self) -> List[datetime]:
        """Generate hourly datetime index."""
        dates = []
        current_date = self.config.start_date
        while current_date <= self.config.end_date:
            for hour in range(24):
                dates.append(current_date.replace(hour=hour))
            current_date += timedelta(days=1)
        return dates
    
    def _generate_minute_index(self) -> List[datetime]:
        """Generate minute-level datetime index."""
        dates = []
        current_date = self.config.start_date
        while current_date <= self.config.end_date:
            for hour in range(24):
                for minute in range(0, 60, 5):  # 5-minute intervals
                    dates.append(current_date.replace(hour=hour, minute=minute))
            current_date += timedelta(days=1)
        return dates
    
    def generate_noise(self, base_values: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Generate noise to add to base values.
        
        Args:
            base_values: Base values to add noise to
            
        Returns:
            Noise array with same shape as base_values
        """
        if isinstance(base_values, (int, float)):
            base_values = np.array([base_values])
        elif isinstance(base_values, list):
            base_values = np.array(base_values)
        
        noise_std = self.config.noise_level * np.abs(base_values)
        
        if self.config.noise_type == "gaussian":
            noise = np.random.normal(0, noise_std)
        elif self.config.noise_type == "uniform":
            noise = np.random.uniform(-noise_std * 1.73, noise_std * 1.73)  # Same variance as gaussian
        elif self.config.noise_type == "exponential":
            # Exponential noise (always positive)
            noise = np.random.exponential(noise_std)
        else:
            raise ValueError(f"Unsupported noise type: {self.config.noise_type}")
        
        return noise
    
    def generate_seasonal_pattern(self, dates: List[datetime], base_value: float = 1.0) -> np.ndarray:
        """
        Generate seasonal pattern based on day of year.
        
        Args:
            dates: List of datetime objects
            base_value: Base value to apply pattern to
            
        Returns:
            Array of seasonal multipliers
        """
        if not self.config.enable_seasonal_patterns:
            return np.ones(len(dates))
        
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # Seasonal pattern: peaks in summer (day 180), low in winter
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365.25 + self.config.seasonal_phase_offset)
        seasonal_multiplier = 1.0 + self.config.seasonal_amplitude * seasonal_factor
        
        return seasonal_multiplier
    
    def generate_weekly_pattern(self, dates: List[datetime], base_value: float = 1.0) -> np.ndarray:
        """
        Generate weekly pattern with weekend effects.
        
        Args:
            dates: List of datetime objects
            base_value: Base value to apply pattern to
            
        Returns:
            Array of weekly multipliers
        """
        if not self.config.enable_weekly_patterns:
            return np.ones(len(dates))
        
        weekly_multiplier = np.ones(len(dates))
        
        for i, date in enumerate(dates):
            day_of_week = date.weekday()  # 0 = Monday, 6 = Sunday
            
            # Apply weekend reduction
            if day_of_week >= 5:  # Saturday (5) or Sunday (6)
                weekly_multiplier[i] = self.config.weekend_multiplier
            else:
                # Weekday pattern: slight variation throughout the week
                weekday_factor = np.sin(2 * np.pi * day_of_week / 7)
                weekly_multiplier[i] = 1.0 + self.config.weekly_amplitude * weekday_factor * 0.1
        
        return weekly_multiplier
    
    def generate_daily_pattern(self, dates: List[datetime], base_value: float = 1.0) -> np.ndarray:
        """
        Generate daily pattern with peak activity hours.
        
        Args:
            dates: List of datetime objects
            base_value: Base value to apply pattern to
            
        Returns:
            Array of daily multipliers
        """
        if not self.config.enable_daily_patterns:
            return np.ones(len(dates))
        
        daily_multiplier = np.ones(len(dates))
        
        for i, date in enumerate(dates):
            if hasattr(date, 'hour'):
                hour = date.hour
            else:
                hour = 12  # Default to midday for date-only objects
            
            # Daily pattern: peak at specified hour, low at night
            hour_factor = np.cos(2 * np.pi * (hour - self.config.peak_hour) / 24)
            daily_multiplier[i] = 1.0 + self.config.daily_amplitude * hour_factor
        
        return daily_multiplier
    
    def apply_combined_patterns(self, dates: List[datetime], base_values: Union[float, np.ndarray]) -> np.ndarray:
        """
        Apply all enabled patterns (seasonal, weekly, daily) to base values.
        
        Args:
            dates: List of datetime objects
            base_values: Base values to apply patterns to
            
        Returns:
            Array with all patterns applied
        """
        if isinstance(base_values, (int, float)):
            base_values = np.full(len(dates), base_values)
        
        result = base_values.copy()
        
        # Apply seasonal pattern
        seasonal = self.generate_seasonal_pattern(dates, 1.0)
        result *= seasonal
        
        # Apply weekly pattern
        weekly = self.generate_weekly_pattern(dates, 1.0)
        result *= weekly
        
        # Apply daily pattern
        daily = self.generate_daily_pattern(dates, 1.0)
        result *= daily
        
        return result
    
    def add_data_quality_issues(self, values: np.ndarray, dates: List[datetime]) -> Tuple[np.ndarray, List[bool]]:
        """
        Add data quality issues (missing data, outliers) to the generated data.
        
        Args:
            values: Array of generated values
            dates: Corresponding dates
            
        Returns:
            Tuple of (modified_values, missing_mask) where missing_mask indicates missing data points
        """
        result_values = values.copy()
        missing_mask = np.zeros(len(values), dtype=bool)
        
        # Add missing data
        if self.config.missing_data_rate > 0:
            n_missing = int(len(values) * self.config.missing_data_rate)
            missing_indices = np.random.choice(len(values), n_missing, replace=False)
            missing_mask[missing_indices] = True
            result_values[missing_indices] = np.nan
        
        # Add outliers
        if self.config.outlier_rate > 0:
            n_outliers = int(len(values) * self.config.outlier_rate)
            outlier_indices = np.random.choice(
                np.where(~missing_mask)[0],  # Don't make missing values outliers
                min(n_outliers, np.sum(~missing_mask)),
                replace=False
            )
            
            for idx in outlier_indices:
                if np.random.random() > 0.5:
                    # Positive outlier
                    result_values[idx] *= self.config.outlier_multiplier
                else:
                    # Negative outlier (but keep positive if original was positive)
                    result_values[idx] *= (1.0 / self.config.outlier_multiplier)
                    if values[idx] > 0:
                        result_values[idx] = max(result_values[idx], 0)
        
        return result_values, missing_mask
    
    def validate_date_range(self) -> bool:
        """
        Validate that the configured date range is valid.
        
        Returns:
            True if date range is valid, False otherwise
        """
        if self.config.start_date >= self.config.end_date:
            self.logger.error("Start date must be before end date")
            return False
        
        if (self.config.end_date - self.config.start_date).days > 3650:  # > 10 years
            self.logger.warning("Date range is very large (>10 years), generation may be slow")
        
        return True
    
    def get_generation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the generation configuration and process.
        
        Returns:
            Dictionary containing generation metadata
        """
        return {
            "generator_class": self.__class__.__name__,
            "config": {
                "random_seed": self.config.random_seed,
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "noise_level": self.config.noise_level,
                "noise_type": self.config.noise_type,
                "patterns_enabled": {
                    "seasonal": self.config.enable_seasonal_patterns,
                    "weekly": self.config.enable_weekly_patterns,
                    "daily": self.config.enable_daily_patterns
                }
            },
            "date_range_days": (self.config.end_date - self.config.start_date).days + 1,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    @abstractmethod
    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Abstract method to generate data. Must be implemented by subclasses.
        
        Args:
            **kwargs: Generator-specific parameters
            
        Returns:
            Dictionary containing generated data and metadata
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        return f"{self.__class__.__name__}(seed={self.config.random_seed})"