"""
Time Series Analysis Foundation for EHS Risk Assessment

This module provides a comprehensive time series analysis framework for detecting
trends, seasonality, anomalies, and change points in EHS-related data.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction classification."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class AnomalyType(Enum):
    """Types of detected anomalies."""
    STATISTICAL = "statistical"  # Based on z-score or modified z-score
    IQR = "iqr"  # Interquartile range method
    ISOLATION = "isolation"  # Isolation forest
    SEASONAL = "seasonal"  # Deviation from seasonal pattern


@dataclass
class TimeSeriesData:
    """Container for time series data."""
    timestamps: List[datetime]
    values: List[float]
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate data consistency."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have the same length")
        if len(self.timestamps) < 2:
            raise ValueError("Time series must have at least 2 data points")
    
    @property
    def df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'value': self.values
        }).set_index('timestamp')
    
    @property
    def length(self) -> int:
        """Number of data points."""
        return len(self.values)
    
    def __len__(self) -> int:
        """Return the length of the time series."""
        return len(self.values)


@dataclass
class SeasonalComponents:
    """Results from seasonal decomposition."""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    seasonal_strength: float
    trend_strength: float
    
    @property
    def has_strong_seasonality(self) -> bool:
        """Check if seasonality is statistically significant."""
        return self.seasonal_strength > 0.6
    
    @property
    def has_strong_trend(self) -> bool:
        """Check if trend is statistically significant."""
        return self.trend_strength > 0.6


@dataclass
class TrendAnalysis:
    """Results from trend analysis."""
    direction: TrendDirection
    slope: float
    p_value: float
    confidence_interval: Tuple[float, float]
    r_squared: float
    mann_kendall_tau: float
    mann_kendall_p_value: float
    
    @property
    def is_significant(self) -> bool:
        """Check if trend is statistically significant."""
        return self.p_value < 0.05 and self.mann_kendall_p_value < 0.05


@dataclass
class AnomalyResult:
    """Results from anomaly detection."""
    indices: List[int]
    values: List[float]
    timestamps: List[datetime]
    scores: List[float]
    anomaly_type: AnomalyType
    threshold: float
    
    @property
    def count(self) -> int:
        """Number of detected anomalies."""
        return len(self.indices)
    
    @property
    def anomaly_rate(self) -> float:
        """Percentage of data points that are anomalies."""
        total_points = len(self.values) if hasattr(self, 'total_points') else 100
        return (self.count / total_points) * 100


@dataclass
class ChangePoint:
    """Represents a detected change point."""
    index: int
    timestamp: datetime
    confidence: float
    magnitude: float
    direction: str  # 'increase' or 'decrease'


@dataclass
class DataQualityReport:
    """Data quality assessment results."""
    missing_values: int
    missing_percentage: float
    duplicate_timestamps: int
    irregular_intervals: int
    outlier_count: int
    data_completeness_score: float
    temporal_consistency_score: float
    overall_quality_score: float
    
    @property
    def is_high_quality(self) -> bool:
        """Check if data meets quality standards."""
        return self.overall_quality_score >= 0.8


@dataclass
class ForecastResult:
    """Results from time series forecasting."""
    forecasted_values: List[float]
    forecast_timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    trend: TrendDirection
    forecast_horizon_days: int
    model_name: str
    confidence_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'forecasted_values': self.forecasted_values,
            'forecast_timestamps': [ts.isoformat() for ts in self.forecast_timestamps],
            'confidence_intervals': self.confidence_intervals,
            'trend': self.trend.value,
            'forecast_horizon_days': self.forecast_horizon_days,
            'model_name': self.model_name,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }


class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis for EHS risk assessment.
    
    This class provides statistical analysis, trend detection, seasonal decomposition,
    anomaly detection, and change point analysis for time series data.
    """
    
    def __init__(self, 
                 cache_size: int = 128,
                 significance_level: float = 0.05,
                 seasonal_periods: Optional[int] = None):
        """
        Initialize the time series analyzer.
        
        Args:
            cache_size: Size of LRU cache for expensive operations
            significance_level: Statistical significance threshold
            seasonal_periods: Expected seasonal period (e.g., 12 for monthly data)
        """
        self.cache_size = cache_size
        self.significance_level = significance_level
        self.seasonal_periods = seasonal_periods
        self.logger = logging.getLogger(__name__)
    
    async def analyze_complete(self, data: TimeSeriesData) -> Dict:
        """
        Perform comprehensive time series analysis.
        
        Args:
            data: Time series data to analyze
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Run all analyses concurrently where possible
            results = await asyncio.gather(
                self.assess_data_quality(data),
                self.detect_trend(data),
                self.decompose_seasonal(data),
                self.detect_anomalies(data, method='statistical'),
                self.detect_changepoints(data),
                return_exceptions=True
            )
            
            quality_report, trend_analysis, seasonal_components, anomalies, changepoints = results
            
            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Analysis step {i} failed: {result}")
            
            return {
                'data_quality': quality_report if not isinstance(quality_report, Exception) else None,
                'trend_analysis': trend_analysis if not isinstance(trend_analysis, Exception) else None,
                'seasonal_components': seasonal_components if not isinstance(seasonal_components, Exception) else None,
                'anomalies': anomalies if not isinstance(anomalies, Exception) else None,
                'changepoints': changepoints if not isinstance(changepoints, Exception) else None,
                'summary_statistics': self.calculate_statistics(data),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Complete analysis failed: {e}")
            raise
    
    async def assess_data_quality(self, data: TimeSeriesData) -> DataQualityReport:
        """
        Assess data quality across multiple dimensions.
        
        Args:
            data: Time series data to assess
            
        Returns:
            Comprehensive data quality report
        """
        df = data.df
        
        # Missing values
        missing_count = df['value'].isna().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        # Duplicate timestamps
        duplicate_timestamps = df.index.duplicated().sum()
        
        # Temporal consistency
        intervals = df.index.to_series().diff().dropna()
        if len(intervals) > 1:
            median_interval = intervals.median()
            irregular_intervals = (abs(intervals - median_interval) > median_interval * 0.1).sum()
            temporal_consistency = 1.0 - (irregular_intervals / len(intervals))
        else:
            irregular_intervals = 0
            temporal_consistency = 1.0
        
        # Outlier detection (simple IQR method)
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df['value'] < (Q1 - 1.5 * IQR)) | 
                        (df['value'] > (Q3 + 1.5 * IQR))).sum()
        
        # Completeness score
        data_completeness = 1.0 - (missing_percentage / 100)
        
        # Overall quality score (weighted average)
        overall_quality = (
            data_completeness * 0.4 +
            temporal_consistency * 0.3 +
            (1.0 - min(duplicate_timestamps / len(df), 1.0)) * 0.2 +
            (1.0 - min(outlier_count / len(df), 0.5)) * 0.1
        )
        
        return DataQualityReport(
            missing_values=missing_count,
            missing_percentage=missing_percentage,
            duplicate_timestamps=duplicate_timestamps,
            irregular_intervals=irregular_intervals,
            outlier_count=outlier_count,
            data_completeness_score=data_completeness,
            temporal_consistency_score=temporal_consistency,
            overall_quality_score=overall_quality
        )
    
    async def detect_trend(self, data: TimeSeriesData) -> TrendAnalysis:
        """
        Detect trends using multiple statistical methods.
        
        Args:
            data: Time series data
            
        Returns:
            Comprehensive trend analysis
        """
        df = data.df
        values = df['value'].dropna()
        
        if len(values) < 3:
            raise ValueError("Insufficient data for trend analysis")
        
        # Linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        confidence_interval = (slope - 1.96 * std_err, slope + 1.96 * std_err)
        
        # Mann-Kendall test for monotonic trend
        tau, mk_p_value = self._mann_kendall_test(values.values)
        
        # Determine trend direction
        if abs(slope) < std_err:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING if p_value < self.significance_level else TrendDirection.STABLE
        else:
            direction = TrendDirection.DECREASING if p_value < self.significance_level else TrendDirection.STABLE
        
        # Check for volatility
        cv = values.std() / values.mean() if values.mean() != 0 else float('inf')
        if cv > 1.0:  # Coefficient of variation > 100%
            direction = TrendDirection.VOLATILE
        
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            p_value=p_value,
            confidence_interval=confidence_interval,
            r_squared=r_value**2,
            mann_kendall_tau=tau,
            mann_kendall_p_value=mk_p_value
        )
    
    async def analyze_trend(self, data: TimeSeriesData) -> TrendAnalysis:
        """
        Analyze trend patterns in time series data.
        
        This method performs comprehensive trend analysis using statistical
        methods to determine direction, significance, and confidence.
        
        Args:
            data: TimeSeriesData object with timestamps and values
            
        Returns:
            TrendAnalysis with statistical measures and trend direction
            
        Raises:
            ValueError: If data is insufficient for analysis
            
        Example:
            >>> analyzer = TimeSeriesAnalyzer()
            >>> data = TimeSeriesData(timestamps=[...], values=[...])
            >>> trend = await analyzer.analyze_trend(data)
            >>> print(f"Trend: {trend.direction.value}")
        """
        try:
            # Input validation
            if data.length < 3:
                raise ValueError("Minimum 3 data points required for trend analysis")
            
            # Delegate to existing implementation
            result = await self.detect_trend(data)
            
            # Add performance logging
            self.logger.info(f"Trend analysis completed: {result.direction.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            raise
    
    async def decompose_seasonal(self, data: TimeSeriesData) -> Optional[SeasonalComponents]:
        """
        Perform seasonal decomposition using STL (Seasonal and Trend decomposition using Loess).
        
        Args:
            data: Time series data
            
        Returns:
            Seasonal decomposition components or None if insufficient data
        """
        df = data.df
        
        # Determine seasonal period
        if self.seasonal_periods is None:
            # Auto-detect seasonal period based on data frequency
            freq = pd.infer_freq(df.index)
            if freq:
                if 'D' in freq or 'B' in freq:  # Daily data
                    period = 7  # Weekly seasonality
                elif 'W' in freq:  # Weekly data
                    period = 52  # Yearly seasonality
                elif 'M' in freq:  # Monthly data
                    period = 12  # Yearly seasonality
                else:
                    period = min(len(df) // 2, 12)  # Default
            else:
                period = min(len(df) // 2, 12)
        else:
            period = self.seasonal_periods
        
        if len(df) < 2 * period:
            self.logger.warning(f"Insufficient data for seasonal decomposition (need at least {2 * period} points)")
            return None
        
        try:
            # Ensure no missing values for STL
            clean_data = df['value'].interpolate()
            
            # Perform STL decomposition
            stl = STL(clean_data, seasonal=period, robust=True)
            decomposition = stl.fit()
            
            # Calculate seasonal and trend strength
            trend_strength = self._calculate_trend_strength(
                clean_data.values, 
                decomposition.trend.values, 
                decomposition.resid.values
            )
            
            seasonal_strength = self._calculate_seasonal_strength(
                clean_data.values,
                decomposition.seasonal.values,
                decomposition.resid.values
            )
            
            return SeasonalComponents(
                trend=decomposition.trend.values,
                seasonal=decomposition.seasonal.values,
                residual=decomposition.resid.values,
                seasonal_strength=seasonal_strength,
                trend_strength=trend_strength
            )
            
        except Exception as e:
            self.logger.error(f"Seasonal decomposition failed: {e}")
            return None
    
    async def detect_anomalies(self, 
                             data: TimeSeriesData, 
                             method: str = 'statistical',
                             threshold: float = 3.0) -> AnomalyResult:
        """
        Detect anomalies using specified method.
        
        Args:
            data: Time series data
            method: Detection method ('statistical', 'iqr', 'modified_zscore')
            threshold: Threshold for anomaly detection
            
        Returns:
            Anomaly detection results
        """
        df = data.df
        values = df['value'].dropna()
        
        if method == 'statistical':
            anomalies = self._detect_statistical_anomalies(values, threshold)
            anomaly_type = AnomalyType.STATISTICAL
        elif method == 'iqr':
            anomalies = self._detect_iqr_anomalies(values)
            anomaly_type = AnomalyType.IQR
        elif method == 'modified_zscore':
            anomalies = self._detect_modified_zscore_anomalies(values, threshold)
            anomaly_type = AnomalyType.STATISTICAL
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        # Get anomaly details
        anomaly_indices = anomalies['indices']
        anomaly_values = [values.iloc[i] for i in anomaly_indices]
        anomaly_timestamps = [data.timestamps[i] for i in anomaly_indices]
        anomaly_scores = anomalies['scores']
        
        return AnomalyResult(
            indices=anomaly_indices,
            values=anomaly_values,
            timestamps=anomaly_timestamps,
            scores=anomaly_scores,
            anomaly_type=anomaly_type,
            threshold=threshold
        )
    
    async def detect_changepoints(self, 
                                data: TimeSeriesData,
                                min_size: int = 10,
                                jump_threshold: float = 1.0) -> List[ChangePoint]:
        """
        Detect change points using CUSUM-based method.
        
        Args:
            data: Time series data
            min_size: Minimum segment size
            jump_threshold: Minimum jump size to consider
            
        Returns:
            List of detected change points
        """
        df = data.df
        values = df['value'].dropna().values
        
        if len(values) < 2 * min_size:
            return []
        
        changepoints = []
        
        # Simple CUSUM-based change point detection
        mean_val = np.mean(values)
        cumsum = np.cumsum(values - mean_val)
        
        # Find peaks in absolute CUSUM
        abs_cumsum = np.abs(cumsum)
        peaks, properties = find_peaks(abs_cumsum, 
                                     height=jump_threshold * np.std(values),
                                     distance=min_size)
        
        for peak in peaks:
            if min_size <= peak <= len(values) - min_size:
                # Calculate confidence based on peak prominence
                confidence = min(properties['peak_heights'][list(peaks).index(peak)] / 
                               (np.std(values) * jump_threshold), 1.0)
                
                # Determine direction of change
                before_mean = np.mean(values[max(0, peak - min_size):peak])
                after_mean = np.mean(values[peak:min(len(values), peak + min_size)])
                direction = 'increase' if after_mean > before_mean else 'decrease'
                
                changepoints.append(ChangePoint(
                    index=peak,
                    timestamp=data.timestamps[peak],
                    confidence=confidence,
                    magnitude=abs(after_mean - before_mean),
                    direction=direction
                ))
        
        return sorted(changepoints, key=lambda x: x.confidence, reverse=True)
    
    def calculate_statistics(self, data: TimeSeriesData) -> Dict:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary of statistical measures
        """
        df = data.df
        values = df['value'].dropna()
        
        if len(values) == 0:
            return {}
        
        # Basic statistics
        stats_dict = {
            'count': len(values),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'var': values.var(),
            'min': values.min(),
            'max': values.max(),
            'range': values.max() - values.min(),
            'q25': values.quantile(0.25),
            'q75': values.quantile(0.75),
            'iqr': values.quantile(0.75) - values.quantile(0.25),
            'skewness': values.skew(),
            'kurtosis': values.kurtosis(),
        }
        
        # Additional statistics
        stats_dict.update({
            'coefficient_of_variation': stats_dict['std'] / stats_dict['mean'] if stats_dict['mean'] != 0 else float('inf'),
            'mad': np.median(np.abs(values - values.median())),  # Median Absolute Deviation
            'autocorr_lag1': values.autocorr(lag=1) if len(values) > 1 else 0,
        })
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            adf_statistic, adf_pvalue, *_ = adfuller(values, autolag='AIC')
            stats_dict.update({
                'adf_statistic': adf_statistic,
                'adf_pvalue': adf_pvalue,
                'is_stationary': adf_pvalue < self.significance_level
            })
        except Exception as e:
            self.logger.warning(f"Stationarity test failed: {e}")
            stats_dict.update({
                'adf_statistic': None,
                'adf_pvalue': None,
                'is_stationary': None
            })
        
        return stats_dict
    
    # Private helper methods
    
    def _mann_kendall_test(self, values: np.ndarray) -> Tuple[float, float]:
        """Perform Mann-Kendall test for monotonic trend."""
        n = len(values)
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(values[j] - values[i])
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate tau
        tau = s / (n * (n - 1) / 2)
        
        # Calculate z-score and p-value
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return tau, p_value
    
    def _calculate_trend_strength(self, 
                                data: np.ndarray, 
                                trend: np.ndarray, 
                                residual: np.ndarray) -> float:
        """Calculate trend strength."""
        detrended = data - trend
        return 1 - np.var(residual) / np.var(detrended)
    
    def _calculate_seasonal_strength(self, 
                                   data: np.ndarray, 
                                   seasonal: np.ndarray, 
                                   residual: np.ndarray) -> float:
        """Calculate seasonal strength."""
        deseasoned = data - seasonal
        return 1 - np.var(residual) / np.var(deseasoned)
    
    def _detect_statistical_anomalies(self, 
                                    values: pd.Series, 
                                    threshold: float) -> Dict:
        """Detect anomalies using z-score method."""
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val == 0:
            return {'indices': [], 'scores': []}
        
        z_scores = np.abs((values - mean_val) / std_val)
        anomaly_indices = np.where(z_scores > threshold)[0].tolist()
        anomaly_scores = z_scores.iloc[anomaly_indices].tolist()
        
        return {'indices': anomaly_indices, 'scores': anomaly_scores}
    
    def _detect_iqr_anomalies(self, values: pd.Series) -> Dict:
        """Detect anomalies using IQR method."""
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomaly_mask = (values < lower_bound) | (values > upper_bound)
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        
        # Calculate scores based on distance from bounds
        scores = []
        for idx in anomaly_indices:
            val = values.iloc[idx]
            if val < lower_bound:
                score = (lower_bound - val) / IQR
            else:
                score = (val - upper_bound) / IQR
            scores.append(score)
        
        return {'indices': anomaly_indices, 'scores': scores}
    
    def _detect_modified_zscore_anomalies(self, 
                                        values: pd.Series, 
                                        threshold: float) -> Dict:
        """Detect anomalies using modified z-score (based on median)."""
        median_val = values.median()
        mad = np.median(np.abs(values - median_val))
        
        if mad == 0:
            return {'indices': [], 'scores': []}
        
        modified_z_scores = 0.6745 * (values - median_val) / mad
        anomaly_indices = np.where(np.abs(modified_z_scores) > threshold)[0].tolist()
        anomaly_scores = np.abs(modified_z_scores).iloc[anomaly_indices].tolist()
        
        return {'indices': anomaly_indices, 'scores': anomaly_scores}


class TimeSeriesPredictor:
    """
    Time series predictor for EHS risk assessment forecasting.
    
    This class provides simple forecasting capabilities using statistical methods
    and trend extrapolation for short to medium-term predictions.
    """
    
    def __init__(self, 
                 default_confidence: float = 0.8,
                 max_forecast_horizon: int = 90):
        """
        Initialize the time series predictor.
        
        Args:
            default_confidence: Default confidence score for forecasts
            max_forecast_horizon: Maximum forecast horizon in days
        """
        self.default_confidence = default_confidence
        self.max_forecast_horizon = max_forecast_horizon
        self.logger = logging.getLogger(__name__)
    
    def predict(self, 
                data: Dict[str, Any], 
                horizon_days: int) -> Optional[Dict[str, Any]]:
        """
        Generate forecast for the given data and horizon.
        
        Args:
            data: Dictionary containing historical data
            horizon_days: Forecast horizon in days
            
        Returns:
            Dictionary containing forecast results or None if prediction fails
        """
        try:
            if horizon_days > self.max_forecast_horizon:
                self.logger.warning(f"Forecast horizon {horizon_days} exceeds maximum {self.max_forecast_horizon}")
                return None
            
            # Extract time series data from the input
            time_series = self._extract_time_series(data)
            if not time_series:
                return None
            
            # Perform simple trend-based forecasting
            forecast_result = self._generate_simple_forecast(time_series, horizon_days)
            
            return forecast_result.to_dict() if forecast_result else None
            
        except Exception as e:
            self.logger.error(f"Forecast generation failed: {e}")
            return None
    
    def _extract_time_series(self, data: Dict[str, Any]) -> Optional[TimeSeriesData]:
        """
        Extract time series data from input dictionary.
        
        Args:
            data: Input data dictionary
            
        Returns:
            TimeSeriesData object or None if extraction fails
        """
        try:
            # Look for common time series fields
            timestamps = []
            values = []
            
            # Handle different data formats
            if 'timestamp' in data and 'value' in data:
                # Single point - create minimal series
                timestamp = data['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                # Create a simple 2-point series for trend
                timestamps = [timestamp - timedelta(days=1), timestamp]
                values = [data.get('value', 0) * 0.95, data.get('value', 0)]
                
            elif 'history' in data and isinstance(data['history'], list):
                # Historical data points
                for point in data['history']:
                    if isinstance(point, dict) and 'timestamp' in point and 'value' in point:
                        ts = point['timestamp']
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        timestamps.append(ts)
                        values.append(float(point['value']))
                        
            elif 'values' in data and isinstance(data['values'], list):
                # Simple values list - generate timestamps
                values = [float(v) for v in data['values']]
                base_time = datetime.now() - timedelta(days=len(values))
                timestamps = [base_time + timedelta(days=i) for i in range(len(values))]
            
            else:
                # Try to extract from general structure
                for key, value in data.items():
                    if key in ['consumption', 'usage', 'kwh', 'gallons', 'tons'] and isinstance(value, (int, float)):
                        # Create simple trend series
                        now = datetime.now()
                        timestamps = [now - timedelta(days=7), now]
                        values = [float(value) * 0.9, float(value)]
                        break
            
            if len(timestamps) >= 2 and len(values) >= 2:
                return TimeSeriesData(
                    timestamps=timestamps,
                    values=values,
                    metadata={'source': 'extracted', 'original_data': data}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Time series extraction failed: {e}")
            return None
    
    def _generate_simple_forecast(self, 
                                 time_series: TimeSeriesData, 
                                 horizon_days: int) -> Optional[ForecastResult]:
        """
        Generate simple trend-based forecast.
        
        Args:
            time_series: Historical time series data
            horizon_days: Forecast horizon in days
            
        Returns:
            ForecastResult or None if forecasting fails
        """
        try:
            df = time_series.df
            values = df['value'].values
            
            if len(values) < 2:
                return None
            
            # Calculate trend using linear regression
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Determine trend direction
            if abs(slope) < std_err:
                trend_direction = TrendDirection.STABLE
            elif slope > 0:
                trend_direction = TrendDirection.INCREASING
            else:
                trend_direction = TrendDirection.DECREASING
            
            # Generate forecast timestamps
            last_timestamp = time_series.timestamps[-1]
            forecast_timestamps = [
                last_timestamp + timedelta(days=i+1) 
                for i in range(horizon_days)
            ]
            
            # Generate forecast values using trend extrapolation
            last_x = len(values) - 1
            forecast_values = []
            confidence_intervals = []
            
            for i in range(horizon_days):
                future_x = last_x + i + 1
                predicted_value = intercept + slope * future_x
                
                # Add some noise based on historical variance
                std_noise = np.std(values) * 0.1 * (i + 1)  # Increasing uncertainty
                
                # Calculate confidence interval
                margin = 1.96 * std_noise  # 95% confidence interval
                lower_bound = predicted_value - margin
                upper_bound = predicted_value + margin
                
                forecast_values.append(max(0, predicted_value))  # Don't allow negative values
                confidence_intervals.append((max(0, lower_bound), upper_bound))
            
            # Calculate confidence score based on trend significance
            confidence_score = min(self.default_confidence, 1.0 - p_value) if p_value < 1.0 else 0.5
            
            return ForecastResult(
                forecasted_values=forecast_values,
                forecast_timestamps=forecast_timestamps,
                confidence_intervals=confidence_intervals,
                trend=trend_direction,
                forecast_horizon_days=horizon_days,
                model_name="simple_linear_trend",
                confidence_score=confidence_score,
                metadata={
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'historical_std': np.std(values),
                    'historical_mean': np.mean(values)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Simple forecast generation failed: {e}")
            return None


# Caching decorator for expensive operations
def cached_analysis(func):
    """Decorator to cache expensive analysis operations."""
    @lru_cache(maxsize=128)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper