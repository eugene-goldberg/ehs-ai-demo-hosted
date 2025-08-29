"""
Trend Analysis Infrastructure for EHS AI Demo Data Foundation

This module provides comprehensive trend analysis functionality including:
- Statistical trend detection methods (linear regression, seasonal decomposition)
- Pattern recognition algorithms (periodicity, cyclic patterns)
- Anomaly detection (statistical outliers, change point detection)
- Seasonal decomposition (trend, seasonal, residual components)
- Change point detection (CUSUM, Bayesian methods)
- Integration with historical metrics and Neo4j storage
- LLM-ready data formatting for AI analysis
- Multiple analysis techniques for comprehensive insights

Author: Claude AI
Created: 2025-08-28
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
import math
import statistics
from collections import defaultdict

from neo4j import GraphDatabase, Transaction
from neo4j.exceptions import Neo4jError

# Configure logging
logger = logging.getLogger(__name__)


class TrendType(Enum):
    """Trend type enumeration"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    CYCLICAL = "cyclical"
    SEASONAL = "seasonal"
    UNKNOWN = "unknown"


class AnomalyType(Enum):
    """Anomaly type enumeration"""
    OUTLIER = "outlier"
    SPIKE = "spike"
    DIP = "dip"
    CHANGE_POINT = "change_point"
    DRIFT = "drift"
    SEASONAL_ANOMALY = "seasonal_anomaly"


class AnalysisMethod(Enum):
    """Analysis method enumeration"""
    LINEAR_REGRESSION = "linear_regression"
    MOVING_AVERAGE = "moving_average"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    CHANGE_POINT_DETECTION = "change_point_detection"
    STATISTICAL_OUTLIER = "statistical_outlier"
    Z_SCORE = "z_score"
    IQR_METHOD = "iqr_method"
    CUSUM = "cusum"
    BAYESIAN_CHANGE_POINT = "bayesian_change_point"


class Seasonality(Enum):
    """Seasonality type enumeration"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NONE = "none"


@dataclass
class DataPoint:
    """Data point for analysis"""
    timestamp: datetime
    value: float
    metric_name: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'metric_name': self.metric_name,
            'metadata': self.metadata or {}
        }


@dataclass
class TrendAnalysisResult:
    """Trend analysis result"""
    analysis_id: str
    metric_name: str
    trend_type: TrendType
    analysis_method: AnalysisMethod
    confidence_score: float
    slope: Optional[float]
    r_squared: Optional[float]
    significance_level: float
    trend_strength: float
    start_date: datetime
    end_date: datetime
    data_points_count: int
    seasonal_component: Optional[Seasonality]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'analysis_id': self.analysis_id,
            'metric_name': self.metric_name,
            'trend_type': self.trend_type.value,
            'analysis_method': self.analysis_method.value,
            'confidence_score': self.confidence_score,
            'slope': self.slope,
            'r_squared': self.r_squared,
            'significance_level': self.significance_level,
            'trend_strength': self.trend_strength,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'data_points_count': self.data_points_count,
            'seasonal_component': self.seasonal_component.value if self.seasonal_component else None,
            'metadata': self.metadata or {}
        }


@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result"""
    anomaly_id: str
    metric_name: str
    anomaly_type: AnomalyType
    detection_method: AnalysisMethod
    timestamp: datetime
    value: float
    expected_value: Optional[float]
    deviation_score: float
    severity: str  # low, medium, high, critical
    confidence: float
    context_window: int
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'anomaly_id': self.anomaly_id,
            'metric_name': self.metric_name,
            'anomaly_type': self.anomaly_type.value,
            'detection_method': self.detection_method.value,
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'expected_value': self.expected_value,
            'deviation_score': self.deviation_score,
            'severity': self.severity,
            'confidence': self.confidence,
            'context_window': self.context_window,
            'metadata': self.metadata or {}
        }


@dataclass
class SeasonalDecompositionResult:
    """Seasonal decomposition result"""
    decomposition_id: str
    metric_name: str
    period: int
    trend_component: List[float]
    seasonal_component: List[float]
    residual_component: List[float]
    seasonality_strength: float
    trend_strength: float
    timestamps: List[datetime]
    analysis_date: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'decomposition_id': self.decomposition_id,
            'metric_name': self.metric_name,
            'period': self.period,
            'trend_component': self.trend_component,
            'seasonal_component': self.seasonal_component,
            'residual_component': self.residual_component,
            'seasonality_strength': self.seasonality_strength,
            'trend_strength': self.trend_strength,
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'analysis_date': self.analysis_date.isoformat(),
            'metadata': self.metadata or {}
        }


@dataclass
class ChangePointResult:
    """Change point detection result"""
    change_point_id: str
    metric_name: str
    change_point_timestamp: datetime
    detection_method: AnalysisMethod
    confidence_score: float
    magnitude_of_change: float
    before_mean: float
    after_mean: float
    before_variance: float
    after_variance: float
    statistical_significance: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'change_point_id': self.change_point_id,
            'metric_name': self.metric_name,
            'change_point_timestamp': self.change_point_timestamp.isoformat(),
            'detection_method': self.detection_method.value,
            'confidence_score': self.confidence_score,
            'magnitude_of_change': self.magnitude_of_change,
            'before_mean': self.before_mean,
            'after_mean': self.after_mean,
            'before_variance': self.before_variance,
            'after_variance': self.after_variance,
            'statistical_significance': self.statistical_significance,
            'metadata': self.metadata or {}
        }


class TrendAnalysisSystem:
    """
    Comprehensive Trend Analysis System
    
    Provides advanced statistical analysis, pattern recognition, and anomaly detection
    capabilities with Neo4j integration for persistent storage and LLM-ready formatting.
    """
    
    def __init__(self, driver: GraphDatabase.driver, database: str = "neo4j"):
        """
        Initialize the Trend Analysis System
        
        Args:
            driver: Neo4j database driver
            database: Database name (default: "neo4j")
        """
        self.driver = driver
        self.database = database
        self._create_constraints()
    
    def _create_constraints(self):
        """Create necessary constraints and indexes for trend analysis"""
        constraints = [
            "CREATE CONSTRAINT trend_analysis_id_unique IF NOT EXISTS FOR (ta:TrendAnalysis) REQUIRE ta.analysis_id IS UNIQUE",
            "CREATE CONSTRAINT anomaly_id_unique IF NOT EXISTS FOR (a:Anomaly) REQUIRE a.anomaly_id IS UNIQUE",
            "CREATE CONSTRAINT decomposition_id_unique IF NOT EXISTS FOR (sd:SeasonalDecomposition) REQUIRE sd.decomposition_id IS UNIQUE",
            "CREATE CONSTRAINT change_point_id_unique IF NOT EXISTS FOR (cp:ChangePoint) REQUIRE cp.change_point_id IS UNIQUE",
            "CREATE INDEX trend_analysis_metric_index IF NOT EXISTS FOR (ta:TrendAnalysis) ON (ta.metric_name)",
            "CREATE INDEX trend_analysis_date_index IF NOT EXISTS FOR (ta:TrendAnalysis) ON (ta.start_date, ta.end_date)",
            "CREATE INDEX anomaly_metric_index IF NOT EXISTS FOR (a:Anomaly) ON (a.metric_name)",
            "CREATE INDEX anomaly_timestamp_index IF NOT EXISTS FOR (a:Anomaly) ON (a.timestamp)",
            "CREATE INDEX anomaly_severity_index IF NOT EXISTS FOR (a:Anomaly) ON (a.severity)",
            "CREATE INDEX change_point_metric_index IF NOT EXISTS FOR (cp:ChangePoint) ON (cp.metric_name)",
            "CREATE INDEX change_point_timestamp_index IF NOT EXISTS FOR (cp:ChangePoint) ON (cp.change_point_timestamp)",
            "CREATE INDEX data_point_timestamp_index IF NOT EXISTS FOR (dp:DataPoint) ON (dp.timestamp)",
            "CREATE INDEX data_point_metric_index IF NOT EXISTS FOR (dp:DataPoint) ON (dp.metric_name)"
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint/index: {constraint}")
                except Neo4jError as e:
                    if "equivalent" not in str(e).lower():
                        logger.warning(f"Failed to create constraint: {e}")
    
    # Statistical Trend Detection Methods
    
    def linear_regression_analysis(self, data_points: List[DataPoint]) -> TrendAnalysisResult:
        """
        Perform linear regression trend analysis
        
        Args:
            data_points: List of data points to analyze
            
        Returns:
            TrendAnalysisResult with linear regression findings
        """
        try:
            if len(data_points) < 3:
                raise ValueError("Need at least 3 data points for linear regression")
            
            # Convert to numerical arrays
            timestamps = [dp.timestamp for dp in data_points]
            values = [dp.value for dp in data_points]
            
            # Convert timestamps to numerical values (days since first timestamp)
            base_time = timestamps[0]
            x = [(ts - base_time).days for ts in timestamps]
            y = values
            
            n = len(x)
            
            # Calculate linear regression parameters
            x_mean = sum(x) / n
            y_mean = sum(y) / n
            
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                slope = 0
                r_squared = 0
            else:
                slope = numerator / denominator
                
                # Calculate R-squared
                y_pred = [slope * (x[i] - x_mean) + y_mean for i in range(n)]
                ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
                ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
                
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend type
            if abs(slope) < 0.01:
                trend_type = TrendType.STABLE
            elif slope > 0:
                trend_type = TrendType.INCREASING
            else:
                trend_type = TrendType.DECREASING
            
            # Calculate confidence and significance
            confidence_score = min(1.0, r_squared + 0.1)  # Adjust for conservative estimate
            trend_strength = abs(slope)
            significance_level = max(0.01, 1.0 - r_squared)
            
            analysis_id = f"trend_lr_{uuid.uuid4().hex[:8]}"
            
            return TrendAnalysisResult(
                analysis_id=analysis_id,
                metric_name=data_points[0].metric_name,
                trend_type=trend_type,
                analysis_method=AnalysisMethod.LINEAR_REGRESSION,
                confidence_score=confidence_score,
                slope=slope,
                r_squared=r_squared,
                significance_level=significance_level,
                trend_strength=trend_strength,
                start_date=timestamps[0],
                end_date=timestamps[-1],
                data_points_count=len(data_points),
                seasonal_component=None,
                metadata={
                    'x_mean': x_mean,
                    'y_mean': y_mean,
                    'intercept': y_mean - slope * x_mean
                }
            )
            
        except Exception as e:
            logger.error(f"Linear regression analysis failed: {e}")
            raise
    
    def moving_average_trend_analysis(self, data_points: List[DataPoint], window_size: int = 7) -> TrendAnalysisResult:
        """
        Perform moving average trend analysis
        
        Args:
            data_points: List of data points to analyze
            window_size: Size of moving average window
            
        Returns:
            TrendAnalysisResult with moving average findings
        """
        try:
            if len(data_points) < window_size * 2:
                raise ValueError(f"Need at least {window_size * 2} data points for moving average analysis")
            
            values = [dp.value for dp in data_points]
            
            # Calculate moving averages
            moving_averages = []
            for i in range(window_size, len(values)):
                window = values[i-window_size:i]
                ma = sum(window) / len(window)
                moving_averages.append(ma)
            
            # Calculate trend from moving averages
            if len(moving_averages) < 2:
                trend_type = TrendType.STABLE
                slope = 0
                trend_strength = 0
            else:
                first_half_avg = sum(moving_averages[:len(moving_averages)//2]) / (len(moving_averages)//2)
                second_half_avg = sum(moving_averages[len(moving_averages)//2:]) / (len(moving_averages) - len(moving_averages)//2)
                
                slope = (second_half_avg - first_half_avg) / (len(moving_averages) / 2)
                
                if abs(slope) < 0.01:
                    trend_type = TrendType.STABLE
                elif slope > 0:
                    trend_type = TrendType.INCREASING
                else:
                    trend_type = TrendType.DECREASING
                
                trend_strength = abs(slope)
            
            # Calculate volatility
            ma_std = statistics.stdev(moving_averages) if len(moving_averages) > 1 else 0
            value_std = statistics.stdev(values)
            
            smoothness_ratio = 1 - (ma_std / value_std) if value_std > 0 else 1
            confidence_score = min(1.0, smoothness_ratio + 0.2)
            
            analysis_id = f"trend_ma_{uuid.uuid4().hex[:8]}"
            
            return TrendAnalysisResult(
                analysis_id=analysis_id,
                metric_name=data_points[0].metric_name,
                trend_type=trend_type,
                analysis_method=AnalysisMethod.MOVING_AVERAGE,
                confidence_score=confidence_score,
                slope=slope,
                r_squared=None,
                significance_level=0.05,
                trend_strength=trend_strength,
                start_date=data_points[0].timestamp,
                end_date=data_points[-1].timestamp,
                data_points_count=len(data_points),
                seasonal_component=None,
                metadata={
                    'window_size': window_size,
                    'smoothness_ratio': smoothness_ratio,
                    'volatility': ma_std
                }
            )
            
        except Exception as e:
            logger.error(f"Moving average analysis failed: {e}")
            raise
    
    # Seasonal Decomposition
    
    def seasonal_decomposition(self, data_points: List[DataPoint], period: int = None) -> SeasonalDecompositionResult:
        """
        Perform seasonal decomposition analysis
        
        Args:
            data_points: List of data points to analyze
            period: Seasonal period (auto-detected if None)
            
        Returns:
            SeasonalDecompositionResult with decomposition components
        """
        try:
            if len(data_points) < 14:  # Minimum for meaningful decomposition
                raise ValueError("Need at least 14 data points for seasonal decomposition")
            
            values = [dp.value for dp in data_points]
            timestamps = [dp.timestamp for dp in data_points]
            
            # Auto-detect period if not provided
            if period is None:
                period = self._detect_period(values)
            
            if period < 2 or period >= len(values) // 2:
                period = min(7, len(values) // 3)  # Default weekly or smaller
            
            # Simple seasonal decomposition using moving averages
            trend_component = self._extract_trend(values, period)
            seasonal_component = self._extract_seasonal(values, trend_component, period)
            residual_component = [values[i] - trend_component[i] - seasonal_component[i] 
                                for i in range(len(values))]
            
            # Calculate strength metrics
            seasonality_strength = self._calculate_seasonality_strength(values, seasonal_component)
            trend_strength = self._calculate_trend_strength(values, trend_component)
            
            decomposition_id = f"decomp_{uuid.uuid4().hex[:8]}"
            
            return SeasonalDecompositionResult(
                decomposition_id=decomposition_id,
                metric_name=data_points[0].metric_name,
                period=period,
                trend_component=trend_component,
                seasonal_component=seasonal_component,
                residual_component=residual_component,
                seasonality_strength=seasonality_strength,
                trend_strength=trend_strength,
                timestamps=timestamps,
                analysis_date=datetime.now(),
                metadata={
                    'residual_variance': statistics.variance(residual_component) if len(residual_component) > 1 else 0,
                    'original_variance': statistics.variance(values) if len(values) > 1 else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Seasonal decomposition failed: {e}")
            raise
    
    def _detect_period(self, values: List[float]) -> int:
        """Detect seasonal period using autocorrelation"""
        try:
            n = len(values)
            max_period = min(n // 3, 30)  # Don't look for periods longer than n/3 or 30
            
            best_period = 7  # Default weekly
            best_correlation = 0
            
            for period in range(2, max_period + 1):
                correlation = self._calculate_autocorrelation(values, period)
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_period = period
            
            return best_period if best_correlation > 0.3 else 7
            
        except Exception:
            return 7  # Default fallback
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        try:
            if lag >= len(values):
                return 0
            
            n = len(values) - lag
            if n <= 1:
                return 0
            
            mean_val = sum(values) / len(values)
            
            numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n))
            denominator = sum((values[i] - mean_val) ** 2 for i in range(len(values)))
            
            return numerator / denominator if denominator > 0 else 0
            
        except Exception:
            return 0
    
    def _extract_trend(self, values: List[float], period: int) -> List[float]:
        """Extract trend component using centered moving average"""
        trend = []
        half_period = period // 2
        
        for i in range(len(values)):
            start_idx = max(0, i - half_period)
            end_idx = min(len(values), i + half_period + 1)
            window = values[start_idx:end_idx]
            trend.append(sum(window) / len(window))
        
        return trend
    
    def _extract_seasonal(self, values: List[float], trend: List[float], period: int) -> List[float]:
        """Extract seasonal component"""
        # Calculate seasonal indices
        seasonal_sums = defaultdict(list)
        
        for i in range(len(values)):
            seasonal_idx = i % period
            if trend[i] != 0:
                seasonal_sums[seasonal_idx].append(values[i] - trend[i])
        
        # Average seasonal effects
        seasonal_averages = {}
        for idx in range(period):
            if idx in seasonal_sums and seasonal_sums[idx]:
                seasonal_averages[idx] = sum(seasonal_sums[idx]) / len(seasonal_sums[idx])
            else:
                seasonal_averages[idx] = 0
        
        # Create seasonal component
        seasonal = []
        for i in range(len(values)):
            seasonal_idx = i % period
            seasonal.append(seasonal_averages[seasonal_idx])
        
        return seasonal
    
    def _calculate_seasonality_strength(self, values: List[float], seasonal: List[float]) -> float:
        """Calculate strength of seasonal component"""
        try:
            if len(values) <= 1 or len(seasonal) <= 1:
                return 0.0
            
            values_var = statistics.variance(values)
            seasonal_var = statistics.variance(seasonal)
            
            if values_var == 0:
                return 0.0
            
            return min(1.0, seasonal_var / values_var)
            
        except Exception:
            return 0.0
    
    def _calculate_trend_strength(self, values: List[float], trend: List[float]) -> float:
        """Calculate strength of trend component"""
        try:
            if len(values) <= 1 or len(trend) <= 1:
                return 0.0
            
            values_var = statistics.variance(values)
            trend_var = statistics.variance(trend)
            
            if values_var == 0:
                return 0.0
            
            return min(1.0, trend_var / values_var)
            
        except Exception:
            return 0.0
    
    # Anomaly Detection Methods
    
    def z_score_anomaly_detection(self, data_points: List[DataPoint], threshold: float = 3.0) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Z-score method
        
        Args:
            data_points: List of data points to analyze
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        try:
            if len(data_points) < 5:
                return []
            
            values = [dp.value for dp in data_points]
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val == 0:
                return []
            
            anomalies = []
            
            for dp in data_points:
                z_score = abs(dp.value - mean_val) / std_val
                
                if z_score > threshold:
                    severity = self._determine_anomaly_severity(z_score, threshold)
                    anomaly_type = AnomalyType.SPIKE if dp.value > mean_val else AnomalyType.DIP
                    
                    anomaly_id = f"anomaly_z_{uuid.uuid4().hex[:8]}"
                    
                    anomalies.append(AnomalyDetectionResult(
                        anomaly_id=anomaly_id,
                        metric_name=dp.metric_name,
                        anomaly_type=anomaly_type,
                        detection_method=AnalysisMethod.Z_SCORE,
                        timestamp=dp.timestamp,
                        value=dp.value,
                        expected_value=mean_val,
                        deviation_score=z_score,
                        severity=severity,
                        confidence=min(1.0, z_score / threshold),
                        context_window=len(data_points),
                        metadata={
                            'mean': mean_val,
                            'std': std_val,
                            'threshold': threshold
                        }
                    ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Z-score anomaly detection failed: {e}")
            return []
    
    def iqr_anomaly_detection(self, data_points: List[DataPoint], iqr_multiplier: float = 1.5) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Interquartile Range (IQR) method
        
        Args:
            data_points: List of data points to analyze
            iqr_multiplier: IQR multiplier for outlier detection
            
        Returns:
            List of detected anomalies
        """
        try:
            if len(data_points) < 5:
                return []
            
            values = [dp.value for dp in data_points]
            values_sorted = sorted(values)
            n = len(values_sorted)
            
            # Calculate quartiles
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            q1 = values_sorted[q1_idx]
            q3 = values_sorted[q3_idx]
            iqr = q3 - q1
            
            if iqr == 0:
                return []
            
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            anomalies = []
            
            for dp in data_points:
                if dp.value < lower_bound or dp.value > upper_bound:
                    # Calculate deviation score
                    if dp.value < lower_bound:
                        deviation_score = (lower_bound - dp.value) / iqr
                        anomaly_type = AnomalyType.DIP
                    else:
                        deviation_score = (dp.value - upper_bound) / iqr
                        anomaly_type = AnomalyType.SPIKE
                    
                    severity = self._determine_anomaly_severity(deviation_score, iqr_multiplier)
                    
                    anomaly_id = f"anomaly_iqr_{uuid.uuid4().hex[:8]}"
                    
                    median_val = values_sorted[n // 2]
                    
                    anomalies.append(AnomalyDetectionResult(
                        anomaly_id=anomaly_id,
                        metric_name=dp.metric_name,
                        anomaly_type=anomaly_type,
                        detection_method=AnalysisMethod.IQR_METHOD,
                        timestamp=dp.timestamp,
                        value=dp.value,
                        expected_value=median_val,
                        deviation_score=deviation_score,
                        severity=severity,
                        confidence=min(1.0, deviation_score / iqr_multiplier),
                        context_window=len(data_points),
                        metadata={
                            'q1': q1,
                            'q3': q3,
                            'iqr': iqr,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }
                    ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"IQR anomaly detection failed: {e}")
            return []
    
    def _determine_anomaly_severity(self, score: float, threshold: float) -> str:
        """Determine anomaly severity based on score"""
        if score > threshold * 3:
            return "critical"
        elif score > threshold * 2:
            return "high"
        elif score > threshold * 1.5:
            return "medium"
        else:
            return "low"
    
    # Change Point Detection
    
    def cusum_change_point_detection(self, data_points: List[DataPoint], threshold: float = 5.0) -> List[ChangePointResult]:
        """
        Detect change points using CUSUM algorithm
        
        Args:
            data_points: List of data points to analyze
            threshold: CUSUM threshold for change point detection
            
        Returns:
            List of detected change points
        """
        try:
            if len(data_points) < 10:
                return []
            
            values = [dp.value for dp in data_points]
            timestamps = [dp.timestamp for dp in data_points]
            
            # Calculate mean and standard deviation
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 1
            
            # CUSUM variables
            cusum_pos = 0
            cusum_neg = 0
            change_points = []
            
            for i in range(1, len(values)):
                # Normalized value
                normalized_val = (values[i] - mean_val) / std_val
                
                # Update CUSUM
                cusum_pos = max(0, cusum_pos + normalized_val - 0.5)
                cusum_neg = max(0, cusum_neg - normalized_val - 0.5)
                
                # Check for change point
                if cusum_pos > threshold or cusum_neg > threshold:
                    # Calculate statistics before and after change point
                    before_values = values[:i]
                    after_values = values[i:]
                    
                    if len(before_values) >= 2 and len(after_values) >= 2:
                        before_mean = statistics.mean(before_values)
                        after_mean = statistics.mean(after_values)
                        before_var = statistics.variance(before_values)
                        after_var = statistics.variance(after_values)
                        
                        magnitude = abs(after_mean - before_mean)
                        confidence = max(cusum_pos, cusum_neg) / threshold
                        
                        change_point_id = f"cp_cusum_{uuid.uuid4().hex[:8]}"
                        
                        change_points.append(ChangePointResult(
                            change_point_id=change_point_id,
                            metric_name=data_points[0].metric_name,
                            change_point_timestamp=timestamps[i],
                            detection_method=AnalysisMethod.CUSUM,
                            confidence_score=min(1.0, confidence),
                            magnitude_of_change=magnitude,
                            before_mean=before_mean,
                            after_mean=after_mean,
                            before_variance=before_var,
                            after_variance=after_var,
                            statistical_significance=0.05,  # Assume 5% significance level
                            metadata={
                                'cusum_pos': cusum_pos,
                                'cusum_neg': cusum_neg,
                                'threshold': threshold,
                                'change_point_index': i
                            }
                        ))
                    
                    # Reset CUSUM
                    cusum_pos = 0
                    cusum_neg = 0
            
            return change_points
            
        except Exception as e:
            logger.error(f"CUSUM change point detection failed: {e}")
            return []
    
    # Data Storage and Retrieval
    
    def store_trend_analysis(self, analysis: TrendAnalysisResult) -> bool:
        """
        Store trend analysis result in Neo4j
        
        Args:
            analysis: TrendAnalysisResult to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.execute_write(self._store_trend_analysis_tx, analysis)
                
            logger.info(f"Stored trend analysis: {analysis.analysis_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store trend analysis: {e}")
            return False
    
    def _store_trend_analysis_tx(self, tx: Transaction, analysis: TrendAnalysisResult):
        """Transaction to store trend analysis"""
        query = """
        CREATE (ta:TrendAnalysis {
            analysis_id: $analysis_id,
            metric_name: $metric_name,
            trend_type: $trend_type,
            analysis_method: $analysis_method,
            confidence_score: $confidence_score,
            slope: $slope,
            r_squared: $r_squared,
            significance_level: $significance_level,
            trend_strength: $trend_strength,
            start_date: datetime($start_date),
            end_date: datetime($end_date),
            data_points_count: $data_points_count,
            seasonal_component: $seasonal_component,
            metadata: $metadata,
            created_date: datetime()
        })
        RETURN ta
        """
        
        params = analysis.to_dict()
        return tx.run(query, params)
    
    def store_anomaly_detection(self, anomaly: AnomalyDetectionResult) -> bool:
        """
        Store anomaly detection result in Neo4j
        
        Args:
            anomaly: AnomalyDetectionResult to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.execute_write(self._store_anomaly_detection_tx, anomaly)
                
            logger.info(f"Stored anomaly detection: {anomaly.anomaly_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store anomaly detection: {e}")
            return False
    
    def _store_anomaly_detection_tx(self, tx: Transaction, anomaly: AnomalyDetectionResult):
        """Transaction to store anomaly detection"""
        query = """
        CREATE (a:Anomaly {
            anomaly_id: $anomaly_id,
            metric_name: $metric_name,
            anomaly_type: $anomaly_type,
            detection_method: $detection_method,
            timestamp: datetime($timestamp),
            value: $value,
            expected_value: $expected_value,
            deviation_score: $deviation_score,
            severity: $severity,
            confidence: $confidence,
            context_window: $context_window,
            metadata: $metadata,
            created_date: datetime()
        })
        RETURN a
        """
        
        params = anomaly.to_dict()
        return tx.run(query, params)
    
    def store_seasonal_decomposition(self, decomposition: SeasonalDecompositionResult) -> bool:
        """
        Store seasonal decomposition result in Neo4j
        
        Args:
            decomposition: SeasonalDecompositionResult to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.execute_write(self._store_seasonal_decomposition_tx, decomposition)
                
            logger.info(f"Stored seasonal decomposition: {decomposition.decomposition_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store seasonal decomposition: {e}")
            return False
    
    def _store_seasonal_decomposition_tx(self, tx: Transaction, decomposition: SeasonalDecompositionResult):
        """Transaction to store seasonal decomposition"""
        query = """
        CREATE (sd:SeasonalDecomposition {
            decomposition_id: $decomposition_id,
            metric_name: $metric_name,
            period: $period,
            trend_component: $trend_component,
            seasonal_component: $seasonal_component,
            residual_component: $residual_component,
            seasonality_strength: $seasonality_strength,
            trend_strength: $trend_strength,
            timestamps: $timestamps,
            analysis_date: datetime($analysis_date),
            metadata: $metadata,
            created_date: datetime()
        })
        RETURN sd
        """
        
        params = decomposition.to_dict()
        return tx.run(query, params)
    
    def store_change_point(self, change_point: ChangePointResult) -> bool:
        """
        Store change point detection result in Neo4j
        
        Args:
            change_point: ChangePointResult to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.execute_write(self._store_change_point_tx, change_point)
                
            logger.info(f"Stored change point: {change_point.change_point_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store change point: {e}")
            return False
    
    def _store_change_point_tx(self, tx: Transaction, change_point: ChangePointResult):
        """Transaction to store change point"""
        query = """
        CREATE (cp:ChangePoint {
            change_point_id: $change_point_id,
            metric_name: $metric_name,
            change_point_timestamp: datetime($change_point_timestamp),
            detection_method: $detection_method,
            confidence_score: $confidence_score,
            magnitude_of_change: $magnitude_of_change,
            before_mean: $before_mean,
            after_mean: $after_mean,
            before_variance: $before_variance,
            after_variance: $after_variance,
            statistical_significance: $statistical_significance,
            metadata: $metadata,
            created_date: datetime()
        })
        RETURN cp
        """
        
        params = change_point.to_dict()
        return tx.run(query, params)
    
    # Comprehensive Analysis Methods
    
    def comprehensive_trend_analysis(self, metric_name: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis using multiple methods
        
        Args:
            metric_name: Name of metric to analyze
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            # Retrieve data points from Neo4j
            data_points = self._get_data_points(metric_name, start_date, end_date)
            
            if len(data_points) < 5:
                return {"error": "Insufficient data points for analysis"}
            
            results = {
                "metric_name": metric_name,
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "data_points_count": len(data_points),
                "analyses": {}
            }
            
            # Linear regression analysis
            try:
                lr_result = self.linear_regression_analysis(data_points)
                results["analyses"]["linear_regression"] = lr_result.to_dict()
                self.store_trend_analysis(lr_result)
            except Exception as e:
                results["analyses"]["linear_regression"] = {"error": str(e)}
            
            # Moving average analysis
            try:
                ma_result = self.moving_average_trend_analysis(data_points)
                results["analyses"]["moving_average"] = ma_result.to_dict()
                self.store_trend_analysis(ma_result)
            except Exception as e:
                results["analyses"]["moving_average"] = {"error": str(e)}
            
            # Seasonal decomposition
            if len(data_points) >= 14:
                try:
                    decomp_result = self.seasonal_decomposition(data_points)
                    results["analyses"]["seasonal_decomposition"] = decomp_result.to_dict()
                    self.store_seasonal_decomposition(decomp_result)
                except Exception as e:
                    results["analyses"]["seasonal_decomposition"] = {"error": str(e)}
            
            # Anomaly detection
            try:
                z_anomalies = self.z_score_anomaly_detection(data_points)
                iqr_anomalies = self.iqr_anomaly_detection(data_points)
                
                results["anomalies"] = {
                    "z_score_anomalies": [a.to_dict() for a in z_anomalies],
                    "iqr_anomalies": [a.to_dict() for a in iqr_anomalies]
                }
                
                # Store anomalies
                for anomaly in z_anomalies + iqr_anomalies:
                    self.store_anomaly_detection(anomaly)
                    
            except Exception as e:
                results["anomalies"] = {"error": str(e)}
            
            # Change point detection
            try:
                change_points = self.cusum_change_point_detection(data_points)
                results["change_points"] = [cp.to_dict() for cp in change_points]
                
                # Store change points
                for cp in change_points:
                    self.store_change_point(cp)
                    
            except Exception as e:
                results["change_points"] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive trend analysis failed: {e}")
            return {"error": str(e)}
    
    def _get_data_points(self, metric_name: str, start_date: datetime, end_date: datetime) -> List[DataPoint]:
        """Retrieve data points from Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (dp:DataPoint)
                    WHERE dp.metric_name = $metric_name
                    AND dp.timestamp >= datetime($start_date)
                    AND dp.timestamp <= datetime($end_date)
                    RETURN dp.timestamp as timestamp, dp.value as value, dp.metric_name as metric_name, dp.metadata as metadata
                    ORDER BY dp.timestamp
                    """,
                    metric_name=metric_name,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )
                
                data_points = []
                for record in result:
                    data_points.append(DataPoint(
                        timestamp=record["timestamp"],
                        value=record["value"],
                        metric_name=record["metric_name"],
                        metadata=record.get("metadata", {})
                    ))
                
                return data_points
                
        except Exception as e:
            logger.error(f"Failed to retrieve data points: {e}")
            return []
    
    # LLM-Ready Data Formatting
    
    def format_for_llm_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format analysis results for LLM consumption
        
        Args:
            analysis_results: Raw analysis results
            
        Returns:
            LLM-formatted analysis data
        """
        try:
            formatted = {
                "summary": {
                    "metric_name": analysis_results.get("metric_name", "Unknown"),
                    "analysis_period": analysis_results.get("analysis_period", {}),
                    "data_points": analysis_results.get("data_points_count", 0)
                },
                "key_findings": [],
                "insights": [],
                "recommendations": [],
                "technical_details": analysis_results.get("analyses", {})
            }
            
            # Extract key findings from analyses
            analyses = analysis_results.get("analyses", {})
            
            # Linear regression insights
            if "linear_regression" in analyses and "error" not in analyses["linear_regression"]:
                lr = analyses["linear_regression"]
                trend_desc = f"{lr['trend_type']} trend"
                if lr.get("slope"):
                    trend_desc += f" (slope: {lr['slope']:.4f})"
                if lr.get("r_squared"):
                    trend_desc += f" with R² of {lr['r_squared']:.3f}"
                
                formatted["key_findings"].append({
                    "type": "trend",
                    "method": "linear_regression",
                    "finding": trend_desc,
                    "confidence": lr.get("confidence_score", 0)
                })
            
            # Anomaly insights
            anomalies = analysis_results.get("anomalies", {})
            total_anomalies = 0
            high_severity_count = 0
            
            for method in ["z_score_anomalies", "iqr_anomalies"]:
                if method in anomalies and not isinstance(anomalies[method], dict):
                    method_anomalies = anomalies[method]
                    total_anomalies += len(method_anomalies)
                    high_severity_count += sum(1 for a in method_anomalies if a.get("severity") in ["high", "critical"])
            
            if total_anomalies > 0:
                formatted["key_findings"].append({
                    "type": "anomalies",
                    "finding": f"Detected {total_anomalies} anomalies",
                    "high_severity": high_severity_count,
                    "impact": "high" if high_severity_count > 0 else "medium"
                })
            
            # Change points insights
            change_points = analysis_results.get("change_points", [])
            if isinstance(change_points, list) and len(change_points) > 0:
                formatted["key_findings"].append({
                    "type": "change_points",
                    "finding": f"Detected {len(change_points)} significant change points",
                    "impact": "high"
                })
            
            # Seasonal insights
            if "seasonal_decomposition" in analyses and "error" not in analyses["seasonal_decomposition"]:
                decomp = analyses["seasonal_decomposition"]
                seasonality_strength = decomp.get("seasonality_strength", 0)
                if seasonality_strength > 0.3:
                    formatted["key_findings"].append({
                        "type": "seasonality",
                        "finding": f"Strong seasonal pattern detected (strength: {seasonality_strength:.3f})",
                        "period": decomp.get("period", "unknown")
                    })
            
            # Generate insights and recommendations
            formatted["insights"] = self._generate_insights(formatted["key_findings"])
            formatted["recommendations"] = self._generate_recommendations(formatted["key_findings"])
            
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format for LLM analysis: {e}")
            return {"error": str(e)}
    
    def _generate_insights(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from findings"""
        insights = []
        
        trend_findings = [f for f in findings if f.get("type") == "trend"]
        anomaly_findings = [f for f in findings if f.get("type") == "anomalies"]
        seasonal_findings = [f for f in findings if f.get("type") == "seasonality"]
        change_point_findings = [f for f in findings if f.get("type") == "change_points"]
        
        if trend_findings:
            trend = trend_findings[0]
            if "increasing" in trend["finding"]:
                insights.append("The metric shows an upward trajectory, indicating positive performance or growth.")
            elif "decreasing" in trend["finding"]:
                insights.append("The metric shows a downward trajectory, which may require attention or intervention.")
            elif "stable" in trend["finding"]:
                insights.append("The metric shows stability over the analysis period.")
        
        if anomaly_findings:
            anomaly = anomaly_findings[0]
            if anomaly.get("high_severity", 0) > 0:
                insights.append("High-severity anomalies detected, indicating potential system issues or unusual events.")
            else:
                insights.append("Minor anomalies detected, which may represent normal variation or minor irregularities.")
        
        if seasonal_findings:
            insights.append("Strong seasonal patterns detected, suggesting predictable cyclical behavior.")
        
        if change_point_findings:
            insights.append("Significant change points detected, indicating structural shifts in the underlying process.")
        
        return insights
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations from findings"""
        recommendations = []
        
        anomaly_findings = [f for f in findings if f.get("type") == "anomalies"]
        change_point_findings = [f for f in findings if f.get("type") == "change_points"]
        seasonal_findings = [f for f in findings if f.get("type") == "seasonality"]
        
        if anomaly_findings:
            anomaly = anomaly_findings[0]
            if anomaly.get("high_severity", 0) > 0:
                recommendations.append("Investigate high-severity anomalies immediately to identify root causes.")
                recommendations.append("Implement monitoring and alerting for similar anomaly patterns.")
            else:
                recommendations.append("Review anomalies for potential process improvements.")
        
        if change_point_findings:
            recommendations.append("Analyze change points to understand what caused the structural shifts.")
            recommendations.append("Update forecasting models to account for structural changes.")
        
        if seasonal_findings:
            recommendations.append("Leverage seasonal patterns for improved forecasting and planning.")
            recommendations.append("Adjust operational strategies to account for seasonal variations.")
        
        if not any(findings):
            recommendations.append("Consider collecting more data or using different metrics for better insights.")
        
        return recommendations
    
    # Query and Reporting Methods
    
    def get_trend_analyses(self, metric_name: Optional[str] = None, limit: int = 100) -> List[TrendAnalysisResult]:
        """
        Get stored trend analyses
        
        Args:
            metric_name: Optional metric filter
            limit: Maximum number of results
            
        Returns:
            List of TrendAnalysisResult objects
        """
        try:
            query = "MATCH (ta:TrendAnalysis)"
            params = {}
            
            if metric_name:
                query += " WHERE ta.metric_name = $metric_name"
                params["metric_name"] = metric_name
            
            query += " RETURN ta ORDER BY ta.created_date DESC LIMIT $limit"
            params["limit"] = limit
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                
                analyses = []
                for record in result:
                    data = dict(record["ta"])
                    analyses.append(self._dict_to_trend_analysis(data))
                
                return analyses
                
        except Exception as e:
            logger.error(f"Failed to get trend analyses: {e}")
            return []
    
    def get_recent_anomalies(self, metric_name: Optional[str] = None, days: int = 7) -> List[AnomalyDetectionResult]:
        """
        Get recent anomalies
        
        Args:
            metric_name: Optional metric filter
            days: Number of days to look back
            
        Returns:
            List of AnomalyDetectionResult objects
        """
        try:
            query = """
            MATCH (a:Anomaly)
            WHERE a.timestamp >= datetime() - duration({days: $days})
            """
            params = {"days": days}
            
            if metric_name:
                query += " AND a.metric_name = $metric_name"
                params["metric_name"] = metric_name
            
            query += " RETURN a ORDER BY a.timestamp DESC"
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                
                anomalies = []
                for record in result:
                    data = dict(record["a"])
                    anomalies.append(self._dict_to_anomaly(data))
                
                return anomalies
                
        except Exception as e:
            logger.error(f"Failed to get recent anomalies: {e}")
            return []
    
    def _dict_to_trend_analysis(self, data: Dict[str, Any]) -> TrendAnalysisResult:
        """Convert dictionary to TrendAnalysisResult"""
        return TrendAnalysisResult(
            analysis_id=data["analysis_id"],
            metric_name=data["metric_name"],
            trend_type=TrendType(data["trend_type"]),
            analysis_method=AnalysisMethod(data["analysis_method"]),
            confidence_score=data["confidence_score"],
            slope=data.get("slope"),
            r_squared=data.get("r_squared"),
            significance_level=data["significance_level"],
            trend_strength=data["trend_strength"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            data_points_count=data["data_points_count"],
            seasonal_component=Seasonality(data["seasonal_component"]) if data.get("seasonal_component") else None,
            metadata=data.get("metadata", {})
        )
    
    def _dict_to_anomaly(self, data: Dict[str, Any]) -> AnomalyDetectionResult:
        """Convert dictionary to AnomalyDetectionResult"""
        return AnomalyDetectionResult(
            anomaly_id=data["anomaly_id"],
            metric_name=data["metric_name"],
            anomaly_type=AnomalyType(data["anomaly_type"]),
            detection_method=AnalysisMethod(data["detection_method"]),
            timestamp=data["timestamp"],
            value=data["value"],
            expected_value=data.get("expected_value"),
            deviation_score=data["deviation_score"],
            severity=data["severity"],
            confidence=data["confidence"],
            context_window=data["context_window"],
            metadata=data.get("metadata", {})
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform system health check
        
        Returns:
            Dictionary containing health status
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Test connection
                result = session.run("RETURN 1 as test")
                result.single()
                
                # Get statistics
                stats_result = session.run(
                    """
                    MATCH (ta:TrendAnalysis)
                    OPTIONAL MATCH (a:Anomaly)
                    OPTIONAL MATCH (sd:SeasonalDecomposition)
                    OPTIONAL MATCH (cp:ChangePoint)
                    RETURN count(DISTINCT ta) as trend_analyses_count,
                           count(DISTINCT a) as anomalies_count,
                           count(DISTINCT sd) as decompositions_count,
                           count(DISTINCT cp) as change_points_count
                    """
                )
                
                stats = stats_result.single()
                
                return {
                    "status": "healthy",
                    "database_connection": "ok",
                    "trend_analyses_count": stats["trend_analyses_count"],
                    "anomalies_count": stats["anomalies_count"],
                    "decompositions_count": stats["decompositions_count"],
                    "change_points_count": stats["change_points_count"],
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }