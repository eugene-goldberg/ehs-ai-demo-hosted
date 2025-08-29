"""
Comprehensive Forecast and Prediction System

This module provides advanced forecasting capabilities with multiple models,
confidence intervals, scenario analysis, and performance tracking.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ForecastModel(Enum):
    """Available forecasting models."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    ARIMA = "arima"
    SEASONAL = "seasonal"
    ENSEMBLE = "ensemble"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ForecastPoint:
    """Individual forecast data point."""
    timestamp: datetime
    value: float
    lower_bound: float
    upper_bound: float
    confidence: float
    model_used: str
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class ForecastResult:
    """Complete forecast result with metadata."""
    forecast_points: List[ForecastPoint]
    model_performance: Dict[str, float]
    selected_model: str
    forecast_horizon: int
    generated_at: datetime
    data_quality_score: float
    explanation: str
    alerts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ScenarioInput:
    """Input parameters for scenario analysis."""
    name: str
    parameter_adjustments: Dict[str, float]
    probability: float
    description: str


class BaseForecastModel(ABC):
    """Abstract base class for forecast models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.parameters = {}
        self.performance_metrics = {}
        
    @abstractmethod
    def fit(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Fit the model to historical data."""
        pass
        
    @abstractmethod
    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with confidence intervals."""
        pass
        
    @abstractmethod
    def get_explanation(self) -> str:
        """Get human-readable explanation of the model."""
        pass


class LinearForecastModel(BaseForecastModel):
    """Linear trend forecasting model."""
    
    def __init__(self):
        super().__init__("Linear Trend")
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Fit linear model to data."""
        try:
            # Convert timestamps to numeric values
            time_numeric = np.array([(t - timestamps[0]).total_seconds() / 3600 
                                   for t in timestamps]).reshape(-1, 1)
            
            # Fit the model
            time_scaled = self.scaler.fit_transform(time_numeric)
            self.model.fit(time_scaled, data)
            
            # Calculate residuals for confidence intervals
            predictions = self.model.predict(time_scaled)
            self.residuals = data - predictions
            self.residual_std = np.std(self.residuals)
            
            # Store parameters
            self.parameters = {
                'slope': self.model.coef_[0],
                'intercept': self.model.intercept_,
                'r_squared': self.model.score(time_scaled, data)
            }
            
            self.is_fitted = True
            logger.info(f"Linear model fitted: slope={self.parameters['slope']:.4f}, R²={self.parameters['r_squared']:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting linear model: {e}")
            raise
    
    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate linear predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate future time points
        future_times = np.arange(1, horizon + 1).reshape(-1, 1)
        future_scaled = self.scaler.transform(future_times)
        
        # Make predictions
        predictions = self.model.predict(future_scaled)
        
        # Calculate confidence intervals (95%)
        confidence_factor = 1.96 * self.residual_std
        lower_bounds = predictions - confidence_factor
        upper_bounds = predictions + confidence_factor
        
        return predictions, lower_bounds, upper_bounds
    
    def get_explanation(self) -> str:
        """Explain the linear model."""
        if not self.is_fitted:
            return "Linear model not fitted yet"
        
        slope = self.parameters['slope']
        r_squared = self.parameters['r_squared']
        
        trend_desc = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        quality_desc = "excellent" if r_squared > 0.9 else "good" if r_squared > 0.7 else "moderate" if r_squared > 0.5 else "poor"
        
        return f"Linear trend model shows a {trend_desc} pattern with {quality_desc} fit (R² = {r_squared:.3f}). The trend changes by {abs(slope):.4f} units per time period."


class ExponentialForecastModel(BaseForecastModel):
    """Exponential growth/decay forecasting model."""
    
    def __init__(self):
        super().__init__("Exponential")
        self.growth_rate = 0
        self.base_value = 0
        
    def fit(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Fit exponential model to data."""
        try:
            # Ensure positive values for log transformation
            min_val = np.min(data)
            if min_val <= 0:
                offset = abs(min_val) + 1
                data_positive = data + offset
            else:
                offset = 0
                data_positive = data
            
            # Log transform and fit linear model
            log_data = np.log(data_positive)
            time_numeric = np.arange(len(data))
            
            # Fit linear regression on log-transformed data
            coeffs = np.polyfit(time_numeric, log_data, 1)
            self.growth_rate = coeffs[0]
            self.log_intercept = coeffs[1]
            self.offset = offset
            
            # Calculate fit quality
            log_predictions = self.growth_rate * time_numeric + self.log_intercept
            predictions = np.exp(log_predictions) - self.offset
            
            self.residuals = data - predictions
            self.residual_std = np.std(self.residuals)
            
            # Calculate R-squared
            ss_res = np.sum(self.residuals ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.parameters = {
                'growth_rate': self.growth_rate,
                'base_value': np.exp(self.log_intercept) - self.offset,
                'r_squared': r_squared
            }
            
            self.is_fitted = True
            logger.info(f"Exponential model fitted: growth_rate={self.growth_rate:.4f}, R²={r_squared:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting exponential model: {e}")
            raise
    
    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate exponential predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate future time points
        future_times = np.arange(1, horizon + 1)
        
        # Make exponential predictions
        log_predictions = self.growth_rate * future_times + self.log_intercept
        predictions = np.exp(log_predictions) - self.offset
        
        # Calculate confidence intervals
        confidence_factor = 1.96 * self.residual_std
        lower_bounds = predictions - confidence_factor
        upper_bounds = predictions + confidence_factor
        
        return predictions, lower_bounds, upper_bounds
    
    def get_explanation(self) -> str:
        """Explain the exponential model."""
        if not self.is_fitted:
            return "Exponential model not fitted yet"
        
        growth_rate = self.parameters['growth_rate']
        r_squared = self.parameters['r_squared']
        
        if growth_rate > 0:
            pattern = f"exponential growth at {growth_rate:.2%} per period"
        elif growth_rate < 0:
            pattern = f"exponential decay at {abs(growth_rate):.2%} per period"
        else:
            pattern = "no significant exponential trend"
        
        quality_desc = "excellent" if r_squared > 0.9 else "good" if r_squared > 0.7 else "moderate" if r_squared > 0.5 else "poor"
        
        return f"Exponential model shows {pattern} with {quality_desc} fit (R² = {r_squared:.3f})."


class ARIMALikeForecastModel(BaseForecastModel):
    """Simplified ARIMA-like model using autoregression."""
    
    def __init__(self, order: int = 3):
        super().__init__(f"ARIMA-like (AR-{order})")
        self.order = order
        self.coefficients = None
        
    def fit(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Fit autoregressive model."""
        try:
            if len(data) < self.order + 1:
                raise ValueError(f"Need at least {self.order + 1} data points for AR({self.order}) model")
            
            # Create lagged features
            X = np.array([data[i:i+self.order] for i in range(len(data) - self.order)])
            y = data[self.order:]
            
            # Fit linear regression
            model = Ridge(alpha=0.1)  # Add regularization
            model.fit(X, y)
            
            self.coefficients = model.coef_
            self.intercept = model.intercept_
            self.last_values = data[-self.order:]
            
            # Calculate residuals
            predictions = model.predict(X)
            self.residuals = y - predictions
            self.residual_std = np.std(self.residuals)
            
            # Calculate R-squared
            ss_res = np.sum(self.residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.parameters = {
                'coefficients': self.coefficients.tolist(),
                'intercept': self.intercept,
                'r_squared': r_squared,
                'order': self.order
            }
            
            self.is_fitted = True
            logger.info(f"ARIMA-like model fitted: order={self.order}, R²={r_squared:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA-like model: {e}")
            raise
    
    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate autoregressive predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        current_values = self.last_values.copy()
        
        for _ in range(horizon):
            # Make prediction based on last 'order' values
            next_val = self.intercept + np.dot(self.coefficients, current_values)
            predictions.append(next_val)
            
            # Update current_values for next prediction
            current_values = np.append(current_values[1:], next_val)
        
        predictions = np.array(predictions)
        
        # Calculate expanding confidence intervals
        confidence_factors = np.array([1.96 * self.residual_std * np.sqrt(i) for i in range(1, horizon + 1)])
        lower_bounds = predictions - confidence_factors
        upper_bounds = predictions + confidence_factors
        
        return predictions, lower_bounds, upper_bounds
    
    def get_explanation(self) -> str:
        """Explain the ARIMA-like model."""
        if not self.is_fitted:
            return "ARIMA-like model not fitted yet"
        
        r_squared = self.parameters['r_squared']
        quality_desc = "excellent" if r_squared > 0.9 else "good" if r_squared > 0.7 else "moderate" if r_squared > 0.5 else "poor"
        
        return f"ARIMA-like autoregressive model of order {self.order} with {quality_desc} fit (R² = {r_squared:.3f}). Predictions based on the last {self.order} observed values."


class SeasonalForecastModel(BaseForecastModel):
    """Seasonal decomposition and forecasting model."""
    
    def __init__(self, season_length: int = 24):
        super().__init__("Seasonal")
        self.season_length = season_length
        
    def fit(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Fit seasonal model."""
        try:
            if len(data) < 2 * self.season_length:
                raise ValueError(f"Need at least {2 * self.season_length} data points for seasonal model")
            
            # Simple seasonal decomposition
            self.trend = self._calculate_trend(data)
            self.seasonal = self._calculate_seasonal(data, self.trend)
            self.residuals = data - self.trend - self.seasonal
            self.residual_std = np.std(self.residuals)
            
            # Calculate fit quality
            fitted_values = self.trend + self.seasonal
            ss_res = np.sum((data - fitted_values) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.parameters = {
                'season_length': self.season_length,
                'r_squared': r_squared,
                'trend_strength': np.std(self.trend) / np.std(data),
                'seasonal_strength': np.std(self.seasonal) / np.std(data)
            }
            
            self.is_fitted = True
            logger.info(f"Seasonal model fitted: season_length={self.season_length}, R²={r_squared:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting seasonal model: {e}")
            raise
    
    def _calculate_trend(self, data: np.ndarray) -> np.ndarray:
        """Calculate trend component using moving average."""
        trend = np.zeros_like(data)
        half_season = self.season_length // 2
        
        for i in range(len(data)):
            start = max(0, i - half_season)
            end = min(len(data), i + half_season + 1)
            trend[i] = np.mean(data[start:end])
        
        return trend
    
    def _calculate_seasonal(self, data: np.ndarray, trend: np.ndarray) -> np.ndarray:
        """Calculate seasonal component."""
        detrended = data - trend
        seasonal_pattern = np.zeros(self.season_length)
        
        for i in range(self.season_length):
            seasonal_indices = np.arange(i, len(detrended), self.season_length)
            if len(seasonal_indices) > 0:
                seasonal_pattern[i] = np.mean(detrended[seasonal_indices])
        
        # Tile the pattern to match data length
        num_full_seasons = len(data) // self.season_length
        remainder = len(data) % self.season_length
        
        seasonal = np.tile(seasonal_pattern, num_full_seasons)
        if remainder > 0:
            seasonal = np.concatenate([seasonal, seasonal_pattern[:remainder]])
        
        return seasonal
    
    def predict(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate seasonal predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extend trend (assuming linear continuation)
        last_trend_values = self.trend[-10:]  # Use last 10 points
        trend_slope = (last_trend_values[-1] - last_trend_values[0]) / len(last_trend_values)
        future_trend = self.trend[-1] + trend_slope * np.arange(1, horizon + 1)
        
        # Extend seasonal pattern
        future_seasonal = np.tile(self.seasonal[-self.season_length:], horizon // self.season_length + 1)[:horizon]
        
        # Combine components
        predictions = future_trend + future_seasonal
        
        # Calculate confidence intervals
        confidence_factor = 1.96 * self.residual_std
        lower_bounds = predictions - confidence_factor
        upper_bounds = predictions + confidence_factor
        
        return predictions, lower_bounds, upper_bounds
    
    def get_explanation(self) -> str:
        """Explain the seasonal model."""
        if not self.is_fitted:
            return "Seasonal model not fitted yet"
        
        r_squared = self.parameters['r_squared']
        trend_strength = self.parameters['trend_strength']
        seasonal_strength = self.parameters['seasonal_strength']
        
        quality_desc = "excellent" if r_squared > 0.9 else "good" if r_squared > 0.7 else "moderate" if r_squared > 0.5 else "poor"
        
        return f"Seasonal model with period {self.season_length} shows {quality_desc} fit (R² = {r_squared:.3f}). Trend strength: {trend_strength:.2f}, Seasonal strength: {seasonal_strength:.2f}."


class ForecastSystem:
    """Main forecasting system with model selection and performance tracking."""
    
    def __init__(self):
        self.models = {
            ForecastModel.LINEAR: LinearForecastModel(),
            ForecastModel.EXPONENTIAL: ExponentialForecastModel(),
            ForecastModel.ARIMA: ARIMALikeForecastModel(),
            ForecastModel.SEASONAL: SeasonalForecastModel()
        }
        
        self.model_performance_history = {}
        self.alert_thresholds = {
            'high_volatility': 2.0,
            'trend_change': 0.5,
            'confidence_drop': 0.3,
            'outlier_detection': 3.0
        }
        
    def generate_forecast(self, 
                         data: List[Dict[str, Any]], 
                         horizon: int = 24,
                         target_column: str = 'value',
                         timestamp_column: str = 'timestamp',
                         model_preference: Optional[ForecastModel] = None) -> ForecastResult:
        """Generate comprehensive forecast with model selection."""
        
        try:
            logger.info(f"Generating forecast for {len(data)} data points, horizon: {horizon}")
            
            # Prepare data
            df = pd.DataFrame(data)
            if timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{timestamp_column}' not found")
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Convert timestamps
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            df = df.sort_values(timestamp_column)
            
            timestamps = df[timestamp_column].tolist()
            values = df[target_column].values
            
            # Data quality assessment
            data_quality_score = self._assess_data_quality(values)
            
            # Select best model
            if model_preference:
                selected_model_type = model_preference
                selected_model = self.models[model_preference]
            else:
                selected_model_type, selected_model = self._select_best_model(values, timestamps)
            
            # Fit selected model
            selected_model.fit(values, timestamps)
            
            # Generate predictions
            predictions, lower_bounds, upper_bounds = selected_model.predict(horizon)
            
            # Create forecast points
            forecast_points = []
            last_timestamp = timestamps[-1]
            
            for i in range(horizon):
                # Assume hourly intervals (adjust as needed)
                future_timestamp = last_timestamp + timedelta(hours=i+1)
                
                point = ForecastPoint(
                    timestamp=future_timestamp,
                    value=float(predictions[i]),
                    lower_bound=float(lower_bounds[i]),
                    upper_bound=float(upper_bounds[i]),
                    confidence=self._calculate_point_confidence(i, selected_model),
                    model_used=selected_model.name,
                    contributing_factors=self._identify_contributing_factors(values, i)
                )
                forecast_points.append(point)
            
            # Generate alerts
            alerts = self._generate_alerts(values, predictions, selected_model)
            
            # Create result
            result = ForecastResult(
                forecast_points=forecast_points,
                model_performance=selected_model.parameters,
                selected_model=selected_model.name,
                forecast_horizon=horizon,
                generated_at=datetime.now(),
                data_quality_score=data_quality_score,
                explanation=selected_model.get_explanation(),
                alerts=alerts
            )
            
            # Update performance history
            self._update_performance_history(selected_model_type, selected_model.parameters)
            
            logger.info(f"Forecast generated successfully using {selected_model.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    def _select_best_model(self, values: np.ndarray, timestamps: List[datetime]) -> Tuple[ForecastModel, BaseForecastModel]:
        """Select the best model based on cross-validation."""
        
        if len(values) < 10:
            logger.warning("Limited data available, using linear model")
            return ForecastModel.LINEAR, self.models[ForecastModel.LINEAR]
        
        model_scores = {}
        
        # Test each model with time series cross-validation
        for model_type, model in self.models.items():
            try:
                scores = self._cross_validate_model(model, values, timestamps)
                model_scores[model_type] = np.mean(scores) if scores else -np.inf
                logger.info(f"{model.name} cross-validation score: {model_scores[model_type]:.4f}")
            except Exception as e:
                logger.warning(f"Could not validate {model.name}: {e}")
                model_scores[model_type] = -np.inf
        
        # Select best model
        best_model_type = max(model_scores.items(), key=lambda x: x[1])[0]
        return best_model_type, self.models[best_model_type]
    
    def _cross_validate_model(self, model: BaseForecastModel, values: np.ndarray, timestamps: List[datetime], n_splits: int = 3) -> List[float]:
        """Perform time series cross-validation."""
        
        scores = []
        min_train_size = max(10, len(values) // 4)
        
        for i in range(n_splits):
            # Calculate split points
            split_point = min_train_size + i * (len(values) - min_train_size) // n_splits
            if split_point >= len(values) - 1:
                break
            
            train_values = values[:split_point]
            train_timestamps = timestamps[:split_point]
            test_values = values[split_point:split_point+5]  # Test on next 5 points
            
            if len(test_values) == 0:
                continue
            
            try:
                # Create a fresh model instance for each fold
                if isinstance(model, LinearForecastModel):
                    fold_model = LinearForecastModel()
                elif isinstance(model, ExponentialForecastModel):
                    fold_model = ExponentialForecastModel()
                elif isinstance(model, ARIMALikeForecastModel):
                    fold_model = ARIMALikeForecastModel()
                elif isinstance(model, SeasonalForecastModel):
                    fold_model = SeasonalForecastModel()
                else:
                    continue
                
                fold_model.fit(train_values, train_timestamps)
                predictions, _, _ = fold_model.predict(len(test_values))
                
                # Calculate score (negative MSE for maximization)
                mse = mean_squared_error(test_values, predictions)
                scores.append(-mse)
                
            except Exception as e:
                logger.debug(f"Cross-validation fold failed: {e}")
                continue
        
        return scores
    
    def _assess_data_quality(self, values: np.ndarray) -> float:
        """Assess data quality score (0-1)."""
        
        quality_factors = []
        
        # Completeness (no NaN values)
        completeness = 1.0 - np.isnan(values).sum() / len(values)
        quality_factors.append(completeness)
        
        # Consistency (low coefficient of variation for differences)
        if len(values) > 1:
            diffs = np.diff(values[~np.isnan(values)])
            if len(diffs) > 0 and np.std(diffs) > 0:
                cv = np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-8)
                consistency = max(0, 1 - cv / 10)  # Normalize CV
            else:
                consistency = 1.0
            quality_factors.append(consistency)
        
        # Sufficient length
        length_score = min(1.0, len(values) / 50)  # Ideal: 50+ points
        quality_factors.append(length_score)
        
        # No extreme outliers (within 5 standard deviations)
        if len(values) > 2:
            std_vals = np.std(values[~np.isnan(values)])
            mean_vals = np.mean(values[~np.isnan(values)])
            outliers = np.abs(values - mean_vals) > 5 * std_vals
            outlier_score = 1.0 - np.sum(outliers) / len(values)
            quality_factors.append(outlier_score)
        
        return np.mean(quality_factors)
    
    def _calculate_point_confidence(self, horizon_step: int, model: BaseForecastModel) -> float:
        """Calculate confidence score for individual forecast point."""
        
        # Base confidence from model performance
        base_confidence = model.parameters.get('r_squared', 0.5)
        
        # Decay confidence with horizon
        decay_factor = np.exp(-horizon_step * 0.1)
        
        # Adjust based on model type
        if isinstance(model, LinearForecastModel):
            type_adjustment = 0.9  # Linear models are generally stable
        elif isinstance(model, ExponentialForecastModel):
            type_adjustment = 0.8  # More uncertain for exponential
        elif isinstance(model, ARIMALikeForecastModel):
            type_adjustment = 0.85  # Good for short-term
        elif isinstance(model, SeasonalForecastModel):
            type_adjustment = 0.87  # Good for cyclical patterns
        else:
            type_adjustment = 0.75
        
        return min(1.0, base_confidence * decay_factor * type_adjustment)
    
    def _identify_contributing_factors(self, historical_values: np.ndarray, horizon_step: int) -> List[str]:
        """Identify factors contributing to the forecast."""
        
        factors = []
        
        # Recent trend
        if len(historical_values) >= 5:
            recent_trend = np.polyfit(range(5), historical_values[-5:], 1)[0]
            if abs(recent_trend) > 0.1:
                factors.append(f"Recent trend: {'increasing' if recent_trend > 0 else 'decreasing'}")
        
        # Volatility
        if len(historical_values) >= 10:
            recent_volatility = np.std(historical_values[-10:])
            overall_volatility = np.std(historical_values)
            if recent_volatility > 1.5 * overall_volatility:
                factors.append("High recent volatility")
        
        # Seasonal patterns (if detectable)
        if len(historical_values) >= 24:
            # Simple seasonality test
            lag_24_corr = np.corrcoef(historical_values[:-24], historical_values[24:])[0, 1]
            if not np.isnan(lag_24_corr) and lag_24_corr > 0.5:
                factors.append("24-period seasonal pattern detected")
        
        # Horizon-specific factors
        if horizon_step < 6:
            factors.append("Short-term forecast")
        elif horizon_step < 24:
            factors.append("Medium-term forecast")
        else:
            factors.append("Long-term forecast")
        
        return factors
    
    def _generate_alerts(self, historical_values: np.ndarray, predictions: np.ndarray, model: BaseForecastModel) -> List[Dict[str, Any]]:
        """Generate alerts based on forecast analysis."""
        
        alerts = []
        
        try:
            # High volatility alert
            if len(historical_values) >= 10:
                recent_volatility = np.std(historical_values[-10:])
                predicted_volatility = np.std(predictions[:10])
                
                if predicted_volatility > self.alert_thresholds['high_volatility'] * recent_volatility:
                    alerts.append({
                        'type': 'high_volatility',
                        'severity': AlertSeverity.MEDIUM.value,
                        'message': f"Predicted volatility ({predicted_volatility:.2f}) significantly higher than recent volatility ({recent_volatility:.2f})",
                        'timestamp': datetime.now(),
                        'recommended_actions': ["Monitor closely", "Consider shorter forecast horizons", "Review model assumptions"]
                    })
            
            # Trend change alert
            if len(historical_values) >= 10:
                historical_trend = np.polyfit(range(len(historical_values[-10:])), historical_values[-10:], 1)[0]
                predicted_trend = np.polyfit(range(len(predictions[:10])), predictions[:10], 1)[0]
                
                if abs(predicted_trend - historical_trend) > self.alert_thresholds['trend_change']:
                    alerts.append({
                        'type': 'trend_change',
                        'severity': AlertSeverity.HIGH.value,
                        'message': f"Significant trend change detected: from {historical_trend:.3f} to {predicted_trend:.3f}",
                        'timestamp': datetime.now(),
                        'recommended_actions': ["Investigate underlying causes", "Validate model assumptions", "Consider external factors"]
                    })
            
            # Low confidence alert
            avg_confidence = np.mean([self._calculate_point_confidence(i, model) for i in range(min(10, len(predictions)))])
            if avg_confidence < self.alert_thresholds['confidence_drop']:
                alerts.append({
                    'type': 'low_confidence',
                    'severity': AlertSeverity.MEDIUM.value,
                    'message': f"Low forecast confidence ({avg_confidence:.2f}) detected",
                    'timestamp': datetime.now(),
                    'recommended_actions': ["Collect more data", "Review data quality", "Consider ensemble methods"]
                })
            
            # Extreme value alert
            historical_range = np.max(historical_values) - np.min(historical_values)
            prediction_range = np.max(predictions) - np.min(predictions)
            
            if prediction_range > self.alert_thresholds['outlier_detection'] * historical_range:
                alerts.append({
                    'type': 'extreme_values',
                    'severity': AlertSeverity.CRITICAL.value,
                    'message': f"Extreme values predicted (range: {prediction_range:.2f} vs historical: {historical_range:.2f})",
                    'timestamp': datetime.now(),
                    'recommended_actions': ["Validate data inputs", "Check for anomalies", "Consider model limitations"]
                })
            
        except Exception as e:
            logger.warning(f"Error generating alerts: {e}")
        
        return alerts
    
    def _update_performance_history(self, model_type: ForecastModel, performance_metrics: Dict[str, float]):
        """Update model performance history."""
        
        if model_type.value not in self.model_performance_history:
            self.model_performance_history[model_type.value] = []
        
        entry = {
            'timestamp': datetime.now(),
            'metrics': performance_metrics.copy()
        }
        
        self.model_performance_history[model_type.value].append(entry)
        
        # Keep only last 100 entries per model
        if len(self.model_performance_history[model_type.value]) > 100:
            self.model_performance_history[model_type.value] = self.model_performance_history[model_type.value][-100:]
    
    def analyze_scenarios(self, 
                         data: List[Dict[str, Any]], 
                         scenarios: List[ScenarioInput],
                         horizon: int = 24,
                         target_column: str = 'value') -> Dict[str, ForecastResult]:
        """Perform scenario analysis with different parameter adjustments."""
        
        results = {}
        
        # Baseline forecast
        baseline_result = self.generate_forecast(data, horizon, target_column)
        results['baseline'] = baseline_result
        
        # Scenario forecasts
        for scenario in scenarios:
            try:
                logger.info(f"Analyzing scenario: {scenario.name}")
                
                # Adjust data based on scenario parameters
                adjusted_data = self._apply_scenario_adjustments(data, scenario)
                
                # Generate forecast for scenario
                scenario_result = self.generate_forecast(adjusted_data, horizon, target_column)
                
                # Add scenario metadata
                scenario_result.explanation += f"\n\nScenario '{scenario.name}': {scenario.description}"
                
                results[scenario.name] = scenario_result
                
            except Exception as e:
                logger.error(f"Error analyzing scenario '{scenario.name}': {e}")
                continue
        
        return results
    
    def _apply_scenario_adjustments(self, data: List[Dict[str, Any]], scenario: ScenarioInput) -> List[Dict[str, Any]]:
        """Apply scenario parameter adjustments to data."""
        
        adjusted_data = []
        
        for point in data:
            adjusted_point = point.copy()
            
            # Apply adjustments
            for param, adjustment in scenario.parameter_adjustments.items():
                if param in adjusted_point:
                    if isinstance(adjustment, (int, float)):
                        # Multiplicative adjustment
                        adjusted_point[param] = adjusted_point[param] * adjustment
                    else:
                        # Additive adjustment
                        adjusted_point[param] = adjusted_point[param] + adjustment
            
            adjusted_data.append(adjusted_point)
        
        return adjusted_data
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance across all historical runs."""
        
        summary = {}
        
        for model_name, history in self.model_performance_history.items():
            if not history:
                continue
            
            # Extract R-squared values
            r_squared_values = [entry['metrics'].get('r_squared', 0) for entry in history]
            
            summary[model_name] = {
                'total_runs': len(history),
                'avg_r_squared': np.mean(r_squared_values),
                'std_r_squared': np.std(r_squared_values),
                'best_r_squared': np.max(r_squared_values),
                'worst_r_squared': np.min(r_squared_values),
                'last_performance': history[-1]['metrics'] if history else None,
                'trend': 'improving' if len(r_squared_values) >= 2 and r_squared_values[-1] > r_squared_values[0] else 'stable'
            }
        
        return summary
    
    def calculate_forecast_accuracy(self, 
                                  predicted_values: List[float], 
                                  actual_values: List[float]) -> Dict[str, float]:
        """Calculate accuracy metrics for forecast validation."""
        
        if len(predicted_values) != len(actual_values):
            raise ValueError("Predicted and actual values must have the same length")
        
        predicted = np.array(predicted_values)
        actual = np.array(actual_values)
        
        # Remove any NaN values
        mask = ~(np.isnan(predicted) | np.isnan(actual))
        predicted = predicted[mask]
        actual = actual[mask]
        
        if len(predicted) == 0:
            return {'error': 'No valid data points for accuracy calculation'}
        
        metrics = {
            'mae': mean_absolute_error(actual, predicted),
            'mse': mean_squared_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mape': np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100,
            'r2_score': r2_score(actual, predicted) if len(actual) > 1 else 0,
            'bias': np.mean(predicted - actual),
            'accuracy_percentage': max(0, 100 - np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100)
        }
        
        return metrics
    
    def export_forecast_report(self, result: ForecastResult) -> str:
        """Export forecast result as human-readable report."""
        
        report = f"""
FORECAST ANALYSIS REPORT
========================

Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
Model Used: {result.selected_model}
Forecast Horizon: {result.forecast_horizon} periods
Data Quality Score: {result.data_quality_score:.2f}/1.00

MODEL EXPLANATION
-----------------
{result.explanation}

FORECAST SUMMARY
---------------
Total Points: {len(result.forecast_points)}
Avg Confidence: {np.mean([p.confidence for p in result.forecast_points]):.2f}
Value Range: {min(p.value for p in result.forecast_points):.2f} - {max(p.value for p in result.forecast_points):.2f}

FIRST 5 PREDICTIONS
-------------------
"""
        
        for i, point in enumerate(result.forecast_points[:5]):
            report += f"Period {i+1}: {point.value:.2f} [{point.lower_bound:.2f}, {point.upper_bound:.2f}] (confidence: {point.confidence:.2f})\n"
        
        if len(result.alerts) > 0:
            report += f"\nALERTS ({len(result.alerts)})\n"
            report += "-" * 20 + "\n"
            for alert in result.alerts:
                report += f"[{alert['severity'].upper()}] {alert['type']}: {alert['message']}\n"
                if 'recommended_actions' in alert:
                    for action in alert['recommended_actions']:
                        report += f"  → {action}\n"
                report += "\n"
        
        return report


# Example usage and testing functions
def create_sample_data(n_points: int = 100, trend: float = 0.1, noise: float = 0.5, seasonal: bool = False) -> List[Dict[str, Any]]:
    """Create sample time series data for testing."""
    
    data = []
    base_time = datetime.now() - timedelta(hours=n_points)
    
    for i in range(n_points):
        timestamp = base_time + timedelta(hours=i)
        
        # Base trend
        value = 100 + trend * i
        
        # Add seasonal component if requested
        if seasonal:
            value += 10 * np.sin(2 * np.pi * i / 24)  # 24-hour cycle
        
        # Add noise
        value += np.random.normal(0, noise)
        
        data.append({
            'timestamp': timestamp,
            'value': value,
            'metadata': {'period': i}
        })
    
    return data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create forecast system
    forecast_system = ForecastSystem()
    
    # Generate sample data
    sample_data = create_sample_data(n_points=72, trend=0.2, seasonal=True)
    
    # Create scenarios
    scenarios = [
        ScenarioInput(
            name="optimistic",
            parameter_adjustments={'value': 1.1},
            probability=0.3,
            description="10% increase in all values"
        ),
        ScenarioInput(
            name="pessimistic", 
            parameter_adjustments={'value': 0.9},
            probability=0.2,
            description="10% decrease in all values"
        )
    ]
    
    try:
        # Generate baseline forecast
        result = forecast_system.generate_forecast(sample_data, horizon=24)
        print("Forecast generated successfully!")
        print(f"Selected model: {result.selected_model}")
        print(f"Data quality: {result.data_quality_score:.2f}")
        print(f"Number of alerts: {len(result.alerts)}")
        
        # Analyze scenarios
        scenario_results = forecast_system.analyze_scenarios(sample_data, scenarios)
        print(f"Analyzed {len(scenario_results)} scenarios")
        
        # Export report
        report = forecast_system.export_forecast_report(result)
        print("\nSample Report:")
        print(report[:500] + "..." if len(report) > 500 else report)
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")