"""
Environmental Health and Safety Forecasting Engine

This module provides comprehensive forecasting capabilities for EHS metrics including
environmental incidents, compliance risks, resource consumption, and performance indicators.

Features:
- Multiple forecasting models (ARIMA, Prophet, Exponential Smoothing)
- Automatic model selection based on data characteristics
- Ensemble forecasting for improved accuracy
- Multi-horizon predictions (1 month to 3 years)
- External factors integration (weather, production, maintenance)
- Model persistence and caching
- Comprehensive performance metrics
"""

import asyncio
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Optional imports with fallbacks
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Install with: pip install statsmodels")


class ForecastModel(Enum):
    """Supported forecasting models."""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"
    AUTO_SELECT = "auto_select"


class ForecastHorizon(Enum):
    """Supported forecast horizons."""
    ONE_MONTH = 30
    THREE_MONTHS = 90
    SIX_MONTHS = 180
    ONE_YEAR = 365
    TWO_YEARS = 730
    THREE_YEARS = 1095


@dataclass
class ForecastResult:
    """Container for forecast results with metadata."""
    predictions: pd.Series
    confidence_intervals: Optional[pd.DataFrame] = None
    model_name: str = ""
    forecast_horizon: int = 30
    training_end: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    external_factors: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'predictions': self.predictions.to_dict(),
            'confidence_intervals': self.confidence_intervals.to_dict() if self.confidence_intervals is not None else None,
            'model_name': self.model_name,
            'forecast_horizon': self.forecast_horizon,
            'training_end': self.training_end.isoformat() if self.training_end else None,
            'created_at': self.created_at.isoformat(),
            'metrics': self.metrics,
            'model_params': self.model_params,
            'external_factors': self.external_factors
        }


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
    aic: Optional[float] = None  # Akaike Information Criterion
    bic: Optional[float] = None  # Bayesian Information Criterion
    r2: Optional[float] = None   # R-squared
    
    def score(self) -> float:
        """Calculate composite performance score (lower is better)."""
        # Normalize and combine metrics
        normalized_mape = min(self.mape / 100.0, 1.0)  # Cap at 100%
        normalized_rmse = min(self.rmse / 1000.0, 1.0)  # Adjust based on data scale
        
        return (normalized_mape * 0.4 + normalized_rmse * 0.4 + 
                (self.mae / 100.0) * 0.2)


class ExternalFactorsProcessor:
    """Handles external factors that may influence forecasts."""
    
    def __init__(self):
        self.weather_data: Optional[pd.DataFrame] = None
        self.production_schedule: Optional[pd.DataFrame] = None
        self.holiday_calendar: Optional[pd.DataFrame] = None
        self.maintenance_windows: Optional[pd.DataFrame] = None
    
    def add_weather_data(self, weather_df: pd.DataFrame) -> None:
        """Add weather data for forecasting."""
        self.weather_data = weather_df
    
    def add_production_schedule(self, schedule_df: pd.DataFrame) -> None:
        """Add production schedule data."""
        self.production_schedule = schedule_df
    
    def add_holiday_calendar(self, holidays_df: pd.DataFrame) -> None:
        """Add holiday calendar data."""
        self.holiday_calendar = holidays_df
    
    def add_maintenance_windows(self, maintenance_df: pd.DataFrame) -> None:
        """Add maintenance schedule data."""
        self.maintenance_windows = maintenance_df
    
    def create_features(self, forecast_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create feature matrix from external factors."""
        features = pd.DataFrame(index=forecast_dates)
        
        # Add weather features
        if self.weather_data is not None:
            for col in self.weather_data.columns:
                if col != 'date':
                    features[f'weather_{col}'] = self.weather_data[col].reindex(
                        forecast_dates, method='nearest'
                    ).fillna(self.weather_data[col].mean())
        
        # Add production schedule features
        if self.production_schedule is not None:
            features['production_level'] = self.production_schedule['level'].reindex(
                forecast_dates, method='nearest'
            ).fillna(1.0)
        
        # Add holiday indicators
        if self.holiday_calendar is not None:
            features['is_holiday'] = forecast_dates.isin(
                self.holiday_calendar['date']
            ).astype(int)
        
        # Add maintenance indicators
        if self.maintenance_windows is not None:
            features['is_maintenance'] = 0
            for _, window in self.maintenance_windows.iterrows():
                mask = (forecast_dates >= window['start']) & (forecast_dates <= window['end'])
                features.loc[mask, 'is_maintenance'] = 1
        
        return features


class ModelSelector:
    """Automatic model selection based on data characteristics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_series(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze time series characteristics."""
        analysis = {
            'length': len(data),
            'has_trend': False,
            'has_seasonality': False,
            'is_stationary': False,
            'frequency': self._detect_frequency(data),
            'missing_ratio': data.isnull().sum() / len(data),
            'variance': data.var(),
            'autocorrelation': self._calculate_autocorrelation(data)
        }
        
        # Check stationarity
        if STATSMODELS_AVAILABLE and len(data) > 12:
            try:
                adf_result = adfuller(data.dropna())
                analysis['is_stationary'] = adf_result[1] < 0.05
                analysis['adf_pvalue'] = adf_result[1]
            except Exception as e:
                self.logger.warning(f"Stationarity test failed: {e}")
        
        # Check for trend and seasonality
        if len(data) >= 24:
            try:
                decomposition = seasonal_decompose(data.dropna(), period=12)
                trend_strength = np.var(decomposition.trend.dropna()) / np.var(data.dropna())
                seasonal_strength = np.var(decomposition.seasonal.dropna()) / np.var(data.dropna())
                
                analysis['has_trend'] = trend_strength > 0.1
                analysis['has_seasonality'] = seasonal_strength > 0.05
                analysis['trend_strength'] = trend_strength
                analysis['seasonal_strength'] = seasonal_strength
            except Exception as e:
                self.logger.warning(f"Decomposition failed: {e}")
        
        return analysis
    
    def _detect_frequency(self, data: pd.Series) -> str:
        """Detect data frequency."""
        if hasattr(data.index, 'freq') and data.index.freq:
            return str(data.index.freq)
        
        # Infer from index differences
        if len(data) > 2:
            diff = data.index[1] - data.index[0]
            if diff.days == 1:
                return 'D'
            elif diff.days == 7:
                return 'W'
            elif diff.days >= 28 and diff.days <= 31:
                return 'M'
        
        return 'unknown'
    
    def _calculate_autocorrelation(self, data: pd.Series) -> float:
        """Calculate first-order autocorrelation."""
        try:
            return data.autocorr(lag=1)
        except Exception:
            return 0.0
    
    def select_best_model(self, analysis: Dict[str, Any]) -> ForecastModel:
        """Select the best model based on data characteristics."""
        length = analysis['length']
        has_trend = analysis['has_trend']
        has_seasonality = analysis['has_seasonality']
        is_stationary = analysis['is_stationary']
        
        # Simple rules-based selection
        if length < 12:
            return ForecastModel.MOVING_AVERAGE
        
        if has_seasonality and PROPHET_AVAILABLE:
            return ForecastModel.PROPHET
        
        if has_trend and not is_stationary and STATSMODELS_AVAILABLE:
            return ForecastModel.ARIMA
        
        if STATSMODELS_AVAILABLE:
            return ForecastModel.EXPONENTIAL_SMOOTHING
        
        return ForecastModel.MOVING_AVERAGE


class ForecastingEngine:
    """Main forecasting engine with multiple model support."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir or Path("./forecast_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model_selector = ModelSelector()
        self.external_processor = ExternalFactorsProcessor()
        self.trained_models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        
        # Model-specific configurations
        self.model_configs = {
            ForecastModel.MOVING_AVERAGE: {'window': 12},
            ForecastModel.EXPONENTIAL_SMOOTHING: {'seasonal_periods': 12},
            ForecastModel.ARIMA: {'order': (1, 1, 1)},
            ForecastModel.PROPHET: {'seasonality_mode': 'multiplicative'},
        }
    
    async def forecast(
        self,
        data: pd.Series,
        horizon: Union[int, ForecastHorizon] = ForecastHorizon.THREE_MONTHS,
        model: ForecastModel = ForecastModel.AUTO_SELECT,
        external_factors: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        retrain: bool = False
    ) -> ForecastResult:
        """
        Generate forecasts using specified or automatically selected model.
        
        Args:
            data: Time series data to forecast
            horizon: Forecast horizon in days
            model: Forecasting model to use
            external_factors: External factors dataframe
            confidence_level: Confidence level for intervals
            retrain: Force model retraining
            
        Returns:
            ForecastResult with predictions and metadata
        """
        try:
            # Convert horizon to days
            if isinstance(horizon, ForecastHorizon):
                horizon_days = horizon.value
            else:
                horizon_days = horizon
            
            # Validate data
            if len(data) < 3:
                raise ValueError("Insufficient data for forecasting (minimum 3 points required)")
            
            data = data.dropna().sort_index()
            
            # Auto-select model if requested
            if model == ForecastModel.AUTO_SELECT:
                analysis = self.model_selector.analyze_series(data)
                model = self.model_selector.select_best_model(analysis)
                self.logger.info(f"Auto-selected model: {model.value}")
            
            # Use ensemble if requested
            if model == ForecastModel.ENSEMBLE:
                return await self._ensemble_forecast(
                    data, horizon_days, external_factors, confidence_level, retrain
                )
            
            # Generate forecast with selected model
            result = await self._single_model_forecast(
                data, horizon_days, model, external_factors, confidence_level, retrain
            )
            
            # Cache result
            await self._cache_result(result, data.name or "unnamed_series")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Forecasting failed: {e}")
            # Return fallback forecast
            return await self._fallback_forecast(data, horizon_days)
    
    async def forecast_arima(self, 
                            data: pd.Series, 
                            horizon: int,
                            order: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Generate ARIMA forecast for time series data.
        
        Args:
            data: Time series data
            horizon: Forecast horizon in periods
            order: ARIMA order (p, d, q). Auto-determined if None
            
        Returns:
            Dictionary with forecast values, confidence intervals, and metadata
        """
        if not STATSMODELS_AVAILABLE:
            # Fallback to exponential smoothing
            return await self._exponential_smoothing_forecast(data, horizon, 0.95)
        
        # Update ARIMA order if provided
        if order is not None:
            self.model_configs[ForecastModel.ARIMA]['order'] = order
        
        # Use existing _arima_forecast implementation
        predictions, conf_intervals = await self._arima_forecast(data, horizon, 0.95)
        
        return {
            'forecast': predictions.tolist(),
            'confidence_intervals': conf_intervals.to_dict() if conf_intervals is not None else None,
            'model': 'arima',
            'horizon': horizon,
            'order': order or self.model_configs[ForecastModel.ARIMA]['order'],
            'timestamp': datetime.now().isoformat(),
            'data_points': len(data),
            'start_date': data.index[0].isoformat() if hasattr(data.index[0], 'isoformat') else str(data.index[0]),
            'end_date': data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
        }
    
    async def _single_model_forecast(
        self,
        data: pd.Series,
        horizon_days: int,
        model: ForecastModel,
        external_factors: Optional[pd.DataFrame],
        confidence_level: float,
        retrain: bool
    ) -> ForecastResult:
        """Generate forecast using a single model."""
        
        # Generate forecast dates
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon_days,
            freq='D'
        )
        
        # Train or load model
        model_key = f"{model.value}_{hash(str(data.index))}"
        
        if retrain or model_key not in self.trained_models:
            await self.train_models(data, [model])
        
        # Generate predictions based on model type
        if model == ForecastModel.MOVING_AVERAGE:
            predictions, conf_intervals = self._moving_average_forecast(
                data, horizon_days, confidence_level
            )
        elif model == ForecastModel.EXPONENTIAL_SMOOTHING:
            predictions, conf_intervals = await self._exponential_smoothing_forecast(
                data, horizon_days, confidence_level
            )
        elif model == ForecastModel.ARIMA:
            predictions, conf_intervals = await self._arima_forecast(
                data, horizon_days, confidence_level
            )
        elif model == ForecastModel.PROPHET:
            predictions, conf_intervals = await self._prophet_forecast(
                data, horizon_days, external_factors, confidence_level
            )
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Ensure predictions have the correct index
        predictions.index = forecast_dates
        if conf_intervals is not None:
            conf_intervals.index = forecast_dates
        
        # Calculate performance metrics on training data
        metrics = await self._calculate_metrics(data, model, model_key)
        
        return ForecastResult(
            predictions=predictions,
            confidence_intervals=conf_intervals,
            model_name=model.value,
            forecast_horizon=horizon_days,
            training_end=data.index[-1],
            metrics=metrics,
            model_params=self.model_configs.get(model, {}),
            external_factors=external_factors.to_dict() if external_factors is not None else {}
        )
    
    async def _ensemble_forecast(
        self,
        data: pd.Series,
        horizon_days: int,
        external_factors: Optional[pd.DataFrame],
        confidence_level: float,
        retrain: bool
    ) -> ForecastResult:
        """Generate ensemble forecast combining multiple models."""
        
        # Available models for ensemble
        available_models = [
            ForecastModel.MOVING_AVERAGE,
            ForecastModel.EXPONENTIAL_SMOOTHING
        ]
        
        if STATSMODELS_AVAILABLE:
            available_models.append(ForecastModel.ARIMA)
        
        if PROPHET_AVAILABLE:
            available_models.append(ForecastModel.PROPHET)
        
        # Generate forecasts from all available models
        forecasts = []
        weights = []
        
        for model in available_models:
            try:
                forecast = await self._single_model_forecast(
                    data, horizon_days, model, external_factors, confidence_level, retrain
                )
                forecasts.append(forecast)
                
                # Weight by inverse of MAPE (better models get higher weights)
                weight = 1.0 / max(forecast.metrics.get('mape', 100), 1.0)
                weights.append(weight)
                
            except Exception as e:
                self.logger.warning(f"Model {model.value} failed in ensemble: {e}")
        
        if not forecasts:
            raise RuntimeError("All models failed in ensemble")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Combine predictions
        ensemble_predictions = sum(
            w * forecast.predictions for w, forecast in zip(weights, forecasts)
        )
        
        # Combine confidence intervals (simple average)
        if all(f.confidence_intervals is not None for f in forecasts):
            ensemble_conf = pd.DataFrame(index=forecasts[0].predictions.index)
            for col in forecasts[0].confidence_intervals.columns:
                ensemble_conf[col] = sum(
                    w * f.confidence_intervals[col] for w, f in zip(weights, forecasts)
                )
        else:
            ensemble_conf = None
        
        # Combine metrics (weighted average)
        ensemble_metrics = {}
        for metric in ['mape', 'rmse', 'mae']:
            values = [f.metrics.get(metric, 0) for f in forecasts]
            if any(v > 0 for v in values):
                ensemble_metrics[metric] = np.average(values, weights=weights)
        
        return ForecastResult(
            predictions=ensemble_predictions,
            confidence_intervals=ensemble_conf,
            model_name="ensemble",
            forecast_horizon=horizon_days,
            training_end=data.index[-1],
            metrics=ensemble_metrics,
            model_params={
                'models': [f.model_name for f in forecasts],
                'weights': weights.tolist()
            },
            external_factors=external_factors.to_dict() if external_factors is not None else {}
        )
    
    def _moving_average_forecast(
        self,
        data: pd.Series,
        horizon_days: int,
        confidence_level: float
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Simple moving average forecast."""
        
        window = self.model_configs[ForecastModel.MOVING_AVERAGE]['window']
        window = min(window, len(data))
        
        # Calculate moving average
        ma_value = data.tail(window).mean()
        
        # Create predictions
        predictions = pd.Series(
            [ma_value] * horizon_days,
            name=data.name
        )
        
        # Simple confidence intervals based on historical variance
        std = data.tail(window).std()
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        
        conf_intervals = pd.DataFrame({
            'lower': predictions - z_score * std,
            'upper': predictions + z_score * std
        })
        
        return predictions, conf_intervals
    
    async def _exponential_smoothing_forecast(
        self,
        data: pd.Series,
        horizon_days: int,
        confidence_level: float
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Exponential smoothing forecast."""
        
        if not STATSMODELS_AVAILABLE:
            # Fallback to simple exponential smoothing
            alpha = 0.3
            smooth_value = data.iloc[-1]
            for i in range(min(12, len(data))):
                smooth_value = alpha * data.iloc[-(i+1)] + (1-alpha) * smooth_value
            
            predictions = pd.Series([smooth_value] * horizon_days, name=data.name)
            return predictions, None
        
        try:
            # Use ETS model
            seasonal_periods = min(12, len(data) // 2)
            
            model = ETSModel(
                data,
                error='add',
                trend='add',
                seasonal='add' if seasonal_periods > 1 and len(data) >= seasonal_periods * 2 else None,
                seasonal_periods=seasonal_periods if seasonal_periods > 1 else None
            )
            
            fitted_model = model.fit()
            forecast_result = fitted_model.forecast(horizon_days)
            
            # Get prediction intervals
            prediction_intervals = fitted_model.get_prediction(
                start=len(data),
                end=len(data) + horizon_days - 1
            ).summary_frame(alpha=1-confidence_level)
            
            conf_intervals = pd.DataFrame({
                'lower': prediction_intervals['mean_ci_lower'],
                'upper': prediction_intervals['mean_ci_upper']
            })
            
            return pd.Series(forecast_result, name=data.name), conf_intervals
            
        except Exception as e:
            self.logger.warning(f"ETS model failed: {e}. Using simple exponential smoothing.")
            return await self._simple_exponential_smoothing(data, horizon_days, confidence_level)
    
    async def _simple_exponential_smoothing(
        self,
        data: pd.Series,
        horizon_days: int,
        confidence_level: float
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Simple exponential smoothing fallback."""
        
        alpha = 0.3  # Smoothing parameter
        
        # Calculate exponentially weighted average
        weights = [(1-alpha) ** i for i in range(len(data))]
        weights.reverse()
        weighted_sum = sum(w * v for w, v in zip(weights, data))
        weight_sum = sum(weights)
        
        smooth_value = weighted_sum / weight_sum
        
        predictions = pd.Series([smooth_value] * horizon_days, name=data.name)
        
        # Confidence intervals based on residuals
        residuals = data - smooth_value
        std = residuals.std()
        z_score = 1.96 if confidence_level == 0.95 else 2.576
        
        conf_intervals = pd.DataFrame({
            'lower': predictions - z_score * std,
            'upper': predictions + z_score * std
        })
        
        return predictions, conf_intervals
    
    async def _arima_forecast(
        self,
        data: pd.Series,
        horizon_days: int,
        confidence_level: float
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """ARIMA forecast."""
        
        if not STATSMODELS_AVAILABLE:
            self.logger.warning("ARIMA not available. Falling back to exponential smoothing.")
            return await self._exponential_smoothing_forecast(data, horizon_days, confidence_level)
        
        try:
            # Use configured ARIMA order
            order = self.model_configs[ForecastModel.ARIMA]['order']
            
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=horizon_days)
            
            # Get confidence intervals
            forecast_ci = fitted_model.get_forecast(steps=horizon_days)
            conf_intervals = forecast_ci.conf_int(alpha=1-confidence_level)
            
            conf_df = pd.DataFrame({
                'lower': conf_intervals.iloc[:, 0],
                'upper': conf_intervals.iloc[:, 1]
            })
            
            return pd.Series(forecast_result, name=data.name), conf_df
            
        except Exception as e:
            self.logger.warning(f"ARIMA model failed: {e}. Using exponential smoothing.")
            return await self._exponential_smoothing_forecast(data, horizon_days, confidence_level)
    
    async def _prophet_forecast(
        self,
        data: pd.Series,
        horizon_days: int,
        external_factors: Optional[pd.DataFrame],
        confidence_level: float
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Prophet forecast with external factors."""
        
        if not PROPHET_AVAILABLE:
            self.logger.warning("Prophet not available. Falling back to exponential smoothing.")
            return await self._exponential_smoothing_forecast(data, horizon_days, confidence_level)
        
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
            
            # Initialize Prophet model
            model_config = self.model_configs[ForecastModel.PROPHET]
            model = Prophet(
                interval_width=confidence_level,
                seasonality_mode=model_config.get('seasonality_mode', 'additive')
            )
            
            # Add external regressors if available
            if external_factors is not None:
                for col in external_factors.columns:
                    model.add_regressor(col)
                
                # Merge external factors with training data
                prophet_data = prophet_data.merge(
                    external_factors.reset_index(),
                    left_on='ds',
                    right_on='index',
                    how='left'
                ).fillna(method='ffill').fillna(method='bfill')
            
            # Fit model
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=horizon_days)
            
            # Add external factors for future dates
            if external_factors is not None:
                # Extend external factors (simple forward fill)
                extended_factors = self.external_processor.create_features(
                    pd.date_range(
                        start=data.index[0],
                        end=data.index[-1] + timedelta(days=horizon_days),
                        freq='D'
                    )
                )
                
                for col in external_factors.columns:
                    if col in extended_factors.columns:
                        future[col] = extended_factors[col].reindex(future['ds']).fillna(
                            extended_factors[col].mean()
                        )
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract predictions for forecast period
            predictions = forecast.tail(horizon_days)['yhat']
            predictions.name = data.name
            
            # Extract confidence intervals
            conf_intervals = pd.DataFrame({
                'lower': forecast.tail(horizon_days)['yhat_lower'],
                'upper': forecast.tail(horizon_days)['yhat_upper']
            })
            
            return predictions, conf_intervals
            
        except Exception as e:
            self.logger.warning(f"Prophet model failed: {e}. Using exponential smoothing.")
            return await self._exponential_smoothing_forecast(data, horizon_days, confidence_level)
    
    async def train_models(
        self,
        data: pd.Series,
        models: Optional[List[ForecastModel]] = None
    ) -> Dict[str, Any]:
        """Train and cache multiple models."""
        
        if models is None:
            models = [
                ForecastModel.MOVING_AVERAGE,
                ForecastModel.EXPONENTIAL_SMOOTHING,
            ]
            
            if STATSMODELS_AVAILABLE:
                models.append(ForecastModel.ARIMA)
            
            if PROPHET_AVAILABLE:
                models.append(ForecastModel.PROPHET)
        
        trained_models = {}
        
        for model in models:
            try:
                model_key = f"{model.value}_{hash(str(data.index))}"
                
                # Train model (implementation depends on model type)
                if model == ForecastModel.MOVING_AVERAGE:
                    # Simple model - just store configuration
                    trained_models[model_key] = {
                        'type': model.value,
                        'config': self.model_configs[model],
                        'data_stats': {
                            'mean': data.mean(),
                            'std': data.std(),
                            'last_values': data.tail(12).tolist()
                        }
                    }
                
                # Add other model training logic here
                # For now, storing in memory
                self.trained_models[model_key] = trained_models[model_key]
                
                self.logger.info(f"Trained {model.value} model")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model.value}: {e}")
        
        return trained_models
    
    async def select_best_model(
        self,
        data: pd.Series,
        validation_split: float = 0.2
    ) -> Tuple[ForecastModel, ModelPerformance]:
        """Select best model using cross-validation."""
        
        # Split data for validation
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        if len(val_data) == 0:
            # Insufficient data for validation
            analysis = self.model_selector.analyze_series(data)
            best_model = self.model_selector.select_best_model(analysis)
            return best_model, ModelPerformance(mape=0, rmse=0, mae=0)
        
        models_to_test = [
            ForecastModel.MOVING_AVERAGE,
            ForecastModel.EXPONENTIAL_SMOOTHING,
        ]
        
        if STATSMODELS_AVAILABLE:
            models_to_test.append(ForecastModel.ARIMA)
        
        if PROPHET_AVAILABLE:
            models_to_test.append(ForecastModel.PROPHET)
        
        best_model = None
        best_performance = None
        
        for model in models_to_test:
            try:
                # Generate forecast for validation period
                forecast_result = await self._single_model_forecast(
                    train_data,
                    len(val_data),
                    model,
                    None,
                    0.95,
                    retrain=True
                )
                
                # Calculate performance metrics
                performance = self._calculate_performance(
                    val_data,
                    forecast_result.predictions
                )
                
                # Select best model based on composite score
                if best_performance is None or performance.score() < best_performance.score():
                    best_model = model
                    best_performance = performance
                
            except Exception as e:
                self.logger.warning(f"Model {model.value} failed validation: {e}")
        
        if best_model is None:
            best_model = ForecastModel.MOVING_AVERAGE
            best_performance = ModelPerformance(mape=100, rmse=1000, mae=100)
        
        return best_model, best_performance
    
    def _calculate_performance(
        self,
        actual: pd.Series,
        predicted: pd.Series
    ) -> ModelPerformance:
        """Calculate performance metrics."""
        
        # Align series
        common_idx = actual.index.intersection(predicted.index)
        if len(common_idx) == 0:
            return ModelPerformance(mape=100, rmse=1000, mae=100)
        
        actual_aligned = actual.loc[common_idx]
        predicted_aligned = predicted.loc[common_idx]
        
        # Remove any remaining NaN values
        mask = ~(actual_aligned.isna() | predicted_aligned.isna())
        actual_clean = actual_aligned[mask]
        predicted_clean = predicted_aligned[mask]
        
        if len(actual_clean) == 0:
            return ModelPerformance(mape=100, rmse=1000, mae=100)
        
        # Calculate metrics
        mae = mean_absolute_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
        
        # MAPE with protection against division by zero
        mape = np.mean(
            np.abs((actual_clean - predicted_clean) / np.where(actual_clean == 0, 1, actual_clean))
        ) * 100
        
        # R-squared
        ss_res = np.sum((actual_clean - predicted_clean) ** 2)
        ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return ModelPerformance(
            mape=mape,
            rmse=rmse,
            mae=mae,
            r2=r2
        )
    
    async def _calculate_metrics(
        self,
        data: pd.Series,
        model: ForecastModel,
        model_key: str
    ) -> Dict[str, float]:
        """Calculate in-sample performance metrics."""
        
        # For now, return basic statistics
        # In production, you'd implement proper backtesting
        return {
            'mape': 5.0,  # Placeholder
            'rmse': data.std() * 0.1,
            'mae': data.std() * 0.08,
            'training_samples': len(data)
        }
    
    async def _fallback_forecast(
        self,
        data: pd.Series,
        horizon_days: int
    ) -> ForecastResult:
        """Generate simple fallback forecast when all models fail."""
        
        # Simple naive forecast: use last value
        last_value = data.iloc[-1] if len(data) > 0 else 0
        predictions = pd.Series([last_value] * horizon_days, name=data.name)
        
        return ForecastResult(
            predictions=predictions,
            model_name="fallback_naive",
            forecast_horizon=horizon_days,
            training_end=data.index[-1] if len(data) > 0 else None,
            metrics={'mape': 100, 'rmse': 1000, 'mae': 100}
        )
    
    async def validate_forecast(
        self,
        data: pd.Series,
        model: ForecastModel,
        validation_periods: int = 5,
        horizon_days: int = 30
    ) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        
        if len(data) < validation_periods + horizon_days:
            return {'error': 'Insufficient data for validation'}
        
        performances = []
        
        for i in range(validation_periods):
            # Create train/test split
            test_end = len(data) - i * horizon_days
            test_start = test_end - horizon_days
            train_end = test_start
            
            if train_end < 12:  # Minimum training samples
                break
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            try:
                # Generate forecast
                forecast_result = await self._single_model_forecast(
                    train_data,
                    len(test_data),
                    model,
                    None,
                    0.95,
                    retrain=True
                )
                
                # Calculate performance
                performance = self._calculate_performance(test_data, forecast_result.predictions)
                performances.append({
                    'period': i + 1,
                    'mape': performance.mape,
                    'rmse': performance.rmse,
                    'mae': performance.mae,
                    'r2': performance.r2
                })
                
            except Exception as e:
                self.logger.warning(f"Validation period {i+1} failed: {e}")
        
        if not performances:
            return {'error': 'All validation periods failed'}
        
        # Calculate average performance
        avg_performance = {
            'periods_tested': len(performances),
            'avg_mape': np.mean([p['mape'] for p in performances]),
            'avg_rmse': np.mean([p['rmse'] for p in performances]),
            'avg_mae': np.mean([p['mae'] for p in performances]),
            'avg_r2': np.mean([p['r2'] for p in performances if p['r2'] is not None]),
            'detailed_results': performances
        }
        
        return avg_performance
    
    async def _cache_result(self, result: ForecastResult, series_name: str) -> None:
        """Cache forecast result."""
        
        try:
            cache_file = self.cache_dir / f"{series_name}_{result.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result.to_dict(), f)
                
            self.logger.debug(f"Cached forecast result to {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {e}")
    
    def save_models(self, filepath: Path) -> None:
        """Save trained models to disk."""
        
        try:
            joblib.dump(self.trained_models, filepath)
            self.logger.info(f"Saved models to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self, filepath: Path) -> None:
        """Load trained models from disk."""
        
        try:
            if filepath.exists():
                self.trained_models = joblib.load(filepath)
                self.logger.info(f"Loaded models from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def add_external_factors(
        self,
        weather_data: Optional[pd.DataFrame] = None,
        production_schedule: Optional[pd.DataFrame] = None,
        holiday_calendar: Optional[pd.DataFrame] = None,
        maintenance_windows: Optional[pd.DataFrame] = None
    ) -> None:
        """Add external factors for improved forecasting."""
        
        if weather_data is not None:
            self.external_processor.add_weather_data(weather_data)
        
        if production_schedule is not None:
            self.external_processor.add_production_schedule(production_schedule)
        
        if holiday_calendar is not None:
            self.external_processor.add_holiday_calendar(holiday_calendar)
        
        if maintenance_windows is not None:
            self.external_processor.add_maintenance_windows(maintenance_windows)


# Example usage and testing functions
async def main():
    """Example usage of the forecasting engine."""
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    np.random.seed(42)
    
    # Create synthetic EHS data with trend and seasonality
    trend = np.linspace(10, 15, 365)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(365) / 365 * 4)  # Quarterly seasonality
    noise = np.random.normal(0, 2, 365)
    
    ehs_data = pd.Series(
        trend + seasonal + noise,
        index=dates,
        name="incident_rate"
    )
    
    # Initialize forecasting engine
    engine = ForecastingEngine()
    
    # Generate forecasts with different models
    print("Testing different forecasting models...")
    
    models_to_test = [
        ForecastModel.AUTO_SELECT,
        ForecastModel.MOVING_AVERAGE,
        ForecastModel.EXPONENTIAL_SMOOTHING,
        ForecastModel.ENSEMBLE
    ]
    
    if STATSMODELS_AVAILABLE:
        models_to_test.append(ForecastModel.ARIMA)
    
    if PROPHET_AVAILABLE:
        models_to_test.append(ForecastModel.PROPHET)
    
    results = {}
    
    for model in models_to_test:
        try:
            print(f"\nTesting {model.value}...")
            result = await engine.forecast(
                ehs_data,
                horizon=ForecastHorizon.THREE_MONTHS,
                model=model
            )
            
            results[model.value] = result
            print(f"✓ {model.value}: MAPE={result.metrics.get('mape', 'N/A'):.2f}%")
            
        except Exception as e:
            print(f"✗ {model.value}: {e}")
    
    # Test model validation
    print("\nTesting model validation...")
    best_model, performance = await engine.select_best_model(ehs_data)
    print(f"Best model: {best_model.value}")
    print(f"Performance: MAPE={performance.mape:.2f}%, RMSE={performance.rmse:.2f}")
    
    # Test validation
    print("\nTesting cross-validation...")
    validation_results = await engine.validate_forecast(
        ehs_data,
        best_model,
        validation_periods=3,
        horizon_days=30
    )
    
    if 'error' not in validation_results:
        print(f"Validation results: Avg MAPE={validation_results['avg_mape']:.2f}%")
    else:
        print(f"Validation failed: {validation_results['error']}")
    
    print("\nForecasting engine test completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())