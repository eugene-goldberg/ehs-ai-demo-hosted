"""
Advanced Anomaly Detection System for EHS Analytics

This module provides a comprehensive anomaly detection system using PyOD best practices
for detecting unusual patterns in environmental, health, and safety data. Features
multiple detection algorithms, ensemble methods, and EHS-specific anomaly types.
"""

import asyncio
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import joblib

# PyOD imports for advanced anomaly detection
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.cblof import CBLOF
    from pyod.models.hbos import HBOS
    from pyod.models.pca import PCA as PyODPCA
    from pyod.models.knn import KNN
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.utils.data import evaluate_print
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

from .base import RiskSeverity


class AnomalyType(Enum):
    """Types of anomalies in EHS data."""
    POINT = "point"  # Single data point anomaly
    CONTEXTUAL = "contextual"  # Anomaly in specific context (time, location)
    COLLECTIVE = "collective"  # Group of data points forming anomaly
    TREND = "trend"  # Unusual trend or pattern
    SEASONAL = "seasonal"  # Deviation from seasonal patterns
    THRESHOLD = "threshold"  # Approaching regulatory thresholds


class DetectorType(Enum):
    """Available anomaly detection algorithms."""
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"
    CLUSTER_BASED_LOF = "cblof"
    HISTOGRAM_BASED_OS = "hbos"
    PCA_BASED = "pca"
    KNN_BASED = "knn"
    COPOD = "copod"
    ECOD = "ecod"


class EnsembleStrategy(Enum):
    """Strategies for combining multiple detectors."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    MAXIMUM = "maximum"
    AVERAGE = "average"
    DYNAMIC_SELECTION = "dynamic_selection"


@dataclass
class AnomalyScore:
    """Represents an anomaly score with confidence and severity."""
    score: float  # Anomaly score (0-1, higher = more anomalous)
    confidence: float  # Confidence in the score (0-1)
    severity: RiskSeverity  # Risk severity level
    detector: str  # Which detector generated this score
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate score ranges."""
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class AnomalyAlert:
    """Represents an anomaly alert with context and recommendations."""
    id: str
    anomaly_type: AnomalyType
    score: AnomalyScore
    data_point: Dict[str, Any]
    context: Dict[str, Any]
    description: str
    recommendations: List[str]
    affected_systems: List[str] = field(default_factory=list)
    regulatory_impact: Optional[str] = None
    estimated_cost: Optional[float] = None
    urgency_hours: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DetectorConfig:
    """Configuration for individual anomaly detectors."""
    detector_type: DetectorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    weight: float = 1.0
    contamination: float = 0.1
    sensitivity: float = 0.5  # 0-1, higher = more sensitive
    
    def __post_init__(self):
        """Set default parameters based on detector type."""
        if not self.parameters:
            self.parameters = self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for each detector type."""
        defaults = {
            DetectorType.ZSCORE: {"threshold": 3.0},
            DetectorType.MODIFIED_ZSCORE: {"threshold": 3.5},
            DetectorType.IQR: {"factor": 1.5},
            DetectorType.ISOLATION_FOREST: {
                "n_estimators": 100,
                "max_samples": "auto",
                "contamination": self.contamination,
                "random_state": 42
            },
            DetectorType.LOCAL_OUTLIER_FACTOR: {
                "n_neighbors": 20,
                "contamination": self.contamination
            },
            DetectorType.ONE_CLASS_SVM: {
                "kernel": "rbf",
                "gamma": "scale",
                "nu": self.contamination
            },
            DetectorType.CLUSTER_BASED_LOF: {
                "contamination": self.contamination,
                "n_clusters": 8
            },
            DetectorType.HISTOGRAM_BASED_OS: {
                "contamination": self.contamination,
                "n_bins": 10
            },
            DetectorType.PCA_BASED: {
                "contamination": self.contamination,
                "n_components": None
            },
            DetectorType.KNN_BASED: {
                "contamination": self.contamination,
                "n_neighbors": 5
            },
            DetectorType.COPOD: {
                "contamination": self.contamination
            },
            DetectorType.ECOD: {
                "contamination": self.contamination
            }
        }
        return defaults.get(self.detector_type, {})


class BaseAnomalyDetector(ABC):
    """Base class for all anomaly detectors."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Fit the detector on training data."""
        pass
    
    @abstractmethod
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for input data."""
        pass
    
    def predict_labels(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary anomaly labels."""
        scores = self.predict_scores(X)
        return (scores > threshold).astype(int)
    
    def save(self, filepath: Path) -> None:
        """Save detector model to disk."""
        model_data = {
            'config': self.config,
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'BaseAnomalyDetector':
        """Load detector model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        detector = cls(model_data['config'])
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.is_fitted = model_data['is_fitted']
        detector.feature_names = model_data['feature_names']
        
        return detector


class StatisticalDetector(BaseAnomalyDetector):
    """Statistical anomaly detector using Z-score, Modified Z-score, or IQR."""
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Fit statistical parameters."""
        self.feature_names = feature_names
        
        if self.config.detector_type == DetectorType.ZSCORE:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        elif self.config.detector_type == DetectorType.MODIFIED_ZSCORE:
            self.median = np.median(X, axis=0)
            self.mad = np.median(np.abs(X - self.median), axis=0)
        elif self.config.detector_type == DetectorType.IQR:
            self.q25 = np.percentile(X, 25, axis=0)
            self.q75 = np.percentile(X, 75, axis=0)
            self.iqr = self.q75 - self.q25
        
        self.is_fitted = True
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores using statistical methods."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        if self.config.detector_type == DetectorType.ZSCORE:
            scores = np.abs((X - self.mean) / (self.std + 1e-8))
            threshold = self.config.parameters.get("threshold", 3.0)
            scores = np.max(scores, axis=1) / threshold
        
        elif self.config.detector_type == DetectorType.MODIFIED_ZSCORE:
            modified_z = 0.6745 * (X - self.median) / (self.mad + 1e-8)
            scores = np.abs(modified_z)
            threshold = self.config.parameters.get("threshold", 3.5)
            scores = np.max(scores, axis=1) / threshold
        
        elif self.config.detector_type == DetectorType.IQR:
            factor = self.config.parameters.get("factor", 1.5)
            lower_bound = self.q25 - factor * self.iqr
            upper_bound = self.q75 + factor * self.iqr
            
            lower_outliers = (X < lower_bound).astype(float)
            upper_outliers = (X > upper_bound).astype(float)
            scores = np.maximum(lower_outliers, upper_outliers)
            scores = np.max(scores, axis=1)
        
        return np.clip(scores, 0, 1)


class MLDetector(BaseAnomalyDetector):
    """Machine learning-based anomaly detector using PyOD or sklearn."""
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Fit ML model."""
        self.feature_names = feature_names
        
        # Scale data for better performance
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model based on detector type
        if PYOD_AVAILABLE:
            self.model = self._create_pyod_model()
        else:
            self.model = self._create_sklearn_model()
        
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def _create_pyod_model(self):
        """Create PyOD model instance."""
        params = self.config.parameters.copy()
        
        model_map = {
            DetectorType.ISOLATION_FOREST: IForest,
            DetectorType.LOCAL_OUTLIER_FACTOR: LOF,
            DetectorType.ONE_CLASS_SVM: OCSVM,
            DetectorType.CLUSTER_BASED_LOF: CBLOF,
            DetectorType.HISTOGRAM_BASED_OS: HBOS,
            DetectorType.PCA_BASED: PyODPCA,
            DetectorType.KNN_BASED: KNN,
            DetectorType.COPOD: COPOD,
            DetectorType.ECOD: ECOD
        }
        
        model_class = model_map.get(self.config.detector_type)
        if model_class:
            return model_class(**params)
        else:
            # Fallback to sklearn
            return self._create_sklearn_model()
    
    def _create_sklearn_model(self):
        """Create sklearn model instance."""
        params = self.config.parameters.copy()
        
        if self.config.detector_type == DetectorType.ISOLATION_FOREST:
            return IsolationForest(**params)
        elif self.config.detector_type == DetectorType.LOCAL_OUTLIER_FACTOR:
            params['novelty'] = True  # For prediction on new data
            return LocalOutlierFactor(**params)
        elif self.config.detector_type == DetectorType.ONE_CLASS_SVM:
            return OneClassSVM(**params)
        else:
            # Default to Isolation Forest
            return IsolationForest(
                contamination=self.config.contamination,
                random_state=42
            )
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores using ML model."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if PYOD_AVAILABLE and hasattr(self.model, 'decision_function'):
            # PyOD models
            scores = self.model.decision_function(X_scaled)
            # Normalize to 0-1 range
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            # sklearn models
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X_scaled)
                # Convert to positive anomaly scores
                scores = -scores  # sklearn returns negative scores for anomalies
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            else:
                # Binary predictions only
                predictions = self.model.predict(X_scaled)
                scores = (predictions == -1).astype(float)  # -1 indicates outlier
        
        return np.clip(scores, 0, 1)


class EnsembleDetector:
    """Ensemble of multiple anomaly detectors."""
    
    def __init__(
        self,
        detectors: List[BaseAnomalyDetector],
        strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE
    ):
        self.detectors = detectors
        self.strategy = strategy
        self.weights = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Fit all detectors in the ensemble."""
        for detector in self.detectors:
            detector.fit(X, feature_names)
        
        # Calculate weights based on detector performance if strategy requires it
        if self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            self.weights = self._calculate_weights(X)
        else:
            self.weights = [detector.config.weight for detector in self.detectors]
        
        self.is_fitted = True
    
    def _calculate_weights(self, X: np.ndarray) -> List[float]:
        """Calculate detector weights based on performance."""
        weights = []
        
        for detector in self.detectors:
            try:
                scores = detector.predict_scores(X)
                # Simple heuristic: weight based on score distribution
                score_std = np.std(scores)
                weight = score_std * detector.config.weight
                weights.append(weight)
            except Exception:
                weights.append(detector.config.weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(self.detectors)] * len(self.detectors)
        
        return weights
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        all_scores = []
        for detector in self.detectors:
            if detector.config.enabled:
                try:
                    scores = detector.predict_scores(X)
                    all_scores.append(scores)
                except Exception as e:
                    print(f"Warning: Detector {detector.config.detector_type} failed: {e}")
                    continue
        
        if not all_scores:
            raise ValueError("No detectors produced valid scores")
        
        scores_matrix = np.column_stack(all_scores)
        
        # Apply ensemble strategy
        if self.strategy == EnsembleStrategy.MAJORITY_VOTE:
            # Binarize scores and take majority vote
            binary_scores = (scores_matrix > 0.5).astype(int)
            final_scores = np.mean(binary_scores, axis=1)
        
        elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            final_scores = np.average(scores_matrix, axis=1, weights=self.weights)
        
        elif self.strategy == EnsembleStrategy.MAXIMUM:
            final_scores = np.max(scores_matrix, axis=1)
        
        elif self.strategy == EnsembleStrategy.AVERAGE:
            final_scores = np.mean(scores_matrix, axis=1)
        
        elif self.strategy == EnsembleStrategy.DYNAMIC_SELECTION:
            # Select best detector per sample based on confidence
            final_scores = np.max(scores_matrix, axis=1)
        
        else:
            final_scores = np.mean(scores_matrix, axis=1)
        
        return np.clip(final_scores, 0, 1)


class AnomalyDetectionSystem:
    """Main anomaly detection system with multiple detectors and EHS-specific features."""
    
    def __init__(
        self,
        detector_configs: Optional[List[DetectorConfig]] = None,
        ensemble_strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
        model_dir: Optional[Path] = None
    ):
        self.detector_configs = detector_configs or self._get_default_configs()
        self.ensemble_strategy = ensemble_strategy
        self.model_dir = Path(model_dir) if model_dir else Path("models/anomaly")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.detectors = []
        self.ensemble = None
        self.is_fitted = False
        self.baseline_stats = {}
        self.drift_detector = None
        
        # EHS-specific thresholds and rules
        self.ehs_thresholds = {
            'electricity_consumption': {'warning': 0.8, 'critical': 0.9},
            'water_consumption': {'warning': 0.8, 'critical': 0.9},
            'waste_generation': {'warning': 0.75, 'critical': 0.9},
            'emissions_co2': {'warning': 0.8, 'critical': 0.95},
            'permit_utilization': {'warning': 0.85, 'critical': 0.95}
        }
        
        self.context_rules = {
            'seasonal_adjustment': True,
            'time_of_day_adjustment': True,
            'equipment_state_consideration': True,
            'weather_adjustment': True
        }
    
    def _get_default_configs(self) -> List[DetectorConfig]:
        """Get default detector configurations optimized for EHS data."""
        configs = [
            DetectorConfig(
                detector_type=DetectorType.ISOLATION_FOREST,
                weight=2.0,
                contamination=0.05,
                parameters={
                    'n_estimators': 200,
                    'max_samples': 'auto',
                    'contamination': 0.05,
                    'random_state': 42
                }
            ),
            DetectorConfig(
                detector_type=DetectorType.LOCAL_OUTLIER_FACTOR,
                weight=1.5,
                contamination=0.05,
                parameters={
                    'n_neighbors': 25,
                    'contamination': 0.05
                }
            ),
            DetectorConfig(
                detector_type=DetectorType.MODIFIED_ZSCORE,
                weight=1.0,
                contamination=0.05,
                parameters={'threshold': 3.0}
            ),
            DetectorConfig(
                detector_type=DetectorType.IQR,
                weight=0.8,
                contamination=0.05,
                parameters={'factor': 2.0}
            )
        ]
        
        # Add PyOD-specific detectors if available
        if PYOD_AVAILABLE:
            configs.extend([
                DetectorConfig(
                    detector_type=DetectorType.COPOD,
                    weight=1.2,
                    contamination=0.05
                ),
                DetectorConfig(
                    detector_type=DetectorType.ECOD,
                    weight=1.3,
                    contamination=0.05
                )
            ])
        
        return configs
    
    async def train_detectors(
        self,
        training_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        save_models: bool = True
    ) -> None:
        """Train all detectors on historical data."""
        print("Training anomaly detectors...")
        
        # Prepare training data
        if feature_columns is None:
            feature_columns = [col for col in training_data.columns 
                             if training_data[col].dtype in ['int64', 'float64']]
        
        X = training_data[feature_columns].values
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Store baseline statistics
        self.baseline_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'median': np.median(X, axis=0),
            'q25': np.percentile(X, 25, axis=0),
            'q75': np.percentile(X, 75, axis=0),
            'feature_names': feature_columns,
            'training_size': len(X),
            'training_date': datetime.now()
        }
        
        # Initialize detectors
        self.detectors = []
        for config in self.detector_configs:
            if config.enabled:
                if config.detector_type in [DetectorType.ZSCORE, DetectorType.MODIFIED_ZSCORE, DetectorType.IQR]:
                    detector = StatisticalDetector(config)
                else:
                    detector = MLDetector(config)
                
                self.detectors.append(detector)
        
        # Train detectors
        tasks = []
        for detector in self.detectors:
            task = asyncio.create_task(self._train_single_detector(detector, X, feature_columns))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Create ensemble
        self.ensemble = EnsembleDetector(self.detectors, self.ensemble_strategy)
        self.ensemble.fit(X, feature_columns)
        
        self.is_fitted = True
        
        if save_models:
            await self._save_models()
        
        print(f"Successfully trained {len(self.detectors)} detectors")
    
    async def _train_single_detector(
        self,
        detector: BaseAnomalyDetector,
        X: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Train a single detector asynchronously."""
        try:
            detector.fit(X, feature_names)
            print(f"Trained {detector.config.detector_type.value} detector")
        except Exception as e:
            print(f"Failed to train {detector.config.detector_type.value}: {e}")
            detector.config.enabled = False
    
    async def detect_anomalies(
        self,
        data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
        return_explanations: bool = True
    ) -> List[AnomalyAlert]:
        """Detect anomalies in new data."""
        if not self.is_fitted:
            raise ValueError("System must be trained before anomaly detection")
        
        alerts = []
        feature_names = self.baseline_stats['feature_names']
        X = data[feature_names].values
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Get ensemble scores
        scores = self.ensemble.predict_scores(X)
        
        # Process each data point
        for idx, (_, row) in enumerate(data.iterrows()):
            score = scores[idx]
            
            # Determine severity
            severity = self._calculate_severity(score, row, context)
            
            if severity != RiskSeverity.LOW:
                # Create anomaly score object
                anomaly_score = AnomalyScore(
                    score=score,
                    confidence=self._calculate_confidence(score, row),
                    severity=severity,
                    detector="ensemble"
                )
                
                # Determine anomaly type
                anomaly_type = self._classify_anomaly_type(row, context, score)
                
                # Generate alert
                alert = await self._create_alert(
                    row, anomaly_score, anomaly_type, context, return_explanations
                )
                
                alerts.append(alert)
        
        return alerts
    
    def detect_statistical_anomalies(self, 
                                    data: np.ndarray, 
                                    threshold: float = 3.0,
                                    method: str = 'zscore') -> Dict[str, Any]:
        """
        Detect statistical anomalies using z-score or modified z-score.
        
        Args:
            data: Input data array
            threshold: Anomaly threshold (default 3.0 for z-score)
            method: Detection method ('zscore', 'modified_zscore', 'iqr')
            
        Returns:
            Dictionary with anomaly indices, scores, and metadata
        """
        if method == 'zscore':
            return self._detect_zscore_anomalies(data, threshold)
        elif method == 'modified_zscore':
            return self._detect_modified_zscore_anomalies(data, threshold)
        elif method == 'iqr':
            return self._detect_iqr_anomalies(data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_zscore_anomalies(self, data: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Detect anomalies using z-score method."""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return {'indices': [], 'scores': [], 'threshold': threshold, 'method': 'zscore'}
        
        z_scores = np.abs((data - mean) / std)
        anomaly_mask = z_scores > threshold
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_scores = z_scores[anomaly_mask].tolist()
        
        return {
            'indices': anomaly_indices,
            'scores': anomaly_scores,
            'threshold': threshold,
            'method': 'zscore',
            'mean': float(mean),
            'std': float(std),
            'num_anomalies': len(anomaly_indices)
        }

    def _detect_modified_zscore_anomalies(self, data: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Detect anomalies using modified z-score (based on median)."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return {'indices': [], 'scores': [], 'threshold': threshold, 'method': 'modified_zscore'}
        
        modified_z_scores = 0.6745 * (data - median) / mad
        abs_modified_z_scores = np.abs(modified_z_scores)
        anomaly_mask = abs_modified_z_scores > threshold
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_scores = abs_modified_z_scores[anomaly_mask].tolist()
        
        return {
            'indices': anomaly_indices,
            'scores': anomaly_scores,
            'threshold': threshold,
            'method': 'modified_zscore',
            'median': float(median),
            'mad': float(mad),
            'num_anomalies': len(anomaly_indices)
        }

    def _detect_iqr_anomalies(self, data: np.ndarray, factor: float = 1.5) -> Dict[str, Any]:
        """Detect anomalies using IQR method."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        anomaly_mask = (data < lower_bound) | (data > upper_bound)
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        
        # Calculate scores based on distance from bounds
        scores = []
        for idx in anomaly_indices:
            val = data[idx]
            if val < lower_bound:
                score = (lower_bound - val) / iqr if iqr > 0 else 1.0
            else:
                score = (val - upper_bound) / iqr if iqr > 0 else 1.0
            scores.append(float(score))
        
        return {
            'indices': anomaly_indices,
            'scores': scores,
            'method': 'iqr',
            'factor': factor,
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'num_anomalies': len(anomaly_indices)
        }
    
    async def real_time_detection(
        self,
        data_stream: asyncio.Queue,
        alert_callback: Callable[[AnomalyAlert], None],
        context_provider: Optional[Callable[[], Dict[str, Any]]] = None
    ) -> None:
        """Real-time anomaly detection from data stream."""
        print("Starting real-time anomaly detection...")
        
        while True:
            try:
                # Get next data point from stream
                data_point = await asyncio.wait_for(data_stream.get(), timeout=1.0)
                
                if data_point is None:  # Shutdown signal
                    break
                
                # Convert to DataFrame for processing
                df = pd.DataFrame([data_point])
                
                # Get current context
                context = context_provider() if context_provider else None
                
                # Detect anomalies
                alerts = await self.detect_anomalies(df, context, return_explanations=False)
                
                # Send alerts
                for alert in alerts:
                    await asyncio.create_task(self._send_alert(alert, alert_callback))
                
                # Check for concept drift periodically
                if np.random.random() < 0.01:  # 1% chance to check drift
                    await self._check_concept_drift(df)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in real-time detection: {e}")
                await asyncio.sleep(1)
    
    async def explain_anomaly(
        self,
        data_point: pd.Series,
        anomaly_score: AnomalyScore,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Provide detailed explanation for why a data point is anomalous."""
        explanation = {
            'overall_score': anomaly_score.score,
            'severity': anomaly_score.severity.value,
            'feature_contributions': {},
            'statistical_analysis': {},
            'contextual_factors': {},
            'similar_historical_cases': [],
            'recommendations': []
        }
        
        # Feature-level analysis
        feature_names = self.baseline_stats['feature_names']
        baseline_mean = self.baseline_stats['mean']
        baseline_std = self.baseline_stats['std']
        
        for i, feature in enumerate(feature_names):
            value = data_point[feature]
            mean_val = baseline_mean[i]
            std_val = baseline_std[i]
            
            z_score = (value - mean_val) / (std_val + 1e-8)
            
            explanation['feature_contributions'][feature] = {
                'value': float(value),
                'baseline_mean': float(mean_val),
                'z_score': float(z_score),
                'deviation_magnitude': abs(z_score),
                'is_outlier': abs(z_score) > 2.0
            }
        
        # Statistical analysis
        explanation['statistical_analysis'] = {
            'multivariate_distance': self._calculate_mahalanobis_distance(data_point),
            'percentile_rank': self._calculate_percentile_rank(data_point),
            'density_estimate': self._calculate_density_estimate(data_point)
        }
        
        # Contextual factors
        if context:
            explanation['contextual_factors'] = self._analyze_contextual_factors(
                data_point, context
            )
        
        # EHS-specific recommendations
        explanation['recommendations'] = self._generate_ehs_recommendations(
            data_point, anomaly_score, context
        )
        
        return explanation
    
    async def update_baseline(
        self,
        new_data: pd.DataFrame,
        update_strategy: str = "incremental"
    ) -> None:
        """Update baseline statistics and retrain if necessary."""
        if update_strategy == "incremental":
            # Incrementally update baseline statistics
            self._update_baseline_incremental(new_data)
        
        elif update_strategy == "full_retrain":
            # Full retraining (expensive)
            await self.train_detectors(new_data)
        
        elif update_strategy == "sliding_window":
            # Update using sliding window approach
            self._update_baseline_sliding_window(new_data)
        
        print(f"Updated baseline using {update_strategy} strategy")
    
    def _calculate_severity(
        self,
        score: float,
        data_point: pd.Series,
        context: Optional[Dict[str, Any]]
    ) -> RiskSeverity:
        """Calculate risk severity based on score and context."""
        # Base severity from score
        if score < 0.3:
            base_severity = RiskSeverity.LOW
        elif score < 0.6:
            base_severity = RiskSeverity.MEDIUM
        elif score < 0.8:
            base_severity = RiskSeverity.HIGH
        else:
            base_severity = RiskSeverity.CRITICAL
        
        # Adjust based on EHS-specific factors
        adjusted_severity = self._adjust_severity_for_ehs_context(
            base_severity, data_point, context
        )
        
        return adjusted_severity
    
    def _calculate_confidence(self, score: float, data_point: pd.Series) -> float:
        """Calculate confidence in the anomaly score."""
        # Simple heuristic: higher scores and consistent detector agreement = higher confidence
        confidence = min(score * 1.2, 1.0)
        
        # Adjust based on data quality
        missing_ratio = data_point.isnull().sum() / len(data_point)
        confidence *= (1.0 - missing_ratio * 0.5)
        
        return max(0.1, confidence)
    
    def _classify_anomaly_type(
        self,
        data_point: pd.Series,
        context: Optional[Dict[str, Any]],
        score: float
    ) -> AnomalyType:
        """Classify the type of anomaly."""
        # Simple heuristic-based classification
        if context and 'timestamp' in context:
            # Check for seasonal/temporal patterns
            if self._is_seasonal_anomaly(data_point, context):
                return AnomalyType.SEASONAL
        
        # Check if it's approaching regulatory thresholds
        if self._is_threshold_anomaly(data_point):
            return AnomalyType.THRESHOLD
        
        # Check for trend anomalies (would need historical context)
        if score > 0.8:
            return AnomalyType.POINT
        
        return AnomalyType.CONTEXTUAL
    
    async def _create_alert(
        self,
        data_point: pd.Series,
        anomaly_score: AnomalyScore,
        anomaly_type: AnomalyType,
        context: Optional[Dict[str, Any]],
        include_explanation: bool
    ) -> AnomalyAlert:
        """Create detailed anomaly alert."""
        alert_id = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(data_point.values)) % 10000}"
        
        # Generate description
        description = self._generate_alert_description(data_point, anomaly_score, anomaly_type)
        
        # Generate recommendations
        recommendations = self._generate_ehs_recommendations(data_point, anomaly_score, context)
        
        # Identify affected systems
        affected_systems = self._identify_affected_systems(data_point, context)
        
        # Estimate regulatory impact and costs
        regulatory_impact = self._assess_regulatory_impact(data_point, anomaly_type)
        estimated_cost = self._estimate_impact_cost(data_point, anomaly_score)
        
        # Calculate urgency
        urgency_hours = self._calculate_urgency_hours(anomaly_score, anomaly_type)
        
        alert = AnomalyAlert(
            id=alert_id,
            anomaly_type=anomaly_type,
            score=anomaly_score,
            data_point=data_point.to_dict(),
            context=context or {},
            description=description,
            recommendations=recommendations,
            affected_systems=affected_systems,
            regulatory_impact=regulatory_impact,
            estimated_cost=estimated_cost,
            urgency_hours=urgency_hours
        )
        
        return alert
    
    def _generate_alert_description(
        self,
        data_point: pd.Series,
        anomaly_score: AnomalyScore,
        anomaly_type: AnomalyType
    ) -> str:
        """Generate human-readable alert description."""
        severity_text = anomaly_score.severity.value.upper()
        score_text = f"{anomaly_score.score:.2%}"
        
        # Find most anomalous features
        feature_names = self.baseline_stats['feature_names']
        baseline_mean = self.baseline_stats['mean']
        baseline_std = self.baseline_stats['std']
        
        anomalous_features = []
        for i, feature in enumerate(feature_names):
            if feature in data_point:
                value = data_point[feature]
                z_score = (value - baseline_mean[i]) / (baseline_std[i] + 1e-8)
                if abs(z_score) > 2.0:
                    anomalous_features.append((feature, value, z_score))
        
        # Sort by deviation magnitude
        anomalous_features.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if anomalous_features:
            top_feature, top_value, top_z = anomalous_features[0]
            description = (
                f"{severity_text} {anomaly_type.value} anomaly detected "
                f"(score: {score_text}). Primary concern: {top_feature} = {top_value:.2f} "
                f"({abs(top_z):.1f} standard deviations from baseline)."
            )
        else:
            description = (
                f"{severity_text} {anomaly_type.value} anomaly detected "
                f"with confidence score of {score_text}."
            )
        
        return description
    
    def _generate_ehs_recommendations(
        self,
        data_point: pd.Series,
        anomaly_score: AnomalyScore,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate EHS-specific recommendations."""
        recommendations = []
        
        # Severity-based recommendations
        if anomaly_score.severity == RiskSeverity.CRITICAL:
            recommendations.extend([
                "Immediately investigate the root cause",
                "Implement emergency response procedures if applicable",
                "Notify relevant regulatory authorities if required",
                "Document all actions taken for compliance reporting"
            ])
        elif anomaly_score.severity == RiskSeverity.HIGH:
            recommendations.extend([
                "Schedule urgent investigation within 24 hours",
                "Review related operational procedures",
                "Consider temporary mitigation measures"
            ])
        elif anomaly_score.severity == RiskSeverity.MEDIUM:
            recommendations.extend([
                "Plan investigation within 72 hours",
                "Monitor related metrics closely",
                "Review maintenance schedules"
            ])
        
        # Feature-specific recommendations
        feature_names = self.baseline_stats['feature_names']
        for feature in feature_names:
            if feature in data_point:
                feature_recommendations = self._get_feature_specific_recommendations(
                    feature, data_point[feature], context
                )
                recommendations.extend(feature_recommendations)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _get_feature_specific_recommendations(
        self,
        feature: str,
        value: float,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Get recommendations specific to a feature."""
        recommendations = []
        
        if "electricity" in feature.lower():
            recommendations.extend([
                "Check electrical equipment for faults",
                "Review HVAC system efficiency",
                "Audit energy management system settings"
            ])
        elif "water" in feature.lower():
            recommendations.extend([
                "Inspect water systems for leaks",
                "Review water treatment processes",
                "Check permit compliance status"
            ])
        elif "waste" in feature.lower():
            recommendations.extend([
                "Review waste segregation practices",
                "Check waste contractor schedules",
                "Verify waste stream classifications"
            ])
        elif "emission" in feature.lower():
            recommendations.extend([
                "Inspect emission control equipment",
                "Review combustion processes",
                "Check environmental monitoring systems"
            ])
        
        return recommendations
    
    def _adjust_severity_for_ehs_context(
        self,
        base_severity: RiskSeverity,
        data_point: pd.Series,
        context: Optional[Dict[str, Any]]
    ) -> RiskSeverity:
        """Adjust severity based on EHS-specific context."""
        # Check if approaching regulatory thresholds
        for feature, thresholds in self.ehs_thresholds.items():
            if feature in data_point:
                value = data_point[feature]
                baseline_max = self.baseline_stats.get('q75', [1000])[0]  # Fallback
                
                utilization = value / baseline_max if baseline_max > 0 else 0
                
                if utilization > thresholds['critical']:
                    return RiskSeverity.CRITICAL
                elif utilization > thresholds['warning'] and base_severity.numeric_value < 3:
                    return RiskSeverity.HIGH
        
        return base_severity
    
    def _is_seasonal_anomaly(
        self,
        data_point: pd.Series,
        context: Dict[str, Any]
    ) -> bool:
        """Check if anomaly is seasonal."""
        # Simple heuristic - would need more sophisticated seasonal analysis
        if 'timestamp' in context:
            timestamp = context['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Check for weekend vs weekday patterns
            if timestamp.weekday() >= 5:  # Weekend
                return True
        
        return False
    
    def _is_threshold_anomaly(self, data_point: pd.Series) -> bool:
        """Check if anomaly is due to approaching thresholds."""
        for feature, thresholds in self.ehs_thresholds.items():
            if feature in data_point:
                value = data_point[feature]
                baseline_max = self.baseline_stats.get('q75', [1000])[0]
                utilization = value / baseline_max if baseline_max > 0 else 0
                
                if utilization > thresholds['warning']:
                    return True
        
        return False
    
    def _identify_affected_systems(
        self,
        data_point: pd.Series,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify systems affected by the anomaly."""
        systems = []
        
        feature_names = list(data_point.index)
        
        for feature in feature_names:
            if "electricity" in feature.lower():
                systems.extend(["Electrical Grid", "HVAC", "Lighting"])
            elif "water" in feature.lower():
                systems.extend(["Water Treatment", "Cooling Systems"])
            elif "waste" in feature.lower():
                systems.extend(["Waste Management", "Recycling"])
            elif "emission" in feature.lower():
                systems.extend(["Emission Control", "Air Quality Monitoring"])
        
        return list(set(systems))
    
    def _assess_regulatory_impact(
        self,
        data_point: pd.Series,
        anomaly_type: AnomalyType
    ) -> Optional[str]:
        """Assess potential regulatory impact."""
        if anomaly_type == AnomalyType.THRESHOLD:
            return "Potential permit violation risk"
        elif anomaly_type == AnomalyType.TREND:
            return "Trend may lead to regulatory non-compliance"
        else:
            return None
    
    def _estimate_impact_cost(
        self,
        data_point: pd.Series,
        anomaly_score: AnomalyScore
    ) -> Optional[float]:
        """Estimate potential cost impact."""
        # Simple cost estimation based on severity and features
        base_cost = {
            RiskSeverity.LOW: 1000,
            RiskSeverity.MEDIUM: 5000,
            RiskSeverity.HIGH: 25000,
            RiskSeverity.CRITICAL: 100000
        }.get(anomaly_score.severity, 1000)
        
        # Adjust based on affected features
        multiplier = 1.0
        feature_names = list(data_point.index)
        
        for feature in feature_names:
            if "emission" in feature.lower():
                multiplier *= 2.0  # Environmental violations are expensive
            elif "permit" in feature.lower():
                multiplier *= 1.5
        
        return base_cost * multiplier
    
    def _calculate_urgency_hours(
        self,
        anomaly_score: AnomalyScore,
        anomaly_type: AnomalyType
    ) -> int:
        """Calculate response urgency in hours."""
        urgency_map = {
            RiskSeverity.CRITICAL: 1,
            RiskSeverity.HIGH: 8,
            RiskSeverity.MEDIUM: 72,
            RiskSeverity.LOW: 168
        }
        
        base_urgency = urgency_map.get(anomaly_score.severity, 168)
        
        # Adjust based on anomaly type
        if anomaly_type == AnomalyType.THRESHOLD:
            base_urgency = min(base_urgency, 4)  # Very urgent for threshold issues
        
        return base_urgency
    
    def _calculate_mahalanobis_distance(self, data_point: pd.Series) -> float:
        """Calculate Mahalanobis distance from baseline."""
        try:
            feature_names = self.baseline_stats['feature_names']
            values = np.array([data_point[f] for f in feature_names])
            mean = self.baseline_stats['mean']
            
            # Simple approximation using diagonal covariance
            std = self.baseline_stats['std']
            normalized = (values - mean) / (std + 1e-8)
            distance = np.sqrt(np.sum(normalized ** 2))
            
            return float(distance)
        except Exception:
            return 0.0
    
    def _calculate_percentile_rank(self, data_point: pd.Series) -> Dict[str, float]:
        """Calculate percentile rank for each feature."""
        ranks = {}
        
        try:
            feature_names = self.baseline_stats['feature_names']
            for feature in feature_names:
                if feature in data_point:
                    value = data_point[feature]
                    # Approximate percentile using z-score
                    mean = self.baseline_stats['mean'][feature_names.index(feature)]
                    std = self.baseline_stats['std'][feature_names.index(feature)]
                    z_score = (value - mean) / (std + 1e-8)
                    percentile = stats.norm.cdf(z_score) * 100
                    ranks[feature] = float(percentile)
        except Exception:
            pass
        
        return ranks
    
    def _calculate_density_estimate(self, data_point: pd.Series) -> float:
        """Estimate data point density."""
        # Simple density estimation using Mahalanobis distance
        distance = self._calculate_mahalanobis_distance(data_point)
        density = np.exp(-0.5 * distance)  # Gaussian-like density
        return float(density)
    
    def _analyze_contextual_factors(
        self,
        data_point: pd.Series,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze contextual factors affecting the anomaly."""
        factors = {}
        
        if 'timestamp' in context:
            timestamp = context['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            factors['time_of_day'] = timestamp.hour
            factors['day_of_week'] = timestamp.strftime('%A')
            factors['is_weekend'] = timestamp.weekday() >= 5
            factors['month'] = timestamp.strftime('%B')
        
        if 'weather' in context:
            factors['weather_conditions'] = context['weather']
        
        if 'equipment_status' in context:
            factors['equipment_status'] = context['equipment_status']
        
        return factors
    
    async def _check_concept_drift(self, recent_data: pd.DataFrame) -> None:
        """Check for concept drift in the data distribution."""
        # Simple drift detection using statistical tests
        try:
            feature_names = self.baseline_stats['feature_names']
            recent_values = recent_data[feature_names].values
            
            # Compare recent data statistics with baseline
            recent_mean = np.mean(recent_values, axis=0)
            baseline_mean = self.baseline_stats['mean']
            
            # Calculate drift score
            drift_scores = np.abs(recent_mean - baseline_mean) / (self.baseline_stats['std'] + 1e-8)
            max_drift = np.max(drift_scores)
            
            if max_drift > 3.0:  # Significant drift detected
                print(f"Concept drift detected (max drift: {max_drift:.2f})")
                # Could trigger model retraining here
        
        except Exception as e:
            print(f"Error checking concept drift: {e}")
    
    async def _send_alert(
        self,
        alert: AnomalyAlert,
        callback: Callable[[AnomalyAlert], None]
    ) -> None:
        """Send alert through callback."""
        try:
            callback(alert)
        except Exception as e:
            print(f"Error sending alert {alert.id}: {e}")
    
    def _update_baseline_incremental(self, new_data: pd.DataFrame) -> None:
        """Incrementally update baseline statistics."""
        feature_names = self.baseline_stats['feature_names']
        new_values = new_data[feature_names].values
        
        # Update statistics incrementally
        old_size = self.baseline_stats['training_size']
        new_size = len(new_values)
        total_size = old_size + new_size
        
        # Update mean
        old_mean = self.baseline_stats['mean']
        new_mean = np.mean(new_values, axis=0)
        updated_mean = (old_mean * old_size + new_mean * new_size) / total_size
        
        # Update other statistics (simplified)
        self.baseline_stats['mean'] = updated_mean
        self.baseline_stats['training_size'] = total_size
        self.baseline_stats['training_date'] = datetime.now()
        
        print(f"Updated baseline with {new_size} new samples")
    
    def _update_baseline_sliding_window(self, new_data: pd.DataFrame) -> None:
        """Update baseline using sliding window approach."""
        # This would maintain a rolling window of recent data
        # Implementation depends on specific requirements
        print("Sliding window baseline update not fully implemented")
    
    async def _save_models(self) -> None:
        """Save all trained models to disk."""
        try:
            # Save individual detectors
            for i, detector in enumerate(self.detectors):
                detector_path = self.model_dir / f"detector_{i}_{detector.config.detector_type.value}.pkl"
                detector.save(detector_path)
            
            # Save baseline statistics
            baseline_path = self.model_dir / "baseline_stats.pkl"
            with open(baseline_path, 'wb') as f:
                pickle.dump(self.baseline_stats, f)
            
            # Save system configuration
            config_path = self.model_dir / "system_config.pkl"
            config_data = {
                'detector_configs': self.detector_configs,
                'ensemble_strategy': self.ensemble_strategy,
                'ehs_thresholds': self.ehs_thresholds,
                'context_rules': self.context_rules
            }
            with open(config_path, 'wb') as f:
                pickle.dump(config_data, f)
            
            print(f"Models saved to {self.model_dir}")
        
        except Exception as e:
            print(f"Error saving models: {e}")
    
    async def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            # Load baseline statistics
            baseline_path = self.model_dir / "baseline_stats.pkl"
            with open(baseline_path, 'rb') as f:
                self.baseline_stats = pickle.load(f)
            
            # Load system configuration
            config_path = self.model_dir / "system_config.pkl"
            with open(config_path, 'rb') as f:
                config_data = pickle.load(f)
                self.detector_configs = config_data['detector_configs']
                self.ensemble_strategy = config_data['ensemble_strategy']
                self.ehs_thresholds = config_data['ehs_thresholds']
                self.context_rules = config_data['context_rules']
            
            # Load detectors
            self.detectors = []
            for i, config in enumerate(self.detector_configs):
                if config.enabled:
                    detector_path = self.model_dir / f"detector_{i}_{config.detector_type.value}.pkl"
                    if detector_path.exists():
                        if config.detector_type in [DetectorType.ZSCORE, DetectorType.MODIFIED_ZSCORE, DetectorType.IQR]:
                            detector = StatisticalDetector.load(detector_path)
                        else:
                            detector = MLDetector.load(detector_path)
                        self.detectors.append(detector)
            
            # Recreate ensemble
            if self.detectors:
                self.ensemble = EnsembleDetector(self.detectors, self.ensemble_strategy)
                self.ensemble.is_fitted = True
                self.is_fitted = True
            
            print(f"Models loaded from {self.model_dir}")
            return True
        
        except Exception as e:
            print(f"Error loading models: {e}")
            return False


# Utility functions for EHS-specific anomaly detection

def create_ehs_anomaly_system(
    contamination_rate: float = 0.05,
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
    model_dir: Optional[str] = None
) -> AnomalyDetectionSystem:
    """Create a pre-configured anomaly detection system for EHS data."""
    
    configs = [
        DetectorConfig(
            detector_type=DetectorType.ISOLATION_FOREST,
            weight=2.0,
            contamination=contamination_rate,
            parameters={
                'n_estimators': 200,
                'max_samples': 'auto',
                'contamination': contamination_rate,
                'random_state': 42
            }
        ),
        DetectorConfig(
            detector_type=DetectorType.LOCAL_OUTLIER_FACTOR,
            weight=1.5,
            contamination=contamination_rate,
            parameters={
                'n_neighbors': 25,
                'contamination': contamination_rate
            }
        ),
        DetectorConfig(
            detector_type=DetectorType.MODIFIED_ZSCORE,
            weight=1.0,
            parameters={'threshold': 3.0}
        ),
        DetectorConfig(
            detector_type=DetectorType.IQR,
            weight=0.8,
            parameters={'factor': 2.0}
        )
    ]
    
    # Add advanced detectors if PyOD is available
    if PYOD_AVAILABLE:
        configs.extend([
            DetectorConfig(
                detector_type=DetectorType.COPOD,
                weight=1.2,
                contamination=contamination_rate
            ),
            DetectorConfig(
                detector_type=DetectorType.ECOD,
                weight=1.3,
                contamination=contamination_rate
            )
        ])
    
    return AnomalyDetectionSystem(
        detector_configs=configs,
        ensemble_strategy=ensemble_strategy,
        model_dir=Path(model_dir) if model_dir else None
    )


async def run_anomaly_detection_example():
    """Example usage of the anomaly detection system."""
    # Create sample EHS data
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = {
        'electricity_consumption': np.random.normal(100, 15, n_samples),
        'water_consumption': np.random.normal(50, 8, n_samples),
        'waste_generation': np.random.normal(25, 5, n_samples),
        'co2_emissions': np.random.normal(75, 12, n_samples),
        'permit_utilization': np.random.uniform(0.6, 0.8, n_samples)
    }
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, 50, replace=False)
    for idx in anomaly_indices:
        # Spike anomalies
        normal_data['electricity_consumption'][idx] *= 2.5
        normal_data['water_consumption'][idx] *= 1.8
    
    df = pd.DataFrame(normal_data)
    
    # Create and train system
    system = create_ehs_anomaly_system()
    await system.train_detectors(df)
    
    # Test detection on new data
    test_data = pd.DataFrame({
        'electricity_consumption': [200, 95, 300],  # One anomaly
        'water_consumption': [52, 48, 45],
        'waste_generation': [26, 24, 28],
        'co2_emissions': [78, 72, 76],
        'permit_utilization': [0.75, 0.68, 0.95]  # One threshold anomaly
    })
    
    alerts = await system.detect_anomalies(test_data)
    
    print(f"Detected {len(alerts)} anomalies:")
    for alert in alerts:
        print(f"  - {alert.description}")
        print(f"    Severity: {alert.score.severity.value}")
        print(f"    Recommendations: {', '.join(alert.recommendations[:2])}")
        print()


if __name__ == "__main__":
    # Run example
    asyncio.run(run_anomaly_detection_example())