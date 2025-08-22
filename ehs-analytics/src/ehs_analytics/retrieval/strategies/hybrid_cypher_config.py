"""
Configuration management for HybridCypher retriever with temporal analytics.

This module provides comprehensive configuration management for the HybridCypher
retriever, including time window configurations, temporal weight decay functions,
pattern matching thresholds, and performance tuning parameters.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TemporalWeightDecayFunction(str, Enum):
    """Types of temporal weight decay functions."""
    
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    STEP = "step"
    NONE = "none"


class TimeWindowGranularity(str, Enum):
    """Time window granularities for aggregation."""
    
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class PerformanceProfile(str, Enum):
    """Performance optimization profiles."""
    
    SPEED = "speed"          # Optimize for fast response times
    ACCURACY = "accuracy"    # Optimize for result quality
    BALANCED = "balanced"    # Balance speed and accuracy
    COMPREHENSIVE = "comprehensive"  # Maximize temporal analysis depth


@dataclass
class TimeWindowConfiguration:
    """Configuration for time windows in temporal queries."""
    
    # Default time windows by query type (in days)
    consumption_window: int = 180  # 6 months
    emissions_window: int = 365    # 1 year
    compliance_window: int = 730   # 2 years
    risk_window: int = 90          # 3 months
    efficiency_window: int = 180   # 6 months
    general_window: int = 180      # 6 months
    
    # Maximum time windows (safety limits)
    max_window_days: int = 1825    # 5 years
    min_window_days: int = 7       # 1 week
    
    # Granularity settings
    default_granularity: TimeWindowGranularity = TimeWindowGranularity.MONTH
    auto_adjust_granularity: bool = True
    
    # Time zone handling
    default_timezone: str = "UTC"
    convert_to_local: bool = False
    
    def get_window_for_query_type(self, query_type: str) -> int:
        """Get appropriate time window for a query type."""
        mapping = {
            "consumption": self.consumption_window,
            "emissions": self.emissions_window,
            "compliance": self.compliance_window,
            "risk": self.risk_window,
            "efficiency": self.efficiency_window
        }
        return mapping.get(query_type.lower(), self.general_window)
    
    def adjust_granularity_for_window(self, window_days: int) -> TimeWindowGranularity:
        """Automatically adjust granularity based on time window size."""
        if not self.auto_adjust_granularity:
            return self.default_granularity
        
        if window_days <= 7:
            return TimeWindowGranularity.HOUR
        elif window_days <= 31:
            return TimeWindowGranularity.DAY
        elif window_days <= 120:
            return TimeWindowGranularity.WEEK
        elif window_days <= 730:
            return TimeWindowGranularity.MONTH
        elif window_days <= 1460:
            return TimeWindowGranularity.QUARTER
        else:
            return TimeWindowGranularity.YEAR


@dataclass
class PatternMatchingThresholds:
    """Thresholds for pattern detection and matching."""
    
    # Seasonal pattern detection
    seasonal_confidence_threshold: float = 0.7
    seasonal_strength_threshold: float = 0.3
    seasonal_min_periods: int = 2
    
    # Trend detection
    trend_confidence_threshold: float = 0.6
    trend_significance_threshold: float = 0.05  # p-value
    trend_min_r_squared: float = 0.4
    
    # Anomaly detection
    anomaly_z_score_threshold: float = 2.5
    anomaly_isolation_threshold: float = 0.1
    anomaly_context_window_days: int = 30
    
    # Cyclical pattern detection
    cyclical_autocorr_threshold: float = 0.6
    cyclical_min_cycles: int = 2
    cyclical_period_tolerance: float = 0.1
    
    # Correlation analysis
    correlation_threshold: float = 0.5
    correlation_significance: float = 0.05
    correlation_min_observations: int = 10
    
    # Sequence pattern matching
    sequence_match_threshold: float = 0.8
    sequence_time_tolerance_hours: int = 24
    sequence_min_events: int = 3


@dataclass
class TemporalQueryConfiguration:
    """Configuration for different types of temporal queries."""
    
    # Weight distribution for different retrieval methods
    vector_weight: float = 0.3
    graph_weight: float = 0.5
    pattern_weight: float = 0.2
    
    # Temporal decay settings
    decay_function: TemporalWeightDecayFunction = TemporalWeightDecayFunction.EXPONENTIAL
    decay_half_life_days: int = 30
    min_temporal_weight: float = 0.1
    max_temporal_weight: float = 2.0
    
    # Result fusion settings
    enable_temporal_boosting: bool = True
    temporal_boost_factor: float = 1.5
    enable_pattern_boosting: bool = True
    pattern_boost_factor: float = 1.3
    
    # Query complexity handling
    max_parallel_retrievals: int = 3
    retrieval_timeout_seconds: int = 30
    enable_progressive_timeout: bool = True


@dataclass
class PerformanceTuningConfig:
    """Performance tuning configuration for temporal queries."""
    
    # Caching settings
    enable_pattern_cache: bool = True
    pattern_cache_ttl_minutes: int = 60
    enable_trend_cache: bool = True
    trend_cache_ttl_minutes: int = 30
    
    # Query optimization
    max_cypher_complexity: int = 100
    enable_query_plan_cache: bool = True
    adaptive_batch_sizing: bool = True
    
    # Resource limits
    max_memory_mb: int = 1024
    max_cpu_threads: int = 4
    enable_resource_monitoring: bool = True
    
    # Performance profiles
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    
    # Timeout configurations
    vector_search_timeout: int = 15
    graph_traversal_timeout: int = 20
    pattern_analysis_timeout: int = 10
    
    def get_timeouts_for_profile(self) -> Dict[str, int]:
        """Get timeout values based on performance profile."""
        if self.performance_profile == PerformanceProfile.SPEED:
            return {
                "vector_search_timeout": 8,
                "graph_traversal_timeout": 10,
                "pattern_analysis_timeout": 5
            }
        elif self.performance_profile == PerformanceProfile.ACCURACY:
            return {
                "vector_search_timeout": 25,
                "graph_traversal_timeout": 35,
                "pattern_analysis_timeout": 20
            }
        elif self.performance_profile == PerformanceProfile.COMPREHENSIVE:
            return {
                "vector_search_timeout": 45,
                "graph_traversal_timeout": 60,
                "pattern_analysis_timeout": 30
            }
        else:  # BALANCED
            return {
                "vector_search_timeout": self.vector_search_timeout,
                "graph_traversal_timeout": self.graph_traversal_timeout,
                "pattern_analysis_timeout": self.pattern_analysis_timeout
            }


@dataclass
class HybridCypherConfig:
    """
    Comprehensive configuration for HybridCypher retriever.
    
    This class combines all configuration aspects for temporal analytics
    including time windows, pattern matching, performance tuning, and
    query-specific settings.
    """
    
    # Core retrieval settings
    vector_top_k: int = 15
    graph_top_k: int = 20
    final_top_k: int = 10
    min_vector_score: float = 0.1
    min_graph_score: float = 0.1
    
    # Temporal analysis settings
    enable_temporal_patterns: bool = True
    enable_vector_search: bool = True
    enable_graph_traversal: bool = True
    enable_pattern_detection: bool = True
    enable_anomaly_detection: bool = True
    
    # Time window configuration
    time_window_config: TimeWindowConfiguration = field(default_factory=TimeWindowConfiguration)
    
    # Pattern matching configuration
    pattern_thresholds: PatternMatchingThresholds = field(default_factory=PatternMatchingThresholds)
    
    # Temporal query configuration
    temporal_config: TemporalQueryConfiguration = field(default_factory=TemporalQueryConfiguration)
    
    # Performance tuning configuration
    performance_config: PerformanceTuningConfig = field(default_factory=PerformanceTuningConfig)
    
    # Query type specific configurations
    query_type_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default query type configurations."""
        if not self.query_type_configs:
            self._initialize_default_query_configs()
    
    def _initialize_default_query_configs(self):
        """Initialize default configurations for different query types."""
        self.query_type_configs = {
            "consumption": {
                "vector_weight": 0.2,
                "graph_weight": 0.6,
                "pattern_weight": 0.2,
                "decay_function": TemporalWeightDecayFunction.LINEAR,
                "enable_seasonal_analysis": True,
                "seasonal_periods": [12],  # Monthly seasonality
                "aggregation_preference": ["sum", "average"]
            },
            "emissions": {
                "vector_weight": 0.3,
                "graph_weight": 0.5,
                "pattern_weight": 0.2,
                "decay_function": TemporalWeightDecayFunction.EXPONENTIAL,
                "enable_seasonal_analysis": True,
                "seasonal_periods": [12, 4],  # Monthly and quarterly
                "aggregation_preference": ["sum", "maximum"]
            },
            "compliance": {
                "vector_weight": 0.4,
                "graph_weight": 0.4,
                "pattern_weight": 0.2,
                "decay_function": TemporalWeightDecayFunction.STEP,
                "enable_deadline_boosting": True,
                "deadline_boost_days": [30, 60, 90],
                "aggregation_preference": ["count", "latest"]
            },
            "risk": {
                "vector_weight": 0.3,
                "graph_weight": 0.4,
                "pattern_weight": 0.3,
                "decay_function": TemporalWeightDecayFunction.EXPONENTIAL,
                "enable_anomaly_detection": True,
                "anomaly_sensitivity": 2.0,
                "aggregation_preference": ["maximum", "count"]
            },
            "efficiency": {
                "vector_weight": 0.3,
                "graph_weight": 0.5,
                "pattern_weight": 0.2,
                "decay_function": TemporalWeightDecayFunction.LINEAR,
                "enable_trend_analysis": True,
                "trend_window_days": 90,
                "aggregation_preference": ["average", "rate_of_change"]
            }
        }
    
    def get_query_type_config(self, query_type: str) -> Dict[str, Any]:
        """Get configuration for a specific query type."""
        return self.query_type_configs.get(query_type.lower(), self.query_type_configs.get("general", {}))
    
    def get_temporal_decay_function(self, query_type: str) -> TemporalWeightDecayFunction:
        """Get the appropriate temporal decay function for a query type."""
        config = self.get_query_type_config(query_type)
        return config.get("decay_function", self.temporal_config.decay_function)
    
    def get_retrieval_weights(self, query_type: str) -> Tuple[float, float, float]:
        """Get vector, graph, and pattern weights for a query type."""
        config = self.get_query_type_config(query_type)
        return (
            config.get("vector_weight", self.temporal_config.vector_weight),
            config.get("graph_weight", self.temporal_config.graph_weight),
            config.get("pattern_weight", self.temporal_config.pattern_weight)
        )
    
    def should_enable_seasonal_analysis(self, query_type: str) -> bool:
        """Check if seasonal analysis should be enabled for a query type."""
        config = self.get_query_type_config(query_type)
        return config.get("enable_seasonal_analysis", False)
    
    def get_seasonal_periods(self, query_type: str) -> List[int]:
        """Get seasonal periods to analyze for a query type."""
        config = self.get_query_type_config(query_type)
        return config.get("seasonal_periods", [12])
    
    def get_aggregation_preferences(self, query_type: str) -> List[str]:
        """Get preferred aggregation methods for a query type."""
        config = self.get_query_type_config(query_type)
        return config.get("aggregation_preference", ["average"])
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in ["time_window", "pattern_thresholds", "temporal", "performance"]:
                # Update nested configurations
                config_map = {
                    "time_window": self.time_window_config,
                    "pattern_thresholds": self.pattern_thresholds,
                    "temporal": self.temporal_config,
                    "performance": self.performance_config
                }
                if key in config_map and isinstance(value, dict):
                    config_obj = config_map[key]
                    for nested_key, nested_value in value.items():
                        if hasattr(config_obj, nested_key):
                            setattr(config_obj, nested_key, nested_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        def convert_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {field: convert_dataclass(getattr(obj, field)) for field in obj.__dataclass_fields__}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert_dataclass(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dataclass(item) for item in obj]
            else:
                return obj
        
        return convert_dataclass(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridCypherConfig':
        """Create configuration from dictionary."""
        # Extract nested configurations
        time_window_data = config_dict.pop('time_window_config', {})
        pattern_thresholds_data = config_dict.pop('pattern_thresholds', {})
        temporal_data = config_dict.pop('temporal_config', {})
        performance_data = config_dict.pop('performance_config', {})
        
        # Create nested configuration objects
        time_window_config = TimeWindowConfiguration(**time_window_data)
        pattern_thresholds = PatternMatchingThresholds(**pattern_thresholds_data)
        temporal_config = TemporalQueryConfiguration(**temporal_data)
        performance_config = PerformanceTuningConfig(**performance_data)
        
        # Create main configuration
        return cls(
            time_window_config=time_window_config,
            pattern_thresholds=pattern_thresholds,
            temporal_config=temporal_config,
            performance_config=performance_config,
            **config_dict
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {filepath}: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'HybridCypherConfig':
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            logger.info(f"Configuration loaded from {filepath}")
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            raise
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate basic settings
        if self.vector_top_k <= 0:
            issues.append("vector_top_k must be positive")
        if self.graph_top_k <= 0:
            issues.append("graph_top_k must be positive")
        if self.final_top_k <= 0:
            issues.append("final_top_k must be positive")
        
        # Validate score thresholds
        if not 0 <= self.min_vector_score <= 1:
            issues.append("min_vector_score must be between 0 and 1")
        if not 0 <= self.min_graph_score <= 1:
            issues.append("min_graph_score must be between 0 and 1")
        
        # Validate time window configuration
        if self.time_window_config.max_window_days <= self.time_window_config.min_window_days:
            issues.append("max_window_days must be greater than min_window_days")
        
        # Validate pattern thresholds
        thresholds = self.pattern_thresholds
        if not 0 <= thresholds.seasonal_confidence_threshold <= 1:
            issues.append("seasonal_confidence_threshold must be between 0 and 1")
        if not 0 <= thresholds.trend_confidence_threshold <= 1:
            issues.append("trend_confidence_threshold must be between 0 and 1")
        
        # Validate temporal configuration
        weights = [self.temporal_config.vector_weight, self.temporal_config.graph_weight, self.temporal_config.pattern_weight]
        if abs(sum(weights) - 1.0) > 0.01:  # Allow small floating point errors
            issues.append("Retrieval weights should sum to 1.0")
        
        # Validate performance configuration
        if self.performance_config.max_memory_mb <= 0:
            issues.append("max_memory_mb must be positive")
        if self.performance_config.max_cpu_threads <= 0:
            issues.append("max_cpu_threads must be positive")
        
        return issues
    
    def optimize_for_query_volume(self, estimated_queries_per_hour: int) -> None:
        """Optimize configuration based on expected query volume."""
        if estimated_queries_per_hour > 1000:
            # High volume: optimize for speed
            self.performance_config.performance_profile = PerformanceProfile.SPEED
            self.performance_config.enable_pattern_cache = True
            self.performance_config.enable_trend_cache = True
            self.vector_top_k = min(self.vector_top_k, 10)
            self.graph_top_k = min(self.graph_top_k, 15)
            
        elif estimated_queries_per_hour > 100:
            # Medium volume: balanced approach
            self.performance_config.performance_profile = PerformanceProfile.BALANCED
            self.performance_config.enable_pattern_cache = True
            
        else:
            # Low volume: optimize for accuracy
            self.performance_config.performance_profile = PerformanceProfile.ACCURACY
            self.enable_pattern_detection = True
            self.enable_anomaly_detection = True
    
    def adapt_to_data_size(self, estimated_nodes: int, estimated_relationships: int) -> None:
        """Adapt configuration based on graph database size."""
        # Adjust limits based on data size
        if estimated_nodes > 1000000:  # Large database
            self.graph_top_k = min(self.graph_top_k, 15)
            self.performance_config.max_cypher_complexity = 50
            self.performance_config.graph_traversal_timeout = 30
            
        elif estimated_nodes > 100000:  # Medium database
            self.graph_top_k = min(self.graph_top_k, 25)
            self.performance_config.max_cypher_complexity = 75
            
        # Adjust memory allocation
        estimated_memory_mb = max(512, min(2048, estimated_nodes // 1000))
        self.performance_config.max_memory_mb = estimated_memory_mb


# Factory functions for common configurations
def create_development_config() -> HybridCypherConfig:
    """Create configuration optimized for development and testing."""
    config = HybridCypherConfig()
    config.performance_config.performance_profile = PerformanceProfile.SPEED
    config.vector_top_k = 5
    config.graph_top_k = 10
    config.final_top_k = 5
    config.performance_config.enable_pattern_cache = False  # Disable caching for testing
    config.performance_config.enable_trend_cache = False
    return config


def create_production_config() -> HybridCypherConfig:
    """Create configuration optimized for production use."""
    config = HybridCypherConfig()
    config.performance_config.performance_profile = PerformanceProfile.BALANCED
    config.performance_config.enable_pattern_cache = True
    config.performance_config.enable_trend_cache = True
    config.performance_config.enable_resource_monitoring = True
    return config


def create_analytics_config() -> HybridCypherConfig:
    """Create configuration optimized for deep analytics workloads."""
    config = HybridCypherConfig()
    config.performance_config.performance_profile = PerformanceProfile.COMPREHENSIVE
    config.enable_pattern_detection = True
    config.enable_anomaly_detection = True
    config.vector_top_k = 20
    config.graph_top_k = 30
    config.final_top_k = 15
    
    # Enhanced pattern detection
    config.pattern_thresholds.seasonal_confidence_threshold = 0.6
    config.pattern_thresholds.trend_confidence_threshold = 0.5
    config.pattern_thresholds.anomaly_z_score_threshold = 2.0
    
    return config


def create_compliance_config() -> HybridCypherConfig:
    """Create configuration optimized for compliance queries."""
    config = HybridCypherConfig()
    
    # Emphasize graph traversal for compliance relationships
    config.temporal_config.vector_weight = 0.2
    config.temporal_config.graph_weight = 0.6
    config.temporal_config.pattern_weight = 0.2
    
    # Use step decay for compliance deadlines
    config.temporal_config.decay_function = TemporalWeightDecayFunction.STEP
    
    # Extended time windows for compliance history
    config.time_window_config.compliance_window = 1095  # 3 years
    
    # Enable deadline boosting
    config.query_type_configs["compliance"]["enable_deadline_boosting"] = True
    
    return config


def create_risk_assessment_config() -> HybridCypherConfig:
    """Create configuration optimized for risk assessment queries."""
    config = HybridCypherConfig()
    
    # Balance all retrieval methods for comprehensive risk analysis
    config.temporal_config.vector_weight = 0.3
    config.temporal_config.graph_weight = 0.4
    config.temporal_config.pattern_weight = 0.3
    
    # Aggressive anomaly detection
    config.enable_anomaly_detection = True
    config.pattern_thresholds.anomaly_z_score_threshold = 2.0
    
    # Shorter time windows for recent risk factors
    config.time_window_config.risk_window = 60  # 2 months
    
    # Exponential decay to emphasize recent events
    config.temporal_config.decay_function = TemporalWeightDecayFunction.EXPONENTIAL
    config.temporal_config.decay_half_life_days = 14  # 2 weeks
    
    return config