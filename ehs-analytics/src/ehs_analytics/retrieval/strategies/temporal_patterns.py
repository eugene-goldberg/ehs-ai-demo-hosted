"""
Temporal patterns analysis for EHS Analytics.

This module provides comprehensive temporal pattern detection and analysis 
capabilities specifically designed for EHS data, including consumption trends,
compliance cycles, seasonal patterns, anomaly detection, and event sequences.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Types of temporal patterns that can be detected in EHS data."""
    
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    TRENDING = "trending"
    PERIODIC = "periodic"
    ANOMALOUS = "anomalous"
    THRESHOLD_BREACH = "threshold_breach"
    CORRELATION = "correlation"
    SEQUENCE = "sequence"


class TimeWindowType(str, Enum):
    """Types of time windows for analysis."""
    
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ROLLING = "rolling"
    CUSTOM = "custom"


class TemporalAggregationType(str, Enum):
    """Types of temporal aggregations."""
    
    SUM = "sum"
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    COUNT = "count"
    VARIANCE = "variance"
    PERCENTILE = "percentile"
    RATE_OF_CHANGE = "rate_of_change"


class SeasonalityType(str, Enum):
    """Types of seasonality patterns."""
    
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    BUSINESS_CYCLE = "business_cycle"
    COMPLIANCE_CYCLE = "compliance_cycle"


@dataclass
class PatternDetectionResult:
    """Result of pattern detection analysis."""
    
    pattern_type: PatternType
    confidence_score: float
    description: str
    metadata: Dict[str, Any]
    time_range: Tuple[datetime, datetime]
    affected_entities: List[str]
    pattern_strength: float = 0.0
    statistical_significance: Optional[float] = None


@dataclass
class TemporalTrend:
    """Represents a temporal trend in the data."""
    
    entity_id: str
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_magnitude: float
    time_period: Tuple[datetime, datetime]
    confidence_score: float
    r_squared: Optional[float] = None
    slope: Optional[float] = None
    seasonal_component: Optional[Dict[str, float]] = None


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""
    
    entity_id: str
    timestamp: datetime
    metric_name: str
    actual_value: float
    expected_value: float
    anomaly_score: float
    anomaly_type: str  # 'spike', 'drop', 'outlier', 'change_point'
    context: Dict[str, Any]


@dataclass
class ComplianceCycleInfo:
    """Information about compliance cycles and deadlines."""
    
    permit_id: str
    cycle_type: str  # 'renewal', 'reporting', 'inspection'
    cycle_length_days: int
    next_deadline: datetime
    last_completion: Optional[datetime]
    compliance_risk_score: float
    related_entities: List[str]


class EHSTemporalPatterns:
    """
    EHS-specific temporal patterns and analysis methods.
    
    This class contains domain knowledge about common temporal patterns
    in environmental, health, and safety data.
    """
    
    # EHS-specific seasonal patterns
    SEASONAL_PATTERNS = {
        "energy_consumption": {
            "winter_months": [12, 1, 2],
            "summer_months": [6, 7, 8],
            "peak_factor": 1.3,
            "description": "Energy consumption typically peaks in winter and summer"
        },
        "water_consumption": {
            "dry_months": [6, 7, 8, 9],
            "wet_months": [12, 1, 2, 3],
            "peak_factor": 1.2,
            "description": "Water consumption often peaks during dry seasons"
        },
        "emissions": {
            "high_activity_months": [3, 4, 5, 9, 10, 11],
            "low_activity_months": [12, 1, 2, 6, 7, 8],
            "peak_factor": 1.15,
            "description": "Emissions may vary with production cycles and weather"
        },
        "incidents": {
            "high_risk_months": [6, 7, 8],  # Summer heat stress
            "winter_risk_months": [12, 1, 2],  # Winter weather hazards
            "peak_factor": 1.4,
            "description": "Safety incidents often correlate with weather extremes"
        }
    }
    
    # Compliance cycle patterns
    COMPLIANCE_CYCLES = {
        "air_emissions_permit": {
            "renewal_cycle_years": 5,
            "reporting_cycle_months": 12,
            "inspection_cycle_months": 24,
            "lead_time_months": 6
        },
        "water_discharge_permit": {
            "renewal_cycle_years": 5,
            "reporting_cycle_months": 3,
            "inspection_cycle_months": 12,
            "lead_time_months": 4
        },
        "waste_management_permit": {
            "renewal_cycle_years": 3,
            "reporting_cycle_months": 12,
            "inspection_cycle_months": 18,
            "lead_time_months": 3
        },
        "safety_certification": {
            "renewal_cycle_years": 3,
            "reporting_cycle_months": 6,
            "inspection_cycle_months": 12,
            "lead_time_months": 2
        }
    }
    
    # Anomaly thresholds by metric type
    ANOMALY_THRESHOLDS = {
        "consumption": {
            "spike_threshold": 2.5,  # Standard deviations
            "drop_threshold": -2.0,
            "change_point_threshold": 1.5
        },
        "emissions": {
            "spike_threshold": 3.0,
            "drop_threshold": -1.5,
            "change_point_threshold": 2.0
        },
        "incidents": {
            "spike_threshold": 2.0,
            "clustering_threshold": 3,  # Number of incidents in time window
            "severity_threshold": 0.8
        },
        "efficiency": {
            "degradation_threshold": -1.8,
            "improvement_threshold": 2.0,
            "stability_threshold": 0.5
        }
    }
    
    # Event sequence patterns
    SEQUENCE_PATTERNS = {
        "incident_escalation": {
            "pattern": ["minor_incident", "investigation", "corrective_action", "major_incident"],
            "max_time_window_days": 90,
            "confidence_threshold": 0.7
        },
        "equipment_degradation": {
            "pattern": ["efficiency_drop", "maintenance_alert", "increased_emissions", "failure"],
            "max_time_window_days": 180,
            "confidence_threshold": 0.6
        },
        "compliance_violation": {
            "pattern": ["warning", "inspection", "violation", "enforcement"],
            "max_time_window_days": 365,
            "confidence_threshold": 0.8
        }
    }


class TemporalPatternAnalyzer:
    """
    Advanced temporal pattern analyzer for EHS data.
    
    This class provides sophisticated pattern detection, trend analysis,
    and anomaly detection capabilities specifically designed for 
    environmental, health, and safety analytics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the temporal pattern analyzer.
        
        Args:
            config: Configuration dictionary for pattern analysis
        """
        self.config = config or {}
        self.ehs_patterns = EHSTemporalPatterns()
        
        # Analysis parameters
        self.min_data_points = self.config.get("min_data_points", 10)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.seasonal_period = self.config.get("seasonal_period", 12)  # months
        self.anomaly_sensitivity = self.config.get("anomaly_sensitivity", 2.0)
        
        # Cached analysis results
        self._pattern_cache = {}
        self._trend_cache = {}
        
        logger.info("Initialized EHS Temporal Pattern Analyzer")
    
    async def initialize(self) -> None:
        """Initialize the pattern analyzer."""
        # Initialize any required resources
        pass
    
    async def analyze_query_requirements(
        self,
        query: str,
        temporal_context: 'TemporalContext',
        temporal_query_type: 'TemporalQueryType'
    ) -> Dict[str, Any]:
        """
        Analyze query to determine required pattern analysis.
        
        Args:
            query: Natural language query
            temporal_context: Temporal context information
            temporal_query_type: Type of temporal query
            
        Returns:
            Dictionary containing analysis requirements
        """
        requirements = {
            "pattern_types": [],
            "analysis_methods": [],
            "aggregation_needs": [],
            "temporal_resolution": "monthly",
            "comparison_baselines": []
        }
        
        query_lower = query.lower()
        
        # Determine required pattern types based on query
        if "seasonal" in query_lower or "monthly" in query_lower or "quarterly" in query_lower:
            requirements["pattern_types"].append(PatternType.SEASONAL)
            requirements["analysis_methods"].append("seasonal_decomposition")
        
        if "trend" in query_lower or "increase" in query_lower or "decrease" in query_lower:
            requirements["pattern_types"].append(PatternType.TRENDING)
            requirements["analysis_methods"].append("trend_analysis")
        
        if "anomaly" in query_lower or "unusual" in query_lower or "spike" in query_lower:
            requirements["pattern_types"].append(PatternType.ANOMALOUS)
            requirements["analysis_methods"].append("anomaly_detection")
        
        if "cycle" in query_lower or "periodic" in query_lower or "regular" in query_lower:
            requirements["pattern_types"].append(PatternType.CYCLICAL)
            requirements["analysis_methods"].append("cyclical_analysis")
        
        if "before" in query_lower or "after" in query_lower or "sequence" in query_lower:
            requirements["pattern_types"].append(PatternType.SEQUENCE)
            requirements["analysis_methods"].append("sequence_analysis")
        
        # Determine temporal resolution
        if "daily" in query_lower:
            requirements["temporal_resolution"] = "daily"
        elif "weekly" in query_lower:
            requirements["temporal_resolution"] = "weekly"
        elif "yearly" in query_lower or "annual" in query_lower:
            requirements["temporal_resolution"] = "yearly"
        
        # Determine aggregation needs
        if "total" in query_lower or "sum" in query_lower:
            requirements["aggregation_needs"].append(TemporalAggregationType.SUM)
        if "average" in query_lower or "mean" in query_lower:
            requirements["aggregation_needs"].append(TemporalAggregationType.AVERAGE)
        if "peak" in query_lower or "maximum" in query_lower:
            requirements["aggregation_needs"].append(TemporalAggregationType.MAXIMUM)
        
        return requirements
    
    async def detect_patterns(
        self,
        query: str,
        temporal_context: 'TemporalContext',
        query_type: 'QueryType'
    ) -> List[PatternDetectionResult]:
        """
        Detect temporal patterns relevant to the query.
        
        Args:
            query: Natural language query
            temporal_context: Temporal context information
            query_type: Type of EHS query
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Detect seasonal patterns
        seasonal_patterns = await self._detect_seasonal_patterns(
            query, temporal_context, query_type
        )
        patterns.extend(seasonal_patterns)
        
        # Detect cyclical patterns
        cyclical_patterns = await self._detect_cyclical_patterns(
            query, temporal_context, query_type
        )
        patterns.extend(cyclical_patterns)
        
        # Detect compliance cycle patterns
        compliance_patterns = await self._detect_compliance_cycles(
            query, temporal_context, query_type
        )
        patterns.extend(compliance_patterns)
        
        return patterns
    
    async def _detect_seasonal_patterns(
        self,
        query: str,
        temporal_context: 'TemporalContext',
        query_type: 'QueryType'
    ) -> List[PatternDetectionResult]:
        """Detect seasonal patterns in EHS data."""
        patterns = []
        
        # Determine relevant seasonal patterns based on query type
        relevant_patterns = []
        if "consumption" in query.lower() or query_type.value == "consumption":
            relevant_patterns.extend(["energy_consumption", "water_consumption"])
        if "emission" in query.lower() or query_type.value == "emissions":
            relevant_patterns.append("emissions")
        if "incident" in query.lower() or "safety" in query.lower():
            relevant_patterns.append("incidents")
        
        # Create pattern detection results for relevant patterns
        for pattern_name in relevant_patterns:
            if pattern_name in self.ehs_patterns.SEASONAL_PATTERNS:
                pattern_info = self.ehs_patterns.SEASONAL_PATTERNS[pattern_name]
                
                pattern_result = PatternDetectionResult(
                    pattern_type=PatternType.SEASONAL,
                    confidence_score=0.8,  # High confidence for known patterns
                    description=pattern_info["description"],
                    metadata={
                        "pattern_name": pattern_name,
                        "peak_factor": pattern_info["peak_factor"],
                        "seasonal_months": pattern_info.get("winter_months", pattern_info.get("high_activity_months", []))
                    },
                    time_range=temporal_context.time_range,
                    affected_entities=[],
                    pattern_strength=pattern_info["peak_factor"] - 1.0
                )
                patterns.append(pattern_result)
        
        return patterns
    
    async def _detect_cyclical_patterns(
        self,
        query: str,
        temporal_context: 'TemporalContext',
        query_type: 'QueryType'
    ) -> List[PatternDetectionResult]:
        """Detect cyclical patterns in EHS data."""
        patterns = []
        
        # Business cycle patterns (quarterly reporting, etc.)
        if "quarter" in query.lower() or "annual" in query.lower():
            pattern_result = PatternDetectionResult(
                pattern_type=PatternType.CYCLICAL,
                confidence_score=0.7,
                description="Business reporting cycle pattern detected",
                metadata={
                    "cycle_type": "business_reporting",
                    "cycle_length_months": 3 if "quarter" in query.lower() else 12,
                    "expected_peaks": [3, 6, 9, 12] if "quarter" in query.lower() else [12]
                },
                time_range=temporal_context.time_range,
                affected_entities=[],
                pattern_strength=0.6
            )
            patterns.append(pattern_result)
        
        return patterns
    
    async def _detect_compliance_cycles(
        self,
        query: str,
        temporal_context: 'TemporalContext',
        query_type: 'QueryType'
    ) -> List[PatternDetectionResult]:
        """Detect compliance cycle patterns."""
        patterns = []
        
        query_lower = query.lower()
        
        # Check for permit-related patterns
        permit_keywords = ["permit", "license", "certification", "compliance"]
        if any(keyword in query_lower for keyword in permit_keywords):
            
            # Determine permit type
            permit_type = None
            if "air" in query_lower or "emission" in query_lower:
                permit_type = "air_emissions_permit"
            elif "water" in query_lower or "discharge" in query_lower:
                permit_type = "water_discharge_permit"
            elif "waste" in query_lower:
                permit_type = "waste_management_permit"
            elif "safety" in query_lower:
                permit_type = "safety_certification"
            
            if permit_type and permit_type in self.ehs_patterns.COMPLIANCE_CYCLES:
                cycle_info = self.ehs_patterns.COMPLIANCE_CYCLES[permit_type]
                
                pattern_result = PatternDetectionResult(
                    pattern_type=PatternType.PERIODIC,
                    confidence_score=0.9,
                    description=f"Compliance cycle pattern for {permit_type.replace('_', ' ')}",
                    metadata={
                        "permit_type": permit_type,
                        "renewal_cycle_years": cycle_info["renewal_cycle_years"],
                        "reporting_cycle_months": cycle_info["reporting_cycle_months"],
                        "inspection_cycle_months": cycle_info["inspection_cycle_months"],
                        "lead_time_months": cycle_info["lead_time_months"]
                    },
                    time_range=temporal_context.time_range,
                    affected_entities=[],
                    pattern_strength=0.8
                )
                patterns.append(pattern_result)
        
        return patterns
    
    async def analyze_trends(
        self,
        data: List[Dict[str, Any]],
        temporal_context: 'TemporalContext'
    ) -> List[TemporalTrend]:
        """
        Analyze temporal trends in the data.
        
        Args:
            data: Time series data
            temporal_context: Temporal context
            
        Returns:
            List of detected trends
        """
        trends = []
        
        if len(data) < self.min_data_points:
            return trends
        
        # Group data by entity
        entity_data = defaultdict(list)
        for item in data:
            entity_id = item.get('facility', item.get('equipment_name', 'unknown'))
            entity_data[entity_id].append(item)
        
        # Analyze trends for each entity
        for entity_id, entity_items in entity_data.items():
            if len(entity_items) < self.min_data_points:
                continue
            
            # Sort by timestamp
            entity_items.sort(key=lambda x: x.get('timestamp', x.get('time_period', '')))
            
            # Extract values for different metrics
            metrics = {}
            for item in entity_items:
                for key, value in item.items():
                    if isinstance(value, (int, float)) and key not in ['score', 'temporal_weight']:
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(value)
            
            # Analyze trend for each metric
            for metric_name, values in metrics.items():
                if len(values) >= self.min_data_points:
                    trend = await self._calculate_trend(
                        entity_id, metric_name, values, entity_items, temporal_context
                    )
                    if trend:
                        trends.append(trend)
        
        return trends
    
    async def _calculate_trend(
        self,
        entity_id: str,
        metric_name: str,
        values: List[float],
        items: List[Dict[str, Any]],
        temporal_context: 'TemporalContext'
    ) -> Optional[TemporalTrend]:
        """Calculate trend for a specific metric."""
        try:
            # Simple linear regression
            n = len(values)
            x = np.arange(n)
            y = np.array(values)
            
            # Calculate slope and R-squared
            slope, intercept = np.polyfit(x, y, 1)
            correlation = np.corrcoef(x, y)[0, 1]
            r_squared = correlation ** 2
            
            # Determine trend direction and magnitude
            if abs(slope) < np.std(values) * 0.1:  # Stable if slope is small relative to variance
                trend_direction = "stable"
                trend_magnitude = 0.0
            elif slope > 0:
                trend_direction = "increasing"
                trend_magnitude = slope
            else:
                trend_direction = "decreasing"
                trend_magnitude = abs(slope)
            
            # Calculate confidence based on R-squared and data points
            confidence_score = min(1.0, r_squared + (n / 50.0))  # Boost confidence with more data points
            
            if confidence_score >= self.confidence_threshold:
                return TemporalTrend(
                    entity_id=entity_id,
                    metric_name=metric_name,
                    trend_direction=trend_direction,
                    trend_magnitude=trend_magnitude,
                    time_period=temporal_context.time_range,
                    confidence_score=confidence_score,
                    r_squared=r_squared,
                    slope=slope
                )
            
        except Exception as e:
            logger.error(f"Error calculating trend for {entity_id}.{metric_name}: {e}")
        
        return None
    
    async def detect_anomalies(
        self,
        data: List[Dict[str, Any]],
        temporal_context: 'TemporalContext'
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in temporal data.
        
        Args:
            data: Time series data
            temporal_context: Temporal context
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(data) < self.min_data_points:
            return anomalies
        
        # Group data by entity and metric
        entity_metrics = defaultdict(lambda: defaultdict(list))
        for item in data:
            entity_id = item.get('facility', item.get('equipment_name', 'unknown'))
            for key, value in item.items():
                if isinstance(value, (int, float)) and key not in ['score', 'temporal_weight']:
                    entity_metrics[entity_id][key].append({
                        'value': value,
                        'timestamp': item.get('timestamp', item.get('time_period')),
                        'item': item
                    })
        
        # Detect anomalies for each entity-metric combination
        for entity_id, metrics in entity_metrics.items():
            for metric_name, metric_data in metrics.items():
                if len(metric_data) >= self.min_data_points:
                    entity_anomalies = await self._detect_metric_anomalies(
                        entity_id, metric_name, metric_data, temporal_context
                    )
                    anomalies.extend(entity_anomalies)
        
        return anomalies
    
    async def _detect_metric_anomalies(
        self,
        entity_id: str,
        metric_name: str,
        metric_data: List[Dict[str, Any]],
        temporal_context: 'TemporalContext'
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies for a specific entity-metric combination."""
        anomalies = []
        
        try:
            # Extract values and timestamps
            values = [item['value'] for item in metric_data]
            timestamps = [item['timestamp'] for item in metric_data]
            
            # Calculate statistical measures
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            if std_value == 0:
                return anomalies  # No variance, no anomalies
            
            # Determine appropriate thresholds based on metric type
            thresholds = self._get_anomaly_thresholds(metric_name)
            
            # Detect different types of anomalies
            for i, item in enumerate(metric_data):
                value = item['value']
                timestamp = item['timestamp']
                
                # Z-score based anomaly detection
                z_score = abs(value - mean_value) / std_value
                
                # Spike detection
                if z_score > thresholds["spike_threshold"]:
                    anomaly_type = "spike" if value > mean_value else "drop"
                    
                    anomaly = AnomalyDetectionResult(
                        entity_id=entity_id,
                        timestamp=self._parse_timestamp(timestamp),
                        metric_name=metric_name,
                        actual_value=value,
                        expected_value=mean_value,
                        anomaly_score=z_score,
                        anomaly_type=anomaly_type,
                        context={
                            "mean": mean_value,
                            "std": std_value,
                            "threshold": thresholds["spike_threshold"],
                            "percentile": self._calculate_percentile(value, values)
                        }
                    )
                    anomalies.append(anomaly)
            
            # Change point detection (simplified)
            change_points = await self._detect_change_points(values, thresholds["change_point_threshold"])
            for cp_index in change_points:
                if cp_index < len(metric_data):
                    item = metric_data[cp_index]
                    anomaly = AnomalyDetectionResult(
                        entity_id=entity_id,
                        timestamp=self._parse_timestamp(item['timestamp']),
                        metric_name=metric_name,
                        actual_value=item['value'],
                        expected_value=mean_value,
                        anomaly_score=thresholds["change_point_threshold"],
                        anomaly_type="change_point",
                        context={
                            "change_point_index": cp_index,
                            "before_mean": np.mean(values[:cp_index]) if cp_index > 0 else mean_value,
                            "after_mean": np.mean(values[cp_index:]) if cp_index < len(values) else mean_value
                        }
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {entity_id}.{metric_name}: {e}")
        
        return anomalies
    
    def _get_anomaly_thresholds(self, metric_name: str) -> Dict[str, float]:
        """Get anomaly detection thresholds for a specific metric."""
        metric_lower = metric_name.lower()
        
        # Determine metric category
        if "consumption" in metric_lower or "amount" in metric_lower:
            return self.ehs_patterns.ANOMALY_THRESHOLDS["consumption"]
        elif "emission" in metric_lower:
            return self.ehs_patterns.ANOMALY_THRESHOLDS["emissions"]
        elif "incident" in metric_lower or "severity" in metric_lower:
            return self.ehs_patterns.ANOMALY_THRESHOLDS["incidents"]
        elif "efficiency" in metric_lower:
            return self.ehs_patterns.ANOMALY_THRESHOLDS["efficiency"]
        else:
            # Default thresholds
            return {
                "spike_threshold": 2.5,
                "drop_threshold": -2.0,
                "change_point_threshold": 1.5
            }
    
    async def _detect_change_points(self, values: List[float], threshold: float) -> List[int]:
        """Detect change points in time series data."""
        change_points = []
        
        if len(values) < 6:  # Need minimum data for change point detection
            return change_points
        
        # Simple change point detection using moving averages
        window_size = max(3, len(values) // 5)
        
        for i in range(window_size, len(values) - window_size):
            before_window = values[i-window_size:i]
            after_window = values[i:i+window_size]
            
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            
            # Calculate change magnitude relative to overall standard deviation
            overall_std = np.std(values)
            if overall_std > 0:
                change_magnitude = abs(after_mean - before_mean) / overall_std
                
                if change_magnitude > threshold:
                    change_points.append(i)
        
        return change_points
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank of a value."""
        try:
            return (sum(1 for v in values if v <= value) / len(values)) * 100
        except:
            return 50.0  # Default to median
    
    def _parse_timestamp(self, timestamp: Union[str, datetime, date]) -> datetime:
        """Parse timestamp to datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, date):
            return datetime.combine(timestamp, datetime.min.time())
        elif isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return datetime.now()
        else:
            return datetime.now()
    
    async def extract_time_range_from_query(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract time range from natural language query."""
        query_lower = query.lower()
        current_time = datetime.now()
        
        # Look for specific time periods
        if "last month" in query_lower:
            start_date = current_time.replace(day=1) - timedelta(days=32)
            start_date = start_date.replace(day=1)
            end_date = current_time.replace(day=1) - timedelta(days=1)
            return (start_date, end_date)
        
        elif "last 6 months" in query_lower or "past 6 months" in query_lower:
            end_date = current_time
            start_date = end_date - timedelta(days=180)
            return (start_date, end_date)
        
        elif "last year" in query_lower or "past year" in query_lower:
            end_date = current_time
            start_date = end_date - timedelta(days=365)
            return (start_date, end_date)
        
        elif "this year" in query_lower:
            start_date = current_time.replace(month=1, day=1)
            end_date = current_time
            return (start_date, end_date)
        
        # Look for specific date patterns using regex
        date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{4})'   # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            if matches:
                try:
                    if pattern.startswith(r'(\d{4})'):  # YYYY-MM-DD
                        year, month, day = map(int, matches[0])
                    else:  # MM/DD/YYYY or MM-DD-YYYY
                        month, day, year = map(int, matches[0])
                    
                    extracted_date = datetime(year, month, day)
                    # Return a range around the extracted date
                    start_date = extracted_date - timedelta(days=30)
                    end_date = extracted_date + timedelta(days=30)
                    return (start_date, end_date)
                except ValueError:
                    continue
        
        return None
    
    async def infer_granularity_from_query(
        self, 
        query: str, 
        time_range: Tuple[datetime, datetime]
    ) -> str:
        """Infer appropriate time granularity from query and time range."""
        query_lower = query.lower()
        
        # Explicit granularity mentions
        if "daily" in query_lower or "day" in query_lower:
            return "daily"
        elif "weekly" in query_lower or "week" in query_lower:
            return "weekly"
        elif "monthly" in query_lower or "month" in query_lower:
            return "monthly"
        elif "quarterly" in query_lower or "quarter" in query_lower:
            return "quarterly"
        elif "yearly" in query_lower or "annual" in query_lower:
            return "yearly"
        
        # Infer from time range duration
        duration_days = (time_range[1] - time_range[0]).days
        
        if duration_days <= 30:
            return "daily"
        elif duration_days <= 120:
            return "weekly"
        elif duration_days <= 730:  # ~2 years
            return "monthly"
        else:
            return "quarterly"
    
    async def infer_pattern_types_from_query(
        self, 
        query: str, 
        query_type: 'QueryType'
    ) -> List[PatternType]:
        """Infer relevant pattern types from query and query type."""
        pattern_types = []
        query_lower = query.lower()
        
        # Seasonal patterns
        if any(keyword in query_lower for keyword in ["seasonal", "monthly", "quarterly", "winter", "summer"]):
            pattern_types.append(PatternType.SEASONAL)
        
        # Trending patterns
        if any(keyword in query_lower for keyword in ["trend", "increase", "decrease", "growth", "decline"]):
            pattern_types.append(PatternType.TRENDING)
        
        # Anomaly patterns
        if any(keyword in query_lower for keyword in ["anomaly", "unusual", "spike", "outlier"]):
            pattern_types.append(PatternType.ANOMALOUS)
        
        # Cyclical patterns
        if any(keyword in query_lower for keyword in ["cycle", "periodic", "recurring", "regular"]):
            pattern_types.append(PatternType.CYCLICAL)
        
        # Sequence patterns
        if any(keyword in query_lower for keyword in ["before", "after", "sequence", "order"]):
            pattern_types.append(PatternType.SEQUENCE)
        
        # Default patterns based on query type
        if not pattern_types:
            if query_type.value in ["consumption", "emissions"]:
                pattern_types.extend([PatternType.SEASONAL, PatternType.TRENDING])
            elif query_type.value == "compliance":
                pattern_types.extend([PatternType.CYCLICAL, PatternType.THRESHOLD_BREACH])
            elif query_type.value == "risk":
                pattern_types.extend([PatternType.ANOMALOUS, PatternType.SEQUENCE])
            else:
                pattern_types.append(PatternType.TRENDING)
        
        return pattern_types
    
    async def infer_aggregation_types_from_query(
        self, 
        query: str, 
        query_type: 'QueryType'
    ) -> List[TemporalAggregationType]:
        """Infer required aggregation types from query."""
        aggregation_types = []
        query_lower = query.lower()
        
        # Explicit aggregation mentions
        if any(keyword in query_lower for keyword in ["total", "sum"]):
            aggregation_types.append(TemporalAggregationType.SUM)
        if any(keyword in query_lower for keyword in ["average", "mean"]):
            aggregation_types.append(TemporalAggregationType.AVERAGE)
        if any(keyword in query_lower for keyword in ["maximum", "peak", "highest"]):
            aggregation_types.append(TemporalAggregationType.MAXIMUM)
        if any(keyword in query_lower for keyword in ["minimum", "lowest"]):
            aggregation_types.append(TemporalAggregationType.MINIMUM)
        if any(keyword in query_lower for keyword in ["count", "number"]):
            aggregation_types.append(TemporalAggregationType.COUNT)
        if any(keyword in query_lower for keyword in ["rate", "change"]):
            aggregation_types.append(TemporalAggregationType.RATE_OF_CHANGE)
        
        # Default aggregations based on query type
        if not aggregation_types:
            if query_type.value in ["consumption", "emissions"]:
                aggregation_types.extend([TemporalAggregationType.SUM, TemporalAggregationType.AVERAGE])
            elif query_type.value == "efficiency":
                aggregation_types.extend([TemporalAggregationType.AVERAGE, TemporalAggregationType.RATE_OF_CHANGE])
            elif query_type.value in ["compliance", "risk"]:
                aggregation_types.extend([TemporalAggregationType.COUNT, TemporalAggregationType.MAXIMUM])
            else:
                aggregation_types.append(TemporalAggregationType.AVERAGE)
        
        return aggregation_types
    
    async def parse_sequence_requirements(self, query: str) -> Dict[str, Any]:
        """Parse sequence analysis requirements from query."""
        requirements = {
            "sequence_type": "unknown",
            "time_window_days": 90,
            "entities": [],
            "events": []
        }
        
        query_lower = query.lower()
        
        # Parse sequence patterns
        if "incident" in query_lower and ("before" in query_lower or "after" in query_lower):
            requirements["sequence_type"] = "incident_sequence"
            if "emission" in query_lower:
                requirements["events"] = ["incident", "emission_change"]
            elif "permit" in query_lower:
                requirements["events"] = ["incident", "permit_status"]
        
        elif "equipment" in query_lower and "failure" in query_lower:
            requirements["sequence_type"] = "equipment_degradation"
            requirements["events"] = ["efficiency_drop", "maintenance", "failure"]
            requirements["time_window_days"] = 180
        
        elif "permit" in query_lower and ("expir" in query_lower or "renew" in query_lower):
            requirements["sequence_type"] = "compliance_timeline"
            requirements["events"] = ["permit_warning", "renewal_deadline", "compliance_status"]
            requirements["time_window_days"] = 365
        
        # Extract time window from query
        if "6 months" in query_lower:
            requirements["time_window_days"] = 180
        elif "year" in query_lower:
            requirements["time_window_days"] = 365
        elif "month" in query_lower:
            requirements["time_window_days"] = 30
        
        return requirements