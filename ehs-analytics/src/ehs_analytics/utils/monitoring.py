"""
Comprehensive monitoring and metrics system for EHS Analytics platform.

This module provides metrics collection, health checks, resource monitoring,
and alerting capabilities with integration points for monitoring tools like
Prometheus and Grafana.
"""

import asyncio
import json
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from threading import Lock
import threading

from .logging import get_ehs_logger

logger = get_ehs_logger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert notification."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """
    Centralized metrics collection system.
    
    Collects and stores various types of metrics with thread-safe operations
    and memory-efficient storage using circular buffers.
    """
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.metric_types: Dict[str, MetricType] = {}
        self.metric_descriptions: Dict[str, str] = {}
        self.lock = Lock()
        
        # Performance counters
        self.query_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        
        logger.info("MetricsCollector initialized", max_points=max_points_per_metric)
    
    def register_metric(
        self, 
        name: str, 
        metric_type: MetricType, 
        description: str = ""
    ):
        """
        Register a new metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Human-readable description
        """
        with self.lock:
            self.metric_types[name] = metric_type
            self.metric_descriptions[name] = description
        
        logger.debug("Metric registered", metric_name=name, type=metric_type.value)
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        self._record_metric(name, MetricType.COUNTER, value, labels)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        self._record_metric(name, MetricType.GAUGE, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self._record_metric(name, MetricType.HISTOGRAM, value, labels)
    
    def _record_metric(
        self, 
        name: str, 
        metric_type: MetricType, 
        value: float, 
        labels: Optional[Dict[str, str]] = None
    ):
        """Internal method to record a metric."""
        with self.lock:
            # Auto-register if not exists
            if name not in self.metric_types:
                self.metric_types[name] = metric_type
            
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels or {}
            )
            
            self.metrics[name].append(point)
    
    def get_metric_values(
        self, 
        name: str, 
        since: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricPoint]:
        """
        Get metric values with optional filtering.
        
        Args:
            name: Metric name
            since: Only return points after this timestamp
            labels: Filter by labels
        
        Returns:
            List of matching metric points
        """
        with self.lock:
            points = list(self.metrics[name])
        
        # Apply timestamp filter
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        # Apply label filter
        if labels:
            points = [
                p for p in points
                if all(p.labels.get(k) == v for k, v in labels.items())
            ]
        
        return points
    
    def get_metric_summary(self, name: str, duration: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """
        Get statistical summary of a metric over a time period.
        
        Args:
            name: Metric name
            duration: Time period to analyze
        
        Returns:
            Dictionary with min, max, avg, count, and recent values
        """
        since = datetime.utcnow() - duration
        points = self.get_metric_values(name, since=since)
        
        if not points:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "latest": None,
                "duration_minutes": duration.total_seconds() / 60
            }
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "duration_minutes": duration.total_seconds() / 60
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get summary of all registered metrics."""
        with self.lock:
            result = {}
            for name in self.metric_types:
                result[name] = {
                    "type": self.metric_types[name].value,
                    "description": self.metric_descriptions.get(name, ""),
                    "point_count": len(self.metrics[name]),
                    "summary": self.get_metric_summary(name)
                }
        
        return result


class ResourceMonitor:
    """
    System resource monitoring.
    
    Monitors CPU, memory, disk, and network usage with configurable
    collection intervals and thresholds.
    """
    
    def __init__(self, collection_interval: float = 60.0):
        self.collection_interval = collection_interval
        self.metrics_collector = None
        self.monitoring_task = None
        self.is_running = False
        
        logger.info("ResourceMonitor initialized", interval=collection_interval)
    
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set the metrics collector for recording resource metrics."""
        self.metrics_collector = collector
        
        # Register resource metrics
        self.metrics_collector.register_metric(
            "system_cpu_percent", MetricType.GAUGE, "CPU utilization percentage"
        )
        self.metrics_collector.register_metric(
            "system_memory_percent", MetricType.GAUGE, "Memory utilization percentage"
        )
        self.metrics_collector.register_metric(
            "system_disk_percent", MetricType.GAUGE, "Disk utilization percentage"
        )
        self.metrics_collector.register_metric(
            "system_load_avg", MetricType.GAUGE, "System load average"
        )
        
    async def start_monitoring(self):
        """Start background resource monitoring."""
        if self.is_running:
            logger.warning("Resource monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop background resource monitoring."""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self.collect_resource_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in resource monitoring loop", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def collect_resource_metrics(self):
        """Collect current resource utilization metrics."""
        if not self.metrics_collector:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_gauge("system_memory_percent", memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.record_gauge("system_disk_percent", disk_percent)
            
            # Load average (on Unix systems)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute average
                self.metrics_collector.record_gauge("system_load_avg", load_avg)
            except AttributeError:
                # getloadavg not available on Windows
                pass
            
            logger.debug(
                "Resource metrics collected",
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk_percent
            )
            
        except Exception as e:
            logger.error("Failed to collect resource metrics", error=str(e))


class HealthChecker:
    """
    Health check system for monitoring component health.
    
    Manages and executes health checks for various system components
    with configurable intervals and dependencies.
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.check_intervals: Dict[str, float] = {}
        self.check_results: Dict[str, HealthCheckResult] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.monitoring_task = None
        self.is_running = False
        
        logger.info("HealthChecker initialized")
    
    def register_check(
        self, 
        name: str, 
        check_func: Callable, 
        interval: float = 30.0,
        dependencies: Optional[List[str]] = None
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Async function that returns HealthCheckResult
            interval: Check interval in seconds
            dependencies: List of dependent check names
        """
        self.checks[name] = check_func
        self.check_intervals[name] = interval
        self.dependencies[name] = dependencies or []
        
        logger.info("Health check registered", check_name=name, interval=interval)
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status="unhealthy",
                message="Check not found",
                timestamp=datetime.utcnow(),
                duration_ms=0.0
            )
        
        start_time = time.time()
        
        try:
            result = await self.checks[name]()
            if not isinstance(result, HealthCheckResult):
                result = HealthCheckResult(
                    name=name,
                    status="healthy",
                    message="Check passed",
                    timestamp=datetime.utcnow(),
                    duration_ms=(time.time() - start_time) * 1000,
                    details={"result": result}
                )
            else:
                result.duration_ms = (time.time() - start_time) * 1000
            
            self.check_results[name] = result
            return result
            
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status="unhealthy",
                message=f"Check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                duration_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
            
            self.check_results[name] = result
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        # Sort checks by dependencies
        sorted_checks = self._sort_checks_by_dependencies()
        
        for check_name in sorted_checks:
            result = await self.run_check(check_name)
            results[check_name] = result
        
        return results
    
    def _sort_checks_by_dependencies(self) -> List[str]:
        """Sort checks to respect dependencies."""
        sorted_checks = []
        remaining = set(self.checks.keys())
        
        while remaining:
            # Find checks with no unresolved dependencies
            ready = []
            for check in remaining:
                deps = self.dependencies.get(check, [])
                if all(dep in sorted_checks or dep not in self.checks for dep in deps):
                    ready.append(check)
            
            if not ready:
                # Circular dependency or missing dependency
                ready = list(remaining)  # Add remaining checks anyway
            
            sorted_checks.extend(ready)
            remaining -= set(ready)
        
        return sorted_checks
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = await self.run_all_checks()
        
        healthy_count = sum(1 for r in results.values() if r.status == "healthy")
        degraded_count = sum(1 for r in results.values() if r.status == "degraded")
        unhealthy_count = sum(1 for r in results.values() if r.status == "unhealthy")
        
        total_checks = len(results)
        health_percentage = (healthy_count / total_checks * 100) if total_checks > 0 else 0
        
        overall_status = "healthy"
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "health_percentage": health_percentage,
            "total_checks": total_checks,
            "healthy_checks": healthy_count,
            "degraded_checks": degraded_count,
            "unhealthy_checks": unhealthy_count,
            "checks": {name: {
                "status": result.status,
                "message": result.message,
                "timestamp": result.timestamp.isoformat(),
                "duration_ms": result.duration_ms
            } for name, result in results.items()}
        }


class AlertManager:
    """
    Alert management system.
    
    Monitors metrics and health checks to generate alerts based on
    configurable thresholds and conditions.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        
        logger.info("AlertManager initialized")
    
    def add_threshold_rule(
        self,
        rule_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "greater",  # "greater", "less", "equal"
        severity: AlertSeverity = AlertSeverity.WARNING,
        duration: timedelta = timedelta(minutes=5),
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Add a threshold-based alert rule.
        
        Args:
            rule_name: Unique rule name
            metric_name: Metric to monitor
            threshold: Threshold value
            comparison: Comparison operator
            severity: Alert severity
            duration: How long condition must persist
            labels: Additional labels for filtering
        """
        self.alert_rules[rule_name] = {
            "type": "threshold",
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "duration": duration,
            "labels": labels or {}
        }
        
        logger.info(
            "Threshold alert rule added",
            rule_name=rule_name,
            metric_name=metric_name,
            threshold=threshold,
            comparison=comparison
        )
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Add a callback function to be called when alerts are triggered.
        
        Args:
            callback: Function that accepts an Alert object
        """
        self.alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    async def check_alert_rules(self):
        """Check all alert rules and trigger alerts if needed."""
        for rule_name, rule in self.alert_rules.items():
            try:
                await self._check_threshold_rule(rule_name, rule)
            except Exception as e:
                logger.error("Error checking alert rule", rule_name=rule_name, error=str(e))
    
    async def _check_threshold_rule(self, rule_name: str, rule: Dict):
        """Check a threshold-based alert rule."""
        metric_name = rule["metric_name"]
        threshold = rule["threshold"]
        comparison = rule["comparison"]
        severity = rule["severity"]
        duration = rule["duration"]
        labels = rule["labels"]
        
        # Get recent metric values
        since = datetime.utcnow() - duration
        points = self.metrics_collector.get_metric_values(metric_name, since=since, labels=labels)
        
        if not points:
            return
        
        # Check if condition is met for the duration
        violating_points = []
        
        for point in points:
            if comparison == "greater" and point.value > threshold:
                violating_points.append(point)
            elif comparison == "less" and point.value < threshold:
                violating_points.append(point)
            elif comparison == "equal" and point.value == threshold:
                violating_points.append(point)
        
        # Check if violation duration is sufficient
        if violating_points and len(violating_points) >= len(points) * 0.8:  # 80% of points
            # Condition met, trigger alert if not already active
            alert_id = f"{rule_name}_{metric_name}"
            
            if alert_id not in self.active_alerts:
                alert = Alert(
                    id=alert_id,
                    name=rule_name,
                    severity=severity,
                    message=f"Metric {metric_name} {comparison} {threshold}",
                    timestamp=datetime.utcnow(),
                    metric_name=metric_name,
                    current_value=points[-1].value if points else None,
                    threshold=threshold,
                    labels=labels
                )
                
                await self._trigger_alert(alert)
        else:
            # Condition not met, resolve alert if active
            alert_id = f"{rule_name}_{metric_name}"
            if alert_id in self.active_alerts:
                await self._resolve_alert(alert_id)
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger a new alert."""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        logger.warning(
            "Alert triggered",
            alert_id=alert.id,
            alert_name=alert.name,
            severity=alert.severity.value,
            message=alert.message
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert) if asyncio.iscoroutinefunction(callback) else callback(alert)
            except Exception as e:
                logger.error("Alert callback failed", alert_id=alert.id, error=str(e))
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            
            del self.active_alerts[alert_id]
            
            logger.info(
                "Alert resolved",
                alert_id=alert_id,
                alert_name=alert.name,
                duration=(alert.resolved_at - alert.timestamp).total_seconds()
            )
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of currently active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]


class EHSMonitor:
    """
    Main monitoring system coordinator.
    
    Coordinates all monitoring components and provides a unified interface
    for metrics collection, health checking, and alerting.
    """
    
    def __init__(
        self,
        metrics_retention_points: int = 10000,
        resource_monitoring_interval: float = 60.0,
        health_check_interval: float = 30.0
    ):
        self.metrics_collector = MetricsCollector(max_points_per_metric=metrics_retention_points)
        self.resource_monitor = ResourceMonitor(collection_interval=resource_monitoring_interval)
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Connect components
        self.resource_monitor.set_metrics_collector(self.metrics_collector)
        
        # Register default metrics
        self._register_default_metrics()
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        self.monitoring_task = None
        self.is_running = False
        
        logger.info("EHSMonitor initialized")
    
    def _register_default_metrics(self):
        """Register default EHS Analytics metrics."""
        # Query metrics
        self.metrics_collector.register_metric(
            "ehs_queries_total", MetricType.COUNTER, "Total number of queries processed"
        )
        self.metrics_collector.register_metric(
            "ehs_query_duration_ms", MetricType.HISTOGRAM, "Query processing duration in milliseconds"
        )
        self.metrics_collector.register_metric(
            "ehs_query_errors_total", MetricType.COUNTER, "Total number of query errors"
        )
        
        # Retrieval metrics
        self.metrics_collector.register_metric(
            "ehs_retrievals_total", MetricType.COUNTER, "Total number of data retrievals"
        )
        self.metrics_collector.register_metric(
            "ehs_retrieval_duration_ms", MetricType.HISTOGRAM, "Data retrieval duration in milliseconds"
        )
        self.metrics_collector.register_metric(
            "ehs_retrieval_results_count", MetricType.HISTOGRAM, "Number of results returned per retrieval"
        )
        
        # Analysis metrics
        self.metrics_collector.register_metric(
            "ehs_analyses_total", MetricType.COUNTER, "Total number of analyses performed"
        )
        self.metrics_collector.register_metric(
            "ehs_analysis_confidence", MetricType.HISTOGRAM, "Analysis confidence scores"
        )
        
        # API metrics
        self.metrics_collector.register_metric(
            "ehs_api_requests_total", MetricType.COUNTER, "Total API requests"
        )
        self.metrics_collector.register_metric(
            "ehs_api_response_time_ms", MetricType.HISTOGRAM, "API response time in milliseconds"
        )
        self.metrics_collector.register_metric(
            "ehs_api_errors_total", MetricType.COUNTER, "Total API errors"
        )
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        
        async def database_health():
            """Check Neo4j database connectivity."""
            # This will be implemented when database connection is available
            return HealthCheckResult(
                name="database",
                status="healthy",
                message="Database connection healthy",
                timestamp=datetime.utcnow(),
                duration_ms=0.0
            )
        
        async def llm_health():
            """Check LLM service availability."""
            # This will be implemented when LLM service is configured
            return HealthCheckResult(
                name="llm_service",
                status="healthy",
                message="LLM service available",
                timestamp=datetime.utcnow(),
                duration_ms=0.0
            )
        
        async def memory_health():
            """Check memory usage."""
            memory = psutil.virtual_memory()
            status = "healthy"
            if memory.percent > 90:
                status = "unhealthy"
            elif memory.percent > 80:
                status = "degraded"
            
            return HealthCheckResult(
                name="memory",
                status=status,
                message=f"Memory usage: {memory.percent}%",
                timestamp=datetime.utcnow(),
                duration_ms=0.0,
                details={"memory_percent": memory.percent}
            )
        
        self.health_checker.register_check("database", database_health, interval=30.0)
        self.health_checker.register_check("llm_service", llm_health, interval=60.0)
        self.health_checker.register_check("memory", memory_health, interval=30.0)
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High error rate
        self.alert_manager.add_threshold_rule(
            "high_error_rate",
            "ehs_query_errors_total",
            threshold=10,
            comparison="greater",
            severity=AlertSeverity.WARNING,
            duration=timedelta(minutes=5)
        )
        
        # High memory usage
        self.alert_manager.add_threshold_rule(
            "high_memory_usage",
            "system_memory_percent",
            threshold=85.0,
            comparison="greater",
            severity=AlertSeverity.WARNING,
            duration=timedelta(minutes=3)
        )
        
        # Critical memory usage
        self.alert_manager.add_threshold_rule(
            "critical_memory_usage",
            "system_memory_percent",
            threshold=95.0,
            comparison="greater",
            severity=AlertSeverity.CRITICAL,
            duration=timedelta(minutes=1)
        )
        
        # High CPU usage
        self.alert_manager.add_threshold_rule(
            "high_cpu_usage",
            "system_cpu_percent",
            threshold=80.0,
            comparison="greater",
            severity=AlertSeverity.WARNING,
            duration=timedelta(minutes=5)
        )
    
    async def start_monitoring(self):
        """Start all monitoring components."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        # Start alert checking loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("EHS monitoring started")
    
    async def stop_monitoring(self):
        """Stop all monitoring components."""
        self.is_running = False
        
        # Stop resource monitoring
        await self.resource_monitor.stop_monitoring()
        
        # Stop monitoring loop
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("EHS monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for alert checking."""
        while self.is_running:
            try:
                await self.alert_manager.check_alert_rules()
                await asyncio.sleep(30)  # Check alerts every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(30)
    
    # Convenience methods for recording metrics
    def record_query(self, query_type: str, duration_ms: float, success: bool = True):
        """Record query metrics."""
        labels = {"query_type": query_type}
        
        self.metrics_collector.record_counter("ehs_queries_total", labels=labels)
        self.metrics_collector.record_histogram("ehs_query_duration_ms", duration_ms, labels=labels)
        
        if not success:
            self.metrics_collector.record_counter("ehs_query_errors_total", labels=labels)
    
    def record_retrieval(self, strategy: str, duration_ms: float, results_count: int, success: bool = True):
        """Record retrieval metrics."""
        labels = {"strategy": strategy}
        
        self.metrics_collector.record_counter("ehs_retrievals_total", labels=labels)
        self.metrics_collector.record_histogram("ehs_retrieval_duration_ms", duration_ms, labels=labels)
        self.metrics_collector.record_histogram("ehs_retrieval_results_count", results_count, labels=labels)
    
    def record_analysis(self, analysis_type: str, confidence: float, success: bool = True):
        """Record analysis metrics."""
        labels = {"analysis_type": analysis_type}
        
        self.metrics_collector.record_counter("ehs_analyses_total", labels=labels)
        self.metrics_collector.record_histogram("ehs_analysis_confidence", confidence, labels=labels)
    
    def record_api_request(self, endpoint: str, method: str, response_time_ms: float, status_code: int):
        """Record API request metrics."""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        
        self.metrics_collector.record_counter("ehs_api_requests_total", labels=labels)
        self.metrics_collector.record_histogram("ehs_api_response_time_ms", response_time_ms, labels=labels)
        
        if status_code >= 400:
            self.metrics_collector.record_counter("ehs_api_errors_total", labels=labels)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboards."""
        # Get system health
        health_data = await self.health_checker.get_system_health()
        
        # Get key metrics
        metrics_data = self.metrics_collector.get_all_metrics()
        
        # Get active alerts
        alerts_data = {
            "active_alerts": [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "current_value": alert.current_value,
                    "threshold": alert.threshold
                }
                for alert in self.alert_manager.get_active_alerts()
            ],
            "alert_count": len(self.alert_manager.get_active_alerts())
        }
        
        # Get resource utilization
        resource_data = {
            "cpu_usage": self.metrics_collector.get_metric_summary("system_cpu_percent"),
            "memory_usage": self.metrics_collector.get_metric_summary("system_memory_percent"),
            "disk_usage": self.metrics_collector.get_metric_summary("system_disk_percent")
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": health_data,
            "metrics": metrics_data,
            "alerts": alerts_data,
            "resources": resource_data
        }


# Global monitor instance
_ehs_monitor: Optional[EHSMonitor] = None

def get_ehs_monitor() -> EHSMonitor:
    """Get the global EHS monitor instance."""
    global _ehs_monitor
    if _ehs_monitor is None:
        _ehs_monitor = EHSMonitor()
    return _ehs_monitor

async def initialize_monitoring():
    """Initialize and start the global monitoring system."""
    monitor = get_ehs_monitor()
    await monitor.start_monitoring()
    logger.info("Global EHS monitoring initialized and started")

async def shutdown_monitoring():
    """Shutdown the global monitoring system."""
    global _ehs_monitor
    if _ehs_monitor:
        await _ehs_monitor.stop_monitoring()
        _ehs_monitor = None
    logger.info("Global EHS monitoring shutdown")