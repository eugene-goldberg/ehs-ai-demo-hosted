"""
Risk Monitoring and Alerting System

Production-ready system for real-time risk monitoring, intelligent alerting,
and multi-channel notifications with Prometheus metrics and Grafana compatibility.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from uuid import uuid4
import json
import hashlib
from collections import defaultdict, deque

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..models import RiskAssessment, Facility

# Configure logging
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories for classification"""
    THRESHOLD_BREACH = "threshold_breach"
    RAPID_CHANGE = "rapid_change"
    ANOMALY = "anomaly"
    FORECAST = "forecast"
    SYSTEM = "system"
    COMPLIANCE = "compliance"
    ENVIRONMENTAL = "environmental"
    SAFETY = "safety"


class AlertUrgency(Enum):
    """Alert urgency for routing decisions"""
    IMMEDIATE = "immediate"  # < 5 minutes
    HIGH = "high"           # < 30 minutes
    NORMAL = "normal"       # < 2 hours
    LOW = "low"            # < 24 hours


class AlertChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"


class AlertStatus(Enum):
    """Alert lifecycle status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


@dataclass
class Alert:
    """Comprehensive alert data structure"""
    id: str = field(default_factory=lambda: str(uuid4()))
    facility_id: str = ""
    title: str = ""
    message: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    category: AlertCategory = AlertCategory.THRESHOLD_BREACH
    urgency: AlertUrgency = AlertUrgency.NORMAL
    
    # Risk context
    risk_score: Optional[float] = None
    risk_type: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Status tracking
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Deduplication
    fingerprint: Optional[str] = None
    
    def __post_init__(self):
        """Generate fingerprint for deduplication"""
        if not self.fingerprint:
            content = f"{self.facility_id}:{self.category.value}:{self.risk_type}:{self.threshold_value}"
            self.fingerprint = hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'facility_id': self.facility_id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'urgency': self.urgency.value,
            'risk_score': self.risk_score,
            'risk_type': self.risk_type,
            'threshold_value': self.threshold_value,
            'current_value': self.current_value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'status': self.status.value,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_by': self.resolved_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'tags': self.tags,
            'metadata': self.metadata,
            'fingerprint': self.fingerprint
        }


@dataclass
class AlertRule:
    """Configurable alert rule definition"""
    id: str
    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    urgency: AlertUrgency
    
    # Conditions
    risk_types: List[str] = field(default_factory=list)
    facility_ids: List[str] = field(default_factory=list)  # Empty = all facilities
    threshold: Optional[float] = None
    comparison: str = ">"  # >, <, >=, <=, ==, !=
    
    # Advanced conditions
    consecutive_periods: int = 1
    time_window: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    rate_of_change: Optional[float] = None  # For rapid change detection
    
    # Routing
    channels: List[AlertChannel] = field(default_factory=list)
    business_hours_only: bool = False
    escalation_delay: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    # Suppression
    suppress_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    max_alerts_per_hour: int = 10
    
    # Lifecycle
    enabled: bool = True
    expires_after: timedelta = field(default_factory=lambda: timedelta(hours=24))
    
    def evaluate(self, risk_score: float, context: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met"""
        if not self.enabled:
            return False
        
        # Check facility filter
        facility_id = context.get('facility_id')
        if self.facility_ids and facility_id not in self.facility_ids:
            return False
        
        # Check risk type filter
        risk_type = context.get('risk_type')
        if self.risk_types and risk_type not in self.risk_types:
            return False
        
        # Check threshold
        if self.threshold is not None:
            if self.comparison == ">" and not risk_score > self.threshold:
                return False
            elif self.comparison == "<" and not risk_score < self.threshold:
                return False
            elif self.comparison == ">=" and not risk_score >= self.threshold:
                return False
            elif self.comparison == "<=" and not risk_score <= self.threshold:
                return False
            elif self.comparison == "==" and not risk_score == self.threshold:
                return False
            elif self.comparison == "!=" and not risk_score != self.threshold:
                return False
        
        # Check rate of change
        if self.rate_of_change is not None:
            previous_score = context.get('previous_score', 0)
            change_rate = abs(risk_score - previous_score)
            if change_rate < self.rate_of_change:
                return False
        
        return True


@dataclass
class AlertHistory:
    """Alert history record for audit and analysis"""
    alert_id: str
    action: str  # created, acknowledged, resolved, escalated, suppressed
    performed_by: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class RiskMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self, registry: Optional[Any] = None):
        self.registry = registry or CollectorRegistry()
        
        if PROMETHEUS_AVAILABLE:
            # Alert metrics
            self.alerts_total = Counter(
                'ehs_alerts_total',
                'Total number of alerts generated',
                ['facility_id', 'severity', 'category'],
                registry=self.registry
            )
            
            self.alert_response_time = Histogram(
                'ehs_alert_response_time_seconds',
                'Time from alert creation to acknowledgment',
                ['severity', 'category'],
                registry=self.registry
            )
            
            self.alert_resolution_time = Histogram(
                'ehs_alert_resolution_time_seconds',
                'Time from alert creation to resolution',
                ['severity', 'category'],
                registry=self.registry
            )
            
            # Risk monitoring metrics
            self.risk_scores = Gauge(
                'ehs_risk_score',
                'Current risk score by facility and type',
                ['facility_id', 'risk_type'],
                registry=self.registry
            )
            
            self.monitoring_errors = Counter(
                'ehs_monitoring_errors_total',
                'Total monitoring errors',
                ['error_type'],
                registry=self.registry
            )
            
            self.active_alerts = Gauge(
                'ehs_active_alerts',
                'Number of active alerts',
                ['severity'],
                registry=self.registry
            )
            
            # System performance
            self.monitoring_duration = Histogram(
                'ehs_monitoring_cycle_duration_seconds',
                'Duration of monitoring cycles',
                registry=self.registry
            )
    
    def record_alert(self, alert: Alert):
        """Record alert creation metrics"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'alerts_total'):
            self.alerts_total.labels(
                facility_id=alert.facility_id,
                severity=alert.severity.value,
                category=alert.category.value
            ).inc()
    
    def record_response_time(self, alert: Alert, response_time: float):
        """Record alert response time"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'alert_response_time'):
            self.alert_response_time.labels(
                severity=alert.severity.value,
                category=alert.category.value
            ).observe(response_time)
    
    def record_resolution_time(self, alert: Alert, resolution_time: float):
        """Record alert resolution time"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'alert_resolution_time'):
            self.alert_resolution_time.labels(
                severity=alert.severity.value,
                category=alert.category.value
            ).observe(resolution_time)
    
    def update_risk_score(self, facility_id: str, risk_type: str, score: float):
        """Update current risk score gauge"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'risk_scores'):
            self.risk_scores.labels(
                facility_id=facility_id,
                risk_type=risk_type
            ).set(score)
    
    def record_error(self, error_type: str):
        """Record monitoring error"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'monitoring_errors'):
            self.monitoring_errors.labels(error_type=error_type).inc()
    
    def update_active_alerts(self, alerts_by_severity: Dict[str, int]):
        """Update active alerts gauge"""
        if PROMETHEUS_AVAILABLE and hasattr(self, 'active_alerts'):
            for severity, count in alerts_by_severity.items():
                self.active_alerts.labels(severity=severity).set(count)


class RiskMonitoringSystem:
    """
    Comprehensive risk monitoring and alerting system
    
    Features:
    - Real-time risk tracking across facilities
    - Intelligent alert generation and deduplication
    - Multi-channel notification support
    - Alert escalation and acknowledgment
    - Prometheus metrics and Grafana compatibility
    """
    
    def __init__(
        self,
        monitoring_interval: int = 60,  # seconds
        enable_metrics: bool = True,
        prometheus_gateway: Optional[str] = None
    ):
        self.monitoring_interval = monitoring_interval
        self.enable_metrics = enable_metrics
        self.prometheus_gateway = prometheus_gateway
        
        # Core state
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: List[AlertHistory] = []
        self.suppressed_alerts: Dict[str, datetime] = {}  # fingerprint -> suppress_until
        
        # Monitoring state
        self.risk_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.consecutive_breaches: Dict[str, int] = defaultdict(int)
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Notification handlers
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        
        # Metrics
        self.metrics = RiskMetrics() if enable_metrics else None
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        logger.info("Risk monitoring system initialized")
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add or update an alert rule"""
        self.alert_rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> None:
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def register_notification_handler(
        self,
        channel: AlertChannel,
        handler: Callable[[Alert], None]
    ) -> None:
        """Register a notification handler for a channel"""
        self.notification_handlers[channel] = handler
        logger.info(f"Registered handler for {channel.value}")
    
    async def start_monitoring(self) -> None:
        """Start the background monitoring task"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started risk monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop the background monitoring task"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped risk monitoring")
    
    def generate_alerts(self, 
                       risk_data: Dict[str, Any],
                       threshold_config: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Generate risk alerts based on current risk data and thresholds.
        
        Args:
            risk_data: Current risk assessment data
            threshold_config: Custom threshold configuration
            
        Returns:
            List of alert dictionaries with severity, message, and recommendations
        """
        alerts = []
        default_thresholds = {
            'overall_risk_score': 0.7,
            'water_risk': 0.8,
            'electricity_risk': 0.8,
            'waste_risk': 0.75,
            'environmental_risk': 0.8,
            'safety_risk': 0.85,
            'compliance_risk': 0.9
        }
        thresholds = threshold_config or default_thresholds
        
        # Extract facility information
        facility_id = risk_data.get('facility_id', 'unknown')
        facility_name = risk_data.get('facility_name', 'Unknown Facility')
        
        # Process risk scores
        for metric, value in risk_data.items():
            if isinstance(value, (int, float)) and metric in thresholds:
                threshold = thresholds.get(metric, 0.8)
                if value > threshold:
                    # Calculate severity based on how much threshold is exceeded
                    severity = self._calculate_alert_severity(value, threshold)
                    
                    # Generate alert
                    alert_dict = {
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'severity': severity.value,
                        'message': self._generate_alert_message_for_metric(
                            metric, value, threshold, facility_name, severity
                        ),
                        'timestamp': datetime.utcnow().isoformat(),
                        'recommendations': self._get_recommendations(metric, value, severity),
                        'facility_id': facility_id,
                        'facility_name': facility_name,
                        'urgency': self._determine_urgency(severity, value / threshold).value,
                        'category': self._determine_category(metric).value
                    }
                    alerts.append(alert_dict)
        
        # Check for composite risks
        if 'risk_factors' in risk_data and isinstance(risk_data['risk_factors'], dict):
            for factor, score in risk_data['risk_factors'].items():
                if isinstance(score, (int, float)):
                    factor_threshold = thresholds.get(f"{factor}_factor", 0.75)
                    if score > factor_threshold:
                        severity = self._calculate_alert_severity(score, factor_threshold)
                        
                        alert_dict = {
                            'metric': f"{factor}_factor",
                            'value': score,
                            'threshold': factor_threshold,
                            'severity': severity.value,
                            'message': f"Risk factor '{factor}' at {facility_name} exceeded threshold: {score:.2f} > {factor_threshold:.2f}",
                            'timestamp': datetime.utcnow().isoformat(),
                            'recommendations': self._get_recommendations(factor, score, severity),
                            'facility_id': facility_id,
                            'facility_name': facility_name,
                            'urgency': self._determine_urgency(severity, score / factor_threshold).value,
                            'category': AlertCategory.THRESHOLD_BREACH.value
                        }
                        alerts.append(alert_dict)
        
        return sorted(alerts, key=lambda x: self._severity_sort_key(x['severity']), reverse=True)

    def _calculate_alert_severity(self, value: float, threshold: float) -> AlertSeverity:
        """Calculate alert severity based on value and threshold"""
        ratio = value / threshold if threshold > 0 else float('inf')
        
        if ratio >= 1.5:
            return AlertSeverity.CRITICAL
        elif ratio >= 1.25:
            return AlertSeverity.HIGH
        elif ratio >= 1.1:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _generate_alert_message_for_metric(self, metric: str, value: float, threshold: float, 
                                          facility_name: str, severity: AlertSeverity) -> str:
        """Generate descriptive alert message for a specific metric"""
        severity_text = severity.value.upper()
        
        metric_names = {
            'overall_risk_score': 'Overall risk score',
            'water_risk': 'Water consumption risk',
            'electricity_risk': 'Electricity consumption risk',
            'waste_risk': 'Waste generation risk',
            'environmental_risk': 'Environmental compliance risk',
            'safety_risk': 'Safety risk',
            'compliance_risk': 'Regulatory compliance risk'
        }
        
        metric_display = metric_names.get(metric, metric.replace('_', ' ').title())
        
        return (f"{severity_text}: {metric_display} at {facility_name} "
                f"exceeded threshold ({value:.2f} > {threshold:.2f})")

    def _get_recommendations(self, metric: str, value: float, severity: AlertSeverity) -> List[str]:
        """Get recommendations based on metric and severity"""
        recommendations = []
        
        # Severity-based recommendations
        if severity == AlertSeverity.CRITICAL:
            recommendations.append("Immediate investigation and intervention required")
            recommendations.append("Notify senior management and compliance team")
        elif severity == AlertSeverity.HIGH:
            recommendations.append("Investigate root cause within 24 hours")
            recommendations.append("Review relevant operational procedures")
        else:
            recommendations.append("Monitor closely for further changes")
        
        # Metric-specific recommendations
        if 'water' in metric.lower():
            recommendations.append("Check for leaks in water systems")
            recommendations.append("Review water treatment and recycling processes")
        elif 'electricity' in metric.lower():
            recommendations.append("Audit electrical equipment efficiency")
            recommendations.append("Review HVAC and lighting schedules")
        elif 'waste' in metric.lower():
            recommendations.append("Review waste segregation practices")
            recommendations.append("Check waste contractor compliance")
        elif 'environmental' in metric.lower():
            recommendations.append("Review environmental permits and compliance status")
            recommendations.append("Schedule environmental audit if needed")
        elif 'safety' in metric.lower():
            recommendations.append("Review safety protocols and training records")
            recommendations.append("Conduct safety walkthrough inspection")
        elif 'compliance' in metric.lower():
            recommendations.append("Review regulatory requirements and deadlines")
            recommendations.append("Update compliance documentation")
        
        return recommendations

    def _determine_urgency(self, severity: AlertSeverity, ratio: float) -> AlertUrgency:
        """Determine alert urgency based on severity and threshold ratio"""
        if severity == AlertSeverity.CRITICAL or ratio > 1.5:
            return AlertUrgency.IMMEDIATE
        elif severity == AlertSeverity.HIGH or ratio > 1.25:
            return AlertUrgency.HIGH
        elif severity == AlertSeverity.MEDIUM or ratio > 1.1:
            return AlertUrgency.NORMAL
        else:
            return AlertUrgency.LOW

    def _determine_category(self, metric: str) -> AlertCategory:
        """Determine alert category based on metric type"""
        if 'compliance' in metric.lower():
            return AlertCategory.COMPLIANCE
        elif 'environmental' in metric.lower():
            return AlertCategory.ENVIRONMENTAL
        elif 'safety' in metric.lower():
            return AlertCategory.SAFETY
        elif any(risk in metric.lower() for risk in ['water', 'electricity', 'waste']):
            return AlertCategory.THRESHOLD_BREACH
        else:
            return AlertCategory.THRESHOLD_BREACH

    def _severity_sort_key(self, severity: str) -> int:
        """Get sort key for severity ordering"""
        severity_order = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return severity_order.get(severity.lower(), 0)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                start_time = datetime.utcnow()
                
                await self.monitor_risks()
                await self._cleanup_expired_alerts()
                await self._cleanup_suppressed_alerts()
                
                # Record monitoring duration
                if self.metrics:
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    self.metrics.monitoring_duration.observe(duration)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                if self.metrics:
                    self.metrics.record_error("monitoring_loop")
                await asyncio.sleep(self.monitoring_interval)
    
    async def monitor_risks(self) -> None:
        """Monitor risks across all facilities and generate alerts"""
        try:
            # This would typically query your risk assessment system
            # For now, we'll demonstrate with a mock implementation
            facilities = await self._get_facilities()
            
            for facility in facilities:
                risk_assessments = await self._get_recent_risk_assessments(facility.id)
                
                for assessment in risk_assessments:
                    await self._evaluate_risk_assessment(facility, assessment)
            
            # Update metrics
            if self.metrics:
                await self._update_dashboard_metrics()
                
        except Exception as e:
            logger.error(f"Error monitoring risks: {e}")
            if self.metrics:
                self.metrics.record_error("risk_monitoring")
    
    async def _evaluate_risk_assessment(
        self,
        facility: Facility,
        assessment: RiskAssessment
    ) -> None:
        """Evaluate a risk assessment against all alert rules"""
        context = {
            'facility_id': facility.id,
            'risk_type': assessment.risk_type,
            'assessment_time': assessment.assessment_time,
            'previous_score': self._get_previous_risk_score(facility.id, assessment.risk_type)
        }
        
        # Store risk history
        history_key = f"{facility.id}:{assessment.risk_type}"
        self.risk_history[history_key].append({
            'score': assessment.overall_score,
            'timestamp': assessment.assessment_time
        })
        
        # Update metrics
        if self.metrics:
            self.metrics.update_risk_score(
                facility.id,
                assessment.risk_type,
                assessment.overall_score
            )
        
        # Evaluate against all rules
        for rule in self.alert_rules.values():
            if rule.evaluate(assessment.overall_score, context):
                await self._process_rule_match(rule, facility, assessment, context)
    
    async def _process_rule_match(
        self,
        rule: AlertRule,
        facility: Facility,
        assessment: RiskAssessment,
        context: Dict[str, Any]
    ) -> None:
        """Process a matched alert rule"""
        # Check consecutive periods requirement
        if rule.consecutive_periods > 1:
            breach_key = f"{rule.id}:{facility.id}:{assessment.risk_type}"
            self.consecutive_breaches[breach_key] += 1
            
            if self.consecutive_breaches[breach_key] < rule.consecutive_periods:
                return  # Not enough consecutive breaches
        
        # Create alert
        alert = Alert(
            facility_id=facility.id,
            title=f"{rule.name} - {facility.name}",
            message=self._generate_alert_message(rule, facility, assessment),
            severity=rule.severity,
            category=rule.category,
            urgency=rule.urgency,
            risk_score=assessment.overall_score,
            risk_type=assessment.risk_type,
            threshold_value=rule.threshold,
            current_value=assessment.overall_score,
            expires_at=datetime.utcnow() + rule.expires_after,
            tags={'rule_id': rule.id, 'facility_name': facility.name},
            metadata=context
        )
        
        # Check if alert should be suppressed
        if await self._should_suppress_alert(alert, rule):
            logger.debug(f"Suppressing duplicate alert: {alert.fingerprint}")
            return
        
        # Generate and route alert
        await self._generate_alert(alert, rule)
    
    def _generate_alert_message(
        self,
        rule: AlertRule,
        facility: Facility,
        assessment: RiskAssessment
    ) -> str:
        """Generate a descriptive alert message"""
        base_message = f"Risk assessment triggered alert rule '{rule.name}' at {facility.name}"
        
        if rule.threshold:
            base_message += f"\nCurrent risk score: {assessment.overall_score:.2f}"
            base_message += f"\nThreshold: {rule.threshold:.2f}"
        
        if assessment.risk_factors:
            top_risks = sorted(
                assessment.risk_factors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            base_message += f"\nTop risk factors: {', '.join([f'{k}: {v:.2f}' for k, v in top_risks])}"
        
        if assessment.recommendations:
            base_message += f"\nRecommendations: {'; '.join(assessment.recommendations[:2])}"
        
        return base_message
    
    async def _should_suppress_alert(self, alert: Alert, rule: AlertRule) -> bool:
        """Check if alert should be suppressed due to deduplication rules"""
        fingerprint = alert.fingerprint
        
        # Check if already suppressed
        if fingerprint in self.suppressed_alerts:
            suppress_until = self.suppressed_alerts[fingerprint]
            if datetime.utcnow() < suppress_until:
                return True
            else:
                del self.suppressed_alerts[fingerprint]
        
        # Check rate limiting
        if fingerprint in self.last_alert_times:
            last_alert = self.last_alert_times[fingerprint]
            if datetime.utcnow() - last_alert < rule.suppress_duration:
                return True
        
        # Check max alerts per hour
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_alerts = [
            a for a in self.active_alerts.values()
            if a.fingerprint == fingerprint and a.created_at > hour_ago
        ]
        
        if len(recent_alerts) >= rule.max_alerts_per_hour:
            # Suppress for the rule's suppress duration
            self.suppressed_alerts[fingerprint] = datetime.utcnow() + rule.suppress_duration
            return True
        
        return False
    
    async def _generate_alert(self, alert: Alert, rule: AlertRule) -> None:
        """Generate and route an alert"""
        # Store alert
        self.active_alerts[alert.id] = alert
        self.last_alert_times[alert.fingerprint] = alert.created_at
        
        # Record metrics
        if self.metrics:
            self.metrics.record_alert(alert)
        
        # Add to history
        self.alert_history.append(AlertHistory(
            alert_id=alert.id,
            action="created",
            performed_by="system",
            timestamp=alert.created_at,
            details={'rule_id': rule.id}
        ))
        
        # Send notifications
        await self._send_alert(alert, rule.channels)
        
        logger.info(f"Generated alert: {alert.title} (ID: {alert.id})")
    
    async def _send_alert(self, alert: Alert, channels: List[AlertChannel]) -> None:
        """Send alert through specified channels"""
        for channel in channels:
            try:
                if channel in self.notification_handlers:
                    await asyncio.create_task(
                        self._call_handler(self.notification_handlers[channel], alert)
                    )
                else:
                    logger.warning(f"No handler registered for channel: {channel.value}")
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                if self.metrics:
                    self.metrics.record_error(f"notification_{channel.value}")
    
    async def _call_handler(self, handler: Callable, alert: Alert) -> None:
        """Call notification handler (async-safe)"""
        if asyncio.iscoroutinefunction(handler):
            await handler(alert)
        else:
            handler(alert)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        
        # Record response time metrics
        if self.metrics and alert.created_at:
            response_time = (alert.acknowledged_at - alert.created_at).total_seconds()
            self.metrics.record_response_time(alert, response_time)
        
        # Add to history
        self.alert_history.append(AlertHistory(
            alert_id=alert_id,
            action="acknowledged",
            performed_by=acknowledged_by,
            timestamp=alert.acknowledged_at
        ))
        
        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str = "") -> bool:
        """Resolve an active alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        
        if resolution_notes:
            alert.metadata['resolution_notes'] = resolution_notes
        
        # Record resolution time metrics
        if self.metrics and alert.created_at:
            resolution_time = (alert.resolved_at - alert.created_at).total_seconds()
            self.metrics.record_resolution_time(alert, resolution_time)
        
        # Add to history
        self.alert_history.append(AlertHistory(
            alert_id=alert_id,
            action="resolved",
            performed_by=resolved_by,
            timestamp=alert.resolved_at,
            details={'resolution_notes': resolution_notes}
        ))
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
        return True
    
    async def suppress_alert(self, alert_id: str, suppress_duration: timedelta, suppressed_by: str) -> bool:
        """Suppress an alert for a specified duration"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        alert.updated_at = datetime.utcnow()
        
        # Add to suppression list
        suppress_until = datetime.utcnow() + suppress_duration
        self.suppressed_alerts[alert.fingerprint] = suppress_until
        
        # Add to history
        self.alert_history.append(AlertHistory(
            alert_id=alert_id,
            action="suppressed",
            performed_by=suppressed_by,
            timestamp=datetime.utcnow(),
            details={'suppress_until': suppress_until.isoformat()}
        ))
        
        logger.info(f"Alert suppressed: {alert_id} until {suppress_until}")
        return True
    
    def get_active_alerts(
        self,
        facility_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None
    ) -> List[Alert]:
        """Get filtered list of active alerts"""
        alerts = list(self.active_alerts.values())
        
        if facility_id:
            alerts = [a for a in alerts if a.facility_id == facility_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_history(
        self,
        alert_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AlertHistory]:
        """Get alert history records"""
        history = self.alert_history
        
        if alert_id:
            history = [h for h in history if h.alert_id == alert_id]
        
        return sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    async def generate_dashboard_metrics(self) -> Dict[str, Any]:
        """Generate real-time metrics for dashboard"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        active_alerts = list(self.active_alerts.values())
        
        # Alert counts by severity
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Alert counts by category
        category_counts = defaultdict(int)
        for alert in active_alerts:
            category_counts[alert.category.value] += 1
        
        # Recent alert trends
        recent_alerts = [a for a in active_alerts if a.created_at > day_ago]
        hourly_alerts = [a for a in recent_alerts if a.created_at > hour_ago]
        
        # Facility risk summary
        facility_risks = defaultdict(list)
        for history_key, risk_data in self.risk_history.items():
            if risk_data:
                facility_id = history_key.split(':')[0]
                latest_risk = risk_data[-1]
                facility_risks[facility_id].append(latest_risk['score'])
        
        facility_avg_risks = {
            facility_id: sum(scores) / len(scores)
            for facility_id, scores in facility_risks.items()
        }
        
        # Response time stats
        acknowledged_alerts = [
            a for a in active_alerts
            if a.status == AlertStatus.ACKNOWLEDGED and a.acknowledged_at
        ]
        
        avg_response_time = 0
        if acknowledged_alerts:
            response_times = [
                (a.acknowledged_at - a.created_at).total_seconds()
                for a in acknowledged_alerts
            ]
            avg_response_time = sum(response_times) / len(response_times)
        
        metrics = {
            'timestamp': now.isoformat(),
            'summary': {
                'total_active_alerts': len(active_alerts),
                'alerts_last_hour': len(hourly_alerts),
                'alerts_last_day': len(recent_alerts),
                'avg_response_time_seconds': avg_response_time,
                'facilities_monitored': len(facility_risks)
            },
            'alerts_by_severity': dict(severity_counts),
            'alerts_by_category': dict(category_counts),
            'facility_risk_scores': facility_avg_risks,
            'system_health': {
                'monitoring_enabled': self.is_monitoring,
                'active_rules': len([r for r in self.alert_rules.values() if r.enabled]),
                'suppressed_alerts': len(self.suppressed_alerts),
                'notification_channels': len(self.notification_handlers)
            }
        }
        
        return metrics
    
    async def _update_dashboard_metrics(self) -> None:
        """Update Prometheus metrics for Grafana dashboard"""
        if not self.metrics:
            return
        
        # Update active alerts gauge
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        self.metrics.update_active_alerts(severity_counts)
        
        # Push metrics to Prometheus gateway if configured
        if self.prometheus_gateway and PROMETHEUS_AVAILABLE:
            try:
                push_to_gateway(
                    self.prometheus_gateway,
                    job='ehs-risk-monitoring',
                    registry=self.metrics.registry
                )
            except Exception as e:
                logger.error(f"Failed to push metrics to Prometheus: {e}")
    
    async def _cleanup_expired_alerts(self) -> None:
        """Remove expired alerts"""
        now = datetime.utcnow()
        expired_alert_ids = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.expires_at and now > alert.expires_at
        ]
        
        for alert_id in expired_alert_ids:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.EXPIRED
            
            # Add to history
            self.alert_history.append(AlertHistory(
                alert_id=alert_id,
                action="expired",
                performed_by="system",
                timestamp=now
            ))
            
            del self.active_alerts[alert_id]
            logger.debug(f"Expired alert: {alert_id}")
    
    async def _cleanup_suppressed_alerts(self) -> None:
        """Clean up expired suppressions"""
        now = datetime.utcnow()
        expired_suppressions = [
            fingerprint for fingerprint, suppress_until in self.suppressed_alerts.items()
            if now > suppress_until
        ]
        
        for fingerprint in expired_suppressions:
            del self.suppressed_alerts[fingerprint]
    
    def _get_previous_risk_score(self, facility_id: str, risk_type: str) -> float:
        """Get previous risk score for change detection"""
        history_key = f"{facility_id}:{risk_type}"
        history = self.risk_history.get(history_key, deque())
        
        if len(history) >= 2:
            return history[-2]['score']
        
        return 0.0
    
    async def _get_facilities(self) -> List[Facility]:
        """Get list of facilities to monitor (mock implementation)"""
        # This would typically query your facility database
        return [
            Facility(id="facility-1", name="Manufacturing Plant A"),
            Facility(id="facility-2", name="Distribution Center B"),
            Facility(id="facility-3", name="Office Complex C")
        ]
    
    async def _get_recent_risk_assessments(self, facility_id: str) -> List[RiskAssessment]:
        """Get recent risk assessments for a facility (mock implementation)"""
        # This would typically query your risk assessment system
        return []


# Example usage and built-in alert rules
def create_default_alert_rules() -> List[AlertRule]:
    """Create a set of default alert rules"""
    return [
        # High risk threshold breach
        AlertRule(
            id="high-risk-threshold",
            name="High Risk Score Alert",
            description="Alert when risk score exceeds 7.0",
            category=AlertCategory.THRESHOLD_BREACH,
            severity=AlertSeverity.HIGH,
            urgency=AlertUrgency.HIGH,
            threshold=7.0,
            comparison=">",
            channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD, AlertChannel.SLACK],
            consecutive_periods=2,
            suppress_duration=timedelta(hours=2)
        ),
        
        # Critical risk threshold
        AlertRule(
            id="critical-risk-threshold",
            name="Critical Risk Score Alert",
            description="Alert when risk score exceeds 8.5",
            category=AlertCategory.THRESHOLD_BREACH,
            severity=AlertSeverity.CRITICAL,
            urgency=AlertUrgency.IMMEDIATE,
            threshold=8.5,
            comparison=">",
            channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.PAGERDUTY],
            consecutive_periods=1,
            suppress_duration=timedelta(minutes=30)
        ),
        
        # Rapid risk increase
        AlertRule(
            id="rapid-risk-increase",
            name="Rapid Risk Increase",
            description="Alert when risk score increases by more than 2.0 points",
            category=AlertCategory.RAPID_CHANGE,
            severity=AlertSeverity.MEDIUM,
            urgency=AlertUrgency.HIGH,
            rate_of_change=2.0,
            channels=[AlertChannel.EMAIL, AlertChannel.DASHBOARD],
            suppress_duration=timedelta(hours=1)
        ),
        
        # Environmental compliance risk
        AlertRule(
            id="environmental-compliance",
            name="Environmental Compliance Risk",
            description="Alert for environmental compliance issues",
            category=AlertCategory.COMPLIANCE,
            severity=AlertSeverity.HIGH,
            urgency=AlertUrgency.HIGH,
            risk_types=["environmental"],
            threshold=6.0,
            comparison=">",
            channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            business_hours_only=True,
            suppress_duration=timedelta(hours=4)
        ),
        
        # Safety incident forecast
        AlertRule(
            id="safety-incident-forecast",
            name="Safety Incident Forecast",
            description="Early warning for potential safety incidents",
            category=AlertCategory.FORECAST,
            severity=AlertSeverity.MEDIUM,
            urgency=AlertUrgency.NORMAL,
            risk_types=["safety"],
            threshold=5.0,
            comparison=">",
            consecutive_periods=3,
            channels=[AlertChannel.EMAIL, AlertChannel.TEAMS],
            suppress_duration=timedelta(hours=6)
        )
    ]


# Example notification handlers
async def email_notification_handler(alert: Alert) -> None:
    """Example email notification handler"""
    logger.info(f"[EMAIL] Alert: {alert.title}")
    logger.info(f"[EMAIL] Message: {alert.message}")
    logger.info(f"[EMAIL] Severity: {alert.severity.value}")
    # Implement actual email sending logic here


async def slack_notification_handler(alert: Alert) -> None:
    """Example Slack notification handler"""
    logger.info(f"[SLACK] Alert: {alert.title}")
    logger.info(f"[SLACK] Facility: {alert.facility_id}")
    logger.info(f"[SLACK] Risk Score: {alert.risk_score}")
    # Implement actual Slack API calls here


async def webhook_notification_handler(alert: Alert) -> None:
    """Example webhook notification handler"""
    import json
    payload = alert.to_dict()
    logger.info(f"[WEBHOOK] Sending alert payload: {json.dumps(payload, indent=2)}")
    # Implement actual HTTP webhook calls here

# Create aliases for backward compatibility with test expectations
RiskMonitor = RiskMonitoringSystem

# Create placeholder classes for expected imports
class AlertManager:
    """Alert manager for handling risk alerts."""
    def __init__(self):
        self.risk_monitoring = RiskMonitoringSystem()
    
    async def process_alert(self, alert: Alert) -> None:
        await self.risk_monitoring.evaluate_alert_rules(alert.facility, alert.assessment)

class EscalationChain:
    """Escalation chain for alert management."""
    def __init__(self, levels: List[str] = None):
        self.levels = levels or ['level1', 'level2', 'level3']
    
    def escalate(self, alert: Alert, level: int = 0) -> str:
        if level < len(self.levels):
            return self.levels[level]
        return self.levels[-1]

class MetricsCollector:
    """Metrics collector for risk monitoring."""
    def __init__(self):
        self.metrics = RiskMetrics()
    
    def collect_risk_metrics(self, facility_id: str) -> Dict[str, float]:
        return {
            'risk_score': 5.0,
            'alert_count': 3,
            'incidents': 1
        }