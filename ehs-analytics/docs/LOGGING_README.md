# EHS Analytics - Comprehensive Logging & Monitoring System

## Overview

This document describes the production-ready logging and monitoring system implemented for the EHS Analytics platform. The system provides structured logging, metrics collection, distributed tracing, health monitoring, and alerting capabilities.

## 🏗️ Architecture

### Core Components

1. **Structured Logging** (`utils/logging.py`)
   - JSON formatted logs for production
   - Context injection (user_id, request_id, trace_id)
   - Log rotation and retention policies
   - Performance logging decorators
   - EHS-specific logging methods

2. **Monitoring & Metrics** (`utils/monitoring.py`)
   - Real-time metrics collection
   - Resource monitoring (CPU, memory, disk)
   - Health checks for system components
   - Alert management with configurable thresholds
   - Dashboard data aggregation

3. **Distributed Tracing** (`utils/tracing.py`)
   - End-to-end request tracing
   - Performance profiling with memory tracking
   - Trace context propagation
   - Analytics and performance insights

## 🚀 Quick Start

### Basic Setup

```python
import asyncio
from src.ehs_analytics.utils.logging import setup_logging, get_ehs_logger
from src.ehs_analytics.utils.monitoring import initialize_monitoring
from src.ehs_analytics.utils.tracing import get_ehs_tracer

# Configure logging
setup_logging(
    log_level="INFO",
    log_dir="logs",
    json_format=True,
    enable_rotation=True
)

# Initialize monitoring
await initialize_monitoring()

# Get logger and start using
logger = get_ehs_logger(__name__)
logger.info("Application started", version="1.0.0")
```

### Using Configuration Script

```bash
# Run the example configuration
python3 logging_config.py

# Or set environment for production
ENVIRONMENT=production python3 logging_config.py
```

## 📝 Logging Features

### Standard Logging

```python
from src.ehs_analytics.utils.logging import get_ehs_logger, log_context

logger = get_ehs_logger(__name__)

# Basic logging with context
with log_context(user_id="123", request_id="req456"):
    logger.info("Processing query", query_type="consumption_analysis")
    logger.error("Query failed", error="Database timeout")
```

### EHS-Specific Logging Methods

```python
# Query operations
logger.query_start("What is water usage?", "consumption_analysis")
logger.query_end("What is water usage?", "consumption_analysis", 1250.0, success=True)

# Data retrieval operations
logger.retrieval_operation("text2cypher", results_count=15, duration_ms=850.0)

# Analysis operations
logger.analysis_operation("risk_assessment", confidence=0.85, duration_ms=450.0)

# Recommendations
logger.recommendation_generated(count=3, total_savings=25000.0)

# Security events
logger.security_event("unauthorized_access", "warning", {"ip": "192.168.1.100"})

# Performance metrics
logger.performance_metric("response_time", 1250.0, "ms")
```

### Performance Decorators

```python
from src.ehs_analytics.utils.logging import performance_logger

@performance_logger(include_args=True, include_result=False)
async def process_ehs_query(query: str, user_id: str):
    # Function automatically logged with execution time
    return await expensive_operation(query)
```

## 📊 Monitoring & Metrics

### Metrics Collection

```python
from src.ehs_analytics.utils.monitoring import get_ehs_monitor

monitor = get_ehs_monitor()

# Record application metrics
monitor.record_query("consumption_analysis", duration_ms=1200, success=True)
monitor.record_retrieval("text2cypher", duration_ms=800, results_count=15)
monitor.record_analysis("risk_assessment", confidence=0.85)

# Record API metrics
monitor.record_api_request("/api/query", "POST", response_time_ms=1200, status_code=200)
```

### Health Checks

```python
# Built-in health checks for:
# - Database connectivity
# - LLM service availability
# - Memory usage
# - System resources

# Get overall system health
health_data = await monitor.health_checker.get_system_health()
print(f"System status: {health_data['overall_status']}")
```

### Custom Alerts

```python
from src.ehs_analytics.utils.monitoring import AlertSeverity
from datetime import timedelta

# Add custom alert rules
monitor.alert_manager.add_threshold_rule(
    rule_name="high_query_latency",
    metric_name="ehs_query_duration_ms",
    threshold=5000,  # 5 seconds
    comparison="greater",
    severity=AlertSeverity.WARNING,
    duration=timedelta(minutes=3)
)

# Add alert callback
async def alert_handler(alert):
    print(f"ALERT: {alert.message}")
    # Send to Slack, email, etc.

monitor.alert_manager.add_alert_callback(alert_handler)
```

## 🔍 Distributed Tracing

### Basic Tracing

```python
from src.ehs_analytics.utils.tracing import get_ehs_tracer, trace_function, SpanKind

tracer = get_ehs_tracer()

# Manual span creation
with tracer.span("query_processing", tags={"query_type": "consumption"}):
    result = process_query()

# Function decorator
@trace_function("data_retrieval", SpanKind.CLIENT, {"service": "neo4j"})
async def retrieve_data(query: str):
    return await database_query(query)
```

### Performance Profiling

```python
from src.ehs_analytics.utils.tracing import get_ehs_profiler

profiler = get_ehs_profiler()

with profiler.profile_operation("expensive_analysis"):
    # Memory and CPU usage automatically tracked
    result = complex_calculation()
```

### Trace Analytics

```python
from src.ehs_analytics.utils.tracing import get_trace_analytics

analytics = get_trace_analytics()
print(f"Total traces: {analytics['total_traces']}")
print(f"Average duration: {analytics['avg_duration_ms']}ms")
print(f"Error rate: {analytics['error_rate']:.1%}")
```

## 🗂️ Log File Structure

The system creates the following log files in the `logs/` directory:

```
logs/
├── ehs_analytics.log          # Main application logs
├── errors.log                 # Warning and error level logs
├── queries.log               # Query-specific operations
├── performance.log           # Performance metrics and timings
├── ehs_analytics.log.1       # Rotated log files
├── ehs_analytics.log.2
└── ...
```

### Log Format (JSON)

```json
{
  "timestamp": "2025-08-20T10:30:45.123Z",
  "level": "INFO",
  "logger": "ehs_analytics.agents.query_router",
  "message": "Query classification completed successfully",
  "module": "query_router",
  "function": "classify_query",
  "line": 285,
  "thread": 140234567890,
  "process": 12345,
  "user_id": "user123",
  "request_id": "req456",
  "trace_id": "trace789",
  "intent_type": "consumption_analysis",
  "confidence_score": 0.92,
  "duration_ms": 1250.5
}
```

## 📈 Dashboard Integration

### Getting Dashboard Data

```python
monitor = get_ehs_monitor()
dashboard_data = await monitor.get_dashboard_data()

# Returns comprehensive monitoring data:
# - System health status
# - Key metrics summaries
# - Active alerts
# - Resource utilization
# - Performance trends
```

### Integration with Prometheus/Grafana

The metrics collector can be easily integrated with Prometheus:

```python
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
from src.ehs_analytics.utils.monitoring import get_ehs_monitor

# Export EHS metrics to Prometheus format
def create_prometheus_metrics():
    registry = CollectorRegistry()
    
    # Create Prometheus metrics
    query_duration = Histogram(
        'ehs_query_duration_seconds',
        'Query processing time',
        ['query_type'],
        registry=registry
    )
    
    # Update from EHS monitor
    monitor = get_ehs_monitor()
    metrics = monitor.metrics_collector.get_all_metrics()
    
    # Export to Prometheus endpoint
    return registry
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Logging configuration
EHS_LOG_LEVEL=INFO
EHS_LOG_DIR=logs
EHS_LOG_JSON=true
EHS_LOG_ROTATION=true
EHS_LOG_MAX_BYTES=50000000
EHS_LOG_BACKUP_COUNT=10

# Monitoring configuration
EHS_MONITOR_RETENTION_POINTS=10000
EHS_RESOURCE_MONITOR_INTERVAL=60
EHS_HEALTH_CHECK_INTERVAL=30

# Tracing configuration
EHS_TRACING_ENABLED=true
EHS_TRACE_SAMPLING_RATE=1.0
```

### Configuration File Example

```python
# logging_config.py
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_dir": "logs",
    "enable_file_logging": True,
    "enable_console_logging": True,
    "enable_rotation": True,
    "max_bytes": 50 * 1024 * 1024,
    "backup_count": 10,
    "json_format": True,
    "include_extra": True
}

MONITORING_CONFIG = {
    "metrics_retention_points": 10000,
    "resource_monitoring_interval": 60.0,
    "health_check_interval": 30.0
}
```

## 🧪 Testing & Development

### Development Setup

```python
# Use human-readable logs in development
setup_logging(
    log_level="DEBUG",
    json_format=False,
    enable_rotation=False
)
```

### Testing Logging

```python
import pytest
from src.ehs_analytics.utils.logging import get_ehs_logger

def test_query_logging(caplog):
    logger = get_ehs_logger("test")
    logger.query_start("test query", "consumption")
    
    assert "Query operation started" in caplog.text
    assert "consumption" in caplog.text
```

## 🔍 Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure the application has write permissions to the log directory
2. **Disk Space**: Monitor disk usage with log rotation enabled
3. **Performance**: Adjust log levels and retention policies for high-volume environments

### Debug Mode

```python
# Enable debug logging for troubleshooting
import logging
logging.getLogger('ehs_analytics').setLevel(logging.DEBUG)

# Check monitoring status
monitor = get_ehs_monitor()
health = await monitor.health_checker.get_system_health()
print(f"System health: {health}")
```

## 🚀 Production Deployment

### Recommended Production Settings

```python
# Production configuration
setup_logging(
    log_level="INFO",
    log_dir="/var/log/ehs-analytics",
    enable_file_logging=True,
    enable_console_logging=False,
    enable_rotation=True,
    max_bytes=100 * 1024 * 1024,  # 100MB
    backup_count=30,
    json_format=True,
    include_extra=False  # Reduce log size
)
```

### Log Aggregation

For production deployments, consider:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Fluentd** for log forwarding
- **Grafana Loki** for log aggregation
- **AWS CloudWatch** or **Azure Monitor**

### Monitoring Integration

- Export metrics to **Prometheus**
- Visualize with **Grafana** dashboards
- Set up **PagerDuty** or **Slack** alerts
- Integrate with **APM tools** like New Relic or DataDog

## 📚 Best Practices

### Logging Best Practices

1. **Use structured logging** with consistent field names
2. **Include context** (user_id, request_id, trace_id) in all log messages
3. **Log at appropriate levels** (DEBUG for debugging, INFO for important events, ERROR for failures)
4. **Avoid logging sensitive data** (passwords, API keys, personal information)
5. **Use performance decorators** for automatic function timing

### Monitoring Best Practices

1. **Set meaningful alert thresholds** based on SLA requirements
2. **Monitor key business metrics** (query success rate, response times)
3. **Track resource usage** to prevent system overload
4. **Implement health checks** for all critical dependencies
5. **Regular review** of metrics and alert effectiveness

### Tracing Best Practices

1. **Trace critical user journeys** end-to-end
2. **Add meaningful span names and tags**
3. **Profile memory-intensive operations**
4. **Use distributed tracing** across microservices
5. **Analyze trace data** for performance optimization

## 📞 Support

For questions or issues with the logging and monitoring system:

1. Check the logs in `./logs/` directory
2. Review the configuration in `logging_config.py`
3. Test with the example usage scripts
4. Monitor system health via the dashboard endpoints

---

The EHS Analytics logging and monitoring system provides comprehensive observability for production environments while maintaining flexibility for development and testing scenarios.