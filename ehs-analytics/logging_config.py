"""
EHS Analytics Logging Configuration Example

This file demonstrates how to configure and initialize the comprehensive
logging and monitoring system for the EHS Analytics platform.
"""

import asyncio
import os
from pathlib import Path

from src.ehs_analytics.utils.logging import setup_logging, configure_external_loggers, get_ehs_logger
from src.ehs_analytics.utils.monitoring import initialize_monitoring, get_ehs_monitor
from src.ehs_analytics.utils.tracing import get_ehs_tracer


def configure_production_logging():
    """Configure logging for production environment."""
    
    # Production logging configuration
    context_filter = setup_logging(
        log_level="INFO",
        log_dir="logs",
        enable_file_logging=True,
        enable_console_logging=True,
        enable_rotation=True,
        max_bytes=50 * 1024 * 1024,  # 50MB
        backup_count=10,
        json_format=True,
        include_extra=True
    )
    
    # Reduce noise from external libraries
    configure_external_loggers()
    
    print("✓ Production logging configured successfully")
    return context_filter


def configure_development_logging():
    """Configure logging for development environment."""
    
    # Development logging configuration
    context_filter = setup_logging(
        log_level="DEBUG",
        log_dir="logs",
        enable_file_logging=True,
        enable_console_logging=True,
        enable_rotation=False,  # No rotation in dev
        json_format=False,  # Human-readable format
        include_extra=True
    )
    
    # Keep more verbose logging for external libraries in dev
    
    print("✓ Development logging configured successfully")
    return context_filter


async def initialize_monitoring_system():
    """Initialize the monitoring and tracing system."""
    
    # Initialize monitoring
    await initialize_monitoring()
    
    # Get monitoring instances
    monitor = get_ehs_monitor()
    tracer = get_ehs_tracer("ehs-analytics")
    
    print("✓ Monitoring system initialized successfully")
    return monitor, tracer


def example_usage():
    """Example of how to use the logging system."""
    
    # Get a logger
    logger = get_ehs_logger(__name__)
    
    # Basic logging
    logger.info("Application starting", version="1.0.0", environment="production")
    logger.debug("Debug information", user_id="user123", request_id="req456")
    
    # EHS-specific logging methods
    logger.query_start("What is the water consumption at Plant A?", "consumption_analysis")
    logger.query_end("What is the water consumption at Plant A?", "consumption_analysis", 1250.5, True)
    
    # Retrieval logging
    logger.retrieval_operation("text2cypher", 15, 850.2, query_type="consumption_analysis")
    
    # Analysis logging
    logger.analysis_operation("risk_assessment", 0.85, 450.1)
    
    # Recommendation logging
    logger.recommendation_generated(3, 25000.0, facility_id="plant_a")
    
    # Security event logging
    logger.security_event("unauthorized_access", "warning", {"ip": "192.168.1.100"})
    
    # Performance metrics
    logger.performance_metric("query_response_time", 1250.5, "ms", endpoint="/api/analytics/query")
    
    print("✓ Example logging completed")


async def example_monitoring_usage():
    """Example of how to use the monitoring system."""
    
    # Get monitor
    monitor = get_ehs_monitor()
    
    # Record various metrics
    monitor.record_query("consumption_analysis", 1250.5, success=True)
    monitor.record_retrieval("text2cypher", 850.2, 15, success=True)
    monitor.record_analysis("risk_assessment", 0.85, success=True)
    monitor.record_api_request("/api/analytics/query", "POST", 1250.5, 200)
    
    # Get dashboard data
    dashboard_data = await monitor.get_dashboard_data()
    print("Dashboard data keys:", list(dashboard_data.keys()))
    
    print("✓ Example monitoring completed")


async def example_tracing_usage():
    """Example of how to use the tracing system."""
    
    # Get tracer
    tracer = get_ehs_tracer()
    
    # Create a trace
    with tracer.span("example_operation", tags={"component": "example"}):
        # Simulate some work
        await asyncio.sleep(0.1)
        
        # Create a child span
        with tracer.span("child_operation", tags={"step": "processing"}):
            await asyncio.sleep(0.05)
    
    # Get trace analytics
    from src.ehs_analytics.utils.tracing import get_trace_analytics
    analytics = get_trace_analytics()
    print("Trace analytics:", analytics)
    
    print("✓ Example tracing completed")


async def main():
    """Main configuration and example execution."""
    
    print("🚀 Configuring EHS Analytics Logging & Monitoring System")
    print("=" * 60)
    
    # Determine environment
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    # Configure logging based on environment
    if environment == "production":
        configure_production_logging()
    else:
        configure_development_logging()
    
    # Initialize monitoring system
    monitor, tracer = await initialize_monitoring_system()
    
    print("\n📊 Running Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_usage()
    await example_monitoring_usage()
    await example_tracing_usage()
    
    print("\n✅ EHS Analytics Logging & Monitoring System Ready!")
    print("=" * 60)
    print("📁 Log files location: ./logs/")
    print("📈 Monitoring dashboard data available via monitor.get_dashboard_data()")
    print("🔍 Tracing analytics available via get_trace_analytics()")
    print("🎯 Integration ready for Prometheus/Grafana monitoring tools")


if __name__ == "__main__":
    asyncio.run(main())