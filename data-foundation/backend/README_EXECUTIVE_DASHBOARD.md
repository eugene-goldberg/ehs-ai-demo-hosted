# Executive Dashboard Service

A comprehensive, production-ready executive dashboard service for the EHS AI Demo Data Foundation. This service integrates with Neo4j to provide real-time KPI monitoring, trend analysis, risk assessment, and dynamic dashboard JSON generation.

## Features

### Core Functionality
- **Real-time KPI Monitoring**: Live tracking of safety and compliance metrics
- **Historical Trend Analysis**: Statistical analysis with anomaly detection
- **Risk Assessment Integration**: Leverages existing risk models for recommendations
- **Dynamic JSON Generation**: Flexible dashboard data structure
- **Advanced Filtering**: Location and date range filtering capabilities
- **Comprehensive Error Handling**: Production-ready error management
- **Performance Optimization**: Built-in caching and monitoring

### Analytics Integration
- **Neo4j Integration**: Direct access to real EHS data
- **Trend Analysis System**: Leverages statistical models for pattern detection
- **Recommendation Engine**: AI-driven recommendations based on current data
- **Forecasting Support**: Framework for predictive analytics (extensible)

### Dashboard Components
- **Summary Metrics**: High-level organizational health indicators
- **KPI Analysis**: Detailed safety and compliance metrics with benchmarking
- **Interactive Charts**: Multiple chart types for data visualization
- **Alert Management**: Real-time alerting with severity levels
- **Trend Insights**: Historical pattern analysis with LLM-ready formatting
- **Recommendations**: Both stored and AI-generated recommendations
- **System Status**: Health monitoring and data quality assessment

## Installation & Setup

### Prerequisites
- Python 3.8+
- Neo4j Database (running and accessible)
- Required environment variables configured

### Environment Configuration

Ensure your `.env` file contains the following:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Optional: Additional API keys for enhanced functionality
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langchain_key
```

### Installation

1. **Install Dependencies** (if not already installed):
   ```bash
   pip install neo4j pandas python-dotenv pydantic
   ```

2. **Verify Neo4j Connection**:
   ```bash
   # Test connection to ensure database is accessible
   python -c "from neo4j import GraphDatabase; print('Neo4j connection test:', GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'your_password')).verify_connectivity())"
   ```

## Usage

### Basic Usage

```python
from services.executive_dashboard.dashboard_service import create_dashboard_service

# Create dashboard service
dashboard_service = create_dashboard_service()

try:
    # Generate basic dashboard
    dashboard_data = dashboard_service.generate_executive_dashboard()
    
    # Get real-time metrics
    real_time_data = dashboard_service.get_real_time_metrics()
    
    # Get detailed KPIs
    kpi_details = dashboard_service.get_kpi_details(date_range_days=30)
    
finally:
    dashboard_service.close()
```

### Advanced Usage with Filtering

```python
from services.executive_dashboard.dashboard_service import (
    create_dashboard_service, LocationFilter, DateRangeFilter, AggregationPeriod
)
from datetime import datetime, timedelta

dashboard_service = create_dashboard_service()

try:
    # Define filters
    location_filter = LocationFilter(facility_ids=["FAC001", "FAC002"])
    date_filter = DateRangeFilter(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        period=AggregationPeriod.WEEKLY
    )
    
    # Generate filtered dashboard with all features
    comprehensive_dashboard = dashboard_service.generate_executive_dashboard(
        location_filter=location_filter,
        date_filter=date_filter,
        include_trends=True,
        include_recommendations=True,
        include_forecasts=True
    )
    
    # Process dashboard data
    if 'error' not in comprehensive_dashboard:
        health_score = comprehensive_dashboard['summary']['overall_health_score']
        alert_level = comprehensive_dashboard['alerts']['summary']['alert_level']
        
        print(f"Health Score: {health_score}")
        print(f"Alert Level: {alert_level}")
    
finally:
    dashboard_service.close()
```

## Testing

### Run Comprehensive Tests
```bash
python test_executive_dashboard.py
```

### Run Usage Examples
```bash
python example_dashboard_usage.py
```

## API Reference

### Core Methods

#### `generate_executive_dashboard()`
Generates comprehensive executive dashboard data with optional filtering and features.

**Parameters:**
- `location_filter` (Optional[LocationFilter]): Location-based filtering
- `date_filter` (Optional[DateRangeFilter]): Date range filtering  
- `include_trends` (bool): Include trend analysis (default: True)
- `include_recommendations` (bool): Include recommendations (default: True)
- `include_forecasts` (bool): Include forecasting data (default: False)

**Returns:** Dict containing complete dashboard JSON structure

#### `get_real_time_metrics(facility_ids=None)`
Retrieves current real-time metrics for immediate dashboard updates.

**Parameters:**
- `facility_ids` (Optional[List[str]]): Filter by specific facilities

**Returns:** Dict with current metrics, alert level, and facility breakdown

#### `get_kpi_details(facility_ids=None, date_range_days=30)`
Gets detailed KPI information with performance analysis.

**Parameters:**
- `facility_ids` (Optional[List[str]]): Filter by specific facilities
- `date_range_days` (int): Number of days for analysis period

**Returns:** Dict with safety and compliance KPIs, targets, and status

#### `health_check()`
Performs comprehensive system health check.

**Returns:** Dict with database, service, and system health status

### Filtering Classes

#### `LocationFilter`
```python
@dataclass
class LocationFilter:
    facility_ids: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    departments: Optional[List[str]] = None
```

#### `DateRangeFilter`
```python
@dataclass  
class DateRangeFilter:
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    period: AggregationPeriod = AggregationPeriod.MONTHLY
```

## Dashboard JSON Structure

The service generates a comprehensive JSON structure with the following components:

```json
{
  "metadata": {
    "generated_at": "ISO timestamp",
    "version": "1.0.0",
    "filters": {...}
  },
  "summary": {
    "period": {...},
    "facilities": {...},
    "incidents": {...},
    "compliance": {...},
    "overall_health_score": 85.2
  },
  "kpis": {
    "summary": {...},
    "metrics": {...},
    "benchmarks": {...}
  },
  "charts": {
    "incident_trend": {...},
    "facility_performance": {...},
    "kpi_status_distribution": {...}
  },
  "alerts": {
    "summary": {...},
    "recent_alerts": [...],
    "escalations": [...]
  },
  "trends": {
    "metric_trends": {...},
    "recent_anomalies": [...],
    "trend_summary": {...}
  },
  "recommendations": {
    "summary": {...},
    "ai_generated_recommendations": [...],
    "analytics": {...}
  },
  "status": {
    "overall_status": "healthy",
    "system_health": {...},
    "data_quality": {...}
  }
}
```

## Performance Considerations

### Caching
- Built-in result caching with configurable TTL
- Cache hit rates monitored for optimization
- Automatic cache invalidation for real-time data

### Database Optimization
- Optimized Neo4j queries with proper indexing
- Connection pooling for concurrent access
- Query result pagination for large datasets

### Error Handling
- Comprehensive error handling with graceful degradation
- Detailed error logging for troubleshooting
- Fallback values for missing data

## Monitoring & Observability

### Health Checks
```python
health_status = dashboard_service.health_check()
```

### Performance Metrics
- Request count and error rate tracking
- Query execution time monitoring
- Cache performance statistics

### Logging
Configured logging levels:
- INFO: Normal operations and key metrics
- WARNING: Non-fatal issues and degraded performance
- ERROR: Failures and exceptions
- DEBUG: Detailed execution information

## Integration Examples

### Web API Integration
```python
from flask import Flask, jsonify
from services.executive_dashboard.dashboard_service import create_dashboard_service

app = Flask(__name__)
dashboard_service = create_dashboard_service()

@app.route('/api/dashboard')
def get_dashboard():
    try:
        dashboard_data = dashboard_service.generate_executive_dashboard()
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/realtime')
def get_realtime():
    real_time_data = dashboard_service.get_real_time_metrics()
    return jsonify(real_time_data)
```

### Scheduled Reporting
```python
import schedule
import time

def generate_daily_report():
    dashboard_service = create_dashboard_service()
    try:
        dashboard_data = dashboard_service.generate_executive_dashboard()
        
        # Save report or send notifications
        timestamp = datetime.now().strftime('%Y%m%d')
        with open(f'daily_report_{timestamp}.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2)
            
    finally:
        dashboard_service.close()

# Schedule daily report at 6 AM
schedule.every().day.at("06:00").do(generate_daily_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Common Issues

**1. Neo4j Connection Issues**
```bash
# Check Neo4j service status
sudo systemctl status neo4j

# Test connection
python -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')).verify_connectivity()"
```

**2. Missing Data**
- Verify data has been loaded into Neo4j
- Check facility IDs in location filters
- Confirm date ranges contain data

**3. Performance Issues**
- Check Neo4j query performance with `EXPLAIN` and `PROFILE`
- Monitor cache hit rates
- Review log files for slow queries

**4. Memory Issues**
```python
# Clear cache if needed
dashboard_service.clear_cache()
```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Production Deployment

### Configuration Checklist
- [ ] Neo4j connection parameters configured
- [ ] Environment variables set correctly  
- [ ] Database indexes created and optimized
- [ ] Monitoring and alerting configured
- [ ] Backup procedures in place
- [ ] Security measures implemented

### Security Considerations
- Use environment variables for sensitive data
- Implement proper authentication/authorization
- Enable Neo4j security features
- Regular security updates and patches

### Scaling Considerations
- Neo4j clustering for high availability
- Horizontal scaling for dashboard service
- Load balancing for multiple instances
- Caching strategies for high-traffic scenarios

## Support & Maintenance

For questions or issues:
1. Check the troubleshooting section above
2. Review log files for error details
3. Test with the provided example scripts
4. Verify Neo4j database connectivity and data

## Version History

- **1.0.0** (2025-08-28): Initial release with comprehensive dashboard functionality

---

This executive dashboard service provides a robust foundation for EHS monitoring and reporting, with production-ready features and extensive customization options.