# Neo4j Data Population Plan for Trend Analysis and Recommendations

> Created: 2025-08-28
> Version: 1.0.0
> Status: Planning

## Overview

This document outlines a comprehensive plan to extend the existing Neo4j schema with historical time-series data, goals/targets, and analytical nodes to support dynamic LLM-based trend analysis and intelligent recommendations for the EHS Analytics Dashboard.

## Current Schema Assessment

Based on the existing `EXECUTIVE_DASHBOARD_NEO4J_SCHEMA.md`, we have:
- ✅ Location hierarchy (Global → Region → State → Site → Facility)
- ✅ MetricData nodes with basic structure
- ✅ Goal and Analysis nodes defined
- ✅ Risk assessment framework
- ⚠️ Limited historical data (sample creation only)
- ⚠️ No systematic trend analysis nodes
- ⚠️ Missing recommendation tracking

## Schema Extensions Required

### 1. Enhanced Time-Series Nodes

#### HistoricalMetric Node (Extension of MetricData)
```cypher
CREATE (:HistoricalMetric {
  id: "hm_001",
  metric_type: "electricity",
  value: 15250.75,
  unit: "kWh",
  cost: 2287.61,
  currency: "USD",
  co2_impact: 7.32,
  co2_unit: "metric_tons",
  measurement_date: date("2025-01-15"),
  measurement_period: "daily",
  data_source: "smart_meter",
  quality_score: 0.95,
  
  // Enhanced properties for trend analysis
  normalized_value: 0.73,           // Value normalized for facility size
  seasonal_factor: 1.12,           // Seasonal adjustment factor
  weather_adjusted: 14500.50,      // Weather-normalized value
  baseline_period: "2024-01",      // Reference baseline
  trend_coefficient: -0.02,        // Linear trend coefficient
  volatility_score: 0.15,          // Measure of value fluctuation
  anomaly_flag: false,             // Automated anomaly detection
  data_completeness: 0.98,         // Percentage of expected data points
  
  created_at: datetime(),
  updated_at: datetime()
})
```

#### TrendPeriod Node
```cypher
CREATE (:TrendPeriod {
  id: "tp_001",
  period_type: "quarterly",        // daily, weekly, monthly, quarterly, annual
  start_date: date("2025-01-01"),
  end_date: date("2025-03-31"),
  period_label: "Q1 2025",
  business_days: 64,
  total_days: 90,
  season: "winter",
  fiscal_period: "Q1 FY25",
  created_at: datetime(),
  updated_at: datetime()
})
```

### 2. Goal and Target Enhancement

#### EnhancedGoal Node (Extension of Goal)
```cypher
CREATE (:EnhancedGoal {
  id: "eg_001",
  metric_type: "electricity",
  target_value: 14000.0,
  target_unit: "kWh",
  target_period: "monthly",
  goal_type: "reduction",
  baseline_value: 16000.0,
  baseline_period: "2024-01",
  target_date: date("2025-12-31"),
  status: "active",
  
  // Enhanced tracking properties
  confidence_interval: [13500.0, 14500.0],  // Statistical confidence range
  achievement_probability: 0.75,             // ML-predicted probability
  historical_variance: 0.08,                 // Historical variance from targets
  adjustment_factor: 1.02,                   // Dynamic adjustment multiplier
  milestone_dates: ["2025-06-30", "2025-09-30"], // Intermediate checkpoints
  milestone_targets: [15000.0, 14500.0],     // Intermediate target values
  risk_factors: ["weather", "equipment_age"], // Factors affecting achievement
  
  created_by: "admin",
  created_at: datetime(),
  updated_at: datetime()
})
```

#### Target Node (Granular target tracking)
```cypher
CREATE (:Target {
  id: "tg_001",
  parent_goal_id: "eg_001",
  target_type: "milestone",        // milestone, threshold, stretch
  metric_type: "electricity",
  target_value: 15000.0,
  target_date: date("2025-06-30"),
  tolerance_range: [14800.0, 15200.0],
  priority: "high",               // high, medium, low
  status: "pending",              // pending, achieved, missed, at_risk
  achievement_date: null,
  actual_value: null,
  variance_percentage: null,
  created_at: datetime(),
  updated_at: datetime()
})
```

### 3. Analytical and Intelligence Nodes

#### TrendAnalysis Node
```cypher
CREATE (:TrendAnalysis {
  id: "ta_001",
  analysis_type: "time_series_decomposition",
  metric_type: "electricity",
  period_analyzed: "6_months",
  start_date: date("2024-08-01"),
  end_date: date("2025-01-31"),
  
  // Statistical analysis results
  trend_direction: "decreasing",    // increasing, decreasing, stable, volatile
  trend_strength: 0.65,            // Strength of trend (0-1)
  seasonal_pattern: "strong",      // none, weak, moderate, strong
  seasonality_period: 7,           // Period in days
  cyclical_component: true,        // Presence of cyclical patterns
  
  // Trend metrics
  linear_slope: -0.03,             // Daily change rate
  r_squared: 0.82,                 // Goodness of fit
  volatility_index: 0.12,          // Measure of variability
  autocorrelation: 0.45,           // Day-to-day correlation
  
  // Predictions
  forecast_horizon_days: 30,       // How far ahead predictions are valid
  confidence_level: 0.95,          // Statistical confidence
  forecast_accuracy: 0.78,         // Historical forecast accuracy
  
  analysis_date: date("2025-02-01"),
  analyst: "ml_system",
  model_version: "trend_v2.1",
  created_at: datetime(),
  updated_at: datetime()
})
```

#### Recommendation Node
```cypher
CREATE (:Recommendation {
  id: "rec_001",
  recommendation_type: "energy_optimization",
  category: "operational",         // operational, strategic, maintenance, behavioral
  priority: "high",               // critical, high, medium, low
  confidence_score: 0.85,         // AI confidence in recommendation
  
  // Recommendation content
  title: "Optimize HVAC Schedule During Low Occupancy",
  description: "Analysis shows 15% energy waste during non-peak hours. Implementing smart scheduling could reduce consumption.",
  action_items: [
    "Install smart thermostats in main production areas",
    "Implement occupancy-based HVAC scheduling",
    "Set up automated weekend/holiday schedules"
  ],
  
  // Impact projections
  projected_savings: 2400.0,      // Monetary savings
  projected_reduction: 850.0,     // Consumption reduction
  reduction_unit: "kWh",
  payback_period_months: 8,
  implementation_difficulty: "medium", // easy, medium, hard, complex
  
  // Tracking
  status: "pending",              // pending, approved, implemented, rejected
  implementation_date: null,
  actual_impact: null,
  roi_actual: null,
  
  // Context
  based_on_period: "2024-08-01_to_2025-01-31",
  trigger_conditions: ["high_baseline_consumption", "seasonal_variation"],
  similar_success_rate: 0.72,     // Success rate of similar recommendations
  
  created_by: "ai_system",
  reviewed_by: null,
  created_at: datetime(),
  updated_at: datetime()
})
```

#### Forecast Node
```cypher
CREATE (:Forecast {
  id: "fc_001",
  forecast_type: "consumption_prediction",
  metric_type: "electricity",
  model_type: "lstm_seasonal",
  model_version: "v1.3.2",
  
  // Forecast parameters
  forecast_date: date("2025-02-01"),
  forecast_period_start: date("2025-02-01"),
  forecast_period_end: date("2025-02-28"),
  forecast_horizon_days: 28,
  
  // Predictions
  predicted_values: [14250.5, 14180.2, 14320.8], // Daily predictions
  prediction_dates: ["2025-02-01", "2025-02-02", "2025-02-03"],
  confidence_intervals: [[13800, 14700], [13750, 14610], [13850, 14790]],
  
  // Model performance
  historical_accuracy: 0.82,      // MAPE on validation set
  bias_correction: 1.02,          // Systematic bias adjustment
  uncertainty_range: 0.15,        // Relative uncertainty
  
  // External factors considered
  weather_adjusted: true,
  holiday_adjusted: true,
  maintenance_scheduled: false,
  capacity_changes: false,
  
  created_at: datetime(),
  updated_at: datetime()
})
```

### 4. Enhanced Relationship Types

#### FOLLOWED_BY (Temporal sequence)
```cypher
// Link consecutive time periods
(period1:TrendPeriod)-[:FOLLOWED_BY {gap_days: 0}]->(period2:TrendPeriod)
```

#### CONTAINS_TREND (Period contains analysis)
```cypher
(period:TrendPeriod)-[:CONTAINS_TREND {analysis_weight: 1.0}]->(trend:TrendAnalysis)
```

#### SUPPORTS_FORECAST (Data supports prediction)
```cypher
(metric:HistoricalMetric)-[:SUPPORTS_FORECAST {weight: 0.8, lag_days: 7}]->(forecast:Forecast)
```

#### GENERATES_RECOMMENDATION (Analysis creates recommendation)
```cypher
(analysis:TrendAnalysis)-[:GENERATES_RECOMMENDATION {confidence: 0.85}]->(rec:Recommendation)
```

#### TRACKS_TARGET (Metric tracks target progress)
```cypher
(metric:HistoricalMetric)-[:TRACKS_TARGET {variance: -0.05}]->(target:Target)
```

#### INFLUENCES_METRIC (External factor affects metric)
```cypher
(factor:ExternalFactor)-[:INFLUENCES_METRIC {correlation: 0.65}]->(metric:HistoricalMetric)
```

## Historical Data Structure (6 Months)

### 1. Data Coverage Requirements

**Time Range**: August 1, 2024 - January 31, 2025 (183 days)

**Metrics to Generate**:
- **Electricity**: Daily consumption (kWh), cost (USD), CO2 impact
- **Water**: Daily consumption (gallons), cost (USD), treatment impact
- **Waste**: Daily generation (tons), disposal cost (USD), recycling rate

**Facility Coverage**:
- 5 Sites across 3 States
- 15 Facilities total (3 per site average)
- 3 Metric types per facility
- Total: 8,235 individual data points (15 facilities × 3 metrics × 183 days)

### 2. Realistic Value Generation Strategy

#### Base Consumption Patterns
```python
# Electricity (kWh/day per facility)
electricity_patterns = {
    "manufacturing": {"base": 15000, "variance": 3000, "seasonal": 0.15},
    "office": {"base": 2500, "variance": 500, "seasonal": 0.25},
    "warehouse": {"base": 8000, "variance": 1200, "seasonal": 0.10}
}

# Water (gallons/day per facility)
water_patterns = {
    "manufacturing": {"base": 5000, "variance": 1000, "seasonal": 0.08},
    "office": {"base": 800, "variance": 200, "seasonal": 0.12},
    "warehouse": {"base": 1200, "variance": 300, "seasonal": 0.05}
}

# Waste (tons/day per facility)
waste_patterns = {
    "manufacturing": {"base": 2.5, "variance": 0.8, "seasonal": 0.20},
    "office": {"base": 0.3, "variance": 0.1, "seasonal": 0.15},
    "warehouse": {"base": 1.2, "variance": 0.4, "seasonal": 0.10}
}
```

#### Temporal Patterns
```python
# Weekly patterns (Monday = 1.0, Weekend = 0.3-0.6)
weekly_multipliers = [1.0, 1.02, 0.98, 1.01, 1.03, 0.6, 0.3]

# Monthly seasonal patterns (January = 1.1, August = 0.9)
monthly_multipliers = {
    8: 0.90, 9: 0.95, 10: 1.00, 11: 1.05, 12: 1.10, 1: 1.12
}

# Holiday impacts (reduced consumption)
holiday_dates = ["2024-09-02", "2024-11-28", "2024-12-25", "2025-01-01"]
holiday_multiplier = 0.4
```

#### Data Quality Simulation
```python
# Quality score distribution
quality_ranges = {
    "excellent": (0.95, 1.00, 0.70),  # (min, max, probability)
    "good": (0.85, 0.95, 0.25),
    "fair": (0.70, 0.85, 0.04),
    "poor": (0.50, 0.70, 0.01)
}

# Missing data simulation (realistic outages)
outage_probability = 0.02  # 2% chance of missing data
outage_duration_range = (1, 3)  # 1-3 days typical outage
```

### 3. Cost and CO2 Calculation Models

#### Electricity Costs
```python
# Rate structures ($/kWh)
electricity_rates = {
    "california": {"base": 0.16, "peak_multiplier": 1.8, "off_peak_multiplier": 0.7},
    "texas": {"base": 0.12, "peak_multiplier": 1.5, "off_peak_multiplier": 0.8},
    "new_york": {"base": 0.19, "peak_multiplier": 2.0, "off_peak_multiplier": 0.6}
}

# CO2 factors (metric tons per kWh)
co2_factors = {
    "california": 0.0004,  # Cleaner grid
    "texas": 0.0008,       # Coal/gas heavy
    "new_york": 0.0005     # Mixed sources
}
```

#### Water and Waste Cost Models
```python
# Water rates ($/gallon) - tiered pricing
water_rates = {
    "tier1": 0.004,  # First 1000 gallons
    "tier2": 0.006,  # 1001-5000 gallons
    "tier3": 0.008   # >5000 gallons
}

# Waste disposal costs ($/ton)
waste_costs = {
    "landfill": 45.0,
    "recycling": -5.0,  # Revenue from recycling
    "hazardous": 180.0,
    "composting": 25.0
}
```

## Integration Points with Existing Schema

### 1. Location Hierarchy Integration
- Link `HistoricalMetric` nodes to existing `Site` and `Facility` nodes via `HAS_METRIC` relationships
- Extend existing `MetricData` nodes with historical variants
- Preserve existing location containment structure

### 2. Goal System Enhancement
- Migrate existing `Goal` nodes to `EnhancedGoal` structure
- Create `Target` nodes for each existing goal's milestones
- Link `HistoricalMetric` to `Target` nodes for progress tracking

### 3. Analysis Framework Extension
- Transform existing `Analysis` nodes to include trend analysis capabilities
- Create `TrendAnalysis` nodes for each significant time period
- Link analyses to specific metric patterns and anomalies

### 4. Recommendation System Integration
- Create `Recommendation` nodes based on existing risk assessments
- Link recommendations to trend analyses and forecasts
- Establish feedback loops for recommendation effectiveness tracking

## Data Generation Implementation Strategy

### Phase 1: Historical Data Foundation (Week 1-2)
1. **Environment Setup**
   - Create data generation Python scripts
   - Set up Neo4j connection utilities
   - Implement batch insertion optimizations

2. **Base Data Generation**
   - Generate 6 months of daily metrics for all facilities
   - Apply realistic patterns, seasonality, and noise
   - Implement data quality variations
   - Create cost and CO2 calculations

3. **Data Validation**
   - Implement statistical validation checks
   - Verify temporal consistency
   - Check location hierarchy integrity

### Phase 2: Enhanced Schema Deployment (Week 3)
1. **Schema Migration**
   - Deploy new node types and relationships
   - Create required indexes for performance
   - Implement data constraints

2. **Historical Data Import**
   - Batch import generated historical metrics
   - Create trend period nodes for analysis
   - Link metrics to existing location hierarchy

3. **Performance Optimization**
   - Optimize queries for time-series analysis
   - Implement caching strategies
   - Create aggregation views

### Phase 3: Intelligence Layer (Week 4)
1. **Trend Analysis Implementation**
   - Develop time-series decomposition algorithms
   - Generate trend analysis nodes for each facility/metric combination
   - Implement seasonal pattern detection

2. **Forecasting System**
   - Create basic forecasting models (linear, seasonal)
   - Generate forecast nodes for next 30 days
   - Implement confidence interval calculations

3. **Recommendation Engine**
   - Develop rule-based recommendation system
   - Create recommendations based on trend patterns
   - Implement recommendation tracking and feedback

## Supporting Dynamic LLM-Based Trend Analysis

### 1. Query Interface for LLM Integration

#### Trend Analysis Queries
```cypher
// Get comprehensive trend data for LLM analysis
MATCH (s:Site {id: $siteId})-[:HAS_METRIC]->(m:HistoricalMetric)
WHERE m.metric_type = $metricType 
  AND m.measurement_date >= date($startDate)
  AND m.measurement_date <= date($endDate)
OPTIONAL MATCH (m)-[:PART_OF]->(p:TrendPeriod)
OPTIONAL MATCH (p)-[:CONTAINS_TREND]->(t:TrendAnalysis)
RETURN 
  m.measurement_date as date,
  m.value as actual_value,
  m.normalized_value as normalized_value,
  m.weather_adjusted as weather_adjusted_value,
  m.seasonal_factor as seasonal_adjustment,
  m.anomaly_flag as is_anomaly,
  t.trend_direction as trend_direction,
  t.trend_strength as trend_strength,
  t.volatility_index as volatility
ORDER BY date;
```

#### Context Data for Recommendations
```cypher
// Get rich context for LLM recommendation generation
MATCH (f:Facility {id: $facilityId})
OPTIONAL MATCH (f)<-[:CONTAINS]-(s:Site)<-[:CONTAINS]-(state:State)
OPTIONAL MATCH (f)-[:HAS_METRIC]->(m:HistoricalMetric)
WHERE m.measurement_date >= date() - duration({days: 90})
OPTIONAL MATCH (f)-[:HAS_GOAL]->(g:EnhancedGoal)
OPTIONAL MATCH (g)-[:HAS_TARGET]->(tg:Target)
OPTIONAL MATCH (f)-[:HAS_RISK]->(r:Risk)
WHERE r.status = 'active'
RETURN 
  f {.*, facility_context: {
    location: {state: state.name, site: s.name},
    recent_metrics: collect(DISTINCT {
      date: m.measurement_date,
      type: m.metric_type,
      value: m.value,
      cost: m.cost,
      anomaly: m.anomaly_flag
    }),
    active_goals: collect(DISTINCT {
      type: g.metric_type,
      target: g.target_value,
      achievement_probability: g.achievement_probability,
      risk_factors: g.risk_factors
    }),
    current_targets: collect(DISTINCT {
      date: tg.target_date,
      value: tg.target_value,
      status: tg.status
    }),
    active_risks: collect(DISTINCT {
      title: r.title,
      level: r.risk_level,
      probability: r.probability
    })
  }};
```

### 2. LLM Analysis Prompt Templates

#### Trend Analysis Prompt
```python
def create_trend_analysis_prompt(facility_data):
    return f"""
    Analyze the following 6-month trend data for {facility_data['name']} ({facility_data['facility_type']}):
    
    FACILITY CONTEXT:
    - Location: {facility_data['location']}
    - Type: {facility_data['facility_type']}
    - Capacity: {facility_data['capacity']} {facility_data['capacity_unit']}
    - Operational since: {facility_data['operational_since']}
    
    METRICS DATA:
    {format_metrics_for_llm(facility_data['metrics'])}
    
    CURRENT GOALS:
    {format_goals_for_llm(facility_data['goals'])}
    
    IDENTIFIED RISKS:
    {format_risks_for_llm(facility_data['risks'])}
    
    Please provide:
    1. Key trend insights (2-3 bullet points)
    2. Seasonal patterns observed
    3. Performance vs. goals analysis
    4. Anomaly assessment
    5. 3 specific actionable recommendations with projected impact
    6. Risk factor analysis
    7. Confidence level in analysis (1-10)
    
    Format response as structured JSON for database storage.
    """
```

### 3. Real-time Analysis Integration

#### Automated Analysis Triggers
```python
# Trigger conditions for automatic LLM analysis
analysis_triggers = {
    "significant_change": {
        "threshold": 0.15,  # 15% change from baseline
        "window_days": 7,
        "min_data_points": 5
    },
    "goal_at_risk": {
        "variance_threshold": 0.20,  # 20% behind target
        "days_to_target": 30
    },
    "anomaly_cluster": {
        "anomaly_count": 3,
        "time_window_days": 5
    },
    "scheduled_review": {
        "frequency": "weekly",  # Weekly comprehensive reviews
        "depth": "full_analysis"
    }
}
```

#### Analysis Results Storage
```cypher
// Store LLM analysis results
CREATE (analysis:LLMAnalysis {
  id: "llm_analysis_001",
  facility_id: "fac_001",
  analysis_date: date("2025-02-01"),
  trigger_type: "significant_change",
  
  // LLM analysis results
  key_insights: [
    "15% reduction in electricity usage during off-peak hours",
    "Strong weekly seasonality with 40% lower weekend consumption",
    "Minor temperature correlation affecting 8% of variance"
  ],
  
  seasonal_patterns: {
    "weekly": "strong_business_hours_pattern",
    "monthly": "moderate_weather_correlation",
    "holiday": "significant_reduction_observed"
  },
  
  goal_performance: {
    "electricity": {"status": "ahead", "variance": -0.08},
    "water": {"status": "on_track", "variance": 0.02}
  },
  
  anomalies_detected: [
    {
      "date": "2024-12-15",
      "type": "spike",
      "magnitude": 2.3,
      "likely_cause": "equipment_malfunction"
    }
  ],
  
  recommendations: [
    {
      "priority": "high",
      "action": "Install smart power strips in office areas",
      "projected_savings": 1200.0,
      "implementation_effort": "low"
    }
  ],
  
  confidence_score: 8.5,
  model_version: "gpt-4-analysis-v1.2",
  processing_time_seconds: 45.2,
  
  created_at: datetime(),
  updated_at: datetime()
})
```

## Implementation Timeline and Milestones

### Week 1: Foundation Setup
- [ ] Create data generation scripts
- [ ] Implement historical data patterns
- [ ] Set up Neo4j connection utilities
- [ ] Generate 6 months of realistic test data

### Week 2: Schema Enhancement
- [ ] Deploy enhanced node types
- [ ] Create relationship definitions
- [ ] Implement performance indexes
- [ ] Migrate existing data to new structure

### Week 3: Data Import and Validation
- [ ] Batch import historical metrics
- [ ] Create trend period nodes
- [ ] Implement data validation checks
- [ ] Performance optimization

### Week 4: Intelligence Layer
- [ ] Implement trend analysis algorithms
- [ ] Create forecasting system
- [ ] Develop recommendation engine
- [ ] Set up LLM integration interfaces

### Week 5: Testing and Optimization
- [ ] End-to-end testing
- [ ] Performance benchmarking
- [ ] LLM prompt optimization
- [ ] Documentation completion

## Success Metrics

### Data Quality Metrics
- **Completeness**: >95% of expected data points present
- **Consistency**: <2% variance in temporal patterns
- **Accuracy**: Generated patterns match real-world expectations
- **Performance**: Query response times <2 seconds for dashboard queries

### Analysis Quality Metrics
- **Trend Detection**: Successfully identify 90% of significant trends
- **Forecast Accuracy**: MAPE <15% for 30-day forecasts
- **Recommendation Relevance**: >80% of recommendations rated as actionable
- **LLM Integration**: <5 second response time for analysis queries

### System Performance Metrics
- **Data Volume**: Successfully handle 10k+ metric points per facility
- **Query Performance**: Complex aggregation queries <3 seconds
- **Scalability**: Support for 100+ facilities without degradation
- **Reliability**: 99.5% uptime for data ingestion pipeline

## Risk Mitigation

### Data Risks
- **Data Quality Issues**: Implement comprehensive validation and monitoring
- **Performance Degradation**: Use staged rollout with performance monitoring
- **Storage Growth**: Implement data archiving strategies for old metrics

### Technical Risks
- **Neo4j Performance**: Optimize indexes and query patterns early
- **LLM Integration**: Build fallback analysis methods for LLM failures
- **Schema Changes**: Version schema changes and maintain backward compatibility

### Operational Risks
- **User Adoption**: Provide clear documentation and training materials
- **Maintenance Overhead**: Automate data generation and validation processes
- **Cost Management**: Monitor cloud resources and optimize usage patterns

This comprehensive plan provides a roadmap for transforming the EHS Analytics platform into an intelligent, trend-aware system capable of supporting sophisticated LLM-based analysis and recommendations while maintaining the robust foundation already established.