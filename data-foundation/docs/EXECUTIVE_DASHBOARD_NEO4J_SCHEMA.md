# Executive Dashboard Neo4j Schema

> Last Updated: 2025-08-27
> Version: 1.0.0

## Overview

This document defines the comprehensive Neo4j schema for storing and retrieving all data points required for the Analytics (Executive) dashboard. The schema supports hierarchical location data, time-series metrics, goals/targets, cost calculations, and risk assessments.

## Schema Architecture

### Core Design Principles

1. **Hierarchical Organization**: Global → Region → State → Site → Facility
2. **Time-Series Data**: Timestamped metrics with flexible aggregation
3. **Goal Tracking**: Targets vs. actual performance with variance analysis
4. **Multi-Dimensional Metrics**: Electricity, water, waste with costs and CO2 impact
5. **Risk Assessment**: Location-based and metric-based risk scoring
6. **Audit Trail**: Complete traceability of data sources and calculations

## Node Types

### 1. Location Hierarchy Nodes

#### Global Node
```cypher
CREATE (:Global {
  id: "global",
  name: "Global Operations",
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique identifier
- `name` (String): Display name
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

#### Region Node
```cypher
CREATE (:Region {
  id: "na",
  name: "North America",
  code: "NA",
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique region identifier
- `name` (String): Full region name
- `code` (String): Region abbreviation
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

#### State Node
```cypher
CREATE (:State {
  id: "ca",
  name: "California",
  code: "CA",
  country: "USA",
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique state identifier
- `name` (String): Full state name
- `code` (String): State abbreviation
- `country` (String): Country code
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

#### Site Node
```cypher
CREATE (:Site {
  id: "site_001",
  name: "San Francisco Manufacturing",
  code: "SFM001",
  address: "123 Industrial Blvd, San Francisco, CA 94103",
  latitude: 37.7749,
  longitude: -122.4194,
  site_type: "Manufacturing",
  operational_since: date("2015-01-01"),
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique site identifier
- `name` (String): Site display name
- `code` (String): Site code
- `address` (String): Physical address
- `latitude` (Float): GPS coordinate
- `longitude` (Float): GPS coordinate
- `site_type` (String): Type of facility
- `operational_since` (Date): Start of operations
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

#### Facility Node
```cypher
CREATE (:Facility {
  id: "fac_001",
  name: "Main Production Floor",
  code: "MPF001",
  facility_type: "Production",
  square_footage: 50000,
  capacity: 1000,
  capacity_unit: "units_per_day",
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique facility identifier
- `name` (String): Facility name
- `code` (String): Facility code
- `facility_type` (String): Type of facility
- `square_footage` (Integer): Floor area
- `capacity` (Float): Production capacity
- `capacity_unit` (String): Unit of capacity measurement
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

### 2. Time-Series Data Nodes

#### MetricData Node
```cypher
CREATE (:MetricData {
  id: "md_001",
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
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique metric data identifier
- `metric_type` (String): Type of metric (electricity, water, waste)
- `value` (Float): Measured value
- `unit` (String): Unit of measurement
- `cost` (Float): Associated cost
- `currency` (String): Currency code
- `co2_impact` (Float): CO2 impact value
- `co2_unit` (String): CO2 measurement unit
- `measurement_date` (Date): Date of measurement
- `measurement_period` (String): Period type (daily, weekly, monthly)
- `data_source` (String): Source of the data
- `quality_score` (Float): Data quality indicator (0-1)
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

### 3. Goals and Targets Nodes

#### Goal Node
```cypher
CREATE (:Goal {
  id: "goal_001",
  metric_type: "electricity",
  target_value: 14000.0,
  target_unit: "kWh",
  target_period: "monthly",
  goal_type: "reduction",
  baseline_value: 16000.0,
  baseline_period: "2024-01",
  target_date: date("2025-12-31"),
  status: "active",
  created_by: "admin",
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique goal identifier
- `metric_type` (String): Type of metric
- `target_value` (Float): Target value to achieve
- `target_unit` (String): Unit of measurement
- `target_period` (String): Period for target
- `goal_type` (String): Type of goal (reduction, increase, maintain)
- `baseline_value` (Float): Starting baseline value
- `baseline_period` (String): Baseline period reference
- `target_date` (Date): Target completion date
- `status` (String): Goal status (active, achieved, suspended)
- `created_by` (String): User who created the goal
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

### 4. Analysis and Recommendations Nodes

#### Analysis Node
```cypher
CREATE (:Analysis {
  id: "analysis_001",
  analysis_type: "performance_review",
  title: "Q1 Energy Performance Analysis",
  description: "Quarterly analysis of energy consumption patterns",
  key_findings: ["10% reduction in peak usage", "Improved efficiency in Building A"],
  recommendations: ["Install smart thermostats", "Optimize HVAC schedules"],
  confidence_score: 0.85,
  analysis_date: date("2025-01-31"),
  analyst: "energy_team",
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique analysis identifier
- `analysis_type` (String): Type of analysis
- `title` (String): Analysis title
- `description` (String): Detailed description
- `key_findings` (List[String]): List of key findings
- `recommendations` (List[String]): List of recommendations
- `confidence_score` (Float): Confidence in analysis (0-1)
- `analysis_date` (Date): Date analysis was performed
- `analyst` (String): Who performed the analysis
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

### 5. Risk Assessment Nodes

#### Risk Node
```cypher
CREATE (:Risk {
  id: "risk_001",
  risk_type: "operational",
  category: "energy_supply",
  title: "Grid Reliability Risk",
  description: "Potential energy supply disruption during peak seasons",
  probability: 0.3,
  impact_score: 8,
  risk_level: "medium",
  mitigation_actions: ["Install backup generators", "Negotiate supply contracts"],
  owner: "facilities_team",
  review_date: date("2025-03-01"),
  status: "active",
  created_at: datetime(),
  updated_at: datetime()
})
```

**Properties:**
- `id` (String): Unique risk identifier
- `risk_type` (String): Type of risk
- `category` (String): Risk category
- `title` (String): Risk title
- `description` (String): Detailed description
- `probability` (Float): Probability of occurrence (0-1)
- `impact_score` (Integer): Impact severity (1-10)
- `risk_level` (String): Overall risk level (low, medium, high, critical)
- `mitigation_actions` (List[String]): Planned mitigation actions
- `owner` (String): Risk owner/responsible party
- `review_date` (Date): Next review date
- `status` (String): Risk status (active, mitigated, closed)
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp

## Relationship Types

### 1. Hierarchical Relationships

#### CONTAINS
- Used for location hierarchy: Global→Region→State→Site→Facility
- Properties: `created_at` (DateTime)

```cypher
// Examples
(global:Global)-[:CONTAINS {created_at: datetime()}]->(region:Region)
(region:Region)-[:CONTAINS {created_at: datetime()}]->(state:State)
(state:State)-[:CONTAINS {created_at: datetime()}]->(site:Site)
(site:Site)-[:CONTAINS {created_at: datetime()}]->(facility:Facility)
```

### 2. Data Relationships

#### HAS_METRIC
- Links locations to their metric data
- Properties: `created_at` (DateTime)

```cypher
(site:Site)-[:HAS_METRIC {created_at: datetime()}]->(metric:MetricData)
(facility:Facility)-[:HAS_METRIC {created_at: datetime()}]->(metric:MetricData)
```

#### HAS_GOAL
- Links locations to their goals/targets
- Properties: `created_at` (DateTime)

```cypher
(site:Site)-[:HAS_GOAL {created_at: datetime()}]->(goal:Goal)
(facility:Facility)-[:HAS_GOAL {created_at: datetime()}]->(goal:Goal)
```

### 3. Analysis Relationships

#### ANALYZED_BY
- Links data to analysis results
- Properties: `weight` (Float), `created_at` (DateTime)

```cypher
(metric:MetricData)-[:ANALYZED_BY {weight: 1.0, created_at: datetime()}]->(analysis:Analysis)
```

#### TARGETS
- Links goals to specific locations
- Properties: `priority` (String), `created_at` (DateTime)

```cypher
(goal:Goal)-[:TARGETS {priority: "high", created_at: datetime()}]->(site:Site)
```

### 4. Risk Relationships

#### HAS_RISK
- Links locations to risk assessments
- Properties: `severity` (String), `created_at` (DateTime)

```cypher
(site:Site)-[:HAS_RISK {severity: "medium", created_at: datetime()}]->(risk:Risk)
```

#### IMPACTS
- Links risks to metrics they might affect
- Properties: `impact_type` (String), `created_at` (DateTime)

```cypher
(risk:Risk)-[:IMPACTS {impact_type: "cost_increase", created_at: datetime()}]->(metric:MetricData)
```

## Essential Indexes

### Primary Indexes
```cypher
// Location hierarchy indexes
CREATE INDEX location_id_idx FOR (n:Global) ON (n.id);
CREATE INDEX region_id_idx FOR (n:Region) ON (n.id);
CREATE INDEX state_id_idx FOR (n:State) ON (n.id);
CREATE INDEX site_id_idx FOR (n:Site) ON (n.id);
CREATE INDEX facility_id_idx FOR (n:Facility) ON (n.id);

// Metric data indexes
CREATE INDEX metric_id_idx FOR (n:MetricData) ON (n.id);
CREATE INDEX metric_type_date_idx FOR (n:MetricData) ON (n.metric_type, n.measurement_date);
CREATE INDEX metric_date_idx FOR (n:MetricData) ON (n.measurement_date);

// Goal indexes
CREATE INDEX goal_id_idx FOR (n:Goal) ON (n.id);
CREATE INDEX goal_type_status_idx FOR (n:Goal) ON (n.metric_type, n.status);

// Analysis and Risk indexes
CREATE INDEX analysis_id_idx FOR (n:Analysis) ON (n.id);
CREATE INDEX analysis_date_idx FOR (n:Analysis) ON (n.analysis_date);
CREATE INDEX risk_id_idx FOR (n:Risk) ON (n.id);
CREATE INDEX risk_level_idx FOR (n:Risk) ON (n.risk_level, n.status);
```

### Composite Indexes
```cypher
// Performance optimization indexes
CREATE INDEX site_metric_date_idx FOR (n:MetricData) ON (n.metric_type, n.measurement_date, n.measurement_period);
CREATE INDEX location_hierarchy_idx FOR (n:Site) ON (n.id, n.site_type);
```

## Core Query Patterns

### 1. Location Hierarchy Queries

#### Get Complete Hierarchy
```cypher
MATCH (g:Global)-[:CONTAINS*]->(location)
RETURN g, location
ORDER BY labels(location), location.name;
```

#### Get Sites in a Region
```cypher
MATCH (r:Region {id: $regionId})-[:CONTAINS*]->(s:Site)
RETURN s
ORDER BY s.name;
```

#### Get Full Path for a Site
```cypher
MATCH path = (g:Global)-[:CONTAINS*]->(s:Site {id: $siteId})
RETURN [node IN nodes(path) | {
  id: node.id,
  name: node.name,
  type: labels(node)[0]
}] as hierarchy;
```

### 2. Time-Series Data Queries

#### Get Recent Metrics for Site
```cypher
MATCH (s:Site {id: $siteId})-[:HAS_METRIC]->(m:MetricData)
WHERE m.measurement_date >= date($startDate)
  AND m.measurement_date <= date($endDate)
  AND m.metric_type = $metricType
RETURN m
ORDER BY m.measurement_date DESC;
```

#### Get Daily Aggregations for Region
```cypher
MATCH (r:Region {id: $regionId})-[:CONTAINS*]->(location)-[:HAS_METRIC]->(m:MetricData)
WHERE m.measurement_date >= date($startDate)
  AND m.measurement_date <= date($endDate)
  AND m.metric_type = $metricType
RETURN 
  m.measurement_date as date,
  sum(m.value) as total_value,
  sum(m.cost) as total_cost,
  sum(m.co2_impact) as total_co2,
  count(m) as measurement_count,
  avg(m.quality_score) as avg_quality
ORDER BY date;
```

#### Get Year-over-Year Comparison
```cypher
MATCH (s:Site {id: $siteId})-[:HAS_METRIC]->(m:MetricData)
WHERE m.metric_type = $metricType
  AND (
    (m.measurement_date >= date($currentYearStart) AND m.measurement_date <= date($currentYearEnd)) OR
    (m.measurement_date >= date($previousYearStart) AND m.measurement_date <= date($previousYearEnd))
  )
WITH 
  CASE 
    WHEN m.measurement_date >= date($currentYearStart) THEN 'current'
    ELSE 'previous'
  END as year_period,
  m.measurement_date.month as month,
  m
RETURN 
  month,
  year_period,
  sum(m.value) as total_value,
  sum(m.cost) as total_cost,
  sum(m.co2_impact) as total_co2
ORDER BY month, year_period;
```

### 3. Goal Tracking Queries

#### Get Goal Performance by Site
```cypher
MATCH (s:Site {id: $siteId})-[:HAS_GOAL]->(g:Goal)
WHERE g.status = 'active'
OPTIONAL MATCH (s)-[:HAS_METRIC]->(m:MetricData)
WHERE m.metric_type = g.metric_type
  AND m.measurement_date >= date($periodStart)
  AND m.measurement_date <= date($periodEnd)
WITH g, s, sum(m.value) as actual_value, sum(m.cost) as actual_cost
RETURN 
  g.id as goal_id,
  g.metric_type,
  g.target_value,
  actual_value,
  actual_cost,
  CASE 
    WHEN actual_value IS NOT NULL THEN (g.target_value - actual_value) / g.target_value * 100
    ELSE NULL
  END as variance_percentage,
  CASE
    WHEN actual_value IS NOT NULL AND actual_value <= g.target_value THEN 'on_track'
    WHEN actual_value IS NOT NULL AND actual_value > g.target_value THEN 'behind'
    ELSE 'no_data'
  END as status;
```

#### Get Regional Goal Summary
```cypher
MATCH (r:Region {id: $regionId})-[:CONTAINS*]->(location)-[:HAS_GOAL]->(g:Goal)
WHERE g.status = 'active'
OPTIONAL MATCH (location)-[:HAS_METRIC]->(m:MetricData)
WHERE m.metric_type = g.metric_type
  AND m.measurement_date >= date($periodStart)
  AND m.measurement_date <= date($periodEnd)
WITH g.metric_type as metric_type, 
     sum(g.target_value) as total_target,
     sum(m.value) as total_actual
RETURN 
  metric_type,
  total_target,
  total_actual,
  CASE 
    WHEN total_actual IS NOT NULL THEN (total_target - total_actual) / total_target * 100
    ELSE NULL
  END as variance_percentage;
```

### 4. Cost and CO2 Analysis Queries

#### Get Cost Breakdown by Location Type
```cypher
MATCH (location)-[:HAS_METRIC]->(m:MetricData)
WHERE m.measurement_date >= date($startDate)
  AND m.measurement_date <= date($endDate)
WITH labels(location)[0] as location_type, 
     m.metric_type,
     sum(m.cost) as total_cost,
     sum(m.co2_impact) as total_co2
RETURN location_type, metric_type, total_cost, total_co2
ORDER BY location_type, metric_type;
```

#### Get Top Cost Contributors
```cypher
MATCH (s:Site)-[:HAS_METRIC]->(m:MetricData)
WHERE m.measurement_date >= date($startDate)
  AND m.measurement_date <= date($endDate)
WITH s, sum(m.cost) as total_cost
ORDER BY total_cost DESC
LIMIT $limit
RETURN s.id, s.name, total_cost;
```

### 5. Risk Assessment Queries

#### Get Active Risks by Location
```cypher
MATCH (location)-[:HAS_RISK]->(r:Risk)
WHERE r.status = 'active'
RETURN 
  labels(location)[0] as location_type,
  location.name as location_name,
  r.risk_level,
  r.title,
  r.probability,
  r.impact_score,
  r.probability * r.impact_score as risk_score
ORDER BY risk_score DESC;
```

#### Get Risk Impact on Metrics
```cypher
MATCH (r:Risk)-[:IMPACTS]->(m:MetricData)
WHERE r.status = 'active'
  AND m.measurement_date >= date($startDate)
WITH r, m.metric_type, sum(m.cost) as impacted_cost
RETURN 
  r.title as risk_title,
  r.risk_level,
  collect({
    metric_type: m.metric_type,
    impacted_cost: impacted_cost
  }) as metric_impacts
ORDER BY r.risk_level, r.title;
```

## Data Creation Scripts

### 1. Create Location Hierarchy
```cypher
// Create Global node
MERGE (global:Global {id: "global"})
SET global.name = "Global Operations",
    global.created_at = datetime(),
    global.updated_at = datetime();

// Create Regions
MERGE (na:Region {id: "na"})
SET na.name = "North America", na.code = "NA",
    na.created_at = datetime(), na.updated_at = datetime();

MERGE (eu:Region {id: "eu"})
SET eu.name = "Europe", eu.code = "EU",
    eu.created_at = datetime(), eu.updated_at = datetime();

// Create States
MERGE (ca:State {id: "ca"})
SET ca.name = "California", ca.code = "CA", ca.country = "USA",
    ca.created_at = datetime(), ca.updated_at = datetime();

MERGE (tx:State {id: "tx"})
SET tx.name = "Texas", tx.code = "TX", tx.country = "USA",
    tx.created_at = datetime(), tx.updated_at = datetime();

// Create Sites
MERGE (site1:Site {id: "site_001"})
SET site1.name = "San Francisco Manufacturing",
    site1.code = "SFM001",
    site1.address = "123 Industrial Blvd, San Francisco, CA 94103",
    site1.latitude = 37.7749,
    site1.longitude = -122.4194,
    site1.site_type = "Manufacturing",
    site1.operational_since = date("2015-01-01"),
    site1.created_at = datetime(),
    site1.updated_at = datetime();

// Create relationships
MERGE (global)-[:CONTAINS {created_at: datetime()}]->(na);
MERGE (na)-[:CONTAINS {created_at: datetime()}]->(ca);
MERGE (ca)-[:CONTAINS {created_at: datetime()}]->(site1);
```

### 2. Create Sample Metrics Data
```cypher
// Create electricity metrics
UNWIND range(1, 30) as day
CREATE (m:MetricData {
  id: "elec_" + toString(day),
  metric_type: "electricity",
  value: toFloat(12000 + (rand() * 6000)),
  unit: "kWh",
  cost: toFloat(1800 + (rand() * 900)),
  currency: "USD",
  co2_impact: toFloat(5.8 + (rand() * 2.9)),
  co2_unit: "metric_tons",
  measurement_date: date("2025-01-" + lpad(toString(day), 2, "0")),
  measurement_period: "daily",
  data_source: "smart_meter",
  quality_score: 0.90 + (rand() * 0.10),
  created_at: datetime(),
  updated_at: datetime()
});

// Link metrics to site
MATCH (s:Site {id: "site_001"})
MATCH (m:MetricData)
WHERE m.metric_type = "electricity"
CREATE (s)-[:HAS_METRIC {created_at: datetime()}]->(m);
```

### 3. Create Goals and Targets
```cypher
// Create electricity reduction goal
CREATE (g:Goal {
  id: "goal_elec_001",
  metric_type: "electricity",
  target_value: 350000.0,
  target_unit: "kWh",
  target_period: "monthly",
  goal_type: "reduction",
  baseline_value: 400000.0,
  baseline_period: "2024-12",
  target_date: date("2025-12-31"),
  status: "active",
  created_by: "admin",
  created_at: datetime(),
  updated_at: datetime()
});

// Link goal to site
MATCH (s:Site {id: "site_001"}), (g:Goal {id: "goal_elec_001"})
CREATE (s)-[:HAS_GOAL {created_at: datetime()}]->(g);
```

## Advanced Aggregation Queries

### 1. Rolling Averages
```cypher
MATCH (s:Site {id: $siteId})-[:HAS_METRIC]->(m:MetricData)
WHERE m.metric_type = $metricType
  AND m.measurement_date >= date($startDate)
  AND m.measurement_date <= date($endDate)
WITH m ORDER BY m.measurement_date
WITH m, 
     collect(m.value)[..7] as last_7_values,
     collect(m.value)[..30] as last_30_values
RETURN 
  m.measurement_date,
  m.value,
  CASE WHEN size(last_7_values) = 7 THEN reduce(sum=0.0, val IN last_7_values | sum + val) / 7.0 END as rolling_7_avg,
  CASE WHEN size(last_30_values) = 30 THEN reduce(sum=0.0, val IN last_30_values | sum + val) / 30.0 END as rolling_30_avg;
```

### 2. Benchmark Comparisons
```cypher
// Compare site performance against regional average
MATCH (r:Region {id: $regionId})-[:CONTAINS*]->(s:Site)-[:HAS_METRIC]->(m:MetricData)
WHERE m.metric_type = $metricType
  AND m.measurement_date >= date($startDate)
  AND m.measurement_date <= date($endDate)
WITH s, avg(m.value) as site_avg, avg(m.cost) as site_cost
WITH collect({site: s, avg_value: site_avg, avg_cost: site_cost}) as sites,
     avg(site_avg) as regional_avg,
     avg(site_cost) as regional_cost
UNWIND sites as site_data
RETURN 
  site_data.site.id,
  site_data.site.name,
  site_data.avg_value,
  regional_avg,
  (site_data.avg_value - regional_avg) / regional_avg * 100 as variance_from_regional_avg,
  CASE 
    WHEN site_data.avg_value <= regional_avg * 0.9 THEN 'excellent'
    WHEN site_data.avg_value <= regional_avg * 1.1 THEN 'good'
    WHEN site_data.avg_value <= regional_avg * 1.3 THEN 'needs_improvement'
    ELSE 'poor'
  END as performance_rating;
```

### 3. Trend Analysis
```cypher
// Calculate month-over-month growth rates
MATCH (s:Site {id: $siteId})-[:HAS_METRIC]->(m:MetricData)
WHERE m.metric_type = $metricType
  AND m.measurement_date >= date($startDate)
WITH 
  m.measurement_date.year as year,
  m.measurement_date.month as month,
  sum(m.value) as monthly_total,
  sum(m.cost) as monthly_cost
ORDER BY year, month
WITH collect({
  year: year, 
  month: month, 
  total: monthly_total, 
  cost: monthly_cost
}) as monthly_data
UNWIND range(1, size(monthly_data)-1) as i
WITH monthly_data[i-1] as prev_month, monthly_data[i] as curr_month
RETURN 
  curr_month.year,
  curr_month.month,
  curr_month.total,
  prev_month.total,
  (curr_month.total - prev_month.total) / prev_month.total * 100 as mom_growth_rate,
  curr_month.cost,
  (curr_month.cost - prev_month.cost) / prev_month.cost * 100 as cost_mom_growth_rate;
```

## Performance Optimization Guidelines

### 1. Query Optimization
- Always use parameterized queries to leverage query plan caching
- Include date range filters early in MATCH clauses
- Use LIMIT clauses for large result sets
- Prefer MERGE over CREATE for idempotent operations

### 2. Data Modeling Best Practices
- Keep frequently accessed properties on nodes rather than relationships
- Use appropriate data types (Date vs DateTime, Integer vs Float)
- Normalize repeated string values using separate nodes when beneficial
- Consider denormalizing heavily aggregated values

### 3. Batch Operations
```cypher
// Batch insert pattern for large datasets
UNWIND $batch as row
MERGE (m:MetricData {id: row.id})
SET m += row.properties,
    m.updated_at = datetime();
```

### 4. Memory Management
- Use periodic commits for large data loads: `USING PERIODIC COMMIT 1000`
- Monitor query memory usage with `PROFILE` and `EXPLAIN`
- Consider data archiving strategies for historical time-series data

## Data Validation Rules

### 1. Consistency Checks
```cypher
// Ensure all sites have parent states
MATCH (s:Site)
WHERE NOT EXISTS((s)<-[:CONTAINS]-(:State))
RETURN s.id, s.name;

// Validate metric data ranges
MATCH (m:MetricData)
WHERE m.value < 0 OR m.cost < 0 OR m.quality_score < 0 OR m.quality_score > 1
RETURN m.id, m.value, m.cost, m.quality_score;
```

### 2. Data Quality Monitoring
```cypher
// Check for missing recent data
MATCH (s:Site)
OPTIONAL MATCH (s)-[:HAS_METRIC]->(m:MetricData)
WHERE m.measurement_date >= date() - duration({days: 7})
WITH s, count(m) as recent_metrics
WHERE recent_metrics = 0
RETURN s.id, s.name, "No recent metrics" as issue;
```

## Backup and Recovery

### 1. Data Export
```cypher
// Export location hierarchy
MATCH (g:Global)-[:CONTAINS*]->(location)
RETURN g, location, relationships(path) as rels;

// Export time-series data with date range
MATCH (m:MetricData)
WHERE m.measurement_date >= date($backupStartDate)
  AND m.measurement_date <= date($backupEndDate)
RETURN m;
```

### 2. Incremental Backups
- Use `created_at` and `updated_at` timestamps for incremental exports
- Implement change logs for critical business data
- Regular validation of backup integrity

This schema provides a comprehensive foundation for the Executive Dashboard's data requirements, supporting hierarchical analysis, time-series tracking, goal management, and risk assessment while maintaining performance and data integrity.