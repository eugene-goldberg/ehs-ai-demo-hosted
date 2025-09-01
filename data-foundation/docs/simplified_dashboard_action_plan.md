# Simplified Executive Dashboard Action Plan

## Overview
This document outlines the MVP implementation for an executive EHS dashboard that displays sustainability metrics (electricity, water, waste) with automated risk assessment and recommendations. The dashboard will be pre-loaded with data stored in Neo4j and use LLM-based risk assessment agents.

## Core Requirements

### Dashboard Categories
- **Electricity**: Consumption tracking, CO2 emissions conversion, goals vs actual
- **Water**: Consumption monitoring, efficiency metrics, conservation goals
- **Waste**: Generation tracking, recycling rates, reduction targets

### Key Features
- Company sustainability goals display
- 6 months historical data visualization
- Automated risk assessment using LLM agents
- Site-specific risks and recommendations
- CO2 emissions calculations for electricity/water consumption

## Implementation Plan

### Phase 1: Neo4j Data Structure (Week 1)

#### Core Nodes
```cypher
// Company Goals
(:Goal {
  category: "electricity|water|waste",
  target_value: float,
  unit: string,
  period: "monthly|quarterly|yearly",
  target_date: date,
  created_at: datetime
})

// Historical Data
(:Measurement {
  category: "electricity|water|waste",
  value: float,
  unit: string,
  measurement_date: date,
  site_id: string,
  created_at: datetime
})

// Sites
(:Site {
  id: string,
  name: string,
  location: string,
  type: string,
  created_at: datetime
})

// Risk Assessments (LLM Generated)
(:RiskAssessment {
  id: string,
  category: "electricity|water|waste",
  risk_level: "low|medium|high|critical",
  description: string,
  assessment_date: datetime,
  factors: [string],
  confidence_score: float
})

// Recommendations (LLM Generated)
(:Recommendation {
  id: string,
  title: string,
  description: string,
  priority: "low|medium|high|urgent",
  estimated_impact: string,
  implementation_effort: "low|medium|high",
  category: "electricity|water|waste",
  created_at: datetime
})

// CO2 Conversion Factors
(:EmissionFactor {
  category: "electricity|water",
  region: string,
  factor: float, // kg CO2 per kWh or liter
  unit: string,
  valid_from: date,
  valid_to: date
})
```

#### Relationships
```cypher
(:Site)-[:HAS_MEASUREMENT]->(:Measurement)
(:Site)-[:HAS_RISK]->(:RiskAssessment)
(:Site)-[:HAS_RECOMMENDATION]->(:Recommendation)
(:RiskAssessment)-[:SUGGESTS]->(:Recommendation)
(:Goal)-[:APPLIES_TO]->(:Site)
(:Measurement)-[:CONTRIBUTES_TO_EMISSION]->(:EmissionCalculation)
```

### Phase 2: Data Loading Strategy (Week 1)

#### Sample Data Preparation Strategy
Generate 6 months of historical data with **intentional patterns** to trigger meaningful risk assessments:

**Risk Scenario Patterns:**
- **CRITICAL Risk Sites**: 15-20% monthly increases, way off target
- **HIGH Risk Sites**: 5-8% monthly increases, trending toward goal failure  
- **MEDIUM Risk Sites**: Flat or slight increases, may miss goals without action
- **LOW Risk Sites**: Decreasing trends, on track to exceed goals

#### Data Generation Scenarios

**Scenario 1: Critical Risk - Manufacturing Site**
```python
# Site experiencing equipment degradation
electricity_pattern = {
    'baseline': 25000,  # kWh/month
    'monthly_increase': 0.18,  # 18% monthly increase
    'annual_goal': 0.15,       # 15% reduction target
    'pattern': 'exponential_increase',
    'factors': ['aging equipment', 'increased production', 'inefficient HVAC']
}
```

**Scenario 2: High Risk - Office Complex**
```python
# Site with seasonal challenges
water_pattern = {
    'baseline': 50000,     # liters/month  
    'monthly_increase': 0.06,  # 6% monthly increase
    'annual_goal': 0.12,       # 12% reduction target
    'pattern': 'summer_peak_with_trend',
    'factors': ['cooling system overuse', 'landscaping needs', 'leak detection needed']
}
```

**Scenario 3: Medium Risk - Warehouse**
```python
# Site with stagnant performance
waste_pattern = {
    'baseline': 8000,      # kg/month
    'monthly_change': 0.02,    # 2% monthly increase
    'annual_goal': 0.10,       # 10% reduction target  
    'pattern': 'seasonal_flat',
    'factors': ['recycling program stalled', 'packaging waste increase']
}
```

**Scenario 4: Low Risk - Research Facility**
```python
# Site performing well
electricity_pattern = {
    'baseline': 15000,     # kWh/month
    'monthly_decrease': 0.03,  # 3% monthly decrease
    'annual_goal': 0.08,       # 8% reduction target
    'pattern': 'steady_improvement',
    'factors': ['LED conversion', 'energy management system', 'staff engagement']
}
```

#### Data Loading Scripts
```python
# /data_loading/load_baseline_data.py

class PatternBasedDataGenerator:
    def generate_consumption_series(self, pattern_config):
        """
        Generate 6 months of data following specific risk patterns
        Returns monthly measurements with realistic seasonal variations
        """
        
    def create_risk_scenarios(self):
        """
        Create diverse site scenarios to test LLM risk assessment:
        - 2 sites with CRITICAL risk patterns  
        - 3 sites with HIGH risk patterns
        - 4 sites with MEDIUM risk patterns
        - 3 sites with LOW risk patterns
        """
        
    def add_seasonal_variations(self, base_pattern):
        """
        Layer seasonal patterns onto base trends:
        - Winter electricity peaks for heating
        - Summer water peaks for cooling
        - Holiday waste reductions
        """
        
    def inject_realistic_anomalies(self, measurements):
        """
        Add realistic consumption spikes:
        - Equipment failures causing temporary increases
        - Weather events driving consumption
        - Maintenance shutdowns causing decreases
        """
```

#### Target Data Patterns for Testing

**Critical Risk Scenarios (Sites that will definitely miss goals):**
```python
critical_sites = [
    {
        'site_id': 'MANUFACTURING_A',
        'category': 'electricity',
        'pattern': 'equipment_degradation',
        'monthly_increase': 0.15,
        'annual_goal_gap': 0.35,  # 35% over target projection
        'expected_risk': 'CRITICAL'
    },
    {
        'site_id': 'DATACENTER_B',
        'category': 'electricity', 
        'pattern': 'cooling_system_failure',
        'monthly_increase': 0.22,
        'annual_goal_gap': 0.45,
        'expected_risk': 'CRITICAL'
    }
]
```

**High Risk Scenarios (Sites likely to miss goals):**
```python
high_risk_sites = [
    {
        'site_id': 'OFFICE_COMPLEX_C',
        'category': 'water',
        'pattern': 'summer_peak_trend',
        'monthly_increase': 0.08,
        'annual_goal_gap': 0.18,
        'expected_risk': 'HIGH'
    },
    {
        'site_id': 'WAREHOUSE_D',
        'category': 'waste',
        'pattern': 'packaging_increase',
        'monthly_increase': 0.05,
        'annual_goal_gap': 0.12,
        'expected_risk': 'HIGH'
    }
]
```

#### Sample Data Validation
```python
# Validation that data patterns will trigger expected risk assessments
def validate_risk_patterns():
    """
    Ensure generated data will produce meaningful risk assessments:
    1. Verify trend calculations match expected patterns
    2. Confirm goal gaps align with risk level thresholds  
    3. Test that LLM prompts receive sufficient signal for analysis
    4. Validate seasonal patterns are realistic for each category
    """
```

#### Data Loading Output
- **12 diverse sites** with varying risk profiles across electricity, water, and waste
- **6 months of measurements** per site with realistic seasonal and anomaly patterns
- **Clear annual goals** that create meaningful gap analysis opportunities  
- **Site characteristics** that explain consumption patterns (size, type, location)
- **Emission conversion factors** for accurate CO2 calculations

### Phase 3: Risk Assessment Agent (Week 2)

#### LLM Analysis Workflow Overview
The Risk Assessment Agent follows a structured 5-step analysis process:

1. **Data Aggregation**: LLM processes 6 months of consumption data
2. **Trend Analysis**: LLM identifies patterns (increasing, decreasing, seasonal)
3. **Goal Comparison**: LLM compares trends to annual reduction goals
4. **Risk Assessment**: LLM determines likelihood of meeting annual targets
5. **Recommendations**: LLM provides specific actions based on risk analysis

#### Risk Assessment Agent Implementation
```python
# /agents/risk_assessment_agent.py

class RiskAssessmentAgent:
    def analyze_site_performance(self, site_id: str, category: str):
        """
        Complete LLM analysis workflow:
        Input: 6 months consumption data + annual reduction goals
        Output: Risk assessment + recommendations
        """
        
        # Step 1: Aggregate 6 months of data
        historical_data = self.get_6month_consumption_data(site_id, category)
        annual_goal = self.get_annual_reduction_goal(site_id, category)
        
        # Step 2: LLM identifies trends and patterns
        trend_analysis = self.llm_analyze_trends(historical_data)
        
        # Step 3: LLM compares trends to goals
        goal_comparison = self.llm_compare_to_goals(trend_analysis, annual_goal)
        
        # Step 4: LLM assesses risk of missing annual target
        risk_assessment = self.llm_assess_goal_risk(goal_comparison)
        
        # Step 5: LLM generates specific recommendations
        recommendations = self.llm_generate_recommendations(risk_assessment)
        
        return {
            'trend_analysis': trend_analysis,
            'risk_assessment': risk_assessment,
            'recommendations': recommendations
        }
    
    def llm_analyze_trends(self, data):
        """
        LLM analyzes 6-month data to identify:
        - Monthly consumption trends (increasing/decreasing)
        - Seasonal patterns
        - Rate of change calculations
        - Statistical anomalies
        """
        
    def llm_compare_to_goals(self, trends, annual_goal):
        """
        LLM projects current trends against annual reduction targets:
        - Calculate projected annual performance based on 6-month trend
        - Determine gap between projection and goal
        - Assess timeline constraints (months remaining in year)
        """
        
    def llm_assess_goal_risk(self, comparison):
        """
        LLM determines risk level based on trend vs goal analysis:
        - LOW: On track to exceed goal
        - MEDIUM: May miss goal without intervention
        - HIGH: Likely to miss goal significantly
        - CRITICAL: Goal impossible without immediate action
        """
        
    def llm_generate_recommendations(self, risk_assessment):
        """
        LLM provides specific actions to get back on track:
        - Immediate actions for critical risks
        - Medium-term strategies for high/medium risks
        - Optimization suggestions for low risks
        """
```

#### LLM Analysis Workflow Details

**Input Data Structure:**
- **Historical Data**: 6 months of daily/weekly consumption measurements
- **Annual Goals**: Target reduction percentages or absolute consumption limits
- **Site Context**: Location, size, type, historical performance
- **Seasonal Factors**: Expected consumption patterns by month

**Step 1: Data Aggregation**
```python
# LLM processes structured consumption data
consumption_data = {
    'site_id': 'SITE_001',
    'category': 'electricity',
    'measurements': [
        {'month': 'Jan', 'consumption': 15000, 'unit': 'kWh'},
        {'month': 'Feb', 'consumption': 16500, 'unit': 'kWh'},
        # ... 6 months of data
    ],
    'annual_goal': {
        'target_reduction': 0.15,  # 15% reduction
        'baseline_year': 2023,
        'baseline_consumption': 180000  # kWh
    }
}
```

**Step 2: Trend Identification**
```python
# LLM identifies patterns in the data
trend_analysis = {
    'overall_trend': 'increasing',  # increasing/decreasing/stable
    'monthly_change_rate': 0.08,   # 8% increase per month
    'seasonal_pattern': 'winter_peak',
    'anomalies': ['Feb spike due to cold weather'],
    'confidence_level': 0.85
}
```

**Step 3: Goal Comparison**
```python
# LLM projects performance against annual target
goal_comparison = {
    'projected_annual_consumption': 195000,  # kWh
    'goal_consumption': 153000,             # 15% below baseline
    'projected_vs_goal_gap': 42000,         # kWh over target
    'goal_achievement_probability': 0.15,    # 15% chance of success
    'months_remaining': 6,
    'required_monthly_reduction': 0.25      # 25% reduction needed
}
```

**Step 4: Risk Assessment**
```python
# LLM determines risk level and factors
risk_assessment = {
    'risk_level': 'HIGH',
    'risk_factors': [
        'Consumption increasing 8% monthly',
        'Already 23% above target pace',
        'Winter peak season approaching',
        'Only 6 months to achieve 15% reduction'
    ],
    'confidence_score': 0.85,
    'assessment_reasoning': 'Current trend shows 8% monthly increase...'
}
```

**Step 5: Recommendations**
```python
# LLM generates specific, actionable recommendations
recommendations = [
    {
        'priority': 'URGENT',
        'action': 'Implement immediate 20% lighting reduction',
        'estimated_impact': '2500 kWh/month savings',
        'implementation_effort': 'LOW',
        'timeline': '1 week'
    },
    {
        'priority': 'HIGH',
        'action': 'Upgrade HVAC systems in Building A',
        'estimated_impact': '5000 kWh/month savings',
        'implementation_effort': 'HIGH',
        'timeline': '2 months'
    }
]
```

#### LLM Prompt Templates

**Trend Analysis Prompt:**
```
Analyze the following 6 months of {category} consumption data for {site_name}:
{consumption_data}

Identify:
1. Overall trend (increasing/decreasing/stable) with monthly rate of change
2. Seasonal patterns or anomalies
3. Statistical significance of observed trends
4. Confidence level in trend projections

Provide analysis in structured format with quantified metrics.
```

**Risk Assessment Prompt:**
```
Based on the trend analysis and annual reduction goal:
- Current trend: {trend_summary}
- Annual goal: {goal_details}
- Time remaining: {months_remaining} months

Assess the risk of missing the annual goal:
1. Calculate projected annual performance
2. Determine gap between projection and goal
3. Assign risk level (LOW/MEDIUM/HIGH/CRITICAL)
4. Explain reasoning with specific metrics

Risk level must be:
- LOW: >90% chance of meeting goal
- MEDIUM: 50-90% chance of meeting goal  
- HIGH: 10-50% chance of meeting goal
- CRITICAL: <10% chance of meeting goal
```

**Recommendation Generation Prompt:**
```
Based on the risk assessment for {category} at {site_name}:

Risk Level: {risk_level}
Current Gap: {goal_gap} {units}
Time Remaining: {months_remaining} months
Key Risk Factors: {risk_factors}

Generate 3-5 specific, actionable recommendations to address this risk:

For each recommendation, provide:
1. Priority level (URGENT/HIGH/MEDIUM/LOW)
2. Specific action description
3. Estimated monthly impact in {units}
4. Implementation effort (LOW/MEDIUM/HIGH)
5. Timeline to implement
6. Resource requirements

Focus on:
- CRITICAL/HIGH risks: Immediate actions with high impact
- MEDIUM risks: Strategic improvements and efficiency measures
- LOW risks: Optimization and best practice implementations

Recommendations must be specific to {category} consumption and realistic for a {site_type} facility.
```

### Phase 4: Dashboard Backend (Week 2)

#### FastAPI Endpoints
```python
# /api/dashboard_endpoints.py

@router.get("/dashboard/overview")
async def get_dashboard_overview():
    """Return summary metrics for all categories"""

@router.get("/dashboard/category/{category}")
async def get_category_details(category: str):
    """Return detailed data for specific category"""

@router.get("/dashboard/site/{site_id}")
async def get_site_dashboard(site_id: str):
    """Return site-specific dashboard data"""

@router.get("/dashboard/risks")
async def get_current_risks():
    """Return current risk assessments"""

@router.get("/dashboard/recommendations")
async def get_active_recommendations():
    """Return prioritized recommendations"""
```

#### Data Service Layer
```python
# /services/dashboard_service.py

class DashboardService:
    def get_category_performance(self, category: str, site_id: str = None)
    def calculate_co2_emissions(self, measurements: list, category: str)
    def get_goal_progress(self, category: str, site_id: str = None)
    def get_risk_summary(self, category: str = None)
    def get_recommendations_by_priority(self, limit: int = 10)
```

### Phase 5: Dashboard Frontend (Week 3)

#### Key Dashboard Components
- **Overview Cards**: Current performance vs goals for each category
- **Historical Charts**: 6-month trend visualization
- **Risk Indicators**: Visual risk level indicators by category/site
- **Recommendations Panel**: Prioritized action items
- **CO2 Emissions Display**: Converted emissions for electricity/water

#### Technology Stack
- React/Vue.js for frontend components
- Chart.js/D3.js for data visualization
- Real-time updates from Neo4j data

## Data Flow Architecture

```
Historical Data → Neo4j Database ← Risk Assessment Agent (LLM)
                     ↓
              Dashboard Service ← FastAPI Endpoints
                     ↓
               Frontend Dashboard
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Set up Neo4j database with schema
- [ ] Create data loading scripts
- [ ] Load 6 months of sample data
- [ ] Load company goals and site information
- [ ] Set up emission conversion factors

### Week 2: Intelligence Layer
- [ ] Implement Risk Assessment Agent
- [ ] Create LLM prompts for risk analysis
- [ ] Build recommendation generation logic
- [ ] Test risk assessment on sample data
- [ ] Create dashboard backend APIs

### Week 3: User Interface
- [ ] Build dashboard frontend components
- [ ] Implement data visualization charts
- [ ] Add risk indicators and recommendation panels
- [ ] Integrate with backend APIs
- [ ] User acceptance testing

## Success Metrics

### Technical Metrics
- All dashboard data loads from Neo4j (no on-the-fly generation)
- Risk assessments complete within 30 seconds
- Dashboard loads in under 2 seconds
- 99.9% API uptime

### LLM Analysis Validation
- **Trend Analysis Accuracy**: LLM correctly identifies increasing/decreasing patterns in 95% of test cases
- **Risk Level Assignment**: Risk levels align with expected thresholds based on goal gaps
  - CRITICAL: <10% goal achievement probability correctly identified
  - HIGH: 10-50% goal achievement probability correctly identified  
  - MEDIUM: 50-90% goal achievement probability correctly identified
  - LOW: >90% goal achievement probability correctly identified
- **Recommendation Relevance**: Recommendations are specific to category and site type in 100% of cases
- **Quantified Impact**: All recommendations include estimated monthly impact in appropriate units

### Business Metrics
- **Goal Performance Visibility**: Clear visualization of 6-month trends vs annual targets for all sites
- **Risk Detection**: Automated identification of sites at risk of missing sustainability goals
- **Actionable Insights**: Each risk assessment generates 3-5 specific, prioritized recommendations
- **Comprehensive Coverage**: Risk analysis covers all three categories (electricity, water, waste) across all sites
- **CO2 Emissions Integration**: Accurate CO2 calculations for electricity and water consumption

### LLM Workflow Validation Criteria
```python
# Validation tests for each step of the LLM analysis
validation_suite = {
    'data_aggregation': 'Verify 6 months of data processed correctly',
    'trend_analysis': 'Confirm trend direction and rate calculations',
    'goal_comparison': 'Validate projected vs target gap analysis',
    'risk_assessment': 'Test risk level assignment logic',
    'recommendations': 'Verify actionable, specific, quantified recommendations'
}
```

## Risk Mitigation

### Technical Risks
- **Neo4j Performance**: Pre-load all data, optimize queries
- **LLM Reliability**: Implement retry logic, fallback assessments
- **Data Quality**: Validation scripts, error handling

### Business Risks
- **Goal Definition**: Work with stakeholders to define clear, measurable goals
- **Data Availability**: Ensure consistent historical data collection
- **User Adoption**: Simple, intuitive dashboard design

## Next Steps After MVP

1. **Real-time Data Integration**: Connect to live data sources
2. **Advanced Analytics**: Predictive modeling, anomaly detection  
3. **Mobile Dashboard**: Responsive design for mobile access
4. **Compliance Reporting**: Automated regulatory reports
5. **Benchmark Comparisons**: Industry standard comparisons

## Dependencies

- Neo4j database setup and configuration
- LLM API access (OpenAI/Anthropic)
- Historical EHS data in structured format
- Company sustainability goals definition
- Site inventory and characteristics

This action plan provides a clear, achievable path to deliver a functional executive dashboard within 3 weeks, with all data pre-loaded in Neo4j and intelligent risk assessment capabilities.