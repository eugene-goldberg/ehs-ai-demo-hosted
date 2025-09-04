# Risk Assessment Agent Documentation

## Overview

The Risk Assessment Agent is a sophisticated AI-powered tool designed to analyze Environmental, Health, and Safety (EHS) data and assess risks related to meeting annual reduction goals. The agent leverages Large Language Models (LLM) and Neo4j graph database to provide comprehensive risk assessments and actionable recommendations.

The agent implements two main approaches:

1. **Simplified Dashboard Implementation**: A 5-step LLM workflow for consumption data analysis
2. **Comprehensive LangGraph Implementation**: A multi-node workflow using LangGraph for detailed facility risk assessment

## Key Features

- 🎯 **Goal-oriented Analysis**: Assesses likelihood of meeting annual EHS reduction targets
- 📊 **Trend Analysis**: Analyzes 6-month consumption patterns and trends
- ⚠️ **Risk Classification**: Provides 4-level risk classification (LOW/MEDIUM/HIGH/CRITICAL)
- 💡 **Smart Recommendations**: Generates specific, actionable mitigation strategies
- 🔗 **Neo4j Integration**: Leverages graph database for comprehensive data analysis
- 📈 **Real-time Processing**: Provides immediate results with detailed explanations

## How It Works: 5-Step Workflow

### Step 1: Data Aggregation
- Retrieves 6 months of consumption data from Neo4j
- Fetches annual reduction goals for the specified category
- Validates data completeness and quality

### Step 2: Trend Analysis (LLM)
The LLM analyzes consumption patterns to identify:
- Overall trend direction (increasing/decreasing/stable)
- Monthly rate of change (percentage)
- Seasonal patterns or anomalies
- Key observations about consumption patterns

### Step 3: Goal Comparison (LLM)
Projects current trends against annual targets:
- Calculates projected annual performance based on 6-month trend
- Determines gap between projection and goal
- Assesses timeline constraints (months remaining in year)

### Step 4: Risk Assessment (LLM)
Determines risk level based on trend vs goal analysis:
- **LOW**: On track to exceed goal (>90% chance)
- **MEDIUM**: May miss goal without intervention (50-90% chance)
- **HIGH**: Likely to miss goal significantly (10-50% chance)
- **CRITICAL**: Goal impossible without immediate action (<10% chance)

### Step 5: Recommendation Generation (LLM)
Provides specific actions based on risk level:
- **LOW**: Optimization and efficiency improvements
- **MEDIUM**: Targeted interventions and monitoring
- **HIGH**: Aggressive conservation measures and system changes
- **CRITICAL**: Emergency protocols and immediate action plans

## Data Requirements in Neo4j

### Required Schema Elements

#### Site Nodes
```cypher
(:Site {id: "SITE_001", name: "Main Manufacturing Facility"})
```

#### Consumption Data Nodes
- **Electricity**: `(:ElectricityConsumption {date: date('2024-01-01'), consumption_kwh: 1500, cost_usd: 200})`
- **Water**: `(:WaterConsumption {date: date('2024-01-01'), consumption_gallons: 5000, cost_usd: 150})`
- **Waste**: `(:WasteGeneration {date: date('2024-01-01'), quantity_lbs: 500, cost_usd: 100})`

#### Goal Nodes
```cypher
(:Goal {
  category: "electricity",
  target_value: -15,
  unit: "%",
  period: "annual",
  target_date: date('2024-12-31')
})
```

#### Relationships
```cypher
(:Site)-[:HAS_ELECTRICITY_CONSUMPTION]->(:ElectricityConsumption)
(:Site)-[:HAS_WATER_CONSUMPTION]->(:WaterConsumption)
(:Site)-[:HAS_WASTE_GENERATION]->(:WasteGeneration)
(:Goal)-[:APPLIES_TO]->(:Site)
```

### Data Requirements

1. **6 Months Historical Data**: At least 24 weeks of consumption data
2. **Annual Goals**: Reduction targets for each category (electricity, water, waste)
3. **Emission Factors**: Environmental impact calculations
4. **Site Information**: Facility type, operational details

## How to Run the Agent

### Method 1: Simplified Dashboard Agent

```python
from src.agents.risk_assessment_agent import RiskAssessmentAgent

# Initialize the agent
agent = RiskAssessmentAgent()

# Run site performance analysis
result = agent.analyze_site_performance(
    site_id="SITE_001",
    category="electricity"  # or "water", "waste"
)

# Access results
print(f"Risk Level: {result['risk_assessment']['risk_level']}")
print(f"Recommendations: {len(result['recommendations']['recommendations'])}")
```

### Method 2: Comprehensive LangGraph Agent

```python
from src.agents.risk_assessment.agent import create_risk_assessment_agent

# Initialize the comprehensive agent
agent = create_risk_assessment_agent(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="your_password",
    llm_model="gpt-4o"
)

# Perform facility risk assessment
result = agent.assess_facility_risk(
    facility_id="DEMO_FACILITY_001",
    assessment_scope={
        "areas": ["environmental", "health", "safety", "compliance"],
        "date_range": "6_months"
    }
)

# Access comprehensive results
print(f"Status: {result['status']}")
print(f"Risk Score: {result['risk_assessment'].risk_score}")
```

### Method 3: Using Test Scripts

```bash
# Standalone test
python3 test_risk_assessment_standalone.py --facility-id DEMO_FACILITY_001 --verbose

# Risk demo
python3 run_risk_demo.py --facility-id SITE_001 --save-results

# Comprehensive tests
python3 run_risk_assessment_tests.py
```

### Environment Setup

Create a `.env` file with required variables:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# LangSmith (Optional)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ehs-risk-assessment
```

## Understanding the Results

### Risk Assessment Output Structure

```python
{
    "trend_analysis": {
        "overall_trend": "increasing",
        "monthly_change_rate": 0.05,
        "seasonal_pattern": "winter_peak",
        "confidence_level": 0.85
    },
    "risk_assessment": {
        "risk_level": "HIGH",
        "risk_probability": 0.3,
        "gap_percentage": 35.2,
        "goal_achievable": false,
        "assessment_date": "2024-09-04T10:30:00Z",
        "confidence_score": 0.8
    },
    "recommendations": {
        "recommendations": [
            {
                "title": "Implement Energy Management System",
                "description": "Deploy automated energy monitoring...",
                "priority": "high",
                "estimated_impact": "15% reduction"
            }
        ],
        "priority": "high",
        "generated_date": "2024-09-04T10:30:00Z"
    }
}
```

### Risk Level Interpretations

#### LOW Risk (>90% success probability)
- **Meaning**: Current trends indicate high likelihood of meeting/exceeding goals
- **Gap**: Less than 10% deviation from target
- **Action**: Continue current practices, focus on optimization

#### MEDIUM Risk (50-90% success probability)
- **Meaning**: Goal achievable but requires attention and minor adjustments
- **Gap**: 10-25% deviation from target
- **Action**: Implement targeted interventions and increased monitoring

#### HIGH Risk (10-50% success probability)
- **Meaning**: Significant intervention required to meet goals
- **Gap**: 25-50% deviation from target
- **Action**: Aggressive conservation measures and system changes needed

#### CRITICAL Risk (<10% success probability)
- **Meaning**: Goal extremely unlikely without immediate, drastic action
- **Gap**: Greater than 50% deviation from target
- **Action**: Emergency protocols and immediate comprehensive overhaul

### Sample Output

```
========================================
RISK ASSESSMENT AGENT - SITE PERFORMANCE ANALYSIS
========================================
Site ID: SITE_001
Category: electricity

Trend Analysis:
  Overall Trend: Increasing
  Monthly Change: 5.2%

Risk Assessment:
  Risk Level: HIGH
  Probability: 30%
  Gap: 35.2%

Recommendations:
  1. Deploy smart energy management system for 15-20% consumption reduction...
  2. Upgrade HVAC system to high-efficiency units reducing peak demand by 25%...
  3. Implement demand response program with utility provider...
========================================
```

## Troubleshooting Common Issues

### 1. Connection Issues

**Problem**: `Failed to connect to Neo4j`

**Solutions**:
- Verify Neo4j server is running: `systemctl status neo4j`
- Check connection parameters in `.env` file
- Test connection manually: `cypher-shell -u neo4j -p your_password`

### 2. Missing Data

**Problem**: `No consumption data found for site/category`

**Solutions**:
- Verify data exists in Neo4j:
```cypher
MATCH (s:Site {id: "SITE_001"})-[:HAS_ELECTRICITY_CONSUMPTION]->(e)
RETURN count(e)
```
- Check date ranges (agent looks for last 6 months)
- Ensure proper node labels and relationships

### 3. API Key Issues

**Problem**: `OpenAI API key not provided` or `Authentication failed`

**Solutions**:
- Verify `OPENAI_API_KEY` in `.env` file
- Check API key validity and quotas
- Ensure proper environment variable loading

### 4. Import Errors

**Problem**: `ModuleNotFoundError` or import issues

**Solutions**:
- Ensure you're in the correct directory (`backend/`)
- Check Python path setup:
```python
import sys
sys.path.append('/path/to/backend/src')
```
- Install requirements: `pip install -r requirements.txt`

### 5. LLM Response Parsing

**Problem**: `JSON parsing failed` or malformed responses

**Solutions**:
- Check LLM model availability (gpt-4, gpt-4o)
- Verify prompt formatting in `risk_assessment_prompts.py`
- Enable debug logging for detailed error information

### 6. Memory or Performance Issues

**Problem**: Agent hangs or runs out of memory

**Solutions**:
- Limit data scope in assessment configuration
- Use pagination for large datasets
- Monitor Neo4j query performance
- Consider using lighter LLM models for testing

### 7. LangGraph Workflow Issues

**Problem**: Workflow fails or gets stuck in loops

**Solutions**:
- Check retry limits and recursion depth settings
- Enable detailed logging: `logging.getLogger().setLevel(logging.DEBUG)`
- Verify state transitions in workflow graph
- Monitor LangSmith traces if enabled

## Advanced Configuration

### Custom Risk Methodologies

```python
agent = RiskAssessmentAgent(
    assessment_methodology="custom",
    max_retries=5,
    max_step_retries=3
)
```

### LangSmith Integration

```python
# Enable comprehensive tracing
agent = create_risk_assessment_agent(
    enable_langsmith=True,
    llm_model="gpt-4o"
)
```

### Batch Processing

```python
# Assess multiple facilities
results = agent.assess_multiple_facilities(
    facility_ids=["SITE_001", "SITE_002", "SITE_003"],
    assessment_scope={"period": "6_months"}
)
```

## Performance Considerations

- **Data Volume**: Optimal performance with 6 months (24 weeks) of data
- **Response Time**: Typical assessment takes 30-60 seconds
- **Concurrent Usage**: Agent supports multiple concurrent assessments
- **Resource Usage**: Requires ~500MB RAM per active assessment

## Integration Points

The Risk Assessment Agent integrates with:
- **Executive Dashboard**: Provides risk metrics and KPIs
- **Ingestion Workflow**: Processes new data for risk updates
- **Notification System**: Triggers alerts for high/critical risks
- **Reporting Engine**: Generates comprehensive risk reports

---

*For additional support or feature requests, please contact the EHS AI development team.*