# Risk Assessment Agent - Architecture Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Logic Flow](#core-logic-flow)
4. [Key Files and Modules](#key-files-and-modules)
5. [Neo4j Interactions](#neo4j-interactions)
6. [LLM Integration](#llm-integration)
7. [CO2e Conversion Logic](#co2e-conversion-logic)
8. [Risk Level Determination](#risk-level-determination)
9. [Dependencies](#dependencies)
10. [Configuration](#configuration)
11. [Testing](#testing)
12. [Known Issues and Solutions](#known-issues-and-solutions)

## Overview

### Purpose and Role in EHS AI System

The Risk Assessment Agent is a core component of the EHS AI Demo system that provides intelligent analysis of environmental consumption data to assess risks of meeting annual reduction goals. The agent serves as the analytical engine that bridges raw consumption data with actionable business insights.

**Key Functions:**
- Analyzes 6 months of historical consumption data across electricity, water, and waste categories
- Projects current trends against annual reduction targets
- Determines risk levels (LOW, MEDIUM, HIGH, CRITICAL) for goal achievement
- Generates specific, prioritized recommendations for risk mitigation
- Stores results in Neo4j for dashboard visualization and historical tracking

**Business Value:**
- Enables proactive risk management for environmental goals
- Provides data-driven insights for decision making
- Supports compliance with regulatory requirements
- Facilitates early intervention to prevent goal failures

## Architecture

### High-Level System Design

The Risk Assessment Agent follows a **5-step LLM-driven workflow** as specified in the simplified dashboard action plan:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   1. Data           │    │   2. Trend          │    │   3. Goal           │
│   Aggregation       │───▶│   Analysis          │───▶│   Comparison        │
│                     │    │   (LLM)             │    │   (LLM)             │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │
                                      ▼
┌─────────────────────┐    ┌─────────────────────┐
│   5. Recommendations│    │   4. Risk           │
│   (LLM)             │◀───│   Assessment        │
│                     │    │   (LLM)             │
└─────────────────────┘    └─────────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   Neo4j Storage     │
                           │   (Results)         │
                           └─────────────────────┘
```

### Key Components

- **RiskAssessmentAgent Class**: Main orchestrator implementing the 5-step workflow
- **Neo4jClient**: Database interface for data retrieval and storage
- **OpenAI LLM**: GPT-4 model for intelligent analysis at each step
- **Prompt Templates**: Structured prompts for consistent LLM interactions
- **CO2e Conversion Engine**: Electricity consumption to emissions converter

### Design Patterns

- **Strategy Pattern**: Different analysis strategies for each consumption category
- **Template Method**: Consistent workflow with customizable steps
- **Factory Pattern**: Prompt generation based on assessment type
- **Repository Pattern**: Neo4j data access abstraction

## Core Logic Flow

### The 5-Step Analysis Process

#### Step 1: Data Aggregation
```python
historical_data = self.get_6month_consumption_data(site_id, category)
annual_goal = self.get_annual_reduction_goal(site_id, category)
```

**Purpose**: Retrieve 6 months of consumption data and annual reduction goals from Neo4j

**Data Sources**:
- ElectricityConsumption nodes (kWh, cost, date)
- WaterConsumption nodes (gallons, cost, date)  
- WasteGeneration nodes (lbs, cost, date)
- Goal nodes (target values, units, periods)

**Processing**:
- Aggregates daily data into monthly summaries
- Calculates CO2e emissions for electricity consumption
- Validates data quality and completeness

#### Step 2: LLM Trend Analysis
```python
trend_analysis = self.llm_analyze_trends(historical_data)
```

**Purpose**: LLM identifies consumption patterns and trends over the 6-month period

**Analysis Focus**:
- Overall trend direction (increasing/decreasing/stable)
- Monthly rate of change calculations
- Seasonal patterns and anomalies
- Statistical confidence levels

**LLM Output**:
```json
{
  "overall_trend": "increasing",
  "monthly_change_rate": 0.025,
  "seasonal_pattern": "summer_peak",
  "confidence_level": 0.85,
  "analysis_text": "Detailed trend explanation..."
}
```

#### Step 3: LLM Goal Comparison
```python
goal_comparison = self.llm_compare_to_goals(trend_analysis, annual_goal, historical_data)
```

**Purpose**: Projects current trends against annual reduction targets

**Analysis Components**:
- Projects annual consumption based on current 6-month trend
- Calculates gap between projection and goal (absolute and percentage)
- Considers timeline constraints (months remaining in year)
- Accounts for baseline vs. current performance

**Key Calculation**:
```python
# For percentage-based goals (e.g., -10% reduction)
target_consumption = baseline * (1 + goal_percentage/100)
gap_percentage = ((projected_annual - target_consumption) / target_consumption) * 100
```

#### Step 4: LLM Risk Assessment
```python
risk_assessment = self.llm_assess_goal_risk(goal_comparison)
```

**Purpose**: Determines risk level based on gap analysis and trend projection

**Risk Level Logic**:
- **LOW**: Gap < 10% (>90% chance of success)
- **MEDIUM**: Gap 10-25% (50-90% chance of success)
- **HIGH**: Gap 25-50% (10-50% chance of success)
- **CRITICAL**: Gap > 50% (<10% chance of success)

#### Step 5: LLM Recommendations
```python
recommendations = self.llm_generate_recommendations(risk_assessment)
```

**Purpose**: Generates specific, actionable recommendations based on risk level

**Recommendation Categories by Risk Level**:
- **LOW**: Optimization and efficiency improvements
- **MEDIUM**: Targeted interventions and monitoring
- **HIGH**: Aggressive conservation measures and system changes
- **CRITICAL**: Emergency protocols and immediate action plans

## Key Files and Modules

### `/src/agents/risk_assessment_agent.py`
**Main agent implementation** (871 lines)

**Key Methods**:
- `analyze_site_performance()`: Main orchestration method
- `llm_analyze_trends()`: Step 2 - Trend analysis with LLM
- `llm_compare_to_goals()`: Step 3 - Goal comparison analysis
- `llm_assess_goal_risk()`: Step 4 - Risk level determination
- `llm_generate_recommendations()`: Step 5 - Recommendation generation

**Helper Methods**:
- `get_6month_consumption_data()`: Neo4j data retrieval
- `get_annual_reduction_goal()`: Goal data retrieval
- `store_risk_assessment_in_neo4j()`: Result storage
- `store_recommendations_in_neo4j()`: Recommendation storage
- `_parse_llm_response()`: **FIXED** JSON parsing with gap calculation handling

**Fixed Issues**:
- Line 310-389: Enhanced JSON parsing to handle nested LLM response structures
- Resolves gap_percentage extraction from complex JSON responses

### `/src/agents/prompts/risk_assessment_prompts.py`
**Prompt templates** (168 lines)

**Core Prompts**:
```python
TREND_ANALYSIS_PROMPT = """
Analyze the following 6 months of {category} consumption data...
"""

RISK_ASSESSMENT_PROMPT = """
Based on the trend analysis below, assess the risk of missing the annual {category} goal...
Risk Levels:
- LOW: >90% chance of meeting goal
- MEDIUM: 50-90% chance of meeting goal  
- HIGH: 10-50% chance of meeting goal
- CRITICAL: <10% chance of meeting goal
"""

RECOMMENDATION_GENERATION_PROMPT = """
Generate 3-5 specific, actionable recommendations...
Focus Areas by Risk Level:
- LOW: Optimization and efficiency improvements
- MEDIUM: Targeted interventions and monitoring
- HIGH: Aggressive conservation measures
- CRITICAL: Emergency protocols
"""
```

**Formatting Functions**:
- `format_trend_analysis_prompt()`: Injects consumption data
- `format_risk_assessment_prompt()`: Injects goal and performance data
- `format_recommendation_prompt()`: Injects risk assessment results

### `/src/agents/risk_assessment/recommendation_context.md`
**Externalized recommendation template** (7 lines)

Simple template for recommendation generation:
```markdown
Generate recommendations for the following risk assessment:
Facility ID: {facility_id}
Risk Assessment: {risk_assessment}
Risk Factors Summary: {risk_factors_summary}
Facility Context: {facility_context}
Generate comprehensive, prioritized recommendations for risk mitigation.
```

### `/src/scripts/run_risk_assessment.py`
**Test script** (185 lines)

**Purpose**: Standalone script for testing and running the Risk Assessment Agent

**Key Features**:
- Environment setup and validation
- Neo4j connection testing
- Agent execution with sample data (algonquin_il, electricity)
- Results validation and debugging output
- Neo4j storage verification

**Usage**:
```bash
cd /backend/src/scripts/
python3 run_risk_assessment.py
```

## Neo4j Interactions

### Data Retrieval Queries

#### Electricity Consumption Data
```cypher
MATCH (s:Site {id: $site_id})-[:HAS_ELECTRICITY_CONSUMPTION]->(e:ElectricityConsumption)
WHERE e.date >= date($start_date)
RETURN e.date as timestamp, e.consumption_kwh as amount, 'kWh' as unit, e.cost_usd as cost
ORDER BY e.date ASC
```

#### Water Consumption Data
```cypher
MATCH (s:Site {id: $site_id})-[:HAS_WATER_CONSUMPTION]->(w:WaterConsumption)
WHERE w.date >= date($start_date)
RETURN w.date as timestamp, w.consumption_gallons as amount, 'gallons' as unit, w.cost_usd as cost
ORDER BY w.date ASC
```

#### Waste Generation Data
```cypher
MATCH (s:Site {id: $site_id})-[:HAS_WASTE_GENERATION]->(w:WasteGeneration)
WHERE w.date >= date($start_date)
RETURN w.date as timestamp, w.quantity_lbs as amount, 'lbs' as unit, w.cost_usd as cost
ORDER BY w.date ASC
```

#### Annual Goals
```cypher
MATCH (g:Goal {category: $category})-[:APPLIES_TO]->(s:Site {id: $site_id})
RETURN g.target_value as target_value, g.unit as unit, g.period as period, g.target_date as target_date
```

### Node Types Used

#### Site Nodes
```cypher
(:Site {id: "algonquin_il", name: "Algonquin IL Facility"})
```

#### Consumption Nodes
```cypher
(:ElectricityConsumption {date: date, consumption_kwh: float, cost_usd: float})
(:WaterConsumption {date: date, consumption_gallons: float, cost_usd: float})
(:WasteGeneration {date: date, quantity_lbs: float, cost_usd: float})
```

#### Goal Nodes
```cypher
(:Goal {category: "electricity", target_value: -10, unit: "%", period: "annual", target_date: "2024-12-31"})
```

### Relationships

- `(Site)-[:HAS_ELECTRICITY_CONSUMPTION]->(ElectricityConsumption)`
- `(Site)-[:HAS_WATER_CONSUMPTION]->(WaterConsumption)`
- `(Site)-[:HAS_WASTE_GENERATION]->(WasteGeneration)`
- `(Goal)-[:APPLIES_TO]->(Site)`
- `(Site)-[:HAS_RISK]->(RiskAssessment)`
- `(Site)-[:HAS_RECOMMENDATION]->(Recommendation)`

### Storage of Results

#### RiskAssessment Nodes
```cypher
CREATE (ra:RiskAssessment {
    id: "site_category_timestamp",
    site_id: $site_id,
    category: $category,
    risk_level: "HIGH",
    description: "Analysis text",
    assessment_date: datetime(),
    factors: ["Gap: 35%"],
    confidence_score: 0.85
})
```

#### Recommendation Nodes
```cypher
CREATE (r:Recommendation {
    id: "site_category_rec_1_timestamp",
    site_id: $site_id,
    title: "Recommendation Title",
    description: "Detailed recommendation",
    priority: "high",
    estimated_impact: "15% reduction",
    category: $category,
    created_date: datetime()
})
```

**Storage Logic**:
- Deletes existing records for same site/category before creating new ones
- Ensures dashboard always shows latest assessment results
- Maintains historical tracking through timestamped IDs

## LLM Integration

### OpenAI Configuration
```python
self.llm = ChatOpenAI(
    temperature=0,           # Deterministic responses
    openai_api_key=self.openai_api_key,
    model_name="gpt-4"      # Latest GPT-4 model
)
```

### Message Structure
Each LLM interaction uses a consistent message pattern:
```python
messages = [
    SystemMessage(content="You are an expert EHS analyst..."),
    HumanMessage(content=formatted_prompt)
]
response = self.llm(messages)
```

### System Prompts by Analysis Step

1. **Trend Analysis**: "Expert EHS data analyst specializing in consumption trend analysis"
2. **Goal Comparison**: "Expert EHS analyst comparing consumption trends to reduction goals" 
3. **Risk Assessment**: Uses algorithmic logic based on gap percentages
4. **Recommendations**: "Expert EHS consultant providing actionable recommendations"

### Response Parsing
**Challenge**: LLM responses can be nested JSON structures that complicate data extraction

**Solution**: Enhanced `_parse_llm_response()` method (lines 310-389):
```python
def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
    """
    Parse LLM response handling both flat and nested JSON structures
    
    FIXED: This method resolves the gap calculation issue by properly extracting
    values from nested JSON structures that the LLM returns.
    """
    # Look for direct fields first (preferred format)
    if 'gap_percentage' in data and data['gap_percentage'] != 0:
        result['gap_percentage'] = float(data['gap_percentage'])
    
    # If direct fields missing, search nested structure
    for key, value in data.items():
        if isinstance(value, dict):
            if 'gap' in key.lower():
                # Extract from gap analysis section
                for subkey, subvalue in value.items():
                    if 'percentage' in subkey.lower():
                        result['gap_percentage'] = float(subvalue)
```

## CO2e Conversion Logic

### Electricity Consumption to Emissions

**Conversion Factor**:
```python
ELECTRICITY_CO2E_FACTOR = 0.000395  # tonnes CO2e per kWh (US grid average)
```

**Calculation**:
```python
co2e_emissions = kwh_consumption * ELECTRICITY_CO2E_FACTOR
```

**Usage Context**:
- Applied when annual goals are specified in "tonnes CO2e"
- Enables comparison between electricity consumption (kWh) and emission reduction targets
- Supports carbon footprint analysis and reporting

**Data Flow**:
1. Retrieve electricity consumption in kWh from Neo4j
2. Calculate CO2e emissions for each data point
3. Aggregate monthly CO2e totals
4. Compare against CO2e reduction goals
5. Store both kWh and CO2e values for comprehensive analysis

**Example**:
```python
# Monthly data processing
for record in records:
    amount = float(record["amount"])  # kWh
    co2e_emissions = amount * self.ELECTRICITY_CO2E_FACTOR  # tonnes CO2e
    
    monthly_data[month_key] = {
        "amount": amount,                    # kWh
        "co2e_emissions": co2e_emissions,   # tonnes CO2e
        "co2e_unit": "tonnes CO2e"
    }
```

## Risk Level Determination

### Gap Percentage Calculation

**Formula**:
```python
gap_percentage = ((projected_annual - target_consumption) / target_consumption) * 100
```

**For Reduction Goals** (e.g., -10% target):
```python
# Target is reduced consumption from baseline
target_consumption = baseline * (1 + goal_percentage/100)  # e.g., baseline * 0.9

# Gap calculation
gap_percentage = abs((projected_annual - target_consumption) / target_consumption) * 100
```

### Risk Thresholds

**Algorithmic Risk Assignment**:
```python
if goal_achievable and gap_percentage < 10:
    risk_level = 'LOW'
    risk_probability = 0.95
elif gap_percentage < 25:
    risk_level = 'MEDIUM' 
    risk_probability = 0.7
elif gap_percentage < 50:
    risk_level = 'HIGH'
    risk_probability = 0.3
else:
    risk_level = 'CRITICAL'
    risk_probability = 0.05
```

### Risk Level Definitions

| Risk Level | Gap Range | Success Probability | Description |
|------------|-----------|-------------------|-------------|
| **LOW** | < 10% | > 90% | On track to exceed goal |
| **MEDIUM** | 10-25% | 50-90% | May miss goal without intervention |
| **HIGH** | 25-50% | 10-50% | Likely to miss goal significantly |
| **CRITICAL** | > 50% | < 10% | Goal impossible without immediate action |

### Success Probability Calculation
Risk probability represents the likelihood of achieving the annual goal based on current trends:
- Considers trend direction and rate of change
- Accounts for remaining time in assessment period
- Incorporates baseline vs. current performance comparison

## Dependencies

### Required Libraries
```python
# Core Python packages
import os, sys, logging, json, traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# LangChain for LLM integration
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Project-specific modules
from database.neo4j_client import Neo4jClient
from agents.prompts.risk_assessment_prompts import (
    format_trend_analysis_prompt,
    format_risk_assessment_prompt, 
    format_recommendation_prompt
)

# Environment management
from dotenv import load_dotenv
```

### External Services
- **OpenAI API**: GPT-4 model for intelligent analysis
- **Neo4j Database**: Graph database for data storage and retrieval
- **Python 3.8+**: Runtime environment

### Internal Dependencies
- **Neo4jClient**: Database abstraction layer
- **Prompt Templates**: Structured prompts for consistent LLM interactions
- **Environment Configuration**: API keys and connection strings

## Configuration

### Environment Variables

**Required**:
```bash
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

**Optional**:
```bash
RISK_ASSESSMENT_LOG_LEVEL=INFO
RISK_ASSESSMENT_MODEL=gpt-4
ELECTRICITY_CO2E_FACTOR=0.000395
```

### Agent Configuration
```python
# Agent initialization parameters
agent = RiskAssessmentAgent(
    neo4j_client=neo4j_client,      # Optional, creates default if None
    openai_api_key=openai_api_key   # Optional, reads from env if None
)

# LLM configuration
self.llm = ChatOpenAI(
    temperature=0,                   # Deterministic responses
    openai_api_key=self.openai_api_key,
    model_name="gpt-4"              # Configurable model
)
```

### Logging Configuration
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

## Running the Risk Assessment Agent

### Prerequisites

Before running the Risk Assessment Agent, ensure you have the following:

1. **Python Environment**:
   - Python 3.8 or higher
   - Virtual environment (venv) activated

2. **Database Setup**:
   - Neo4j database running (local or remote)
   - Database populated with consumption data and goals
   - Proper Neo4j credentials configured

3. **API Access**:
   - Valid OpenAI API key with GPT-4 access
   - Sufficient API credits for analysis requests

4. **Required Data**:
   - At least 6 months of historical consumption data
   - Annual reduction goals configured for target sites
   - Valid site IDs in the database

### Environment Setup

#### 1. Activate Python Virtual Environment
```bash
# Navigate to project root
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

#### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python3 -c "import langchain, neo4j, openai; print('Dependencies installed successfully')"
```

#### 3. Configure Environment Variables
Create or update your `.env` file in the backend directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Optional Settings
RISK_ASSESSMENT_LOG_LEVEL=INFO
RISK_ASSESSMENT_MODEL=gpt-4
ELECTRICITY_CO2E_FACTOR=0.000395
```

#### 4. Verify Database Connection
```bash
# Test Neo4j connection
python3 -c "
from database.neo4j_client import Neo4jClient
from dotenv import load_dotenv
load_dotenv()
client = Neo4jClient()
print('Neo4j connection successful' if client.verify_connectivity() else 'Connection failed')
"
```

### Running via Script

#### Quick Start - Test Script
The easiest way to run the agent is using the provided test script:

```bash
# Navigate to scripts directory
cd src/scripts/

# Run the agent with default test data
python3 run_risk_assessment.py
```

#### Script Output
The script will display:
```
=== Risk Assessment Agent Test ===
Environment setup complete.
Neo4j client initialized successfully.
Running risk assessment for site: algonquin_il, category: electricity

Step 1/5: Data Aggregation...
Retrieved 180 consumption records for analysis.

Step 2/5: LLM Trend Analysis...
Trend analysis completed: increasing trend detected.

Step 3/5: LLM Goal Comparison...
Goal comparison completed: 35.2% gap identified.

Step 4/5: Risk Assessment...
Risk level determined: HIGH

Step 5/5: Recommendation Generation...
Generated 4 actionable recommendations.

Results stored in Neo4j successfully.
=== Test completed successfully ===
```

### Running Programmatically

#### Basic Usage Example
```python
#!/usr/bin/env python3
"""
Example: Run Risk Assessment Agent programmatically
"""
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/src')

from agents.risk_assessment_agent import RiskAssessmentAgent
from database.neo4j_client import Neo4jClient

def run_risk_assessment():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    neo4j_client = Neo4jClient()
    agent = RiskAssessmentAgent(
        neo4j_client=neo4j_client,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Run analysis
    try:
        result = agent.analyze_site_performance(
            site_id="algonquin_il",
            category="electricity"
        )
        
        print(f"Risk Level: {result.get('risk_assessment', {}).get('risk_level', 'Unknown')}")
        print(f"Gap Percentage: {result.get('risk_assessment', {}).get('gap_percentage', 0)}%")
        print(f"Recommendations: {len(result.get('recommendations', {}).get('recommendations', []))}")
        
        return result
        
    except Exception as e:
        print(f"Error running risk assessment: {e}")
        return None
    finally:
        neo4j_client.close()

if __name__ == "__main__":
    run_risk_assessment()
```

#### Advanced Usage with Custom Parameters
```python
def run_custom_assessment(site_id: str, category: str):
    """Run risk assessment with custom site and category"""
    
    # Initialize agent
    agent = RiskAssessmentAgent()
    
    try:
        # Run analysis
        result = agent.analyze_site_performance(
            site_id=site_id,
            category=category
        )
        
        # Process results
        risk_data = result.get('risk_assessment', {})
        recommendations = result.get('recommendations', {}).get('recommendations', [])
        
        print(f"\n=== Risk Assessment Results ===")
        print(f"Site: {site_id}")
        print(f"Category: {category}")
        print(f"Risk Level: {risk_data.get('risk_level', 'Unknown')}")
        print(f"Gap Percentage: {risk_data.get('gap_percentage', 0):.1f}%")
        print(f"Goal Achievable: {risk_data.get('goal_achievable', False)}")
        
        print(f"\n=== Recommendations ({len(recommendations)}) ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        return result
        
    except Exception as e:
        print(f"Assessment failed: {e}")
        return None

# Usage examples
run_custom_assessment("algonquin_il", "electricity")
run_custom_assessment("algonquin_il", "water")  
run_custom_assessment("algonquin_il", "waste")
```

### Command Line Examples

#### Run with Different Sites and Categories
```bash
# Electricity analysis for Algonquin facility
python3 -c "
import sys; sys.path.append('src')
from agents.risk_assessment_agent import RiskAssessmentAgent
agent = RiskAssessmentAgent()
result = agent.analyze_site_performance('algonquin_il', 'electricity')
print(f'Risk Level: {result[\"risk_assessment\"][\"risk_level\"]}')
"

# Water consumption analysis
python3 -c "
import sys; sys.path.append('src')
from agents.risk_assessment_agent import RiskAssessmentAgent
agent = RiskAssessmentAgent()
result = agent.analyze_site_performance('algonquin_il', 'water')
print(f'Gap: {result[\"risk_assessment\"][\"gap_percentage\"]}%')
"

# Waste generation analysis
python3 -c "
import sys; sys.path.append('src')
from agents.risk_assessment_agent import RiskAssessmentAgent
agent = RiskAssessmentAgent()
result = agent.analyze_site_performance('algonquin_il', 'waste')
print(f'Recommendations: {len(result[\"recommendations\"][\"recommendations\"])}')
"
```

#### Batch Analysis Script
```bash
# Create batch analysis script
cat > run_batch_assessment.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.append('src')
from agents.risk_assessment_agent import RiskAssessmentAgent

def run_batch_analysis():
    agent = RiskAssessmentAgent()
    
    # Define analysis targets
    targets = [
        ("algonquin_il", "electricity"),
        ("algonquin_il", "water"),
        ("algonquin_il", "waste")
    ]
    
    results = {}
    for site_id, category in targets:
        print(f"Analyzing {site_id} - {category}...")
        try:
            result = agent.analyze_site_performance(site_id, category)
            risk_level = result['risk_assessment']['risk_level']
            gap = result['risk_assessment']['gap_percentage']
            results[f"{site_id}_{category}"] = {
                'risk_level': risk_level,
                'gap_percentage': gap
            }
            print(f"  Result: {risk_level} ({gap}% gap)")
        except Exception as e:
            print(f"  Error: {e}")
    
    return results

if __name__ == "__main__":
    run_batch_analysis()
EOF

# Run batch analysis
python3 run_batch_assessment.py
```

### Parameters

#### Available Site IDs
Currently configured test sites:
- `algonquin_il` - Algonquin IL Facility (primary test site)

#### Available Categories
- `electricity` - Electrical consumption analysis (kWh/CO2e)
- `water` - Water consumption analysis (gallons)
- `waste` - Waste generation analysis (lbs)

#### Agent Configuration Parameters
```python
# Agent initialization options
agent = RiskAssessmentAgent(
    neo4j_client=None,           # Optional: custom Neo4j client
    openai_api_key=None,         # Optional: custom API key (defaults to env)
    model_name="gpt-4",          # Optional: LLM model selection
    temperature=0,               # Optional: LLM response consistency
    co2e_factor=0.000395         # Optional: custom CO2e conversion factor
)

# Analysis method parameters
result = agent.analyze_site_performance(
    site_id="algonquin_il",      # Required: target site identifier
    category="electricity"        # Required: consumption category
)
```

### Expected Output

#### Successful Analysis Result
```python
{
    'trend_analysis': {
        'overall_trend': 'increasing',
        'monthly_change_rate': 0.025,
        'seasonal_pattern': 'summer_peak',
        'confidence_level': 0.85,
        'analysis_text': 'Consumption shows a consistent 2.5% monthly increase...'
    },
    'goal_comparison': {
        'projected_annual': 850000,
        'target_consumption': 810000,
        'gap_amount': 40000,
        'gap_percentage': 4.9,
        'analysis_text': 'Current trend projects 4.9% over target...'
    },
    'risk_assessment': {
        'risk_level': 'MEDIUM',
        'gap_percentage': 4.9,
        'goal_achievable': True,
        'confidence_score': 0.78,
        'factors': ['Seasonal variation', 'Recent efficiency improvements'],
        'analysis_text': 'Medium risk due to increasing trend...'
    },
    'recommendations': {
        'recommendations': [
            'Implement LED lighting upgrades in high-usage areas',
            'Optimize HVAC scheduling during peak hours',
            'Install smart meters for real-time monitoring',
            'Conduct energy audit of major equipment'
        ]
    }
}
```

#### Console Output Indicators
- **Success**: "Results stored in Neo4j successfully"
- **Data Issues**: "Warning: Limited historical data available"
- **API Issues**: "Error: OpenAI API request failed"
- **Database Issues**: "Error: Neo4j connection failed"

### Troubleshooting

#### Common Issues and Solutions

**1. OpenAI API Key Issues**
```bash
# Error: "No API key provided"
# Solution: Check .env file and verify key is valid
echo $OPENAI_API_KEY
python3 -c "import openai; print('API key configured')"
```

**2. Neo4j Connection Problems**
```bash
# Error: "Failed to connect to Neo4j"
# Solution: Verify Neo4j is running and credentials are correct
sudo systemctl status neo4j  # Linux
brew services list | grep neo4j  # macOS

# Test connection manually
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
driver.verify_connectivity()
print('Neo4j connection successful')
"
```

**3. Missing Dependencies**
```bash
# Error: "ModuleNotFoundError: No module named 'langchain'"
# Solution: Reinstall dependencies
pip install --upgrade langchain langchain-openai neo4j python-dotenv
```

**4. Insufficient Historical Data**
```bash
# Error: "Not enough historical data for analysis"
# Solution: Check data availability in Neo4j
python3 -c "
from database.neo4j_client import Neo4jClient
client = Neo4jClient()
records = client.get_electricity_consumption_data('algonquin_il', '2024-03-01')
print(f'Found {len(records)} consumption records')
"
```

**5. LLM Response Parsing Errors**
```bash
# Error: "Failed to parse LLM response"
# Solution: This is typically a transient issue, retry the analysis
# The enhanced parsing logic should handle most response variations
```

**6. Permission and Path Issues**
```bash
# Error: "Permission denied" or "Module not found"
# Solution: Ensure correct working directory and Python path
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python3 -c "import agents.risk_assessment_agent; print('Import successful')"
```

**7. Memory or Performance Issues**
```bash
# Error: "Out of memory" or slow performance
# Solution: Monitor resource usage and optimize
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Reduce batch size if processing multiple sites
# Consider running analyses sequentially rather than in parallel
```

**8. Database Query Timeouts**
```bash
# Error: "Query timeout" 
# Solution: Optimize Neo4j queries or increase timeout
# Check Neo4j performance
MATCH (n) RETURN COUNT(n)  # Total node count
MATCH ()-[r]-() RETURN COUNT(r)  # Total relationship count
```

#### Debug Mode
Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
agent = RiskAssessmentAgent()
result = agent.analyze_site_performance("algonquin_il", "electricity")
```

#### Getting Help
1. **Check Logs**: Review console output for specific error messages
2. **Verify Data**: Ensure test site has sufficient historical data
3. **Test Components**: Run individual components to isolate issues
4. **Documentation**: Refer to the architecture section for detailed technical information

## Testing

### How to Test the Agent

#### 1. Standalone Testing
```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/src/scripts/
python3 run_risk_assessment.py
```

#### 2. Unit Testing
```bash
# Run comprehensive test suite
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/
python3 -m pytest tests/agents/risk_assessment/ -v
```

#### 3. Integration Testing
```bash
# Test with real Neo4j data
python3 test_risk_assessment_standalone.py
```

### Test Scenarios

#### Test Site Configuration
```python
test_site_id = "algonquin_il"      # Test site with historical data
test_category = "electricity"       # Primary test category
```

#### Expected Outputs
```python
{
    'trend_analysis': {
        'overall_trend': 'increasing',
        'monthly_change_rate': 0.025,
        'confidence_level': 0.85
    },
    'risk_assessment': {
        'risk_level': 'HIGH',
        'gap_percentage': 35.2,
        'goal_achievable': False
    },
    'recommendations': {
        'recommendations': [
            'Implement LED lighting upgrades',
            'Optimize HVAC scheduling'
        ]
    }
}
```

#### Validation Steps
1. **Data Retrieval**: Verify 6-month consumption data is retrieved correctly
2. **Trend Analysis**: Confirm LLM identifies meaningful patterns
3. **Gap Calculation**: Validate projected vs. target calculations
4. **Risk Assignment**: Ensure risk levels align with gap percentages
5. **Neo4j Storage**: Confirm results are stored correctly in database

### Test Data Requirements
- **Historical Data**: Minimum 6 months of consumption data
- **Annual Goals**: Properly configured Goal nodes with target values
- **Site Configuration**: Valid Site nodes with proper relationships

## Known Issues and Solutions

### 1. Gap Percentage Extraction Issue (FIXED)

**Problem**: LLM responses contained gap percentage data in nested JSON structures, causing extraction failures and zero values.

**Root Cause**: 
```python
# Original parsing only looked for direct fields
if 'gap_percentage' in data:
    result['gap_percentage'] = data['gap_percentage']
```

**Solution** (Lines 310-389 in risk_assessment_agent.py):
```python
def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
    """Enhanced parsing handles both flat and nested JSON structures"""
    
    # First check for direct fields (preferred format)
    if 'gap_percentage' in data and data['gap_percentage'] != 0:
        result['gap_percentage'] = float(data['gap_percentage'])
    
    # If direct fields missing or zero, search nested structure
    if result['gap_percentage'] == 0:
        for key, value in data.items():
            if isinstance(value, dict) and 'gap' in key.lower():
                for subkey, subvalue in value.items():
                    if 'percentage' in subkey.lower():
                        result['gap_percentage'] = float(subvalue)
```

**Verification**: Test with complex LLM responses to ensure gap extraction works correctly.

### 2. Date Handling in Neo4j Queries

**Problem**: Neo4j date objects require proper conversion for Python datetime operations.

**Solution**:
```python
# Handle Neo4j date objects properly
timestamp_obj = record["timestamp"]
if hasattr(timestamp_obj, 'to_native'):
    timestamp = timestamp_obj.to_native()
else:
    timestamp = timestamp_obj
```

### 3. CO2e Unit Handling

**Problem**: Inconsistent unit handling between kWh consumption and CO2e emission goals.

**Solution**: Dynamic unit detection and conversion:
```python
if consumption_unit == "tonnes CO2e" and category == "electricity":
    # Use CO2e values for electricity when goal is in CO2e
    co2e_emissions = kwh_amount * self.ELECTRICITY_CO2E_FACTOR
    actual_consumption += co2e_emissions
else:
    # Use raw consumption values for other units
    actual_consumption += month_data.get('amount', 0)
```

### 4. LLM Response Variability

**Problem**: LLM responses can vary in format and structure, affecting parsing reliability.

**Solution**: 
- Robust parsing with fallback mechanisms
- Clear system prompts requesting JSON format
- Validation and default value assignment for critical fields

### 5. Data Quality Validation

**Problem**: Incomplete or missing historical data affects analysis accuracy.

**Solution**:
```python
# Data quality assessment
data_quality = "good" if len(records) >= 24 else "fair"  # 24 = 4 weeks/month * 6 months
```

**Recommendations**:
- Implement data completeness checks before analysis
- Provide warnings for insufficient data
- Consider extending data collection period if needed

---

## Development Notes

**Last Updated**: 2025-09-05  
**Author**: AI Assistant  
**Version**: 1.2 (Fixed gap calculation parsing issue)

**Recent Changes**:
- Enhanced JSON parsing to handle nested LLM responses
- Improved CO2e conversion logic for mixed unit scenarios  
- Added comprehensive error handling and logging
- Updated documentation with architectural details

**Future Enhancements**:
- Support for additional consumption categories (natural gas, renewables)
- Machine learning model integration for trend prediction
- Advanced correlation analysis across multiple sites
- Real-time monitoring and alert capabilities