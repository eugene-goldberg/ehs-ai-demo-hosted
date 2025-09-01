# Environmental Prompts Module

A comprehensive LLM prompt templates module for environmental facts analysis, specifically designed for analyzing electricity, water, and waste consumption data.

## Overview

This module provides structured prompt templates that can be used with any LLM to analyze environmental consumption data. The templates are designed to extract actionable insights, identify trends, and provide recommendations for sustainability improvements.

## Features

- **Multi-Domain Analysis**: Support for electricity, water, waste, and cross-domain analysis
- **Structured Output**: JSON schemas for consistent, parseable responses
- **LangChain Integration**: Native support for LangChain ChatPromptTemplate
- **Fact Extraction**: Specialized prompts for extracting key insights from raw data
- **Temporal Analysis**: Time-series trend analysis capabilities
- **Expert System Prompts**: Domain-specific expertise built into system prompts
- **Validation**: Schema compliance validation for LLM responses
- **Convenience Functions**: Easy-to-use functions for common scenarios

## Installation

The module is already integrated into the project. Import the components you need:

```python
from src.llm.prompts import (
    EnvironmentalPromptTemplates,
    AnalysisType,
    OutputFormat,
    PromptContext,
    create_electricity_analysis_prompt,
    create_water_analysis_prompt,
    create_waste_analysis_prompt,
    create_multi_domain_analysis_prompt
)
```

## Quick Start

### Simple Electricity Analysis

```python
from src.llm.prompts import create_electricity_analysis_prompt

# Your electricity data
electricity_data = """
Monthly consumption: 45,000 kWh
Peak demand: 120 kW
Total cost: $4,500
Previous month: 42,000 kWh
"""

# Create analysis prompt
prompts = create_electricity_analysis_prompt(
    consumption_data=electricity_data,
    facility_name="Manufacturing Plant A",
    time_period="January 2024"
)

print(prompts["system"])  # System prompt
print(prompts["user"])    # User prompt
```

### Advanced Multi-Domain Analysis

```python
from src.llm.prompts import EnvironmentalPromptTemplates, AnalysisType, PromptContext

# Create templates instance
templates = EnvironmentalPromptTemplates()

# Define context
context = PromptContext(
    facility_name="ACME Manufacturing",
    time_period="Q1 2024",
    data_types=["electricity", "water", "waste"],
    analysis_goals=["identify correlations", "optimize resource usage"]
)

# Create cross-domain analysis prompt
prompts = templates.create_consumption_analysis_prompt(
    AnalysisType.CROSS_DOMAIN,
    your_consumption_data,
    context
)
```

### LangChain Integration

```python
from src.llm.prompts.integration_guide import LangChainEnvironmentalPrompts
from src.llm import get_llm
from langchain.chains import LLMChain

# Initialize components
llm, model_name = get_llm("openai")
langchain_prompts = LangChainEnvironmentalPrompts()

# Create template
template = langchain_prompts.create_simple_chat_template(
    analysis_type=AnalysisType.ELECTRICITY
)

# Create and run chain
chain = LLMChain(llm=llm, prompt=template)
result = chain.run(
    consumption_data=your_data,
    facility_name="Your Facility",
    time_period="Current Period"
)
```

## Analysis Types

### 1. Electricity Analysis (`AnalysisType.ELECTRICITY`)
- Consumption patterns and trends
- Peak demand analysis
- Cost optimization opportunities
- Energy efficiency metrics
- Load balancing recommendations

### 2. Water Analysis (`AnalysisType.WATER`)
- Usage patterns and conservation opportunities
- Leak detection and efficiency metrics
- Seasonal variation analysis
- Water quality and treatment costs
- Reuse and recycling potential

### 3. Waste Analysis (`AnalysisType.WASTE`)
- Waste stream categorization
- Diversion rate optimization
- Circular economy opportunities
- Compliance and disposal costs
- Source reduction strategies

### 4. Cross-Domain Analysis (`AnalysisType.CROSS_DOMAIN`)
- Correlations between resource types
- Integrated sustainability metrics
- Multi-domain optimization
- Trade-off analysis
- Holistic environmental impact

### 5. Temporal Analysis (`AnalysisType.TEMPORAL`)
- Trend identification and forecasting
- Seasonal pattern analysis
- Performance benchmarking
- Conservation measure effectiveness
- Predictive insights

## Output Formats

### JSON Format (Default)
Structured JSON with predefined schemas for consistent parsing:

```json
{
  "analysis_summary": {
    "total_consumption": 45000,
    "consumption_unit": "kWh",
    "consumption_trend": "increasing",
    "key_findings": ["7.1% increase vs previous month"]
  },
  "recommendations": [
    {
      "category": "Peak Demand Management",
      "recommendation": "Implement load scheduling",
      "priority": "high",
      "estimated_savings": "$300-500/month"
    }
  ]
}
```

### Structured Text Format
Organized text with clear sections and headers for easy reading.

### Markdown Format
Well-formatted Markdown with tables, bullet points, and emphasized key findings.

## Advanced Features

### Fact Extraction
Extract structured facts from raw environmental data:

```python
facts_prompt = templates.create_fact_extraction_prompt(
    data_content=raw_bill_data,
    extraction_type="consumption_facts",
    specific_focus=["cost efficiency", "usage trends"]
)
```

### Custom Analysis
Create custom analysis with specific requirements:

```python
custom_prompt = templates.create_custom_prompt(
    custom_instructions="Focus on circular economy opportunities...",
    consumption_data=data,
    context=context
)
```

### Schema Validation
Validate LLM responses against expected schemas:

```python
validation = templates.validate_schema_compliance(
    llm_response_json,
    "consumption_analysis"
)
```

## Expert System Prompts

The module includes specialized system prompts that establish the AI as an expert in:

- **Environmental Sustainability**: General environmental expertise
- **Utility Data Analysis**: Electricity consumption and demand management
- **Water Management**: Water conservation and treatment
- **Waste Management**: Circular economy and waste optimization

## File Structure

```
src/llm/prompts/
├── __init__.py                    # Module exports
├── environmental_prompts.py       # Core templates and classes
├── integration_guide.py          # LangChain integration utilities
├── usage_examples.py             # Comprehensive usage examples
└── README.md                      # This documentation
```

## Usage Examples

See `usage_examples.py` for comprehensive examples covering:
- Basic consumption analysis
- Multi-domain correlations
- Fact extraction
- Temporal trend analysis
- Custom analysis scenarios

## Integration Examples

See `integration_guide.py` for LangChain integration examples:
- ChatPromptTemplate creation
- Chain setup and execution
- Response parsing and validation
- Error handling patterns

## Best Practices

1. **Data Quality**: Ensure consumption data is clean and well-formatted
2. **Context Setting**: Provide facility name, time periods, and analysis goals
3. **Output Validation**: Use schema validation for JSON responses
4. **Domain Expertise**: Choose appropriate analysis types for your data
5. **Iterative Analysis**: Start with broad analysis, then drill down into specifics

## Schema Reference

The module includes predefined JSON schemas for:
- `consumption_analysis`: Single-domain consumption analysis
- `cross_domain_analysis`: Multi-domain correlation analysis
- `temporal_analysis`: Time-series trend analysis

Each schema defines required fields, data types, and validation rules for consistent LLM responses.

## Support

For issues or questions:
1. Check the usage examples in `usage_examples.py`
2. Review integration patterns in `integration_guide.py`
3. Examine the core templates in `environmental_prompts.py`

The module is designed to be modular, extensible, and compatible with the existing project architecture.