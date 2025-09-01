"""
Integration guide for using Environmental Prompts with LangChain.
Shows how to integrate the prompt templates with LangChain's ChatPromptTemplate system.
"""

import json
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage

from src.llm.prompts import (
    EnvironmentalPromptTemplates,
    AnalysisType,
    OutputFormat,
    PromptContext,
    create_electricity_analysis_prompt
)


class LangChainEnvironmentalPrompts:
    """
    Integration wrapper for using Environmental Prompts with LangChain.
    Converts environmental prompt templates to LangChain ChatPromptTemplate format.
    """
    
    def __init__(self):
        """Initialize the LangChain integration."""
        self.templates = EnvironmentalPromptTemplates()
    
    def create_simple_chat_template(
        self,
        analysis_type: AnalysisType,
        output_format: OutputFormat = OutputFormat.JSON
    ) -> ChatPromptTemplate:
        """
        Create a simple LangChain ChatPromptTemplate for environmental analysis.
        Uses basic variable substitution to avoid conflicts with JSON schemas.
        
        Args:
            analysis_type: Type of environmental analysis
            output_format: Desired output format
            
        Returns:
            ChatPromptTemplate ready for use with LangChain
        """
        # Get system prompt
        system_prompt = self.templates.get_system_prompt(analysis_type)
        
        # Create a simplified user template that avoids JSON schema conflicts
        if analysis_type == AnalysisType.ELECTRICITY:
            user_template = """Analyze the following electricity consumption data for {facility_name} during {time_period}.

DATA TO ANALYZE:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Identify total consumption, peak demand patterns, and usage trends
- Calculate energy efficiency metrics and cost per kWh
- Detect any unusual consumption spikes or patterns
- Assess power quality indicators if available
- Evaluate demand charges vs consumption charges
- Identify opportunities for load shifting and peak demand reduction
- Consider renewable energy integration possibilities
- Analyze time-of-use patterns and rate optimization opportunities

OUTPUT FORMAT: Provide your analysis as structured JSON with sections for:
- analysis_summary (total consumption, trends, key findings)
- efficiency_metrics (consumption per unit, efficiency rating)
- cost_analysis (total cost, cost trends, savings opportunities)
- recommendations (category, recommendation, priority, estimated savings)
- environmental_impact (carbon emissions, sustainability score)

Focus on actionable insights that can reduce energy costs and improve sustainability."""

        elif analysis_type == AnalysisType.WATER:
            user_template = """Analyze the following water consumption data for {facility_name} during {time_period}.

DATA TO ANALYZE:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Identify total water consumption and usage patterns
- Calculate water efficiency metrics and cost per gallon/cubic meter
- Detect potential leaks or unusual consumption patterns
- Assess seasonal variations and operational impacts
- Evaluate water quality parameters if available
- Identify water conservation opportunities
- Analyze wastewater generation and treatment costs
- Consider water reuse and recycling potential

OUTPUT FORMAT: Provide your analysis as structured JSON with sections for:
- analysis_summary (total consumption, trends, key findings)
- efficiency_metrics (consumption per unit, efficiency rating)
- cost_analysis (total cost, cost trends, savings opportunities)
- recommendations (category, recommendation, priority, estimated savings)
- environmental_impact (conservation potential, sustainability score)

Focus on water conservation strategies and cost-effective improvements."""

        elif analysis_type == AnalysisType.WASTE:
            user_template = """Analyze the following waste generation and disposal data for {facility_name} during {time_period}.

DATA TO ANALYZE:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Categorize waste streams by type and disposal method
- Calculate waste generation rates and diversion rates
- Identify recycling and composting opportunities
- Assess hazardous waste management compliance
- Evaluate waste disposal costs by category
- Analyze waste-to-landfill ratios and sustainability metrics
- Identify source reduction opportunities
- Consider circular economy and waste-to-energy options

OUTPUT FORMAT: Provide your analysis as structured JSON with sections for:
- analysis_summary (total waste, diversion rates, key findings)
- waste_breakdown (by category, amounts, disposal methods)
- cost_analysis (disposal costs by category, optimization opportunities)
- recommendations (category, recommendation, priority, estimated savings)
- environmental_impact (emissions avoided, circular economy opportunities)

Focus on waste reduction, increased diversion rates, and circular economy opportunities."""

        else:  # Cross-domain or default
            user_template = """Perform a comprehensive cross-domain analysis examining relationships between electricity, water, and waste consumption for {facility_name} during {time_period}.

MULTI-DOMAIN DATA:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Identify correlations between different resource consumption patterns
- Analyze how changes in one domain affect others
- Calculate integrated sustainability metrics
- Assess overall resource efficiency and circular economy potential
- Identify optimization opportunities that benefit multiple domains
- Evaluate trade-offs and synergies between resource management strategies
- Calculate total environmental footprint and carbon equivalent

OUTPUT FORMAT: Provide your analysis as structured JSON with sections for:
- correlation_analysis (relationships between domains)
- integrated_insights (cross-domain findings and opportunities)
- optimization_opportunities (multi-domain improvements)
- environmental_impact (total footprint, integrated sustainability score)
- recommendations (integrated strategies, priorities, estimated impact)

Focus on integrated sustainability strategies that optimize across all domains."""

        # Create LangChain prompt template
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_template)
        ])
        
        return chat_template
    
    def create_fact_extraction_chat_template(
        self,
        extraction_type: str = "consumption_facts"
    ) -> ChatPromptTemplate:
        """
        Create a LangChain ChatPromptTemplate for fact extraction.
        
        Args:
            extraction_type: Type of fact extraction to perform
            
        Returns:
            ChatPromptTemplate for fact extraction
        """
        # Create simplified fact extraction template
        user_template = """Extract key consumption facts from the following environmental data:

{data_content}

Extract the following types of facts:
1. QUANTITATIVE FACTS - Specific numbers, measurements, percentages
2. TEMPORAL FACTS - Time periods, dates, durations, trends over time
3. COMPARATIVE FACTS - Comparisons between periods, benchmarks, targets
4. EFFICIENCY FACTS - Efficiency metrics, ratios, performance indicators
5. COST FACTS - Financial data, cost per unit, budget impacts
6. COMPLIANCE FACTS - Regulatory compliance, permit requirements, violations
7. OPERATIONAL FACTS - Operational changes, equipment performance, usage patterns

Format each fact as:
- Fact Type: [Category from above]
- Statement: [Clear, specific fact statement]
- Value: [Numerical value if applicable]
- Unit: [Unit of measurement if applicable]
- Context: [Additional context or explanation]
- Confidence: [High/Medium/Low based on data quality]

Prioritize facts that are actionable for decision-making and relevant to sustainability goals."""
        
        # Create chat template
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                self.templates.SYSTEM_PROMPTS["base_environmental_expert"]
            ),
            HumanMessagePromptTemplate.from_template(user_template)
        ])
        
        return chat_template
    
    def create_custom_chat_template(
        self,
        system_prompt: str,
        user_template: str
    ) -> ChatPromptTemplate:
        """
        Create a custom ChatPromptTemplate with specified prompts.
        
        Args:
            system_prompt: System prompt text
            user_template: User prompt template with variables
            
        Returns:
            Custom ChatPromptTemplate
        """
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_template)
        ])
        
        return chat_template


def example_langchain_integration():
    """
    Example of how to use Environmental Prompts with LangChain.
    This would typically be used with an LLM from the project's llm.py module.
    """
    print("=== LANGCHAIN INTEGRATION EXAMPLE ===")
    
    # Create the integration wrapper
    langchain_prompts = LangChainEnvironmentalPrompts()
    
    # Create a ChatPromptTemplate for electricity analysis
    electricity_template = langchain_prompts.create_simple_chat_template(
        analysis_type=AnalysisType.ELECTRICITY,
        output_format=OutputFormat.JSON
    )
    
    print("Created LangChain ChatPromptTemplate for electricity analysis")
    print(f"Input variables: {electricity_template.input_variables}")
    
    # Example of formatting the template with actual data
    sample_values = {
        "consumption_data": "Monthly consumption: 45,000 kWh, Peak demand: 120 kW, Cost: $4,500",
        "facility_name": "ACME Manufacturing",
        "time_period": "January 2024"
    }
    
    # Format the prompt (this creates the actual messages to send to LLM)
    formatted_messages = electricity_template.format_messages(**sample_values)
    
    print(f"\nGenerated {len(formatted_messages)} messages:")
    for i, msg in enumerate(formatted_messages):
        print(f"Message {i+1} ({msg.__class__.__name__}): {msg.content[:100]}...")
    
    print("\n" + "="*60)
    
    # Example with fact extraction
    fact_extraction_template = langchain_prompts.create_fact_extraction_chat_template()
    
    print("Created fact extraction template")
    print(f"Input variables: {fact_extraction_template.input_variables}")


def example_with_project_llm():
    """
    Example showing how to use with the project's LLM system.
    This demonstrates the complete integration pattern.
    """
    print("=== COMPLETE INTEGRATION EXAMPLE ===")
    
    # This would be the typical usage pattern in the project
    usage_example = '''
    # Import project LLM and environmental prompts
    from src.llm import get_llm
    from src.llm.prompts.integration_guide import LangChainEnvironmentalPrompts
    from src.llm.prompts import AnalysisType, OutputFormat
    
    # Initialize components  
    llm, model_name = get_llm("openai")  # or your preferred model
    langchain_prompts = LangChainEnvironmentalPrompts()
    
    # Create prompt template for electricity analysis
    electricity_template = langchain_prompts.create_simple_chat_template(
        analysis_type=AnalysisType.ELECTRICITY,
        output_format=OutputFormat.JSON
    )
    
    # Create chain
    from langchain.chains import LLMChain
    analysis_chain = LLMChain(llm=llm, prompt=electricity_template)
    
    # Run analysis
    consumption_data = """
    Electricity Bill Data:
    - Consumption: 87,450 kWh
    - Peak Demand: 340 kW  
    - Total Cost: $8,547.23
    - Previous Month: 82,100 kWh
    """
    
    result = analysis_chain.run(
        consumption_data=consumption_data,
        facility_name="Manufacturing Plant A",
        time_period="January 2024"
    )
    
    # The result will be structured JSON
    print(result)
    
    # Parse and validate the result
    import json
    try:
        parsed_result = json.loads(result)
        print("Successfully parsed LLM response as JSON")
        
        # Extract key insights
        if "analysis_summary" in parsed_result:
            summary = parsed_result["analysis_summary"]
            print(f"Total consumption: {summary.get('total_consumption')} {summary.get('consumption_unit', 'units')}")
            print(f"Trend: {summary.get('consumption_trend')}")
        
        if "recommendations" in parsed_result:
            recommendations = parsed_result["recommendations"]
            print(f"Generated {len(recommendations)} recommendations")
            
    except json.JSONDecodeError:
        print("Response was not valid JSON, treating as text")
    '''
    
    print("Complete integration pattern:")
    print("-" * 40)
    print(usage_example)


def example_structured_output_parsing():
    """
    Example of parsing structured outputs using the environmental prompts.
    """
    print("=== STRUCTURED OUTPUT PARSING EXAMPLE ===")
    
    parsing_example = '''
    # After getting LLM response, validate and parse it
    from src.llm.prompts import EnvironmentalPromptTemplates
    import json
    
    templates = EnvironmentalPromptTemplates()
    
    # Assume we got this response from the LLM
    llm_response = """{
        "analysis_summary": {
            "total_consumption": 87450,
            "consumption_unit": "kWh",
            "time_period": "January 2024", 
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
    }"""
    
    # Parse JSON response
    try:
        parsed_response = json.loads(llm_response)
        
        # Validate against schema (basic validation)
        validation = templates.validate_schema_compliance(
            parsed_response, 
            "consumption_analysis"
        )
        
        if validation["valid"]:
            # Extract specific insights
            consumption = parsed_response["analysis_summary"]["total_consumption"]
            trend = parsed_response["analysis_summary"]["consumption_trend"]
            recommendations = len(parsed_response["recommendations"])
            
            print(f"✓ Valid response: {consumption} kWh, trend: {trend}")
            print(f"✓ Generated {recommendations} recommendations")
            
            # Extract actionable items
            for rec in parsed_response["recommendations"]:
                category = rec.get("category", "General")
                priority = rec.get("priority", "medium")
                savings = rec.get("estimated_savings", "TBD")
                print(f"  - {category} ({priority} priority): {savings} potential savings")
                
        else:
            print(f"✗ Invalid response: {validation['error']}")
            
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
    '''
    
    print("Structured output parsing pattern:")
    print("-" * 40)
    print(parsing_example)


def example_convenience_functions():
    """
    Example of using the convenience functions for quick prompt creation.
    """
    print("=== CONVENIENCE FUNCTIONS EXAMPLE ===")
    
    # Use convenience functions for quick prompt creation
    electricity_data = """
    Monthly Electricity Bill:
    - Consumption: 45,000 kWh
    - Peak Demand: 120 kW
    - Cost: $4,500
    - Previous Month: 42,000 kWh
    """
    
    # Create electricity analysis prompt
    prompts = create_electricity_analysis_prompt(
        consumption_data=electricity_data,
        facility_name="Test Facility",
        time_period="January 2024",
        output_format=OutputFormat.JSON
    )
    
    print("Created electricity analysis prompt using convenience function:")
    print(f"- System prompt length: {len(prompts['system'])} characters")
    print(f"- User prompt length: {len(prompts['user'])} characters")
    print(f"- System prompt preview: {prompts['system'][:100]}...")
    print()
    
    # Show how to use with LangChain
    langchain_usage = '''
    # Convert convenience function output to LangChain template
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    
    prompts = create_electricity_analysis_prompt(electricity_data, "Facility", "Jan 2024")
    
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompts["system"]),
        HumanMessagePromptTemplate.from_template(prompts["user"])
    ])
    
    # Now you can use chat_template with any LLM
    '''
    
    print("LangChain integration with convenience functions:")
    print("-" * 40)
    print(langchain_usage)


if __name__ == "__main__":
    """Run integration examples."""
    print("Environmental Prompts - LangChain Integration Guide")
    print("=" * 60)
    print()
    
    example_langchain_integration()
    example_with_project_llm() 
    example_structured_output_parsing()
    example_convenience_functions()
    
    print("\nIntegration examples completed!")