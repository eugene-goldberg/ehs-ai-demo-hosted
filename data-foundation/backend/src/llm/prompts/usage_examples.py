"""
Usage examples for the Environmental Prompts module.
Demonstrates how to use the prompt templates for various environmental analysis tasks.
"""

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


def example_electricity_analysis():
    """Example of electricity consumption analysis prompt generation."""
    print("=== ELECTRICITY ANALYSIS EXAMPLE ===")
    
    # Sample electricity data
    electricity_data = """
    Monthly Electricity Bill - Manufacturing Facility
    
    Account: ACME Manufacturing - Building A
    Billing Period: January 1-31, 2024
    Service Address: 123 Industrial Blvd, Manufacturing District
    
    Consumption Summary:
    - Total kWh Consumed: 87,450 kWh
    - Peak Demand: 340 kW (occurred on Jan 15 at 2:00 PM)
    - Power Factor: 0.92
    - Previous Month (December): 82,100 kWh
    
    Rate Structure:
    - Energy Charge: $0.0845 per kWh
    - Demand Charge: $18.50 per kW
    - Power Factor Penalty: $125.00
    - Total Bill: $8,547.23
    
    Usage Pattern:
    - Peak Hours (12-6 PM): 35% of total consumption
    - Off-Peak Hours: 65% of total consumption
    - Weekend Usage: 12% of total consumption
    """
    
    # Create prompt using convenience function
    prompts = create_electricity_analysis_prompt(
        consumption_data=electricity_data,
        facility_name="ACME Manufacturing Building A",
        time_period="January 2024",
        output_format=OutputFormat.JSON
    )
    
    print("System Prompt:")
    print("-" * 50)
    print(prompts["system"][:300] + "...")
    print("\nUser Prompt:")
    print("-" * 50)
    print(prompts["user"][:500] + "...")
    print()


def example_multi_domain_analysis():
    """Example of cross-domain environmental analysis."""
    print("=== MULTI-DOMAIN ANALYSIS EXAMPLE ===")
    
    # Sample multi-domain data
    multi_domain_data = """
    ACME Manufacturing - Q1 2024 Environmental Dashboard
    
    ELECTRICITY:
    - Total Consumption: 245,600 kWh
    - Average Monthly: 81,867 kWh
    - Peak Demand: 385 kW
    - Total Cost: $24,890
    - YoY Change: +7.2%
    
    WATER:
    - Total Consumption: 2.4 million gallons
    - Average Monthly: 800,000 gallons
    - Peak Usage Day: March 15 (35,000 gallons)
    - Total Cost: $18,450
    - Wastewater Generated: 1.8 million gallons
    - YoY Change: +3.8%
    
    WASTE:
    - Total Generated: 45.6 tons
    - Recycled: 28.3 tons (62% diversion rate)
    - Landfill: 17.3 tons
    - Hazardous: 2.1 tons (properly disposed)
    - Total Disposal Cost: $8,950
    - YoY Change: -5.2% (improvement)
    
    OPERATIONAL CONTEXT:
    - Production increased 12% compared to Q1 2023
    - New efficiency equipment installed in February
    - Implemented waste reduction program in January
    """
    
    # Create context
    context = PromptContext(
        facility_name="ACME Manufacturing",
        time_period="Q1 2024",
        data_types=["electricity", "water", "waste"],
        analysis_goals=[
            "Identify correlations between resource consumption",
            "Assess environmental efficiency improvements",
            "Optimize integrated resource management"
        ]
    )
    
    # Create prompt using class method
    templates = EnvironmentalPromptTemplates()
    prompts = templates.create_consumption_analysis_prompt(
        AnalysisType.CROSS_DOMAIN,
        multi_domain_data,
        context,
        OutputFormat.JSON,
        include_examples=True
    )
    
    print("System Prompt:")
    print("-" * 50)
    print(prompts["system"][:300] + "...")
    print("\nUser Prompt:")
    print("-" * 50)
    print(prompts["user"][:600] + "...")
    print()


def example_fact_extraction():
    """Example of fact extraction from environmental data."""
    print("=== FACT EXTRACTION EXAMPLE ===")
    
    # Sample utility bill content
    utility_bill_content = """
    Pacific Electric Utility Bill
    Account: 789456123
    Service Period: February 1-28, 2024
    
    Current Reading: 245,890 kWh
    Previous Reading: 201,340 kWh
    Usage: 44,550 kWh
    
    Charges:
    Energy Charge (44,550 kWh × $0.0892): $3,974.26
    Basic Service Charge: $45.00
    Peak Demand Charge (156 kW × $16.75): $2,613.00
    Power Factor Adjustment: -$125.30 (Credit)
    Environmental Surcharge: $267.30
    
    Total Current Charges: $6,774.26
    Previous Balance: $6,234.18
    Payment Received: $6,234.18
    Current Amount Due: $6,774.26
    Due Date: March 25, 2024
    
    Usage Comparison:
    This Month: 44,550 kWh
    Last Month: 41,230 kWh
    Same Month Last Year: 47,890 kWh
    """
    
    # Create fact extraction prompt
    templates = EnvironmentalPromptTemplates()
    prompts = templates.create_fact_extraction_prompt(
        data_content=utility_bill_content,
        extraction_type="consumption_facts",
        specific_focus=["cost efficiency", "usage trends", "billing anomalies"]
    )
    
    print("System Prompt:")
    print("-" * 50)
    print(prompts["system"][:300] + "...")
    print("\nUser Prompt:")
    print("-" * 50)
    print(prompts["user"][:600] + "...")
    print()


def example_temporal_analysis():
    """Example of temporal trend analysis."""
    print("=== TEMPORAL ANALYSIS EXAMPLE ===")
    
    # Sample time series data
    temporal_data = """
    Water Consumption Trends - Office Complex
    12-Month Historical Data (March 2023 - February 2024)
    
    Monthly Consumption (thousands of gallons):
    Mar 2023: 145.2    Apr 2023: 138.7    May 2023: 152.3
    Jun 2023: 168.9    Jul 2023: 189.2    Aug 2023: 201.4
    Sep 2023: 176.8    Oct 2023: 157.3    Nov 2023: 142.8
    Dec 2023: 134.5    Jan 2024: 139.2    Feb 2024: 148.7
    
    Key Events:
    - May 2023: Cooling system upgrade completed
    - July 2023: Peak summer demand period
    - September 2023: Irrigation system leak detected and repaired
    - November 2023: Water conservation program launched
    - January 2024: New low-flow fixtures installed
    
    Operational Factors:
    - Building Occupancy: 850-900 employees
    - HVAC System: Evaporative cooling (seasonal)
    - Landscaping: 2.5 acres of irrigated grounds
    - Cafeteria: Full-service kitchen (250 meals/day)
    """
    
    # Create context for temporal analysis
    context = PromptContext(
        facility_name="Office Complex",
        time_period="March 2023 - February 2024",
        data_types=["water consumption"],
        baseline_period="March-May 2023",
        target_metrics=["seasonal patterns", "conservation effectiveness", "operational efficiency"]
    )
    
    # Create temporal analysis prompt
    templates = EnvironmentalPromptTemplates()
    prompts = templates.create_temporal_analysis_prompt(
        consumption_data=temporal_data,
        context=context,
        output_format=OutputFormat.JSON
    )
    
    print("System Prompt:")
    print("-" * 50)
    print(prompts["system"][:300] + "...")
    print("\nUser Prompt:")  
    print("-" * 50)
    print(prompts["user"][:600] + "...")
    print()


def example_custom_analysis():
    """Example of custom analysis with specific instructions."""
    print("=== CUSTOM ANALYSIS EXAMPLE ===")
    
    # Sample waste data
    waste_data = """
    Manufacturing Waste Audit - Q4 2024
    
    Waste Categories (tons):
    - Metal Scrap: 12.5 (100% recycled)
    - Plastic Waste: 8.3 (65% recycled, 35% landfill)
    - Paper/Cardboard: 4.7 (95% recycled)
    - Electronic Waste: 2.1 (100% certified disposal)
    - Chemical Waste: 1.8 (100% hazardous disposal)
    - Food Waste: 6.2 (80% composted)
    - Mixed Refuse: 9.4 (100% landfill)
    
    Disposal Costs:
    - Recycling Programs: $2,450
    - Hazardous Disposal: $3,200
    - Landfill Fees: $1,890
    - Composting: $340
    Total: $7,880
    
    Vendor Performance:
    - RecycleCorp: Metal and plastic (95% reliability)
    - EcoWaste Solutions: Electronics (100% certified)
    - GreenCycle: Organics (92% diversion rate)
    """
    
    # Custom analysis instructions
    custom_instructions = """
    Perform a circular economy assessment focusing on:
    1. Calculate the current circular economy score (0-100 scale)
    2. Identify waste streams with the highest improvement potential
    3. Estimate cost savings from achieving 90% overall diversion rate
    4. Recommend specific circular economy strategies for each major waste type
    5. Prioritize recommendations by ROI and implementation difficulty
    6. Assess vendor performance and optimization opportunities
    7. Calculate environmental impact reduction potential (CO2 equivalent)
    
    Use circular economy principles:
    - Reduce: Source reduction opportunities
    - Reuse: Internal reuse potential
    - Recycle: Recycling optimization
    - Recover: Energy/material recovery options
    
    Provide specific, actionable recommendations with quantified benefits.
    """
    
    # Create context
    context = PromptContext(
        facility_name="Manufacturing Plant",
        time_period="Q4 2024",
        analysis_goals=["circular economy optimization", "cost reduction", "sustainability improvement"]
    )
    
    # Create custom prompt
    templates = EnvironmentalPromptTemplates()
    prompts = templates.create_custom_prompt(
        custom_instructions=custom_instructions,
        consumption_data=waste_data,
        context=context,
        analysis_type=AnalysisType.WASTE,
        output_format=OutputFormat.JSON
    )
    
    print("System Prompt:")
    print("-" * 50)
    print(prompts["system"][:300] + "...")
    print("\nUser Prompt:")
    print("-" * 50)
    print(prompts["user"][:800] + "...")
    print()


def demonstrate_json_schema_validation():
    """Demonstrate JSON schema validation functionality."""
    print("=== JSON SCHEMA VALIDATION EXAMPLE ===")
    
    templates = EnvironmentalPromptTemplates()
    
    # Sample LLM response for validation
    sample_response = {
        "analysis_summary": {
            "total_consumption": 45000,
            "consumption_unit": "kWh", 
            "time_period": "January 2024",
            "consumption_trend": "increasing",
            "key_findings": [
                "7.1% increase compared to previous month",
                "Peak demand optimization opportunity identified"
            ]
        },
        "recommendations": [
            {
                "category": "Peak Demand Management",
                "recommendation": "Implement load scheduling during high-cost periods",
                "priority": "high",
                "estimated_savings": "$200-400/month",
                "implementation_effort": "medium",
                "timeframe": "30-60 days"
            }
        ]
    }
    
    # Validate against consumption analysis schema
    validation_result = templates.validate_schema_compliance(
        sample_response, 
        "consumption_analysis"
    )
    
    print("Validation Result:")
    print("-" * 50)
    print(f"Valid: {validation_result['valid']}")
    if validation_result['valid']:
        print(f"Message: {validation_result['message']}")
    else:
        print(f"Error: {validation_result['error']}")
    print()


def show_available_templates():
    """Display all available prompt templates."""
    print("=== AVAILABLE PROMPT TEMPLATES ===")
    
    templates = EnvironmentalPromptTemplates()
    available = templates.get_available_templates()
    
    for category, template_list in available.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for template in template_list:
            print(f"  • {template}")
    print()


if __name__ == "__main__":
    """Run all examples."""
    print("Environmental Prompts Module - Usage Examples")
    print("=" * 60)
    print()
    
    # Show available templates
    show_available_templates()
    
    # Run examples
    example_electricity_analysis()
    example_multi_domain_analysis()
    example_fact_extraction()
    example_temporal_analysis()
    example_custom_analysis()
    demonstrate_json_schema_validation()
    
    print("All examples completed successfully!")