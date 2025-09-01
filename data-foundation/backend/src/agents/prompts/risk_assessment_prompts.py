"""
Risk Assessment Prompts for EHS AI Demo

This module contains prompt templates used by the Risk Assessment Agent
to analyze consumption trends, assess risks, and generate recommendations.
"""

# Trend Analysis Prompt (lines 434-446)
TREND_ANALYSIS_PROMPT = """
Analyze the following 6 months of {category} consumption data for {site_name}:

{consumption_data}

Provide a structured analysis including:
1. Overall trend direction (increasing/decreasing/stable)
2. Monthly rate of change (percentage)
3. Seasonal patterns or anomalies
4. Key observations about consumption patterns

Format your response with quantified metrics where possible.
Focus on trends that could impact annual goal achievement.
"""

# Risk Assessment Prompt (lines 449-466)
RISK_ASSESSMENT_PROMPT = """
Based on the trend analysis below, assess the risk of missing the annual {category} goal:

Trend Summary: {trend_summary}
Annual Goal: {annual_goal} {units}
Current Performance: {current_performance} {units} ({months_elapsed} months)
Months Remaining: {months_remaining}

Calculate the projected annual performance and assign a risk level:

Risk Levels:
- LOW: >90% chance of meeting goal (projected performance within 10% of goal)
- MEDIUM: 50-90% chance of meeting goal (projected performance 10-25% from goal)
- HIGH: 10-50% chance of meeting goal (projected performance 25-50% from goal)
- CRITICAL: <10% chance of meeting goal (projected performance >50% from goal)

Provide:
1. Projected annual consumption
2. Gap analysis (difference from goal)
3. Risk level assignment with justification
4. Key risk factors
"""

# Recommendation Generation Prompt (lines 469-493)
RECOMMENDATION_GENERATION_PROMPT = """
Generate 3-5 specific, actionable recommendations for {site_name} ({site_type}) to address {category} consumption concerns.

Context:
- Risk Level: {risk_level}
- Goal Gap: {goal_gap} {units}
- Key Risk Factors: {risk_factors}
- Months Remaining: {months_remaining}

For each recommendation, provide:
1. Priority Level (High/Medium/Low)
2. Specific Action Description
3. Estimated Monthly Impact ({units} reduction/improvement)
4. Implementation Effort (Low/Medium/High)
5. Timeline (Immediate/Short-term/Long-term)
6. Resource Requirements

Focus Areas by Risk Level:
- LOW: Optimization and efficiency improvements
- MEDIUM: Targeted interventions and monitoring
- HIGH: Aggressive conservation measures and system changes
- CRITICAL: Emergency protocols and immediate action plans

Ensure recommendations are:
- Specific to the site type and consumption category
- Quantifiable where possible
- Realistic and implementable
- Aligned with industry best practices
"""

def format_trend_analysis_prompt(category, site_name, consumption_data):
    """
    Format the trend analysis prompt with specific data.
    
    Args:
        category (str): Consumption category (e.g., 'energy', 'water', 'waste')
        site_name (str): Name of the site being analyzed
        consumption_data (str): Formatted consumption data for the last 6 months
        
    Returns:
        str: Formatted prompt ready for LLM processing
    """
    return TREND_ANALYSIS_PROMPT.format(
        category=category,
        site_name=site_name,
        consumption_data=consumption_data
    )

def format_risk_assessment_prompt(trend_summary, goal_details, months_remaining):
    """
    Format the risk assessment prompt with analysis results.
    
    Args:
        trend_summary (str): Summary from trend analysis
        goal_details (dict): Dictionary containing:
            - category: consumption category
            - annual_goal: target value for the year
            - current_performance: actual consumption so far
            - months_elapsed: months completed
            - units: measurement units
        months_remaining (int): Months left in the assessment period
        
    Returns:
        str: Formatted prompt ready for LLM processing
    """
    return RISK_ASSESSMENT_PROMPT.format(
        category=goal_details['category'],
        trend_summary=trend_summary,
        annual_goal=goal_details['annual_goal'],
        current_performance=goal_details['current_performance'],
        months_elapsed=goal_details['months_elapsed'],
        months_remaining=months_remaining,
        units=goal_details['units']
    )

def format_recommendation_prompt(category, site_name, site_type, risk_level, 
                               goal_gap, risk_factors, months_remaining, units):
    """
    Format the recommendation generation prompt with assessment results.
    
    Args:
        category (str): Consumption category
        site_name (str): Name of the site
        site_type (str): Type of site (e.g., 'Manufacturing', 'Office', 'Warehouse')
        risk_level (str): Risk level from assessment (LOW/MEDIUM/HIGH/CRITICAL)
        goal_gap (float): Difference between projected and target performance
        risk_factors (str): Key risk factors identified
        months_remaining (int): Months left in assessment period
        units (str): Measurement units
        
    Returns:
        str: Formatted prompt ready for LLM processing
    """
    return RECOMMENDATION_GENERATION_PROMPT.format(
        site_name=site_name,
        site_type=site_type,
        category=category,
        risk_level=risk_level,
        goal_gap=goal_gap,
        risk_factors=risk_factors,
        months_remaining=months_remaining,
        units=units
    )

# Export all prompts and helper functions for import by Risk Assessment Agent
__all__ = [
    'TREND_ANALYSIS_PROMPT',
    'RISK_ASSESSMENT_PROMPT', 
    'RECOMMENDATION_GENERATION_PROMPT',
    'format_trend_analysis_prompt',
    'format_risk_assessment_prompt',
    'format_recommendation_prompt'
]