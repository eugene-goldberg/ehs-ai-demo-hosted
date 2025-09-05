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

# Risk Assessment Prompt (lines 449-466) - Updated to include CO2e and baseline data
RISK_ASSESSMENT_PROMPT = """
Based on the trend analysis below, assess the risk of missing the annual {category} goal:

Trend Summary: {trend_summary}
Annual Goal: {annual_goal} {units}
Current Performance: {current_performance} {units} ({months_elapsed} months)
Baseline Consumption: {baseline_consumption} {units}
Months Remaining: {months_remaining}

Calculate the projected annual performance and assign a risk level:

Risk Levels:
- LOW: >90% chance of meeting goal (projected performance within 10% of goal)
- MEDIUM: 50-90% chance of meeting goal (projected performance 10-25% from goal)
- HIGH: 10-50% chance of meeting goal (projected performance 25-50% from goal)
- CRITICAL: <10% chance of meeting goal (projected performance >50% from goal)

Provide:
1. Projected annual consumption based on current trend
2. Gap analysis (difference from goal in absolute terms and percentage)
3. Risk level assignment with justification
4. Key risk factors considering baseline vs current performance
5. Assessment of whether goal is achievable given current trend

Note: For CO2e targets, consider that the goal is typically a reduction target (e.g., -10% from baseline).
For percentage-based goals like "-10%", calculate the target as: baseline * (1 + goal_percentage/100).
"""

# Recommendation Generation Prompt (lines 469-493) - Enhanced with industrial best practices
RECOMMENDATION_GENERATION_PROMPT = """
Generate 3-5 specific, actionable recommendations for {site_name} ({site_type}) to address {category} consumption concerns.

Context:
- Risk Level: {risk_level}
- Goal Gap: {goal_gap} {units}
- Key Risk Factors: {risk_factors}
- Months Remaining: {months_remaining}

INDUSTRIAL BEST PRACTICES TO CONSIDER:

1. **Energy Efficiency Measures**
   - Advanced lighting systems (LED upgrades, smart controls)
   - Building envelope optimization (insulation, windows, HVAC systems)
   - Equipment modernization and right-sizing
   - Real-time energy monitoring and management systems
   - Example: General Motors achieved 35% energy reduction across facilities through comprehensive efficiency programs

2. **Process Electrification**
   - Converting combustion-based processes to electric alternatives
   - Electric heating systems, induction heating, electric ovens
   - Electric vehicle fleet conversion
   - Heat pump installations for heating and cooling
   - Example: Cleveland-Cliffs implemented electric arc furnace technology, reducing CO2 emissions by 75% compared to traditional blast furnaces

3. **Renewable Energy Procurement**
   - On-site solar and wind installations
   - Power Purchase Agreements (PPAs) with renewable energy providers
   - Virtual power purchase agreements (VPPAs)
   - Battery energy storage systems integration
   - Example: RMS Company achieved 100% renewable electricity through a combination of on-site solar and renewable energy contracts

4. **Fuel Switching**
   - Natural gas to renewable natural gas conversion
   - Hydrogen adoption for high-temperature processes
   - Biofuels for transportation and heating
   - Electric alternatives to fossil fuel equipment
   - Example: ArcelorMittal is piloting hydrogen-based steelmaking to replace coal-fired processes

5. **Carbon Capture, Utilization, and Storage (CCUS)**
   - Direct air capture systems
   - Point-source carbon capture on industrial processes
   - Carbon utilization in manufacturing processes
   - Partnerships with carbon storage providers
   - Example: Microsoft has committed to removing all historical CO2 emissions through CCUS technologies and nature-based solutions

CITATION REQUIREMENTS:
When generating recommendations, you MUST:
- Reference specific best practices from the list above when applicable
- Include citations to real company examples when making recommendations
- Specify which industrial best practice category each recommendation falls under
- Provide evidence-based justification for proposed solutions

For each recommendation, provide:
1. Priority Level (High/Medium/Low)
2. Specific Action Description
3. Best Practice Category (from the 5 categories above)
4. Estimated Monthly Impact ({units} reduction/improvement)
5. Implementation Effort (Low/Medium/High)
6. Timeline (Immediate/Short-term/Long-term)
7. Resource Requirements
8. Supporting Evidence/Citation (reference to best practice or company example)

Focus Areas by Risk Level:
- LOW: Optimization and efficiency improvements (focus on Energy Efficiency Measures)
- MEDIUM: Targeted interventions and monitoring (combine Energy Efficiency with Process Electrification)
- HIGH: Aggressive conservation measures and system changes (incorporate Renewable Energy Procurement and Fuel Switching)
- CRITICAL: Emergency protocols and immediate action plans (consider all best practices including CCUS for long-term strategy)

Ensure recommendations are:
- Specific to the site type and consumption category
- Quantifiable where possible
- Realistic and implementable
- Aligned with industry best practices and supported by real-world examples
- Include specific citations to best practices or company examples
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
            - baseline_consumption: baseline consumption for comparison
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
        baseline_consumption=goal_details.get('baseline_consumption', 0),
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