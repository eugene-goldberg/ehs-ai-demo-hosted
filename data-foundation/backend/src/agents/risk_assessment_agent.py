#!/usr/bin/env python3
"""
Risk Assessment Agent - Simplified Dashboard Implementation

This agent implements the Risk Assessment Agent as specified in the simplified
dashboard action plan (lines 276-341). It provides LLM-based analysis of consumption
data to assess risks of meeting annual EHS reduction goals.

The agent follows the exact method structure defined in the plan:
- analyze_site_performance: Main orchestration method
- llm_analyze_trends: Analyzes 6-month consumption data  
- llm_compare_to_goals: Projects trends vs annual goals
- llm_assess_goal_risk: Determines risk level (LOW/MEDIUM/HIGH/CRITICAL)
- llm_generate_recommendations: Creates specific action items

Author: AI Assistant
Date: 2025-09-01
Updated: 2025-09-05 - Fixed gap calculation parsing issue
"""

import os
import sys
import logging
import json
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from database.neo4j_client import Neo4jClient
    from agents.prompts.risk_assessment_prompts import (
        format_trend_analysis_prompt,
        format_risk_assessment_prompt, 
        format_recommendation_prompt
    )
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskAssessmentAgent:
    """
    Risk Assessment Agent as specified in simplified dashboard action plan
    
    Implements 5-step LLM workflow:
    1. Data Aggregation: Get 6 months consumption + annual goals
    2. Trend Analysis: LLM identifies patterns and trends  
    3. Goal Comparison: LLM projects trends vs annual reduction goals
    4. Risk Assessment: LLM determines likelihood of meeting targets
    5. Recommendations: LLM generates specific actions based on risk
    """
    
    # Standard emission factors for conversions
    ELECTRICITY_CO2E_FACTOR = 0.000395  # tonnes CO2e per kWh (US grid average)
    
    def __init__(self, neo4j_client: Neo4jClient = None, openai_api_key: str = None):
        """Initialize the Risk Assessment Agent"""
        logger.info("Initializing Risk Assessment Agent")
        
        # Initialize Neo4j client
        if neo4j_client:
            self.neo4j_client = neo4j_client
        else:
            self.neo4j_client = Neo4jClient()
        
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key,
            model_name="gpt-4"
        )
        
        logger.info("Risk Assessment Agent initialized successfully")
    
    def analyze_site_performance(self, site_id: str, category: str) -> Dict[str, Any]:
        """
        Complete LLM analysis workflow as specified in action plan (lines 279-306)
        
        Input: 6 months consumption data + annual reduction goals
        Output: Risk assessment + recommendations
        
        Args:
            site_id: Site identifier
            category: EHS category (electricity, water, waste)
            
        Returns:
            Dictionary containing trend_analysis, risk_assessment, recommendations
        """
        logger.info(f"Starting site performance analysis for {site_id}/{category}")
        
        try:
            # Step 1: Aggregate 6 months of data
            historical_data = self.get_6month_consumption_data(site_id, category)
            annual_goal = self.get_annual_reduction_goal(site_id, category)
            
            if not historical_data or "error" in historical_data:
                logger.error(f"Failed to retrieve consumption data")
                return {"error": "Data retrieval failed"}
            
            if not annual_goal or "error" in annual_goal:
                logger.error(f"Failed to retrieve annual goal")
                return {"error": "Goal retrieval failed"}
            
            # Step 2: LLM identifies trends and patterns  
            trend_analysis = self.llm_analyze_trends(historical_data)
            
            # Step 3: LLM compares trends to goals
            goal_comparison = self.llm_compare_to_goals(trend_analysis, annual_goal, historical_data)
            
            # Step 4: LLM assesses risk of missing annual target
            risk_assessment = self.llm_assess_goal_risk(goal_comparison)
            
            # Step 5: LLM generates specific recommendations
            recommendations = self.llm_generate_recommendations(risk_assessment)
            
            # Store results in Neo4j
            self.store_risk_assessment_in_neo4j(site_id, category, risk_assessment)
            self.store_recommendations_in_neo4j(site_id, category, recommendations)
            
            return {
                'trend_analysis': trend_analysis,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_site_performance: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def llm_analyze_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM analyzes 6-month data to identify (lines 308-316):
        - Monthly consumption trends (increasing/decreasing)
        - Seasonal patterns
        - Rate of change calculations
        - Statistical anomalies
        
        Args:
            data: 6 months of consumption data from get_6month_consumption_data
            
        Returns:
            Dictionary with trend analysis results
        """
        logger.info("Analyzing consumption trends with LLM")
        
        try:
            # Format consumption data for LLM
            formatted_data = json.dumps(data.get('monthly_data', {}), indent=2)
            site_name = f"Site {data.get('site_id', 'Unknown')}"
            category = data.get('category', 'unknown')
            
            prompt = format_trend_analysis_prompt(category, site_name, formatted_data)
            
            messages = [
                SystemMessage(content="You are an expert EHS data analyst specializing in consumption trend analysis. Respond in JSON format with quantified metrics."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            
            try:
                trend_analysis = json.loads(response.content)
                # Ensure required fields exist
                if 'overall_trend' not in trend_analysis:
                    trend_analysis['overall_trend'] = 'unknown'
                if 'monthly_change_rate' not in trend_analysis:
                    trend_analysis['monthly_change_rate'] = 0.0
                if 'confidence_level' not in trend_analysis:
                    trend_analysis['confidence_level'] = 0.5
                    
            except json.JSONDecodeError:
                logger.warning("LLM response not in JSON format, parsing manually")
                trend_analysis = {
                    'analysis_text': response.content,
                    'overall_trend': 'unknown',
                    'monthly_change_rate': 0.0,
                    'seasonal_pattern': 'unknown',
                    'confidence_level': 0.5
                }
            
            logger.info(f"Trend analysis completed: {trend_analysis.get('overall_trend', 'unknown')}")
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error in llm_analyze_trends: {e}")
            return {"error": str(e), "overall_trend": "unknown"}
    
    def llm_compare_to_goals(self, trends: Dict[str, Any], annual_goal: Dict[str, Any], historical_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        LLM projects current trends against annual reduction targets (lines 317-323):
        - Calculate projected annual performance based on 6-month trend
        - Determine gap between projection and goal  
        - Assess timeline constraints (months remaining in year)
        
        Args:
            trends: Results from llm_analyze_trends
            annual_goal: Annual reduction goal data
            historical_data: Raw consumption data with CO2e conversions
            
        Returns:
            Dictionary with goal comparison analysis
        """
        logger.info("Comparing trends to annual goals with LLM")
        
        try:
            # Calculate months remaining in year
            current_month = datetime.now().month
            months_remaining = 12 - current_month
            
            # Create goal comparison prompt
            trend_summary = trends.get('analysis_text', json.dumps(trends))
            
            # Calculate actual consumption values from historical data
            actual_consumption = 0
            baseline_consumption = 0
            consumption_unit = annual_goal.get('unit', '%')
            
            if historical_data and historical_data.get('monthly_data'):
                monthly_data = historical_data['monthly_data']
                category = historical_data.get('category', 'electricity')
                
                # Calculate actual consumption based on goal unit
                if consumption_unit == "tonnes CO2e" and category == "electricity":
                    # Use CO2e values for electricity when goal is in CO2e
                    for month_data in monthly_data.values():
                        if 'co2e_emissions' in month_data:
                            actual_consumption += month_data['co2e_emissions']
                        elif 'amount' in month_data:
                            # Convert kWh to CO2e if not already converted
                            kwh_amount = month_data['amount']
                            actual_consumption += kwh_amount * self.ELECTRICITY_CO2E_FACTOR
                else:
                    # Use raw consumption values for other units
                    for month_data in monthly_data.values():
                        actual_consumption += month_data.get('amount', 0)
                
                # Calculate baseline (assume first month as baseline)
                first_month = min(monthly_data.keys())
                if consumption_unit == "tonnes CO2e" and category == "electricity":
                    if 'co2e_emissions' in monthly_data[first_month]:
                        baseline_consumption = monthly_data[first_month]['co2e_emissions']
                    else:
                        baseline_kwh = monthly_data[first_month].get('amount', 0)
                        baseline_consumption = baseline_kwh * self.ELECTRICITY_CO2E_FACTOR
                else:
                    baseline_consumption = monthly_data[first_month].get('amount', 0)
            
            goal_details = {
                'category': annual_goal.get('category', 'unknown'),
                'annual_goal': annual_goal.get('target_value', 0),
                'current_performance': actual_consumption,
                'baseline_consumption': baseline_consumption,
                'months_elapsed': current_month,
                'units': consumption_unit
            }
            
            prompt = format_risk_assessment_prompt(trend_summary, goal_details, months_remaining)
            
            messages = [
                SystemMessage(content="You are an expert EHS analyst comparing consumption trends to reduction goals. Respond in JSON format with projected performance calculations."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            
            try:
                # FIXED: Use improved parsing that handles nested JSON structure
                goal_comparison = self._parse_llm_response(response.content)
                                
            except json.JSONDecodeError:
                logger.warning("LLM response not in JSON format")
                goal_comparison = {
                    'analysis_text': response.content,
                    'goal_achievable': False,
                    'projected_annual_consumption': 0,
                    'gap_percentage': 0,
                    'confidence': 0.5
                }
            
            logger.info(f"Goal comparison completed: achievable={goal_comparison.get('goal_achievable', False)}, gap={goal_comparison.get('gap_percentage', 0):.1f}%")
            return goal_comparison
            
        except Exception as e:
            logger.error(f"Error in llm_compare_to_goals: {e}")
            return {"error": str(e), "goal_achievable": False}
    
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse LLM response handling both flat and nested JSON structures
        
        FIXED: This method resolves the gap calculation issue by properly extracting
        values from nested JSON structures that the LLM returns.
        
        Args:
            response_content: Raw JSON response from LLM
            
        Returns:
            Dictionary with extracted gap_percentage, projected_annual_consumption, etc.
        """
        
        try:
            data = json.loads(response_content)
            
            # Initialize with defaults
            result = {
                'gap_percentage': 0,
                'projected_annual_consumption': 0,
                'goal_achievable': False,
                'confidence': 0.5,
                'analysis_text': response_content
            }
            
            if isinstance(data, dict):
                # First check for direct fields (preferred format)
                if 'gap_percentage' in data and data['gap_percentage'] != 0:
                    result['gap_percentage'] = float(data['gap_percentage'])
                if 'projected_annual_consumption' in data and data['projected_annual_consumption'] != 0:
                    result['projected_annual_consumption'] = float(data['projected_annual_consumption'])
                if 'goal_achievable' in data:
                    result['goal_achievable'] = bool(data['goal_achievable'])
                
                # If direct fields are missing or zero, look in nested structure
                if result['gap_percentage'] == 0 or result['projected_annual_consumption'] == 0:
                    # Look for gap analysis and projected consumption sections
                    for key, value in data.items():
                        if isinstance(value, dict):
                            # Look for gap analysis section
                            if 'gap' in key.lower():
                                for subkey, subvalue in value.items():
                                    if 'percentage' in subkey.lower():
                                        try:
                                            result['gap_percentage'] = float(subvalue)
                                            logger.debug(f"Extracted gap percentage from nested: {result['gap_percentage']}")
                                        except (ValueError, TypeError):
                                            pass
                            
                            # Look for projected consumption section  
                            elif 'projected' in key.lower():
                                for subkey, subvalue in value.items():
                                    if 'consumption' in subkey.lower():
                                        try:
                                            result['projected_annual_consumption'] = float(subvalue)
                                            logger.debug(f"Extracted projected consumption from nested: {result['projected_annual_consumption']}")
                                        except (ValueError, TypeError):
                                            pass
                            
                            # Look for goal achievability
                            elif 'goal' in key.lower():
                                for subkey, subvalue in value.items():
                                    if 'achievable' in subkey.lower():
                                        if isinstance(subvalue, str):
                                            result['goal_achievable'] = subvalue.lower() in ['yes', 'true', 'achievable']
            
            logger.info(f"Parsed LLM response: gap={result['gap_percentage']:.1f}%, projected={result['projected_annual_consumption']:.1f}")
            return result
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {
                'gap_percentage': 0,
                'projected_annual_consumption': 0,
                'goal_achievable': False,
                'confidence': 0.5,
                'analysis_text': response_content,
                'parsing_error': str(e)
            }
    
    def llm_assess_goal_risk(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM determines risk level based on trend vs goal analysis (lines 325-332):
        - LOW: On track to exceed goal (>90% chance)
        - MEDIUM: May miss goal without intervention (50-90% chance)  
        - HIGH: Likely to miss goal significantly (10-50% chance)
        - CRITICAL: Goal impossible without immediate action (<10% chance)
        
        Args:
            comparison: Results from llm_compare_to_goals
            
        Returns:
            Dictionary with risk assessment
        """
        logger.info("Assessing goal achievement risk with LLM")
        
        try:
            # Determine risk based on goal comparison
            goal_achievable = comparison.get('goal_achievable', False)
            gap_percentage = abs(comparison.get('gap_percentage', 0))
            
            # Create risk assessment based on gap analysis
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
            
            analysis_text = comparison.get('analysis_text', '')
            if not analysis_text:
                analysis_text = f"Gap analysis shows {gap_percentage}% deviation from target"
            
            risk_assessment = {
                'risk_level': risk_level,
                'risk_probability': risk_probability,
                'gap_percentage': gap_percentage,
                'goal_achievable': goal_achievable,
                'analysis_text': analysis_text,
                'assessment_date': datetime.now().isoformat(),
                'confidence_score': comparison.get('confidence', 0.8)
            }
            
            logger.info(f"Risk assessment completed: {risk_level} ({risk_probability:.0%} chance)")
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error in llm_assess_goal_risk: {e}")
            return {"error": str(e), "risk_level": "HIGH"}
    
    def llm_generate_recommendations(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM provides specific actions to get back on track (lines 334-340):
        - Immediate actions for critical risks
        - Medium-term strategies for high/medium risks  
        - Optimization suggestions for low risks
        
        Args:
            risk_assessment: Results from llm_assess_goal_risk
            
        Returns:
            Dictionary with specific recommendations
        """
        logger.info("Generating recommendations with LLM")
        
        try:
            risk_level = risk_assessment.get('risk_level', 'MEDIUM')
            gap_percentage = risk_assessment.get('gap_percentage', 0)
            
            # Create recommendation prompt parameters
            site_name = "Site"  # Could be enhanced with actual site name
            site_type = "Manufacturing"  # Could be enhanced with actual site type
            category = "consumption"  # Could be enhanced with actual category
            goal_gap = f"{gap_percentage}%"
            risk_factors = f"Risk level: {risk_level}"
            months_remaining = 12 - datetime.now().month
            units = "units"
            
            prompt = format_recommendation_prompt(
                category, site_name, site_type, risk_level,
                goal_gap, risk_factors, months_remaining, units
            )
            
            messages = [
                SystemMessage(content="You are an expert EHS consultant providing actionable recommendations. Respond in JSON format with structured recommendations."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            
            try:
                recommendations = json.loads(response.content)
                if 'recommendations' not in recommendations:
                    # Parse text response into structured format
                    recommendations = {
                        'recommendations': [response.content],
                        'priority': risk_level.lower(),
                        'generated_date': datetime.now().isoformat()
                    }
            except json.JSONDecodeError:
                logger.warning("LLM response not in JSON format")
                recommendations = {
                    'recommendations': [response.content],
                    'priority': risk_level.lower(),
                    'generated_date': datetime.now().isoformat()
                }
            
            logger.info(f"Generated {len(recommendations.get('recommendations', []))} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in llm_generate_recommendations: {e}")
            return {"error": str(e), "recommendations": []}
    
    # Helper Methods as specified in the plan
    
    def get_6month_consumption_data(self, site_id: str, category: str) -> Dict[str, Any]:
        """
        Retrieve 6 months of consumption data from Neo4j
        
        Args:
            site_id: Site identifier  
            category: EHS category (electricity, water, waste)
            
        Returns:
            Dictionary containing 6 months of consumption data
        """
        logger.info(f"Retrieving 6-month consumption data for {site_id}/{category}")
        
        # Calculate 6 months ago
        six_months_ago = datetime.now() - timedelta(days=180)
        
        try:
            with self.neo4j_client.session_scope() as session:
                # Category-specific Neo4j queries as specified in plan
                if category == "electricity":
                    query = """
                    MATCH (s:Site {id: $site_id})-[:HAS_ELECTRICITY_CONSUMPTION]->(e:ElectricityConsumption)
                    WHERE e.date >= date($start_date)
                    RETURN e.date as timestamp, e.consumption_kwh as amount, 'kWh' as unit, e.cost_usd as cost
                    ORDER BY e.date ASC
                    """
                elif category == "water":
                    query = """
                    MATCH (s:Site {id: $site_id})-[:HAS_WATER_CONSUMPTION]->(w:WaterConsumption)
                    WHERE w.date >= date($start_date)
                    RETURN w.date as timestamp, w.consumption_gallons as amount, 'gallons' as unit, w.cost_usd as cost
                    ORDER BY w.date ASC
                    """
                elif category == "waste":
                    query = """
                    MATCH (s:Site {id: $site_id})-[:HAS_WASTE_GENERATION]->(w:WasteGeneration)
                    WHERE w.date >= date($start_date)
                    RETURN w.date as timestamp, w.quantity_lbs as amount, 'lbs' as unit, w.cost_usd as cost
                    ORDER BY w.date ASC
                    """
                else:
                    logger.error(f"Unknown category: {category}")
                    return {"error": f"Unknown category: {category}"}
                
                result = session.run(query, {
                    "site_id": site_id,
                    "start_date": six_months_ago.strftime('%Y-%m-%d')
                })
                
                records = list(result)
                
                if not records:
                    logger.warning(f"No consumption data found for {site_id}/{category}")
                    return {}
                
                # Process data into monthly aggregations
                monthly_data = {}
                total_amount = 0
                total_cost = 0
                total_co2e = 0
                
                for record in records:
                    # Handle Neo4j date objects properly as specified
                    timestamp_obj = record["timestamp"]
                    if hasattr(timestamp_obj, 'to_native'):
                        timestamp = timestamp_obj.to_native()
                    else:
                        timestamp = timestamp_obj
                    
                    month_key = timestamp.strftime("%Y-%m")
                    amount = float(record["amount"]) if record["amount"] else 0
                    cost = float(record["cost"]) if record["cost"] else 0
                    
                    # Calculate CO2e emissions for electricity
                    co2e_emissions = 0
                    if category == "electricity" and amount > 0:
                        co2e_emissions = amount * self.ELECTRICITY_CO2E_FACTOR
                    
                    if month_key not in monthly_data:
                        monthly_data[month_key] = {
                            "amount": 0,
                            "cost": 0,
                            "count": 0,
                            "unit": record["unit"]
                        }
                        # Add CO2e emissions for electricity
                        if category == "electricity":
                            monthly_data[month_key]["co2e_emissions"] = 0
                            monthly_data[month_key]["co2e_unit"] = "tonnes CO2e"
                    
                    monthly_data[month_key]["amount"] += amount
                    monthly_data[month_key]["cost"] += cost
                    monthly_data[month_key]["count"] += 1
                    
                    if category == "electricity":
                        monthly_data[month_key]["co2e_emissions"] += co2e_emissions
                    
                    total_amount += amount
                    total_cost += cost
                    total_co2e += co2e_emissions

                result_data = {
                    "site_id": site_id,
                    "category": category,
                    "period": "6_months",
                    "start_date": six_months_ago.strftime('%Y-%m-%d'),
                    "end_date": datetime.now().strftime('%Y-%m-%d'),
                    "monthly_data": monthly_data,
                    "totals": {
                        "amount": total_amount,
                        "cost": total_cost,
                        "records": len(records)
                    },
                    "data_quality": "good" if len(records) >= 24 else "fair"
                }
                
                # Add total CO2e for electricity
                if category == "electricity":
                    result_data["totals"]["co2e_emissions"] = total_co2e
                    result_data["totals"]["co2e_unit"] = "tonnes CO2e"
                
                return result_data
                
        except Exception as e:
            logger.error(f"Error retrieving consumption data: {e}")
            return {"error": str(e)}
    
    def get_annual_reduction_goal(self, site_id: str, category: str) -> Dict[str, Any]:
        """Retrieve annual reduction goal from Neo4j for a specific site and category."""
        try:
            with self.neo4j_client.session_scope() as session:
                # Query for goal that applies to this site
                query = """
                MATCH (g:Goal {category: $category})-[:APPLIES_TO]->(s:Site {id: $site_id})
                RETURN g.target_value as target_value, g.unit as unit, 
                       g.period as period, g.target_date as target_date
                """
                
                result = session.run(query, {
                    "site_id": site_id,
                    "category": category  # Already "electricity", "water", or "waste"
                })
                
                record = result.single()
                if not record:
                    logger.warning(f"No goal found for site {site_id}, category {category}")
                    return {}
                
                return {
                    'target_value': record['target_value'],
                    'unit': record['unit'],
                    'period': record['period'],
                    'target_date': str(record['target_date']) if record['target_date'] else None,
                    'category': category
                }
                
        except Exception as e:
            logger.error(f"Error retrieving goal: {e}")
            return {}
    
    def store_risk_assessment_in_neo4j(self, site_id: str, category: str, assessment: Dict[str, Any]) -> bool:
        """
        Store risk assessment results in Neo4j as RiskAssessment nodes.
        Deletes existing risk assessments for the same site and category before creating new one.
        
        Args:
            site_id: Site identifier
            category: EHS category
            assessment: Risk assessment results
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Storing risk assessment for {site_id}/{category}")
        
        try:
            with self.neo4j_client.session_scope() as session:
                query = """
                MATCH (s:Site {id: $site_id})
                // Delete existing risk assessments for this site and category
                OPTIONAL MATCH (s)-[r:HAS_RISK]->(existing:RiskAssessment {category: $category})
                DELETE r, existing
                // Create new risk assessment
                CREATE (ra:RiskAssessment {
                    id: $assessment_id,
                    site_id: $site_id,
                    category: $category,
                    risk_level: $risk_level,
                    description: $description,
                    assessment_date: datetime($assessment_date),
                    factors: $factors,
                    confidence_score: $confidence_score
                })
                CREATE (s)-[:HAS_RISK]->(ra)
                RETURN ra.id as id
                """
                
                assessment_id = f"{site_id}_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                result = session.run(query, {
                    "site_id": site_id,
                    "assessment_id": assessment_id,
                    "category": category,
                    "risk_level": assessment.get('risk_level', 'MEDIUM'),
                    "description": assessment.get('analysis_text', ''),
                    "assessment_date": assessment.get('assessment_date', datetime.now().isoformat()),
                    "factors": [f"Gap: {assessment.get('gap_percentage', 0)}%"],
                    "confidence_score": assessment.get('confidence_score', 0.5)
                })
                
                record = result.single()
                if record:
                    logger.info(f"Stored risk assessment: {record['id']}")
                    return True
                else:
                    logger.error("Failed to store risk assessment")
                    return False
                    
        except Exception as e:
            logger.error(f"Error storing risk assessment: {e}")
            return False
    
    def store_recommendations_in_neo4j(self, site_id: str, category: str, recommendations: Dict[str, Any]) -> bool:
        """
        Store recommendations in Neo4j as Recommendation nodes.
        Deletes existing recommendations for the same site and category before creating new ones.
        
        Args:
            site_id: Site identifier
            category: EHS category
            recommendations: Recommendation results
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Storing recommendations for {site_id}/{category}")
        
        try:
            with self.neo4j_client.session_scope() as session:
                # First, delete existing recommendations for this site and category
                delete_query = """
                MATCH (s:Site {id: $site_id})-[r:HAS_RECOMMENDATION]->(existing:Recommendation {category: $category})
                DELETE r, existing
                """
                
                session.run(delete_query, {
                    "site_id": site_id,
                    "category": category
                })
                
                logger.info(f"Deleted existing recommendations for {site_id}/{category}")
                
                # Then create new recommendations
                for i, rec in enumerate(recommendations.get('recommendations', [])):
                    query = """
                    MATCH (s:Site {id: $site_id})
                    CREATE (r:Recommendation {
                        id: $recommendation_id,
                        site_id: $site_id,
                        title: $title,
                        description: $description,
                        priority: $priority,
                        estimated_impact: $estimated_impact,
                        category: $category,
                        created_date: datetime($created_date)
                    })
                    CREATE (s)-[:HAS_RECOMMENDATION]->(r)
                    RETURN r.id as id
                    """
                    
                    recommendation_id = f"{site_id}_{category}_rec_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Handle both string and dict recommendation formats
                    if isinstance(rec, str):
                        title = f"Recommendation {i+1}"
                        description = rec
                        priority = recommendations.get('priority', 'medium')
                        estimated_impact = "TBD"
                    else:
                        title = rec.get('title', f"Recommendation {i+1}")
                        description = rec.get('description', str(rec))
                        priority = rec.get('priority', 'medium')
                        estimated_impact = rec.get('estimated_impact', 'TBD')
                    
                    result = session.run(query, {
                        "site_id": site_id,
                        "recommendation_id": recommendation_id,
                        "title": title,
                        "description": description,
                        "priority": priority,
                        "estimated_impact": estimated_impact,
                        "category": category,
                        "created_date": recommendations.get('generated_date', datetime.now().isoformat())
                    })
                    
                    record = result.single()
                    if record:
                        logger.info(f"Stored recommendation: {record['id']}")
                    else:
                        logger.warning(f"Failed to store recommendation {i+1}")
                
                return True
                    
        except Exception as e:
            logger.error(f"Error storing recommendations: {e}")
            return False

def main():
    """Test function for the Risk Assessment Agent"""
    
    # Test configuration
    test_site_id = "SITE_001"
    test_category = "electricity"
    
    try:
        logger.info("Starting Risk Assessment Agent test")
        
        # Initialize agent
        agent = RiskAssessmentAgent()
        
        # Run site performance analysis
        result = agent.analyze_site_performance(test_site_id, test_category)
        
        # Print results
        print("\n" + "="*60)
        print("RISK ASSESSMENT AGENT - SITE PERFORMANCE ANALYSIS")
        print("="*60)
        print(f"Site ID: {test_site_id}")
        print(f"Category: {test_category}")
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"\nTrend Analysis:")
            trend = result.get('trend_analysis', {})
            print(f"  Overall Trend: {trend.get('overall_trend', 'Unknown')}")
            print(f"  Monthly Change: {trend.get('monthly_change_rate', 0):.1%}")
            
            print(f"\nRisk Assessment:")
            risk = result.get('risk_assessment', {})
            print(f"  Risk Level: {risk.get('risk_level', 'Unknown')}")
            print(f"  Probability: {risk.get('risk_probability', 0):.0%}")
            print(f"  Gap: {risk.get('gap_percentage', 0):.1f}%")
            
            print(f"\nRecommendations:")
            recommendations = result.get('recommendations', {}).get('recommendations', [])
            for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                rec_text = rec if isinstance(rec, str) else str(rec)
                print(f"  {i}. {rec_text[:100]}...")
        
        print("="*60)
        logger.info("Risk Assessment Agent test completed")
        
    except Exception as e:
        logger.error(f"Error in main test: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()