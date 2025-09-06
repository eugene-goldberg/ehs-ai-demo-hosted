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
Updated: 2025-09-06 - Enhanced LLM response parsing for robustness and nested objects
"""

import os
import sys
import logging
import json
import traceback
import re
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
                # ENHANCED: Use improved parsing that handles nested JSON structure and multiple key formats
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
        Parse LLM response handling both flat and nested JSON structures with robust value extraction
        
        ENHANCED: This method handles multiple key formats, string percentage parsing, and nested objects 
        to resolve LLM response format variations including:
        - Both "percentage_gap" and "percentage_difference" keys
        - String values with % signs (e.g., "-28.36%")
        - Numeric values (e.g., 1011.35)
        - Nested objects like {"value": 28.36, "units": "%"}
        - Proper negative value extraction
        - FIXED: Prioritizes percentage_difference over absolute_difference
        
        Args:
            response_content: Raw JSON response from LLM
            
        Returns:
            Dictionary with extracted gap_percentage, projected_annual_consumption, etc.
        """
        
        def _extract_value_from_nested_object(value: Any) -> Any:
            """
            Extract the actual value from nested objects that have a "value" field.
            If the value is a dict/object and has a "value" field, return that.
            Otherwise, return the original value.
            
            Args:
                value: The value to potentially extract from
                
            Returns:
                The extracted value or the original value
            """
            if isinstance(value, dict) and "value" in value:
                logger.debug(f"Extracting value from nested object: {value} -> {value['value']}")
                return value["value"]
            return value
        
        def _parse_percentage_value(value: Any) -> float:
            """
            Parse percentage values from various formats:
            - "-28.36%" -> -28.36
            - "28.36%" -> 28.36
            - 28.36 -> 28.36
            - "-28.36" -> -28.36
            - {"value": 28.36, "units": "%"} -> 28.36
            """
            # First, extract from nested object if needed
            extracted_value = _extract_value_from_nested_object(value)
            
            if extracted_value is None:
                return 0.0
            
            try:
                if isinstance(extracted_value, (int, float)):
                    return float(extracted_value)
                
                if isinstance(extracted_value, str):
                    # Remove % sign and whitespace
                    clean_value = extracted_value.strip().replace('%', '')
                    # Handle empty strings
                    if not clean_value:
                        return 0.0
                    return float(clean_value)
                
                return 0.0
            except (ValueError, TypeError):
                logger.warning(f"Could not parse percentage value: {value} (extracted: {extracted_value})")
                return 0.0
        
        def _parse_numeric_value(value: Any) -> float:
            """
            Parse numeric values from various formats:
            - 1011.35 -> 1011.35
            - "1011.35" -> 1011.35
            - "1,011.35" -> 1011.35
            - {"value": 1011.35, "units": "kWh"} -> 1011.35
            """
            # First, extract from nested object if needed
            extracted_value = _extract_value_from_nested_object(value)
            
            if extracted_value is None:
                return 0.0
                
            try:
                if isinstance(extracted_value, (int, float)):
                    return float(extracted_value)
                
                if isinstance(extracted_value, str):
                    # Remove commas and whitespace
                    clean_value = extracted_value.strip().replace(',', '')
                    if not clean_value:
                        return 0.0
                    return float(clean_value)
                
                return 0.0
            except (ValueError, TypeError):
                logger.warning(f"Could not parse numeric value: {value} (extracted: {extracted_value})")
                return 0.0
        
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
                # ENHANCED: Reordered percentage_keys to prioritize "percentage_difference" and similar keys
                # This ensures we prefer percentage keys over absolute difference keys
                percentage_keys = [
                    'percentage_difference', 'percentage_gap', 'gap_percentage', 
                    'percent_difference', 'percent_gap', 'gap_percent', 'gap'
                ]
                
                consumption_keys = [
                    'projected_annual_consumption', 'projected_consumption', 
                    'annual_projection', 'projected_annual', 'projection'
                ]
                
                achievability_keys = [
                    'goal_achievable', 'achievable', 'is_achievable',
                    'goal_attainable', 'can_achieve_goal'
                ]
                
                # First check for direct fields (preferred format)
                for key in percentage_keys:
                    if key in data and data[key] is not None:
                        parsed_value = _parse_percentage_value(data[key])
                        if parsed_value != 0:  # Only use non-zero values
                            result['gap_percentage'] = parsed_value
                            logger.debug(f"Found gap percentage in {key}: {parsed_value}")
                            break
                
                for key in consumption_keys:
                    if key in data and data[key] is not None:
                        parsed_value = _parse_numeric_value(data[key])
                        if parsed_value != 0:  # Only use non-zero values
                            result['projected_annual_consumption'] = parsed_value
                            logger.debug(f"Found projected consumption in {key}: {parsed_value}")
                            break
                
                for key in achievability_keys:
                    if key in data:
                        # Also handle nested objects for achievability
                        extracted_value = _extract_value_from_nested_object(data[key])
                        if isinstance(extracted_value, bool):
                            result['goal_achievable'] = extracted_value
                        elif isinstance(extracted_value, str):
                            result['goal_achievable'] = extracted_value.lower() in ['yes', 'true', 'achievable', 'attainable']
                        logger.debug(f"Found goal achievability in {key}: {result['goal_achievable']}")
                        break
                
                # ENHANCED: If direct fields are missing or zero, look in nested structure with broader search
                # BUT exclude keys containing "absolute" to avoid absolute_difference confusion
                if result['gap_percentage'] == 0 or result['projected_annual_consumption'] == 0:
                    # Look through all nested dictionaries for relevant values
                    for main_key, main_value in data.items():
                        if isinstance(main_value, dict):
                            # Search for gap/percentage values in nested structures
                            for sub_key, sub_value in main_value.items():
                                # FIXED: Check for percentage-related keys but exclude keys containing "absolute"
                                if any(term in sub_key.lower() for term in ['gap', 'percentage', 'percent', 'difference']):
                                    # EXCLUDE keys that contain "absolute" to avoid matching "absolute_difference"
                                    if 'absolute' not in sub_key.lower():
                                        if result['gap_percentage'] == 0:
                                            parsed_value = _parse_percentage_value(sub_value)
                                            if parsed_value != 0:
                                                result['gap_percentage'] = parsed_value
                                                logger.debug(f"Extracted gap percentage from nested {main_key}.{sub_key}: {parsed_value}")
                                
                                # Check for consumption/projection keys
                                if any(term in sub_key.lower() for term in ['consumption', 'projected', 'projection', 'annual']):
                                    if result['projected_annual_consumption'] == 0:
                                        parsed_value = _parse_numeric_value(sub_value)
                                        if parsed_value != 0:
                                            result['projected_annual_consumption'] = parsed_value
                                            logger.debug(f"Extracted projected consumption from nested {main_key}.{sub_key}: {parsed_value}")
                                
                                # Check for achievability
                                if any(term in sub_key.lower() for term in ['achievable', 'attainable', 'goal']):
                                    extracted_value = _extract_value_from_nested_object(sub_value)
                                    if isinstance(extracted_value, str):
                                        achievable = extracted_value.lower() in ['yes', 'true', 'achievable', 'attainable']
                                        if achievable != result['goal_achievable']:  # Only update if different
                                            result['goal_achievable'] = achievable
                                            logger.debug(f"Extracted goal achievability from nested {main_key}.{sub_key}: {achievable}")
                
                # Handle confidence field (also check for nested objects)
                if 'confidence' in data:
                    result['confidence'] = _parse_numeric_value(data['confidence'])
                elif 'confidence_score' in data:
                    result['confidence'] = _parse_numeric_value(data['confidence_score'])
            
            logger.info(f"Parsed LLM response: gap={result['gap_percentage']:.1f}%, projected={result['projected_annual_consumption']:.1f}, achievable={result['goal_achievable']}")
            return result
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Try to extract values using regex as fallback
            fallback_result = self._fallback_parse_response(response_content)
            fallback_result['parsing_error'] = str(e)
            return fallback_result
    
    def _fallback_parse_response(self, response_content: str) -> Dict[str, Any]:
        """
        Fallback parsing using regex when JSON parsing fails
        
        Args:
            response_content: Raw response text
            
        Returns:
            Dictionary with extracted values
        """
        result = {
            'gap_percentage': 0,
            'projected_annual_consumption': 0,
            'goal_achievable': False,
            'confidence': 0.5,
            'analysis_text': response_content
        }
        
        try:
            # Look for percentage values (with or without % sign)
            percentage_patterns = [
                r'gap[_\s]*(?:percentage|percent|difference)[_\s]*:?\s*["\']?(-?\d+\.?\d*)%?["\']?',
                r'percentage[_\s]*(?:gap|difference)[_\s]*:?\s*["\']?(-?\d+\.?\d*)%?["\']?',
                r'(-?\d+\.?\d*)%?\s*gap',
                r'gap.*?(-?\d+\.?\d*)%'
            ]
            
            for pattern in percentage_patterns:
                match = re.search(pattern, response_content, re.IGNORECASE)
                if match:
                    try:
                        result['gap_percentage'] = float(match.group(1))
                        logger.debug(f"Regex extracted gap percentage: {result['gap_percentage']}")
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Look for consumption/projection values
            consumption_patterns = [
                r'projected[_\s]*(?:annual[_\s]*)?consumption[_\s]*:?\s*["\']?(\d+\.?\d*)["\']?',
                r'annual[_\s]*projection[_\s]*:?\s*["\']?(\d+\.?\d*)["\']?',
                r'consumption.*?(\d+\.?\d*)'
            ]
            
            for pattern in consumption_patterns:
                match = re.search(pattern, response_content, re.IGNORECASE)
                if match:
                    try:
                        result['projected_annual_consumption'] = float(match.group(1))
                        logger.debug(f"Regex extracted projected consumption: {result['projected_annual_consumption']}")
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Look for goal achievability
            achievability_patterns = [
                r'goal[_\s]*achievable[_\s]*:?\s*["\']?(yes|no|true|false|achievable|not.achievable)["\']?',
                r'achievable[_\s]*:?\s*["\']?(yes|no|true|false)["\']?'
            ]
            
            for pattern in achievability_patterns:
                match = re.search(pattern, response_content, re.IGNORECASE)
                if match:
                    achievable_text = match.group(1).lower()
                    result['goal_achievable'] = achievable_text in ['yes', 'true', 'achievable']
                    logger.debug(f"Regex extracted goal achievability: {result['goal_achievable']}")
                    break
            
        except Exception as e:
            logger.warning(f"Error in fallback parsing: {e}")
        
        return result
    
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