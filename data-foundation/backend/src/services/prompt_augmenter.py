"""
Prompt Augmenter Service
Creates RAG-augmented prompts by combining user queries with Neo4j context data
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PromptAugmenter:
    """Creates augmented prompts for RAG-based responses"""
    
    def __init__(self):
        self.base_instructions = """You are an EHS (Environmental, Health, and Safety) AI Assistant with access to real-time data from our facilities.

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the provided context data
2. Include specific numbers and dates from the data
3. If the data is insufficient to answer the question, clearly state what information is missing
4. Be concise and professional
5. Use units consistently (kWh for electricity, $ for costs, tons for CO2)
6. Round numbers appropriately for readability
7. If asked about data outside the provided context, say you don't have that information"""

    def create_augmented_prompt(self, 
                               user_query: str,
                               context_data: Dict[str, Any],
                               intent_type: str) -> str:
        """
        Create an augmented prompt combining query and context
        
        Args:
            user_query: The original user question
            context_data: Retrieved context from Neo4j
            intent_type: The classified intent type
            
        Returns:
            Complete augmented prompt for LLM
        """
        
        # Format context based on intent type
        if intent_type == 'electricity_consumption':
            formatted_context = self._format_electricity_context(context_data)
        elif intent_type == 'water_consumption':
            formatted_context = self._format_water_context(context_data)
        elif intent_type == 'waste_generation':
            formatted_context = self._format_waste_context(context_data)
        elif intent_type == 'co2_goals':
            formatted_context = self._format_goals_context(context_data)
        elif intent_type == 'risk_assessment':
            formatted_context = self._format_risk_context(context_data)
        elif intent_type == 'recommendations':
            formatted_context = self._format_recommendations_context(context_data)
        else:
            formatted_context = self._format_general_context(context_data)
        
        # Build the complete prompt
        prompt = f"""{self.base_instructions}

CONTEXT DATA:
{formatted_context}

USER QUESTION: {user_query}

Please provide a clear, data-driven response based on the context above."""
        
        return prompt
    
    def _format_electricity_context(self, context: Dict[str, Any]) -> str:
        """Format electricity consumption context"""
        
        if context.get('record_count', 0) == 0:
            return "No electricity consumption data available for the specified criteria."
        
        lines = []
        
        # Basic information
        lines.append(f"Site: {context.get('site', 'Unknown')}")
        
        # Period information
        period = context.get('period', {})
        if period.get('start') and period.get('end'):
            lines.append(f"Period: {period['start']} to {period['end']}")
        
        lines.append(f"Number of records: {context.get('record_count', 0)}")
        
        # Aggregated data
        if 'aggregates' in context:
            agg = context['aggregates']
            lines.append("\nAGGREGATED METRICS:")
            lines.append(f"- Total consumption: {agg.get('total', 0):,.0f} kWh")
            lines.append(f"- Average daily consumption: {agg.get('average', 0):,.0f} kWh")
            lines.append(f"- Minimum daily: {agg.get('min', 0):,.0f} kWh")
            lines.append(f"- Maximum daily: {agg.get('max', 0):,.0f} kWh")
            lines.append(f"- Total cost: ${agg.get('total_cost', 0):,.2f}")
            lines.append(f"- Total CO2 emissions: {agg.get('total_co2', 0):,.2f} tons")
            if agg.get('avg_cost_per_kwh', 0) > 0:
                lines.append(f"- Average cost per kWh: ${agg['avg_cost_per_kwh']:.3f}")
        
        # Recent data samples
        if 'recent_data' in context and context['recent_data']:
            lines.append("\nRECENT DAILY DATA:")
            for i, record in enumerate(context['recent_data'][:5], 1):
                lines.append(f"{i}. {record['date']}: {record['consumption']:,.0f} kWh, Cost: ${record['cost']:,.2f}")
        
        return "\n".join(lines)
    
    def _format_water_context(self, context: Dict[str, Any]) -> str:
        """Format water consumption context (stub)"""
        return "Water consumption data retrieval not yet implemented.\n" + json.dumps(context, indent=2)
    
    def _format_waste_context(self, context: Dict[str, Any]) -> str:
        """Format waste generation context (stub)"""
        return "Waste generation data retrieval not yet implemented.\n" + json.dumps(context, indent=2)
    
    def _format_goals_context(self, context: Dict[str, Any]) -> str:
        """Format CO2 goals context (stub)"""
        return "CO2 goals data retrieval not yet implemented.\n" + json.dumps(context, indent=2)
    
    def _format_risk_context(self, context: Dict[str, Any]) -> str:
        """Format risk assessment context (stub)"""
        return "Risk assessment data retrieval not yet implemented.\n" + json.dumps(context, indent=2)
    
    def _format_recommendations_context(self, context: Dict[str, Any]) -> str:
        """Format recommendations context (stub)"""
        return "Recommendations data retrieval not yet implemented.\n" + json.dumps(context, indent=2)
    
    def _format_general_context(self, context: Dict[str, Any]) -> str:
        """Format general context"""
        return json.dumps(context, indent=2)
    
    def create_simple_prompt(self, user_query: str, context_json: str, intent_type: str = None) -> str:
        """
        Convenience method that takes JSON string context
        
        Args:
            user_query: The user's question
            context_json: Context data as JSON string
            intent_type: Optional intent type
            
        Returns:
            Augmented prompt string
        """
        try:
            context_data = json.loads(context_json) if isinstance(context_json, str) else context_json
        except json.JSONDecodeError:
            context_data = {'error': 'Failed to parse context data'}
        
        return self.create_augmented_prompt(user_query, context_data, intent_type or 'general')
