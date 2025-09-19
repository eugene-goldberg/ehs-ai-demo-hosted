"""
Prompt Augmenter Service
Creates RAG-augmented prompts by combining user queries with Neo4j context data
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import calendar

logger = logging.getLogger(__name__)

class PromptAugmenter:
    """Creates augmented prompts for RAG-based responses"""

    def __init__(self):
        # Get current date context for temporal understanding
        now = datetime.now()
        current_date_context = f"""
**Current Date:** {now.strftime("%B %d, %Y")} ({now.strftime("%Y-%m-%d")})
**Current Month:** {now.strftime("%B %Y")}
**Current Year:** {now.year}

When users ask about relative time periods like "last month", "this month", etc., interpret them relative to the current date above.
"""

        self.base_instructions = f"""You are an EHS (Environmental, Health, and Safety) AI Assistant with access to real-time data from our facilities.

{current_date_context}

### IMPORTANT RESPONSE FORMATTING INSTRUCTIONS:

**Content Requirements:**
1. Answer based ONLY on the provided context data
2. Include specific numbers and dates from the data
3. If the data is insufficient to answer the question, clearly state what information is missing
4. Be concise and professional
5. Use units consistently (kWh for electricity, $ for costs, tons for CO2)
6. Round numbers appropriately for readability
7. If asked about data outside the provided context, say you don't have that information
8. When interpreting relative time references ("last month", "this quarter", etc.), use the current date context provided above

**Formatting Requirements:**
- Use clear markdown formatting with proper headers (###)
- Use bullet points (-) for lists and key metrics
- Use **bold** for emphasis on important numbers and key findings
- Create clear sections with appropriate spacing
- Present data using bullet points and clear sections
- Structure responses with logical flow: overview, details, insights
- Use consistent formatting for units and numbers
- Add clear section breaks between different types of information

**Response Structure:**
1. Start with a brief summary of findings
2. Present detailed data in organized sections
3. Highlight key insights or trends
4. Include relevant context and explanations"""

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
        if intent_type == "electricity_consumption":
            formatted_context = self._format_electricity_context(context_data)
        elif intent_type == "water_consumption":
            formatted_context = self._format_water_context(context_data)
        elif intent_type == "waste_generation":
            formatted_context = self._format_waste_context(context_data)
        elif intent_type == "co2_goals":
            formatted_context = self._format_goals_context(context_data)
        elif intent_type == "risk_assessment":
            formatted_context = self._format_risk_context(context_data)
        elif intent_type == "recommendations":
            formatted_context = self._format_recommendations_context(context_data)
        else:
            formatted_context = self._format_general_context(context_data)

        # Build the complete prompt
        prompt = f"""{self.base_instructions}

### CONTEXT DATA:
{formatted_context}

### USER QUESTION:
{user_query}

Please provide a clear, data-driven response based on the context above, following the formatting guidelines specified in the instructions."""

        return prompt

    def _format_electricity_context(self, context: Dict[str, Any]) -> str:
        """Format electricity consumption context with simple markdown structure"""

        if context.get("record_count", 0) == 0:
            return "**No electricity consumption data available** for the specified criteria."

        lines = []

        # Header
        lines.append("### Electricity Consumption Data")
        lines.append("")

        # Basic information
        lines.append("#### Site Information")
        lines.append(f"- **Site:** {context.get('site', 'Unknown')}")

        # Period information
        period = context.get("period", {})
        if period.get("start") and period.get("end"):
            lines.append(f"- **Period:** {period['start']} to {period['end']}")

        lines.append(f"- **Records Available:** {context.get('record_count', 0):,}")
        lines.append("")

        # Aggregated data with simple bullet points instead of tables
        if "aggregates" in context:
            agg = context["aggregates"]
            lines.append("#### Summary Metrics")
            lines.append("")
            lines.append(f"- **Total Consumption:** {agg.get('total', 0):,.0f} kWh")
            lines.append(f"- **Average Daily:** {agg.get('average', 0):,.0f} kWh")
            lines.append(f"- **Minimum Daily:** {agg.get('min', 0):,.0f} kWh")
            lines.append(f"- **Maximum Daily:** {agg.get('max', 0):,.0f} kWh")
            lines.append(f"- **Total Cost:** ${agg.get('total_cost', 0):,.2f}")
            lines.append(f"- **Total CO2 Emissions:** {agg.get('total_co2', 0):,.2f} tons")
            if agg.get("avg_cost_per_kwh", 0) > 0:
                lines.append(f"- **Average Cost per kWh:** ${agg['avg_cost_per_kwh']:.3f}")
            lines.append("")

        # Recent data samples
        if "recent_data" in context and context["recent_data"]:
            lines.append("#### Recent Daily Data")
            lines.append("")
            for i, record in enumerate(context["recent_data"][:5], 1):
                lines.append(f"- **{record['date']}:** {record['consumption']:,.0f} kWh, Cost: ${record['cost']:,.2f}")
            lines.append("")

        return "\n".join(lines)

    def _format_water_context(self, context: Dict[str, Any]) -> str:
        """Format water consumption context with simple markdown structure"""

        if context.get("record_count", 0) == 0:
            return "**No water consumption data available** for the specified criteria."

        lines = []

        # Header
        lines.append("### Water Consumption Data")
        lines.append("")

        # Basic information
        lines.append("#### Site Information")
        lines.append(f"- **Site:** {context.get('site', 'Unknown')}")

        # Period information
        period = context.get("period", {})
        if period.get("start") and period.get("end"):
            lines.append(f"- **Period:** {period['start']} to {period['end']}")

        lines.append(f"- **Records Available:** {context.get('record_count', 0):,}")
        lines.append("")

        # Aggregated data with simple bullet points instead of tables
        if "aggregates" in context:
            agg = context["aggregates"]
            lines.append("#### Summary Metrics")
            lines.append("")
            lines.append(f"- **Total Consumption:** {agg.get('total', 0):,.0f} gallons")
            lines.append(f"- **Average Daily:** {agg.get('average', 0):,.0f} gallons")
            lines.append(f"- **Minimum Daily:** {agg.get('min', 0):,.0f} gallons")
            lines.append(f"- **Maximum Daily:** {agg.get('max', 0):,.0f} gallons")
            if agg.get('total_cost', 0) > 0:
                lines.append(f"- **Total Cost:** ${agg.get('total_cost', 0):,.2f}")
            lines.append("")

        # Recent data samples
        if "recent_data" in context and context["recent_data"]:
            lines.append("#### Recent Daily Data")
            lines.append("")
            for i, record in enumerate(context["recent_data"][:5], 1):
                cost_info = f", Cost: ${record['cost']:,.2f}" if 'cost' in record else ""
                lines.append(f"- **{record['date']}:** {record['consumption']:,.0f} gallons{cost_info}")
            lines.append("")

        return "\n".join(lines)
    def _format_waste_context(self, context: Dict[str, Any]) -> str:
        """Format waste generation context with simple markdown structure"""

        if context.get("record_count", 0) == 0:
            return "**No waste generation data available** for the specified criteria."

        lines = []

        # Header
        lines.append("### Waste Generation Data")
        lines.append("")

        # Basic information
        lines.append("#### Site Information")
        lines.append(f"- **Site:** {context.get('site', 'Unknown')}")

        # Period information
        period = context.get("period", {})
        if period.get("start") and period.get("end"):
            lines.append(f"- **Period:** {period['start']} to {period['end']}")

        lines.append(f"- **Records Available:** {context.get('record_count', 0):,}")
        lines.append("")

        # Waste type breakdown with simple formatting
        if "waste_types" in context:
            lines.append("#### Waste Types Summary")
            lines.append("")
            for waste_type, data in context["waste_types"].items():
                amount = data.get('total_amount', 0)
                cost = data.get('total_cost', 0)
                lines.append(f"- **{waste_type}:** {amount:,.2f} tons, Disposal Cost: ${cost:,.2f}")
            lines.append("")

        # Aggregated data with simple bullet points
        if "aggregates" in context:
            agg = context["aggregates"]
            lines.append("#### Overall Summary")
            lines.append("")
            lines.append(f"- **Total Waste Generated:** {agg.get('total_amount', 0):,.2f} tons")
            lines.append(f"- **Average Daily:** {agg.get('average', 0):,.2f} tons")
            lines.append(f"- **Total Disposal Cost:** ${agg.get('total_cost', 0):,.2f}")
            lines.append("")

        # Recent data samples
        if "recent_data" in context and context["recent_data"]:
            lines.append("#### Recent Daily Data")
            lines.append("")
            for i, record in enumerate(context["recent_data"][:5], 1):
                waste_type = record.get('waste_type', 'Unknown')
                amount = record.get('amount', 0)
                cost_info = f", Cost: ${record['cost']:,.2f}" if 'cost' in record else ""
                lines.append(f"- **{record['date']}:** {waste_type} - {amount:,.2f} tons{cost_info}")
            lines.append("")

        return "\n".join(lines)

    def _format_goals_context(self, context: Dict[str, Any]) -> str:
        """Format CO2 goals context with simple markdown structure"""

        lines = []

        # Header
        lines.append("### CO2 Emissions Goals & Progress")
        lines.append("")

        # Goals information
        if "goals" in context:
            lines.append("#### Emissions Reduction Goals")
            lines.append("")
            for goal in context["goals"]:
                target_year = goal.get('target_year', 'Unknown')
                reduction_target = goal.get('reduction_percentage', 0)
                baseline_year = goal.get('baseline_year', 'Unknown')
                lines.append(f"- **{target_year} Target:** {reduction_target}% reduction from {baseline_year} baseline")
            lines.append("")

        # Current progress with simple bullet points
        if "current_progress" in context:
            progress = context["current_progress"]
            lines.append("#### Current Progress")
            lines.append("")
            lines.append(f"- **Current Year Emissions:** {progress.get('current_emissions', 0):,.2f} tons CO2")
            lines.append(f"- **Baseline Emissions:** {progress.get('baseline_emissions', 0):,.2f} tons CO2")
            lines.append(f"- **Reduction Achieved:** {progress.get('reduction_percentage', 0):.1f}%")
            lines.append(f"- **Goal Progress:** {progress.get('goal_progress', 0):.1f}% complete")
            lines.append("")

        # Facility breakdown with simple formatting
        if "facility_breakdown" in context:
            lines.append("#### Facility Breakdown")
            lines.append("")
            for facility in context["facility_breakdown"]:
                name = facility.get('name', 'Unknown')
                current = facility.get('current_emissions', 0)
                target = facility.get('target_emissions', 0)
                progress = facility.get('progress_percentage', 0)
                lines.append(f"**{name}**")
                lines.append(f"- **Current Emissions:** {current:,.2f} tons")
                lines.append(f"- **Target Emissions:** {target:,.2f} tons")
                lines.append(f"- **Progress:** {progress:.1f}%")
                lines.append("")

        return "\n".join(lines)

    def _format_risk_context(self, context: Dict[str, Any]) -> str:
        """Format risk assessment context with simple, clean markdown structure"""

        if not context or context.get("risk_count", 0) == 0:
            return "**No risk assessment data available** for the specified criteria."

        lines = []

        # Header
        lines.append("### Risk Assessment Data")
        lines.append("")

        # Risk summary with simple bullet points instead of tables
        if "risk_summary" in context:
            summary = context["risk_summary"]
            lines.append("#### Risk Level Summary")
            lines.append("")
            for level in ['Critical', 'High', 'Medium', 'Low']:
                count = summary.get(level.lower(), 0)
                percentage = summary.get(f"{level.lower()}_percentage", 0)
                lines.append(f"- **{level} Risk:** {count} cases ({percentage:.1f}%)")
            lines.append("")

        # Active risks by category with clean formatting
        if "risk_categories" in context:
            lines.append("#### Risks by Category")
            lines.append("")
            for category, risks in context["risk_categories"].items():
                lines.append(f"**{category}**")
                lines.append("")
                for risk in risks:
                    risk_level = risk.get('level', 'Unknown')
                    description = risk.get('description', 'No description')
                    likelihood = risk.get('likelihood', 'Unknown')
                    impact = risk.get('impact', 'Unknown')

                    lines.append(f"- **Risk ID {risk.get('id', 'N/A')}** - {risk_level} Risk")
                    lines.append(f"  **Description:** {description}")
                    lines.append(f"  **Likelihood:** {likelihood}")
                    lines.append(f"  **Impact:** {impact}")

                    # Mitigation strategies
                    if 'mitigation_strategies' in risk:
                        lines.append(f"  **Mitigation Strategies:**")
                        for strategy in risk['mitigation_strategies']:
                            lines.append(f"    - {strategy}")
                    lines.append("")

        # Recent assessments with clean formatting
        if "recent_assessments" in context:
            lines.append("#### Recent Risk Assessments")
            lines.append("")
            for assessment in context["recent_assessments"][:5]:
                date = assessment.get('date', 'Unknown')
                assessor = assessment.get('assessor', 'Unknown')
                facility = assessment.get('facility', 'Unknown')
                new_risks = assessment.get('new_risks_identified', 0)
                
                lines.append(f"**Assessment Date:** {date}")
                lines.append(f"**Facility:** {facility}")
                lines.append(f"**Assessor:** {assessor}")
                lines.append(f"**New Risks Identified:** {new_risks}")
                lines.append("")

        return "\n".join(lines)

    def _format_recommendations_context(self, context: Dict[str, Any]) -> str:
        """Format recommendations context with simple markdown structure"""

        if not context or context.get("recommendation_count", 0) == 0:
            return "**No recommendations available** for the specified criteria."

        lines = []

        # Header
        lines.append("### Environmental Recommendations")
        lines.append("")

        # Summary with simple bullet points instead of table
        if "summary" in context:
            summary = context["summary"]
            lines.append("#### Recommendations Summary")
            lines.append("")

            for category, data in summary.items():
                if isinstance(data, dict):
                    count = data.get('total', 0)
                    high = data.get('high_priority', 0)
                    medium = data.get('medium_priority', 0)
                    low = data.get('low_priority', 0)
                    lines.append(f"**{category}**")
                    lines.append(f"- **Total Count:** {count}")
                    lines.append(f"- **Priority Breakdown:** High: {high}, Medium: {medium}, Low: {low}")
                    lines.append("")

        # Detailed recommendations by category
        if "recommendations" in context:
            lines.append("#### Detailed Recommendations")
            lines.append("")

            for category, recommendations in context["recommendations"].items():
                lines.append(f"##### {category}")
                lines.append("")

                for i, rec in enumerate(recommendations, 1):
                    title = rec.get('title', f'Recommendation {i}')
                    priority = rec.get('priority', 'Medium')
                    description = rec.get('description', 'No description available')

                    lines.append(f"**{i}. {title}** (*{priority} Priority*)")
                    lines.append("")
                    lines.append(f"{description}")
                    lines.append("")

                    # Implementation details
                    if 'implementation' in rec:
                        impl = rec['implementation']
                        lines.append("**Implementation:**")
                        if 'estimated_cost' in impl:
                            lines.append(f"- **Estimated Cost:** ${impl['estimated_cost']:,.2f}")
                        if 'timeline' in impl:
                            lines.append(f"- **Timeline:** {impl['timeline']}")
                        if 'resources_required' in impl:
                            lines.append(f"- **Resources Required:** {impl['resources_required']}")
                        lines.append("")

                    # Expected benefits
                    if 'benefits' in rec:
                        benefits = rec['benefits']
                        lines.append("**Expected Benefits:**")
                        for benefit_type, value in benefits.items():
                            if isinstance(value, (int, float)):
                                if 'cost' in benefit_type.lower() or 'saving' in benefit_type.lower():
                                    lines.append(f"- **{benefit_type.replace('_', ' ').title()}:** ${value:,.2f}")
                                elif 'reduction' in benefit_type.lower() or 'emission' in benefit_type.lower():
                                    lines.append(f"- **{benefit_type.replace('_', ' ').title()}:** {value:,.2f} tons CO2")
                                else:
                                    lines.append(f"- **{benefit_type.replace('_', ' ').title()}:** {value}")
                            else:
                                lines.append(f"- **{benefit_type.replace('_', ' ').title()}:** {value}")
                        lines.append("")

                    lines.append("---")
                    lines.append("")

        # Citations and sources
        if "sources" in context:
            lines.append("#### Sources and References")
            lines.append("")
            for i, source in enumerate(context["sources"], 1):
                if isinstance(source, dict):
                    title = source.get('title', f'Source {i}')
                    url = source.get('url', '')
                    date = source.get('date', '')

                    if url:
                        lines.append(f"{i}. **{title}** - [{url}]({url})")
                    else:
                        lines.append(f"{i}. **{title}**")

                    if date:
                        lines.append(f"   *Published: {date}*")
                else:
                    lines.append(f"{i}. {source}")
                lines.append("")

        return "\n".join(lines)

    def _format_general_context(self, context: Dict[str, Any]) -> str:
        """Format general context with basic structure"""
        lines = []
        lines.append("### General Context Data")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(context, indent=2))
        lines.append("```")
        return "\n".join(lines)

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
            context_data = {"error": "Failed to parse context data"}

        return self.create_augmented_prompt(user_query, context_data, intent_type or "general")