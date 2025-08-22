"""
Query Router Agent for EHS Analytics

This module contains the Query Router Agent responsible for classifying natural language
queries about EHS data into specific intent types and extracting relevant entities.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI

# Import our logging and monitoring utilities
from ..utils.logging import get_ehs_logger, performance_logger, log_context
from ..utils.monitoring import get_ehs_monitor
from ..utils.tracing import trace_function, SpanKind

logger = get_ehs_logger(__name__)


class IntentType(str, Enum):
    """Enumeration of supported EHS query intent types."""
    
    CONSUMPTION_ANALYSIS = "consumption_analysis"
    COMPLIANCE_CHECK = "compliance_check"  
    RISK_ASSESSMENT = "risk_assessment"
    EMISSION_TRACKING = "emission_tracking"
    EQUIPMENT_EFFICIENCY = "equipment_efficiency"
    PERMIT_STATUS = "permit_status"
    GENERAL_INQUIRY = "general_inquiry"


class RetrieverType(str, Enum):
    """Enumeration of available data retrievers."""
    
    CONSUMPTION_RETRIEVER = "consumption_retriever"
    COMPLIANCE_RETRIEVER = "compliance_retriever"
    RISK_RETRIEVER = "risk_retriever"
    EMISSION_RETRIEVER = "emission_retriever"
    EQUIPMENT_RETRIEVER = "equipment_retriever"
    PERMIT_RETRIEVER = "permit_retriever"
    GENERAL_RETRIEVER = "general_retriever"


@dataclass
class EntityExtraction:
    """Container for extracted entities from user queries."""
    
    facilities: List[str]
    date_ranges: List[str]
    equipment: List[str]
    pollutants: List[str]
    regulations: List[str]
    departments: List[str]
    metrics: List[str]


@dataclass
class QueryClassification:
    """Result of query classification and entity extraction."""
    
    intent_type: IntentType
    confidence_score: float
    entities_identified: EntityExtraction
    suggested_retriever: RetrieverType
    reasoning: str
    query_rewrite: Optional[str] = None


class QueryRouterAgent:
    """
    Query Router Agent for EHS Analytics.
    
    Classifies natural language queries about EHS data into specific intent types,
    extracts relevant entities, and suggests appropriate data retrievers.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize the Query Router Agent.
        
        Args:
            llm: Language model for classification (defaults to GPT-3.5-turbo)
            temperature: Temperature for LLM inference
            max_tokens: Maximum tokens for LLM response
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Intent keywords and patterns for classification
        self.intent_patterns = {
            IntentType.CONSUMPTION_ANALYSIS: [
                r'\b(consumption|usage|energy|water|electricity|gas|utility)\b',
                r'\b(trending|pattern|analyze|analysis|usage data)\b',
                r'\b(kWh|gallons|cubic feet|BTU|consumption rate)\b'
            ],
            IntentType.COMPLIANCE_CHECK: [
                r'\b(compliance|regulation|requirement|standard|violation)\b',
                r'\b(EPA|OSHA|ISO|permit|license|audit)\b',
                r'\b(non-compliant|violation|regulatory|status)\b'
            ],
            IntentType.RISK_ASSESSMENT: [
                r'\b(risk|assessment|hazard|danger|safety|threat)\b',
                r'\b(probability|likelihood|impact|consequence)\b',
                r'\b(environmental risk|safety risk|operational risk)\b'
            ],
            IntentType.EMISSION_TRACKING: [
                r'\b(emission|carbon|CO2|greenhouse|footprint|discharge)\b',
                r'\b(scope 1|scope 2|scope 3|carbon tracking)\b',
                r'\b(air quality|pollutant|NOx|SOx|particulate)\b'
            ],
            IntentType.EQUIPMENT_EFFICIENCY: [
                r'\b(equipment|machinery|efficiency|performance|maintenance)\b',
                r'\b(downtime|utilization|effectiveness|productivity)\b',
                r'\b(asset|machine|system|operational efficiency)\b'
            ],
            IntentType.PERMIT_STATUS: [
                r'\b(permit|license|authorization|approval|expiration)\b',
                r'\b(renewal|validity|status|compliance date)\b',
                r'\b(environmental permit|operating permit)\b'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'facilities': [
                r'\b(facility|plant|site|location|building|campus)\s+([A-Z][A-Za-z0-9\s-]+)',
                r'\b([A-Z][A-Za-z\s]+\s+(Plant|Facility|Site|Building))\b'
            ],
            'date_ranges': [
                r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                r'\b(last|past|previous)\s+(week|month|quarter|year)\b',
                r'\b(Q[1-4]\s+\d{4})\b'
            ],
            'equipment': [
                r'\b(boiler|chiller|compressor|pump|motor|generator|turbine)\s*(\w*)\b',
                r'\b(HVAC|system|unit|equipment)\s+([A-Z0-9-]+)\b'
            ],
            'pollutants': [
                r'\b(CO2|NOx|SOx|PM2\.5|PM10|VOC|methane|benzene|mercury)\b',
                r'\b(carbon dioxide|nitrogen oxide|sulfur oxide|particulate matter)\b'
            ],
            'regulations': [
                r'\b(EPA|OSHA|ISO\s*\d+|NESHAP|NSPS|CAA|CWA|RCRA)\b',
                r'\b(Title\s+V|Part\s+\d+|Section\s+\d+)\b'
            ],
            'departments': [
                r'\b(EHS|Environmental|Safety|Compliance|Operations|Maintenance)\s*(Department|Team)?\b'
            ],
            'metrics': [
                r'\b(efficiency|utilization|consumption|emission rate|compliance rate)\b',
                r'\b(kWh|BTU|gallons|cubic feet|tons|percentage|%)\b'
            ]
        }
        
        # Mapping intents to retrievers
        self.intent_retriever_map = {
            IntentType.CONSUMPTION_ANALYSIS: RetrieverType.CONSUMPTION_RETRIEVER,
            IntentType.COMPLIANCE_CHECK: RetrieverType.COMPLIANCE_RETRIEVER,
            IntentType.RISK_ASSESSMENT: RetrieverType.RISK_RETRIEVER,
            IntentType.EMISSION_TRACKING: RetrieverType.EMISSION_RETRIEVER,
            IntentType.EQUIPMENT_EFFICIENCY: RetrieverType.EQUIPMENT_RETRIEVER,
            IntentType.PERMIT_STATUS: RetrieverType.PERMIT_RETRIEVER,
            IntentType.GENERAL_INQUIRY: RetrieverType.GENERAL_RETRIEVER
        }
        
        logger.info(
            "QueryRouterAgent initialized",
            model=self.llm.model_name if hasattr(self.llm, 'model_name') else "unknown",
            temperature=temperature,
            max_tokens=max_tokens
        )

    @performance_logger(include_args=True)
    @trace_function("query_classification", SpanKind.INTERNAL, {"component": "query_router"})
    def classify_query(self, query: str, user_id: Optional[str] = None) -> QueryClassification:
        """
        Classify a natural language query into an EHS intent type.
        
        Args:
            query: The user's natural language query
            user_id: Optional user identifier for logging context
            
        Returns:
            QueryClassification object with intent, confidence, and entities
        """
        # Set up logging context
        with log_context(
            component="query_router", 
            operation="classify_query",
            user_id=user_id,
            query_length=len(query)
        ):
            logger.query_start(query, "classification", user_id=user_id)
            
            monitor = get_ehs_monitor()
            start_time = datetime.now()
            
            try:
                # Input validation
                if not query or not query.strip():
                    logger.warning("Empty or invalid query received", query=query)
                    raise ValueError("Query cannot be empty")
                
                logger.info("Starting query classification", query_preview=query[:100])
                
                # First pass: Pattern-based classification
                logger.debug("Starting pattern-based classification")
                pattern_scores = self._calculate_pattern_scores(query)
                logger.debug("Pattern scores calculated", pattern_scores=pattern_scores)
                
                # Second pass: LLM-based classification for refinement
                logger.debug("Starting LLM-based classification")
                llm_classification = self._llm_classify(query, pattern_scores)
                logger.debug("LLM classification completed", llm_result=llm_classification)
                
                # Extract entities
                logger.debug("Extracting entities from query")
                entities = self._extract_entities(query)
                entity_counts = {
                    entity_type: len(entity_list) 
                    for entity_type, entity_list in entities.__dict__.items()
                }
                logger.debug("Entity extraction completed", entity_counts=entity_counts)
                
                # Determine final intent and confidence
                final_intent = llm_classification["intent"]
                final_confidence = self._calculate_final_confidence(
                    pattern_scores, llm_classification["confidence"]
                )
                
                # Get suggested retriever
                suggested_retriever = self.intent_retriever_map[final_intent]
                
                # Create result
                result = QueryClassification(
                    intent_type=final_intent,
                    confidence_score=final_confidence,
                    entities_identified=entities,
                    suggested_retriever=suggested_retriever,
                    reasoning=llm_classification["reasoning"],
                    query_rewrite=self._rewrite_query(query, final_intent, entities)
                )
                
                # Log successful completion
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.query_end(
                    query, 
                    "classification", 
                    duration_ms, 
                    success=True,
                    intent_type=final_intent.value,
                    confidence_score=final_confidence,
                    entity_count=sum(entity_counts.values())
                )
                
                # Record metrics
                monitor.record_query(
                    query_type="classification",
                    duration_ms=duration_ms,
                    success=True
                )
                
                logger.info(
                    "Query classification completed successfully",
                    intent_type=final_intent.value,
                    confidence_score=final_confidence,
                    suggested_retriever=suggested_retriever.value,
                    duration_ms=duration_ms
                )
                
                return result
                
            except Exception as e:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.error(
                    "Query classification failed",
                    query=query,
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=duration_ms,
                    exc_info=True
                )
                
                # Log failed query
                logger.query_end(query, "classification", duration_ms, success=False)
                
                # Record error metrics
                monitor.record_query(
                    query_type="classification",
                    duration_ms=duration_ms,
                    success=False
                )
                
                raise

    @trace_function("pattern_scoring", SpanKind.INTERNAL)
    def _calculate_pattern_scores(self, query: str) -> Dict[IntentType, float]:
        """Calculate pattern-based scores for each intent type."""
        query_lower = query.lower()
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    matches += 1
                    score += 1.0
            
            # Normalize score
            if patterns:
                scores[intent] = min(score / len(patterns), 1.0)
            else:
                scores[intent] = 0.0
        
        logger.debug("Pattern scoring completed", total_patterns_checked=sum(len(p) for p in self.intent_patterns.values()))
        return scores

    @trace_function("llm_classification", SpanKind.CLIENT, {"service": "openai"})
    def _llm_classify(self, query: str, pattern_scores: Dict[IntentType, float]) -> Dict[str, Any]:
        """Use LLM for more sophisticated classification."""
        
        # Prepare context with pattern scores
        pattern_info = "\n".join([
            f"- {intent.value}: {score:.2f}" 
            for intent, score in pattern_scores.items()
        ])
        
        system_prompt = f"""You are an expert EHS (Environmental, Health, Safety) data analyst. 
        Your task is to classify user queries about EHS data into one of these intent types:

        1. consumption_analysis: Queries about utility consumption patterns, energy usage, water consumption
        2. compliance_check: Queries about regulatory compliance status, violations, audit results
        3. risk_assessment: Queries about environmental or safety risk evaluation
        4. emission_tracking: Queries about carbon footprint, air emissions, greenhouse gases
        5. equipment_efficiency: Queries about equipment performance, maintenance, efficiency metrics
        6. permit_status: Queries about permit compliance, expiration dates, renewals
        7. general_inquiry: Other EHS-related questions that don't fit above categories

        Pattern-based scores for this query:
        {pattern_info}

        Respond with a JSON object containing:
        - "intent": the most appropriate intent type
        - "confidence": confidence score between 0.0 and 1.0
        - "reasoning": brief explanation of your classification decision
        """

        user_prompt = f"Classify this EHS query: '{query}'"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            logger.debug("Sending query to LLM for classification")
            response = self.llm.invoke(messages)
            result = self._parse_llm_response(response.content)
            
            # Validate intent type
            if result["intent"] not in [intent.value for intent in IntentType]:
                logger.warning("Invalid intent from LLM, falling back", llm_intent=result["intent"])
                result["intent"] = IntentType.GENERAL_INQUIRY.value
                
            result["intent"] = IntentType(result["intent"])
            
            logger.debug("LLM classification successful", llm_intent=result["intent"].value, confidence=result["confidence"])
            return result
            
        except Exception as e:
            logger.error("LLM classification failed, falling back to pattern matching", error=str(e))
            
            # Fallback to highest pattern score
            best_intent = max(pattern_scores.items(), key=lambda x: x[1])
            return {
                "intent": best_intent[0],
                "confidence": best_intent[1],
                "reasoning": f"LLM classification failed, using pattern matching: {str(e)}"
            }

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response with fallback."""
        try:
            import json
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["intent", "confidence", "reasoning"]
                if all(field in parsed for field in required_fields):
                    return parsed
                else:
                    raise ValueError(f"Missing required fields: {required_fields}")
                
        except Exception as e:
            logger.warning("Failed to parse LLM response as JSON", error=str(e), response_preview=response[:200])
            
        # Fallback parsing
        return {
            "intent": "general_inquiry",
            "confidence": 0.5,
            "reasoning": "Could not parse LLM response, using fallback"
        }

    @trace_function("entity_extraction", SpanKind.INTERNAL)
    def _extract_entities(self, query: str) -> EntityExtraction:
        """Extract relevant entities from the query."""
        entities = {
            'facilities': [],
            'date_ranges': [],
            'equipment': [],
            'pollutants': [],
            'regulations': [],
            'departments': [],
            'metrics': []
        }
        
        total_matches = 0
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    # Get the full match or the first captured group
                    entity_value = match.group(1) if match.groups() else match.group(0)
                    entity_value = entity_value.strip()
                    
                    if entity_value and entity_value not in entities[entity_type]:
                        entities[entity_type].append(entity_value)
                        total_matches += 1
        
        logger.debug("Entity extraction completed", total_entities_found=total_matches)
        
        return EntityExtraction(
            facilities=entities['facilities'],
            date_ranges=entities['date_ranges'], 
            equipment=entities['equipment'],
            pollutants=entities['pollutants'],
            regulations=entities['regulations'],
            departments=entities['departments'],
            metrics=entities['metrics']
        )

    def _calculate_final_confidence(
        self, 
        pattern_scores: Dict[IntentType, float], 
        llm_confidence: float
    ) -> float:
        """Calculate final confidence combining pattern and LLM scores."""
        # Weight: 30% pattern matching, 70% LLM confidence
        max_pattern_score = max(pattern_scores.values()) if pattern_scores else 0.0
        final_confidence = 0.3 * max_pattern_score + 0.7 * llm_confidence
        
        return min(max(final_confidence, 0.0), 1.0)

    @trace_function("query_rewrite", SpanKind.INTERNAL)
    def _rewrite_query(
        self, 
        original_query: str, 
        intent: IntentType, 
        entities: EntityExtraction
    ) -> Optional[str]:
        """Rewrite query to be more specific for retrieval."""
        
        # Create a more structured query based on intent and entities
        query_parts = []
        
        # Add intent-specific context
        if intent == IntentType.CONSUMPTION_ANALYSIS:
            query_parts.append("Analyze consumption patterns and usage data")
        elif intent == IntentType.COMPLIANCE_CHECK:
            query_parts.append("Check regulatory compliance status")
        elif intent == IntentType.RISK_ASSESSMENT:
            query_parts.append("Evaluate environmental and safety risks")
        elif intent == IntentType.EMISSION_TRACKING:
            query_parts.append("Track emissions and carbon footprint")
        elif intent == IntentType.EQUIPMENT_EFFICIENCY:
            query_parts.append("Analyze equipment performance and efficiency")
        elif intent == IntentType.PERMIT_STATUS:
            query_parts.append("Check permit status and compliance")
        
        # Add entity information
        if entities.facilities:
            query_parts.append(f"for facilities: {', '.join(entities.facilities)}")
        if entities.date_ranges:
            query_parts.append(f"during: {', '.join(entities.date_ranges)}")
        if entities.equipment:
            query_parts.append(f"equipment: {', '.join(entities.equipment)}")
        if entities.pollutants:
            query_parts.append(f"pollutants: {', '.join(entities.pollutants)}")
        if entities.regulations:
            query_parts.append(f"regulations: {', '.join(entities.regulations)}")
        
        if len(query_parts) > 1:
            rewritten = " ".join(query_parts)
            logger.debug("Query rewritten for better retrieval", original_length=len(original_query), rewritten_length=len(rewritten))
            return rewritten
        
        return None

    def get_intent_examples(self) -> Dict[IntentType, List[str]]:
        """Get example queries for each intent type."""
        return {
            IntentType.CONSUMPTION_ANALYSIS: [
                "What is the electricity consumption trend for Plant A over the last quarter?",
                "Show me water usage patterns for all facilities in 2024",
                "Analyze energy consumption efficiency across our manufacturing sites"
            ],
            IntentType.COMPLIANCE_CHECK: [
                "Are we compliant with EPA air quality standards?",
                "Check OSHA compliance status for safety violations",
                "Show me any regulatory non-compliance issues this month"
            ],
            IntentType.RISK_ASSESSMENT: [
                "What are the environmental risks at our chemical facility?",
                "Assess safety risks for equipment maintenance operations",
                "Evaluate operational risks from our emission sources"
            ],
            IntentType.EMISSION_TRACKING: [
                "Track our carbon footprint for Scope 1 emissions",
                "Show CO2 emissions from our manufacturing processes",
                "What are our greenhouse gas emission trends?"
            ],
            IntentType.EQUIPMENT_EFFICIENCY: [
                "How efficient is our HVAC system performing?",
                "Show equipment utilization rates for our boilers",
                "Analyze maintenance schedules and downtime patterns"
            ],
            IntentType.PERMIT_STATUS: [
                "When do our environmental permits expire?",
                "Check the status of our air quality permits",
                "Are there any permits requiring renewal this quarter?"
            ],
            IntentType.GENERAL_INQUIRY: [
                "What EHS data do we have available?",
                "Explain our environmental management system",
                "What are the key EHS metrics we track?"
            ]
        }
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about classification performance."""
        # This would typically be implemented with actual usage data
        # For now, return placeholder stats
        return {
            "total_classifications": 0,
            "intent_distribution": {intent.value: 0 for intent in IntentType},
            "average_confidence": 0.0,
            "average_processing_time_ms": 0.0
        }