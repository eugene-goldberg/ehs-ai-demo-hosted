"""
Intent Classifier Service

Purpose: Classifies user queries into EHS categories and extracts relevant metadata.

This service:
- Uses real LLM calls (OpenAI GPT models) for classification
- Classifies queries into predefined EHS categories
- Extracts site information and time period data
- Returns structured classification results

Categories:
- ELECTRICITY_CONSUMPTION
- WATER_CONSUMPTION  
- WASTE_GENERATION
- CO2_GOALS
- RISK_ASSESSMENT
- RECOMMENDATIONS
- GENERAL
"""

import logging
import json
import re
import sys
import os
import importlib.util
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import the LLM module directly from the file path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
llm_file_path = os.path.join(backend_dir, 'src', 'llm.py')

# Load the llm.py module directly
spec = importlib.util.spec_from_file_location("llm_direct", llm_file_path)
llm_direct = importlib.util.module_from_spec(spec)
sys.modules["llm_direct"] = llm_direct
spec.loader.exec_module(llm_direct)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Structured result from intent classification."""
    intent: str
    confidence: float
    site: Optional[str] = None
    time_period: Optional[Dict[str, Any]] = None
    extracted_entities: Optional[Dict[str, Any]] = None
    raw_query: Optional[str] = None


class IntentClassifier:
    """Service for classifying user queries into EHS categories."""
    
    # Supported EHS categories
    SUPPORTED_INTENTS = [
        "ELECTRICITY_CONSUMPTION",
        "WATER_CONSUMPTION", 
        "WASTE_GENERATION",
        "CO2_GOALS",
        "RISK_ASSESSMENT",
        "RECOMMENDATIONS",
        "GENERAL"
    ]
    
    # Known sites
    KNOWN_SITES = [
        "houston_texas",
        "algonquin_illinois"
    ]
    
    def __init__(self, model_name: str = "openai_gpt_4o"):
        """
        Initialize the Intent Classifier.
        
        Args:
            model_name: The LLM model to use for classification
        """
        self.model_name = model_name
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM client."""
        try:
            llm_tuple = llm_direct.get_llm(self.model_name)
            self.llm = llm_tuple[0] if isinstance(llm_tuple, tuple) else llm_tuple
            logger.info(f"Intent Classifier initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM for Intent Classifier: {e}")
            raise
    
    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a user query into EHS categories and extract metadata.
        
        Args:
            query: The user query to classify
            
        Returns:
            ClassificationResult with intent, confidence, and extracted metadata
        """
        try:
            logger.info(f"Classifying query: {query[:100]}...")
            
            # Create classification prompt
            prompt = self._create_classification_prompt(query)
            
            # Get LLM response
            if isinstance(self.llm, tuple):
                # Handle tuple return from get_llm
                logger.warning("LLM returned tuple, using fallback classification")
                return self._fallback_classification(query)
            response = self.llm.invoke(prompt)
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
            classification_result = self._parse_classification_response(result_text, query)
            
            logger.info(f"Classification result: {classification_result.intent} (confidence: {classification_result.confidence})")
            
            return classification_result
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            # Return a default classification for GENERAL queries
            return ClassificationResult(
                intent="GENERAL",
                confidence=0.5,
                raw_query=query,
                extracted_entities={"error": str(e)}
            )
    
    def _create_classification_prompt(self, query: str) -> str:
        """Create the classification prompt for the LLM."""
        prompt = f"""You are an expert EHS (Environmental, Health, and Safety) data analyst. 
Classify the following user query into one of these categories and extract relevant metadata.

CATEGORIES:
1. ELECTRICITY_CONSUMPTION - Questions about electrical energy usage, power consumption, kilowatt hours
2. WATER_CONSUMPTION - Questions about water usage, gallons consumed, water metrics
3. WASTE_GENERATION - Questions about waste production, disposal, waste metrics, recycling
4. CO2_GOALS - Questions about carbon emissions, CO2 targets, sustainability goals, carbon footprint
5. RISK_ASSESSMENT - Questions about environmental risks, safety assessments, compliance issues
6. RECOMMENDATIONS - Requests for suggestions, improvements, best practices, action items
7. GENERAL - General questions, greetings, or queries that don't fit other categories

SITES (extract if mentioned):
- houston_texas
- algonquin_illinois

TIME PERIODS (extract if mentioned):
- Specific dates, months, years
- Relative time phrases (last month, this year, Q1, etc.)
- Date ranges

USER QUERY: "{query}"

Respond with a JSON object in this exact format:
{{
    "intent": "CATEGORY_NAME",
    "confidence": 0.95,
    "site": "site_name_or_null",
    "time_period": {{
        "type": "specific|relative|range|null",
        "value": "extracted_time_info_or_null",
        "start_date": "YYYY-MM-DD_or_null",
        "end_date": "YYYY-MM-DD_or_null"
    }},
    "extracted_entities": {{
        "metrics": ["list_of_metrics"],
        "locations": ["list_of_locations"],
        "keywords": ["key_terms_from_query"]
    }}
}}

Be precise and confident in your classification. If uncertain between categories, choose the most specific one that applies."""
        
        return prompt
    
    def _parse_classification_response(self, response_text: str, original_query: str) -> ClassificationResult:
        """
        Parse the LLM response into a structured ClassificationResult.
        
        Args:
            response_text: Raw response from LLM
            original_query: Original user query
            
        Returns:
            ClassificationResult object
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)
            
            # Validate intent
            intent = parsed_data.get("intent", "GENERAL")
            if intent not in self.SUPPORTED_INTENTS:
                logger.warning(f"Unknown intent '{intent}', defaulting to GENERAL")
                intent = "GENERAL"
            
            # Extract confidence
            confidence = float(parsed_data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            
            # Extract site
            site = parsed_data.get("site")
            if site and site.lower() != "null":
                # Normalize site name
                site = self._normalize_site_name(site)
            else:
                site = None
            
            # Extract time period
            time_period = parsed_data.get("time_period")
            if time_period and isinstance(time_period, dict):
                # Clean up null values
                for key in time_period:
                    if time_period[key] == "null" or time_period[key] == "":
                        time_period[key] = None
            else:
                time_period = None
            
            # Extract entities
            extracted_entities = parsed_data.get("extracted_entities", {})
            
            return ClassificationResult(
                intent=intent,
                confidence=confidence,
                site=site,
                time_period=time_period,
                extracted_entities=extracted_entities,
                raw_query=original_query
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse classification response: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Fallback classification
            return self._fallback_classification(original_query)
    
    def _normalize_site_name(self, site: str) -> Optional[str]:
        """Normalize site name to match known sites."""
        if not site:
            return None
            
        site_lower = site.lower().strip()
        
        # Direct matches
        if site_lower in self.KNOWN_SITES:
            return site_lower
        
        # Fuzzy matching
        if "houston" in site_lower or "texas" in site_lower:
            return "houston_texas"
        elif "algonquin" in site_lower or "illinois" in site_lower:
            return "algonquin_illinois"
        
        return None
    
    def _fallback_classification(self, query: str) -> ClassificationResult:
        """
        Provide a fallback classification using simple keyword matching.
        
        Args:
            query: Original user query
            
        Returns:
            ClassificationResult with basic classification
        """
        query_lower = query.lower()
        
        # Simple keyword-based classification
        if any(word in query_lower for word in ["electricity", "electric", "power", "kwh", "kilowatt"]):
            intent = "ELECTRICITY_CONSUMPTION"
            confidence = 0.7
        elif any(word in query_lower for word in ["water", "gallons", "usage", "consumption"]):
            intent = "WATER_CONSUMPTION"
            confidence = 0.7
        elif any(word in query_lower for word in ["waste", "trash", "disposal", "recycling"]):
            intent = "WASTE_GENERATION"
            confidence = 0.7
        elif any(word in query_lower for word in ["co2", "carbon", "emissions", "goals", "sustainability"]):
            intent = "CO2_GOALS"
            confidence = 0.7
        elif any(word in query_lower for word in ["risk", "assessment", "safety", "compliance"]):
            intent = "RISK_ASSESSMENT"
            confidence = 0.7
        elif any(word in query_lower for word in ["recommend", "suggest", "improve", "best practice"]):
            intent = "RECOMMENDATIONS"
            confidence = 0.7
        else:
            intent = "GENERAL"
            confidence = 0.6
        
        # Extract site using simple matching
        site = None
        if "houston" in query_lower or "texas" in query_lower:
            site = "houston_texas"
        elif "algonquin" in query_lower or "illinois" in query_lower:
            site = "algonquin_illinois"
        
        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            site=site,
            raw_query=query,
            extracted_entities={"fallback": True}
        )
    
    def classify_batch(self, queries: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple queries in batch.
        
        Args:
            queries: List of queries to classify
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        for query in queries:
            result = self.classify(query)
            results.append(result)
        
        return results
    
    def get_supported_intents(self) -> List[str]:
        """Get the list of supported intent categories."""
        return self.SUPPORTED_INTENTS.copy()
    
    def get_known_sites(self) -> List[str]:
        """Get the list of known site identifiers."""
        return self.KNOWN_SITES.copy()


# Factory function for easy instantiation
def create_intent_classifier(model_name: str = "openai_gpt_4o") -> IntentClassifier:
    """
    Create and return an IntentClassifier instance.
    
    Args:
        model_name: LLM model to use for classification
        
    Returns:
        Initialized IntentClassifier
    """
    return IntentClassifier(model_name=model_name)
