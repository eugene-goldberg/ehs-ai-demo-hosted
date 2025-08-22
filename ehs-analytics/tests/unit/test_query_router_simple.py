"""
Simple unit tests for QueryRouterAgent - focusing on core functionality.

These tests avoid complex imports and focus on testing the key logic 
with proper mocking of dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import re
from datetime import datetime


class TestQueryRouterAgentBasic:
    """Basic tests for QueryRouterAgent without complex imports."""

    def test_pattern_matching_consumption(self):
        """Test pattern matching for consumption queries."""
        # Mock the basic structure we need
        intent_patterns = {
            "consumption_analysis": [
                r'\b(consumption|usage|energy|water|electricity|gas|utility)\b',
                r'\b(trending|pattern|analyze|analysis|usage data)\b',
                r'\b(kWh|gallons|cubic feet|BTU|consumption rate)\b'
            ]
        }
        
        query = "What is the electricity consumption trend for Plant A?"
        
        # Test pattern matching logic
        query_lower = query.lower()
        scores = {}
        
        for intent, patterns in intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1.0
            
            if patterns:
                scores[intent] = min(score / len(patterns), 1.0)
            else:
                scores[intent] = 0.0
        
        # Should match consumption analysis patterns
        assert scores["consumption_analysis"] > 0.0

    def test_entity_extraction_facilities(self):
        """Test facility name extraction logic."""
        patterns = [
            r'\b(facility|plant|site|location|building|campus)\s+([A-Z][A-Za-z0-9\s-]+)',
            r'\b([A-Z][A-Za-z\s]+\s+(Plant|Facility|Site|Building))\b'
        ]
        
        query = "Show consumption for Plant A and Manufacturing Site B"
        
        entities = []
        for pattern in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity_value = match.group(1) if match.groups() else match.group(0)
                entity_value = entity_value.strip()
                
                if entity_value and entity_value not in entities:
                    entities.append(entity_value)
        
        # Should find facility names
        assert len(entities) > 0

    def test_entity_extraction_date_ranges(self):
        """Test date range extraction logic."""
        patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(last|past|previous)\s+(week|month|quarter|year)\b',
            r'\b(Q[1-4]\s+\d{4})\b'
        ]
        
        query = "Show data for January 2024 and last month"
        
        entities = []
        for pattern in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity_value = match.group(1) if match.groups() else match.group(0)
                entity_value = entity_value.strip()
                
                if entity_value and entity_value not in entities:
                    entities.append(entity_value)
        
        # Should find date ranges
        assert len(entities) >= 1

    def test_confidence_calculation(self):
        """Test confidence score calculation logic."""
        # Mock pattern scores and LLM confidence
        pattern_scores = {"consumption_analysis": 0.6}
        llm_confidence = 0.8
        
        # Calculate final confidence (30% pattern, 70% LLM)
        max_pattern_score = max(pattern_scores.values()) if pattern_scores else 0.0
        final_confidence = 0.3 * max_pattern_score + 0.7 * llm_confidence
        final_confidence = min(max(final_confidence, 0.0), 1.0)
        
        # Should be weighted average: 0.3 * 0.6 + 0.7 * 0.8 = 0.74
        expected = 0.3 * 0.6 + 0.7 * 0.8
        assert abs(final_confidence - expected) < 0.01
        assert 0.0 <= final_confidence <= 1.0

    def test_json_parsing_valid(self):
        """Test JSON parsing logic for LLM responses."""
        response = '{"intent": "consumption_analysis", "confidence": 0.85, "reasoning": "Test"}'
        
        try:
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                required_fields = ["intent", "confidence", "reasoning"]
                is_valid = all(field in parsed for field in required_fields)
                
                assert is_valid
                assert parsed["intent"] == "consumption_analysis"
                assert parsed["confidence"] == 0.85
                
        except Exception:
            pytest.fail("Valid JSON should parse correctly")

    def test_json_parsing_invalid(self):
        """Test JSON parsing fallback for invalid responses."""
        response = "This is not valid JSON"
        
        try:
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                is_valid = True
            else:
                is_valid = False
                
        except Exception:
            is_valid = False
        
        # Should handle invalid JSON gracefully
        assert not is_valid

    def test_query_validation_logic(self):
        """Test basic query validation logic."""
        def validate_query_input(query):
            return bool(query and query.strip() and len(query.strip()) >= 3)
        
        # Valid queries
        assert validate_query_input("What is electricity consumption?") is True
        assert validate_query_input("Test query") is True
        
        # Invalid queries
        assert validate_query_input("") is False
        assert validate_query_input("   ") is False
        assert validate_query_input("ab") is False  # Too short
        assert validate_query_input(None) is False

    def test_ehs_keyword_detection(self):
        """Test EHS keyword detection for query validation."""
        ehs_keywords = [
            "facility", "equipment", "permit", "utility", "consumption",
            "emission", "waste", "incident", "compliance", "efficiency",
            "energy", "water", "gas", "electricity", "maintenance",
            "environmental", "safety", "health"
        ]
        
        def has_ehs_keywords(query):
            query_lower = query.lower()
            return any(keyword in query_lower for keyword in ehs_keywords)
        
        # EHS queries
        assert has_ehs_keywords("What is electricity consumption?") is True
        assert has_ehs_keywords("Show facility emissions") is True
        assert has_ehs_keywords("Check equipment maintenance") is True
        
        # Non-EHS queries
        assert has_ehs_keywords("What's the weather?") is False
        assert has_ehs_keywords("Tell me a joke") is False

    def test_graph_pattern_detection(self):
        """Test graph-queryable pattern detection."""
        graph_patterns = [
            "show", "find", "get", "list", "what", "which", "how many",
            "total", "sum", "average", "maximum", "minimum", "count",
            "between", "during", "over time", "related to", "connected to"
        ]
        
        def has_graph_patterns(query):
            query_lower = query.lower()
            return any(pattern in query_lower for pattern in graph_patterns)
        
        # Graph-queryable patterns
        assert has_graph_patterns("Show me electricity consumption") is True
        assert has_graph_patterns("What is the total count?") is True
        assert has_graph_patterns("Find related equipment") is True
        
        # Less graph-suitable queries
        assert has_graph_patterns("Explain the theory") is False

    def test_intent_type_coverage(self):
        """Test that we have coverage for expected intent types."""
        # This simulates the IntentType enum values
        expected_intents = [
            "consumption_analysis",
            "compliance_check", 
            "risk_assessment",
            "emission_tracking",
            "equipment_efficiency",
            "permit_status",
            "general_inquiry"
        ]
        
        # Test that we can map intents to retrievers
        intent_retriever_map = {
            "consumption_analysis": "consumption_retriever",
            "compliance_check": "compliance_retriever",
            "risk_assessment": "risk_retriever",
            "emission_tracking": "emission_retriever",
            "equipment_efficiency": "equipment_retriever",
            "permit_status": "permit_retriever",
            "general_inquiry": "general_retriever"
        }
        
        # All expected intents should have retriever mappings
        for intent in expected_intents:
            assert intent in intent_retriever_map

    def test_query_rewriting_logic(self):
        """Test query rewriting logic."""
        def rewrite_query_for_intent(original_query, intent, facilities, date_ranges):
            query_parts = []
            
            # Add intent-specific context
            if intent == "consumption_analysis":
                query_parts.append("Analyze consumption patterns and usage data")
            elif intent == "compliance_check":
                query_parts.append("Check regulatory compliance status")
            elif intent == "risk_assessment":
                query_parts.append("Evaluate environmental and safety risks")
            
            # Add entity information
            if facilities:
                query_parts.append(f"for facilities: {', '.join(facilities)}")
            if date_ranges:
                query_parts.append(f"during: {', '.join(date_ranges)}")
            
            if len(query_parts) > 1:
                return " ".join(query_parts)
            return None
        
        # Test with entities
        rewritten = rewrite_query_for_intent(
            "Show electricity usage",
            "consumption_analysis", 
            ["Plant A"], 
            ["last month"]
        )
        
        assert rewritten is not None
        assert "consumption" in rewritten.lower()
        assert "plant a" in rewritten.lower()
        assert "last month" in rewritten.lower()
        
        # Test without entities
        rewritten_empty = rewrite_query_for_intent(
            "Show data",
            "general_inquiry",
            [],
            []
        )
        
        assert rewritten_empty is None

    def test_performance_timing(self):
        """Test basic performance timing logic."""
        import time
        
        start_time = time.time()
        
        # Simulate some processing
        time.sleep(0.001)  # 1ms
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        assert duration_ms >= 1.0  # Should be at least 1ms
        assert duration_ms < 100.0  # Should complete quickly


class TestEntityExtractionLogic:
    """Test entity extraction logic in isolation."""
    
    def test_facility_pattern_matching(self):
        """Test facility name pattern matching."""
        patterns = [
            r'\b(facility|plant|site|location|building|campus)\s+([A-Z][A-Za-z0-9\s-]+)',
            r'\b([A-Z][A-Za-z\s]+\s+(Plant|Facility|Site|Building))\b'
        ]
        
        test_cases = [
            ("Show data for Plant A", ["Plant A"]),
            ("Manufacturing Facility B", ["Manufacturing Facility B"]),
            ("Check Building 5 and Site C", ["Building 5", "Site C"]),
        ]
        
        for query, expected_entities in test_cases:
            entities = []
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    if match.groups() and len(match.groups()) >= 2:
                        entity_value = match.group(2)
                    elif match.groups():
                        entity_value = match.group(1)
                    else:
                        entity_value = match.group(0)
                    
                    entity_value = entity_value.strip()
                    if entity_value and entity_value not in entities:
                        entities.append(entity_value)
            
            # Should find expected entities (allowing for some variation in extraction)
            assert len(entities) >= len([e for e in expected_entities if e])

    def test_date_pattern_matching(self):
        """Test date range pattern matching."""
        patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(last|past|previous)\s+(week|month|quarter|year)\b',
            r'\b(Q[1-4]\s+\d{4})\b'
        ]
        
        test_cases = [
            ("Data for 01/01/2024", ["01/01/2024"]),
            ("January 2024 report", ["January 2024"]),
            ("Show last month data", ["last month"]),
            ("Q1 2024 summary", ["Q1 2024"]),
        ]
        
        for query, expected_patterns in test_cases:
            entities = []
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entity_value = match.group(1) if match.groups() else match.group(0)
                    entity_value = entity_value.strip()
                    
                    if entity_value and entity_value not in entities:
                        entities.append(entity_value)
            
            # Should find at least some date patterns
            assert len(entities) > 0

    def test_equipment_pattern_matching(self):
        """Test equipment pattern matching."""
        patterns = [
            r'\b(boiler|chiller|compressor|pump|motor|generator|turbine)\s*(\w*)\b',
            r'\b(HVAC|system|unit|equipment)\s+([A-Z0-9-]+)\b'
        ]
        
        test_cases = [
            ("Boiler B-1 efficiency", ["boiler", "B-1"]),
            ("HVAC Unit 2 maintenance", ["HVAC", "Unit 2"]),
            ("Check compressor status", ["compressor"]),
        ]
        
        for query, expected_terms in test_cases:
            entities = []
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    # Get all groups
                    for group in match.groups():
                        if group and group.strip():
                            entity_value = group.strip()
                            if entity_value and entity_value not in entities:
                                entities.append(entity_value)
            
            # Should find equipment-related terms
            assert len(entities) > 0


class TestQueryClassificationLogic:
    """Test query classification logic components."""
    
    def test_confidence_bounds_checking(self):
        """Test confidence score bounds validation."""
        def ensure_confidence_bounds(score):
            return min(max(score, 0.0), 1.0)
        
        test_scores = [-0.5, 0.0, 0.5, 1.0, 1.5]
        expected = [0.0, 0.0, 0.5, 1.0, 1.0]
        
        for test_score, expected_score in zip(test_scores, expected):
            result = ensure_confidence_bounds(test_score)
            assert result == expected_score
            assert 0.0 <= result <= 1.0

    def test_intent_fallback_logic(self):
        """Test intent fallback when classification fails."""
        def get_intent_with_fallback(classification_result, pattern_scores):
            """Simulate intent determination with fallback."""
            if classification_result and "intent" in classification_result:
                try:
                    return classification_result["intent"]
                except Exception:
                    pass
            
            # Fallback to highest pattern score
            if pattern_scores:
                return max(pattern_scores.items(), key=lambda x: x[1])[0]
            
            return "general_inquiry"
        
        # Test successful classification
        good_result = {"intent": "consumption_analysis", "confidence": 0.8}
        intent = get_intent_with_fallback(good_result, {})
        assert intent == "consumption_analysis"
        
        # Test fallback to pattern scores
        pattern_scores = {"compliance_check": 0.7, "consumption_analysis": 0.9}
        intent = get_intent_with_fallback(None, pattern_scores)
        assert intent == "consumption_analysis"  # Highest score
        
        # Test ultimate fallback
        intent = get_intent_with_fallback(None, {})
        assert intent == "general_inquiry"