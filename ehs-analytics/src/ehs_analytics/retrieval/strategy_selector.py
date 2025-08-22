"""
Strategy Selector for EHS Analytics Retrieval Orchestrator.

This module provides intelligent strategy selection logic that analyzes queries
to determine the optimal retriever(s) for execution. It includes rule-based
selection, ML-based selection (placeholder), and adaptive performance-based selection.
"""

import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

from .base import RetrievalStrategy, QueryType

logger = logging.getLogger(__name__)


class SelectionMethod(str, Enum):
    """Strategy selection methods."""
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"  # Placeholder for future ML implementation
    HYBRID = "hybrid"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class QueryCharacteristics:
    """Characteristics extracted from query analysis."""
    
    # Content analysis
    contains_time_references: bool = False
    contains_numerical_values: bool = False
    contains_entity_names: bool = False
    contains_relationship_terms: bool = False
    contains_aggregation_terms: bool = False
    
    # Query complexity
    word_count: int = 0
    sentence_count: int = 0
    complexity_score: float = 0.0  # 0-1 scale
    
    # Domain specificity
    ehs_domain_terms: Set[str] = field(default_factory=set)
    facility_references: Set[str] = field(default_factory=set)
    equipment_references: Set[str] = field(default_factory=set)
    
    # Query intent signals
    is_lookup_query: bool = False  # "What is", "Show me"
    is_analytical_query: bool = False  # "Analyze", "Compare", "Trend"
    is_comparative_query: bool = False  # "vs", "compared to", "better"
    is_temporal_query: bool = False  # "over time", "last month", "trends"
    is_predictive_query: bool = False  # "predict", "forecast", "will"


@dataclass
class SelectionResult:
    """Result of strategy selection process."""
    
    selected_strategies: List[RetrievalStrategy]
    strategy_confidences: Dict[RetrievalStrategy, float]
    selection_method: SelectionMethod
    reasoning: str
    execution_time_ms: float
    query_characteristics: QueryCharacteristics
    fallback_strategies: List[RetrievalStrategy] = field(default_factory=list)


class StrategySelector:
    """
    Intelligent strategy selector for retrieval orchestration.
    
    Analyzes incoming queries to determine optimal retrieval strategies
    based on query characteristics, performance history, and domain knowledge.
    """
    
    def __init__(self):
        """Initialize the strategy selector."""
        
        # Performance tracking
        self.strategy_performance: Dict[RetrievalStrategy, Dict[str, float]] = defaultdict(
            lambda: {
                'success_rate': 0.8,  # Default success rate
                'avg_response_time': 1000.0,  # Default response time in ms
                'avg_result_quality': 0.7,  # Default quality score
                'usage_count': 0
            }
        )
        
        # Query pattern tracking
        self.query_patterns: Dict[QueryType, List[str]] = {
            QueryType.CONSUMPTION: [
                r'\b(consumption|usage|use|consumed|spent)\b',
                r'\b(water|electricity|energy|gas|fuel)\b',
                r'\b(monthly|daily|yearly|annual|total)\b',
                r'\b(cost|bill|expense|charge)\b'
            ],
            QueryType.EFFICIENCY: [
                r'\b(efficiency|performance|optimize|improvement)\b',
                r'\b(equipment|machine|system|device)\b',
                r'\b(rating|score|benchmark|compare)\b',
                r'\b(energy efficient|power consumption)\b'
            ],
            QueryType.COMPLIANCE: [
                r'\b(permit|license|compliance|regulation)\b',
                r'\b(expired|expiring|due|deadline)\b',
                r'\b(status|valid|current|active)\b',
                r'\b(authority|agency|regulatory)\b'
            ],
            QueryType.EMISSIONS: [
                r'\b(emission|carbon|pollution|discharge)\b',
                r'\b(co2|methane|nox|particulate)\b',
                r'\b(level|amount|concentration|rate)\b',
                r'\b(source|generated|produced)\b'
            ],
            QueryType.RISK: [
                r'\b(risk|hazard|danger|safety|incident)\b',
                r'\b(assessment|analysis|evaluation)\b',
                r'\b(probability|likelihood|chance)\b',
                r'\b(mitigation|prevention|control)\b'
            ],
            QueryType.RECOMMENDATION: [
                r'\b(recommend|suggest|advice|best practice)\b',
                r'\b(improve|optimize|reduce|minimize)\b',
                r'\b(should|could|might|consider)\b',
                r'\b(action|step|measure|solution)\b'
            ]
        }
        
        # Strategy routing rules
        self.strategy_rules = self._initialize_strategy_rules()
        
        # EHS domain vocabulary
        self.ehs_terms = {
            'facilities': {'facility', 'plant', 'site', 'location', 'building', 'campus'},
            'equipment': {'equipment', 'machine', 'device', 'system', 'instrument', 'tool'},
            'utilities': {'water', 'electricity', 'gas', 'energy', 'power', 'fuel'},
            'emissions': {'emission', 'carbon', 'co2', 'pollution', 'discharge', 'waste'},
            'compliance': {'permit', 'license', 'regulation', 'compliance', 'authority'},
            'safety': {'safety', 'risk', 'hazard', 'incident', 'accident', 'danger'},
            'time': {'daily', 'weekly', 'monthly', 'yearly', 'annual', 'trend', 'over time', 'since', 'until'}
        }
    
    def _initialize_strategy_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rule-based strategy selection logic."""
        
        return {
            # Simple lookup queries - prefer Text2Cypher
            'simple_lookup': {
                'patterns': [
                    r'\b(what is|show me|list|find|get)\b',
                    r'\b(facility|equipment|permit) (named|called|with)\b',
                    r'\bstatus of\b'
                ],
                'strategies': [RetrievalStrategy.TEXT2CYPHER],
                'confidence': 0.9,
                'conditions': ['word_count < 10', 'not contains_aggregation_terms']
            },
            
            # Document search queries - prefer Vector
            'document_search': {
                'patterns': [
                    r'\b(document|report|manual|guide|procedure)\b',
                    r'\b(search for|find documents|documentation)\b',
                    r'\b(similar to|like|related to)\b'
                ],
                'strategies': [RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID],
                'confidence': 0.85,
                'conditions': ['word_count > 5']
            },
            
            # Relationship queries - prefer VectorCypher
            'relationship_queries': {
                'patterns': [
                    r'\b(connect|relate|associate|link)\b',
                    r'\b(between|and|with|from|to)\b',
                    r'\b(path|route|connection|relationship)\b'
                ],
                'strategies': [RetrievalStrategy.VECTOR_CYPHER, RetrievalStrategy.HYBRID_CYPHER],
                'confidence': 0.8,
                'conditions': ['contains_relationship_terms']
            },
            
            # Temporal analysis - prefer HybridCypher
            'temporal_analysis': {
                'patterns': [
                    r'\b(trend|over time|pattern|change)\b',
                    r'\b(last|past|previous|since|until)\b',
                    r'\b(month|year|day|week|quarter)\b',
                    r'\b(forecast|predict|future)\b'
                ],
                'strategies': [RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.HYBRID],
                'confidence': 0.85,
                'conditions': ['contains_time_references']
            },
            
            # Aggregation queries - prefer Hybrid approaches
            'aggregation_queries': {
                'patterns': [
                    r'\b(total|sum|average|count|maximum|minimum)\b',
                    r'\b(aggregate|summarize|analyze|compare)\b',
                    r'\b(group by|breakdown|distribution)\b'
                ],
                'strategies': [RetrievalStrategy.HYBRID, RetrievalStrategy.HYBRID_CYPHER],
                'confidence': 0.8,
                'conditions': ['contains_aggregation_terms']
            },
            
            # Complex analytical queries - prefer multiple strategies
            'complex_analysis': {
                'patterns': [
                    r'\b(analyze|assessment|evaluation|study)\b',
                    r'\b(correlation|impact|effect|influence)\b',
                    r'\b(optimization|improvement|efficiency)\b'
                ],
                'strategies': [
                    RetrievalStrategy.HYBRID_CYPHER,
                    RetrievalStrategy.VECTOR_CYPHER,
                    RetrievalStrategy.HYBRID
                ],
                'confidence': 0.75,
                'conditions': ['complexity_score > 0.6']
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the strategy selector."""
        logger.info("Strategy selector initialized")
    
    async def select_strategies(
        self,
        query: str,
        query_type: QueryType,
        available_strategies: List[RetrievalStrategy],
        max_strategies: int = 3,
        min_confidence: float = 0.7,
        selection_method: SelectionMethod = SelectionMethod.HYBRID
    ) -> SelectionResult:
        """
        Select optimal retrieval strategies for the given query.
        
        Args:
            query: Natural language query
            query_type: Classified query type
            available_strategies: List of available retrieval strategies
            max_strategies: Maximum number of strategies to select
            min_confidence: Minimum confidence threshold
            selection_method: Method to use for selection
            
        Returns:
            SelectionResult with selected strategies and metadata
        """
        start_time = time.time()
        
        # Analyze query characteristics
        characteristics = self._analyze_query(query, query_type)
        
        # Select strategies based on method
        if selection_method == SelectionMethod.RULE_BASED:
            selected_strategies, confidences, reasoning = self._rule_based_selection(
                query, query_type, characteristics, available_strategies
            )
        elif selection_method == SelectionMethod.PERFORMANCE_BASED:
            selected_strategies, confidences, reasoning = self._performance_based_selection(
                query, query_type, characteristics, available_strategies
            )
        elif selection_method == SelectionMethod.ML_BASED:
            # Placeholder for future ML implementation
            selected_strategies, confidences, reasoning = self._ml_based_selection(
                query, query_type, characteristics, available_strategies
            )
        else:  # HYBRID
            selected_strategies, confidences, reasoning = self._hybrid_selection(
                query, query_type, characteristics, available_strategies
            )
        
        # Filter by confidence threshold and limit
        filtered_strategies = []
        filtered_confidences = {}
        
        for strategy in selected_strategies:
            if (strategy in available_strategies and 
                confidences.get(strategy, 0.0) >= min_confidence and
                len(filtered_strategies) < max_strategies):
                filtered_strategies.append(strategy)
                filtered_confidences[strategy] = confidences[strategy]
        
        # Ensure at least one strategy is selected (fallback)
        if not filtered_strategies and available_strategies:
            fallback_strategy = self._select_fallback_strategy(query_type, available_strategies)
            filtered_strategies = [fallback_strategy]
            filtered_confidences = {fallback_strategy: 0.5}
            reasoning += " Applied fallback strategy selection."
        
        # Determine fallback strategies
        fallback_strategies = [
            strategy for strategy in available_strategies 
            if strategy not in filtered_strategies
        ][:2]  # Keep top 2 as fallbacks
        
        execution_time = (time.time() - start_time) * 1000
        
        result = SelectionResult(
            selected_strategies=filtered_strategies,
            strategy_confidences=filtered_confidences,
            selection_method=selection_method,
            reasoning=reasoning,
            execution_time_ms=execution_time,
            query_characteristics=characteristics,
            fallback_strategies=fallback_strategies
        )
        
        logger.info(
            f"Selected {len(filtered_strategies)} strategies for query type {query_type.value}: "
            f"{[s.value for s in filtered_strategies]}"
        )
        
        return result
    
    def _analyze_query(self, query: str, query_type: QueryType) -> QueryCharacteristics:
        """Analyze query to extract characteristics for strategy selection."""
        
        query_lower = query.lower()
        words = query_lower.split()
        sentences = query.split('.')
        
        characteristics = QueryCharacteristics(
            word_count=len(words),
            sentence_count=len(sentences),
            complexity_score=self._calculate_complexity_score(query)
        )
        
        # Detect time references
        time_patterns = [
            r'\b(last|past|previous|since|until|ago)\b',
            r'\b(day|week|month|year|quarter)\b',
            r'\b(daily|weekly|monthly|yearly|annual)\b',
            r'\b(trend|over time|pattern|change)\b'
        ]
        characteristics.contains_time_references = any(
            re.search(pattern, query_lower) for pattern in time_patterns
        )
        
        # Detect numerical values
        characteristics.contains_numerical_values = bool(
            re.search(r'\b\d+(?:\.\d+)?\b', query)
        )
        
        # Detect entity names (capitalized words)
        characteristics.contains_entity_names = bool(
            re.search(r'\b[A-Z][a-z]+\b', query)
        )
        
        # Detect relationship terms
        relationship_terms = ['between', 'and', 'with', 'from', 'to', 'connect', 'relate', 'associate']
        characteristics.contains_relationship_terms = any(
            term in query_lower for term in relationship_terms
        )
        
        # Detect aggregation terms
        aggregation_terms = ['total', 'sum', 'average', 'count', 'maximum', 'minimum', 'aggregate']
        characteristics.contains_aggregation_terms = any(
            term in query_lower for term in aggregation_terms
        )
        
        # Extract EHS domain terms
        for category, terms in self.ehs_terms.items():
            found_terms = {term for term in terms if term in query_lower}
            if found_terms:
                characteristics.ehs_domain_terms.update(found_terms)
                if category == 'facilities':
                    characteristics.facility_references.update(found_terms)
                elif category == 'equipment':
                    characteristics.equipment_references.update(found_terms)
        
        # Detect query intent
        characteristics.is_lookup_query = bool(
            re.search(r'\b(what is|show me|list|find|get|display)\b', query_lower)
        )
        
        characteristics.is_analytical_query = bool(
            re.search(r'\b(analyze|assessment|evaluation|study|examine)\b', query_lower)
        )
        
        characteristics.is_comparative_query = bool(
            re.search(r'\b(vs|versus|compared to|compare|better|worse)\b', query_lower)
        )
        
        characteristics.is_temporal_query = characteristics.contains_time_references
        
        characteristics.is_predictive_query = bool(
            re.search(r'\b(predict|forecast|future|will|expect)\b', query_lower)
        )
        
        return characteristics
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Calculate query complexity score (0-1 scale)."""
        
        factors = []
        
        # Word count factor
        word_count = len(query.split())
        factors.append(min(word_count / 20.0, 1.0))  # Normalize to 20 words
        
        # Sentence count factor
        sentence_count = len(query.split('.'))
        factors.append(min(sentence_count / 3.0, 1.0))  # Normalize to 3 sentences
        
        # Complex terms factor
        complex_terms = [
            'analyze', 'assessment', 'evaluation', 'optimization',
            'correlation', 'comparison', 'prediction', 'forecast'
        ]
        complex_term_count = sum(1 for term in complex_terms if term in query.lower())
        factors.append(min(complex_term_count / 3.0, 1.0))
        
        # Conditional/logical terms factor
        logical_terms = ['and', 'or', 'but', 'however', 'if', 'when', 'where']
        logical_term_count = sum(1 for term in logical_terms if term in query.lower())
        factors.append(min(logical_term_count / 3.0, 1.0))
        
        return sum(factors) / len(factors)
    
    def _rule_based_selection(
        self,
        query: str,
        query_type: QueryType,
        characteristics: QueryCharacteristics,
        available_strategies: List[RetrievalStrategy]
    ) -> Tuple[List[RetrievalStrategy], Dict[RetrievalStrategy, float], str]:
        """Rule-based strategy selection."""
        
        strategy_scores = defaultdict(float)
        reasoning_parts = []
        
        query_lower = query.lower()
        
        # Apply rule-based patterns
        for rule_name, rule in self.strategy_rules.items():
            pattern_matches = sum(
                1 for pattern in rule['patterns']
                if re.search(pattern, query_lower)
            )
            
            if pattern_matches > 0:
                # Check conditions
                conditions_met = True
                for condition in rule.get('conditions', []):
                    if not self._evaluate_condition(condition, characteristics):
                        conditions_met = False
                        break
                
                if conditions_met:
                    confidence_boost = rule['confidence'] * (pattern_matches / len(rule['patterns']))
                    for strategy in rule['strategies']:
                        if strategy in available_strategies:
                            strategy_scores[strategy] += confidence_boost
                    
                    reasoning_parts.append(
                        f"Rule '{rule_name}' matched with {pattern_matches} patterns"
                    )
        
        # Query type specific boosting
        if query_type == QueryType.CONSUMPTION:
            strategy_scores[RetrievalStrategy.TEXT2CYPHER] += 0.2
            strategy_scores[RetrievalStrategy.HYBRID] += 0.1
        elif query_type == QueryType.EFFICIENCY:
            strategy_scores[RetrievalStrategy.HYBRID] += 0.2
            strategy_scores[RetrievalStrategy.VECTOR_CYPHER] += 0.1
        elif query_type == QueryType.COMPLIANCE:
            strategy_scores[RetrievalStrategy.TEXT2CYPHER] += 0.3
        elif query_type == QueryType.EMISSIONS:
            strategy_scores[RetrievalStrategy.HYBRID_CYPHER] += 0.2
            strategy_scores[RetrievalStrategy.VECTOR_CYPHER] += 0.1
        elif query_type == QueryType.RISK:
            strategy_scores[RetrievalStrategy.HYBRID_CYPHER] += 0.3
            strategy_scores[RetrievalStrategy.HYBRID] += 0.2
        elif query_type == QueryType.RECOMMENDATION:
            strategy_scores[RetrievalStrategy.HYBRID_CYPHER] += 0.2
            strategy_scores[RetrievalStrategy.HYBRID] += 0.2
            strategy_scores[RetrievalStrategy.VECTOR_CYPHER] += 0.1
        
        # Sort strategies by score
        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_strategies = [strategy for strategy, _ in sorted_strategies if strategy in available_strategies]
        confidences = dict(sorted_strategies)
        
        reasoning = "Rule-based selection: " + "; ".join(reasoning_parts)
        
        return selected_strategies, confidences, reasoning
    
    def _performance_based_selection(
        self,
        query: str,
        query_type: QueryType,
        characteristics: QueryCharacteristics,
        available_strategies: List[RetrievalStrategy]
    ) -> Tuple[List[RetrievalStrategy], Dict[RetrievalStrategy, float], str]:
        """Performance-based strategy selection using historical data."""
        
        strategy_scores = {}
        
        for strategy in available_strategies:
            if strategy in self.strategy_performance:
                perf = self.strategy_performance[strategy]
                
                # Calculate composite performance score
                success_weight = 0.4
                speed_weight = 0.3
                quality_weight = 0.3
                
                # Normalize response time (lower is better)
                normalized_speed = max(0, 1.0 - (perf['avg_response_time'] / 5000.0))
                
                composite_score = (
                    perf['success_rate'] * success_weight +
                    normalized_speed * speed_weight +
                    perf['avg_result_quality'] * quality_weight
                )
                
                strategy_scores[strategy] = composite_score
            else:
                # Default score for unknown strategies
                strategy_scores[strategy] = 0.6
        
        # Sort by performance score
        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_strategies = [strategy for strategy, _ in sorted_strategies]
        confidences = dict(sorted_strategies)
        
        reasoning = f"Performance-based selection using historical data"
        
        return selected_strategies, confidences, reasoning
    
    def _ml_based_selection(
        self,
        query: str,
        query_type: QueryType,
        characteristics: QueryCharacteristics,
        available_strategies: List[RetrievalStrategy]
    ) -> Tuple[List[RetrievalStrategy], Dict[RetrievalStrategy, float], str]:
        """ML-based strategy selection (placeholder for future implementation)."""
        
        # Placeholder implementation - falls back to rule-based for now
        logger.info("ML-based selection not yet implemented, falling back to rule-based")
        
        return self._rule_based_selection(query, query_type, characteristics, available_strategies)
    
    def _hybrid_selection(
        self,
        query: str,
        query_type: QueryType,
        characteristics: QueryCharacteristics,
        available_strategies: List[RetrievalStrategy]
    ) -> Tuple[List[RetrievalStrategy], Dict[RetrievalStrategy, float], str]:
        """Hybrid selection combining rule-based and performance-based approaches."""
        
        # Get rule-based results
        rule_strategies, rule_confidences, rule_reasoning = self._rule_based_selection(
            query, query_type, characteristics, available_strategies
        )
        
        # Get performance-based results
        perf_strategies, perf_confidences, perf_reasoning = self._performance_based_selection(
            query, query_type, characteristics, available_strategies
        )
        
        # Combine scores with weights
        rule_weight = 0.7
        perf_weight = 0.3
        
        combined_scores = {}
        for strategy in available_strategies:
            rule_score = rule_confidences.get(strategy, 0.0)
            perf_score = perf_confidences.get(strategy, 0.5)
            
            combined_scores[strategy] = (
                rule_score * rule_weight + perf_score * perf_weight
            )
        
        # Sort by combined score
        sorted_strategies = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_strategies = [strategy for strategy, _ in sorted_strategies]
        confidences = dict(sorted_strategies)
        
        reasoning = f"Hybrid selection: {rule_reasoning}; {perf_reasoning}"
        
        return selected_strategies, confidences, reasoning
    
    def _evaluate_condition(self, condition: str, characteristics: QueryCharacteristics) -> bool:
        """Evaluate a condition string against query characteristics."""
        
        # Simple condition evaluation
        if 'word_count < 10' in condition:
            return characteristics.word_count < 10
        elif 'word_count > 5' in condition:
            return characteristics.word_count > 5
        elif 'not contains_aggregation_terms' in condition:
            return not characteristics.contains_aggregation_terms
        elif 'contains_relationship_terms' in condition:
            return characteristics.contains_relationship_terms
        elif 'contains_time_references' in condition:
            return characteristics.contains_time_references
        elif 'contains_aggregation_terms' in condition:
            return characteristics.contains_aggregation_terms
        elif 'complexity_score > 0.6' in condition:
            return characteristics.complexity_score > 0.6
        
        return True  # Default to true for unknown conditions
    
    def _select_fallback_strategy(
        self,
        query_type: QueryType,
        available_strategies: List[RetrievalStrategy]
    ) -> RetrievalStrategy:
        """Select fallback strategy based on query type."""
        
        # Define fallback preferences by query type
        fallback_preferences = {
            QueryType.CONSUMPTION: [RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.HYBRID],
            QueryType.EFFICIENCY: [RetrievalStrategy.HYBRID, RetrievalStrategy.VECTOR_CYPHER],
            QueryType.COMPLIANCE: [RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR],
            QueryType.EMISSIONS: [RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.HYBRID],
            QueryType.RISK: [RetrievalStrategy.HYBRID_CYPHER, RetrievalStrategy.VECTOR_CYPHER],
            QueryType.RECOMMENDATION: [RetrievalStrategy.HYBRID, RetrievalStrategy.HYBRID_CYPHER],
            QueryType.GENERAL: [RetrievalStrategy.HYBRID, RetrievalStrategy.TEXT2CYPHER]
        }
        
        preferences = fallback_preferences.get(query_type, [RetrievalStrategy.TEXT2CYPHER])
        
        for preferred_strategy in preferences:
            if preferred_strategy in available_strategies:
                return preferred_strategy
        
        # Final fallback - return first available strategy
        return available_strategies[0] if available_strategies else RetrievalStrategy.TEXT2CYPHER
    
    def update_performance_metrics(
        self,
        strategy: RetrievalStrategy,
        success: bool,
        response_time_ms: float,
        result_quality: float = 0.7
    ) -> None:
        """Update performance metrics for a strategy."""
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'success_rate': 0.8,
                'avg_response_time': 1000.0,
                'avg_result_quality': 0.7,
                'usage_count': 0
            }
        
        perf = self.strategy_performance[strategy]
        perf['usage_count'] += 1
        
        # Update running averages
        count = perf['usage_count']
        alpha = 1.0 / count  # Simple moving average
        
        perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * (1.0 if success else 0.0)
        perf['avg_response_time'] = (1 - alpha) * perf['avg_response_time'] + alpha * response_time_ms
        perf['avg_result_quality'] = (1 - alpha) * perf['avg_result_quality'] + alpha * result_quality
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get current strategy performance statistics."""
        
        return {
            'strategy_performance': dict(self.strategy_performance),
            'total_selections': sum(
                perf['usage_count'] for perf in self.strategy_performance.values()
            )
        }