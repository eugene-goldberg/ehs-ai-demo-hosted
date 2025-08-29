"""
LLM Query Templates for Neo4j Enhanced Queries

This module provides comprehensive templates for LLM-driven data retrieval,
analysis, and context aggregation from Neo4j graphs.

Created: 2025-08-28
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Enum for different types of LLM queries"""
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    PATTERN_DISCOVERY = "pattern_discovery"
    CONTEXTUAL_SEARCH = "contextual_search"
    SUMMARIZATION = "summarization"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"


class AnalysisType(Enum):
    """Enum for different analysis types"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


@dataclass
class QueryTemplate:
    """Template structure for LLM queries"""
    name: str
    description: str
    cypher_template: str
    prompt_template: str
    parameters: Dict[str, Any]
    expected_output: str
    use_case: str
    performance_hints: List[str]


class CypherQueryTemplates:
    """Cypher query templates for LLM data retrieval"""

    @staticmethod
    def knowledge_extraction_basic() -> str:
        """Basic knowledge extraction query template"""
        return """
        MATCH (n:{node_label})
        WHERE {filter_conditions}
        OPTIONAL MATCH (n)-[r]->(m)
        WITH n, collect({{
            relationship: type(r),
            target: labels(m)[0],
            target_properties: m{{.name, .id, .type}}
        }}) as relationships
        RETURN n{{
            .id,
            .name,
            .type,
            .description,
            .properties,
            .created_at,
            .updated_at
        }} as node,
        relationships,
        labels(n) as node_labels
        ORDER BY n.name
        LIMIT {limit}
        """

    @staticmethod
    def relationship_analysis_deep() -> str:
        """Deep relationship analysis query template"""
        return """
        MATCH path = (start:{start_label})-[*1..{max_depth}]-(end:{end_label})
        WHERE {start_conditions} AND {end_conditions}
        WITH path,
             [rel in relationships(path) | {{
                 type: type(rel),
                 properties: properties(rel),
                 start_node: startNode(rel){{.name, .id}},
                 end_node: endNode(rel){{.name, .id}}
             }}] as relationship_chain,
             length(path) as path_length
        WHERE path_length >= {min_depth}
        RETURN {{
            start_node: startNode(path){{.*, labels: labels(startNode(path))}},
            end_node: endNode(path){{.*, labels: labels(endNode(path))}},
            relationship_chain: relationship_chain,
            path_length: path_length,
            path_strength: reduce(s = 0.0, rel in relationships(path) | s + coalesce(rel.weight, 1.0))
        }} as analysis_result
        ORDER BY path_length, path_strength DESC
        LIMIT {limit}
        """

    @staticmethod
    def pattern_discovery_clustering() -> str:
        """Pattern discovery with clustering query template"""
        return """
        MATCH (n:{node_label})
        WHERE {node_conditions}
        OPTIONAL MATCH (n)-[r:{relationship_type}]-(m)
        WITH n, 
             collect(DISTINCT type(r)) as relationship_types,
             collect(DISTINCT labels(m)[0]) as connected_node_types,
             count(DISTINCT m) as connection_count
        WITH n, relationship_types, connected_node_types, connection_count,
             // Create pattern signature
             apoc.text.join(apoc.coll.sort(relationship_types + connected_node_types), "_") as pattern_signature
        WITH pattern_signature, 
             collect({{
                 node: n{{.*}},
                 connection_count: connection_count,
                 relationship_types: relationship_types,
                 connected_node_types: connected_node_types
             }}) as nodes_with_pattern,
             count(*) as pattern_frequency
        WHERE pattern_frequency >= {min_pattern_frequency}
        RETURN {{
            pattern_signature: pattern_signature,
            pattern_frequency: pattern_frequency,
            sample_nodes: nodes_with_pattern[0..{sample_size}],
            pattern_strength: pattern_frequency * avg([node in nodes_with_pattern | node.connection_count])
        }} as pattern_result
        ORDER BY pattern_strength DESC
        LIMIT {limit}
        """

    @staticmethod
    def contextual_search_semantic() -> str:
        """Semantic contextual search query template"""
        return """
        CALL db.index.vector.queryNodes('{vector_index_name}', {k}, {query_vector})
        YIELD node as primary_node, score as similarity_score
        WHERE similarity_score >= {similarity_threshold}
        
        // Expand context around semantically similar nodes
        OPTIONAL MATCH (primary_node)-[r1:{context_relationship}*1..{context_depth}]-(context_node)
        WITH primary_node, similarity_score, 
             collect(DISTINCT {{
                 node: context_node{{.*}},
                 labels: labels(context_node),
                 distance: length([rel in r1 | rel])
             }}) as context_nodes
        
        // Add temporal context if available
        OPTIONAL MATCH (primary_node)-[:OCCURRED_AT|CREATED_AT|MODIFIED_AT]->(time_node)
        WITH primary_node, similarity_score, context_nodes,
             collect(time_node{{.*}}) as temporal_context
        
        RETURN {{
            primary_node: primary_node{{.*}},
            similarity_score: similarity_score,
            context_nodes: context_nodes[0..{max_context_nodes}],
            temporal_context: temporal_context,
            context_summary: {{
                total_context_nodes: size(context_nodes),
                context_types: [node in context_nodes | node.labels[0]]
            }}
        }} as search_result
        ORDER BY similarity_score DESC
        LIMIT {limit}
        """

    @staticmethod
    def aggregation_summary() -> str:
        """Context aggregation and summarization query template"""
        return """
        MATCH (n:{primary_label})
        WHERE {primary_conditions}
        
        // Aggregate different types of related information
        OPTIONAL MATCH (n)-[:HAS_ATTRIBUTE]->(attr)
        WITH n, collect(attr{{.*}}) as attributes
        
        OPTIONAL MATCH (n)-[:RELATES_TO]->(related)
        WITH n, attributes, 
             collect({{
                 entity: related{{.name, .type, .id}},
                 labels: labels(related)
             }}) as related_entities
        
        OPTIONAL MATCH (n)-[:HAS_EVENT]->(event)
        WITH n, attributes, related_entities,
             collect(event{{.*, timestamp: event.timestamp}}) as events
        
        OPTIONAL MATCH (n)-[:HAS_METRIC]->(metric)
        WITH n, attributes, related_entities, events,
             collect({{
                 name: metric.name,
                 value: metric.value,
                 unit: metric.unit,
                 timestamp: metric.timestamp
             }}) as metrics
        
        RETURN {{
            entity: n{{.*}},
            summary: {{
                attributes_count: size(attributes),
                related_entities_count: size(related_entities),
                events_count: size(events),
                metrics_count: size(metrics),
                last_update: max([event in events | event.timestamp])
            }},
            attributes: attributes,
            related_entities: related_entities[0..{max_related}],
            recent_events: [event in events WHERE event.timestamp > {recent_threshold}][0..{max_events}],
            key_metrics: metrics[0..{max_metrics}]
        }} as aggregated_result
        ORDER BY n.importance DESC, n.updated_at DESC
        LIMIT {limit}
        """

    @staticmethod
    def multi_step_analysis() -> str:
        """Multi-step query chain template"""
        return """
        // Step 1: Identify seed nodes
        WITH {seed_parameters} as seed_params
        MATCH (seed:{seed_label})
        WHERE {seed_conditions}
        
        // Step 2: Find related entities
        WITH seed, seed_params
        MATCH (seed)-[r1:{step1_relationship}*1..{step1_depth}]-(related1)
        WHERE {step1_conditions}
        
        // Step 3: Cross-reference analysis
        WITH seed, related1, seed_params
        MATCH (related1)-[r2:{step2_relationship}]-(related2)
        WHERE {step2_conditions}
        
        // Step 4: Pattern scoring and ranking
        WITH seed, related1, related2,
             // Calculate relationship strength
             reduce(strength = 0.0, rel in r1 | strength + coalesce(rel.weight, 1.0)) as step1_strength,
             coalesce(r2.weight, 1.0) as step2_strength
        
        // Step 5: Aggregate results
        WITH seed, 
             collect(DISTINCT {{
                 intermediate: related1{{.*}},
                 target: related2{{.*}},
                 path_strength: step1_strength + step2_strength,
                 confidence: (step1_strength + step2_strength) / ({step1_depth} + 1)
             }}) as analysis_chain
        
        RETURN {{
            seed_entity: seed{{.*}},
            analysis_chain: [item in analysis_chain WHERE item.confidence >= {min_confidence}][0..{max_results}],
            total_paths: size(analysis_chain),
            avg_confidence: avg([item in analysis_chain | item.confidence])
        }} as multi_step_result
        ORDER BY avg_confidence DESC
        LIMIT {limit}
        """

    @staticmethod
    def performance_optimized_batch() -> str:
        """Performance-optimized batch processing query template"""
        return """
        // Use parameterized batch processing for better performance
        UNWIND {batch_items} as item
        
        // Use index hints for optimal performance
        USING INDEX n:{node_label}({index_property})
        MATCH (n:{node_label} {{{index_property}: item.{parameter_key}}})
        
        // Collect related data efficiently
        OPTIONAL MATCH (n)-[r:{relationship_type}]->(related)
        WITH n, item, collect({{
            related: related{{.{required_properties}}},
            relationship: r{{.{relationship_properties}}}
        }}) as related_data
        
        // Apply business logic filters
        WHERE {business_logic_conditions}
        
        RETURN {{
            input_item: item,
            result: n{{.{required_node_properties}}},
            related_data: related_data[0..{max_related_per_item}],
            processing_metadata: {{
                query_timestamp: timestamp(),
                result_count: size(related_data)
            }}
        }} as batch_result
        """

    @staticmethod
    def error_handling_template() -> str:
        """Error handling and validation query template"""
        return """
        // Validate input parameters
        WITH {input_parameters} as params
        WHERE params.{required_field} IS NOT NULL AND size(params.{required_field}) > 0
        
        // Safe node matching with existence checks
        OPTIONAL MATCH (n:{node_label})
        WHERE n.{identifier_field} = params.{required_field}
        
        WITH params, n,
             CASE 
                 WHEN n IS NULL THEN {{error: "Node not found", code: "NODE_NOT_FOUND"}}
                 ELSE NULL
             END as error
        
        // Continue processing only if no errors
        WHERE error IS NULL
        
        // Safe relationship traversal
        OPTIONAL MATCH (n)-[r:{relationship_type}]-(related)
        WITH params, n, collect({{
            node: related{{.*}},
            relationship: r{{.*}},
            valid: related IS NOT NULL AND r IS NOT NULL
        }}) as relationships,
        CASE 
            WHEN size([rel in relationships WHERE rel.valid]) = 0 
            THEN {{error: "No valid relationships found", code: "NO_RELATIONSHIPS"}}
            ELSE NULL
        END as relationship_error
        
        RETURN {{
            success: relationship_error IS NULL,
            result: CASE 
                WHEN relationship_error IS NULL THEN {{
                    node: n{{.*}},
                    relationships: [rel in relationships WHERE rel.valid],
                    metadata: {{
                        query_timestamp: timestamp(),
                        relationship_count: size([rel in relationships WHERE rel.valid])
                    }}
                }}
                ELSE NULL
            END,
            error: relationship_error
        }} as query_result
        """


class PromptTemplates:
    """Prompt templates for different analysis types"""

    @staticmethod
    def knowledge_extraction_prompt() -> str:
        """Prompt template for knowledge extraction"""
        return """
        Based on the following graph data, extract key knowledge and insights:

        Context:
        - Primary Entity: {primary_entity}
        - Entity Type: {entity_type}
        - Analysis Focus: {analysis_focus}

        Graph Data:
        {graph_data}

        Please analyze this data and provide:
        1. Key facts and attributes about the primary entity
        2. Important relationships and their significance
        3. Patterns or trends you observe
        4. Notable characteristics or anomalies
        5. Contextual insights based on connected entities

        Format your response as a structured analysis with clear sections and bullet points.
        Focus on actionable insights and meaningful patterns.
        """

    @staticmethod
    def relationship_analysis_prompt() -> str:
        """Prompt template for relationship analysis"""
        return """
        Analyze the relationship patterns in the following graph data:

        Analysis Parameters:
        - Start Entity: {start_entity}
        - End Entity: {end_entity}
        - Relationship Depth: {relationship_depth}
        - Analysis Type: {analysis_type}

        Relationship Data:
        {relationship_data}

        Provide a comprehensive analysis including:
        1. Relationship pathway description
        2. Strength and significance of connections
        3. Intermediate entities and their roles
        4. Potential implications of these relationships
        5. Recommendations based on the relationship patterns

        Consider both direct and indirect relationships in your analysis.
        Highlight any unusual or particularly significant connection patterns.
        """

    @staticmethod
    def pattern_discovery_prompt() -> str:
        """Prompt template for pattern discovery"""
        return """
        Identify and analyze patterns in the following graph data:

        Pattern Analysis Context:
        - Data Domain: {data_domain}
        - Pattern Type: {pattern_type}
        - Minimum Pattern Frequency: {min_frequency}

        Pattern Data:
        {pattern_data}

        Analyze the data and provide:
        1. Description of discovered patterns
        2. Pattern frequency and distribution
        3. Significance and implications of each pattern
        4. Potential causes or explanations for patterns
        5. Actionable recommendations based on patterns
        6. Anomalies or outliers that don't fit the patterns

        Focus on patterns that have business or operational significance.
        Explain the methodology used to identify these patterns.
        """

    @staticmethod
    def contextual_search_prompt() -> str:
        """Prompt template for contextual search analysis"""
        return """
        Analyze the contextual search results and provide relevant insights:

        Search Context:
        - Query Intent: {query_intent}
        - Search Domain: {search_domain}
        - Similarity Threshold: {similarity_threshold}

        Search Results:
        {search_results}

        Provide analysis including:
        1. Relevance of search results to the query intent
        2. Key insights from the most relevant results
        3. Contextual connections between results
        4. Additional context that might be relevant
        5. Recommendations for further exploration
        6. Quality assessment of the search results

        Prioritize results with high semantic similarity and rich contextual information.
        Explain how the contextual information enhances understanding.
        """

    @staticmethod
    def summarization_prompt() -> str:
        """Prompt template for data summarization"""
        return """
        Create a comprehensive summary of the following graph data:

        Summarization Parameters:
        - Data Scope: {data_scope}
        - Summary Level: {summary_level}
        - Target Audience: {target_audience}

        Data to Summarize:
        {summarization_data}

        Create a summary that includes:
        1. Executive Overview
        2. Key Entities and Their Roles
        3. Important Relationships and Connections
        4. Quantitative Insights and Metrics
        5. Notable Trends or Patterns
        6. Significant Events or Changes
        7. Recommendations or Next Steps

        Tailor the summary to the specified audience level.
        Focus on the most important and actionable information.
        Use clear, concise language while maintaining technical accuracy.
        """

    @staticmethod
    def recommendation_prompt() -> str:
        """Prompt template for generating recommendations"""
        return """
        Generate actionable recommendations based on the following graph analysis:

        Analysis Context:
        - Business Objective: {business_objective}
        - Decision Context: {decision_context}
        - Constraints: {constraints}

        Analysis Results:
        {analysis_results}

        Provide recommendations that include:
        1. Primary recommendations with clear rationale
        2. Alternative options and trade-offs
        3. Risk assessment for each recommendation
        4. Implementation considerations
        5. Expected outcomes and success metrics
        6. Timeline and resource requirements
        7. Monitoring and evaluation approaches

        Prioritize recommendations by potential impact and feasibility.
        Consider both short-term and long-term implications.
        Ensure recommendations are specific, measurable, and actionable.
        """

    @staticmethod
    def anomaly_detection_prompt() -> str:
        """Prompt template for anomaly detection analysis"""
        return """
        Analyze the following data for anomalies and unusual patterns:

        Anomaly Detection Parameters:
        - Baseline Period: {baseline_period}
        - Detection Sensitivity: {detection_sensitivity}
        - Anomaly Types: {anomaly_types}

        Data for Analysis:
        {anomaly_data}

        Identify and analyze:
        1. Detected anomalies and their characteristics
        2. Severity and significance of each anomaly
        3. Potential causes or explanations
        4. Impact assessment on related entities
        5. Recommended actions for each anomaly
        6. Prevention strategies for future occurrences
        7. Monitoring recommendations

        Classify anomalies by type, severity, and urgency.
        Provide confidence levels for anomaly detection.
        Focus on actionable insights and clear next steps.
        """

    @staticmethod
    def trend_analysis_prompt() -> str:
        """Prompt template for trend analysis"""
        return """
        Analyze trends and temporal patterns in the following data:

        Trend Analysis Parameters:
        - Time Period: {time_period}
        - Trend Types: {trend_types}
        - Granularity: {granularity}

        Trend Data:
        {trend_data}

        Provide analysis including:
        1. Identified trends and their direction
        2. Trend strength and statistical significance
        3. Seasonal or cyclical patterns
        4. Trend drivers and contributing factors
        5. Trend projections and forecasts
        6. Potential inflection points or changes
        7. Business implications and recommendations

        Use appropriate statistical measures to validate trends.
        Consider external factors that might influence trends.
        Provide actionable insights based on trend analysis.
        """


class ResultFormattingTemplates:
    """Templates for formatting query results"""

    @staticmethod
    def standard_result_format() -> Dict[str, Any]:
        """Standard result format template"""
        return {
            "query_metadata": {
                "query_id": "{query_id}",
                "timestamp": "{timestamp}",
                "execution_time": "{execution_time}",
                "query_type": "{query_type}"
            },
            "data": {
                "primary_results": [],
                "secondary_results": [],
                "aggregations": {},
                "statistics": {}
            },
            "analysis": {
                "summary": "",
                "insights": [],
                "patterns": [],
                "recommendations": []
            },
            "metadata": {
                "result_count": 0,
                "confidence_score": 0.0,
                "data_quality": {},
                "processing_notes": []
            }
        }

    @staticmethod
    def relationship_result_format() -> Dict[str, Any]:
        """Relationship analysis result format template"""
        return {
            "relationship_analysis": {
                "source_entity": {},
                "target_entity": {},
                "relationship_path": [],
                "path_strength": 0.0,
                "path_length": 0,
                "intermediate_entities": []
            },
            "analysis_details": {
                "relationship_types": [],
                "connection_strength": {},
                "significance_scores": {},
                "context_information": {}
            },
            "insights": {
                "key_findings": [],
                "relationship_patterns": [],
                "anomalies": [],
                "recommendations": []
            }
        }

    @staticmethod
    def pattern_result_format() -> Dict[str, Any]:
        """Pattern discovery result format template"""
        return {
            "pattern_discovery": {
                "discovered_patterns": [],
                "pattern_frequency": {},
                "pattern_significance": {},
                "pattern_examples": []
            },
            "statistical_analysis": {
                "distribution_metrics": {},
                "correlation_analysis": {},
                "outlier_detection": {},
                "confidence_intervals": {}
            },
            "business_insights": {
                "actionable_patterns": [],
                "trend_implications": [],
                "risk_indicators": [],
                "opportunity_identification": []
            }
        }

    @staticmethod
    def contextual_result_format() -> Dict[str, Any]:
        """Contextual search result format template"""
        return {
            "search_results": {
                "primary_matches": [],
                "contextual_matches": [],
                "semantic_similarity": {},
                "relevance_scores": {}
            },
            "context_analysis": {
                "context_graph": {},
                "entity_relationships": {},
                "temporal_context": {},
                "spatial_context": {}
            },
            "enriched_insights": {
                "comprehensive_view": {},
                "cross_references": [],
                "related_topics": [],
                "suggested_explorations": []
            }
        }


class QueryChainTemplates:
    """Templates for multi-step query chains"""

    @staticmethod
    def exploration_chain() -> List[Dict[str, Any]]:
        """Exploration query chain template"""
        return [
            {
                "step": 1,
                "name": "initial_discovery",
                "description": "Discover primary entities of interest",
                "query_template": CypherQueryTemplates.knowledge_extraction_basic(),
                "prompt_template": PromptTemplates.knowledge_extraction_prompt(),
                "output_format": "entities_list"
            },
            {
                "step": 2,
                "name": "relationship_mapping",
                "description": "Map relationships between discovered entities",
                "query_template": CypherQueryTemplates.relationship_analysis_deep(),
                "prompt_template": PromptTemplates.relationship_analysis_prompt(),
                "output_format": "relationship_graph",
                "depends_on": ["initial_discovery"]
            },
            {
                "step": 3,
                "name": "pattern_identification",
                "description": "Identify patterns in the relationship graph",
                "query_template": CypherQueryTemplates.pattern_discovery_clustering(),
                "prompt_template": PromptTemplates.pattern_discovery_prompt(),
                "output_format": "pattern_analysis",
                "depends_on": ["relationship_mapping"]
            },
            {
                "step": 4,
                "name": "insight_synthesis",
                "description": "Synthesize insights from all previous steps",
                "query_template": "RETURN 'synthesis' as step",
                "prompt_template": PromptTemplates.summarization_prompt(),
                "output_format": "comprehensive_insights",
                "depends_on": ["initial_discovery", "relationship_mapping", "pattern_identification"]
            }
        ]

    @staticmethod
    def investigation_chain() -> List[Dict[str, Any]]:
        """Investigation query chain template"""
        return [
            {
                "step": 1,
                "name": "anomaly_detection",
                "description": "Detect anomalies in the data",
                "query_template": CypherQueryTemplates.performance_optimized_batch(),
                "prompt_template": PromptTemplates.anomaly_detection_prompt(),
                "output_format": "anomaly_report"
            },
            {
                "step": 2,
                "name": "context_gathering",
                "description": "Gather context around detected anomalies",
                "query_template": CypherQueryTemplates.contextual_search_semantic(),
                "prompt_template": PromptTemplates.contextual_search_prompt(),
                "output_format": "context_analysis",
                "depends_on": ["anomaly_detection"]
            },
            {
                "step": 3,
                "name": "root_cause_analysis",
                "description": "Analyze potential root causes",
                "query_template": CypherQueryTemplates.multi_step_analysis(),
                "prompt_template": PromptTemplates.relationship_analysis_prompt(),
                "output_format": "root_cause_report",
                "depends_on": ["context_gathering"]
            },
            {
                "step": 4,
                "name": "recommendation_generation",
                "description": "Generate actionable recommendations",
                "query_template": "RETURN 'recommendations' as step",
                "prompt_template": PromptTemplates.recommendation_prompt(),
                "output_format": "action_plan",
                "depends_on": ["root_cause_analysis"]
            }
        ]


class PerformanceOptimizations:
    """Performance optimization templates and hints"""

    @staticmethod
    def get_performance_hints() -> Dict[str, List[str]]:
        """Get performance optimization hints for different query types"""
        return {
            "general": [
                "Use LIMIT to restrict result sets",
                "Apply WHERE clauses as early as possible",
                "Use index hints with USING INDEX when appropriate",
                "Avoid unnecessary OPTIONAL MATCH when MATCH suffices",
                "Use WITH to pass only necessary data between query parts"
            ],
            "relationship_traversal": [
                "Limit traversal depth with specific ranges [*1..3]",
                "Use relationship type filters to reduce search space",
                "Consider using shortest path functions for path queries",
                "Apply node filters before relationship traversal",
                "Use variable length relationship patterns judiciously"
            ],
            "aggregation": [
                "Use DISTINCT judiciously to avoid unnecessary deduplication",
                "Consider using collect() with size limits",
                "Apply filters before aggregation operations",
                "Use appropriate aggregation functions (count, sum, avg)",
                "Consider using apoc functions for complex aggregations"
            ],
            "vector_operations": [
                "Set appropriate similarity thresholds",
                "Limit vector query results with topK parameter",
                "Consider using vector indexes for large datasets",
                "Batch vector operations when possible",
                "Use appropriate distance metrics for your use case"
            ],
            "batch_processing": [
                "Process data in appropriately sized batches",
                "Use UNWIND for parameterized batch operations",
                "Consider using apoc.periodic.iterate for large datasets",
                "Implement proper error handling in batch operations",
                "Monitor memory usage during batch processing"
            ]
        }

    @staticmethod
    def get_query_optimization_checklist() -> List[str]:
        """Get query optimization checklist"""
        return [
            "✓ Index usage verified for key properties",
            "✓ Query complexity assessed and limited",
            "✓ Result set size controlled with LIMIT",
            "✓ Filters applied as early as possible",
            "✓ Unnecessary data excluded from results",
            "✓ Relationship traversal depth limited",
            "✓ Memory usage estimated and acceptable",
            "✓ Query plan reviewed with EXPLAIN/PROFILE",
            "✓ Error handling implemented",
            "✓ Performance tested with realistic data volumes"
        ]


class DocumentationTemplates:
    """Documentation and example templates"""

    @staticmethod
    def get_usage_examples() -> Dict[str, str]:
        """Get usage examples for different query types"""
        return {
            "knowledge_extraction": """
# Knowledge Extraction Example

query_template = CypherQueryTemplates.knowledge_extraction_basic()
prompt_template = PromptTemplates.knowledge_extraction_prompt()

parameters = {
    'node_label': 'Person',
    'filter_conditions': 'n.age > 25 AND n.department = "Engineering"',
    'limit': 10
}

prompt_params = {
    'primary_entity': 'Engineering Team Members',
    'entity_type': 'Person',
    'analysis_focus': 'Team composition and skills'
}
            """,
            
            "relationship_analysis": """
# Relationship Analysis Example

query_template = CypherQueryTemplates.relationship_analysis_deep()
prompt_template = PromptTemplates.relationship_analysis_prompt()

parameters = {
    'start_label': 'Company',
    'end_label': 'Technology',
    'start_conditions': 'start.industry = "Healthcare"',
    'end_conditions': 'end.category = "AI"',
    'max_depth': 3,
    'min_depth': 1,
    'limit': 20
}
            """,
            
            "pattern_discovery": """
# Pattern Discovery Example

query_template = CypherQueryTemplates.pattern_discovery_clustering()
prompt_template = PromptTemplates.pattern_discovery_prompt()

parameters = {
    'node_label': 'Transaction',
    'node_conditions': 'n.amount > 1000 AND n.date > date("2024-01-01")',
    'relationship_type': 'INVOLVES',
    'min_pattern_frequency': 5,
    'sample_size': 3,
    'limit': 10
}
            """
        }

    @staticmethod
    def get_integration_guide() -> str:
        """Get integration guide documentation"""
        return """
# LLM Query Templates Integration Guide

## Overview
This module provides comprehensive templates for integrating LLM capabilities
with Neo4j graph queries. It supports various analysis types and provides
optimized query patterns.

## Basic Usage

1. **Import the required templates:**
```python
from neo4j_enhancements.queries.templates.llm_query_templates import (
    CypherQueryTemplates, 
    PromptTemplates,
    QueryType,
    AnalysisType
)
```

2. **Get a query template:**
```python
query_template = CypherQueryTemplates.knowledge_extraction_basic()
```

3. **Format with parameters:**
```python
formatted_query = query_template.format(**your_parameters)
```

4. **Get corresponding prompt:**
```python
prompt_template = PromptTemplates.knowledge_extraction_prompt()
formatted_prompt = prompt_template.format(**your_prompt_parameters)
```

## Advanced Features

### Multi-Step Query Chains
Use QueryChainTemplates for complex analysis workflows:

```python
chain = QueryChainTemplates.exploration_chain()
for step in chain:
    # Execute each step in sequence
    result = execute_query_step(step)
```

### Performance Optimization
Apply performance hints from PerformanceOptimizations:

```python
hints = PerformanceOptimizations.get_performance_hints()
checklist = PerformanceOptimizations.get_query_optimization_checklist()
```

### Error Handling
Use the error handling template for robust queries:

```python
error_safe_query = CypherQueryTemplates.error_handling_template()
```

## Best Practices

1. Always validate input parameters
2. Use appropriate LIMIT clauses
3. Apply WHERE filters early in queries
4. Consider index usage for performance
5. Implement proper error handling
6. Test with realistic data volumes
7. Monitor query performance
8. Use semantic versioning for template updates

## Template Customization

Templates can be extended or customized for specific use cases:

```python
class CustomQueryTemplates(CypherQueryTemplates):
    @staticmethod
    def domain_specific_query():
        return '''
        // Your custom query template here
        MATCH (n:YourDomain)
        WHERE your_conditions
        RETURN your_results
        '''
```

## Troubleshooting

Common issues and solutions:
- Query timeout: Reduce LIMIT or add more specific filters
- Memory issues: Process data in smaller batches
- Performance issues: Check index usage and query plan
- Result formatting: Use appropriate result format templates
"""

    @staticmethod
    def get_api_reference() -> str:
        """Get API reference documentation"""
        return """
# API Reference

## Classes

### CypherQueryTemplates
Static methods providing Cypher query templates for various use cases.

Methods:
- `knowledge_extraction_basic()` - Basic entity and relationship extraction
- `relationship_analysis_deep()` - Deep relationship path analysis
- `pattern_discovery_clustering()` - Pattern identification with clustering
- `contextual_search_semantic()` - Semantic search with context expansion
- `aggregation_summary()` - Data aggregation and summarization
- `multi_step_analysis()` - Multi-step analysis chains
- `performance_optimized_batch()` - Batch processing optimization
- `error_handling_template()` - Error-safe query template

### PromptTemplates
Static methods providing LLM prompt templates for different analysis types.

Methods:
- `knowledge_extraction_prompt()` - Knowledge extraction analysis
- `relationship_analysis_prompt()` - Relationship pattern analysis
- `pattern_discovery_prompt()` - Pattern identification analysis
- `contextual_search_prompt()` - Contextual search analysis
- `summarization_prompt()` - Data summarization
- `recommendation_prompt()` - Recommendation generation
- `anomaly_detection_prompt()` - Anomaly detection analysis
- `trend_analysis_prompt()` - Trend and temporal analysis

### ResultFormattingTemplates
Templates for standardizing query result formats.

Methods:
- `standard_result_format()` - Standard result structure
- `relationship_result_format()` - Relationship analysis results
- `pattern_result_format()` - Pattern discovery results
- `contextual_result_format()` - Contextual search results

### QueryChainTemplates
Templates for multi-step query workflows.

Methods:
- `exploration_chain()` - Data exploration workflow
- `investigation_chain()` - Investigation and analysis workflow

### PerformanceOptimizations
Performance optimization utilities and guidelines.

Methods:
- `get_performance_hints()` - Performance optimization hints
- `get_query_optimization_checklist()` - Optimization checklist

## Enums

### QueryType
Enumeration of different query types for classification.

### AnalysisType
Enumeration of different analysis approaches.

## Data Classes

### QueryTemplate
Structure for defining complete query templates with metadata.

Attributes:
- `name` - Template name
- `description` - Template description
- `cypher_template` - Cypher query template
- `prompt_template` - LLM prompt template
- `parameters` - Required parameters
- `expected_output` - Expected output format
- `use_case` - Primary use case
- `performance_hints` - Performance optimization hints
"""


# Example usage and testing
if __name__ == "__main__":
    # Example of using the templates
    print("LLM Query Templates Module")
    print("=" * 40)
    
    # Get a knowledge extraction query
    query = CypherQueryTemplates.knowledge_extraction_basic()
    print("Knowledge Extraction Query Template:")
    print(query[:200] + "...")
    
    # Get corresponding prompt
    prompt = PromptTemplates.knowledge_extraction_prompt()
    print("\nCorresponding Prompt Template:")
    print(prompt[:200] + "...")
    
    # Get performance hints
    hints = PerformanceOptimizations.get_performance_hints()
    print(f"\nPerformance Hints Available: {list(hints.keys())}")
    
    # Get usage examples
    examples = DocumentationTemplates.get_usage_examples()
    print(f"\nUsage Examples Available: {list(examples.keys())}")
    
    print("\nModule loaded successfully!")