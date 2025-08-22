# EHS Analytics Retrieval Orchestrator

The Retrieval Orchestrator is an intelligent coordination system that selects and combines multiple retrieval strategies to provide optimal results for EHS Analytics queries.

## Overview

The orchestrator consists of three main components:

1. **Strategy Selector** - Analyzes queries to determine optimal retrieval strategies
2. **Result Merger** - Combines and deduplicates results from multiple retrievers  
3. **Orchestrator** - Coordinates the entire retrieval process with performance optimization

## Architecture

```
Query Input
    ↓
Strategy Selector (analyzes query characteristics)
    ↓
Orchestrator (executes selected strategies)
    ↓
Result Merger (deduplicates and ranks results)
    ↓
Merged Results
```

## Key Features

### Intelligent Strategy Selection

The orchestrator automatically routes queries to the most appropriate retriever(s):

- **Simple lookups** → Text2Cypher (direct database queries)
- **Document search** → Vector or Hybrid (semantic similarity)
- **Relationship queries** → VectorCypher (graph traversal + similarity)
- **Temporal analysis** → HybridCypher (time-series + graph analysis)

### Execution Modes

- **Single**: Use the best single retriever
- **Parallel**: Execute multiple retrievers simultaneously  
- **Sequential**: Try retrievers in order with fallback
- **Adaptive**: Automatically choose optimal mode based on query

### Result Optimization

- **Deduplication**: Remove duplicate results across strategies
- **Score Normalization**: Normalize confidence scores between strategies
- **Intelligent Ranking**: Combine multiple ranking factors
- **Source Tracking**: Maintain transparency about result origins

## Usage

### Basic Usage

```python
from src.ehs_analytics.retrieval import (
    create_ehs_retrieval_orchestrator,
    OrchestrationConfig,
    OrchestrationMode,
    RetrievalStrategy,
    QueryType
)

# Configure retrieval strategies
configs = {
    RetrievalStrategy.TEXT2CYPHER: {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "llm_provider": "openai"
    },
    RetrievalStrategy.VECTOR: {
        "vector_store_type": "neo4j",
        "embedding_model": "openai"
    },
    RetrievalStrategy.HYBRID: {
        # Combined configuration
    }
}

# Create orchestrator
orchestrator = await create_ehs_retrieval_orchestrator(
    configs=configs,
    orchestration_config=OrchestrationConfig(
        max_strategies=3,
        enable_parallel_execution=True
    )
)

# Execute query
result = await orchestrator.retrieve(
    query="Show water consumption trends for the last 6 months",
    query_type=QueryType.CONSUMPTION,
    mode=OrchestrationMode.ADAPTIVE,
    limit=10
)

print(f"Found {len(result.data)} results")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Strategies used: {result.source_strategies}")
```

### Advanced Configuration

```python
from src.ehs_analytics.retrieval import (
    OrchestrationConfig,
    MergerConfig,
    DeduplicationMethod,
    RankingMethod
)

# Advanced orchestration configuration
config = OrchestrationConfig(
    max_strategies=3,
    min_confidence_threshold=0.7,
    enable_parallel_execution=True,
    enable_caching=True,
    cache_ttl_seconds=300,
    strategy_bias_weights={
        RetrievalStrategy.HYBRID_CYPHER: 1.3,
        RetrievalStrategy.HYBRID: 1.2,
        RetrievalStrategy.VECTOR_CYPHER: 1.1,
        RetrievalStrategy.TEXT2CYPHER: 1.0,
        RetrievalStrategy.VECTOR: 0.8
    }
)
```

## Strategy Selection Logic

### Rule-Based Selection

The orchestrator uses pattern matching to identify query types:

```python
# Simple lookup patterns
r'\b(what is|show me|list|find|get)\b'
r'\bstatus of\b'
→ Prefers TEXT2CYPHER

# Temporal analysis patterns  
r'\b(trend|over time|pattern|change)\b'
r'\b(last|past|previous|since|until)\b'
→ Prefers HYBRID_CYPHER

# Relationship patterns
r'\b(connect|relate|associate|link)\b'
r'\b(between|and|with|from|to)\b'
→ Prefers VECTOR_CYPHER
```

### Query Characteristics Analysis

The selector analyzes multiple query characteristics:

- **Complexity Score**: Based on word count, sentence structure, complex terms
- **Domain Terms**: EHS-specific vocabulary (facilities, equipment, utilities)
- **Intent Signals**: Lookup, analytical, comparative, temporal, predictive
- **Entity References**: Named entities and specific IDs

### Performance-Based Adaptation

The orchestrator learns from execution history:

- **Success Rates**: Track which strategies work best for query types
- **Response Times**: Optimize for performance requirements
- **Result Quality**: Learn from user feedback and result effectiveness

## Result Merging

### Deduplication Methods

1. **Content Hash**: Remove identical content based on hash comparison
2. **Entity ID**: Deduplicate based on unique entity identifiers
3. **Similarity Threshold**: Remove results above similarity threshold
4. **Semantic Similarity**: Advanced semantic deduplication (future)

### Ranking Methods

1. **Score-Based**: Rank by normalized confidence scores
2. **Strategy-Weighted**: Apply strategy-specific weights
3. **Consensus-Based**: Boost results found by multiple strategies
4. **Relevance-Optimized**: Consider query-result relevance
5. **Hybrid**: Combine multiple ranking factors

### Score Normalization

Results from different strategies are normalized using:

- **Min-Max Normalization**: Scale scores to 0-1 range
- **Z-Score Normalization**: Standardize based on mean and standard deviation
- **Softmax**: Apply softmax normalization per strategy

## Performance Monitoring

### Metrics Tracking

```python
# Get performance metrics
metrics = await orchestrator.get_performance_metrics()

print(f"Total operations: {metrics['total_operations']}")
print(f"Average execution time: {metrics['recent_performance']['avg_execution_time_ms']:.2f}ms")
print(f"Cache hit rate: {metrics['recent_performance']['cache_hit_rate']:.1%}")

# Strategy-specific performance
for strategy, perf in metrics['strategy_performance'].items():
    print(f"{strategy}: success={perf['success_rate']:.1%}, time={perf['avg_response_time']:.0f}ms")
```

### Health Monitoring

```python
# Check system health
health = await orchestrator.health_check()
print(f"Orchestrator status: {health['orchestrator']['status']}")

for strategy, status in health['retrievers'].items():
    print(f"{strategy}: {status['status']}")
```

## Configuration Options

### OrchestrationConfig

- `max_strategies`: Maximum number of strategies to use (default: 3)
- `min_confidence_threshold`: Minimum confidence for strategy selection (default: 0.7)
- `enable_parallel_execution`: Enable parallel retriever execution (default: True)
- `enable_caching`: Enable query result caching (default: True)
- `cache_ttl_seconds`: Cache time-to-live (default: 300)
- `enable_adaptive_selection`: Enable adaptive strategy selection (default: True)

### MergerConfig

- `deduplication_method`: Method for removing duplicates
- `similarity_threshold`: Threshold for similarity-based deduplication (default: 0.85)
- `ranking_method`: Method for ranking merged results
- `strategy_weights`: Weights for different strategies
- `max_results_per_strategy`: Limit results per strategy (default: 20)

## Best Practices

### Query Design

1. **Be Specific**: Include relevant entity names and types
2. **Use EHS Terminology**: Leverage domain-specific vocabulary
3. **Specify Time Ranges**: Include temporal context when relevant
4. **Indicate Intent**: Use clear action words (analyze, find, show, compare)

### Performance Optimization

1. **Set Appropriate Limits**: Don't request more results than needed
2. **Use Caching**: Enable caching for repeated queries
3. **Monitor Performance**: Track metrics and adjust configuration
4. **Strategy Selection**: Use single mode for simple queries

### Error Handling

```python
try:
    result = await orchestrator.retrieve(query, query_type)
    if result.confidence_score < 0.5:
        print("Low confidence results, consider refining query")
except Exception as e:
    print(f"Retrieval failed: {e}")
    # Implement fallback logic
```

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI
from src.ehs_analytics.retrieval import create_ehs_retrieval_orchestrator

app = FastAPI()
orchestrator = None

@app.on_event("startup")
async def startup():
    global orchestrator
    orchestrator = await create_ehs_retrieval_orchestrator(configs)

@app.post("/query")
async def query_endpoint(query: str, query_type: str):
    result = await orchestrator.retrieve(
        query=query,
        query_type=QueryType(query_type),
        mode=OrchestrationMode.ADAPTIVE
    )
    return {
        "results": result.data,
        "confidence": result.confidence_score,
        "strategies": [s.value for s in result.source_strategies]
    }
```

### With LangGraph

```python
from langgraph import StateGraph
from src.ehs_analytics.retrieval import RetrievalOrchestrator

class QueryState:
    query: str
    results: list
    confidence: float

def retrieval_node(state: QueryState) -> QueryState:
    result = await orchestrator.retrieve(state.query)
    state.results = result.data
    state.confidence = result.confidence_score
    return state

# Add to LangGraph workflow
graph = StateGraph(QueryState)
graph.add_node("retrieval", retrieval_node)
```

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - Check query specificity and EHS terminology
   - Verify strategy configuration
   - Review available data in retrievers

2. **Slow Performance**
   - Reduce max_strategies limit
   - Enable caching
   - Use single mode for simple queries
   - Check individual retriever performance

3. **No Results**
   - Verify retriever initialization
   - Check database connectivity
   - Review query patterns and strategy rules

### Debug Mode

```python
import logging
logging.getLogger('ehs_analytics.retrieval').setLevel(logging.DEBUG)

# Enable detailed logging for troubleshooting
result = await orchestrator.retrieve(query, query_type)
```

## Future Enhancements

- **ML-Based Selection**: Machine learning models for strategy selection
- **Semantic Deduplication**: Advanced semantic similarity for deduplication
- **Query Expansion**: Automatic query expansion for better results
- **A/B Testing**: Built-in A/B testing for strategy optimization
- **Real-time Learning**: Continuous learning from user feedback