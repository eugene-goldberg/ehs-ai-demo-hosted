"""
VectorCypher retriever implementation for relationship-aware vector search.

This module provides a retriever that combines vector similarity search with
graph traversal to find contextually relevant EHS documents through relationships.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from neo4j_graphrag.retrievers import VectorCypherRetriever as GraphRAGVectorCypherRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j import GraphDatabase, Driver
import numpy as np

from ..base import BaseRetriever, RetrievalResult, RetrievalMetadata, RetrievalStrategy
from ..config import RetrieverConfig
from .ehs_vector_config import EHSVectorConfig, DocumentType, FacilityType
from ..base import QueryType
from .vector_cypher_config import (
    VectorCypherConfig, VectorCypherConfigManager, RelationshipType, TraversalDepth
)
from .graph_patterns import (
    EHSPatternLibrary, GraphPath, PathScorer, ContextAggregator, TraversalPattern
)
from ..embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class VectorCypherResult:
    """Result from vector-cypher search combining similarity and relationships."""
    content: str
    vector_score: float
    relationship_score: float
    combined_score: float
    metadata: Dict[str, Any]
    document_id: str
    document_type: DocumentType
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    traversal_paths: List[GraphPath] = field(default_factory=list)
    context_expansion: Optional[str] = None


@dataclass
class VectorCypherSearchMetrics:
    """Metrics for vector-cypher search operation."""
    vector_search_time: float
    graph_traversal_time: float
    context_aggregation_time: float
    total_search_time: float
    vector_results_count: int
    graph_paths_count: int
    final_results_count: int
    cache_hits: int = 0


class EHSVectorCypherRetriever(BaseRetriever):
    """
    Retriever that combines vector similarity with graph relationship traversal.
    
    This retriever performs vector search to find semantically similar documents,
    then uses graph traversal to expand context through relationships and find
    additional relevant content that may not be semantically similar but is
    contextually connected.
    """
    
    def __init__(self, 
                 neo4j_driver: Driver,
                 embedding_manager: EmbeddingManager,
                 config: Optional[VectorCypherConfig] = None,
                 ehs_config: Optional[EHSVectorConfig] = None):
        """Initialize the VectorCypher retriever."""
        # Initialize with config dict as expected by base class
        config_dict = {
            "vector_cypher_config": config or VectorCypherConfig(),
            "ehs_config": ehs_config or EHSVectorConfig()
        }
        super().__init__(config_dict)
        
        self.driver = neo4j_driver
        self.embedding_manager = embedding_manager
        self.vector_cypher_config = config or VectorCypherConfig()
        self.ehs_config = ehs_config or EHSVectorConfig()
        
        # Initialize configuration manager
        self.config_manager = VectorCypherConfigManager(self.vector_cypher_config)
        
        # Initialize pattern library and scoring components
        self.pattern_library = EHSPatternLibrary()
        self.path_scorer = self.pattern_library.create_path_scorer(
            self.vector_cypher_config.path_scoring_config.relationship_weights
        )
        self.context_aggregator = self.pattern_library.create_context_aggregator("balanced")
        
        # Cache for search results
        self._result_cache = {}
        
    async def initialize(self) -> None:
        """Initialize the VectorCypher retriever."""
        try:
            # Setup the GraphRAG VectorCypher retriever
            await self._setup_graphrag_retriever()
            self._initialized = True
            logger.info("EHS VectorCypher retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VectorCypher retriever: {e}")
            raise
    
    async def _setup_graphrag_retriever(self):
        """Setup the neo4j-graphrag-python VectorCypher retriever."""
        try:
            # Create OpenAI embeddings instance
            embeddings = OpenAIEmbeddings(
                model=self.ehs_config.embedding_model,
                api_key=self.embedding_manager.openai_client.api_key
            )
            
            # Initialize the GraphRAG VectorCypher retriever
            self.graphrag_retriever = GraphRAGVectorCypherRetriever(
                driver=self.driver,
                index_name="document_embeddings",  # Assuming this index exists
                embedder=embeddings,
                return_properties=["content", "document_type", "created_date", "facility_id"],
                retrieval_query=self._get_base_retrieval_query()
            )
            
        except Exception as e:
            logger.warning(f"Failed to initialize GraphRAG VectorCypher retriever: {e}")
            self.graphrag_retriever = None
    
    def _get_base_retrieval_query(self) -> str:
        """Get the base Cypher query for retrieval."""
        return """
        MATCH (doc:Document)
        WHERE doc.embedding IS NOT NULL
        WITH doc, score
        
        // Expand to related entities
        OPTIONAL MATCH (doc)-[r1:DOCUMENTS]->(entity)
        OPTIONAL MATCH (entity)-[r2:LOCATED_AT]->(facility:Facility)
        OPTIONAL MATCH (doc)-[r3:CREATED_FOR]->(event)
        
        RETURN 
            doc.content as content,
            doc.document_id as document_id,
            doc.document_type as document_type,
            doc.created_date as created_date,
            doc.facility_id as facility_id,
            score,
            collect(DISTINCT {
                type: type(r1),
                entity: entity.name,
                entity_type: labels(entity)[0]
            }) as direct_relations,
            collect(DISTINCT {
                type: type(r2), 
                facility: facility.name,
                facility_type: facility.facility_type
            }) as facility_relations,
            collect(DISTINCT {
                type: type(r3),
                event: event.event_id,
                event_type: labels(event)[0]
            }) as event_relations
        """
    
    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        limit: int = 10,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve documents using vector similarity and graph relationships.
        
        Args:
            query: The search query
            query_type: Type of query for configuration optimization
            limit: Maximum number of results to return
            **kwargs: Additional parameters (facility_filter, document_types, etc.)
            
        Returns:
            RetrievalResult containing data, metadata, and execution information
        """
        start_time = datetime.now()
        
        # Extract parameters from kwargs
        facility_filter = kwargs.get('facility_filter')
        document_types = kwargs.get('document_types')
        
        # Get query-optimized configuration
        query_config = self.config_manager.get_config_for_query(query_type)
        
        # Check cache if enabled
        cache_key = self._get_cache_key(query, query_type, facility_filter, document_types)
        if query_config.enable_result_caching and cache_key in self._result_cache:
            cached_result, cache_time = self._result_cache[cache_key]
            if (datetime.now() - cache_time).seconds < query_config.cache_ttl_seconds:
                logger.debug(f"Returning cached results for query: {query[:50]}...")
                return cached_result
        
        try:
            # Step 1: Vector similarity search
            vector_start = datetime.now()
            vector_results = await self._vector_search(
                query, query_config, facility_filter, document_types
            )
            vector_time = (datetime.now() - vector_start).total_seconds()
            
            # Step 2: Graph traversal and relationship expansion
            graph_start = datetime.now()
            expanded_results = await self._expand_with_relationships(
                vector_results, query, query_type, query_config
            )
            graph_time = (datetime.now() - graph_start).total_seconds()
            
            # Step 3: Context aggregation and final formatting
            context_start = datetime.now()
            final_data = await self._aggregate_context(
                expanded_results, query, query_config, limit
            )
            context_time = (datetime.now() - context_start).total_seconds()
            
            # Create metrics
            total_time = (datetime.now() - start_time).total_seconds()
            metrics = VectorCypherSearchMetrics(
                vector_search_time=vector_time,
                graph_traversal_time=graph_time,
                context_aggregation_time=context_time,
                total_search_time=total_time,
                vector_results_count=len(vector_results),
                graph_paths_count=sum(len(r.traversal_paths) for r in expanded_results),
                final_results_count=len(final_data)
            )
            
            # Create retrieval metadata
            retrieval_metadata = RetrievalMetadata(
                strategy=RetrievalStrategy.VECTOR_CYPHER,
                query_type=query_type,
                confidence_score=self._calculate_average_confidence(final_data),
                execution_time_ms=total_time * 1000,
                nodes_retrieved=sum(len(r.related_entities) for r in expanded_results),
                relationships_retrieved=sum(len(r.traversal_paths) for r in expanded_results)
            )
            
            # Create final result
            result = RetrievalResult(
                data=final_data,
                metadata=retrieval_metadata,
                success=True,
                message=f"Retrieved {len(final_data)} results using VectorCypher strategy"
            )
            
            # Cache results if enabled
            if query_config.enable_result_caching:
                self._result_cache[cache_key] = (result, datetime.now())
                
            # Log performance metrics
            logger.info(f"VectorCypher search completed: {len(final_data)} results in {total_time:.2f}s")
            logger.debug(f"Performance breakdown - Vector: {vector_time:.2f}s, Graph: {graph_time:.2f}s, Context: {context_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in VectorCypher retrieval: {e}", exc_info=True)
            # Return fallback result
            return await self._fallback_vector_search(query, query_type, limit, **kwargs)
    
    async def validate_query(self, query: str) -> bool:
        """
        Validate if the query can be processed by this retriever.
        
        Args:
            query: Natural language query to validate
            
        Returns:
            True if the query can be processed, False otherwise
        """
        try:
            # Basic validation checks
            if not query or len(query.strip()) < 3:
                return False
                
            # Check if embeddings are available
            if not self.embedding_manager:
                return False
                
            # Check if neo4j driver is available
            if not self.driver:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return False
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get the retrieval strategy identifier."""
        return RetrievalStrategy.VECTOR_CYPHER
    
    async def _vector_search(self, 
                            query: str, 
                            config: VectorCypherConfig,
                            facility_filter: Optional[str] = None,
                            document_types: Optional[List[DocumentType]] = None) -> List[VectorCypherResult]:
        """Perform vector similarity search."""
        try:
            if self.graphrag_retriever:
                # Use GraphRAG VectorCypher retriever
                search_results = self.graphrag_retriever.search(
                    query_text=query,
                    top_k=config.max_vector_results
                )
                
                results = []
                for result in search_results:
                    vector_result = VectorCypherResult(
                        content=result.content,
                        vector_score=result.score,
                        relationship_score=0.0,
                        combined_score=result.score * config.vector_weight,
                        metadata=result.metadata or {},
                        document_id=result.metadata.get('document_id', ''),
                        document_type=DocumentType(result.metadata.get('document_type', 'UNKNOWN')),
                        related_entities=[]
                    )
                    results.append(vector_result)
                
                return results
            else:
                # Fallback to manual vector search
                return await self._manual_vector_search(query, config, facility_filter, document_types)
                
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return await self._manual_vector_search(query, config, facility_filter, document_types)
    
    async def _manual_vector_search(self, 
                                   query: str, 
                                   config: VectorCypherConfig,
                                   facility_filter: Optional[str] = None,
                                   document_types: Optional[List[DocumentType]] = None) -> List[VectorCypherResult]:
        """Manual vector search implementation."""
        try:
            # Get query embedding
            query_embedding = await self.embedding_manager.get_embedding(query)
            
            # Build Cypher query for vector similarity
            cypher_query = self._build_vector_similarity_query(
                facility_filter, document_types, config.max_vector_results
            )
            
            # Execute search
            with self.driver.session() as session:
                result = session.run(cypher_query, {
                    "query_embedding": query_embedding,
                    "similarity_threshold": config.vector_similarity_threshold,
                    "facility_filter": facility_filter,
                    "document_types": [dt.value for dt in document_types] if document_types else None
                })
                
                vector_results = []
                for record in result:
                    vector_result = VectorCypherResult(
                        content=record["content"],
                        vector_score=record["similarity_score"],
                        relationship_score=0.0,
                        combined_score=record["similarity_score"] * config.vector_weight,
                        metadata={
                            "document_id": record["document_id"],
                            "document_type": record["document_type"],
                            "created_date": record.get("created_date"),
                            "facility_id": record.get("facility_id")
                        },
                        document_id=record["document_id"],
                        document_type=DocumentType(record["document_type"]),
                        related_entities=record.get("related_entities", [])
                    )
                    vector_results.append(vector_result)
                
                return vector_results
                
        except Exception as e:
            logger.error(f"Error in manual vector search: {e}")
            return []
    
    def _build_vector_similarity_query(self, 
                                      facility_filter: Optional[str],
                                      document_types: Optional[List[DocumentType]],
                                      max_results: int) -> str:
        """Build Cypher query for vector similarity search."""
        query_parts = [
            "MATCH (doc:Document)",
            "WHERE doc.embedding IS NOT NULL"
        ]
        
        # Add facility filter
        if facility_filter:
            query_parts.append("AND doc.facility_id = $facility_filter")
        
        # Add document type filter
        if document_types:
            query_parts.append("AND doc.document_type IN $document_types")
        
        # Calculate similarity and filter
        query_parts.extend([
            "WITH doc,",
            "gds.similarity.cosine(doc.embedding, $query_embedding) AS similarity_score",
            "WHERE similarity_score >= $similarity_threshold",
            "",
            "// Get related entities",
            "OPTIONAL MATCH (doc)-[:DOCUMENTS]->(entity)",
            "OPTIONAL MATCH (entity)-[:LOCATED_AT]->(facility:Facility)",
            "",
            "RETURN",
            "doc.content as content,",
            "doc.document_id as document_id,",
            "doc.document_type as document_type,",
            "doc.created_date as created_date,",
            "doc.facility_id as facility_id,",
            "similarity_score,",
            "collect(DISTINCT {",
            "  entity_id: entity.entity_id,",
            "  entity_type: labels(entity)[0],",
            "  entity_name: entity.name",
            "}) as related_entities",
            "",
            "ORDER BY similarity_score DESC",
            f"LIMIT {max_results}"
        ])
        
        return "\n".join(query_parts)
    
    async def _expand_with_relationships(self, 
                                        vector_results: List[VectorCypherResult],
                                        query: str,
                                        query_type: QueryType,
                                        config: VectorCypherConfig) -> List[VectorCypherResult]:
        """Expand vector results with relationship traversal."""
        if not vector_results:
            return vector_results
        
        try:
            # Get relevant traversal patterns for query type
            patterns = self.pattern_library.get_patterns_for_query_type(query_type)
            
            expanded_results = []
            
            for vector_result in vector_results:
                # Find traversal paths from this document
                paths = await self._find_traversal_paths(
                    vector_result.document_id, patterns, config
                )
                
                # Score and rank paths
                ranked_paths = self.path_scorer.rank_paths(
                    paths, {"temporal_focus": True, "query_type": query_type}
                )
                
                # Calculate relationship score based on best paths
                relationship_score = self._calculate_relationship_score(ranked_paths)
                
                # Update combined score
                combined_score = (
                    vector_result.vector_score * config.vector_weight +
                    relationship_score * config.graph_weight
                )
                
                # Create expanded result
                expanded_result = VectorCypherResult(
                    content=vector_result.content,
                    vector_score=vector_result.vector_score,
                    relationship_score=relationship_score,
                    combined_score=combined_score,
                    metadata=vector_result.metadata,
                    document_id=vector_result.document_id,
                    document_type=vector_result.document_type,
                    related_entities=vector_result.related_entities,
                    traversal_paths=ranked_paths[:10]  # Keep top 10 paths
                )
                
                expanded_results.append(expanded_result)
            
            # Sort by combined score
            expanded_results.sort(key=lambda r: r.combined_score, reverse=True)
            
            # Limit results
            return expanded_results[:config.max_total_results]
            
        except Exception as e:
            logger.error(f"Error in relationship expansion: {e}")
            return vector_results
    
    async def _find_traversal_paths(self, 
                                   document_id: str,
                                   patterns: List[TraversalPattern],
                                   config: VectorCypherConfig) -> List[GraphPath]:
        """Find traversal paths from a document using given patterns."""
        all_paths = []
        
        with self.driver.session() as session:
            for pattern in patterns:
                try:
                    # Execute pattern query
                    cypher_query = self._adapt_pattern_query(pattern, document_id)
                    result = session.run(cypher_query, {
                        "document_id": document_id,
                        "max_depth": pattern.max_depth
                    })
                    
                    # Parse results into paths
                    pattern_paths = self._parse_traversal_results(result, pattern)
                    all_paths.extend(pattern_paths)
                    
                except Exception as e:
                    logger.warning(f"Error executing pattern {pattern.name}: {e}")
                    continue
        
        return all_paths
    
    def _adapt_pattern_query(self, pattern: TraversalPattern, document_id: str) -> str:
        """Adapt a pattern query for a specific document."""
        # Start from the document
        base_query = f"""
        MATCH (doc:Document {{document_id: $document_id}})
        
        // Follow the pattern
        MATCH path = (doc){self._build_pattern_path(pattern)}
        
        // Calculate path score
        WITH path, nodes(path) as path_nodes, relationships(path) as path_rels
        
        RETURN 
            [node in path_nodes | coalesce(node.entity_id, node.document_id, node.id)] as node_ids,
            [rel in path_rels | type(rel)] as relationship_types,
            length(path) as path_length,
            1.0 / (length(path) + 1) as base_score
        
        ORDER BY base_score DESC
        LIMIT 20
        """
        
        return base_query
    
    def _build_pattern_path(self, pattern: TraversalPattern) -> str:
        """Build Cypher path expression from pattern."""
        if not pattern.relationship_sequence:
            return ""
        
        path_parts = []
        for i, rel_type in enumerate(pattern.relationship_sequence):
            if pattern.bidirectional:
                path_parts.append(f"-[:{rel_type.value}]-")
            else:
                path_parts.append(f"-[:{rel_type.value}]->")
            
            # Add node constraint if specified
            if i < len(pattern.end_node_types):
                node_type = pattern.end_node_types[i]
                path_parts.append(f"({node_type.lower()}:{node_type})")
            else:
                path_parts.append("()")
        
        return "".join(path_parts)
    
    def _parse_traversal_results(self, result, pattern: TraversalPattern) -> List[GraphPath]:
        """Parse Cypher results into GraphPath objects."""
        paths = []
        
        for record in result:
            try:
                node_ids = record["node_ids"]
                relationship_types = [RelationshipType(rt) for rt in record["relationship_types"]]
                base_score = record["base_score"]
                
                # Create scores for each step (simplified)
                scores = [base_score * (0.9 ** i) for i in range(len(relationship_types))]
                
                path = GraphPath(
                    nodes=node_ids,
                    relationships=relationship_types,
                    scores=scores,
                    total_score=base_score,
                    metadata={
                        "pattern": pattern.name,
                        "path_length": len(relationship_types)
                    }
                )
                
                paths.append(path)
                
            except Exception as e:
                logger.warning(f"Error parsing traversal result: {e}")
                continue
        
        return paths
    
    def _calculate_relationship_score(self, paths: List[GraphPath]) -> float:
        """Calculate overall relationship score from traversal paths."""
        if not paths:
            return 0.0
        
        # Use weighted average of top paths
        top_paths = paths[:5]  # Top 5 paths
        
        total_score = 0.0
        total_weight = 0.0
        
        for i, path in enumerate(top_paths):
            # Weight decreases for lower-ranked paths
            weight = 1.0 / (i + 1)
            total_score += path.total_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def _aggregate_context(self, 
                                expanded_results: List[VectorCypherResult],
                                query: str,
                                config: VectorCypherConfig,
                                limit: int) -> List[Dict[str, Any]]:
        """Aggregate context from expanded results and format for output."""
        final_data = []
        
        # Limit results
        results_to_process = expanded_results[:limit]
        
        for result in results_to_process:
            try:
                # Get context content if paths exist
                context_content = ""
                if result.traversal_paths and config.context_expansion.max_context_nodes > 0:
                    context_content = await self._get_context_content(
                        result.traversal_paths, config
                    )
                
                # Create final data entry
                data_entry = {
                    "content": result.content,
                    "score": result.combined_score,
                    "vector_score": result.vector_score,
                    "relationship_score": result.relationship_score,
                    "document_id": result.document_id,
                    "document_type": result.document_type.value,
                    "metadata": result.metadata,
                    "related_entities": result.related_entities,
                    "traversal_paths_count": len(result.traversal_paths),
                    "context_expansion": context_content if context_content else None,
                    "facility_id": result.metadata.get("facility_id"),
                    "created_date": result.metadata.get("created_date")
                }
                
                final_data.append(data_entry)
                
            except Exception as e:
                logger.warning(f"Error aggregating context for result {result.document_id}: {e}")
                continue
        
        return final_data
    
    async def _get_context_content(self, 
                                  paths: List[GraphPath],
                                  config: VectorCypherConfig) -> str:
        """Get aggregated context content from traversal paths."""
        try:
            # Collect node IDs from all paths
            node_ids = set()
            for path in paths:
                node_ids.update(path.nodes)
            
            # Limit context nodes
            if len(node_ids) > config.context_expansion.max_context_nodes:
                # Keep nodes from highest-scoring paths
                priority_nodes = set()
                for path in sorted(paths, key=lambda p: p.total_score, reverse=True):
                    priority_nodes.update(path.nodes)
                    if len(priority_nodes) >= config.context_expansion.max_context_nodes:
                        break
                node_ids = priority_nodes
            
            # Fetch content for nodes
            node_contents = {}
            node_metadata = {}
            
            with self.driver.session() as session:
                # Get content from different node types
                content_query = """
                MATCH (n)
                WHERE n.entity_id IN $node_ids OR n.document_id IN $node_ids
                RETURN 
                    coalesce(n.entity_id, n.document_id) as node_id,
                    coalesce(n.content, n.description, n.name) as content,
                    labels(n)[0] as node_type,
                    n.created_date as created_date
                """
                
                result = session.run(content_query, {"node_ids": list(node_ids)})
                
                for record in result:
                    node_id = record["node_id"]
                    content = record["content"]
                    if content:
                        node_contents[node_id] = content
                        node_metadata[node_id] = {
                            "node_type": record["node_type"],
                            "created_date": record["created_date"]
                        }
            
            # Use context aggregator to combine content
            aggregated_context = self.context_aggregator.aggregate_context(
                paths, node_contents, node_metadata
            )
            
            return aggregated_context
            
        except Exception as e:
            logger.error(f"Error getting context content: {e}")
            return ""
    
    async def _fallback_vector_search(self, 
                                     query: str, 
                                     query_type: QueryType,
                                     limit: int,
                                     **kwargs) -> RetrievalResult:
        """Fallback to vector search only."""
        try:
            config = self.vector_cypher_config
            facility_filter = kwargs.get('facility_filter')
            document_types = kwargs.get('document_types')
            
            vector_results = await self._manual_vector_search(
                query, config, facility_filter, document_types
            )
            
            # Convert to final data format
            final_data = []
            for result in vector_results[:limit]:
                data_entry = {
                    "content": result.content,
                    "score": result.vector_score,
                    "vector_score": result.vector_score,
                    "relationship_score": 0.0,
                    "document_id": result.document_id,
                    "document_type": result.document_type.value,
                    "metadata": result.metadata,
                    "fallback_mode": True
                }
                final_data.append(data_entry)
            
            # Create metadata
            retrieval_metadata = RetrievalMetadata(
                strategy=RetrievalStrategy.VECTOR,
                query_type=query_type,
                confidence_score=self._calculate_average_confidence(final_data),
                execution_time_ms=0.0,  # Not tracking time in fallback
                nodes_retrieved=len(final_data),
                relationships_retrieved=0
            )
            
            return RetrievalResult(
                data=final_data,
                metadata=retrieval_metadata,
                success=True,
                message=f"Fallback vector search returned {len(final_data)} results"
            )
            
        except Exception as e:
            logger.error(f"Error in fallback vector search: {e}")
            
            # Return empty result
            retrieval_metadata = RetrievalMetadata(
                strategy=RetrievalStrategy.VECTOR,
                query_type=query_type,
                confidence_score=0.0,
                execution_time_ms=0.0,
                error_message=str(e)
            )
            
            return RetrievalResult(
                data=[],
                metadata=retrieval_metadata,
                success=False,
                message=f"Error in fallback search: {e}"
            )
    
    def _calculate_average_confidence(self, data: List[Dict[str, Any]]) -> float:
        """Calculate average confidence score from results."""
        if not data:
            return 0.0
        
        scores = [item.get("score", 0.0) for item in data]
        return sum(scores) / len(scores)
    
    def _get_cache_key(self, 
                      query: str, 
                      query_type: Optional[QueryType],
                      facility_filter: Optional[str],
                      document_types: Optional[List[DocumentType]]) -> str:
        """Generate cache key for search parameters."""
        key_parts = [
            query,
            query_type.value if query_type else "none",
            facility_filter or "none",
            ",".join(dt.value for dt in document_types) if document_types else "none"
        ]
        return "|".join(key_parts)
    
    def clear_cache(self):
        """Clear the result cache."""
        self._result_cache.clear()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._result_cache),
            "cache_keys": list(self._result_cache.keys())
        }


def create_ehs_vector_cypher_retriever(
    neo4j_driver: Driver,
    embedding_manager: EmbeddingManager,
    performance_profile: str = "balanced",
    custom_config: Optional[VectorCypherConfig] = None
) -> EHSVectorCypherRetriever:
    """
    Factory function to create an EHS VectorCypher retriever with optimal configuration.
    
    Args:
        neo4j_driver: Neo4j database driver
        embedding_manager: Embedding manager instance
        performance_profile: "fast", "balanced", or "comprehensive"
        custom_config: Optional custom configuration
        
    Returns:
        Configured EHSVectorCypherRetriever instance
    """
    # Create base configuration
    if custom_config:
        config = custom_config
    else:
        config = VectorCypherConfig()
    
    # Create configuration manager and set performance profile
    config_manager = VectorCypherConfigManager(config)
    config_manager.set_performance_profile(performance_profile)
    
    # Create EHS vector configuration
    ehs_config = EHSVectorConfig()
    
    # Create and return retriever
    retriever = EHSVectorCypherRetriever(
        neo4j_driver=neo4j_driver,
        embedding_manager=embedding_manager,
        config=config_manager.base_config,
        ehs_config=ehs_config
    )
    
    logger.info(f"Created EHS VectorCypher retriever with {performance_profile} profile")
    
    return retriever