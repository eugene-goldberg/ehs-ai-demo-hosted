"""
Vector retriever implementation for EHS documents using neo4j-graphrag-python.

This module provides vector-based document retrieval capabilities specifically
designed for EHS (Environmental, Health, Safety) documents including utility bills,
permits, compliance reports, and incident reports.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, date
from dataclasses import dataclass
import numpy as np

from neo4j_graphrag.retrievers import VectorRetriever as GraphRAGVectorRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from neo4j import GraphDatabase

from ..base import BaseRetriever, RetrievalResult, RetrievalMetadata, RetrievalStrategy
from ..config import RetrieverConfig
from .ehs_vector_config import EHSVectorConfig, DocumentType, QueryType
from ..embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    document_type: DocumentType


@dataclass
class VectorRetrieverConfig(RetrieverConfig):
    """Configuration for vector retriever."""
    embedding_model: str = "text-embedding-ada-002"
    similarity_threshold: float = 0.7
    max_results: int = 10
    include_metadata: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None
    chunk_overlap: int = 50
    rerank_results: bool = True
    
    # EHS-specific configurations
    facility_filter: Optional[str] = None
    date_range: Optional[Tuple[date, date]] = None
    document_types: Optional[List[DocumentType]] = None
    compliance_level_filter: Optional[str] = None


class EHSVectorRetriever(BaseRetriever):
    """
    Vector retriever for EHS documents with specialized handling for:
    - Utility bills and consumption data
    - Environmental permits and compliance documents
    - Incident reports and safety documentation
    - Facility-specific document filtering
    """
    
    def __init__(
        self,
        neo4j_driver,
        config: Optional[Dict[str, Any]] = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        """
        Initialize the EHS Vector Retriever.
        
        Args:
            neo4j_driver: Neo4j database driver
            config: Configuration dictionary
            embedding_manager: Manager for embeddings and chunking
        """
        super().__init__(config or {})
        self.neo4j_driver = neo4j_driver
        
        # Initialize configuration
        self.vector_config = VectorRetrieverConfig(**config) if config else VectorRetrieverConfig()
        
        # Initialize embedding manager
        self.embedding_manager = embedding_manager or EmbeddingManager(
            model_name=self.vector_config.embedding_model
        )
        
        # Initialize EHS-specific configuration
        self.ehs_config = EHSVectorConfig()
        
        # Initialize GraphRAG vector retriever
        self._init_graphrag_retriever()
        
        logger.info(f"Initialized EHS Vector Retriever with model: {self.vector_config.embedding_model}")
    
    async def initialize(self) -> None:
        """
        Initialize the vector retriever.
        
        This method sets up any necessary resources for vector retrieval.
        """
        if self._initialized:
            logger.debug("Vector retriever already initialized")
            return
            
        try:
            logger.info("Initializing EHS Vector Retriever")
            
            # Verify Neo4j connection
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                if result.single()["test"] != 1:
                    raise RuntimeError("Neo4j connection test failed")
            
            # Mark as initialized
            self._initialized = True
            logger.info("EHS Vector Retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector retriever: {e}")
            raise
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get the retrieval strategy identifier."""
        return RetrievalStrategy.VECTOR
    
    async def validate_query(self, query: str) -> bool:
        """Validate if the query can be processed by this retriever."""
        # Vector retriever can handle any text query
        return len(query.strip()) > 0
    
    def _init_graphrag_retriever(self):
        """Initialize the neo4j-graphrag-python VectorRetriever."""
        try:
            # Configure embedding provider based on model
            if self.vector_config.embedding_model.startswith("text-embedding"):
                # OpenAI embeddings
                embedder = OpenAIEmbeddings(
                    model=self.vector_config.embedding_model
                )
            else:
                # Sentence transformer embeddings
                embedder = SentenceTransformerEmbeddings(
                    model=self.vector_config.embedding_model
                )
            
            # Initialize GraphRAG vector retriever
            self.graphrag_retriever = GraphRAGVectorRetriever(
                driver=self.neo4j_driver,
                index_name="ehs_document_chunks",
                embedder=embedder,
                return_properties=["content", "metadata", "document_id", "chunk_id", "document_type"]
            )
            
            logger.info("GraphRAG Vector Retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG Vector Retriever: {e}")
            self.graphrag_retriever = None
    
    async def retrieve(
        self,
        query: str,
        query_type: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve EHS documents using vector similarity search.
        
        Args:
            query: Natural language query
            query_type: Type of query (optional)
            limit: Maximum number of results
            **kwargs: Additional retrieval parameters
        
        Returns:
            RetrievalResult with relevant documents and metadata
        """
        try:
            start_time = datetime.now()
            
            # Update config with kwargs
            config_updates = self._extract_config_updates(kwargs, limit)
            
            # Preprocess query for EHS context
            processed_query = self._preprocess_ehs_query(query)
            
            # Perform vector search
            search_results = self._vector_search(processed_query, config_updates)
            
            # Apply EHS-specific post-processing
            filtered_results = self._apply_ehs_filters(search_results, config_updates)
            
            # Rerank results if enabled
            if self.vector_config.rerank_results:
                reranked_results = self._rerank_results(filtered_results, processed_query)
            else:
                reranked_results = filtered_results
            
            # Format response
            response = self._format_response(reranked_results, query, start_time)
            
            logger.info(f"Vector retrieval completed: {len(reranked_results)} results in {response.metadata.execution_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return RetrievalResult(
                success=False,
                data=[],
                metadata=RetrievalMetadata(
                    query=query,
                    query_type=query_type,
                    strategy=self.get_strategy(),
                    execution_time=processing_time,
                    total_results=0,
                    error_message=str(e)
                )
            )
    
    def _extract_config_updates(self, kwargs: Dict[str, Any], limit: Optional[int]) -> Dict[str, Any]:
        """Extract configuration updates from kwargs."""
        config_updates = {}
        
        # Standard parameters
        if limit is not None:
            config_updates["max_results"] = limit
        if "max_results" in kwargs:
            config_updates["max_results"] = kwargs["max_results"]
        if "similarity_threshold" in kwargs:
            config_updates["similarity_threshold"] = kwargs["similarity_threshold"]
        
        # EHS-specific parameters
        if "facility" in kwargs:
            config_updates["facility_filter"] = kwargs["facility"]
        if "date_range" in kwargs:
            config_updates["date_range"] = kwargs["date_range"]
        if "document_types" in kwargs:
            config_updates["document_types"] = kwargs["document_types"]
        if "compliance_level" in kwargs:
            config_updates["compliance_level_filter"] = kwargs["compliance_level"]
        
        return config_updates
    
    def _preprocess_ehs_query(self, query: str) -> str:
        """
        Preprocess query to enhance EHS document retrieval.
        
        Args:
            query: Original query string
            
        Returns:
            Preprocessed query optimized for EHS context
        """
        # Add EHS context keywords if not present
        ehs_keywords = [
            "environmental", "health", "safety", "compliance", "permit",
            "utility", "consumption", "incident", "report", "facility"
        ]
        
        query_lower = query.lower()
        missing_context = []
        
        # Check if query needs EHS context
        has_ehs_context = any(keyword in query_lower for keyword in ehs_keywords)
        
        if not has_ehs_context:
            # Infer context from query type
            if any(word in query_lower for word in ["water", "electricity", "gas", "bill", "usage"]):
                missing_context.append("utility consumption")
            elif any(word in query_lower for word in ["permit", "license", "approval", "regulatory"]):
                missing_context.append("environmental compliance")
            elif any(word in query_lower for word in ["accident", "incident", "injury", "safety"]):
                missing_context.append("safety incident")
        
        # Enhance query with missing context
        if missing_context:
            enhanced_query = f"{query} {' '.join(missing_context)}"
        else:
            enhanced_query = query
        
        logger.debug(f"Preprocessed query: '{query}' -> '{enhanced_query}'")
        return enhanced_query
    
    def _vector_search(self, query: str, config_updates: Dict[str, Any]) -> List[VectorSearchResult]:
        """
        Perform vector similarity search using GraphRAG.
        
        Args:
            query: Preprocessed query string
            config_updates: Configuration updates
            
        Returns:
            List of vector search results
        """
        if not self.graphrag_retriever:
            logger.warning("GraphRAG retriever not available, falling back to custom implementation")
            return self._fallback_vector_search(query, config_updates)
        
        try:
            # Configure search parameters
            top_k = config_updates.get("max_results", self.vector_config.max_results)
            
            # Build Cypher filter conditions
            filter_conditions = self._build_filter_conditions(config_updates)
            
            # Perform GraphRAG vector search
            results = self.graphrag_retriever.search(
                query_text=query,
                top_k=top_k,
                filter=filter_conditions
            )
            
            # Convert to VectorSearchResult objects
            search_results = []
            for result in results:
                search_results.append(VectorSearchResult(
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                    chunk_id=result.get("chunk_id", ""),
                    document_id=result.get("document_id", ""),
                    document_type=DocumentType(result.get("document_type", "unknown"))
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"GraphRAG vector search failed: {e}")
            return self._fallback_vector_search(query, config_updates)
    
    def _fallback_vector_search(self, query: str, config_updates: Dict[str, Any]) -> List[VectorSearchResult]:
        """
        Fallback vector search implementation using direct Neo4j queries.
        
        Args:
            query: Query string
            config_updates: Configuration updates
            
        Returns:
            List of vector search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embedding(query)
            
            # Build Cypher query for vector similarity
            cypher_query = """
            MATCH (chunk:DocumentChunk)
            WHERE chunk.embedding IS NOT NULL
            {filter_conditions}
            WITH chunk, 
                 gds.similarity.cosine(chunk.embedding, $query_embedding) AS score
            WHERE score >= $similarity_threshold
            MATCH (chunk)-[:PART_OF]->(doc:Document)
            RETURN chunk.content AS content,
                   score,
                   chunk.metadata AS metadata,
                   chunk.id AS chunk_id,
                   doc.id AS document_id,
                   doc.type AS document_type
            ORDER BY score DESC
            LIMIT $max_results
            """
            
            # Add filter conditions
            filter_conditions = self._build_cypher_filters(config_updates)
            cypher_query = cypher_query.format(filter_conditions=filter_conditions)
            
            # Execute query
            with self.neo4j_driver.session() as session:
                results = session.run(
                    cypher_query,
                    query_embedding=query_embedding.tolist(),
                    similarity_threshold=config_updates.get("similarity_threshold", self.vector_config.similarity_threshold),
                    max_results=config_updates.get("max_results", self.vector_config.max_results)
                )
                
                search_results = []
                for record in results:
                    search_results.append(VectorSearchResult(
                        content=record["content"],
                        score=record["score"],
                        metadata=record["metadata"] or {},
                        chunk_id=record["chunk_id"],
                        document_id=record["document_id"],
                        document_type=DocumentType(record["document_type"])
                    ))
                
                return search_results
                
        except Exception as e:
            logger.error(f"Fallback vector search failed: {e}")
            return []
    
    def _build_filter_conditions(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Build filter conditions for GraphRAG search."""
        filters = {}
        
        # Facility filter
        if config_updates.get("facility_filter"):
            filters["facility"] = config_updates["facility_filter"]
        
        # Document type filter
        if config_updates.get("document_types"):
            filters["document_type"] = [dt.value for dt in config_updates["document_types"]]
        
        # Date range filter
        if config_updates.get("date_range"):
            start_date, end_date = config_updates["date_range"]
            filters["date_created"] = {
                "gte": start_date.isoformat(),
                "lte": end_date.isoformat()
            }
        
        # Compliance level filter
        if config_updates.get("compliance_level_filter"):
            filters["compliance_level"] = config_updates["compliance_level_filter"]
        
        return filters
    
    def _build_cypher_filters(self, config_updates: Dict[str, Any]) -> str:
        """Build Cypher WHERE conditions for fallback search."""
        conditions = []
        
        # Facility filter
        if config_updates.get("facility_filter"):
            conditions.append(f"chunk.metadata.facility = '{config_updates['facility_filter']}'")
        
        # Document type filter
        if config_updates.get("document_types"):
            doc_types = [f"'{dt.value}'" for dt in config_updates["document_types"]]
            conditions.append(f"doc.type IN [{', '.join(doc_types)}]")
        
        # Date range filter
        if config_updates.get("date_range"):
            start_date, end_date = config_updates["date_range"]
            conditions.append(f"doc.date_created >= date('{start_date.isoformat()}')")
            conditions.append(f"doc.date_created <= date('{end_date.isoformat()}')")
        
        # Compliance level filter
        if config_updates.get("compliance_level_filter"):
            conditions.append(f"chunk.metadata.compliance_level = '{config_updates['compliance_level_filter']}'")
        
        if conditions:
            return "AND " + " AND ".join(conditions)
        return ""
    
    def _apply_ehs_filters(self, results: List[VectorSearchResult], config_updates: Dict[str, Any]) -> List[VectorSearchResult]:
        """Apply EHS-specific filtering to search results."""
        filtered_results = results
        
        # Apply similarity threshold
        similarity_threshold = config_updates.get("similarity_threshold", self.vector_config.similarity_threshold)
        filtered_results = [r for r in filtered_results if r.score >= similarity_threshold]
        
        # Apply document type filtering (client-side backup)
        if config_updates.get("document_types"):
            allowed_types = config_updates["document_types"]
            filtered_results = [r for r in filtered_results if r.document_type in allowed_types]
        
        # Apply EHS-specific quality filters
        quality_filtered = []
        for result in filtered_results:
            if self._meets_ehs_quality_criteria(result):
                quality_filtered.append(result)
        
        return quality_filtered
    
    def _meets_ehs_quality_criteria(self, result: VectorSearchResult) -> bool:
        """Check if result meets EHS-specific quality criteria."""
        # Minimum content length
        if len(result.content.strip()) < 50:
            return False
        
        # Check for EHS relevance indicators
        content_lower = result.content.lower()
        ehs_indicators = [
            "environmental", "health", "safety", "compliance", "permit",
            "utility", "consumption", "incident", "facility", "regulation"
        ]
        
        # Require at least one EHS indicator
        has_ehs_indicator = any(indicator in content_lower for indicator in ehs_indicators)
        
        # Check metadata quality
        metadata = result.metadata
        has_good_metadata = bool(
            metadata.get("document_title") or 
            metadata.get("facility") or 
            metadata.get("document_type")
        )
        
        return has_ehs_indicator and has_good_metadata
    
    def _rerank_results(self, results: List[VectorSearchResult], query: str) -> List[VectorSearchResult]:
        """Rerank results using EHS-specific criteria."""
        # Document type priority based on query
        query_lower = query.lower()
        type_priority = self.ehs_config.get_document_type_priority(query_lower)
        
        # Rerank with combined score
        reranked_results = []
        for result in results:
            # Base similarity score
            base_score = result.score
            
            # Document type boost
            type_boost = type_priority.get(result.document_type, 0.0)
            
            # Recency boost (if date available)
            recency_boost = self._calculate_recency_boost(result)
            
            # Facility relevance boost
            facility_boost = self._calculate_facility_boost(result, query)
            
            # Combined score
            combined_score = base_score + (type_boost * 0.1) + (recency_boost * 0.05) + (facility_boost * 0.05)
            
            # Update result with new score
            reranked_result = VectorSearchResult(
                content=result.content,
                score=combined_score,
                metadata=result.metadata,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_type=result.document_type
            )
            reranked_results.append(reranked_result)
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        return reranked_results
    
    def _calculate_recency_boost(self, result: VectorSearchResult) -> float:
        """Calculate recency boost for result."""
        try:
            date_created = result.metadata.get("date_created")
            if not date_created:
                return 0.0
            
            if isinstance(date_created, str):
                doc_date = datetime.fromisoformat(date_created).date()
            else:
                doc_date = date_created
            
            # Calculate days since creation
            days_old = (datetime.now().date() - doc_date).days
            
            # More recent documents get higher boost (up to 0.1)
            if days_old <= 30:
                return 0.1
            elif days_old <= 90:
                return 0.05
            elif days_old <= 365:
                return 0.02
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_facility_boost(self, result: VectorSearchResult, query: str) -> float:
        """Calculate facility relevance boost."""
        facility = result.metadata.get("facility")
        if not facility:
            return 0.0
        
        # Check if facility is mentioned in query
        if facility.lower() in query.lower():
            return 0.1
        
        return 0.0
    
    def _format_response(self, results: List[VectorSearchResult], query: str, start_time: datetime) -> RetrievalResult:
        """Format search results into RetrievalResult."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert results to response format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.content,
                "score": result.score,
                "metadata": {
                    **result.metadata,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "document_type": result.document_type.value
                }
            })
        
        return RetrievalResult(
            success=True,
            data=formatted_results,
            metadata=RetrievalMetadata(
                query=query,
                strategy=self.get_strategy(),
                execution_time=processing_time,
                total_results=len(formatted_results),
                config_used={
                    "embedding_model": self.vector_config.embedding_model,
                    "similarity_threshold": self.vector_config.similarity_threshold,
                    "max_results": self.vector_config.max_results,
                    "include_metadata": self.vector_config.include_metadata,
                    "rerank_results": self.vector_config.rerank_results
                }
            )
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector index.
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for doc in documents:
                # Process document with embedding manager
                processed_chunks = self.embedding_manager.process_document(
                    content=doc["content"],
                    metadata=doc.get("metadata", {}),
                    document_id=doc.get("id"),
                    document_type=doc.get("type", "unknown")
                )
                
                # Store chunks in Neo4j
                self._store_document_chunks(processed_chunks)
            
            logger.info(f"Successfully added {len(documents)} documents to vector index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector index: {e}")
            return False
    
    def _store_document_chunks(self, chunks: List[Dict[str, Any]]):
        """Store document chunks in Neo4j with embeddings."""
        cypher_query = """
        MERGE (doc:Document {id: $document_id})
        SET doc.type = $document_type,
            doc.title = $document_title,
            doc.date_created = date($date_created)
        
        WITH doc
        CREATE (chunk:DocumentChunk)
        SET chunk.id = $chunk_id,
            chunk.content = $content,
            chunk.embedding = $embedding,
            chunk.metadata = $metadata
        
        CREATE (chunk)-[:PART_OF]->(doc)
        """
        
        with self.neo4j_driver.session() as session:
            for chunk in chunks:
                session.run(
                    cypher_query,
                    document_id=chunk["document_id"],
                    document_type=chunk["document_type"],
                    document_title=chunk.get("document_title", ""),
                    date_created=chunk.get("date_created", datetime.now().isoformat()),
                    chunk_id=chunk["chunk_id"],
                    content=chunk["content"],
                    embedding=chunk["embedding"].tolist(),
                    metadata=chunk["metadata"]
                )