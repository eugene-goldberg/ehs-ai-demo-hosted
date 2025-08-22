"""
Context Builder for EHS RAG Agent

This module provides context window management, result prioritization and filtering,
metadata preservation for citations, context compression, and relevance scoring.
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import retrieval types
from ..retrieval.base import RetrievalResult, QueryType
from .query_router import QueryClassification, IntentType

# Import logging and monitoring
from ..utils.logging import get_ehs_logger, performance_logger, log_context
from ..utils.tracing import trace_function, SpanKind

logger = get_ehs_logger(__name__)


class SourceType(str, Enum):
    """Types of sources in context."""
    
    DATABASE_RECORD = "database_record"
    DOCUMENT = "document"
    KNOWLEDGE_BASE = "knowledge_base"
    REGULATION = "regulation"
    EQUIPMENT_DATA = "equipment_data"
    FACILITY_DATA = "facility_data"


@dataclass
class ContextSource:
    """Individual source in the context window."""
    
    id: str
    content: str
    source_type: SourceType
    relevance_score: float
    confidence_score: float
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None
    retriever_strategy: Optional[str] = None
    
    def get_citation(self) -> str:
        """Generate citation string for this source."""
        if self.source_type == SourceType.DATABASE_RECORD:
            return f"Database Record {self.id}"
        elif self.source_type == SourceType.DOCUMENT:
            title = self.metadata.get("title", "Document")
            return f"{title} (ID: {self.id})"
        elif self.source_type == SourceType.REGULATION:
            reg_name = self.metadata.get("regulation_name", "Regulation")
            return f"{reg_name} (ID: {self.id})"
        else:
            return f"{self.source_type.value.title()} {self.id}"


@dataclass
class ContextWindow:
    """Context window containing processed content and sources."""
    
    content: str
    sources: List[ContextSource]
    metadata: Dict[str, Any]
    total_length: int = field(init=False)
    source_count: int = field(init=False)
    average_relevance: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.total_length = len(self.content)
        self.source_count = len(self.sources)
        self.average_relevance = (
            sum(source.relevance_score for source in self.sources) / len(self.sources)
            if self.sources else 0.0
        )
    
    def get_citations(self) -> List[str]:
        """Get formatted citations for all sources."""
        return [source.get_citation() for source in self.sources]
    
    def get_top_sources(self, n: int = 5) -> List[ContextSource]:
        """Get top N sources by relevance score."""
        return sorted(self.sources, key=lambda s: s.relevance_score, reverse=True)[:n]


class ContextBuilder:
    """
    Context Builder for RAG processing.
    
    Manages context windows, prioritizes and filters results, preserves metadata
    for citations, compresses content for long documents, and scores relevance.
    """
    
    def __init__(
        self,
        max_length: int = 8000,
        compression_ratio: float = 0.8,
        include_metadata: bool = True,
        min_relevance_threshold: float = 0.3
    ):
        """
        Initialize the context builder.
        
        Args:
            max_length: Maximum context window length
            compression_ratio: Ratio for context compression (0.0 to 1.0)
            include_metadata: Whether to include metadata in context
            min_relevance_threshold: Minimum relevance score for inclusion
        """
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        self.include_metadata = include_metadata
        self.min_relevance_threshold = min_relevance_threshold
        
        logger.info(
            "Context Builder initialized",
            max_length=max_length,
            compression_ratio=compression_ratio,
            include_metadata=include_metadata,
            min_relevance_threshold=min_relevance_threshold
        )
    
    @performance_logger(include_args=True, include_result=False)
    @trace_function("build_context", SpanKind.INTERNAL, {"component": "context_builder"})
    async def build_context(
        self,
        query: str,
        classification: QueryClassification,
        retrieval_results: List[RetrievalResult]
    ) -> ContextWindow:
        """
        Build context window from retrieval results.
        
        Args:
            query: Original user query
            classification: Query classification result
            retrieval_results: Results from retrievers
            
        Returns:
            ContextWindow with processed content and sources
        """
        with log_context(
            component="context_builder",
            operation="build_context",
            query_length=len(query),
            retrieval_count=len(retrieval_results)
        ):
            logger.debug("Building context window from retrieval results")
            
            # Step 1: Extract and process sources from retrieval results
            raw_sources = await self._extract_sources(retrieval_results)
            
            # Step 2: Score relevance against query and classification
            scored_sources = await self._score_relevance(query, classification, raw_sources)
            
            # Step 3: Filter and prioritize sources
            filtered_sources = await self._filter_and_prioritize(scored_sources)
            
            # Step 4: Build context content respecting length limits
            context_content = await self._build_context_content(
                query, classification, filtered_sources
            )
            
            # Step 5: Apply compression if needed
            if len(context_content) > self.max_length:
                context_content = await self._compress_content(context_content)
            
            # Step 6: Create context window
            context_window = ContextWindow(
                content=context_content,
                sources=filtered_sources,
                metadata={
                    "query": query,
                    "intent_type": classification.intent_type.value,
                    "classification_confidence": classification.confidence_score,
                    "total_retrieval_results": len(retrieval_results),
                    "successful_retrievals": len([r for r in retrieval_results if r.success]),
                    "context_length": len(context_content),
                    "compression_applied": len(context_content) < self.max_length * self.compression_ratio,
                    "build_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(
                "Context window built successfully",
                content_length=context_window.total_length,
                source_count=context_window.source_count,
                average_relevance=context_window.average_relevance
            )
            
            return context_window
    
    @trace_function("extract_sources", SpanKind.INTERNAL, {"context_step": "extraction"})
    async def _extract_sources(self, retrieval_results: List[RetrievalResult]) -> List[ContextSource]:
        """Extract sources from retrieval results."""
        sources = []
        
        for result in retrieval_results:
            if not result.success or not result.data:
                continue
                
            for i, data_item in enumerate(result.data):
                source = await self._create_context_source(data_item, result, i)
                if source:
                    sources.append(source)
        
        logger.debug(f"Extracted {len(sources)} sources from retrieval results")
        return sources
    
    async def _create_context_source(
        self,
        data_item: Dict[str, Any],
        retrieval_result: RetrievalResult,
        index: int
    ) -> Optional[ContextSource]:
        """Create a context source from a data item."""
        try:
            # Extract content and metadata
            content = self._extract_content_from_data(data_item)
            if not content:
                return None
            
            # Determine source type
            source_type = self._determine_source_type(data_item)
            
            # Generate unique ID
            source_id = self._generate_source_id(data_item, retrieval_result, index)
            
            # Extract metadata
            metadata = self._extract_source_metadata(data_item)
            
            return ContextSource(
                id=source_id,
                content=content,
                source_type=source_type,
                relevance_score=0.0,  # Will be calculated in scoring step
                confidence_score=retrieval_result.metadata.confidence_score,
                metadata=metadata,
                timestamp=datetime.utcnow(),
                retriever_strategy=retrieval_result.metadata.strategy.value
            )
            
        except Exception as e:
            logger.warning(f"Failed to create context source: {str(e)}")
            return None
    
    def _extract_content_from_data(self, data_item: Dict[str, Any]) -> Optional[str]:
        """Extract textual content from data item."""
        # Try common content fields
        content_fields = ["content", "text", "description", "summary", "name", "title"]
        
        for field in content_fields:
            if field in data_item and data_item[field]:
                return str(data_item[field])
        
        # If no direct content field, create from key-value pairs
        content_parts = []
        for key, value in data_item.items():
            if isinstance(value, (str, int, float)) and value is not None:
                content_parts.append(f"{key}: {value}")
        
        return "; ".join(content_parts) if content_parts else None
    
    def _determine_source_type(self, data_item: Dict[str, Any]) -> SourceType:
        """Determine the type of source based on data item structure."""
        # Check for node labels or types
        labels = data_item.get("labels", data_item.get("node_labels", []))
        
        if "Equipment" in labels:
            return SourceType.EQUIPMENT_DATA
        elif "Facility" in labels:
            return SourceType.FACILITY_DATA
        elif "Permit" in labels or "regulation" in str(data_item).lower():
            return SourceType.REGULATION
        elif "document" in str(data_item).lower() or "title" in data_item:
            return SourceType.DOCUMENT
        else:
            return SourceType.DATABASE_RECORD
    
    def _generate_source_id(
        self,
        data_item: Dict[str, Any],
        retrieval_result: RetrievalResult,
        index: int
    ) -> str:
        """Generate unique ID for source."""
        # Try to use actual ID from data
        if "id" in data_item:
            return f"{retrieval_result.metadata.strategy.value}_{data_item['id']}"
        elif "elementId" in data_item:
            return f"{retrieval_result.metadata.strategy.value}_{data_item['elementId']}"
        else:
            return f"{retrieval_result.metadata.strategy.value}_{index}"
    
    def _extract_source_metadata(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from data item."""
        metadata = {}
        
        # Copy important fields
        important_fields = [
            "name", "type", "facility", "equipment", "date", "timestamp",
            "created_at", "updated_at", "labels", "properties"
        ]
        
        for field in important_fields:
            if field in data_item:
                metadata[field] = data_item[field]
        
        return metadata
    
    @trace_function("score_relevance", SpanKind.INTERNAL, {"context_step": "relevance_scoring"})
    async def _score_relevance(
        self,
        query: str,
        classification: QueryClassification,
        sources: List[ContextSource]
    ) -> List[ContextSource]:
        """Score relevance of sources against query and classification."""
        logger.debug(f"Scoring relevance for {len(sources)} sources")
        
        query_lower = query.lower()
        query_terms = self._extract_query_terms(query_lower)
        
        for source in sources:
            content_lower = source.content.lower()
            
            # Term matching score
            term_score = self._calculate_term_matching_score(query_terms, content_lower)
            
            # Intent-specific scoring
            intent_score = self._calculate_intent_specific_score(
                classification.intent_type, source
            )
            
            # Entity matching score
            entity_score = self._calculate_entity_matching_score(
                classification.entities_identified, source
            )
            
            # Temporal relevance score
            temporal_score = self._calculate_temporal_relevance_score(source)
            
            # Combine scores with weights
            relevance_score = (
                0.4 * term_score +
                0.3 * intent_score +
                0.2 * entity_score +
                0.1 * temporal_score
            )
            
            source.relevance_score = min(max(relevance_score, 0.0), 1.0)
        
        logger.debug("Relevance scoring completed")
        return sources
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract important terms from query."""
        # Remove stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "what", "how", "when",
            "where", "why", "show", "tell", "give", "find"
        }
        
        # Split and clean terms
        terms = re.findall(r'\b\w+\b', query.lower())
        meaningful_terms = [term for term in terms if term not in stop_words and len(term) > 2]
        
        return meaningful_terms
    
    def _calculate_term_matching_score(self, query_terms: List[str], content: str) -> float:
        """Calculate score based on term matching."""
        if not query_terms:
            return 0.0
        
        matches = sum(1 for term in query_terms if term in content)
        return matches / len(query_terms)
    
    def _calculate_intent_specific_score(self, intent: IntentType, source: ContextSource) -> float:
        """Calculate score based on intent-source type matching."""
        intent_source_weights = {
            IntentType.CONSUMPTION_ANALYSIS: {
                SourceType.DATABASE_RECORD: 0.9,
                SourceType.EQUIPMENT_DATA: 0.8,
                SourceType.FACILITY_DATA: 0.7,
                SourceType.DOCUMENT: 0.3
            },
            IntentType.COMPLIANCE_CHECK: {
                SourceType.REGULATION: 0.9,
                SourceType.DATABASE_RECORD: 0.7,
                SourceType.DOCUMENT: 0.6,
                SourceType.FACILITY_DATA: 0.5
            },
            IntentType.RISK_ASSESSMENT: {
                SourceType.DATABASE_RECORD: 0.8,
                SourceType.EQUIPMENT_DATA: 0.7,
                SourceType.FACILITY_DATA: 0.7,
                SourceType.DOCUMENT: 0.6
            },
            IntentType.EMISSION_TRACKING: {
                SourceType.DATABASE_RECORD: 0.9,
                SourceType.EQUIPMENT_DATA: 0.8,
                SourceType.FACILITY_DATA: 0.7,
                SourceType.DOCUMENT: 0.4
            },
            IntentType.EQUIPMENT_EFFICIENCY: {
                SourceType.EQUIPMENT_DATA: 0.9,
                SourceType.DATABASE_RECORD: 0.8,
                SourceType.FACILITY_DATA: 0.6,
                SourceType.DOCUMENT: 0.4
            },
            IntentType.PERMIT_STATUS: {
                SourceType.REGULATION: 0.9,
                SourceType.DATABASE_RECORD: 0.8,
                SourceType.FACILITY_DATA: 0.6,
                SourceType.DOCUMENT: 0.5
            },
            IntentType.GENERAL_INQUIRY: {
                SourceType.DATABASE_RECORD: 0.7,
                SourceType.DOCUMENT: 0.7,
                SourceType.KNOWLEDGE_BASE: 0.8,
                SourceType.FACILITY_DATA: 0.6
            }
        }
        
        weights = intent_source_weights.get(intent, {})
        return weights.get(source.source_type, 0.5)  # Default weight
    
    def _calculate_entity_matching_score(self, entities, source: ContextSource) -> float:
        """Calculate score based on entity matching."""
        content_lower = source.content.lower()
        total_entities = 0
        matched_entities = 0
        
        # Check facility matches
        for facility in entities.facilities:
            total_entities += 1
            if facility.lower() in content_lower:
                matched_entities += 1
        
        # Check equipment matches
        for equipment in entities.equipment:
            total_entities += 1
            if equipment.lower() in content_lower:
                matched_entities += 1
        
        # Check pollutant matches
        for pollutant in entities.pollutants:
            total_entities += 1
            if pollutant.lower() in content_lower:
                matched_entities += 1
        
        return matched_entities / total_entities if total_entities > 0 else 0.0
    
    def _calculate_temporal_relevance_score(self, source: ContextSource) -> float:
        """Calculate score based on temporal relevance."""
        # More recent data gets higher score
        if source.timestamp:
            days_old = (datetime.utcnow() - source.timestamp).days
            # Decay factor - data older than 365 days gets lower score
            return max(0.1, 1.0 - (days_old / 365.0))
        
        return 0.5  # Default for sources without timestamps
    
    @trace_function("filter_and_prioritize", SpanKind.INTERNAL, {"context_step": "filtering"})
    async def _filter_and_prioritize(self, sources: List[ContextSource]) -> List[ContextSource]:
        """Filter and prioritize sources based on relevance and quality."""
        logger.debug(f"Filtering and prioritizing {len(sources)} sources")
        
        # Filter by minimum relevance threshold
        filtered_sources = [
            source for source in sources
            if source.relevance_score >= self.min_relevance_threshold
        ]
        
        # Sort by relevance score (descending)
        filtered_sources.sort(key=lambda s: s.relevance_score, reverse=True)
        
        # Limit to top sources to manage context length
        max_sources = min(20, len(filtered_sources))  # Reasonable limit
        prioritized_sources = filtered_sources[:max_sources]
        
        logger.debug(
            f"Filtered to {len(prioritized_sources)} sources "
            f"(threshold: {self.min_relevance_threshold})"
        )
        
        return prioritized_sources
    
    @trace_function("build_context_content", SpanKind.INTERNAL, {"context_step": "content_building"})
    async def _build_context_content(
        self,
        query: str,
        classification: QueryClassification,
        sources: List[ContextSource]
    ) -> str:
        """Build the actual context content string."""
        content_parts = []
        
        # Add query and classification context
        content_parts.append(f"User Query: {query}")
        content_parts.append(f"Query Intent: {classification.intent_type.value}")
        
        if classification.entities_identified.facilities:
            content_parts.append(f"Facilities: {', '.join(classification.entities_identified.facilities)}")
        
        if classification.entities_identified.equipment:
            content_parts.append(f"Equipment: {', '.join(classification.entities_identified.equipment)}")
        
        content_parts.append("\nRelevant Information:")
        
        # Add source content
        for i, source in enumerate(sources, 1):
            if self.include_metadata and source.metadata:
                metadata_str = self._format_metadata(source.metadata)
                content_parts.append(
                    f"\n[Source {i}] ({source.source_type.value}, relevance: {source.relevance_score:.2f})\n"
                    f"Content: {source.content}\n"
                    f"Metadata: {metadata_str}"
                )
            else:
                content_parts.append(
                    f"\n[Source {i}] {source.content}"
                )
        
        return "\n".join(content_parts)
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for inclusion in context."""
        formatted_items = []
        for key, value in metadata.items():
            if value is not None:
                formatted_items.append(f"{key}={value}")
        
        return "; ".join(formatted_items)
    
    @trace_function("compress_content", SpanKind.INTERNAL, {"context_step": "compression"})
    async def _compress_content(self, content: str) -> str:
        """Apply content compression to fit within length limits."""
        target_length = int(self.max_length * self.compression_ratio)
        
        if len(content) <= target_length:
            return content
        
        logger.debug(f"Compressing content from {len(content)} to {target_length} characters")
        
        # Simple truncation with ellipsis (more sophisticated compression could be implemented)
        # Try to break at sentence boundaries
        sentences = content.split('.')
        compressed_parts = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) + 1 > target_length - 10:  # Leave room for ellipsis
                break
            compressed_parts.append(sentence)
            current_length += len(sentence) + 1
        
        compressed = '.'.join(compressed_parts)
        if len(compressed) < len(content):
            compressed += "... [content truncated]"
        
        return compressed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context builder statistics."""
        return {
            "max_length": self.max_length,
            "compression_ratio": self.compression_ratio,
            "include_metadata": self.include_metadata,
            "min_relevance_threshold": self.min_relevance_threshold,
            "total_contexts_built": 0,  # Would be tracked in production
            "average_context_length": 0.0,  # Would be calculated from history
            "average_source_count": 0.0     # Would be calculated from history
        }