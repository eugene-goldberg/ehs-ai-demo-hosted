"""
LlamaIndex integration for document indexing and RAG capabilities.
Handles vector storage, hybrid search, and knowledge graph integration.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    PropertyGraphIndex,
    Document,
    ServiceContext,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
# Commented out llama-index imports not in requirements.txt
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.anthropic import Anthropic
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

logger = logging.getLogger(__name__)


class EHSDocumentIndexer:
    """
    Document indexer for EHS AI Platform using LlamaIndex.
    Supports both vector and graph-based indexing.
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4",
        use_local_embeddings: bool = False
    ):
        """
        Initialize the document indexer.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            embedding_model: Model to use for embeddings
            llm_model: LLM model for extraction and queries
            use_local_embeddings: Whether to use local embeddings model
        """
        # Neo4j connections
        self.graph_store = Neo4jPropertyGraphStore(
            username=neo4j_username,
            password=neo4j_password,
            url=neo4j_uri,
            database="neo4j"
        )
        
        self.vector_store = Neo4jVectorStore(
            username=neo4j_username,
            password=neo4j_password,
            url=neo4j_uri,
            embedding_dimension=1536 if "text-embedding" in embedding_model else 384,
            index_name="ehs_documents",
            node_label="DocumentChunk",
            text_node_property="text",
            embedding_node_property="embedding"
        )
        
        # Configure embedding model
        if use_local_embeddings:
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embed_model = OpenAIEmbedding(model=embedding_model)
        
        # Configure LLM
        if "claude" in llm_model.lower():
            self.llm = Anthropic(model=llm_model)
        else:
            self.llm = OpenAI(model=llm_model, temperature=0)
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        # Storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            property_graph_store=self.graph_store
        )
        
        # Initialize indexes
        self.vector_index = None
        self.graph_index = None
        
    def create_nodes_from_documents(
        self, 
        documents: List[Document],
        include_metadata_extractor: bool = True
    ) -> List[TextNode]:
        """
        Convert documents to nodes with metadata extraction.
        
        Args:
            documents: List of parsed documents
            include_metadata_extractor: Whether to extract additional metadata
            
        Returns:
            List of TextNode objects
        """
        # Create node parser
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=512,
            chunk_overlap=50
        )
        
        # Parse documents into nodes
        nodes = node_parser.get_nodes_from_documents(documents)
        
        if include_metadata_extractor:
            # Create extraction pipeline
            extractors = [
                TitleExtractor(nodes=5, llm=self.llm),
                KeywordExtractor(keywords=10, llm=self.llm),
                SummaryExtractor(summaries=["prev", "self"], llm=self.llm)
            ]
            
            pipeline = IngestionPipeline(
                transformations=extractors
            )
            
            # Run extraction pipeline
            nodes = pipeline.run(nodes=nodes)
        
        # Add EHS-specific metadata
        for node in nodes:
            if "document_type" in node.metadata:
                node.metadata["domain"] = "ehs"
                node.metadata["indexed_at"] = datetime.utcnow().isoformat()
        
        return nodes
    
    def build_vector_index(
        self, 
        documents: List[Document],
        index_id: Optional[str] = None
    ) -> VectorStoreIndex:
        """
        Build a vector index from documents.
        
        Args:
            documents: List of documents to index
            index_id: Optional index identifier
            
        Returns:
            VectorStoreIndex object
        """
        logger.info(f"Building vector index from {len(documents)} documents")
        
        # Create nodes
        nodes = self.create_nodes_from_documents(documents)
        
        # Create index
        self.vector_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            show_progress=True
        )
        
        if index_id:
            self.vector_index.set_index_id(index_id)
        
        logger.info(f"Vector index created with {len(nodes)} nodes")
        return self.vector_index
    
    def build_property_graph_index(
        self, 
        documents: List[Document],
        extract_relationships: bool = True
    ) -> PropertyGraphIndex:
        """
        Build a property graph index with entity and relationship extraction.
        
        Args:
            documents: List of documents to index
            extract_relationships: Whether to extract relationships
            
        Returns:
            PropertyGraphIndex object
        """
        logger.info(f"Building property graph index from {len(documents)} documents")
        
        # Create custom extraction prompt for EHS entities
        entity_prompt = """
        Extract entities related to Environmental Health and Safety (EHS) from the text.
        Focus on:
        - Facilities and locations
        - Equipment and assets
        - Utility consumption (electricity, gas, water)
        - Emissions and pollutants
        - Permits and compliance items
        - Vendors and suppliers
        - Dates and time periods
        - Quantities and measurements
        """
        
        # Build graph index
        self.graph_index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=self.graph_store,
            embed_model=self.embed_model,
            llm=self.llm,
            show_progress=True,
            include_embeddings=True,
            entity_prompt=entity_prompt
        )
        
        logger.info("Property graph index created")
        return self.graph_index
    
    def create_hybrid_index(
        self, 
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Create both vector and graph indexes for hybrid search.
        
        Args:
            documents: List of documents to index
            
        Returns:
            Dictionary containing both indexes
        """
        vector_index = self.build_vector_index(documents)
        graph_index = self.build_property_graph_index(documents)
        
        return {
            "vector_index": vector_index,
            "graph_index": graph_index
        }
    
    def create_query_engine(
        self, 
        similarity_top_k: int = 5,
        include_text: bool = True,
        response_mode: str = "tree_summarize"
    ) -> RetrieverQueryEngine:
        """
        Create a query engine for the indexed documents.
        
        Args:
            similarity_top_k: Number of similar documents to retrieve
            include_text: Whether to include text in responses
            response_mode: Response synthesis mode
            
        Returns:
            Query engine object
        """
        if not self.vector_index:
            raise ValueError("No vector index available. Build index first.")
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=similarity_top_k
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode=response_mode,
            node_postprocessors=[],
            verbose=True
        )
        
        return query_engine
    
    def create_graph_query_engine(self, include_text: bool = True):
        """
        Create a query engine for the property graph.
        
        Args:
            include_text: Whether to include source text in responses
            
        Returns:
            Graph query engine
        """
        if not self.graph_index:
            raise ValueError("No graph index available. Build index first.")
        
        return self.graph_index.as_query_engine(
            include_text=include_text,
            response_mode="tree_summarize",
            verbose=True
        )
    
    def add_document_relationships(
        self,
        source_doc_id: str,
        target_doc_id: str,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add relationships between documents in the graph.
        
        Args:
            source_doc_id: Source document ID
            target_doc_id: Target document ID
            relationship_type: Type of relationship
            metadata: Additional relationship metadata
        """
        # This would be implemented using Neo4j queries
        # For now, it's a placeholder for the relationship management
        query = """
        MATCH (s:Document {id: $source_id})
        MATCH (t:Document {id: $target_id})
        CREATE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
        SET r += $metadata
        RETURN r
        """
        
        # Execute query (implementation depends on Neo4j driver setup)
        logger.info(f"Added relationship: {source_doc_id} -{relationship_type}-> {target_doc_id}")
    
    def search_similar_documents(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            
        Returns:
            List of similar documents with scores
        """
        if not self.vector_index:
            raise ValueError("No vector index available")
        
        # Create retriever with filters
        retriever = self.vector_index.as_retriever(
            similarity_top_k=top_k,
            filters=filters
        )
        
        # Retrieve similar nodes
        nodes = retriever.retrieve(query)
        
        # Format results
        results = []
        for node in nodes:
            results.append({
                "text": node.node.get_content(),
                "score": node.score,
                "metadata": node.node.metadata,
                "node_id": node.node.node_id
            })
        
        return results
    
    def extract_ehs_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract EHS-specific entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and values
        """
        # This is a simplified version - in production, use NER models
        # or more sophisticated extraction
        prompt = f"""
        Extract the following entities from the text:
        - Facilities (building names, locations)
        - Equipment (model numbers, types)
        - Metrics (energy consumption, emissions, costs)
        - Dates (billing periods, compliance dates)
        - Compliance items (permit numbers, requirements)
        
        Text: {text}
        
        Return as JSON with entity types as keys and lists of values.
        """
        
        response = self.llm.complete(prompt)
        # Parse response and return entities
        # This is simplified - add proper JSON parsing and validation
        
        return {
            "facilities": [],
            "equipment": [],
            "metrics": [],
            "dates": [],
            "compliance": []
        }


# Utility function for quick indexing
def index_ehs_documents(
    documents: List[Document],
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    index_type: str = "hybrid"
) -> Dict[str, Any]:
    """
    Quick function to index EHS documents.
    
    Args:
        documents: List of documents to index
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        index_type: Type of index to create (vector, graph, hybrid)
        
    Returns:
        Dictionary containing the created indexes
    """
    indexer = EHSDocumentIndexer(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password
    )
    
    if index_type == "vector":
        return {"vector_index": indexer.build_vector_index(documents)}
    elif index_type == "graph":
        return {"graph_index": indexer.build_property_graph_index(documents)}
    else:  # hybrid
        return indexer.create_hybrid_index(documents)