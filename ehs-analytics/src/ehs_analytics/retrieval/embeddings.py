"""
Embedding manager for EHS documents with support for multiple models and EHS-specific processing.

This module handles document chunking, metadata extraction, and embedding generation
specifically optimized for EHS documents including utility bills, permits, compliance
reports, and incident reports.
"""

import logging
import re
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI

from ..config import Settings

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Supported embedding models."""
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    SENTENCE_TRANSFORMER_MPNET = "all-mpnet-base-v2"
    SENTENCE_TRANSFORMER_MINILM = "all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMER_DISTILBERT = "all-distilroberta-v1"


class DocumentType(Enum):
    """EHS document types for specialized processing."""
    UTILITY_BILL = "utility_bill"
    ENVIRONMENTAL_PERMIT = "environmental_permit"
    COMPLIANCE_REPORT = "compliance_report"
    INCIDENT_REPORT = "incident_report"
    SAFETY_INSPECTION = "safety_inspection"
    WASTE_MANIFEST = "waste_manifest"
    EMISSION_REPORT = "emission_report"
    TRAINING_RECORD = "training_record"
    UNKNOWN = "unknown"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata and embedding."""
    chunk_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    document_id: Optional[str] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0


@dataclass
class ChunkingStrategy:
    """Configuration for document chunking strategy."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    
    # EHS-specific chunking parameters
    preserve_tables: bool = True
    preserve_sections: bool = True
    section_markers: List[str] = None
    
    def __post_init__(self):
        if self.section_markers is None:
            self.section_markers = [
                "Summary", "Overview", "Details", "Analysis", "Recommendations",
                "Consumption", "Usage", "Permit", "Compliance", "Incident",
                "Safety", "Environmental", "Waste", "Emission"
            ]


class EHSMetadataExtractor:
    """Extracts EHS-specific metadata from document content."""
    
    def __init__(self):
        self.facility_patterns = [
            r"facility[:\s]+([^,\n]+)",
            r"location[:\s]+([^,\n]+)",
            r"site[:\s]+([^,\n]+)",
            r"plant[:\s]+([^,\n]+)"
        ]
        
        self.date_patterns = [
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{4}-\d{2}-\d{2})",
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
        ]
        
        self.compliance_patterns = [
            r"permit\s+(?:number|#)[:\s]*([A-Z0-9-]+)",
            r"compliance\s+(?:level|status)[:\s]*([^,\n]+)",
            r"violation[:\s]*([^,\n]+)",
            r"non-compliance[:\s]*([^,\n]+)"
        ]
        
        self.consumption_patterns = [
            r"(\d+(?:\.\d+)?)\s*(kWh|kwh|kilowatt.?hours?)",
            r"(\d+(?:\.\d+)?)\s*(gallons?|gal)",
            r"(\d+(?:\.\d+)?)\s*(cubic\s+feet|cf|ccf)",
            r"(\d+(?:\.\d+)?)\s*(therms?)"
        ]
    
    def extract_metadata(self, content: str, document_type: DocumentType) -> Dict[str, Any]:
        """
        Extract EHS-specific metadata from document content.
        
        Args:
            content: Document content text
            document_type: Type of EHS document
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "document_type": document_type.value,
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        # Extract facility information
        facility = self._extract_facility(content)
        if facility:
            metadata["facility"] = facility
        
        # Extract dates
        dates = self._extract_dates(content)
        if dates:
            metadata["dates"] = dates
            metadata["primary_date"] = dates[0]  # Most recent or first found
        
        # Extract compliance information
        compliance_info = self._extract_compliance_info(content)
        metadata.update(compliance_info)
        
        # Document type specific extraction
        if document_type == DocumentType.UTILITY_BILL:
            metadata.update(self._extract_utility_metadata(content))
        elif document_type == DocumentType.INCIDENT_REPORT:
            metadata.update(self._extract_incident_metadata(content))
        elif document_type == DocumentType.ENVIRONMENTAL_PERMIT:
            metadata.update(self._extract_permit_metadata(content))
        elif document_type == DocumentType.COMPLIANCE_REPORT:
            metadata.update(self._extract_compliance_metadata(content))
        
        # Calculate content metrics
        metadata.update(self._calculate_content_metrics(content))
        
        return metadata
    
    def _extract_facility(self, content: str) -> Optional[str]:
        """Extract facility name from content."""
        for pattern in self.facility_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                facility = match.group(1).strip()
                # Clean up facility name
                facility = re.sub(r'\s+', ' ', facility)
                return facility[:100]  # Limit length
        return None
    
    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates from content."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            dates.extend(matches)
        
        # Remove duplicates and sort
        unique_dates = list(set(dates))
        return unique_dates[:10]  # Limit to first 10 dates
    
    def _extract_compliance_info(self, content: str) -> Dict[str, Any]:
        """Extract compliance-related information."""
        compliance_info = {}
        
        for pattern in self.compliance_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                key = pattern.split('\\s+')[0].replace('\\', '').lower()
                compliance_info[f"{key}_numbers"] = matches[:5]  # Limit to 5 matches
        
        return compliance_info
    
    def _extract_utility_metadata(self, content: str) -> Dict[str, Any]:
        """Extract utility bill specific metadata."""
        metadata = {}
        
        # Extract consumption values
        consumption_data = []
        for pattern in self.consumption_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for value, unit in matches:
                consumption_data.append({
                    "value": float(value),
                    "unit": unit.lower()
                })
        
        if consumption_data:
            metadata["consumption_data"] = consumption_data[:20]  # Limit entries
        
        # Extract utility type
        utility_types = []
        if re.search(r"electric|electricity|kWh", content, re.IGNORECASE):
            utility_types.append("electricity")
        if re.search(r"water|gallon|gal", content, re.IGNORECASE):
            utility_types.append("water")
        if re.search(r"gas|therm|ccf", content, re.IGNORECASE):
            utility_types.append("gas")
        
        if utility_types:
            metadata["utility_types"] = utility_types
        
        return metadata
    
    def _extract_incident_metadata(self, content: str) -> Dict[str, Any]:
        """Extract incident report specific metadata."""
        metadata = {}
        
        # Extract severity indicators
        severity_keywords = {
            "high": ["fatal", "death", "serious", "major", "critical"],
            "medium": ["injury", "accident", "incident", "moderate"],
            "low": ["minor", "near miss", "potential", "observation"]
        }
        
        content_lower = content.lower()
        for severity, keywords in severity_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["severity"] = severity
                break
        
        # Extract injury types
        injury_patterns = [
            r"(cuts?|burns?|bruises?|fractures?|sprains?|lacerations?)",
            r"(head|back|leg|arm|hand|eye)\s+injur[yi]",
        ]
        
        injuries = []
        for pattern in injury_patterns:
            matches = re.findall(pattern, content_lower)
            injuries.extend(matches)
        
        if injuries:
            metadata["injury_types"] = list(set(injuries))[:10]
        
        return metadata
    
    def _extract_permit_metadata(self, content: str) -> Dict[str, Any]:
        """Extract environmental permit specific metadata."""
        metadata = {}
        
        # Extract permit types
        permit_types = []
        permit_keywords = {
            "air": ["air quality", "emission", "stack", "pollutant"],
            "water": ["water discharge", "wastewater", "stormwater", "npdes"],
            "waste": ["waste management", "hazardous waste", "solid waste"],
            "construction": ["construction", "building", "development"]
        }
        
        content_lower = content.lower()
        for permit_type, keywords in permit_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                permit_types.append(permit_type)
        
        if permit_types:
            metadata["permit_types"] = permit_types
        
        # Extract regulatory agency
        agencies = ["epa", "dep", "dnr", "aqmd", "rwqcb"]
        for agency in agencies:
            if agency in content_lower:
                metadata["regulatory_agency"] = agency.upper()
                break
        
        return metadata
    
    def _extract_compliance_metadata(self, content: str) -> Dict[str, Any]:
        """Extract compliance report specific metadata."""
        metadata = {}
        
        # Extract compliance status
        if re.search(r"compliant|compliance|met|satisfied", content, re.IGNORECASE):
            metadata["compliance_status"] = "compliant"
        elif re.search(r"non.?compliant|violation|breach|failed", content, re.IGNORECASE):
            metadata["compliance_status"] = "non_compliant"
        
        # Extract regulation references
        regulation_patterns = [
            r"(CFR\s+\d+\.\d+)",
            r"(40\s+CFR\s+\d+)",
            r"(29\s+CFR\s+\d+)",
            r"(USC\s+\d+)"
        ]
        
        regulations = []
        for pattern in regulation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            regulations.extend(matches)
        
        if regulations:
            metadata["regulations_referenced"] = regulations[:10]
        
        return metadata
    
    def _calculate_content_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate content quality and complexity metrics."""
        return {
            "char_count": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(re.findall(r'[.!?]+', content)),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "has_numbers": bool(re.search(r'\d', content)),
            "has_dates": bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', content)),
            "has_tables": bool(re.search(r'\t.*\t|[\|].*[\|]', content)),
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:16]
        }


class EmbeddingManager:
    """
    Manages embedding generation for EHS documents with multiple model support.
    
    Handles document chunking, metadata extraction, and batch embedding generation
    optimized for EHS document types.
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        config: Optional[Settings] = None
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the embedding model to use
            config: EHS Analytics configuration
        """
        self.model_name = model_name
        self.config = config or Settings()
        self.metadata_extractor = EHSMetadataExtractor()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Define document type specific chunking strategies
        self.chunking_strategies = self._init_chunking_strategies()
        
        logger.info(f"Initialized EmbeddingManager with model: {model_name}")
    
    def _init_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        try:
            if self.model_name.startswith("text-embedding"):
                # OpenAI embeddings
                self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
                self.embedding_dim = 1536 if "ada-002" in self.model_name else 1536
                self.model_type = "openai"
                
            elif self.model_name.startswith("all-"):
                # Sentence Transformers
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.model_type = "sentence_transformer"
                
            else:
                raise ValueError(f"Unsupported embedding model: {self.model_name}")
                
            logger.info(f"Embedding model initialized: {self.model_name} (dim: {self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _init_chunking_strategies(self) -> Dict[DocumentType, ChunkingStrategy]:
        """Initialize document type specific chunking strategies."""
        strategies = {}
        
        # Utility bills - smaller chunks for precise consumption data
        strategies[DocumentType.UTILITY_BILL] = ChunkingStrategy(
            chunk_size=800,
            chunk_overlap=100,
            preserve_tables=True,
            section_markers=["Summary", "Usage", "Charges", "Consumption", "Billing"]
        )
        
        # Environmental permits - larger chunks for context
        strategies[DocumentType.ENVIRONMENTAL_PERMIT] = ChunkingStrategy(
            chunk_size=1500,
            chunk_overlap=300,
            preserve_sections=True,
            section_markers=["Permit", "Conditions", "Requirements", "Limits", "Monitoring"]
        )
        
        # Compliance reports - medium chunks with section preservation
        strategies[DocumentType.COMPLIANCE_REPORT] = ChunkingStrategy(
            chunk_size=1200,
            chunk_overlap=200,
            preserve_sections=True,
            section_markers=["Executive", "Summary", "Findings", "Compliance", "Recommendations"]
        )
        
        # Incident reports - medium chunks preserving narrative flow
        strategies[DocumentType.INCIDENT_REPORT] = ChunkingStrategy(
            chunk_size=1000,
            chunk_overlap=150,
            preserve_sentences=True,
            section_markers=["Incident", "Description", "Analysis", "Actions", "Follow-up"]
        )
        
        # Default strategy for unknown document types
        strategies[DocumentType.UNKNOWN] = ChunkingStrategy(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        return strategies
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        try:
            if self.model_type == "openai":
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
                
            elif self.model_type == "sentence_transformer":
                embedding = self.model.encode(text)
                
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of numpy arrays containing embedding vectors
        """
        try:
            if self.model_type == "openai":
                # OpenAI batch processing
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                embeddings = [np.array(data.embedding) for data in response.data]
                
            elif self.model_type == "sentence_transformer":
                # Sentence Transformers batch processing
                embeddings = self.model.encode(texts)
                embeddings = [emb for emb in embeddings]
                
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def chunk_document(
        self,
        content: str,
        document_type: DocumentType = DocumentType.UNKNOWN,
        custom_strategy: Optional[ChunkingStrategy] = None
    ) -> List[str]:
        """
        Chunk document content using appropriate strategy for document type.
        
        Args:
            content: Document content to chunk
            document_type: Type of EHS document
            custom_strategy: Custom chunking strategy (optional)
            
        Returns:
            List of document chunks
        """
        strategy = custom_strategy or self.chunking_strategies.get(
            document_type, 
            self.chunking_strategies[DocumentType.UNKNOWN]
        )
        
        # Clean content
        cleaned_content = self._clean_content(content)
        
        # Apply chunking strategy
        if strategy.preserve_sections:
            chunks = self._chunk_by_sections(cleaned_content, strategy)
        else:
            chunks = self._chunk_by_size(cleaned_content, strategy)
        
        # Post-process chunks
        processed_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) >= strategy.min_chunk_size:
                processed_chunks.append(chunk.strip())
        
        logger.debug(f"Chunked document into {len(processed_chunks)} chunks using {document_type.value} strategy")
        return processed_chunks
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might interfere with processing
        content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\\@\#\$\%\^\&\*\+\=\|\`\~]', ' ', content)
        
        # Normalize line breaks
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        return content.strip()
    
    def _chunk_by_sections(self, content: str, strategy: ChunkingStrategy) -> List[str]:
        """Chunk content by preserving document sections."""
        chunks = []
        
        # Find section boundaries
        section_pattern = '|'.join(strategy.section_markers)
        sections = re.split(f'({section_pattern})', content, flags=re.IGNORECASE)
        
        current_chunk = ""
        for section in sections:
            # Check if adding this section would exceed chunk size
            if len(current_chunk) + len(section) > strategy.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = section
            else:
                current_chunk += section
        
        # Add remaining content
        if current_chunk:
            chunks.append(current_chunk)
        
        # Further split large sections if needed
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > strategy.max_chunk_size:
                sub_chunks = self._chunk_by_size(chunk, strategy)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _chunk_by_size(self, content: str, strategy: ChunkingStrategy) -> List[str]:
        """Chunk content by size with overlap."""
        chunks = []
        start = 0
        content_length = len(content)
        
        while start < content_length:
            end = start + strategy.chunk_size
            
            # Adjust end to preserve sentences if enabled
            if strategy.preserve_sentences and end < content_length:
                # Find the nearest sentence boundary
                sentence_end = content.rfind('.', start, end)
                if sentence_end > start + strategy.min_chunk_size:
                    end = sentence_end + 1
            
            # Extract chunk
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - strategy.chunk_overlap
        
        return chunks
    
    def process_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        document_type: Union[str, DocumentType] = DocumentType.UNKNOWN
    ) -> List[DocumentChunk]:
        """
        Process a complete document: chunk, extract metadata, and generate embeddings.
        
        Args:
            content: Document content
            metadata: Additional metadata (optional)
            document_id: Unique document identifier
            document_type: Type of EHS document
            
        Returns:
            List of DocumentChunk objects with embeddings and metadata
        """
        try:
            # Convert document type if string
            if isinstance(document_type, str):
                try:
                    document_type = DocumentType(document_type)
                except ValueError:
                    document_type = DocumentType.UNKNOWN
            
            # Generate document ID if not provided
            if not document_id:
                document_id = str(uuid.uuid4())
            
            # Extract document-level metadata
            extracted_metadata = self.metadata_extractor.extract_metadata(content, document_type)
            if metadata:
                extracted_metadata.update(metadata)
            
            # Chunk the document
            text_chunks = self.chunk_document(content, document_type)
            
            # Generate embeddings for all chunks
            embeddings = self.generate_embeddings_batch(text_chunks)
            
            # Create DocumentChunk objects
            document_chunks = []
            for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Calculate chunk position in original content
                start_char = content.find(text[:50])  # Approximate start position
                end_char = start_char + len(text) if start_char >= 0 else len(text)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=text,
                    embedding=embedding,
                    metadata={
                        **extracted_metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "document_id": document_id,
                        "processing_timestamp": datetime.now().isoformat()
                    },
                    document_id=document_id,
                    document_type=document_type,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=end_char
                )
                document_chunks.append(chunk)
            
            logger.info(f"Processed document {document_id}: {len(document_chunks)} chunks with embeddings")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "embedding_dimension": self.embedding_dim,
            "supported_document_types": [dt.value for dt in DocumentType],
            "chunking_strategies": {
                dt.value: {
                    "chunk_size": strategy.chunk_size,
                    "chunk_overlap": strategy.chunk_overlap,
                    "preserve_sections": strategy.preserve_sections
                }
                for dt, strategy in self.chunking_strategies.items()
            }
        }