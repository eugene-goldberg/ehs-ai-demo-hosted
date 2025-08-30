"""
LlamaParse integration for EHS document parsing.
Handles utility bills, permits, invoices, and other EHS documents.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser

# Import transcript logging utilities
try:
    from utils.transcript_forwarder import forward_transcript_entry
except ImportError:
    # If import fails, create a no-op function
    def forward_transcript_entry(role, content, context=None):
        pass
from langchain.text_splitter import TokenTextSplitter

# Apply nest_asyncio to handle async in notebooks/existing event loops
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class EHSDocumentParser:
    """
    Enhanced document parser for EHS-specific documents using LlamaParse.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the EHS document parser.
        
        Args:
            api_key: LlamaParse API key. If not provided, reads from environment.
        """
        self.api_key = api_key or os.getenv("LLAMA_PARSE_API_KEY")
        if not self.api_key:
            raise ValueError("LlamaParse API key not provided")
        
        # Document type specific parsing instructions
        self.parsing_instructions = {
            "utility_bill": """
                Extract the following information:
                - Billing period (start and end dates)
                - Total energy consumption (kWh)
                - Peak demand (kW)
                - Total cost
                - Rate structure
                - All line items in tables
                - Meter readings
                - Carbon emissions if mentioned
                Preserve all tabular data in markdown format.
            """,
            "water_bill": """
                Extract the following information from this water utility bill:
                - Account number and service address
                - Customer name and billing address (billed to)
                - Facility name and address (service location)
                - Water utility provider information
                - Billing period (start and end dates)
                - Statement date and due date
                - Water consumption in gallons (and cubic meters if provided)
                - Meter readings (previous, current, usage)
                - All charges (water consumption, sewer service, stormwater fees, conservation tax, infrastructure surcharge)
                - Rate information (cost per gallon or per unit)
                - Total amount due
                Preserve all tabular data in markdown format.
            """,
            "waste_manifest": """
                Extract the following information from this waste manifest:
                - Manifest tracking number and type (hazardous/non-hazardous)
                - Issue date and document status
                - Generator information (company name, EPA ID, contact person, phone, address)
                - Transporter information (company name, EPA ID, vehicle ID, driver name, driver license)
                - Receiving facility information (facility name, EPA ID, contact person, phone, address)
                - Waste items with descriptions, container types, quantities, units, and classifications
                - All certification dates and signatures (generator, transporter, facility)
                - Special handling instructions
                - Total waste quantity and unit
                Preserve all tabular data in markdown format.
            """,
            "permit": """
                Extract the following information:
                - Permit number
                - Issue date and expiry date
                - Facility/location information
                - Permitted activities
                - Compliance requirements
                - Emission limits
                - Monitoring requirements
                - Special conditions
            """,
            "invoice": """
                Extract the following information:
                - Invoice number and date
                - Vendor information
                - All line items with quantities and costs
                - Environmental product categories
                - Waste disposal fees
                - Recycling information
                - Total amounts
            """,
            "equipment_spec": """
                Extract the following information:
                - Equipment model and manufacturer
                - Energy efficiency ratings
                - Operating parameters
                - Emission factors
                - Maintenance requirements
                - Safety specifications
            """,
            "default": """
                Extract all structured data including:
                - Tables with numerical data
                - Dates and time periods
                - Quantities and measurements
                - Costs and financial information
                - Compliance-related information
                - Environmental metrics
            """
        }
        
    def detect_document_type(self, file_path: str) -> str:
        """
        Detect document type based on filename and content patterns.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document type string
        """
        filename = Path(file_path).name.lower()
        
        # Simple pattern matching for now
        if "waste" in filename and "manifest" in filename:
            return "waste_manifest"
        elif "water" in filename:
            return "water_bill"
        elif any(term in filename for term in ["utility", "electric", "gas", "energy"]):
            return "utility_bill"
        elif any(term in filename for term in ["permit", "license", "authorization"]):
            return "permit"
        elif any(term in filename for term in ["invoice", "bill", "receipt"]):
            return "invoice"
        elif any(term in filename for term in ["equipment", "spec", "datasheet"]):
            return "equipment_spec"
        else:
            return "default"
    
    def parse_document(
        self, 
        file_path: str, 
        document_type: Optional[str] = None,
        custom_instruction: Optional[str] = None
    ) -> List[Document]:
        """
        Parse a document using LlamaParse with EHS-specific instructions.
        
        Args:
            file_path: Path to the document
            document_type: Type of document (auto-detected if not provided)
            custom_instruction: Custom parsing instruction to override defaults
            
        Returns:
            List of parsed Document objects
        """
        # Auto-detect document type if not provided
        if not document_type:
            document_type = self.detect_document_type(file_path)
            logger.info(f"Auto-detected document type: {document_type}")
        
        # Get parsing instruction
        parsing_instruction = custom_instruction or self.parsing_instructions.get(
            document_type, 
            self.parsing_instructions["default"]
        )
        
        try:
            # Initialize parser with specific instructions
            parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                parsing_instruction=parsing_instruction,
                verbose=True,
                invalidate_cache=True,  # Always get fresh parse for accuracy
                do_not_unroll_columns=False,  # Unroll columns for better table extraction
                page_separator="\\n---\\n"  # Clear page separation
            )
            
            logger.info(f"Parsing document: {file_path}")
            
            # Log LlamaParse request
            try:
                forward_transcript_entry(
                    role="user",
                    content=f"LlamaParse Document Request:\nFile: {file_path}\nDocument Type: {document_type}\nParsing Instructions: {parsing_instruction[:500]}... (truncated)",
                    context={
                        "component": "llama_parser",
                        "function": "parse_document",
                        "document_type": document_type,
                        "file_path": str(file_path),
                        "timestamp": ""
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to forward LlamaParse request transcript: {e}")
            
            documents = parser.load_data(file_path)
            
            # Log LlamaParse response
            try:
                doc_summary = f"Parsed {len(documents)} pages from {document_type} document"
                if documents:
                    doc_summary += f"\nFirst page preview: {documents[0].text[:200]}..."
                    
                forward_transcript_entry(
                    role="assistant",
                    content=doc_summary,
                    context={
                        "component": "llama_parser",
                        "function": "parse_document",
                        "document_type": document_type,
                        "page_count": len(documents),
                        "timestamp": ""
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to forward LlamaParse response transcript: {e}")
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": file_path,
                    "document_type": document_type,
                    "parser": "llamaparse"
                })
            
            logger.info(f"Successfully parsed {len(documents)} pages")
            return documents
            
        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {str(e)}")
            raise
    
    def parse_batch(
        self, 
        file_paths: List[str], 
        document_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[Document]]:
        """
        Parse multiple documents in batch.
        
        Args:
            file_paths: List of file paths to parse
            document_types: Optional mapping of file paths to document types
            
        Returns:
            Dictionary mapping file paths to parsed documents
        """
        results = {}
        document_types = document_types or {}
        
        for file_path in file_paths:
            try:
                doc_type = document_types.get(file_path)
                documents = self.parse_document(file_path, doc_type)
                results[file_path] = documents
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {str(e)}")
                results[file_path] = []
        
        return results
    
    def extract_tables(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract tables from parsed documents.
        
        Args:
            documents: List of parsed documents
            
        Returns:
            List of extracted tables with metadata
        """
        tables = []
        
        for doc in documents:
            content = doc.get_content()
            # Simple table extraction from markdown
            # In production, use more sophisticated table parsing
            lines = content.split('\n')
            in_table = False
            current_table = []
            
            for line in lines:
                if '|' in line and not line.strip().startswith('```'):
                    in_table = True
                    current_table.append(line)
                elif in_table and '|' not in line:
                    if current_table:
                        tables.append({
                            "content": '\n'.join(current_table),
                            "source": doc.metadata.get("source", ""),
                            "page": doc.metadata.get("page", 0)
                        })
                    in_table = False
                    current_table = []
            
            # Don't forget the last table
            if current_table:
                tables.append({
                    "content": '\n'.join(current_table),
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", 0)
                })
        
        return tables
    
    def create_chunks(
        self, 
        documents: List[Document], 
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        Create optimized chunks for EHS documents.
        
        Args:
            documents: List of parsed documents
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked documents
        """
        # Use token-based splitting for consistent chunk sizes
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunked_docs = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.get_content())
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_docs.append(chunk_doc)
        
        return chunked_docs


# Utility functions for common operations
def parse_utility_bill(file_path: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a utility bill and extract key metrics.
    
    Args:
        file_path: Path to utility bill PDF
        api_key: Optional LlamaParse API key
        
    Returns:
        Dictionary containing extracted metrics
    """
    parser = EHSDocumentParser(api_key)
    documents = parser.parse_document(file_path, document_type="utility_bill")
    
    # Extract tables
    tables = parser.extract_tables(documents)
    
    # TODO: Implement specific extraction logic for utility bills
    # This would use an LLM to extract structured data from the parsed content
    
    return {
        "documents": documents,
        "tables": tables,
        "document_type": "utility_bill"
    }


def parse_permit(file_path: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a permit document and extract compliance requirements.
    
    Args:
        file_path: Path to permit PDF
        api_key: Optional LlamaParse API key
        
    Returns:
        Dictionary containing extracted permit information
    """
    parser = EHSDocumentParser(api_key)
    documents = parser.parse_document(file_path, document_type="permit")
    
    return {
        "documents": documents,
        "document_type": "permit"
    }