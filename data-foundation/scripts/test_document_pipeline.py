#!/usr/bin/env python3
"""
Test script for the EHS document processing pipeline.
Tests the full pipeline with a real electric bill PDF.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.src.parsers.llama_parser import EHSDocumentParser, parse_utility_bill
from backend.src.indexing.document_indexer import EHSDocumentIndexer
from backend.src.workflows.ingestion_workflow import IngestionWorkflow
from backend.src.extractors.ehs_extractors import UtilityBillExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_llamaparse_only():
    """Test just the LlamaParse component."""
    logger.info("=" * 50)
    logger.info("Testing LlamaParse Component")
    logger.info("=" * 50)
    
    # Get API key
    llama_parse_key = os.getenv("LLAMA_PARSE_API_KEY")
    if not llama_parse_key:
        logger.error("LLAMA_PARSE_API_KEY not set!")
        return None
    
    try:
        # Initialize parser
        parser = EHSDocumentParser(api_key=llama_parse_key)
        
        # Parse the electric bill
        file_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/electric_bill.pdf"
        logger.info(f"Parsing document: {file_path}")
        
        documents = parser.parse_document(
            file_path=file_path,
            document_type="utility_bill"
        )
        
        logger.info(f"Parsed {len(documents)} pages")
        
        # Extract tables
        tables = parser.extract_tables(documents)
        logger.info(f"Extracted {len(tables)} tables")
        
        # Print first page content (truncated)
        if documents:
            content = documents[0].get_content()
            logger.info("\nFirst page content (first 500 chars):")
            logger.info(content[:500] + "...")
        
        # Print first table if found
        if tables:
            logger.info("\nFirst table found:")
            logger.info(tables[0]['content'])
        
        return documents
        
    except Exception as e:
        logger.error(f"Error in LlamaParse test: {str(e)}")
        return None


def test_extraction_only(documents):
    """Test just the extraction component."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Data Extraction Component")
    logger.info("=" * 50)
    
    if not documents:
        logger.error("No documents to extract from!")
        return None
    
    # Get LLM API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY not set!")
        return None
    
    try:
        # Initialize extractor
        extractor = UtilityBillExtractor(llm_model="gpt-4")
        
        # Combine document content
        full_content = "\n".join([doc.get_content() for doc in documents])
        
        logger.info("Extracting structured data from utility bill...")
        
        # Extract data
        extracted_data = extractor.extract(
            content=full_content,
            metadata={"source": "electric_bill.pdf"}
        )
        
        logger.info("\nExtracted Data:")
        logger.info(json.dumps(extracted_data, indent=2, default=str))
        
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error in extraction test: {str(e)}")
        return None


def test_full_workflow():
    """Test the complete workflow."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Full Document Processing Workflow")
    logger.info("=" * 50)
    
    # Check environment variables
    required_vars = ["LLAMA_PARSE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return
    
    # Neo4j connection details
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "EhsAI2024!"
    
    try:
        # Initialize workflow
        workflow = IngestionWorkflow(
            llama_parse_api_key=os.getenv("LLAMA_PARSE_API_KEY"),
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            llm_model="gpt-4"
        )
        
        # Process document
        file_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/electric_bill.pdf"
        document_id = f"electric_bill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Processing document: {file_path}")
        logger.info(f"Document ID: {document_id}")
        
        # Run workflow
        result = workflow.process_document(
            file_path=file_path,
            document_id=document_id,
            metadata={
                "upload_date": datetime.now().isoformat(),
                "document_source": "test_script",
                "facility": "Test Facility"
            }
        )
        
        # Log results
        logger.info(f"\nWorkflow Status: {result['status']}")
        logger.info(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
        
        if result['errors']:
            logger.error(f"Errors encountered: {result['errors']}")
        
        if result['extracted_data']:
            logger.info("\nExtracted Data Summary:")
            data = result['extracted_data']
            logger.info(f"  Billing Period: {data.get('billing_period_start')} to {data.get('billing_period_end')}")
            logger.info(f"  Total kWh: {data.get('total_kwh')}")
            logger.info(f"  Total Cost: ${data.get('total_cost')}")
            logger.info(f"  Peak Demand: {data.get('peak_demand_kw')} kW")
        
        if result['neo4j_nodes']:
            logger.info(f"\nCreated {len(result['neo4j_nodes'])} Neo4j nodes")
            logger.info(f"Created {len(result.get('neo4j_relationships', []))} relationships")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in workflow test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    logger.info("Starting EHS Document Pipeline Tests")
    logger.info(f"Test started at: {datetime.now()}")
    
    # Test 1: LlamaParse only
    documents = test_llamaparse_only()
    
    if documents:
        # Test 2: Extraction only
        extracted_data = test_extraction_only(documents)
    
    # Test 3: Full workflow
    result = test_full_workflow()
    
    logger.info(f"\nTest completed at: {datetime.now()}")


if __name__ == "__main__":
    main()