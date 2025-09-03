#!/usr/bin/env python3
"""
Batch ingestion runner for EHS AI Demo
Run this from the backend directory to process documents
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Also add the src directory to Python path
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Set up environment
os.environ.setdefault('PYTHONPATH', f"{str(current_dir)}:{str(src_dir)}")

from ehs_workflows.ingestion_workflow import IngestionWorkflow
# from ingestion.database.document_store import DocumentStore
import logging
from dotenv import load_dotenv

async def main():
    """Main function to run batch ingestion"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('batch_ingestion')

    # Load environment variables
    load_dotenv(current_dir / ".env")
    
    # Initialize document store
    # doc_store = DocumentStore()
    
    # Initialize the original workflow (which works properly)
    workflow = IngestionWorkflow(
        llama_parse_api_key=os.getenv("LLAMA_PARSE_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Define input and output directories
    input_dir = Path('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data')
    output_dir = current_dir / 'data' / 'processed'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting batch ingestion from {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Check if input directory exists and has files
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return
            
        # Get all PDF files from input directory
        pdf_files = list(input_dir.glob('*.pdf'))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF file
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            
            try:
                # Run the integrated workflow with risk assessment
                result = workflow.process_document(
                    file_path=str(pdf_file),
                    document_id=pdf_file.stem,  # Use filename without extension as document ID
                    metadata={
                        "output_dir": str(output_dir),
                        "original_filename": pdf_file.name,
                        "batch_processing": True
                    }
                )
                
                if result:
                    logger.info(f"Successfully processed {pdf_file.name}")
                    logger.info(f"Document ID: {result.get('document_id', 'N/A')}")
                    logger.info(f"Risk score: {result.get('risk_score', 'N/A')}")
                else:
                    logger.error(f"Failed to process {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}", exc_info=True)
                continue
                
        logger.info("Batch ingestion completed")
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())