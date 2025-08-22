#!/usr/bin/env python3
"""
Simple test script to directly test waste manifest extraction without the full workflow.
This script will:
1. Load environment variables
2. Parse the waste_manifest.pdf using LlamaParse
3. Extract data using WasteManifestExtractor
4. Print the full extracted data as JSON
5. Show which fields are present and which are missing
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from backend/.env
env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path, override=True)

from llama_parse import LlamaParse
from backend.src.extractors.ehs_extractors import WasteManifestExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_direct_waste_extraction():
    """Test direct waste manifest extraction."""
    logger.info("=" * 80)
    logger.info("Direct Waste Manifest Extraction Test")
    logger.info("=" * 80)
    
    # Document path
    pdf_path = Path(__file__).parent.parent / "data" / "waste_manifest.pdf"
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    try:
        # Step 1: Parse PDF with LlamaParse
        logger.info("Step 1: Parsing PDF with LlamaParse...")
        
        # Get API key (trying both possible names)
        llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_PARSE_API_KEY")
        if not llama_cloud_api_key:
            logger.error("LLAMA_CLOUD_API_KEY or LLAMA_PARSE_API_KEY not found in environment variables")
            return
        
        # Initialize LlamaParse
        parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="text"
        )
        
        # Parse the document
        logger.info(f"Parsing document: {pdf_path}")
        documents = parser.load_data(str(pdf_path))
        
        if not documents:
            logger.error("No documents returned from LlamaParse")
            return
        
        # Extract text content
        document_text = documents[0].text
        logger.info(f"Extracted text length: {len(document_text)} characters")
        
        # Step 2: Extract data using WasteManifestExtractor
        logger.info("\nStep 2: Extracting data with WasteManifestExtractor...")
        
        # Initialize extractor with GPT-4
        extractor = WasteManifestExtractor(llm_model="gpt-4")
        
        # Extract data
        extracted_data = extractor.extract(document_text)
        
        # Step 3: Print full extracted data as JSON
        logger.info("\nStep 3: Full Extracted Data (JSON):")
        logger.info("=" * 50)
        print(json.dumps(extracted_data, indent=2, default=str))
        
        # Step 4: Analyze field completeness
        logger.info("\nStep 4: Field Completeness Analysis:")
        logger.info("=" * 50)
        
        # Define expected fields from WasteManifestData model
        expected_fields = {
            # Manifest Information
            "manifest_tracking_number": "Manifest tracking number",
            "manifest_type": "Type of waste manifest",
            "issue_date": "Date manifest was issued",
            "document_status": "Document status",
            
            # Generator Information
            "generator_name": "Waste generator company name",
            "generator_epa_id": "Generator EPA ID number",
            "generator_contact_person": "Generator contact person",
            "generator_phone": "Generator phone number",
            "generator_address": "Generator company address",
            
            # Transporter Information
            "transporter_name": "Waste transporter company name",
            "transporter_epa_id": "Transporter EPA ID number",
            "vehicle_id": "Transport vehicle identification",
            "driver_name": "Driver name",
            "driver_license": "Driver license number",
            
            # Receiving Facility Information
            "facility_name": "Receiving facility name",
            "facility_epa_id": "Facility EPA ID number",
            "facility_contact_person": "Facility contact person",
            "facility_phone": "Facility phone number",
            "facility_address": "Facility address",
            
            # Waste Line Items
            "waste_items": "List of waste items",
            
            # Certifications
            "generator_certification_date": "Generator certification date",
            "generator_signature": "Generator signature/name",
            "transporter_acknowledgment_date": "Transporter acknowledgment date",
            "transporter_signature": "Transporter signature/name",
            "facility_certification_date": "Facility certification date",
            "facility_signature": "Facility signature/name",
            
            # Special Handling
            "special_handling_instructions": "Special handling instructions",
            
            # Calculated Fields
            "total_waste_quantity": "Total weight of all waste items",
            "total_waste_unit": "Unit for total waste quantity"
        }
        
        # Count present and missing fields
        present_fields = []
        missing_fields = []
        
        for field, description in expected_fields.items():
            value = extracted_data.get(field)
            if value is not None and value != "" and value != []:
                present_fields.append((field, description, value))
            else:
                missing_fields.append((field, description))
        
        # Print present fields
        logger.info(f"PRESENT FIELDS ({len(present_fields)}/{len(expected_fields)}):")
        for field, description, value in present_fields:
            if isinstance(value, list):
                logger.info(f"  ✓ {field}: {description} ({len(value)} items)")
            elif isinstance(value, str) and len(value) > 50:
                logger.info(f"  ✓ {field}: {description} ({len(value)} chars)")
            else:
                logger.info(f"  ✓ {field}: {description} = {value}")
        
        # Print missing fields
        logger.info(f"\nMISSING FIELDS ({len(missing_fields)}/{len(expected_fields)}):")
        for field, description in missing_fields:
            logger.info(f"  ✗ {field}: {description}")
        
        # Special analysis for waste_items
        if extracted_data.get("waste_items"):
            logger.info(f"\nWASTE ITEMS ANALYSIS:")
            logger.info("=" * 30)
            for i, item in enumerate(extracted_data["waste_items"], 1):
                logger.info(f"Item {i}:")
                for key, value in item.items():
                    logger.info(f"  {key}: {value}")
                logger.info("")
        
        # Step 5: Validation insights
        logger.info("\nStep 5: Validation Insights:")
        logger.info("=" * 50)
        
        # Check critical fields for Neo4j validation
        critical_fields = [
            "manifest_tracking_number",
            "generator_name", 
            "generator_epa_id",
            "facility_name",
            "facility_epa_id",
            "waste_items"
        ]
        
        missing_critical = []
        for field in critical_fields:
            if not extracted_data.get(field):
                missing_critical.append(field)
        
        if missing_critical:
            logger.warning(f"Missing critical fields for validation: {missing_critical}")
            logger.warning("These fields are likely required for successful Neo4j ingestion")
        else:
            logger.info("All critical fields are present")
        
        # Check waste items structure
        if extracted_data.get("waste_items"):
            waste_items = extracted_data["waste_items"]
            logger.info(f"Waste items count: {len(waste_items)}")
            
            for i, item in enumerate(waste_items):
                required_item_fields = ["description", "container_type", "container_quantity"]
                missing_item_fields = [f for f in required_item_fields if not item.get(f)]
                if missing_item_fields:
                    logger.warning(f"Waste item {i+1} missing: {missing_item_fields}")
        
        # Summary
        completion_rate = len(present_fields) / len(expected_fields) * 100
        logger.info(f"\nExtraction completion rate: {completion_rate:.1f}%")
        
        if completion_rate >= 80:
            logger.info("✓ Good extraction rate - likely sufficient for validation")
        elif completion_rate >= 60:
            logger.warning("⚠ Moderate extraction rate - some validation issues possible")
        else:
            logger.error("✗ Low extraction rate - likely validation failures")
        
    except Exception as e:
        logger.error(f"Extraction test failed: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("Direct Waste Extraction Test Completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_direct_waste_extraction()