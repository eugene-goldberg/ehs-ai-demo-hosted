#!/usr/bin/env python3
"""
Integration test for transcript flow between web app upload and transcript retrieval.

This test verifies the complete flow:
1. Clear existing transcript data
2. Upload a document to web app API (port 8001) using multipart/form-data
3. Wait for processing
4. Fetch transcript from web app backend (port 8001)
5. Verify entries were captured
"""

import requests
import json
import time
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API URLs
INGESTION_API_URL = "http://localhost:8000"
WEB_APP_API_URL = "http://localhost:8001"

def test_transcript_flow():
    """Test the complete transcript flow from upload to web app."""
    
    try:
        # Step 1: Clear existing transcript data
        logger.info("Step 1: Clearing existing transcript data...")
        clear_response = requests.delete(f"{WEB_APP_API_URL}/api/data/transcript")
        
        if clear_response.status_code == 200:
            logger.info("✓ Transcript cleared successfully")
        else:
            logger.warning(f"Clear transcript returned status {clear_response.status_code}: {clear_response.text}")
        
        # Step 2: Find and upload test document
        logger.info("Step 2: Looking for test document (electric_bill.pdf)...")
        
        # Look for the test file in common locations
        test_file_paths = [
            "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tests/electric_bill.pdf",
            "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/electric_bill.pdf",
            "/Users/eugene/dev/ai/agentos/ehs-ai-demo/electric_bill.pdf",
            "./electric_bill.pdf",
            "./tests/electric_bill.pdf"
        ]
        
        test_file_path = None
        for path in test_file_paths:
            if os.path.exists(path):
                test_file_path = path
                logger.info(f"✓ Found test file at: {path}")
                break
        
        if not test_file_path:
            logger.error("✗ Test file 'electric_bill.pdf' not found in any expected location")
            logger.error("Expected locations:")
            for path in test_file_paths:
                logger.error(f"  - {path}")
            return False
        
        # Upload the document using multipart/form-data
        logger.info(f"Step 3: Uploading document to web app API using multipart/form-data...")
        
        # Upload using files parameter with binary mode
        with open(test_file_path, 'rb') as f:
            files = {'file': ('electric_bill.pdf', f, 'application/pdf')}
            
            upload_response = requests.post(
                f"{WEB_APP_API_URL}/api/data/upload",
                files=files,
                timeout=30
            )
        
        if upload_response.status_code == 200:
            upload_data = upload_response.json()
            logger.info(f"✓ Document uploaded successfully: {upload_data}")
        else:
            logger.error(f"✗ Upload failed with status {upload_response.status_code}: {upload_response.text}")
            return False
        
        # Step 4: Wait for processing
        logger.info("Step 4: Waiting for document processing...")
        time.sleep(5)  # Wait 5 seconds for processing
        
        # Step 5: Fetch transcript from web app backend
        logger.info("Step 5: Fetching transcript from web app backend...")
        
        transcript_response = requests.get(f"{WEB_APP_API_URL}/api/data/transcript")
        
        if transcript_response.status_code == 200:
            transcript_data = transcript_response.json()
            logger.info(f"✓ Transcript fetched successfully")
            
            # Step 6: Verify and display entries
            if isinstance(transcript_data, list) and len(transcript_data) > 0:
                logger.info(f"✓ Found {len(transcript_data)} transcript entries:")
                
                for i, entry in enumerate(transcript_data, 1):
                    logger.info(f"\n--- Entry {i} ---")
                    logger.info(f"Timestamp: {entry.get('timestamp', 'N/A')}")
                    logger.info(f"Action: {entry.get('action', 'N/A')}")
                    logger.info(f"Details: {entry.get('details', 'N/A')}")
                    logger.info(f"Status: {entry.get('status', 'N/A')}")
                    if 'metadata' in entry:
                        logger.info(f"Metadata: {json.dumps(entry['metadata'], indent=2)}")
                
                logger.info("\n✓ Integration test completed successfully!")
                return True
            else:
                logger.warning("✗ No transcript entries found after processing")
                logger.warning(f"Transcript response: {transcript_data}")
                return False
        else:
            logger.error(f"✗ Failed to fetch transcript with status {transcript_response.status_code}: {transcript_response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"✗ Connection error: {e}")
        logger.error("Make sure both APIs are running:")
        logger.error(f"  - Ingestion API: {INGESTION_API_URL}")
        logger.error(f"  - Web App API: {WEB_APP_API_URL}")
        return False
    except requests.exceptions.Timeout as e:
        logger.error(f"✗ Request timeout: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False

def check_api_health():
    """Check if both APIs are running and healthy."""
    logger.info("Checking API health...")
    
    try:
        # Check ingestion API
        ingestion_response = requests.get(f"{INGESTION_API_URL}/health", timeout=5)
        if ingestion_response.status_code == 200:
            logger.info("✓ Ingestion API (8000) is healthy")
        else:
            logger.warning(f"Ingestion API health check returned {ingestion_response.status_code}")
    except Exception as e:
        logger.error(f"✗ Ingestion API (8000) is not responding: {e}")
        return False
    
    try:
        # Check web app API
        webapp_response = requests.get(f"{WEB_APP_API_URL}/health", timeout=5)
        if webapp_response.status_code == 200:
            logger.info("✓ Web App API (8001) is healthy")
        else:
            logger.warning(f"Web App API health check returned {webapp_response.status_code}")
    except Exception as e:
        logger.error(f"✗ Web App API (8001) is not responding: {e}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting transcript integration test...")
    logger.info("="*50)
    
    # First check if APIs are healthy
    if not check_api_health():
        logger.error("API health check failed. Please ensure both services are running:")
        logger.error("  python3 app.py (ingestion API on port 8000)")
        logger.error("  python3 web_app.py (web app on port 8001)")
        exit(1)
    
    logger.info("\nRunning transcript flow test...")
    logger.info("-"*30)
    
    success = test_transcript_flow()
    
    logger.info("\n" + "="*50)
    if success:
        logger.info("🎉 INTEGRATION TEST PASSED!")
        exit(0)
    else:
        logger.error("❌ INTEGRATION TEST FAILED!")
        exit(1)