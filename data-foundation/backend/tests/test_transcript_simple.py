"""
Simplified transcript functionality test script for the web app backend (port 8001).

This test script focuses exclusively on testing the transcript functionality of the web app
backend running on port 8001. It tests:
1. Adding transcript entries manually via the transcript/add endpoint
2. Fetching and displaying the transcript
3. Clearing the transcript
4. Basic error handling

This script does NOT involve file uploads or the ingestion API.
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/test_transcript_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
WEB_APP_BASE_URL = "http://localhost:8001"
TRANSCRIPT_ADD_ENDPOINT = f"{WEB_APP_BASE_URL}/api/data/transcript/add"
TRANSCRIPT_GET_ENDPOINT = f"{WEB_APP_BASE_URL}/api/data/transcript"
TRANSCRIPT_CLEAR_ENDPOINT = f"{WEB_APP_BASE_URL}/api/data/transcript"

class TranscriptTester:
    """Test class for transcript functionality."""
    
    def __init__(self):
        """Initialize the transcript tester."""
        self.session = requests.Session()
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        logger.info("TranscriptTester initialized")
    
    def check_server_health(self) -> bool:
        """Check if the web app server is running and accessible."""
        try:
            logger.info(f"Checking server health at {WEB_APP_BASE_URL}")
            response = self.session.get(f"{WEB_APP_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is healthy and accessible")
                return True
            else:
                logger.error(f"❌ Server health check failed with status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Server health check failed: {str(e)}")
            return False
    
    def add_transcript_entry(self, content: str, entry_type: str = "user_message") -> bool:
        """
        Add a single transcript entry via the API.
        
        Args:
            content: The content of the transcript entry
            entry_type: Type of entry (e.g., "user_message", "assistant_response", "system")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Map entry types to proper role values
            role_mapping = {
                "user_message": "user",
                "assistant_response": "assistant",
                "test_entry": "user",
                "test_clear": "user",
                "system": "system"
            }
            
            role = role_mapping.get(entry_type, "user")
            
            payload = {
                "content": content,
                "role": role,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Adding transcript entry: {role} - {content[:50]}...")
            response = self.session.post(
                TRANSCRIPT_ADD_ENDPOINT,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info("✅ Transcript entry added successfully")
                return True
            else:
                logger.error(f"❌ Failed to add transcript entry. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error adding transcript entry: {str(e)}")
            return False
    
    def get_transcript(self) -> Optional[Dict[Any, Any]]:
        """
        Fetch the current transcript from the API.
        
        Returns:
            Dict containing transcript data or None if failed
        """
        try:
            logger.info("Fetching transcript from API")
            response = self.session.get(TRANSCRIPT_GET_ENDPOINT, timeout=10)
            
            if response.status_code == 200:
                transcript_data = response.json()
                logger.info(f"✅ Successfully fetched transcript with {len(transcript_data.get('entries', []))} entries")
                return transcript_data
            else:
                logger.error(f"❌ Failed to fetch transcript. Status: {response.status_code}, Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching transcript: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing transcript JSON: {str(e)}")
            return None
    
    def clear_transcript(self) -> bool:
        """
        Clear the transcript via the API using DELETE method.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Clearing transcript")
            response = self.session.delete(TRANSCRIPT_CLEAR_ENDPOINT, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ Transcript cleared successfully")
                return True
            else:
                logger.error(f"❌ Failed to clear transcript. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error clearing transcript: {str(e)}")
            return False
    
    def display_transcript(self, transcript_data: Dict[Any, Any]) -> None:
        """
        Display the transcript in a readable format.
        
        Args:
            transcript_data: The transcript data dictionary
        """
        logger.info("=== TRANSCRIPT DISPLAY ===")
        
        if not transcript_data or not transcript_data.get('entries'):
            logger.info("📝 Transcript is empty")
            return
        
        entries = transcript_data.get('entries', [])
        logger.info(f"📝 Transcript contains {len(entries)} entries:")
        
        for i, entry in enumerate(entries, 1):
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            timestamp = entry.get('timestamp', 'no timestamp')
            
            logger.info(f"  [{i}] {role.upper()}: {content}")
            logger.info(f"      Timestamp: {timestamp}")
            logger.info("")
        
        logger.info("=== END TRANSCRIPT ===")
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """
        Run a single test and track results.
        
        Args:
            test_name: Name of the test
            test_func: Function to execute
            *args, **kwargs: Arguments for the test function
            
        Returns:
            bool: True if test passed, False otherwise
        """
        self.test_results["total_tests"] += 1
        logger.info(f"🧪 Running test: {test_name}")
        
        try:
            result = test_func(*args, **kwargs)
            if result:
                self.test_results["passed"] += 1
                logger.info(f"✅ Test PASSED: {test_name}")
                return True
            else:
                self.test_results["failed"] += 1
                self.test_results["errors"].append(f"Test failed: {test_name}")
                logger.error(f"❌ Test FAILED: {test_name}")
                return False
        except Exception as e:
            self.test_results["failed"] += 1
            error_msg = f"Test error in {test_name}: {str(e)}"
            self.test_results["errors"].append(error_msg)
            logger.error(f"❌ Test ERROR: {error_msg}")
            return False
    
    def test_add_multiple_entries(self) -> bool:
        """Test adding multiple transcript entries."""
        test_entries = [
            {"content": "Hello, I need help with EHS data extraction", "entry_type": "user_message"},
            {"content": "I can help you with that. What specific data are you looking for?", "entry_type": "assistant_response"},
            {"content": "I need electrical consumption data for facility FAC-001", "entry_type": "user_message"},
            {"content": "Let me extract that data for you", "entry_type": "assistant_response"},
            {"content": "Data extraction completed successfully", "entry_type": "system"}
        ]
        
        success_count = 0
        for entry in test_entries:
            if self.add_transcript_entry(entry["content"], entry["entry_type"]):
                success_count += 1
            time.sleep(0.5)  # Small delay between additions
        
        logger.info(f"Added {success_count}/{len(test_entries)} transcript entries")
        return success_count == len(test_entries)
    
    def test_fetch_transcript(self) -> bool:
        """Test fetching the transcript."""
        transcript_data = self.get_transcript()
        if transcript_data is not None:
            self.display_transcript(transcript_data)
            return True
        return False
    
    def test_transcript_persistence(self) -> bool:
        """Test that transcript entries persist correctly."""
        # Add a specific test entry
        test_content = f"Test persistence entry - {datetime.now().isoformat()}"
        
        if not self.add_transcript_entry(test_content, "test_entry"):
            return False
        
        # Fetch transcript and verify the entry exists
        transcript_data = self.get_transcript()
        if not transcript_data:
            return False
        
        entries = transcript_data.get('entries', [])
        for entry in entries:
            if entry.get('content') == test_content and entry.get('role') == 'user':
                logger.info("✅ Transcript persistence verified")
                return True
        
        logger.error("❌ Test entry not found in transcript")
        return False
    
    def test_clear_functionality(self) -> bool:
        """Test the transcript clear functionality."""
        # First ensure there's something to clear
        if not self.add_transcript_entry("Entry to be cleared", "test_clear"):
            return False
        
        # Clear the transcript
        if not self.clear_transcript():
            return False
        
        # Verify it's empty
        transcript_data = self.get_transcript()
        if transcript_data is None:
            return False
        
        entries = transcript_data.get('entries', [])
        if len(entries) == 0:
            logger.info("✅ Transcript successfully cleared")
            return True
        else:
            logger.error(f"❌ Transcript not fully cleared, {len(entries)} entries remain")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling scenarios."""
        # Test with invalid JSON payload
        try:
            logger.info("Testing error handling with invalid payload")
            response = self.session.post(
                TRANSCRIPT_ADD_ENDPOINT,
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [400, 422]:
                logger.info("✅ API correctly handled invalid JSON payload")
                return True
            else:
                logger.error(f"❌ Unexpected response to invalid JSON: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error in error handling test: {str(e)}")
            return False
    
    def run_all_tests(self) -> None:
        """Run all transcript tests."""
        logger.info("🚀 Starting comprehensive transcript tests")
        logger.info("="*60)
        
        # Check server health first
        if not self.run_test("Server Health Check", self.check_server_health):
            logger.error("❌ Server is not accessible. Cannot proceed with tests.")
            self.print_test_summary()
            return
        
        # Clear transcript at start to ensure clean state
        self.run_test("Initial Transcript Clear", self.clear_transcript)
        
        # Run all tests
        self.run_test("Add Multiple Entries", self.test_add_multiple_entries)
        self.run_test("Fetch Transcript", self.test_fetch_transcript)
        self.run_test("Transcript Persistence", self.test_transcript_persistence)
        self.run_test("Clear Functionality", self.test_clear_functionality)
        self.run_test("Error Handling", self.test_error_handling)
        
        # Print final summary
        self.print_test_summary()
    
    def print_test_summary(self) -> None:
        """Print a summary of all test results."""
        logger.info("="*60)
        logger.info("🎯 TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {self.test_results['total_tests']}")
        logger.info(f"Passed: {self.test_results['passed']} ✅")
        logger.info(f"Failed: {self.test_results['failed']} ❌")
        
        if self.test_results['errors']:
            logger.info("\n❌ ERRORS:")
            for error in self.test_results['errors']:
                logger.info(f"  - {error}")
        
        success_rate = (self.test_results['passed'] / max(self.test_results['total_tests'], 1)) * 100
        logger.info(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            logger.info("🎉 ALL TESTS PASSED! 🎉")
        elif success_rate >= 80:
            logger.info("⚠️ Most tests passed, but some issues detected")
        else:
            logger.info("🚨 Multiple test failures detected")
        
        logger.info("="*60)


def main():
    """Main function to run the transcript tests."""
    logger.info("Starting simplified transcript functionality test")
    logger.info(f"Testing web app backend at: {WEB_APP_BASE_URL}")
    logger.info(f"Log file: /tmp/test_transcript_simple.log")
    
    # Create tester instance and run all tests
    tester = TranscriptTester()
    tester.run_all_tests()
    
    logger.info("Test run completed. Check log file for detailed results.")


if __name__ == "__main__":
    main()