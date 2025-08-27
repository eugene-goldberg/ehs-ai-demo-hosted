"""
Transcript Forwarder Utility

This module provides functionality to forward transcript entries from the backend
ingestion API to the web app backend in a thread-safe and non-blocking manner.
"""

import requests
import logging
import threading
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import json


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool executor for non-blocking requests
executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="transcript_forwarder")

# Web app backend URL
WEB_APP_BACKEND_URL = "http://localhost:8001/api/data/transcript/add"

# Request timeout in seconds
REQUEST_TIMEOUT = 10


def _make_request(role: str, content: str, context: Optional[Dict] = None) -> bool:
    """
    Internal function to make the actual HTTP request.
    
    Args:
        role (str): The role of the transcript entry (e.g., 'user', 'assistant')
        content (str): The content of the transcript entry
        context (Optional[Dict]): Additional context data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare the payload
        payload = {
            "role": role,
            "content": content
        }
        
        if context:
            payload["context"] = context
            
        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Make the POST request
        logger.info(f"Forwarding transcript entry - Role: {role}, Content length: {len(content)} chars")
        logger.debug(f"Forwarding to URL: {WEB_APP_BACKEND_URL}")
        logger.debug(f"Payload: {json.dumps(payload)[:200]}...")
        
        response = requests.post(
            WEB_APP_BACKEND_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        
        # Check if request was successful
        if response.status_code in [200, 201]:
            logger.info(f"Successfully forwarded transcript entry. Status: {response.status_code}")
            return True
        else:
            logger.error(f"Failed to forward transcript entry. Status: {response.status_code}, Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout while forwarding transcript entry to {WEB_APP_BACKEND_URL}")
        return False
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error while forwarding transcript entry to {WEB_APP_BACKEND_URL}")
        return False
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception while forwarding transcript entry: {str(e)}")
        return False
        
    except json.JSONEncodeError as e:
        logger.error(f"JSON encoding error while preparing transcript payload: {str(e)}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error while forwarding transcript entry: {str(e)}")
        return False


def forward_transcript_entry(role: str, content: str, context: Optional[Dict] = None) -> bool:
    """
    Forward a transcript entry to the web app backend in a thread-safe and non-blocking manner.
    
    This function submits the request to a thread pool executor to ensure it doesn't
    block the calling thread. The actual HTTP request is handled asynchronously.
    
    Args:
        role (str): The role of the transcript entry (e.g., 'user', 'assistant', 'system')
        content (str): The content of the transcript entry
        context (Optional[Dict]): Additional context data to include with the entry
        
    Returns:
        bool: True if the request was successfully submitted to the thread pool,
              False if there was an error submitting the request
              
    Example:
        >>> success = forward_transcript_entry(
        ...     role="user",
        ...     content="What are the safety requirements for handling chemicals?",
        ...     context={"session_id": "abc123", "timestamp": "2024-08-27T10:30:00Z"}
        ... )
        >>> print(f"Request submitted: {success}")
    """
    try:
        # Input validation
        if not isinstance(role, str) or not role.strip():
            logger.error("Role must be a non-empty string")
            return False
            
        if not isinstance(content, str) or not content.strip():
            logger.error("Content must be a non-empty string")
            return False
            
        if context is not None and not isinstance(context, dict):
            logger.error("Context must be a dictionary if provided")
            return False
        
        # Submit the request to the thread pool for non-blocking execution
        future = executor.submit(_make_request, role, content, context)
        
        # Log that the request has been submitted
        logger.info(f"Transcript forwarding request submitted to thread pool - Role: {role}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error submitting transcript forwarding request to thread pool: {str(e)}")
        return False


def forward_transcript_entry_sync(role: str, content: str, context: Optional[Dict] = None) -> bool:
    """
    Forward a transcript entry to the web app backend synchronously.
    
    This function blocks until the request is complete and returns the actual result.
    Use this when you need to know if the forwarding was successful before proceeding.
    
    Args:
        role (str): The role of the transcript entry (e.g., 'user', 'assistant', 'system')
        content (str): The content of the transcript entry
        context (Optional[Dict]): Additional context data to include with the entry
        
    Returns:
        bool: True if the request was successful, False otherwise
        
    Example:
        >>> success = forward_transcript_entry_sync(
        ...     role="assistant",
        ...     content="Based on the safety data sheet, you should wear protective gloves...",
        ...     context={"query_id": "q456", "confidence": 0.95}
        ... )
        >>> if success:
        ...     print("Transcript entry successfully forwarded")
    """
    try:
        # Input validation
        if not isinstance(role, str) or not role.strip():
            logger.error("Role must be a non-empty string")
            return False
            
        if not isinstance(content, str) or not content.strip():
            logger.error("Content must be a non-empty string")
            return False
            
        if context is not None and not isinstance(context, dict):
            logger.error("Context must be a dictionary if provided")
            return False
        
        # Make the request synchronously
        return _make_request(role, content, context)
        
    except Exception as e:
        logger.error(f"Error in synchronous transcript forwarding: {str(e)}")
        return False


def shutdown_forwarder():
    """
    Shutdown the transcript forwarder thread pool gracefully.
    
    Call this function when shutting down the application to ensure
    all pending requests are completed before termination.
    """
    try:
        logger.info("Shutting down transcript forwarder thread pool...")
        executor.shutdown(wait=True, timeout=30)
        logger.info("Transcript forwarder thread pool shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down transcript forwarder thread pool: {str(e)}")


# Module-level configuration
def configure_forwarder(web_app_url: str = None, timeout: int = None, max_workers: int = None):
    """
    Configure the transcript forwarder with custom settings.
    
    Args:
        web_app_url (str): Custom web app backend URL
        timeout (int): Custom request timeout in seconds
        max_workers (int): Custom maximum number of worker threads
    """
    global WEB_APP_BACKEND_URL, REQUEST_TIMEOUT, executor
    
    if web_app_url:
        WEB_APP_BACKEND_URL = web_app_url
        logger.info(f"Web app backend URL configured to: {WEB_APP_BACKEND_URL}")
        
    if timeout:
        REQUEST_TIMEOUT = timeout
        logger.info(f"Request timeout configured to: {REQUEST_TIMEOUT} seconds")
        
    if max_workers:
        # Shutdown existing executor and create new one with different worker count
        executor.shutdown(wait=True, timeout=10)
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="transcript_forwarder")
        logger.info(f"Thread pool reconfigured with {max_workers} max workers")


if __name__ == "__main__":
    # Example usage and testing
    import time
    
    # Test the forwarder
    logger.info("Testing transcript forwarder...")
    
    # Test asynchronous forwarding
    success = forward_transcript_entry(
        role="user",
        content="What are the safety requirements for chemical storage?",
        context={"session_id": "test_session", "timestamp": "2024-08-27T10:30:00Z"}
    )
    
    print(f"Async request submitted: {success}")
    
    # Test synchronous forwarding
    success_sync = forward_transcript_entry_sync(
        role="assistant",
        content="Chemical storage requires proper ventilation, appropriate containers, and temperature control.",
        context={"query_id": "q123", "confidence": 0.89}
    )
    
    print(f"Sync request result: {success_sync}")
    
    # Wait a bit for async request to complete
    time.sleep(2)
    
    # Shutdown gracefully
    shutdown_forwarder()