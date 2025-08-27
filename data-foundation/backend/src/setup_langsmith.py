"""
Simple LangSmith Tracing Setup

This module provides a straightforward way to enable LangSmith tracing
for the EHS AI Demo application without complex configuration.

Usage:
    from setup_langsmith import setup_langsmith_tracing, set_langsmith_project
    
    # Enable tracing
    if setup_langsmith_tracing():
        print("LangSmith tracing enabled!")
    
    # Change project dynamically
    set_langsmith_project("my-custom-project")
"""

import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_PROJECT = "ehs-ai-demo-ingestion"
DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


def setup_langsmith_tracing() -> bool:
    """
    Set up LangSmith tracing if properly configured.
    
    Checks for required environment variables and enables tracing
    if all conditions are met.
    
    Returns:
        bool: True if LangSmith is properly configured and enabled,
              False if not configured or disabled
    
    Environment Variables Required:
        LANGCHAIN_API_KEY: Your LangSmith API key (required)
        LANGCHAIN_TRACING_V2: Set to "true" to enable (required)
        LANGCHAIN_PROJECT: Project name (optional, defaults to "ehs-ai-demo-ingestion")
        LANGCHAIN_ENDPOINT: API endpoint (optional, defaults to "https://api.smith.langchain.com")
    """
    
    # Check if API key is set and not empty
    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    if not api_key:
        logger.info("LangSmith tracing disabled: LANGCHAIN_API_KEY not set or empty")
        return False
    
    # Check if tracing is enabled
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
    if tracing_enabled != "true":
        logger.info("LangSmith tracing disabled: LANGCHAIN_TRACING_V2 not set to 'true'")
        return False
    
    # Set default project if not specified
    project = os.getenv("LANGCHAIN_PROJECT", "").strip()
    if not project:
        os.environ["LANGCHAIN_PROJECT"] = DEFAULT_PROJECT
        project = DEFAULT_PROJECT
        logger.info(f"Using default LangSmith project: {project}")
    
    # Set default endpoint if not specified
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "").strip()
    if not endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = DEFAULT_ENDPOINT
        endpoint = DEFAULT_ENDPOINT
        logger.info(f"Using default LangSmith endpoint: {endpoint}")
    
    logger.info(f"LangSmith tracing enabled for project: {project}")
    logger.debug(f"LangSmith endpoint: {endpoint}")
    
    return True


def set_langsmith_project(project_name: str) -> bool:
    """
    Dynamically set the LangSmith project name.
    
    Args:
        project_name (str): Name of the LangSmith project
        
    Returns:
        bool: True if successfully set, False if project_name is empty
        
    Example:
        set_langsmith_project("ehs-demo-testing")
    """
    
    if not project_name or not project_name.strip():
        logger.warning("Cannot set LangSmith project: project_name is empty")
        return False
    
    project_name = project_name.strip()
    os.environ["LANGCHAIN_PROJECT"] = project_name
    logger.info(f"LangSmith project changed to: {project_name}")
    
    return True


def get_langsmith_status() -> dict:
    """
    Get the current LangSmith configuration status.
    
    Returns:
        dict: Current configuration status including:
            - enabled: bool - Whether tracing is enabled
            - api_key_set: bool - Whether API key is configured
            - project: str - Current project name
            - endpoint: str - Current endpoint URL
    """
    
    api_key_set = bool(os.getenv("LANGCHAIN_API_KEY", "").strip())
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", DEFAULT_PROJECT)
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", DEFAULT_ENDPOINT)
    
    return {
        "enabled": api_key_set and tracing_enabled,
        "api_key_set": api_key_set,
        "project": project,
        "endpoint": endpoint
    }


def print_langsmith_status():
    """
    Print the current LangSmith configuration status to console.
    
    Useful for debugging and verification.
    """
    
    status = get_langsmith_status()
    
    print("\n" + "="*50)
    print("LangSmith Configuration Status")
    print("="*50)
    print(f"Tracing Enabled: {'✓' if status['enabled'] else '✗'}")
    print(f"API Key Set: {'✓' if status['api_key_set'] else '✗'}")
    print(f"Project: {status['project']}")
    print(f"Endpoint: {status['endpoint']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Example usage when run as a script
    print("Testing LangSmith Setup...")
    
    # Show current status
    print_langsmith_status()
    
    # Try to setup tracing
    if setup_langsmith_tracing():
        print("✓ LangSmith tracing is ready!")
    else:
        print("✗ LangSmith tracing is not configured.")
        print("\nTo enable LangSmith tracing, set these environment variables:")
        print("  export LANGCHAIN_API_KEY='your-api-key-here'")
        print("  export LANGCHAIN_TRACING_V2='true'")
        print("  export LANGCHAIN_PROJECT='ehs-ai-demo-ingestion'  # optional")
        print("  export LANGCHAIN_ENDPOINT='https://api.smith.langchain.com'  # optional")
    
    # Show final status
    print_langsmith_status()