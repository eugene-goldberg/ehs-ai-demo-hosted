"""
LangSmith Configuration Module

This module provides configuration and helper functions for LangSmith tracing
in the EHS AI Demo data foundation backend.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Set up logging
logger = logging.getLogger(__name__)

# Global tracing state
_tracing_enabled = False
_current_project = None
_langsmith_available = False

# Check if LangSmith is available and properly configured
try:
    from langsmith import Client
    from langsmith.utils import tracing_is_enabled
    import langchain
    from langchain import callbacks
    
    # Check for API key - prefer LANGCHAIN_API_KEY over LANGSMITH_API_KEY
    _api_key = os.getenv('LANGCHAIN_API_KEY') or os.getenv('LANGSMITH_API_KEY')
    if _api_key:
        _langsmith_available = True
        logger.info("LangSmith is available and configured")
    else:
        logger.warning("LangSmith API key not found in environment variables")
        
except ImportError as e:
    logger.warning(f"LangSmith not available: {e}")
    _langsmith_available = False


class LangSmithConfig:
    """Configuration class for LangSmith tracing."""
    
    def __init__(self):
        self.client = None
        self.default_project = "ehs-ai-demo-data-foundation"
        self.current_project = self.default_project
        
        if _langsmith_available and _api_key:
            try:
                self.client = Client(api_key=_api_key)
                logger.info("LangSmith client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith client: {e}")
                self.client = None
    
    @property
    def is_available(self) -> bool:
        """Check if LangSmith is available and properly configured."""
        return _langsmith_available and self.client is not None
    
    @property
    def is_enabled(self) -> bool:
        """Check if tracing is currently enabled."""
        return _tracing_enabled and self.is_available
    
    def enable_tracing(self, project_name: Optional[str] = None) -> bool:
        """
        Enable LangSmith tracing.
        
        Args:
            project_name: Optional project name to set. If None, uses current project.
            
        Returns:
            bool: True if tracing was enabled successfully, False otherwise.
        """
        global _tracing_enabled
        
        if not self.is_available:
            logger.warning("Cannot enable tracing: LangSmith is not available")
            return False
        
        try:
            if project_name:
                self.set_project(project_name)
            
            # Set environment variable for LangSmith
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_PROJECT'] = self.current_project
            
            _tracing_enabled = True
            logger.info(f"LangSmith tracing enabled for project: {self.current_project}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable tracing: {e}")
            return False
    
    def disable_tracing(self) -> bool:
        """
        Disable LangSmith tracing.
        
        Returns:
            bool: True if tracing was disabled successfully, False otherwise.
        """
        global _tracing_enabled
        
        try:
            # Remove environment variables
            os.environ.pop('LANGCHAIN_TRACING_V2', None)
            os.environ.pop('LANGCHAIN_PROJECT', None)
            
            _tracing_enabled = False
            logger.info("LangSmith tracing disabled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable tracing: {e}")
            return False
    
    def set_project(self, project_name: str) -> bool:
        """
        Set the current project name for tracing.
        
        Args:
            project_name: Name of the project to use for traces.
            
        Returns:
            bool: True if project was set successfully, False otherwise.
        """
        try:
            self.current_project = project_name
            if _tracing_enabled:
                os.environ['LANGCHAIN_PROJECT'] = project_name
            
            logger.info(f"LangSmith project set to: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set project: {e}")
            return False
    
    def get_project(self) -> str:
        """Get the current project name."""
        return self.current_project
    
    def create_session_project(self, ingestion_id: str, base_name: Optional[str] = None) -> str:
        """
        Create a project name for a specific ingestion session.
        
        Args:
            ingestion_id: Unique identifier for the ingestion session.
            base_name: Base project name. If None, uses default project.
            
        Returns:
            str: Generated project name for the session.
        """
        if not base_name:
            base_name = self.default_project
        
        session_project = f"{base_name}-session-{ingestion_id}"
        return session_project
    
    def tag_trace(self, **metadata: Any) -> Dict[str, Any]:
        """
        Create metadata tags for traces.
        
        Args:
            **metadata: Key-value pairs to include as metadata.
            
        Returns:
            dict: Metadata dictionary formatted for LangSmith.
        """
        tags = {
            'service': 'ehs-ai-demo-data-foundation',
            'component': 'backend',
            **metadata
        }
        return tags


# Global configuration instance
config = LangSmithConfig()


# Convenience functions
def is_langsmith_available() -> bool:
    """Check if LangSmith is available and configured."""
    return config.is_available


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return config.is_enabled


def enable_tracing(project_name: Optional[str] = None) -> bool:
    """Enable LangSmith tracing with optional project name."""
    return config.enable_tracing(project_name)


def disable_tracing() -> bool:
    """Disable LangSmith tracing."""
    return config.disable_tracing()


def set_project(project_name: str) -> bool:
    """Set the current project name for tracing."""
    return config.set_project(project_name)


def get_project() -> str:
    """Get the current project name."""
    return config.get_project()


def create_ingestion_session(ingestion_id: str, enable_tracing: bool = True) -> Optional[str]:
    """
    Create and optionally enable a new ingestion session with tracing.
    
    Args:
        ingestion_id: Unique identifier for the ingestion session.
        enable_tracing: Whether to enable tracing for this session.
        
    Returns:
        Optional[str]: Session project name if successful, None otherwise.
    """
    if not config.is_available:
        logger.warning("LangSmith not available for ingestion session")
        return None
    
    session_project = config.create_session_project(ingestion_id)
    
    if enable_tracing:
        if config.enable_tracing(session_project):
            logger.info(f"Ingestion session created with tracing: {session_project}")
            return session_project
        else:
            logger.error(f"Failed to enable tracing for session: {session_project}")
            return None
    else:
        config.set_project(session_project)
        logger.info(f"Ingestion session created without tracing: {session_project}")
        return session_project


def tag_ingestion_trace(ingestion_id: str, **additional_metadata: Any) -> Dict[str, Any]:
    """
    Create metadata tags for ingestion traces.
    
    Args:
        ingestion_id: Unique identifier for the ingestion session.
        **additional_metadata: Additional key-value pairs to include.
        
    Returns:
        dict: Metadata dictionary with ingestion-specific tags.
    """
    tags = config.tag_trace(
        ingestion_id=ingestion_id,
        operation_type='data_ingestion',
        **additional_metadata
    )
    return tags


@contextmanager
def tracing_context(project_name: Optional[str] = None, ingestion_id: Optional[str] = None):
    """
    Context manager for temporary tracing configuration.
    
    Args:
        project_name: Optional project name to use during context.
        ingestion_id: Optional ingestion ID to create session project.
        
    Usage:
        with tracing_context(project_name="my-project"):
            # Tracing is enabled with specified project
            pass
        # Previous tracing state is restored
    """
    if not config.is_available:
        logger.warning("LangSmith not available for tracing context")
        yield
        return
    
    # Save current state
    original_enabled = config.is_enabled
    original_project = config.current_project
    
    try:
        # Set up context
        if ingestion_id:
            session_project = create_ingestion_session(ingestion_id, enable_tracing=True)
            if not session_project:
                logger.warning("Failed to create ingestion session, using original settings")
        elif project_name:
            config.enable_tracing(project_name)
        else:
            config.enable_tracing()
        
        yield
        
    finally:
        # Restore original state
        if original_enabled:
            config.enable_tracing(original_project)
        else:
            config.disable_tracing()
            config.set_project(original_project)


# Initialize logging
def setup_logging():
    """Set up logging configuration for LangSmith module."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Auto-setup if imported
if __name__ != "__main__":
    # Only set up logging if we're being imported
    pass
else:
    # If run directly, provide some status information
    setup_logging()
    print(f"LangSmith Available: {is_langsmith_available()}")
    print(f"Tracing Enabled: {is_tracing_enabled()}")
    print(f"Current Project: {get_project()}")
    
    if is_langsmith_available():
        print("LangSmith configuration is ready for use.")
    else:
        print("LangSmith is not available. Please check your configuration:")
        print("1. Install langsmith: pip install langsmith")
        print("2. Set LANGCHAIN_API_KEY or LANGSMITH_API_KEY environment variable")
        print("3. Ensure langchain is installed for tracing integration")