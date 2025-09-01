"""
LLM module for environmental data analysis.

Note: This package contains prompt templates for environmental analysis.
The main LLM functionality (get_llm, etc.) is in src/llm.py at the project root.
"""

# Re-export the main LLM functions to maintain compatibility
try:
    import sys
    import os
    # Add the parent directory to sys.path to import from src.llm
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from llm import get_llm, get_llm_model_name, get_combined_chunks
    
    __all__ = ['get_llm', 'get_llm_model_name', 'get_combined_chunks']
    
except ImportError:
    # If import fails, create placeholder functions to avoid import errors
    def get_llm(*args, **kwargs):
        raise ImportError("LLM functions not available. Import directly from src.llm")
    
    def get_llm_model_name(*args, **kwargs):
        raise ImportError("LLM functions not available. Import directly from src.llm")
    
    def get_combined_chunks(*args, **kwargs):
        raise ImportError("LLM functions not available. Import directly from src.llm")
    
    __all__ = []