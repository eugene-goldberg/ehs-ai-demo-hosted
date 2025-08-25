"""
API package for EHS data foundation backend.

This package contains API endpoints and routers for various services.
"""

from .simple_rejection_api import simple_rejection_router

__all__ = ['simple_rejection_router']