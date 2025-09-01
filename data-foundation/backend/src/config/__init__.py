"""
Configuration module for EHS Data Foundation Backend

This module provides configuration classes and utilities for the EHS
(Environmental Health & Safety) data foundation system.
"""

from .ehs_goals_config import (
    EHSGoalsConfig,
    EHSGoal,
    SiteLocation,
    EHSCategory,
    ehs_goals_config,
    get_goal,
    get_reduction_percentage,
    get_all_goals,
    get_goals_summary,
)

__all__ = [
    "EHSGoalsConfig",
    "EHSGoal", 
    "SiteLocation",
    "EHSCategory",
    "ehs_goals_config",
    "get_goal",
    "get_reduction_percentage",
    "get_all_goals",
    "get_goals_summary",
]