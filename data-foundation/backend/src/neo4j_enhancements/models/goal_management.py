"""
Goal Management System for EHS AI Demo Data Foundation

This module provides comprehensive goal management functionality including:
- Goal creation and management
- Target assignment and tracking
- Progress calculation methods
- Hierarchical goal relationships
- Goal achievement analysis
- Integration with historical metrics
- CRUD operations for goals and targets

Author: Claude AI
Created: 2025-08-28
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import uuid

from neo4j import GraphDatabase, Transaction
from neo4j.exceptions import Neo4jError


# Configure logging
logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Goal status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class GoalType(Enum):
    """Goal type enumeration"""
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    TACTICAL = "tactical"
    PERSONAL = "personal"
    TEAM = "team"
    ORGANIZATIONAL = "organizational"


class TargetType(Enum):
    """Target type enumeration"""
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    REDUCTION = "reduction"
    INCREASE = "increase"
    MAINTAIN = "maintain"


class MeasurementUnit(Enum):
    """Measurement unit enumeration"""
    COUNT = "count"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    HOURS = "hours"
    DAYS = "days"
    RATE = "rate"
    SCORE = "score"


@dataclass
class Goal:
    """Goal data class"""
    goal_id: str
    title: str
    description: str
    goal_type: GoalType
    status: GoalStatus
    created_date: datetime
    start_date: datetime
    end_date: datetime
    owner_id: str
    department: str
    priority: int
    parent_goal_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary"""
        return {
            'goal_id': self.goal_id,
            'title': self.title,
            'description': self.description,
            'goal_type': self.goal_type.value,
            'status': self.status.value,
            'created_date': self.created_date.isoformat(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'owner_id': self.owner_id,
            'department': self.department,
            'priority': self.priority,
            'parent_goal_id': self.parent_goal_id,
            'metadata': self.metadata or {}
        }


@dataclass
class Target:
    """Target data class"""
    target_id: str
    goal_id: str
    metric_name: str
    target_value: float
    current_value: float
    target_type: TargetType
    unit: MeasurementUnit
    baseline_value: Optional[float] = None
    threshold_values: Optional[Dict[str, float]] = None
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert target to dictionary"""
        return {
            'target_id': self.target_id,
            'goal_id': self.goal_id,
            'metric_name': self.metric_name,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'target_type': self.target_type.value,
            'unit': self.unit.value,
            'baseline_value': self.baseline_value,
            'threshold_values': self.threshold_values or {},
            'weight': self.weight
        }


class GoalManagementSystem:
    """
    Comprehensive Goal Management System
    
    Provides functionality for managing goals, targets, and progress tracking
    with Neo4j integration for persistent storage and relationship management.
    """
    
    def __init__(self, driver: GraphDatabase.driver, database: str = "neo4j"):
        """
        Initialize the Goal Management System
        
        Args:
            driver: Neo4j database driver
            database: Database name (default: "neo4j")
        """
        self.driver = driver
        self.database = database
        self._create_constraints()
    
    def _create_constraints(self):
        """Create necessary constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT goal_id_unique IF NOT EXISTS FOR (g:Goal) REQUIRE g.goal_id IS UNIQUE",
            "CREATE CONSTRAINT target_id_unique IF NOT EXISTS FOR (t:Target) REQUIRE t.target_id IS UNIQUE",
            "CREATE INDEX goal_status_index IF NOT EXISTS FOR (g:Goal) ON (g.status)",
            "CREATE INDEX goal_type_index IF NOT EXISTS FOR (g:Goal) ON (g.goal_type)",
            "CREATE INDEX goal_department_index IF NOT EXISTS FOR (g:Goal) ON (g.department)",
            "CREATE INDEX goal_owner_index IF NOT EXISTS FOR (g:Goal) ON (g.owner_id)",
            "CREATE INDEX goal_dates_index IF NOT EXISTS FOR (g:Goal) ON (g.start_date, g.end_date)",
            "CREATE INDEX target_metric_index IF NOT EXISTS FOR (t:Target) ON (t.metric_name)"
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint/index: {constraint}")
                except Neo4jError as e:
                    if "equivalent" not in str(e).lower():
                        logger.warning(f"Failed to create constraint: {e}")
    
    # Goal CRUD Operations
    
    def create_goal(self, 
                   title: str,
                   description: str,
                   goal_type: GoalType,
                   start_date: datetime,
                   end_date: datetime,
                   owner_id: str,
                   department: str,
                   priority: int = 5,
                   parent_goal_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Goal:
        """
        Create a new goal
        
        Args:
            title: Goal title
            description: Goal description
            goal_type: Type of goal
            start_date: Goal start date
            end_date: Goal end date
            owner_id: Goal owner identifier
            department: Department/team
            priority: Goal priority (1-10, higher is more important)
            parent_goal_id: Parent goal ID for hierarchical goals
            metadata: Additional metadata
            
        Returns:
            Created Goal object
            
        Raises:
            ValueError: If validation fails
            Neo4jError: If database operation fails
        """
        try:
            # Validate inputs
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            if priority < 1 or priority > 10:
                raise ValueError("Priority must be between 1 and 10")
            
            # Generate unique goal ID
            goal_id = f"goal_{uuid.uuid4().hex[:8]}"
            
            goal = Goal(
                goal_id=goal_id,
                title=title,
                description=description,
                goal_type=goal_type,
                status=GoalStatus.DRAFT,
                created_date=datetime.now(),
                start_date=start_date,
                end_date=end_date,
                owner_id=owner_id,
                department=department,
                priority=priority,
                parent_goal_id=parent_goal_id,
                metadata=metadata or {}
            )
            
            # Store in Neo4j
            with self.driver.session(database=self.database) as session:
                result = session.execute_write(self._create_goal_tx, goal)
                
            logger.info(f"Created goal: {goal_id}")
            return goal
            
        except Exception as e:
            logger.error(f"Failed to create goal: {e}")
            raise
    
    def _create_goal_tx(self, tx: Transaction, goal: Goal) -> Dict[str, Any]:
        """Transaction to create a goal"""
        query = """
        CREATE (g:Goal {
            goal_id: ,
            title: ,
            description: ,
            goal_type: ,
            status: 0,
            created_date: datetime(),
            start_date: datetime(),
            end_date: datetime(),
            owner_id: ,
            department: ,
            priority: ,
            metadata: 
        })
        RETURN g
        """
        
        result = tx.run(query, goal.to_dict())
        
        # Create parent relationship if specified
        if goal.parent_goal_id:
            parent_query = """
            MATCH (parent:Goal {goal_id: })
            MATCH (child:Goal {goal_id: })
            CREATE (child)-[:CHILD_OF]->(parent)
            """
            tx.run(parent_query, {
                "parent_goal_id": goal.parent_goal_id,
                "goal_id": goal.goal_id
            })
        
        return result.single()["g"]
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Retrieve a goal by ID
        
        Args:
            goal_id: Goal identifier
            
        Returns:
            Goal object if found, None otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    "MATCH (g:Goal {goal_id: }) RETURN g",
                    goal_id=goal_id
                )
                
                record = result.single()
                if record:
                    return self._dict_to_goal(dict(record["g"]))
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get goal {goal_id}: {e}")
            return None
    
    def update_goal(self, goal_id: str, **updates) -> bool:
        """
        Update a goal
        
        Args:
            goal_id: Goal identifier
            **updates: Fields to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Build update query dynamically
            set_clauses = []
            params = {"goal_id": goal_id}
            
            for key, value in updates.items():
                if key in ["title", "description", "status", "priority", "metadata"]:
                    set_clauses.append(f"g.{key} = ")
                    params[key] = value.value if isinstance(value, Enum) else value
            
            if not set_clauses:
                return False
            
            query = f"""
            MATCH (g:Goal {{goal_id: }})
            SET {', '.join(set_clauses)}
            RETURN g
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                
                if result.single():
                    logger.info(f"Updated goal: {goal_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update goal {goal_id}: {e}")
            return False
    
    def delete_goal(self, goal_id: str) -> bool:
        """
        Delete a goal and its relationships
        
        Args:
            goal_id: Goal identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (g:Goal {goal_id: })
                    DETACH DELETE g
                    RETURN count(g) as deleted_count
                    """,
                    goal_id=goal_id
                )
                
                deleted_count = result.single()["deleted_count"]
                
                if deleted_count > 0:
                    logger.info(f"Deleted goal: {goal_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete goal {goal_id}: {e}")
            return False
    
    # Target CRUD Operations
    
    def create_target(self,
                     goal_id: str,
                     metric_name: str,
                     target_value: float,
                     target_type: TargetType,
                     unit: MeasurementUnit,
                     baseline_value: Optional[float] = None,
                     threshold_values: Optional[Dict[str, float]] = None,
                     weight: float = 1.0) -> Target:
        """
        Create a new target for a goal
        
        Args:
            goal_id: Associated goal ID
            metric_name: Name of the metric
            target_value: Target value to achieve
            target_type: Type of target
            unit: Measurement unit
            baseline_value: Baseline/starting value
            threshold_values: Threshold values (e.g., warning, critical)
            weight: Target weight for progress calculation
            
        Returns:
            Created Target object
            
        Raises:
            ValueError: If validation fails
            Neo4jError: If database operation fails
        """
        try:
            # Validate goal exists
            if not self.get_goal(goal_id):
                raise ValueError(f"Goal {goal_id} does not exist")
            
            if weight <= 0:
                raise ValueError("Weight must be positive")
            
            target_id = f"target_{uuid.uuid4().hex[:8]}"
            
            target = Target(
                target_id=target_id,
                goal_id=goal_id,
                metric_name=metric_name,
                target_value=target_value,
                current_value=baseline_value or 0.0,
                target_type=target_type,
                unit=unit,
                baseline_value=baseline_value,
                threshold_values=threshold_values or {},
                weight=weight
            )
            
            # Store in Neo4j
            with self.driver.session(database=self.database) as session:
                result = session.execute_write(self._create_target_tx, target)
            
            logger.info(f"Created target: {target_id} for goal: {goal_id}")
            return target
            
        except Exception as e:
            logger.error(f"Failed to create target: {e}")
            raise
    
    def _create_target_tx(self, tx: Transaction, target: Target) -> Dict[str, Any]:
        """Transaction to create a target"""
        query = """
        MATCH (g:Goal {goal_id: })
        CREATE (t:Target {
            target_id: ,
            goal_id: ,
            metric_name: ,
            target_value: ,
            current_value: ,
            target_type: ,
            unit: ,
            baseline_value: ,
            threshold_values: ,
            weight: 
        })
        CREATE (t)-[:TARGET_OF]->(g)
        RETURN t
        """
        
        result = tx.run(query, target.to_dict())
        return result.single()["t"]
    
    def update_target_progress(self, target_id: str, current_value: float) -> bool:
        """
        Update target current value
        
        Args:
            target_id: Target identifier
            current_value: New current value
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (t:Target {target_id: })
                    SET t.current_value = ,
                        t.last_updated = datetime()
                    RETURN t
                    """,
                    target_id=target_id,
                    current_value=current_value
                )
                
                if result.single():
                    logger.info(f"Updated target progress: {target_id} = {current_value}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update target progress: {e}")
            return False
    
    def get_goal_targets(self, goal_id: str) -> List[Target]:
        """
        Get all targets for a goal
        
        Args:
            goal_id: Goal identifier
            
        Returns:
            List of Target objects
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (g:Goal {goal_id: })<-[:TARGET_OF]-(t:Target)
                    RETURN t
                    ORDER BY t.metric_name
                    """,
                    goal_id=goal_id
                )
                
                targets = []
                for record in result:
                    targets.append(self._dict_to_target(dict(record["t"])))
                
                return targets
                
        except Exception as e:
            logger.error(f"Failed to get targets for goal {goal_id}: {e}")
            return []
    
    # Progress Calculation Methods
    
    def calculate_target_progress(self, target: Target) -> float:
        """
        Calculate progress percentage for a target
        
        Args:
            target: Target object
            
        Returns:
            Progress percentage (0-100)
        """
        try:
            if target.target_type == TargetType.ABSOLUTE:
                if target.target_value == 0:
                    return 100.0 if target.current_value >= target.target_value else 0.0
                return min(100.0, (target.current_value / target.target_value) * 100)
            
            elif target.target_type == TargetType.PERCENTAGE:
                return min(100.0, target.current_value)
            
            elif target.target_type == TargetType.REDUCTION:
                if target.baseline_value is None or target.baseline_value == 0:
                    return 0.0
                
                reduction_needed = target.baseline_value - target.target_value
                reduction_achieved = target.baseline_value - target.current_value
                
                if reduction_needed <= 0:
                    return 100.0 if reduction_achieved >= 0 else 0.0
                
                return min(100.0, max(0.0, (reduction_achieved / reduction_needed) * 100))
            
            elif target.target_type == TargetType.INCREASE:
                if target.baseline_value is None:
                    return 0.0
                
                increase_needed = target.target_value - target.baseline_value
                increase_achieved = target.current_value - target.baseline_value
                
                if increase_needed <= 0:
                    return 100.0 if increase_achieved >= 0 else 0.0
                
                return min(100.0, max(0.0, (increase_achieved / increase_needed) * 100))
            
            elif target.target_type == TargetType.MAINTAIN:
                # For maintain targets, check if current value is within acceptable range
                tolerance = target.threshold_values.get("tolerance", 0.05)  # 5% default tolerance
                lower_bound = target.target_value * (1 - tolerance)
                upper_bound = target.target_value * (1 + tolerance)
                
                if lower_bound <= target.current_value <= upper_bound:
                    return 100.0
                else:
                    deviation = abs(target.current_value - target.target_value) / target.target_value
                    return max(0.0, 100.0 - (deviation * 100))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate target progress: {e}")
            return 0.0
    
    def calculate_goal_progress(self, goal_id: str) -> Dict[str, Any]:
        """
        Calculate overall progress for a goal based on its targets
        
        Args:
            goal_id: Goal identifier
            
        Returns:
            Dictionary containing progress metrics
        """
        try:
            targets = self.get_goal_targets(goal_id)
            
            if not targets:
                return {
                    "overall_progress": 0.0,
                    "target_count": 0,
                    "achieved_targets": 0,
                    "target_progress": []
                }
            
            target_progress = []
            total_weighted_progress = 0.0
            total_weight = 0.0
            achieved_targets = 0
            
            for target in targets:
                progress = self.calculate_target_progress(target)
                target_progress.append({
                    "target_id": target.target_id,
                    "metric_name": target.metric_name,
                    "progress": progress,
                    "current_value": target.current_value,
                    "target_value": target.target_value,
                    "weight": target.weight
                })
                
                total_weighted_progress += progress * target.weight
                total_weight += target.weight
                
                if progress >= 100.0:
                    achieved_targets += 1
            
            overall_progress = total_weighted_progress / total_weight if total_weight > 0 else 0.0
            
            return {
                "overall_progress": round(overall_progress, 2),
                "target_count": len(targets),
                "achieved_targets": achieved_targets,
                "target_progress": target_progress
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate goal progress: {e}")
            return {
                "overall_progress": 0.0,
                "target_count": 0,
                "achieved_targets": 0,
                "target_progress": []
            }
    
    # Hierarchical Goal Methods
    
    def get_child_goals(self, parent_goal_id: str) -> List[Goal]:
        """
        Get all child goals for a parent goal
        
        Args:
            parent_goal_id: Parent goal identifier
            
        Returns:
            List of child Goal objects
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (parent:Goal {goal_id: })<-[:CHILD_OF]-(child:Goal)
                    RETURN child
                    ORDER BY child.priority DESC, child.created_date
                    """,
                    parent_goal_id=parent_goal_id
                )
                
                goals = []
                for record in result:
                    goals.append(self._dict_to_goal(dict(record["child"])))
                
                return goals
                
        except Exception as e:
            logger.error(f"Failed to get child goals: {e}")
            return []
    
    def get_goal_hierarchy(self, goal_id: str) -> Dict[str, Any]:
        """
        Get complete goal hierarchy (parents and children)
        
        Args:
            goal_id: Goal identifier
            
        Returns:
            Dictionary containing hierarchy information
        """
        try:
            goal = self.get_goal(goal_id)
            if not goal:
                return {}
            
            # Get parent chain
            parent_chain = []
            current_goal = goal
            
            while current_goal and current_goal.parent_goal_id:
                parent = self.get_goal(current_goal.parent_goal_id)
                if parent:
                    parent_chain.append(parent.to_dict())
                    current_goal = parent
                else:
                    break
            
            # Get children
            children = [child.to_dict() for child in self.get_child_goals(goal_id)]
            
            # Get sibling goals (same parent)
            siblings = []
            if goal.parent_goal_id:
                all_children = self.get_child_goals(goal.parent_goal_id)
                siblings = [child.to_dict() for child in all_children if child.goal_id != goal_id]
            
            return {
                "goal": goal.to_dict(),
                "parent_chain": list(reversed(parent_chain)),  # Root to immediate parent
                "children": children,
                "siblings": siblings
            }
            
        except Exception as e:
            logger.error(f"Failed to get goal hierarchy: {e}")
            return {}
    
    def calculate_hierarchical_progress(self, goal_id: str) -> Dict[str, Any]:
        """
        Calculate progress including child goal progress
        
        Args:
            goal_id: Goal identifier
            
        Returns:
            Dictionary containing hierarchical progress metrics
        """
        try:
            # Get direct progress
            direct_progress = self.calculate_goal_progress(goal_id)
            
            # Get child goals and their progress
            child_goals = self.get_child_goals(goal_id)
            child_progress = []
            total_child_progress = 0.0
            
            for child in child_goals:
                child_prog = self.calculate_goal_progress(child.goal_id)
                child_progress.append({
                    "goal_id": child.goal_id,
                    "title": child.title,
                    "progress": child_prog["overall_progress"],
                    "priority": child.priority
                })
                total_child_progress += child_prog["overall_progress"] * (child.priority / 10.0)
            
            # Calculate combined progress (weighted average of direct and child progress)
            if child_goals:
                child_weight = 0.6  # 60% weight to child goals
                direct_weight = 0.4  # 40% weight to direct targets
                
                avg_child_progress = total_child_progress / len(child_goals) if child_goals else 0.0
                combined_progress = (direct_progress["overall_progress"] * direct_weight + 
                                   avg_child_progress * child_weight)
            else:
                combined_progress = direct_progress["overall_progress"]
            
            return {
                "goal_id": goal_id,
                "direct_progress": direct_progress,
                "child_progress": child_progress,
                "combined_progress": round(combined_progress, 2),
                "has_children": len(child_goals) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate hierarchical progress: {e}")
            return {
                "goal_id": goal_id,
                "direct_progress": {"overall_progress": 0.0},
                "child_progress": [],
                "combined_progress": 0.0,
                "has_children": False
            }
    
    # Goal Achievement Analysis
    
    def analyze_goal_achievement(self, goal_id: str) -> Dict[str, Any]:
        """
        Analyze goal achievement status and provide insights
        
        Args:
            goal_id: Goal identifier
            
        Returns:
            Dictionary containing achievement analysis
        """
        try:
            goal = self.get_goal(goal_id)
            if not goal:
                return {"error": "Goal not found"}
            
            progress_data = self.calculate_goal_progress(goal_id)
            
            # Time analysis
            now = datetime.now()
            total_duration = (goal.end_date - goal.start_date).days
            elapsed_days = max(0, (now - goal.start_date).days)
            remaining_days = max(0, (goal.end_date - now).days)
            
            time_progress = min(100.0, (elapsed_days / total_duration * 100)) if total_duration > 0 else 100.0
            
            # Performance analysis
            expected_progress = time_progress
            actual_progress = progress_data["overall_progress"]
            performance_ratio = actual_progress / expected_progress if expected_progress > 0 else 0.0
            
            # Status determination
            if actual_progress >= 100.0:
                status = "achieved"
            elif now > goal.end_date and actual_progress < 100.0:
                status = "failed"
            elif performance_ratio >= 1.2:
                status = "ahead"
            elif performance_ratio >= 0.8:
                status = "on_track"
            elif performance_ratio >= 0.5:
                status = "behind"
            else:
                status = "at_risk"
            
            # Risk assessment
            risk_factors = []
            
            if performance_ratio < 0.8:
                risk_factors.append("Below expected progress")
            
            if remaining_days < total_duration * 0.1 and actual_progress < 90.0:
                risk_factors.append("Limited time remaining")
            
            if progress_data["achieved_targets"] == 0 and elapsed_days > total_duration * 0.5:
                risk_factors.append("No targets achieved past midpoint")
            
            # Recommendations
            recommendations = []
            
            if status == "behind" or status == "at_risk":
                recommendations.append("Review and adjust target values or timeline")
                recommendations.append("Identify and address blocking factors")
                recommendations.append("Consider reallocating resources")
            
            if performance_ratio > 1.5 and remaining_days > total_duration * 0.2:
                recommendations.append("Consider stretching targets for greater impact")
            
            return {
                "goal_id": goal_id,
                "status": status,
                "progress_data": progress_data,
                "time_analysis": {
                    "total_duration_days": total_duration,
                    "elapsed_days": elapsed_days,
                    "remaining_days": remaining_days,
                    "time_progress": round(time_progress, 2)
                },
                "performance_analysis": {
                    "expected_progress": round(expected_progress, 2),
                    "actual_progress": round(actual_progress, 2),
                    "performance_ratio": round(performance_ratio, 2)
                },
                "risk_factors": risk_factors,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze goal achievement: {e}")
            return {"error": str(e)}
    
    # Historical Integration Methods
    
    def integrate_historical_metrics(self, goal_id: str, metric_history: Dict[str, List[Tuple[datetime, float]]]) -> bool:
        """
        Integrate historical metric data for target progress tracking
        
        Args:
            goal_id: Goal identifier
            metric_history: Dictionary mapping metric names to historical data points
            
        Returns:
            True if integration successful, False otherwise
        """
        try:
            targets = self.get_goal_targets(goal_id)
            
            for target in targets:
                if target.metric_name in metric_history:
                    history = metric_history[target.metric_name]
                    
                    # Update current value with latest data point
                    if history:
                        latest_date, latest_value = max(history, key=lambda x: x[0])
                        self.update_target_progress(target.target_id, latest_value)
                    
                    # Store historical data in Neo4j
                    with self.driver.session(database=self.database) as session:
                        session.execute_write(self._store_metric_history_tx, target.target_id, history)
            
            logger.info(f"Integrated historical metrics for goal: {goal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate historical metrics: {e}")
            return False
    
    def _store_metric_history_tx(self, tx: Transaction, target_id: str, history: List[Tuple[datetime, float]]):
        """Transaction to store metric history"""
        # First, remove existing history
        tx.run(
            "MATCH (t:Target {target_id: })-[:HAS_HISTORY]->(h:MetricHistory) DETACH DELETE h",
            target_id=target_id
        )
        
        # Add new history points
        for timestamp, value in history:
            tx.run(
                """
                MATCH (t:Target {target_id: })
                CREATE (h:MetricHistory {
                    timestamp: datetime(),
                    value: 
                })
                CREATE (t)-[:HAS_HISTORY]->(h)
                """,
                target_id=target_id,
                timestamp=timestamp.isoformat(),
                value=value
            )
    
    def get_metric_trend(self, target_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get metric trend analysis
        
        Args:
            target_id: Target identifier
            days: Number of days to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (t:Target {target_id: })-[:HAS_HISTORY]->(h:MetricHistory)
                    WHERE h.timestamp >= datetime() - duration({days: })
                    RETURN h.timestamp as timestamp, h.value as value
                    ORDER BY h.timestamp
                    """,
                    target_id=target_id,
                    days=days
                )
                
                data_points = [(record["timestamp"], record["value"]) for record in result]
                
                if len(data_points) < 2:
                    return {"trend": "insufficient_data", "data_points": len(data_points)}
                
                # Calculate trend
                values = [point[1] for point in data_points]
                n = len(values)
                
                # Simple linear regression for trend
                x_sum = sum(range(n))
                y_sum = sum(values)
                xy_sum = sum(i * values[i] for i in range(n))
                x2_sum = sum(i * i for i in range(n))
                
                slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
                
                trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
                
                return {
                    "trend": trend,
                    "slope": slope,
                    "data_points": n,
                    "latest_value": values[-1],
                    "period_start": values[0],
                    "period_change": values[-1] - values[0],
                    "period_change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get metric trend: {e}")
            return {"trend": "error", "error": str(e)}
    
    # Query and Reporting Methods
    
    def get_goals_by_status(self, status: GoalStatus, department: Optional[str] = None) -> List[Goal]:
        """
        Get goals by status and optionally by department
        
        Args:
            status: Goal status to filter by
            department: Optional department filter
            
        Returns:
            List of Goal objects
        """
        try:
            query = "MATCH (g:Goal {status: 0})"
            params = {"status": status.value}
            
            if department:
                query += " WHERE g.department = "
                params["department"] = department
            
            query += " RETURN g ORDER BY g.priority DESC, g.created_date DESC"
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                
                goals = []
                for record in result:
                    goals.append(self._dict_to_goal(dict(record["g"])))
                
                return goals
                
        except Exception as e:
            logger.error(f"Failed to get goals by status: {e}")
            return []
    
    def get_goals_by_owner(self, owner_id: str) -> List[Goal]:
        """
        Get all goals for a specific owner
        
        Args:
            owner_id: Owner identifier
            
        Returns:
            List of Goal objects
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (g:Goal {owner_id: })
                    RETURN g
                    ORDER BY g.status, g.priority DESC, g.end_date
                    """,
                    owner_id=owner_id
                )
                
                goals = []
                for record in result:
                    goals.append(self._dict_to_goal(dict(record["g"])))
                
                return goals
                
        except Exception as e:
            logger.error(f"Failed to get goals by owner: {e}")
            return []
    
    def get_overdue_goals(self) -> List[Dict[str, Any]]:
        """
        Get all overdue goals with progress information
        
        Returns:
            List of dictionaries containing goal and progress information
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (g:Goal)
                    WHERE g.end_date < datetime() AND g.status IN ['active', 'draft']
                    RETURN g
                    ORDER BY g.end_date
                    """
                )
                
                overdue_goals = []
                for record in result:
                    goal = self._dict_to_goal(dict(record["g"]))
                    progress = self.calculate_goal_progress(goal.goal_id)
                    
                    overdue_goals.append({
                        "goal": goal.to_dict(),
                        "progress": progress,
                        "days_overdue": (datetime.now() - goal.end_date).days
                    })
                
                return overdue_goals
                
        except Exception as e:
            logger.error(f"Failed to get overdue goals: {e}")
            return []
    
    def generate_dashboard_data(self, department: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard data
        
        Args:
            department: Optional department filter
            
        Returns:
            Dictionary containing dashboard metrics
        """
        try:
            query_filter = f" WHERE g.department = '{department}'" if department else ""
            
            with self.driver.session(database=self.database) as session:
                # Get status counts
                result = session.run(f"""
                    MATCH (g:Goal){query_filter}
                    RETURN g.status as status, count(g) as count
                    """)
                
                status_counts = {record["status"]: record["count"] for record in result}
                
                # Get priority distribution
                result = session.run(f"""
                    MATCH (g:Goal){query_filter}
                    RETURN g.priority as priority, count(g) as count
                    ORDER BY g.priority
                    """)
                
                priority_distribution = {record["priority"]: record["count"] for record in result}
                
                # Get active goals with progress
                active_goals = self.get_goals_by_status(GoalStatus.ACTIVE, department)
                active_goals_progress = []
                
                for goal in active_goals[:10]:  # Top 10 for dashboard
                    progress = self.calculate_goal_progress(goal.goal_id)
                    active_goals_progress.append({
                        "goal_id": goal.goal_id,
                        "title": goal.title,
                        "progress": progress["overall_progress"],
                        "end_date": goal.end_date.isoformat(),
                        "owner_id": goal.owner_id
                    })
                
                # Get overdue goals
                overdue_goals = self.get_overdue_goals()
                if department:
                    overdue_goals = [g for g in overdue_goals if g["goal"]["department"] == department]
                
                return {
                    "status_summary": status_counts,
                    "priority_distribution": priority_distribution,
                    "active_goals_progress": active_goals_progress,
                    "overdue_goals_count": len(overdue_goals),
                    "total_goals": sum(status_counts.values()),
                    "department": department
                }
                
        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")
            return {}
    
    # Utility Methods
    
    def _dict_to_goal(self, data: Dict[str, Any]) -> Goal:
        """Convert dictionary to Goal object"""
        return Goal(
            goal_id=data["goal_id"],
            title=data["title"],
            description=data["description"],
            goal_type=GoalType(data["goal_type"]),
            status=GoalStatus(data["status"]),
            created_date=data["created_date"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            owner_id=data["owner_id"],
            department=data["department"],
            priority=data["priority"],
            parent_goal_id=data.get("parent_goal_id"),
            metadata=data.get("metadata", {})
        )
    
    def _dict_to_target(self, data: Dict[str, Any]) -> Target:
        """Convert dictionary to Target object"""
        return Target(
            target_id=data["target_id"],
            goal_id=data["goal_id"],
            metric_name=data["metric_name"],
            target_value=data["target_value"],
            current_value=data["current_value"],
            target_type=TargetType(data["target_type"]),
            unit=MeasurementUnit(data["unit"]),
            baseline_value=data.get("baseline_value"),
            threshold_values=data.get("threshold_values", {}),
            weight=data.get("weight", 1.0)
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform system health check
        
        Returns:
            Dictionary containing health status
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Test connection
                result = session.run("RETURN 1 as test")
                result.single()
                
                # Get statistics
                stats_result = session.run(
                    """
                    MATCH (g:Goal)
                    OPTIONAL MATCH (g)<-[:TARGET_OF]-(t:Target)
                    RETURN count(DISTINCT g) as goals_count, count(t) as targets_count
                    """
                )
                
                stats = stats_result.single()
                
                return {
                    "status": "healthy",
                    "database_connection": "ok",
                    "goals_count": stats["goals_count"],
                    "targets_count": stats["targets_count"],
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
