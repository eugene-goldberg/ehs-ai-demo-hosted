"""
Neo4j-based Recommendation Storage System

This module provides a comprehensive framework for storing, managing, and tracking
EHS recommendations with full lifecycle management, priority scoring, and
effectiveness measurement.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import json
import logging

from neo4j import GraphDatabase, Transaction
from pydantic import BaseModel, Field, validator


class RecommendationStatus(str, Enum):
    """Enumeration of recommendation lifecycle statuses"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(str, Enum):
    """Types of recommendations"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PROCESS_IMPROVEMENT = "process_improvement"
    TRAINING = "training"
    POLICY_UPDATE = "policy_update"
    EQUIPMENT_UPGRADE = "equipment_upgrade"
    BEHAVIORAL_CHANGE = "behavioral_change"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class ImplementationComplexity(str, Enum):
    """Implementation complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ResourceType(str, Enum):
    """Types of resources required"""
    FINANCIAL = "financial"
    HUMAN = "human"
    TIME = "time"
    EQUIPMENT = "equipment"
    TECHNOLOGY = "technology"
    TRAINING = "training"
    EXTERNAL_SERVICES = "external_services"


@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    type: ResourceType
    amount: float
    unit: str
    description: str
    estimated_cost: Optional[float] = None
    availability_date: Optional[datetime] = None


@dataclass
class PriorityScore:
    """Priority scoring breakdown"""
    risk_level: float = 0.0  # 0-100 scale
    potential_impact: float = 0.0  # 0-100 scale
    regulatory_urgency: float = 0.0  # 0-100 scale
    business_value: float = 0.0  # 0-100 scale
    implementation_ease: float = 0.0  # 0-100 scale (reverse scored)
    stakeholder_pressure: float = 0.0  # 0-100 scale
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall priority score"""
        weights = {
            'risk_level': 0.25,
            'potential_impact': 0.20,
            'regulatory_urgency': 0.20,
            'business_value': 0.15,
            'implementation_ease': 0.10,
            'stakeholder_pressure': 0.10
        }
        
        return (
            self.risk_level * weights['risk_level'] +
            self.potential_impact * weights['potential_impact'] +
            self.regulatory_urgency * weights['regulatory_urgency'] +
            self.business_value * weights['business_value'] +
            (100 - self.implementation_ease) * weights['implementation_ease'] +  # Reverse scoring
            self.stakeholder_pressure * weights['stakeholder_pressure']
        )


@dataclass
class ApprovalWorkflow:
    """Approval workflow tracking"""
    workflow_id: str
    current_stage: str
    required_approvers: List[str]
    completed_approvals: List[Dict[str, Any]] = field(default_factory=list)
    pending_approvals: List[str] = field(default_factory=list)
    rejection_history: List[Dict[str, Any]] = field(default_factory=list)
    workflow_started: datetime = field(default_factory=datetime.utcnow)
    workflow_completed: Optional[datetime] = None


@dataclass
class EffectivenessMeasurement:
    """Effectiveness measurement framework"""
    measurement_id: str
    baseline_metrics: Dict[str, float]
    target_metrics: Dict[str, float]
    actual_metrics: Dict[str, float] = field(default_factory=dict)
    measurement_dates: Dict[str, datetime] = field(default_factory=dict)
    effectiveness_score: Optional[float] = None
    roi_analysis: Optional[Dict[str, Any]] = None
    lessons_learned: List[str] = field(default_factory=list)


class RecommendationModel(BaseModel):
    """Pydantic model for recommendation data validation"""
    
    recommendation_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10)
    recommendation_type: RecommendationType
    status: RecommendationStatus = RecommendationStatus.DRAFT
    priority: RecommendationPriority = RecommendationPriority.MEDIUM
    
    # Scoring and assessment
    priority_scores: Dict[str, float] = Field(default_factory=dict)
    implementation_complexity: ImplementationComplexity = ImplementationComplexity.MODERATE
    estimated_effort_hours: Optional[float] = None
    estimated_duration_days: Optional[int] = None
    
    # Resource requirements
    resource_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_total_cost: Optional[float] = None
    
    # Relationships
    related_incidents: List[str] = Field(default_factory=list)
    related_trends: List[str] = Field(default_factory=list)
    related_goals: List[str] = Field(default_factory=list)
    dependent_recommendations: List[str] = Field(default_factory=list)
    blocking_recommendations: List[str] = Field(default_factory=list)
    
    # Lifecycle tracking
    created_by: str
    created_date: datetime = Field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    implementation_start_date: Optional[datetime] = None
    implementation_end_date: Optional[datetime] = None
    
    # Approval workflow
    approval_workflow: Optional[Dict[str, Any]] = None
    approval_required: bool = True
    
    # Effectiveness measurement
    effectiveness_measurement: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list)
    department: Optional[str] = None
    location: Optional[str] = None
    regulatory_references: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RecommendationStorage:
    """Neo4j-based recommendation storage and management system"""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = logging.getLogger(__name__)
        self._ensure_schema()
    
    def close(self):
        """Close Neo4j driver"""
        self.driver.close()
    
    def _ensure_schema(self):
        """Ensure Neo4j schema exists with proper indexes and constraints"""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT recommendation_id_unique IF NOT EXISTS FOR (r:Recommendation) REQUIRE r.recommendation_id IS UNIQUE",
                "CREATE CONSTRAINT approval_workflow_id_unique IF NOT EXISTS FOR (a:ApprovalWorkflow) REQUIRE a.workflow_id IS UNIQUE",
                "CREATE CONSTRAINT effectiveness_measurement_id_unique IF NOT EXISTS FOR (e:EffectivenessMeasurement) REQUIRE e.measurement_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    self.logger.warning(f"Constraint creation failed (may already exist): {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX recommendation_status_idx IF NOT EXISTS FOR (r:Recommendation) ON (r.status)",
                "CREATE INDEX recommendation_priority_idx IF NOT EXISTS FOR (r:Recommendation) ON (r.priority)",
                "CREATE INDEX recommendation_type_idx IF NOT EXISTS FOR (r:Recommendation) ON (r.recommendation_type)",
                "CREATE INDEX recommendation_created_date_idx IF NOT EXISTS FOR (r:Recommendation) ON (r.created_date)",
                "CREATE INDEX recommendation_due_date_idx IF NOT EXISTS FOR (r:Recommendation) ON (r.due_date)",
                "CREATE INDEX recommendation_department_idx IF NOT EXISTS FOR (r:Recommendation) ON (r.department)",
                "CREATE INDEX recommendation_assigned_to_idx IF NOT EXISTS FOR (r:Recommendation) ON (r.assigned_to)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    self.logger.warning(f"Index creation failed (may already exist): {e}")
    
    def create_recommendation(self, recommendation: RecommendationModel) -> str:
        """Create a new recommendation"""
        with self.driver.session() as session:
            return session.execute_write(self._create_recommendation_tx, recommendation)
    
    def _create_recommendation_tx(self, tx: Transaction, recommendation: RecommendationModel) -> str:
        """Transaction for creating recommendation"""
        
        # Calculate priority score
        priority_score_obj = self._calculate_priority_score(recommendation.priority_scores)
        
        # Create main recommendation node
        query = """
        CREATE (r:Recommendation {
            recommendation_id: $recommendation_id,
            title: $title,
            description: $description,
            recommendation_type: $recommendation_type,
            status: $status,
            priority: $priority,
            priority_score: $priority_score,
            implementation_complexity: $implementation_complexity,
            estimated_effort_hours: $estimated_effort_hours,
            estimated_duration_days: $estimated_duration_days,
            estimated_total_cost: $estimated_total_cost,
            created_by: $created_by,
            created_date: $created_date,
            assigned_to: $assigned_to,
            due_date: $due_date,
            implementation_start_date: $implementation_start_date,
            implementation_end_date: $implementation_end_date,
            approval_required: $approval_required,
            tags: $tags,
            department: $department,
            location: $location,
            regulatory_references: $regulatory_references,
            priority_scores: $priority_scores_json,
            resource_requirements: $resource_requirements_json
        })
        RETURN r.recommendation_id
        """
        
        result = tx.run(query, {
            'recommendation_id': recommendation.recommendation_id,
            'title': recommendation.title,
            'description': recommendation.description,
            'recommendation_type': recommendation.recommendation_type.value,
            'status': recommendation.status.value,
            'priority': recommendation.priority.value,
            'priority_score': priority_score_obj.overall_score,
            'implementation_complexity': recommendation.implementation_complexity.value,
            'estimated_effort_hours': recommendation.estimated_effort_hours,
            'estimated_duration_days': recommendation.estimated_duration_days,
            'estimated_total_cost': recommendation.estimated_total_cost,
            'created_by': recommendation.created_by,
            'created_date': recommendation.created_date.isoformat(),
            'assigned_to': recommendation.assigned_to,
            'due_date': recommendation.due_date.isoformat() if recommendation.due_date else None,
            'implementation_start_date': recommendation.implementation_start_date.isoformat() if recommendation.implementation_start_date else None,
            'implementation_end_date': recommendation.implementation_end_date.isoformat() if recommendation.implementation_end_date else None,
            'approval_required': recommendation.approval_required,
            'tags': recommendation.tags,
            'department': recommendation.department,
            'location': recommendation.location,
            'regulatory_references': recommendation.regulatory_references,
            'priority_scores_json': json.dumps(recommendation.priority_scores),
            'resource_requirements_json': json.dumps(recommendation.resource_requirements)
        })
        
        recommendation_id = result.single()['r.recommendation_id']
        
        # Create relationships
        self._create_recommendation_relationships(tx, recommendation)
        
        # Initialize approval workflow if required
        if recommendation.approval_required:
            self._initialize_approval_workflow(tx, recommendation_id)
        
        return recommendation_id
    
    def _calculate_priority_score(self, priority_scores: Dict[str, float]) -> PriorityScore:
        """Calculate comprehensive priority score"""
        return PriorityScore(
            risk_level=priority_scores.get('risk_level', 0.0),
            potential_impact=priority_scores.get('potential_impact', 0.0),
            regulatory_urgency=priority_scores.get('regulatory_urgency', 0.0),
            business_value=priority_scores.get('business_value', 0.0),
            implementation_ease=priority_scores.get('implementation_ease', 0.0),
            stakeholder_pressure=priority_scores.get('stakeholder_pressure', 0.0)
        )
    
    def _create_recommendation_relationships(self, tx: Transaction, recommendation: RecommendationModel):
        """Create relationships to incidents, trends, and goals"""
        
        # Link to related incidents
        for incident_id in recommendation.related_incidents:
            tx.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})
                MATCH (i:Incident {incident_id: $incident_id})
                CREATE (r)-[:ADDRESSES_INCIDENT]->(i)
            """, rec_id=recommendation.recommendation_id, incident_id=incident_id)
        
        # Link to related trends
        for trend_id in recommendation.related_trends:
            tx.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})
                MATCH (t:Trend {trend_id: $trend_id})
                CREATE (r)-[:ADDRESSES_TREND]->(t)
            """, rec_id=recommendation.recommendation_id, trend_id=trend_id)
        
        # Link to related goals
        for goal_id in recommendation.related_goals:
            tx.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})
                MATCH (g:Goal {goal_id: $goal_id})
                CREATE (r)-[:SUPPORTS_GOAL]->(g)
            """, rec_id=recommendation.recommendation_id, goal_id=goal_id)
        
        # Link dependencies
        for dep_id in recommendation.dependent_recommendations:
            tx.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})
                MATCH (d:Recommendation {recommendation_id: $dep_id})
                CREATE (r)-[:DEPENDS_ON]->(d)
            """, rec_id=recommendation.recommendation_id, dep_id=dep_id)
        
        # Link blocking relationships
        for blocking_id in recommendation.blocking_recommendations:
            tx.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})
                MATCH (b:Recommendation {recommendation_id: $blocking_id})
                CREATE (r)-[:BLOCKS]->(b)
            """, rec_id=recommendation.recommendation_id, blocking_id=blocking_id)
    
    def _initialize_approval_workflow(self, tx: Transaction, recommendation_id: str):
        """Initialize approval workflow"""
        workflow_id = str(uuid4())
        
        tx.run("""
            CREATE (w:ApprovalWorkflow {
                workflow_id: $workflow_id,
                current_stage: $current_stage,
                workflow_started: $workflow_started,
                required_approvers: $required_approvers,
                completed_approvals: $completed_approvals,
                pending_approvals: $pending_approvals,
                rejection_history: $rejection_history
            })
        """, {
            'workflow_id': workflow_id,
            'current_stage': 'initial_review',
            'workflow_started': datetime.utcnow().isoformat(),
            'required_approvers': json.dumps(['supervisor', 'safety_manager']),
            'completed_approvals': json.dumps([]),
            'pending_approvals': json.dumps(['supervisor', 'safety_manager']),
            'rejection_history': json.dumps([])
        })
        
        # Link workflow to recommendation
        tx.run("""
            MATCH (r:Recommendation {recommendation_id: $rec_id})
            MATCH (w:ApprovalWorkflow {workflow_id: $workflow_id})
            CREATE (r)-[:HAS_WORKFLOW]->(w)
        """, rec_id=recommendation_id, workflow_id=workflow_id)
    
    def update_recommendation_status(self, recommendation_id: str, new_status: RecommendationStatus, 
                                   updated_by: str, notes: Optional[str] = None) -> bool:
        """Update recommendation status with audit trail"""
        with self.driver.session() as session:
            return session.execute_write(self._update_status_tx, recommendation_id, new_status, updated_by, notes)
    
    def _update_status_tx(self, tx: Transaction, recommendation_id: str, new_status: RecommendationStatus, 
                         updated_by: str, notes: Optional[str]) -> bool:
        """Transaction for updating recommendation status"""
        
        # Get current status for audit trail
        result = tx.run("""
            MATCH (r:Recommendation {recommendation_id: $rec_id})
            RETURN r.status as current_status
        """, rec_id=recommendation_id)
        
        record = result.single()
        if not record:
            return False
        
        current_status = record['current_status']
        
        # Update status
        tx.run("""
            MATCH (r:Recommendation {recommendation_id: $rec_id})
            SET r.status = $new_status,
                r.last_updated = $timestamp,
                r.last_updated_by = $updated_by
        """, {
            'rec_id': recommendation_id,
            'new_status': new_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'updated_by': updated_by
        })
        
        # Create audit trail entry
        tx.run("""
            MATCH (r:Recommendation {recommendation_id: $rec_id})
            CREATE (a:StatusAudit {
                audit_id: $audit_id,
                previous_status: $previous_status,
                new_status: $new_status,
                changed_by: $changed_by,
                change_timestamp: $timestamp,
                notes: $notes
            })
            CREATE (r)-[:HAS_AUDIT_ENTRY]->(a)
        """, {
            'rec_id': recommendation_id,
            'audit_id': str(uuid4()),
            'previous_status': current_status,
            'new_status': new_status.value,
            'changed_by': updated_by,
            'timestamp': datetime.utcnow().isoformat(),
            'notes': notes
        })
        
        return True
    
    def process_approval(self, recommendation_id: str, approver: str, approved: bool, 
                        comments: Optional[str] = None) -> bool:
        """Process approval decision"""
        with self.driver.session() as session:
            return session.execute_write(self._process_approval_tx, recommendation_id, approver, approved, comments)
    
    def _process_approval_tx(self, tx: Transaction, recommendation_id: str, approver: str, 
                           approved: bool, comments: Optional[str]) -> bool:
        """Transaction for processing approval"""
        
        # Get workflow
        result = tx.run("""
            MATCH (r:Recommendation {recommendation_id: $rec_id})-[:HAS_WORKFLOW]->(w:ApprovalWorkflow)
            RETURN w.workflow_id as workflow_id, w.pending_approvals as pending_approvals,
                   w.completed_approvals as completed_approvals, w.rejection_history as rejection_history
        """, rec_id=recommendation_id)
        
        record = result.single()
        if not record:
            return False
        
        workflow_id = record['workflow_id']
        pending_approvals = json.loads(record['pending_approvals'])
        completed_approvals = json.loads(record['completed_approvals'])
        rejection_history = json.loads(record['rejection_history'])
        
        if approver not in pending_approvals:
            return False  # Approver not authorized
        
        # Process approval/rejection
        approval_entry = {
            'approver': approver,
            'approved': approved,
            'timestamp': datetime.utcnow().isoformat(),
            'comments': comments
        }
        
        if approved:
            completed_approvals.append(approval_entry)
            pending_approvals.remove(approver)
            
            # Check if all approvals completed
            if not pending_approvals:
                # All approved - update recommendation status
                tx.run("""
                    MATCH (r:Recommendation {recommendation_id: $rec_id})
                    SET r.status = 'approved'
                """, rec_id=recommendation_id)
                
                # Mark workflow as completed
                tx.run("""
                    MATCH (w:ApprovalWorkflow {workflow_id: $workflow_id})
                    SET w.current_stage = 'completed',
                        w.workflow_completed = $timestamp
                """, workflow_id=workflow_id, timestamp=datetime.utcnow().isoformat())
        else:
            rejection_history.append(approval_entry)
            # Rejection - update recommendation status
            tx.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})
                SET r.status = 'rejected'
            """, rec_id=recommendation_id)
        
        # Update workflow
        tx.run("""
            MATCH (w:ApprovalWorkflow {workflow_id: $workflow_id})
            SET w.pending_approvals = $pending_approvals,
                w.completed_approvals = $completed_approvals,
                w.rejection_history = $rejection_history
        """, {
            'workflow_id': workflow_id,
            'pending_approvals': json.dumps(pending_approvals),
            'completed_approvals': json.dumps(completed_approvals),
            'rejection_history': json.dumps(rejection_history)
        })
        
        return True
    
    def start_implementation(self, recommendation_id: str, implementer: str, 
                           start_date: Optional[datetime] = None) -> bool:
        """Start recommendation implementation"""
        if start_date is None:
            start_date = datetime.utcnow()
        
        with self.driver.session() as session:
            result = session.execute_write("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})
                WHERE r.status = 'approved'
                SET r.status = 'in_progress',
                    r.implementation_start_date = $start_date,
                    r.assigned_to = $implementer,
                    r.last_updated = $timestamp,
                    r.last_updated_by = $implementer
                RETURN r.recommendation_id
            """, {
                'rec_id': recommendation_id,
                'start_date': start_date.isoformat(),
                'implementer': implementer,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return result.single() is not None
    
    def complete_implementation(self, recommendation_id: str, completed_by: str,
                              effectiveness_data: Optional[Dict[str, Any]] = None) -> bool:
        """Complete recommendation implementation and initialize effectiveness measurement"""
        completion_date = datetime.utcnow()
        
        with self.driver.session() as session:
            return session.execute_write(self._complete_implementation_tx, recommendation_id, 
                                       completed_by, completion_date, effectiveness_data)
    
    def _complete_implementation_tx(self, tx: Transaction, recommendation_id: str, completed_by: str,
                                   completion_date: datetime, effectiveness_data: Optional[Dict[str, Any]]) -> bool:
        """Transaction for completing implementation"""
        
        # Update recommendation status
        result = tx.run("""
            MATCH (r:Recommendation {recommendation_id: $rec_id})
            WHERE r.status = 'in_progress'
            SET r.status = 'completed',
                r.implementation_end_date = $completion_date,
                r.last_updated = $timestamp,
                r.last_updated_by = $completed_by
            RETURN r.recommendation_id
        """, {
            'rec_id': recommendation_id,
            'completion_date': completion_date.isoformat(),
            'timestamp': completion_date.isoformat(),
            'completed_by': completed_by
        })
        
        if not result.single():
            return False
        
        # Initialize effectiveness measurement if data provided
        if effectiveness_data:
            measurement_id = str(uuid4())
            
            tx.run("""
                CREATE (e:EffectivenessMeasurement {
                    measurement_id: $measurement_id,
                    baseline_metrics: $baseline_metrics,
                    target_metrics: $target_metrics,
                    actual_metrics: $actual_metrics,
                    measurement_started: $start_date,
                    created_by: $completed_by
                })
            """, {
                'measurement_id': measurement_id,
                'baseline_metrics': json.dumps(effectiveness_data.get('baseline_metrics', {})),
                'target_metrics': json.dumps(effectiveness_data.get('target_metrics', {})),
                'actual_metrics': json.dumps({}),
                'start_date': completion_date.isoformat(),
                'completed_by': completed_by
            })
            
            # Link to recommendation
            tx.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})
                MATCH (e:EffectivenessMeasurement {measurement_id: $measurement_id})
                CREATE (r)-[:HAS_EFFECTIVENESS_MEASUREMENT]->(e)
            """, rec_id=recommendation_id, measurement_id=measurement_id)
        
        return True
    
    def update_effectiveness_measurement(self, recommendation_id: str, metrics: Dict[str, float],
                                       measurement_date: Optional[datetime] = None) -> bool:
        """Update effectiveness measurement with new metrics"""
        if measurement_date is None:
            measurement_date = datetime.utcnow()
        
        with self.driver.session() as session:
            return session.execute_write(self._update_effectiveness_tx, recommendation_id, 
                                       metrics, measurement_date)
    
    def _update_effectiveness_tx(self, tx: Transaction, recommendation_id: str, 
                               metrics: Dict[str, float], measurement_date: datetime) -> bool:
        """Transaction for updating effectiveness measurement"""
        
        result = tx.run("""
            MATCH (r:Recommendation {recommendation_id: $rec_id})-[:HAS_EFFECTIVENESS_MEASUREMENT]->(e:EffectivenessMeasurement)
            RETURN e.measurement_id as measurement_id, e.actual_metrics as current_metrics,
                   e.baseline_metrics as baseline_metrics, e.target_metrics as target_metrics
        """, rec_id=recommendation_id)
        
        record = result.single()
        if not record:
            return False
        
        current_metrics = json.loads(record['current_metrics'])
        baseline_metrics = json.loads(record['baseline_metrics'])
        target_metrics = json.loads(record['target_metrics'])
        
        # Update metrics with new data
        current_metrics.update(metrics)
        
        # Calculate effectiveness score
        effectiveness_score = self._calculate_effectiveness_score(baseline_metrics, target_metrics, current_metrics)
        
        # Update measurement
        tx.run("""
            MATCH (e:EffectivenessMeasurement {measurement_id: $measurement_id})
            SET e.actual_metrics = $actual_metrics,
                e.effectiveness_score = $effectiveness_score,
                e.last_measured = $measurement_date
        """, {
            'measurement_id': record['measurement_id'],
            'actual_metrics': json.dumps(current_metrics),
            'effectiveness_score': effectiveness_score,
            'measurement_date': measurement_date.isoformat()
        })
        
        return True
    
    def _calculate_effectiveness_score(self, baseline: Dict[str, float], target: Dict[str, float], 
                                     actual: Dict[str, float]) -> float:
        """Calculate effectiveness score based on target achievement"""
        if not target or not actual:
            return 0.0
        
        scores = []
        for metric, target_value in target.items():
            if metric in actual and metric in baseline:
                baseline_value = baseline[metric]
                actual_value = actual[metric]
                
                # Calculate improvement percentage
                if target_value > baseline_value:  # Higher is better
                    improvement = (actual_value - baseline_value) / (target_value - baseline_value)
                else:  # Lower is better
                    improvement = (baseline_value - actual_value) / (baseline_value - target_value)
                
                scores.append(min(max(improvement, 0.0), 1.0))  # Clamp between 0 and 1
        
        return sum(scores) / len(scores) * 100 if scores else 0.0
    
    def get_recommendations_by_status(self, status: RecommendationStatus, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recommendations by status"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:Recommendation {status: $status})
                RETURN r
                ORDER BY r.priority_score DESC, r.created_date DESC
                LIMIT $limit
            """, status=status.value, limit=limit)
            
            return [record['r'] for record in result]
    
    def get_recommendations_by_priority(self, priority: RecommendationPriority, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recommendations by priority"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:Recommendation {priority: $priority})
                RETURN r
                ORDER BY r.priority_score DESC, r.created_date DESC
                LIMIT $limit
            """, priority=priority.value, limit=limit)
            
            return [record['r'] for record in result]
    
    def get_overdue_recommendations(self, as_of_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get overdue recommendations"""
        if as_of_date is None:
            as_of_date = datetime.utcnow()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:Recommendation)
                WHERE r.due_date < $as_of_date 
                AND r.status IN ['approved', 'in_progress']
                RETURN r
                ORDER BY r.due_date ASC
            """, as_of_date=as_of_date.isoformat())
            
            return [record['r'] for record in result]
    
    def get_recommendation_analytics(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Get recommendation analytics for specified time period"""
        start_date = datetime.utcnow() - timedelta(days=time_period_days)
        
        with self.driver.session() as session:
            # Status distribution
            status_result = session.run("""
                MATCH (r:Recommendation)
                WHERE r.created_date >= $start_date
                RETURN r.status as status, count(*) as count
            """, start_date=start_date.isoformat())
            
            status_distribution = {record['status']: record['count'] for record in status_result}
            
            # Priority distribution
            priority_result = session.run("""
                MATCH (r:Recommendation)
                WHERE r.created_date >= $start_date
                RETURN r.priority as priority, count(*) as count
            """, start_date=start_date.isoformat())
            
            priority_distribution = {record['priority']: record['count'] for record in priority_result}
            
            # Type distribution
            type_result = session.run("""
                MATCH (r:Recommendation)
                WHERE r.created_date >= $start_date
                RETURN r.recommendation_type as type, count(*) as count
            """, start_date=start_date.isoformat())
            
            type_distribution = {record['type']: record['count'] for record in type_result}
            
            # Completion metrics
            completion_result = session.run("""
                MATCH (r:Recommendation)
                WHERE r.created_date >= $start_date
                AND r.status = 'completed'
                AND r.implementation_start_date IS NOT NULL
                AND r.implementation_end_date IS NOT NULL
                RETURN avg(duration.between(date(r.implementation_start_date), date(r.implementation_end_date)).days) as avg_completion_time,
                       count(*) as completed_count
            """, start_date=start_date.isoformat())
            
            completion_record = completion_result.single()
            
            # Effectiveness scores
            effectiveness_result = session.run("""
                MATCH (r:Recommendation)-[:HAS_EFFECTIVENESS_MEASUREMENT]->(e:EffectivenessMeasurement)
                WHERE r.created_date >= $start_date
                AND e.effectiveness_score IS NOT NULL
                RETURN avg(e.effectiveness_score) as avg_effectiveness,
                       count(*) as measured_count
            """, start_date=start_date.isoformat())
            
            effectiveness_record = effectiveness_result.single()
            
            return {
                'time_period_days': time_period_days,
                'status_distribution': status_distribution,
                'priority_distribution': priority_distribution,
                'type_distribution': type_distribution,
                'avg_completion_time_days': completion_record['avg_completion_time'] if completion_record else None,
                'completed_recommendations': completion_record['completed_count'] if completion_record else 0,
                'avg_effectiveness_score': effectiveness_record['avg_effectiveness'] if effectiveness_record else None,
                'measured_recommendations': effectiveness_record['measured_count'] if effectiveness_record else 0,
                'generated_at': datetime.utcnow().isoformat()
            }
    
    def get_recommendation_dependencies(self, recommendation_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get recommendation dependencies and blocking relationships"""
        with self.driver.session() as session:
            # Dependencies (what this recommendation depends on)
            deps_result = session.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})-[:DEPENDS_ON]->(dep:Recommendation)
                RETURN dep
            """, rec_id=recommendation_id)
            
            dependencies = [record['dep'] for record in deps_result]
            
            # Blocked by (what recommendations are blocked by this one)
            blocked_result = session.run("""
                MATCH (r:Recommendation {recommendation_id: $rec_id})-[:BLOCKS]->(blocked:Recommendation)
                RETURN blocked
            """, rec_id=recommendation_id)
            
            blocked_recommendations = [record['blocked'] for record in blocked_result]
            
            return {
                'dependencies': dependencies,
                'blocks': blocked_recommendations
            }
    
    def archive_old_recommendations(self, days_old: int = 365, dry_run: bool = True) -> Dict[str, Any]:
        """Archive old completed recommendations"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        with self.driver.session() as session:
            # Find recommendations to archive
            result = session.run("""
                MATCH (r:Recommendation)
                WHERE r.status = 'completed'
                AND r.implementation_end_date < $cutoff_date
                RETURN count(*) as archive_count, collect(r.recommendation_id) as recommendation_ids
            """, cutoff_date=cutoff_date.isoformat())
            
            record = result.single()
            archive_count = record['archive_count']
            recommendation_ids = record['recommendation_ids']
            
            if not dry_run and archive_count > 0:
                # Actually archive the recommendations
                session.run("""
                    MATCH (r:Recommendation)
                    WHERE r.recommendation_id IN $recommendation_ids
                    SET r.status = 'archived', r.archived_date = $archive_date
                """, {
                    'recommendation_ids': recommendation_ids,
                    'archive_date': datetime.utcnow().isoformat()
                })
            
            return {
                'archive_count': archive_count,
                'recommendation_ids': recommendation_ids,
                'dry_run': dry_run,
                'cutoff_date': cutoff_date.isoformat()
            }


# Example usage and factory functions
def create_recommendation_storage(neo4j_uri: str, username: str, password: str) -> RecommendationStorage:
    """Factory function to create recommendation storage instance"""
    return RecommendationStorage(neo4j_uri, username, password)


def create_sample_recommendation(title: str, description: str, created_by: str) -> RecommendationModel:
    """Factory function to create a sample recommendation"""
    return RecommendationModel(
        title=title,
        description=description,
        recommendation_type=RecommendationType.PREVENTIVE,
        priority=RecommendationPriority.MEDIUM,
        created_by=created_by,
        priority_scores={
            'risk_level': 60.0,
            'potential_impact': 70.0,
            'regulatory_urgency': 50.0,
            'business_value': 80.0,
            'implementation_ease': 60.0,
            'stakeholder_pressure': 40.0
        }
    )


if __name__ == "__main__":
    # Example usage
    storage = create_recommendation_storage("bolt://localhost:7687", "neo4j", "password")
    
    # Create sample recommendation
    recommendation = create_sample_recommendation(
        "Implement Safety Training Program",
        "Develop comprehensive safety training program to reduce workplace incidents",
        "safety_manager"
    )
    
    try:
        rec_id = storage.create_recommendation(recommendation)
        print(f"Created recommendation: {rec_id}")
        
        # Get analytics
        analytics = storage.get_recommendation_analytics()
        print(f"Analytics: {analytics}")
        
    finally:
        storage.close()