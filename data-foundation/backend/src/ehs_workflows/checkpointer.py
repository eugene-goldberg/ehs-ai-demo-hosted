"""
Workflow checkpointer module for LlamaIndex compatibility.
This module provides checkpointing functionality for workflows.
"""

from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod
import logging
import json
import time
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Represents a workflow checkpoint."""
    
    checkpoint_id: str
    workflow_id: str
    step_name: str
    state: Dict[str, Any]
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "step_name": self.step_name,
            "state": self.state,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data["workflow_id"],
            step_name=data["step_name"],
            state=data["state"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )


class CheckpointCallback(ABC):
    """Abstract base class for checkpoint callbacks."""
    
    @abstractmethod
    def on_checkpoint_created(self, checkpoint: Checkpoint) -> None:
        """Called when a checkpoint is created."""
        pass
    
    @abstractmethod
    def on_checkpoint_loaded(self, checkpoint: Checkpoint) -> None:
        """Called when a checkpoint is loaded."""
        pass
    
    @abstractmethod
    def on_checkpoint_deleted(self, checkpoint_id: str) -> None:
        """Called when a checkpoint is deleted."""
        pass


class DefaultCheckpointCallback(CheckpointCallback):
    """Default implementation of checkpoint callback."""
    
    def on_checkpoint_created(self, checkpoint: Checkpoint) -> None:
        logger.debug(f"Checkpoint created: {checkpoint.checkpoint_id} for workflow {checkpoint.workflow_id}")
    
    def on_checkpoint_loaded(self, checkpoint: Checkpoint) -> None:
        logger.debug(f"Checkpoint loaded: {checkpoint.checkpoint_id} for workflow {checkpoint.workflow_id}")
    
    def on_checkpoint_deleted(self, checkpoint_id: str) -> None:
        logger.debug(f"Checkpoint deleted: {checkpoint_id}")


class WorkflowCheckpointer:
    """
    Manages workflow checkpoints for state persistence and recovery.
    This is a basic implementation for LlamaIndex compatibility.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        callback: Optional[CheckpointCallback] = None,
        max_checkpoints: int = 10
    ):
        """
        Initialize the workflow checkpointer.
        
        Args:
            storage_path: Path to store checkpoints (optional)
            callback: Checkpoint callback handler
            max_checkpoints: Maximum number of checkpoints to retain per workflow
        """
        self.storage_path = storage_path
        self.callback = callback or DefaultCheckpointCallback()
        self.max_checkpoints = max_checkpoints
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        
        logger.info(f"WorkflowCheckpointer initialized with storage_path={storage_path}")
    
    def create_checkpoint(
        self,
        workflow_id: str,
        step_name: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        Create a new checkpoint for a workflow.
        
        Args:
            workflow_id: Unique identifier for the workflow
            step_name: Name of the current workflow step
            state: Current workflow state
            metadata: Additional metadata for the checkpoint
            
        Returns:
            Created checkpoint
        """
        checkpoint_id = f"{workflow_id}_{step_name}_{int(time.time() * 1000)}"
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            step_name=step_name,
            state=state.copy() if state else {},
            timestamp=time.time(),
            metadata=metadata
        )
        
        # Store checkpoint
        if workflow_id not in self._checkpoints:
            self._checkpoints[workflow_id] = []
        
        self._checkpoints[workflow_id].append(checkpoint)
        
        # Enforce max checkpoints limit
        if len(self._checkpoints[workflow_id]) > self.max_checkpoints:
            old_checkpoint = self._checkpoints[workflow_id].pop(0)
            logger.debug(f"Removed old checkpoint: {old_checkpoint.checkpoint_id}")
        
        # Persist if storage path is configured
        if self.storage_path:
            self._persist_checkpoint(checkpoint)
        
        # Trigger callback
        self.callback.on_checkpoint_created(checkpoint)
        
        logger.debug(f"Created checkpoint {checkpoint_id} for workflow {workflow_id} at step {step_name}")
        return checkpoint
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: ID of the checkpoint to load
            
        Returns:
            Loaded checkpoint or None if not found
        """
        for workflow_checkpoints in self._checkpoints.values():
            for checkpoint in workflow_checkpoints:
                if checkpoint.checkpoint_id == checkpoint_id:
                    self.callback.on_checkpoint_loaded(checkpoint)
                    return checkpoint
        
        # Try loading from persistent storage if available
        if self.storage_path:
            checkpoint = self._load_checkpoint_from_storage(checkpoint_id)
            if checkpoint:
                self.callback.on_checkpoint_loaded(checkpoint)
                return checkpoint
        
        logger.warning(f"Checkpoint not found: {checkpoint_id}")
        return None
    
    def get_latest_checkpoint(self, workflow_id: str) -> Optional[Checkpoint]:
        """
        Get the latest checkpoint for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        if workflow_id not in self._checkpoints or not self._checkpoints[workflow_id]:
            return None
        
        return self._checkpoints[workflow_id][-1]
    
    def list_checkpoints(self, workflow_id: str) -> List[Checkpoint]:
        """
        List all checkpoints for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            List of checkpoints for the workflow
        """
        return self._checkpoints.get(workflow_id, []).copy()
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to delete
            
        Returns:
            True if checkpoint was deleted, False if not found
        """
        for workflow_id, checkpoints in self._checkpoints.items():
            for i, checkpoint in enumerate(checkpoints):
                if checkpoint.checkpoint_id == checkpoint_id:
                    del checkpoints[i]
                    self.callback.on_checkpoint_deleted(checkpoint_id)
                    
                    # Remove from persistent storage if available
                    if self.storage_path:
                        self._delete_checkpoint_from_storage(checkpoint_id)
                    
                    logger.debug(f"Deleted checkpoint: {checkpoint_id}")
                    return True
        
        logger.warning(f"Checkpoint not found for deletion: {checkpoint_id}")
        return False
    
    def clear_workflow_checkpoints(self, workflow_id: str) -> int:
        """
        Clear all checkpoints for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Number of checkpoints cleared
        """
        if workflow_id not in self._checkpoints:
            return 0
        
        count = len(self._checkpoints[workflow_id])
        checkpoint_ids = [cp.checkpoint_id for cp in self._checkpoints[workflow_id]]
        
        del self._checkpoints[workflow_id]
        
        # Notify callback
        for checkpoint_id in checkpoint_ids:
            self.callback.on_checkpoint_deleted(checkpoint_id)
        
        logger.debug(f"Cleared {count} checkpoints for workflow {workflow_id}")
        return count
    
    def _persist_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist checkpoint to storage (placeholder implementation)."""
        # This would implement actual file/database persistence
        logger.debug(f"Persisting checkpoint {checkpoint.checkpoint_id} (not implemented)")
    
    def _load_checkpoint_from_storage(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from persistent storage (placeholder implementation)."""
        # This would implement actual file/database loading
        logger.debug(f"Loading checkpoint {checkpoint_id} from storage (not implemented)")
        return None
    
    def _delete_checkpoint_from_storage(self, checkpoint_id: str) -> None:
        """Delete checkpoint from persistent storage (placeholder implementation)."""
        # This would implement actual file/database deletion
        logger.debug(f"Deleting checkpoint {checkpoint_id} from storage (not implemented)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpointer statistics."""
        total_checkpoints = sum(len(checkpoints) for checkpoints in self._checkpoints.values())
        workflow_count = len(self._checkpoints)
        
        return {
            "total_checkpoints": total_checkpoints,
            "workflow_count": workflow_count,
            "max_checkpoints_per_workflow": self.max_checkpoints,
            "storage_path": self.storage_path,
            "workflows": {
                workflow_id: len(checkpoints)
                for workflow_id, checkpoints in self._checkpoints.items()
            }
        }


# Create default instance for convenience
default_checkpointer = WorkflowCheckpointer()


def create_checkpoint(
    workflow_id: str,
    step_name: str,
    state: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Checkpoint:
    """Convenience function to create a checkpoint using the default checkpointer."""
    return default_checkpointer.create_checkpoint(workflow_id, step_name, state, metadata)


def load_checkpoint(checkpoint_id: str) -> Optional[Checkpoint]:
    """Convenience function to load a checkpoint using the default checkpointer."""
    return default_checkpointer.load_checkpoint(checkpoint_id)


def get_latest_checkpoint(workflow_id: str) -> Optional[Checkpoint]:
    """Convenience function to get the latest checkpoint using the default checkpointer."""
    return default_checkpointer.get_latest_checkpoint(workflow_id)