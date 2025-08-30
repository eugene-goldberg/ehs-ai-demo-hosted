"""
Workflow context module for LlamaIndex compatibility.
This module provides context management functionality for workflows.
"""

from typing import Any, Dict, Optional, List, Union, Callable
from contextlib import contextmanager
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkflowEvent:
    """Represents a workflow event."""
    
    event_id: str
    event_type: str
    source_step: str
    target_step: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Context:
    """
    Workflow context for managing state, events, and communication between workflow steps.
    This provides compatibility with LlamaIndex workflow system.
    """
    
    def __init__(
        self,
        workflow_id: Optional[str] = None,
        parent_context: Optional["Context"] = None
    ):
        """
        Initialize workflow context.
        
        Args:
            workflow_id: Unique identifier for the workflow
            parent_context: Parent context if this is a child context
        """
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.parent_context = parent_context
        self._state: Dict[str, Any] = {}
        self._events: List[WorkflowEvent] = []
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._step_stack: List[str] = []
        self._metadata: Dict[str, Any] = {}
        
        logger.debug(f"Context initialized with workflow_id: {self.workflow_id}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context state.
        
        Args:
            key: State key
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the context state.
        
        Args:
            key: State key
            value: State value
        """
        with self._lock:
            self._state[key] = value
            logger.debug(f"Context state updated: {key} = {type(value).__name__}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple state values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        with self._lock:
            self._state.update(updates)
            logger.debug(f"Context state batch updated: {len(updates)} keys")
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the context state.
        
        Args:
            key: State key to delete
            
        Returns:
            True if key was deleted, False if key didn't exist
        """
        with self._lock:
            if key in self._state:
                del self._state[key]
                logger.debug(f"Context state key deleted: {key}")
                return True
            return False
    
    def has(self, key: str) -> bool:
        """
        Check if a key exists in the context state.
        
        Args:
            key: State key to check
            
        Returns:
            True if key exists, False otherwise
        """
        with self._lock:
            return key in self._state
    
    def keys(self) -> List[str]:
        """
        Get all state keys.
        
        Returns:
            List of state keys
        """
        with self._lock:
            return list(self._state.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context state to dictionary.
        
        Returns:
            Dictionary representation of state
        """
        with self._lock:
            return self._state.copy()
    
    def emit_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        target_step: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowEvent:
        """
        Emit an event to the workflow.
        
        Args:
            event_type: Type of event
            data: Event data
            target_step: Target step for the event
            metadata: Event metadata
            
        Returns:
            Created event
        """
        with self._lock:
            current_step = self._step_stack[-1] if self._step_stack else "unknown"
            
            event = WorkflowEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                source_step=current_step,
                target_step=target_step,
                data=data or {},
                metadata=metadata or {}
            )
            
            self._events.append(event)
            
            # Notify subscribers
            for callback in self._subscribers.get(event_type, []):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {e}")
            
            logger.debug(f"Event emitted: {event_type} from {current_step}")
            return event
    
    def subscribe(self, event_type: str, callback: Callable[[WorkflowEvent], None]) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Callback function to handle events
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable[[WorkflowEvent], None]) -> bool:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            callback: Callback function to remove
            
        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from event type: {event_type}")
                    return True
                except ValueError:
                    pass
            return False
    
    def get_events(self, event_type: Optional[str] = None) -> List[WorkflowEvent]:
        """
        Get events from the context.
        
        Args:
            event_type: Optional event type filter
            
        Returns:
            List of events, optionally filtered by type
        """
        with self._lock:
            if event_type is None:
                return self._events.copy()
            return [event for event in self._events if event.event_type == event_type]
    
    @contextmanager
    def step(self, step_name: str):
        """
        Context manager for tracking the current workflow step.
        
        Args:
            step_name: Name of the current step
        """
        with self._lock:
            self._step_stack.append(step_name)
            logger.debug(f"Entered step: {step_name}")
        
        try:
            yield
        finally:
            with self._lock:
                if self._step_stack and self._step_stack[-1] == step_name:
                    self._step_stack.pop()
                    logger.debug(f"Exited step: {step_name}")
    
    def get_current_step(self) -> Optional[str]:
        """
        Get the current workflow step.
        
        Returns:
            Current step name or None if no step is active
        """
        with self._lock:
            return self._step_stack[-1] if self._step_stack else None
    
    def get_step_stack(self) -> List[str]:
        """
        Get the current step stack.
        
        Returns:
            List of active steps (most recent last)
        """
        with self._lock:
            return self._step_stack.copy()
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the context.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        with self._lock:
            self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata from the context.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        with self._lock:
            return self._metadata.get(key, default)
    
    def create_child_context(self, child_id: Optional[str] = None) -> "Context":
        """
        Create a child context that inherits from this context.
        
        Args:
            child_id: Optional ID for the child context
            
        Returns:
            New child context
        """
        child_context = Context(
            workflow_id=child_id or f"{self.workflow_id}_child_{uuid.uuid4()}",
            parent_context=self
        )
        
        # Inherit state (shallow copy)
        with self._lock:
            child_context._state = self._state.copy()
            child_context._metadata = self._metadata.copy()
        
        logger.debug(f"Created child context: {child_context.workflow_id}")
        return child_context
    
    def clear(self) -> None:
        """Clear all context state and events."""
        with self._lock:
            self._state.clear()
            self._events.clear()
            self._step_stack.clear()
            self._metadata.clear()
            logger.debug("Context cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get context statistics.
        
        Returns:
            Dictionary with context statistics
        """
        with self._lock:
            return {
                "workflow_id": self.workflow_id,
                "state_keys": len(self._state),
                "events_count": len(self._events),
                "active_steps": len(self._step_stack),
                "current_step": self.get_current_step(),
                "subscribers": {
                    event_type: len(callbacks)
                    for event_type, callbacks in self._subscribers.items()
                },
                "has_parent": self.parent_context is not None,
                "metadata_keys": len(self._metadata)
            }
    
    def __str__(self) -> str:
        """String representation of the context."""
        return f"Context(workflow_id={self.workflow_id}, state_keys={len(self._state)}, events={len(self._events)})"
    
    def __repr__(self) -> str:
        """Detailed representation of the context."""
        return f"Context(workflow_id={self.workflow_id}, state={len(self._state)} keys, events={len(self._events)}, step={self.get_current_step()})"


# Global context registry for workflow management
_context_registry: Dict[str, Context] = {}
_registry_lock = threading.RLock()


def get_context(workflow_id: str) -> Optional[Context]:
    """
    Get a context by workflow ID.
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        Context instance or None if not found
    """
    with _registry_lock:
        return _context_registry.get(workflow_id)


def register_context(context: Context) -> None:
    """
    Register a context in the global registry.
    
    Args:
        context: Context to register
    """
    with _registry_lock:
        _context_registry[context.workflow_id] = context
        logger.debug(f"Context registered: {context.workflow_id}")


def unregister_context(workflow_id: str) -> bool:
    """
    Unregister a context from the global registry.
    
    Args:
        workflow_id: Workflow identifier
        
    Returns:
        True if context was found and removed, False otherwise
    """
    with _registry_lock:
        if workflow_id in _context_registry:
            del _context_registry[workflow_id]
            logger.debug(f"Context unregistered: {workflow_id}")
            return True
        return False


def create_context(workflow_id: Optional[str] = None) -> Context:
    """
    Create and register a new context.
    
    Args:
        workflow_id: Optional workflow identifier
        
    Returns:
        New context instance
    """
    context = Context(workflow_id=workflow_id)
    register_context(context)
    return context


def list_contexts() -> List[str]:
    """
    List all registered context workflow IDs.
    
    Returns:
        List of workflow IDs
    """
    with _registry_lock:
        return list(_context_registry.keys())


def clear_registry() -> int:
    """
    Clear all contexts from the registry.
    
    Returns:
        Number of contexts that were cleared
    """
    with _registry_lock:
        count = len(_context_registry)
        _context_registry.clear()
        logger.debug(f"Context registry cleared: {count} contexts")
        return count