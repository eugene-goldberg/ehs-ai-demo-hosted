"""
Distributed tracing and performance profiling system for EHS Analytics.

This module provides comprehensive tracing capabilities for workflow execution,
span management, performance profiling, and trace context propagation across
distributed components.
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator
from threading import local
import functools

from .logging import get_ehs_logger

logger = get_ehs_logger(__name__)


class SpanKind(Enum):
    """Types of spans in the trace."""
    SERVER = "server"  # Incoming request
    CLIENT = "client"  # Outgoing request
    INTERNAL = "internal"  # Internal operation
    PRODUCER = "producer"  # Message producer
    CONSUMER = "consumer"  # Message consumer


class SpanStatus(Enum):
    """Span status codes."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """Trace context information."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanContext":
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {})
        )


@dataclass
class Span:
    """Individual span representing an operation in a trace."""
    context: SpanContext
    operation_name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: Optional[float] = None
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the span."""
        self.tags[key] = value
    
    def set_tags(self, tags: Dict[str, Any]):
        """Set multiple tags on the span."""
        self.tags.update(tags)
    
    def log(self, event: str, **fields):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            **fields
        }
        self.logs.append(log_entry)
        
        logger.debug(
            f"Span log: {event}",
            trace_id=self.context.trace_id,
            span_id=self.context.span_id,
            operation=self.operation_name,
            **fields
        )
    
    def finish(self, status: Optional[SpanStatus] = None):
        """Finish the span."""
        self.end_time = datetime.utcnow()
        if status:
            self.status = status
        
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        logger.debug(
            "Span finished",
            trace_id=self.context.trace_id,
            span_id=self.context.span_id,
            operation=self.operation_name,
            duration_ms=self.duration_ms,
            status=self.status.value
        )
    
    def set_error(self, error: Exception):
        """Mark span as error with exception details."""
        self.status = SpanStatus.ERROR
        self.set_tags({
            "error": True,
            "error.type": type(error).__name__,
            "error.message": str(error)
        })
        self.log("error", error_type=type(error).__name__, error_message=str(error))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "context": self.context.to_dict(),
            "operation_name": self.operation_name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "tags": self.tags,
            "logs": self.logs
        }


class Trace:
    """Complete trace containing multiple spans."""
    
    def __init__(self, trace_id: str, root_operation: str):
        self.trace_id = trace_id
        self.root_operation = root_operation
        self.spans: Dict[str, Span] = {}
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
    
    def add_span(self, span: Span):
        """Add a span to the trace."""
        self.spans[span.context.span_id] = span
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        return self.spans.get(span_id)
    
    def get_root_span(self) -> Optional[Span]:
        """Get the root span (span with no parent)."""
        for span in self.spans.values():
            if span.context.parent_span_id is None:
                return span
        return None
    
    def finish(self):
        """Finish the trace."""
        self.end_time = datetime.utcnow()
        
        # Finish any unfinished spans
        for span in self.spans.values():
            if span.end_time is None:
                span.finish()
    
    def get_duration_ms(self) -> Optional[float]:
        """Get total trace duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "root_operation": self.root_operation,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.get_duration_ms(),
            "metadata": self.metadata,
            "spans": [span.to_dict() for span in self.spans.values()]
        }


class TraceStorage:
    """Storage backend for traces."""
    
    def __init__(self, max_traces: int = 1000):
        self.max_traces = max_traces
        self.traces: Dict[str, Trace] = {}
        self.trace_order: List[str] = []
    
    def store_trace(self, trace: Trace):
        """Store a completed trace."""
        self.traces[trace.trace_id] = trace
        self.trace_order.append(trace.trace_id)
        
        # Remove old traces if at capacity
        while len(self.trace_order) > self.max_traces:
            old_trace_id = self.trace_order.pop(0)
            self.traces.pop(old_trace_id, None)
        
        logger.debug("Trace stored", trace_id=trace.trace_id, span_count=len(trace.spans))
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self.traces.get(trace_id)
    
    def get_recent_traces(self, limit: int = 50) -> List[Trace]:
        """Get recent traces."""
        recent_trace_ids = self.trace_order[-limit:]
        return [self.traces[tid] for tid in recent_trace_ids if tid in self.traces]
    
    def get_traces_by_operation(self, operation: str, limit: int = 50) -> List[Trace]:
        """Get traces filtered by root operation."""
        matching_traces = []
        for trace_id in reversed(self.trace_order):
            if len(matching_traces) >= limit:
                break
            trace = self.traces.get(trace_id)
            if trace and trace.root_operation == operation:
                matching_traces.append(trace)
        return matching_traces
    
    def get_error_traces(self, limit: int = 50) -> List[Trace]:
        """Get traces that contain errors."""
        error_traces = []
        for trace_id in reversed(self.trace_order):
            if len(error_traces) >= limit:
                break
            trace = self.traces.get(trace_id)
            if trace:
                has_errors = any(span.status == SpanStatus.ERROR for span in trace.spans.values())
                if has_errors:
                    error_traces.append(trace)
        return error_traces
    
    def get_slow_traces(self, threshold_ms: float = 1000, limit: int = 50) -> List[Trace]:
        """Get traces that took longer than threshold."""
        slow_traces = []
        for trace_id in reversed(self.trace_order):
            if len(slow_traces) >= limit:
                break
            trace = self.traces.get(trace_id)
            if trace:
                duration = trace.get_duration_ms()
                if duration and duration > threshold_ms:
                    slow_traces.append(trace)
        return slow_traces


class Tracer:
    """Main tracer for creating and managing traces."""
    
    def __init__(self, service_name: str, storage: Optional[TraceStorage] = None):
        self.service_name = service_name
        self.storage = storage or TraceStorage()
        self.active_traces: Dict[str, Trace] = {}
        self.context = local()  # Thread-local storage for current span
        
        logger.info("Tracer initialized", service_name=service_name)
    
    def start_trace(self, operation_name: str, trace_id: Optional[str] = None) -> Trace:
        """Start a new trace."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        trace = Trace(trace_id, operation_name)
        trace.metadata["service_name"] = self.service_name
        
        self.active_traces[trace_id] = trace
        
        logger.debug("Trace started", trace_id=trace_id, operation=operation_name)
        return trace
    
    def start_span(
        self,
        operation_name: str,
        parent_context: Optional[SpanContext] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        span_id = str(uuid.uuid4())
        
        # Determine trace and parent
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            # Check for current span context
            current_span = getattr(self.context, 'current_span', None)
            if current_span:
                trace_id = current_span.context.trace_id
                parent_span_id = current_span.context.span_id
            else:
                # Start new trace
                trace_id = str(uuid.uuid4())
                parent_span_id = None
                trace = self.start_trace(operation_name, trace_id)
        
        # Create span context
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        
        # Create span
        span = Span(
            context=span_context,
            operation_name=operation_name,
            kind=kind,
            start_time=datetime.utcnow()
        )
        
        if tags:
            span.set_tags(tags)
        
        # Add to trace
        trace = self.active_traces.get(trace_id)
        if trace:
            trace.add_span(span)
        
        logger.debug(
            "Span started",
            trace_id=trace_id,
            span_id=span_id,
            operation=operation_name,
            parent_span_id=parent_span_id
        )
        
        return span
    
    def finish_span(self, span: Span):
        """Finish a span and update trace."""
        span.finish()
        
        trace = self.active_traces.get(span.context.trace_id)
        if trace:
            # Check if this is the last span in the trace
            active_spans = [s for s in trace.spans.values() if s.end_time is None]
            if not active_spans:
                self.finish_trace(trace.trace_id)
    
    def finish_trace(self, trace_id: str):
        """Finish and store a trace."""
        trace = self.active_traces.get(trace_id)
        if trace:
            trace.finish()
            self.storage.store_trace(trace)
            del self.active_traces[trace_id]
            
            logger.debug(
                "Trace finished",
                trace_id=trace_id,
                duration_ms=trace.get_duration_ms(),
                span_count=len(trace.spans)
            )
    
    @contextmanager
    def span(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Context manager for creating spans."""
        span = self.start_span(operation_name, kind=kind, tags=tags)
        
        # Set as current span
        old_span = getattr(self.context, 'current_span', None)
        self.context.current_span = span
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)
            self.context.current_span = old_span
    
    @asynccontextmanager
    async def async_span(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Async context manager for creating spans."""
        span = self.start_span(operation_name, kind=kind, tags=tags)
        
        # Set as current span
        old_span = getattr(self.context, 'current_span', None)
        self.context.current_span = span
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)
            self.context.current_span = old_span
    
    def get_current_span(self) -> Optional[Span]:
        """Get the currently active span."""
        return getattr(self.context, 'current_span', None)
    
    def inject_context(self, span_context: SpanContext) -> Dict[str, str]:
        """Inject span context into headers for propagation."""
        return {
            "X-Trace-Id": span_context.trace_id,
            "X-Span-Id": span_context.span_id,
            "X-Parent-Span-Id": span_context.parent_span_id or "",
            "X-Baggage": json.dumps(span_context.baggage)
        }
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from headers."""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        
        if not trace_id or not span_id:
            return None
        
        parent_span_id = headers.get("X-Parent-Span-Id")
        if parent_span_id == "":
            parent_span_id = None
        
        baggage = {}
        baggage_header = headers.get("X-Baggage")
        if baggage_header:
            try:
                baggage = json.loads(baggage_header)
            except json.JSONDecodeError:
                pass
        
        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )


class PerformanceProfiler:
    """Performance profiler for detailed operation analysis."""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.profiles: Dict[str, Dict[str, Any]] = {}
    
    @contextmanager
    def profile_operation(self, operation_name: str, tags: Optional[Dict[str, Any]] = None):
        """Profile a specific operation with detailed metrics."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        with self.tracer.span(operation_name, tags=tags) as span:
            span.log("profiling_started", memory_mb=start_memory)
            
            try:
                yield span
                
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                duration_ms = (end_time - start_time) * 1000
                memory_delta = end_memory - start_memory
                
                # Add performance metrics to span
                span.set_tags({
                    "performance.duration_ms": duration_ms,
                    "performance.memory_start_mb": start_memory,
                    "performance.memory_end_mb": end_memory,
                    "performance.memory_delta_mb": memory_delta
                })
                
                span.log(
                    "profiling_completed",
                    duration_ms=duration_ms,
                    memory_delta_mb=memory_delta
                )
                
                # Store profile summary
                self._store_profile(operation_name, {
                    "duration_ms": duration_ms,
                    "memory_delta_mb": memory_delta,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                span.set_tags({
                    "performance.duration_ms": duration_ms,
                    "performance.error": True
                })
                
                span.log("profiling_error", duration_ms=duration_ms, error=str(e))
                raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _store_profile(self, operation: str, profile_data: Dict[str, Any]):
        """Store profile data for analysis."""
        if operation not in self.profiles:
            self.profiles[operation] = {
                "samples": [],
                "total_samples": 0,
                "avg_duration_ms": 0.0,
                "avg_memory_delta_mb": 0.0
            }
        
        self.profiles[operation]["samples"].append(profile_data)
        self.profiles[operation]["total_samples"] += 1
        
        # Keep only last 100 samples
        if len(self.profiles[operation]["samples"]) > 100:
            self.profiles[operation]["samples"].pop(0)
        
        # Update averages
        samples = self.profiles[operation]["samples"]
        self.profiles[operation]["avg_duration_ms"] = sum(s["duration_ms"] for s in samples) / len(samples)
        self.profiles[operation]["avg_memory_delta_mb"] = sum(s.get("memory_delta_mb", 0) for s in samples) / len(samples)
    
    def get_profile_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance profile summary."""
        if operation:
            return self.profiles.get(operation, {})
        return self.profiles


def trace_function(
    operation_name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    tags: Optional[Dict[str, Any]] = None
):
    """
    Decorator to automatically trace function execution.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        kind: Span kind
        tags: Additional tags to add to span
    
    Usage:
        @trace_function("my_operation", tags={"component": "data_retrieval"})
        async def my_async_function(arg1, arg2):
            return "result"
    """
    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__qualname__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_ehs_tracer()
                async with tracer.async_span(operation_name, kind=kind, tags=tags) as span:
                    try:
                        # Add function arguments to span if debug mode
                        span.set_tags({
                            "function.module": func.__module__,
                            "function.name": func.__name__,
                            "function.args_count": len(args),
                            "function.kwargs_count": len(kwargs)
                        })
                        
                        result = await func(*args, **kwargs)
                        span.set_tag("function.success", True)
                        return result
                    except Exception as e:
                        span.set_error(e)
                        span.set_tag("function.success", False)
                        raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = get_ehs_tracer()
                with tracer.span(operation_name, kind=kind, tags=tags) as span:
                    try:
                        # Add function arguments to span if debug mode
                        span.set_tags({
                            "function.module": func.__module__,
                            "function.name": func.__name__,
                            "function.args_count": len(args),
                            "function.kwargs_count": len(kwargs)
                        })
                        
                        result = func(*args, **kwargs)
                        span.set_tag("function.success", True)
                        return result
                    except Exception as e:
                        span.set_error(e)
                        span.set_tag("function.success", False)
                        raise
            return sync_wrapper
    
    return decorator


# Global tracer and profiler instances
_ehs_tracer: Optional[Tracer] = None
_ehs_profiler: Optional[PerformanceProfiler] = None


def get_ehs_tracer(service_name: str = "ehs-analytics") -> Tracer:
    """Get the global EHS tracer instance."""
    global _ehs_tracer
    if _ehs_tracer is None:
        _ehs_tracer = Tracer(service_name)
        logger.info("Global EHS tracer initialized", service_name=service_name)
    return _ehs_tracer


def get_ehs_profiler() -> PerformanceProfiler:
    """Get the global EHS profiler instance."""
    global _ehs_profiler
    if _ehs_profiler is None:
        tracer = get_ehs_tracer()
        _ehs_profiler = PerformanceProfiler(tracer)
        logger.info("Global EHS profiler initialized")
    return _ehs_profiler


def create_trace_context(
    trace_id: Optional[str] = None,
    operation_name: str = "ehs_operation"
) -> SpanContext:
    """
    Create a new trace context for EHS operations.
    
    Args:
        trace_id: Existing trace ID or None for new trace
        operation_name: Name of the root operation
    
    Returns:
        SpanContext for the new trace
    """
    tracer = get_ehs_tracer()
    
    if trace_id:
        # Create child span context
        span_id = str(uuid.uuid4())
        return SpanContext(trace_id=trace_id, span_id=span_id)
    else:
        # Start new trace
        trace = tracer.start_trace(operation_name)
        span_id = str(uuid.uuid4())
        return SpanContext(trace_id=trace.trace_id, span_id=span_id)


def get_trace_analytics() -> Dict[str, Any]:
    """Get analytics data from stored traces."""
    tracer = get_ehs_tracer()
    recent_traces = tracer.storage.get_recent_traces(100)
    
    if not recent_traces:
        return {
            "total_traces": 0,
            "avg_duration_ms": 0,
            "error_rate": 0,
            "operations": {}
        }
    
    # Calculate metrics
    total_traces = len(recent_traces)
    total_duration = sum(t.get_duration_ms() or 0 for t in recent_traces)
    avg_duration = total_duration / total_traces if total_traces > 0 else 0
    
    error_traces = sum(1 for t in recent_traces 
                      if any(s.status == SpanStatus.ERROR for s in t.spans.values()))
    error_rate = error_traces / total_traces if total_traces > 0 else 0
    
    # Operation breakdown
    operations = {}
    for trace in recent_traces:
        op = trace.root_operation
        if op not in operations:
            operations[op] = {
                "count": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0,
                "error_count": 0,
                "error_rate": 0
            }
        
        operations[op]["count"] += 1
        duration = trace.get_duration_ms() or 0
        operations[op]["total_duration_ms"] += duration
        operations[op]["avg_duration_ms"] = operations[op]["total_duration_ms"] / operations[op]["count"]
        
        has_error = any(s.status == SpanStatus.ERROR for s in trace.spans.values())
        if has_error:
            operations[op]["error_count"] += 1
        
        operations[op]["error_rate"] = operations[op]["error_count"] / operations[op]["count"]
    
    return {
        "total_traces": total_traces,
        "avg_duration_ms": avg_duration,
        "error_rate": error_rate,
        "operations": operations,
        "timestamp": datetime.utcnow().isoformat()
    }