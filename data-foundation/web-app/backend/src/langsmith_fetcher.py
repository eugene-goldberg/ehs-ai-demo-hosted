"""
LangSmith API client for fetching traces and run data.

This module provides a client interface to interact with the LangSmith API
for retrieving traces, run details, and project information.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

try:
    from langsmith import Client as LangSmithSDKClient
    from langsmith.schemas import Run
except ImportError:
    raise ImportError(
        "langsmith package is required. Install with: pip install langsmith"
    )


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TraceData:
    """Structured trace data for frontend consumption."""
    run_id: str
    name: str
    run_type: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    start_time: str
    end_time: Optional[str]
    latency_ms: Optional[float]
    tokens_used: Optional[int]
    model_name: Optional[str]
    status: str
    error: Optional[str]
    parent_run_id: Optional[str]
    child_runs: List[str]
    metadata: Dict[str, Any]


class LangSmithFetcherError(Exception):
    """Custom exception for LangSmith fetcher errors."""
    pass


class LangSmithClient:
    """Client for interacting with LangSmith API to fetch traces and runs."""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize LangSmith client.
        
        Args:
            api_key: LangSmith API key. If None, will try to get from LANGCHAIN_API_KEY env var
            api_url: LangSmith API URL. If None, will use default
        """
        self.api_key = api_key or os.getenv("LANGCHAIN_API_KEY")
        if not self.api_key:
            raise LangSmithFetcherError(
                "LangSmith API key not provided. Set LANGCHAIN_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        try:
            self.client = LangSmithSDKClient(
                api_key=self.api_key,
                api_url=api_url
            )
            logger.info("LangSmith client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith client: {e}")
            raise LangSmithFetcherError(f"Failed to initialize LangSmith client: {e}")
    
    def fetch_traces_for_project(
        self, 
        project_name: str, 
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        limit: int = 100
    ) -> List[TraceData]:
        """
        Fetch traces for a specific project.
        
        Args:
            project_name: Name of the LangSmith project
            start_time: Start time for trace filtering (ISO string or datetime)
            end_time: End time for trace filtering (ISO string or datetime)
            limit: Maximum number of traces to fetch
            
        Returns:
            List of TraceData objects containing formatted trace information
            
        Raises:
            LangSmithFetcherError: If API call fails or project not found
        """
        try:
            logger.info(f"Fetching traces for project: {project_name}")
            
            # Convert string timestamps to datetime if needed
            start_dt = self._parse_timestamp(start_time) if start_time else None
            end_dt = self._parse_timestamp(end_time) if end_time else None
            
            # Fetch runs from the project
            runs = list(self.client.list_runs(
                project_name=project_name,
                start_time=start_dt,
                end_time=end_dt,
                limit=limit,
                is_root=True  # Only get root runs initially
            ))
            
            if not runs:
                logger.warning(f"No traces found for project: {project_name}")
                return []
            
            # Convert runs to TraceData format
            traces = []
            for run in runs:
                try:
                    trace_data = self._format_run_as_trace(run)
                    traces.append(trace_data)
                except Exception as e:
                    logger.warning(f"Failed to format run {run.id}: {e}")
                    continue
            
            logger.info(f"Successfully fetched {len(traces)} traces")
            return traces
            
        except Exception as e:
            logger.error(f"Failed to fetch traces for project {project_name}: {e}")
            raise LangSmithFetcherError(f"Failed to fetch traces: {e}")
    
    def fetch_trace_details(self, run_id: str) -> TraceData:
        """
        Fetch detailed information about a specific run.
        
        Args:
            run_id: ID of the run to fetch details for
            
        Returns:
            TraceData object with detailed run information
            
        Raises:
            LangSmithFetcherError: If run not found or API call fails
        """
        try:
            logger.info(f"Fetching details for run: {run_id}")
            
            # Get the specific run
            run = self.client.read_run(run_id)
            if not run:
                raise LangSmithFetcherError(f"Run with ID {run_id} not found")
            
            # Get child runs for this trace
            child_runs = list(self.client.list_runs(
                trace_id=run.trace_id,
                limit=1000  # Get all child runs
            ))
            
            # Format the main run with child information
            trace_data = self._format_run_as_trace(run, child_runs)
            
            logger.info(f"Successfully fetched details for run {run_id}")
            return trace_data
            
        except Exception as e:
            logger.error(f"Failed to fetch details for run {run_id}: {e}")
            raise LangSmithFetcherError(f"Failed to fetch run details: {e}")
    
    def list_projects(self) -> List[Dict[str, str]]:
        """
        List all available projects.
        
        Returns:
            List of dictionaries with project information
        """
        try:
            projects = list(self.client.list_projects())
            return [
                {
                    "id": str(project.id),
                    "name": project.name,
                    "description": project.description or "",
                    "created_at": self._format_timestamp(project.start_time)
                }
                for project in projects
            ]
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            raise LangSmithFetcherError(f"Failed to list projects: {e}")
    
    def _format_run_as_trace(self, run: Run, child_runs: Optional[List[Run]] = None) -> TraceData:
        """
        Format a LangSmith Run object as TraceData.
        
        Args:
            run: The Run object to format
            child_runs: Optional list of child runs
            
        Returns:
            TraceData object
        """
        # Calculate latency
        latency_ms = None
        if run.start_time and run.end_time:
            latency_ms = (run.end_time - run.start_time).total_seconds() * 1000
        
        # Extract token usage from metadata or extra fields
        tokens_used = self._extract_token_usage(run)
        
        # Extract model name
        model_name = self._extract_model_name(run)
        
        # Determine status
        status = "completed"
        error_msg = None
        if run.error:
            status = "error"
            error_msg = str(run.error)
        elif not run.end_time:
            status = "running"
        
        # Get child run IDs
        child_run_ids = []
        if child_runs:
            child_run_ids = [str(child.id) for child in child_runs if child.parent_run_id == run.id]
        
        return TraceData(
            run_id=str(run.id),
            name=run.name or "Unnamed Run",
            run_type=run.run_type or "unknown",
            inputs=self._safe_dict_conversion(run.inputs),
            outputs=self._safe_dict_conversion(run.outputs),
            start_time=self._format_timestamp(run.start_time),
            end_time=self._format_timestamp(run.end_time),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            model_name=model_name,
            status=status,
            error=error_msg,
            parent_run_id=str(run.parent_run_id) if run.parent_run_id else None,
            child_runs=child_run_ids,
            metadata=self._extract_metadata(run)
        )
    
    def _extract_token_usage(self, run: Run) -> Optional[int]:
        """Extract token usage from run data."""
        try:
            # Check various possible locations for token data
            if hasattr(run, "total_tokens") and run.total_tokens:
                return run.total_tokens
            
            if run.extra and isinstance(run.extra, dict):
                # Check for common token fields
                for key in ["total_tokens", "tokens", "usage_metadata"]:
                    if key in run.extra:
                        value = run.extra[key]
                        if isinstance(value, int):
                            return value
                        elif isinstance(value, dict) and "total_tokens" in value:
                            return value["total_tokens"]
            
            return None
        except Exception:
            return None
    
    def _extract_model_name(self, run: Run) -> Optional[str]:
        """Extract model name from run data."""
        try:
            # Check various possible locations for model info
            if run.extra and isinstance(run.extra, dict):
                for key in ["model", "model_name", "llm_model", "invocation_params"]:
                    if key in run.extra:
                        value = run.extra[key]
                        if isinstance(value, str):
                            return value
                        elif isinstance(value, dict) and "model" in value:
                            return value["model"]
            
            # Check serialized data
            if run.serialized and "name" in run.serialized:
                return run.serialized["name"]
                
            return None
        except Exception:
            return None
    
    def _extract_metadata(self, run: Run) -> Dict[str, Any]:
        """Extract relevant metadata from run."""
        metadata = {}
        
        try:
            if run.extra:
                metadata.update(run.extra)
            
            if run.serialized:
                metadata["serialized"] = run.serialized
            
            # Add run-specific metadata
            metadata.update({
                "trace_id": str(run.trace_id),
                "session_id": str(run.session_id) if run.session_id else None,
                "run_type": run.run_type,
                "tags": run.tags or []
            })
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    def _safe_dict_conversion(self, data: Any) -> Dict[str, Any]:
        """Safely convert data to dictionary format."""
        if data is None:
            return {}
        
        if isinstance(data, dict):
            return data
        
        try:
            # Try to convert to string if not serializable
            return {"data": str(data)}
        except Exception:
            return {"data": "<non-serializable>"}
    
    def _parse_timestamp(self, timestamp: Union[str, datetime]) -> datetime:
        """Parse timestamp string or datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        
        try:
            # Try parsing ISO format
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            # Try other common formats
            formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            raise ValueError(f"Unable to parse timestamp: {timestamp}")
    
    def _format_timestamp(self, dt: Optional[datetime]) -> Optional[str]:
        """Format datetime as ISO string."""
        if dt is None:
            return None
        
        try:
            return dt.isoformat()
        except Exception:
            return str(dt)


def group_traces_by_parent(traces: List[TraceData]) -> Dict[str, List[TraceData]]:
    """
    Group traces by their parent run ID.
    
    Args:
        traces: List of TraceData objects
        
    Returns:
        Dictionary mapping parent_run_id to list of child traces
    """
    grouped = {}
    for trace in traces:
        parent_id = trace.parent_run_id or "root"
        if parent_id not in grouped:
            grouped[parent_id] = []
        grouped[parent_id].append(trace)
    
    return grouped


def format_duration(latency_ms: Optional[float]) -> str:
    """
    Format duration in a human-readable format.
    
    Args:
        latency_ms: Duration in milliseconds
        
    Returns:
        Formatted duration string
    """
    if latency_ms is None:
        return "N/A"
    
    if latency_ms < 1000:
        return f"{latency_ms:.0f}ms"
    elif latency_ms < 60000:
        return f"{latency_ms/1000:.1f}s"
    else:
        minutes = latency_ms / 60000
        return f"{minutes:.1f}m"


def extract_relevant_info(trace: TraceData) -> Dict[str, Any]:
    """
    Extract the most relevant information from a trace for display.
    
    Args:
        trace: TraceData object
        
    Returns:
        Dictionary with key information
    """
    return {
        "id": trace.run_id,
        "name": trace.name,
        "type": trace.run_type,
        "status": trace.status,
        "duration": format_duration(trace.latency_ms),
        "tokens": trace.tokens_used or 0,
        "model": trace.model_name,
        "start_time": trace.start_time,
        "has_children": len(trace.child_runs) > 0,
        "error": trace.error
    }


# Example usage and testing functions
if __name__ == "__main__":
    """Example usage of the LangSmith client."""
    
    try:
        # Initialize client
        client = LangSmithClient()
        
        # List available projects
        print("Available projects:")
        projects = client.list_projects()
        for project in projects[:5]:  # Show first 5
            print(f"  - {project['name']} (ID: {project['id']})")
        
        # Example: Fetch traces for a project
        if projects:
            project_name = projects[0]["name"]
            print(f"\nFetching traces for project: {project_name}")
            
            traces = client.fetch_traces_for_project(
                project_name=project_name,
                limit=10
            )
            
            print(f"Found {len(traces)} traces")
            
            # Display summary
            for trace in traces[:3]:  # Show first 3
                info = extract_relevant_info(trace)
                print(f"  - {info['name']} ({info['status']}) - {info['duration']}")
    
    except LangSmithFetcherError as e:
        print(f"LangSmith error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")