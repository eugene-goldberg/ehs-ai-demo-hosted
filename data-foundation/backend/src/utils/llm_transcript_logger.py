"""
LLM Transcript Logger

A thread-safe logger for capturing LLM interactions during the ingestion workflow.
Stores messages with role, content, timestamp, and context information.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class TranscriptLogger:
    """
    Thread-safe logger for capturing LLM interactions.
    
    Stores messages with role (system/user/assistant), content, timestamp, and context.
    Provides methods for logging, retrieval, persistence, and management of transcript data.
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize the TranscriptLogger.
        
        Args:
            max_entries: Maximum number of entries to store in memory (default: 10000)
        """
        self._transcript: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._logger = logging.getLogger(__name__)
        
        # Configure logging if not already configured
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
    
    def log_interaction(
        self, 
        role: str, 
        content: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an LLM interaction.
        
        Args:
            role: The role of the message sender (system/user/assistant)
            content: The content of the message
            context: Optional context information (e.g., model_name, tokens, etc.)
        
        Raises:
            ValueError: If role or content is empty
        """
        if not role or not role.strip():
            raise ValueError("Role cannot be empty")
        
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        
        # Validate role
        valid_roles = {'system', 'user', 'assistant'}
        if role.lower() not in valid_roles:
            self._logger.warning(f"Unusual role '{role}' - expected one of {valid_roles}")
        
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'unix_timestamp': time.time(),
            'role': role.lower(),
            'content': content.strip(),
            'context': context or {},
            'entry_id': len(self._transcript) + 1
        }
        
        with self._lock:
            try:
                self._transcript.append(entry)
                
                # Maintain maximum entries limit
                if len(self._transcript) > self._max_entries:
                    removed_entries = len(self._transcript) - self._max_entries
                    self._transcript = self._transcript[-self._max_entries:]
                    self._logger.warning(
                        f"Transcript exceeded max entries. Removed {removed_entries} oldest entries."
                    )
                
                self._logger.debug(f"Logged {role} interaction with {len(content)} characters")
                
            except Exception as e:
                self._logger.error(f"Failed to log interaction: {e}")
                raise
    
    def get_transcript(
        self, 
        start_index: Optional[int] = None, 
        end_index: Optional[int] = None,
        role_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the transcript entries.
        
        Args:
            start_index: Starting index for slicing (inclusive)
            end_index: Ending index for slicing (exclusive)
            role_filter: Optional role filter (system/user/assistant)
        
        Returns:
            List of transcript entries
        """
        with self._lock:
            try:
                transcript_copy = self._transcript.copy()
                
                # Apply role filter if specified
                if role_filter:
                    transcript_copy = [
                        entry for entry in transcript_copy 
                        if entry['role'] == role_filter.lower()
                    ]
                
                # Apply index slicing if specified
                if start_index is not None or end_index is not None:
                    transcript_copy = transcript_copy[start_index:end_index]
                
                return transcript_copy
                
            except Exception as e:
                self._logger.error(f"Failed to get transcript: {e}")
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the transcript.
        
        Returns:
            Dictionary containing transcript statistics
        """
        with self._lock:
            try:
                if not self._transcript:
                    return {
                        'total_entries': 0,
                        'role_counts': {},
                        'first_entry_time': None,
                        'last_entry_time': None,
                        'total_content_length': 0
                    }
                
                role_counts = {}
                total_content_length = 0
                
                for entry in self._transcript:
                    role = entry['role']
                    role_counts[role] = role_counts.get(role, 0) + 1
                    total_content_length += len(entry['content'])
                
                return {
                    'total_entries': len(self._transcript),
                    'role_counts': role_counts,
                    'first_entry_time': self._transcript[0]['timestamp'],
                    'last_entry_time': self._transcript[-1]['timestamp'],
                    'total_content_length': total_content_length
                }
                
            except Exception as e:
                self._logger.error(f"Failed to get stats: {e}")
                raise
    
    def clear_transcript(self) -> int:
        """
        Clear all transcript entries.
        
        Returns:
            Number of entries that were cleared
        """
        with self._lock:
            try:
                cleared_count = len(self._transcript)
                self._transcript.clear()
                self._logger.info(f"Cleared {cleared_count} transcript entries")
                return cleared_count
                
            except Exception as e:
                self._logger.error(f"Failed to clear transcript: {e}")
                raise
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """
        Save the transcript to a JSON file.
        
        Args:
            filepath: Path to the output file
        
        Raises:
            IOError: If file cannot be written
            ValueError: If filepath is invalid
        """
        if not filepath:
            raise ValueError("Filepath cannot be empty")
        
        filepath = Path(filepath)
        
        with self._lock:
            try:
                # Create parent directories if they don't exist
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                transcript_data = {
                    'metadata': {
                        'saved_at': datetime.utcnow().isoformat(),
                        'total_entries': len(self._transcript),
                        'max_entries': self._max_entries
                    },
                    'transcript': self._transcript.copy()
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                
                self._logger.info(f"Saved {len(self._transcript)} entries to {filepath}")
                
            except Exception as e:
                self._logger.error(f"Failed to save transcript to {filepath}: {e}")
                raise IOError(f"Failed to save transcript: {e}")
    
    def load_from_file(self, filepath: Union[str, Path], replace: bool = True) -> int:
        """
        Load transcript from a JSON file.
        
        Args:
            filepath: Path to the input file
            replace: If True, replace current transcript; if False, append to current
        
        Returns:
            Number of entries loaded
        
        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid
        """
        if not filepath:
            raise ValueError("Filepath cannot be empty")
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise IOError(f"File does not exist: {filepath}")
        
        with self._lock:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate file format
                if not isinstance(data, dict) or 'transcript' not in data:
                    raise ValueError("Invalid transcript file format")
                
                loaded_entries = data['transcript']
                
                # Validate entries format
                for i, entry in enumerate(loaded_entries):
                    required_fields = ['timestamp', 'role', 'content']
                    for field in required_fields:
                        if field not in entry:
                            raise ValueError(f"Entry {i} missing required field: {field}")
                
                if replace:
                    self._transcript = loaded_entries.copy()
                else:
                    self._transcript.extend(loaded_entries)
                
                # Maintain max entries limit
                if len(self._transcript) > self._max_entries:
                    excess = len(self._transcript) - self._max_entries
                    self._transcript = self._transcript[-self._max_entries:]
                    self._logger.warning(f"Truncated {excess} entries to maintain max limit")
                
                loaded_count = len(loaded_entries)
                self._logger.info(f"Loaded {loaded_count} entries from {filepath}")
                return loaded_count
                
            except json.JSONDecodeError as e:
                self._logger.error(f"Invalid JSON in {filepath}: {e}")
                raise ValueError(f"Invalid JSON format: {e}")
            except Exception as e:
                self._logger.error(f"Failed to load transcript from {filepath}: {e}")
                raise IOError(f"Failed to load transcript: {e}")
    
    def export_to_text(self, filepath: Union[str, Path], include_context: bool = False) -> None:
        """
        Export transcript to a human-readable text file.
        
        Args:
            filepath: Path to the output text file
            include_context: Whether to include context information in the export
        
        Raises:
            IOError: If file cannot be written
        """
        if not filepath:
            raise ValueError("Filepath cannot be empty")
        
        filepath = Path(filepath)
        
        with self._lock:
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("LLM Interaction Transcript\n")
                    f.write("=" * 50 + "\n\n")
                    
                    if not self._transcript:
                        f.write("No interactions recorded.\n")
                        return
                    
                    for i, entry in enumerate(self._transcript, 1):
                        f.write(f"Entry {i}: {entry['role'].upper()}\n")
                        f.write(f"Timestamp: {entry['timestamp']}\n")
                        
                        if include_context and entry.get('context'):
                            f.write(f"Context: {json.dumps(entry['context'], indent=2)}\n")
                        
                        f.write("Content:\n")
                        f.write("-" * 20 + "\n")
                        f.write(entry['content'])
                        f.write("\n" + "=" * 50 + "\n\n")
                
                self._logger.info(f"Exported {len(self._transcript)} entries to {filepath}")
                
            except Exception as e:
                self._logger.error(f"Failed to export transcript to {filepath}: {e}")
                raise IOError(f"Failed to export transcript: {e}")
    
    def __len__(self) -> int:
        """Return the number of entries in the transcript."""
        with self._lock:
            return len(self._transcript)
    
    def __str__(self) -> str:
        """Return a string representation of the logger."""
        with self._lock:
            stats = self.get_stats()
            return (f"TranscriptLogger(entries={stats['total_entries']}, "
                   f"roles={list(stats['role_counts'].keys())})")
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the logger."""
        return (f"TranscriptLogger(max_entries={self._max_entries}, "
               f"current_entries={len(self._transcript)})")


# Singleton instance for global use
_global_logger: Optional[TranscriptLogger] = None
_global_lock = threading.Lock()


def get_global_logger() -> TranscriptLogger:
    """
    Get the global TranscriptLogger instance.
    
    Returns:
        Global TranscriptLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        with _global_lock:
            if _global_logger is None:
                _global_logger = TranscriptLogger()
    
    return _global_logger


def log_llm_interaction(
    role: str, 
    content: str, 
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to log an interaction using the global logger.
    
    Args:
        role: The role of the message sender (system/user/assistant)
        content: The content of the message
        context: Optional context information
    """
    logger = get_global_logger()
    logger.log_interaction(role, content, context)


if __name__ == "__main__":
    # Example usage
    logger = TranscriptLogger()
    
    # Log some sample interactions
    logger.log_interaction("system", "You are a helpful assistant.", {"model": "gpt-4"})
    logger.log_interaction("user", "What is the capital of France?")
    logger.log_interaction("assistant", "The capital of France is Paris.", {"tokens": 150})
    
    # Print stats
    print("Transcript Stats:", logger.get_stats())
    
    # Save to file
    logger.save_to_file("/tmp/sample_transcript.json")
    
    # Export to text
    logger.export_to_text("/tmp/sample_transcript.txt", include_context=True)
    
    print(f"Logger: {logger}")