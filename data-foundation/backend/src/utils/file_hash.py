"""
File hash utilities for duplicate detection and document ID generation.

This module provides functions to calculate file hashes using SHA-256 with
efficient streaming for large files and consistent document ID generation.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Buffer size for streaming hash calculation (64KB)
BUFFER_SIZE = 65536


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> Optional[str]:
    """
    Calculate the hash of a file using the specified algorithm.
    
    Uses streaming to handle large files efficiently without loading
    the entire file into memory.
    
    Args:
        file_path: Path to the file to hash
        algorithm: Hash algorithm to use (default: "sha256")
        
    Returns:
        Hexadecimal hash string, or None if error occurs
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file cannot be read
        OSError: If there's an I/O error reading the file
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
            
        # Create hash object
        hash_obj = hashlib.new(algorithm)
        
        # Read file in chunks for memory efficiency
        with open(file_path, 'rb') as file:
            while chunk := file.read(BUFFER_SIZE):
                hash_obj.update(chunk)
                
        return hash_obj.hexdigest()
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied reading file: {file_path}")
        raise
    except OSError as e:
        logger.error(f"I/O error reading file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calculating hash for {file_path}: {e}")
        return None


def calculate_sha256_hash(file_path: Union[str, Path]) -> Optional[str]:
    """
    Calculate SHA-256 hash of a file using efficient streaming.
    
    This is a convenience wrapper around calculate_file_hash() specifically
    for SHA-256 hashing, which is the most common use case for duplicate detection.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        SHA-256 hash as hexadecimal string, or None if error occurs
        
    Example:
        >>> hash_value = calculate_sha256_hash("/path/to/document.pdf")
        >>> print(hash_value)  # e.g., "a1b2c3d4e5f6..."
    """
    return calculate_file_hash(file_path, "sha256")


def generate_document_id(file_path: Union[str, Path], prefix: str = "doc") -> Optional[str]:
    """
    Generate a consistent document ID from file hash.
    
    Creates a document ID by combining a prefix with the first 16 characters
    of the file's SHA-256 hash. This provides a good balance between uniqueness
    and readability while maintaining consistency across runs.
    
    Args:
        file_path: Path to the file
        prefix: Prefix for the document ID (default: "doc")
        
    Returns:
        Document ID string in format "{prefix}_{hash_prefix}", or None if error
        
    Example:
        >>> doc_id = generate_document_id("/path/to/document.pdf", "ehs")
        >>> print(doc_id)  # e.g., "ehs_a1b2c3d4e5f6789a"
    """
    try:
        file_hash = calculate_sha256_hash(file_path)
        if file_hash is None:
            return None
            
        # Use first 16 characters of hash for document ID
        hash_prefix = file_hash[:16]
        return f"{prefix}_{hash_prefix}"
        
    except Exception as e:
        logger.error(f"Error generating document ID for {file_path}: {e}")
        return None


def verify_file_integrity(file_path: Union[str, Path], expected_hash: str) -> bool:
    """
    Verify file integrity by comparing with expected hash.
    
    Args:
        file_path: Path to the file to verify
        expected_hash: Expected SHA-256 hash value
        
    Returns:
        True if file hash matches expected hash, False otherwise
        
    Example:
        >>> is_valid = verify_file_integrity("/path/to/file.pdf", "a1b2c3d4...")
        >>> if not is_valid:
        ...     print("File may be corrupted!")
    """
    try:
        actual_hash = calculate_sha256_hash(file_path)
        if actual_hash is None:
            return False
            
        return actual_hash.lower() == expected_hash.lower()
        
    except Exception as e:
        logger.error(f"Error verifying file integrity for {file_path}: {e}")
        return False


def get_file_info_with_hash(file_path: Union[str, Path]) -> Optional[dict]:
    """
    Get comprehensive file information including hash.
    
    Returns file metadata along with hash for duplicate detection and
    file tracking purposes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information and hash, or None if error
        
    Example:
        >>> info = get_file_info_with_hash("/path/to/document.pdf")
        >>> print(info)
        {
            'path': '/path/to/document.pdf',
            'name': 'document.pdf',
            'size': 1024576,
            'sha256': 'a1b2c3d4e5f6...',
            'document_id': 'doc_a1b2c3d4e5f6789a'
        }
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists() or not file_path.is_file():
            return None
            
        # Calculate hash
        file_hash = calculate_sha256_hash(file_path)
        if file_hash is None:
            return None
            
        # Get file stats
        stat_info = file_path.stat()
        
        return {
            'path': str(file_path.absolute()),
            'name': file_path.name,
            'size': stat_info.st_size,
            'sha256': file_hash,
            'document_id': generate_document_id(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return None


def find_duplicate_files(file_paths: list[Union[str, Path]]) -> dict[str, list[str]]:
    """
    Find duplicate files by comparing their SHA-256 hashes.
    
    Groups files by their hash values to identify duplicates.
    Only returns groups with more than one file (actual duplicates).
    
    Args:
        file_paths: List of file paths to check for duplicates
        
    Returns:
        Dictionary mapping hash values to lists of duplicate file paths
        
    Example:
        >>> duplicates = find_duplicate_files(["/path/a.pdf", "/path/b.pdf", "/path/c.pdf"])
        >>> for hash_val, files in duplicates.items():
        ...     print(f"Duplicate files (hash: {hash_val[:8]}...):")
        ...     for file in files:
        ...         print(f"  - {file}")
    """
    hash_to_files = {}
    
    for file_path in file_paths:
        try:
            file_hash = calculate_sha256_hash(file_path)
            if file_hash is not None:
                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []
                hash_to_files[file_hash].append(str(Path(file_path).absolute()))
                
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            continue
    
    # Return only groups with duplicates (more than 1 file)
    return {hash_val: files for hash_val, files in hash_to_files.items() if len(files) > 1}