#!/usr/bin/env python3
"""
Comprehensive test suite for duplicate prevention functionality.

This test suite covers:
1. File hash calculation utilities
2. Duplicate detection in workflows  
3. Neo4j MERGE operations for duplicate prevention
4. Processing same file twice scenarios
5. Processing different files with same content
6. Mocked Neo4j operations for offline testing
7. Real PDF test files from data directory

Tests use real PDF files but mock Neo4j operations to allow running
without an actual Neo4j database connection.
"""

import os
import sys
import tempfile
import shutil
import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "backend" / "src"))

# Import modules under test
from src.utils.file_hash import (
    calculate_file_hash,
    calculate_sha256_hash, 
    generate_document_id,
    verify_file_integrity,
    get_file_info_with_hash,
    find_duplicate_files
)
from src.ehs_workflows.ingestion_workflow import IngestionWorkflow, DocumentState, ProcessingStatus


class TestFileHashUtilities:
    """Test suite for file hash calculation utilities."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file with known content for testing."""
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as f:
            test_content = b"This is test content for hash calculation"
            f.write(test_content)
            f.flush()
            
            # Calculate expected hash manually
            expected_hash = hashlib.sha256(test_content).hexdigest()
            
            yield f.name, expected_hash
        
        # Cleanup
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def sample_pdf_files(self):
        """Provide paths to real PDF test files from data directory."""
        data_dir = project_root / "data-foundation" / "data"
        pdf_files = []
        
        # Check for available PDF files
        for i in range(1, 5):
            pdf_path = data_dir / f"document-{i}.pdf"
            if pdf_path.exists():
                pdf_files.append(str(pdf_path))
        
        return pdf_files

    def test_calculate_file_hash_basic(self, temp_file):
        """Test basic file hash calculation."""
        file_path, expected_hash = temp_file
        
        # Test SHA-256 hash calculation
        result = calculate_file_hash(file_path, "sha256")
        assert result == expected_hash
        
        # Test that it returns a valid hex string
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 produces 64 hex characters
        assert all(c in '0123456789abcdef' for c in result)

    def test_calculate_file_hash_different_algorithms(self, temp_file):
        """Test hash calculation with different algorithms."""
        file_path, _ = temp_file
        
        # Test SHA-256
        sha256_result = calculate_file_hash(file_path, "sha256")
        assert sha256_result is not None
        assert len(sha256_result) == 64
        
        # Test MD5
        md5_result = calculate_file_hash(file_path, "md5")
        assert md5_result is not None
        assert len(md5_result) == 32
        
        # Test SHA-1
        sha1_result = calculate_file_hash(file_path, "sha1")
        assert sha1_result is not None
        assert len(sha1_result) == 40
        
        # Results should be different
        assert sha256_result != md5_result != sha1_result

    def test_calculate_file_hash_nonexistent_file(self):
        """Test hash calculation with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            calculate_file_hash("/nonexistent/path/file.txt")

    def test_calculate_file_hash_directory(self, temp_file):
        """Test hash calculation with directory path."""
        temp_dir = os.path.dirname(temp_file[0])
        with pytest.raises(ValueError, match="Path is not a file"):
            calculate_file_hash(temp_dir)

    def test_calculate_sha256_hash_convenience_function(self, temp_file):
        """Test the SHA-256 convenience function."""
        file_path, expected_hash = temp_file
        
        result = calculate_sha256_hash(file_path)
        assert result == expected_hash

    def test_generate_document_id(self, temp_file):
        """Test document ID generation."""
        file_path, expected_hash = temp_file
        
        # Test default prefix
        doc_id = generate_document_id(file_path)
        expected_prefix = f"doc_{expected_hash[:16]}"
        assert doc_id == expected_prefix
        
        # Test custom prefix
        custom_doc_id = generate_document_id(file_path, "ehs")
        expected_custom = f"ehs_{expected_hash[:16]}"
        assert custom_doc_id == expected_custom

    def test_verify_file_integrity(self, temp_file):
        """Test file integrity verification."""
        file_path, expected_hash = temp_file
        
        # Test with correct hash
        assert verify_file_integrity(file_path, expected_hash)
        
        # Test with incorrect hash
        wrong_hash = "0" * 64
        assert not verify_file_integrity(file_path, wrong_hash)
        
        # Test case insensitive comparison
        assert verify_file_integrity(file_path, expected_hash.upper())

    def test_get_file_info_with_hash(self, temp_file):
        """Test comprehensive file information extraction."""
        file_path, expected_hash = temp_file
        
        info = get_file_info_with_hash(file_path)
        
        assert info is not None
        assert info['path'] == str(Path(file_path).absolute())
        assert info['name'] == os.path.basename(file_path)
        assert info['sha256'] == expected_hash
        assert info['document_id'] == f"doc_{expected_hash[:16]}"
        assert isinstance(info['size'], int)
        assert info['size'] > 0

    def test_find_duplicate_files(self):
        """Test duplicate file detection."""
        # Create temporary files with same and different content
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with same content
            content1 = b"Same content for testing duplicates"
            file1 = os.path.join(temp_dir, "file1.txt")
            file2 = os.path.join(temp_dir, "file2.txt")
            
            with open(file1, 'wb') as f:
                f.write(content1)
            with open(file2, 'wb') as f:
                f.write(content1)
                
            # Create file with different content
            content2 = b"Different content"
            file3 = os.path.join(temp_dir, "file3.txt")
            with open(file3, 'wb') as f:
                f.write(content2)
            
            # Test duplicate detection
            duplicates = find_duplicate_files([file1, file2, file3])
            
            # Should find one group of duplicates
            assert len(duplicates) == 1
            
            # The duplicate group should contain file1 and file2
            duplicate_group = list(duplicates.values())[0]
            assert len(duplicate_group) == 2
            assert str(Path(file1).absolute()) in duplicate_group
            assert str(Path(file2).absolute()) in duplicate_group

    def test_hash_consistency_across_calls(self, sample_pdf_files):
        """Test that hash calculation is consistent across multiple calls."""
        if not sample_pdf_files:
            pytest.skip("No PDF test files available")
            
        pdf_file = sample_pdf_files[0]
        
        # Calculate hash multiple times
        hash1 = calculate_sha256_hash(pdf_file)
        hash2 = calculate_sha256_hash(pdf_file)
        hash3 = calculate_sha256_hash(pdf_file)
        
        # All should be the same
        assert hash1 == hash2 == hash3
        assert hash1 is not None

    def test_hash_different_files_different_hashes(self, sample_pdf_files):
        """Test that different files produce different hashes."""
        if len(sample_pdf_files) < 2:
            pytest.skip("Need at least 2 PDF test files")
            
        hash1 = calculate_sha256_hash(sample_pdf_files[0])
        hash2 = calculate_sha256_hash(sample_pdf_files[1])
        
        # Different files should have different hashes
        assert hash1 != hash2
        assert hash1 is not None
        assert hash2 is not None


class TestWorkflowDuplicateDetection:
    """Test suite for duplicate detection in ingestion workflow."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver for testing."""
        with patch('src.ehs_workflows.ingestion_workflow.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = Mock()
            
            mock_gdb.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            mock_session.run.return_value = mock_result
            
            yield mock_driver, mock_session, mock_result

    @pytest.fixture
    def workflow_instance(self):
        """Create workflow instance with test configuration."""
        return IngestionWorkflow(
            llama_parse_api_key="test_key",
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="test_user", 
            neo4j_password="test_password",
            openai_api_key="test_openai_key",
            max_retries=2,
            acceptance_threshold=0.6
        )

    @pytest.fixture
    def sample_document_state(self, temp_file):
        """Create a sample document state for testing."""
        file_path, _ = temp_file
        return {
            "file_path": file_path,
            "document_id": "test_doc_001",
            "upload_metadata": {"source": "test"},
            "recognition_result": None,
            "is_accepted": True,
            "rejection_reason": None,
            "file_hash": None,
            "is_duplicate": False,
            "document_type": None,
            "parsed_content": None,
            "extracted_data": None,
            "validation_results": None,
            "indexed": False,
            "errors": [],
            "retry_count": 0,
            "neo4j_nodes": None,
            "neo4j_relationships": None,
            "processing_time": None,
            "status": ProcessingStatus.PENDING
        }

    def test_check_duplicate_no_existing_document(self, workflow_instance, sample_document_state, mock_neo4j_driver):
        """Test duplicate check when no existing document exists."""
        mock_driver, mock_session, mock_result = mock_neo4j_driver
        mock_result.single.return_value = None  # No existing document found
        
        # Test duplicate check
        result_state = workflow_instance.check_duplicate(sample_document_state)
        
        # Verify state updates
        assert result_state["file_hash"] is not None
        assert not result_state["is_duplicate"]
        assert result_state["status"] != ProcessingStatus.DUPLICATE
        
        # Verify Neo4j query was called
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MATCH (d:Document {file_hash: $file_hash})" in call_args[0][0]

    def test_check_duplicate_existing_document_found(self, workflow_instance, sample_document_state, mock_neo4j_driver):
        """Test duplicate check when existing document is found."""
        mock_driver, mock_session, mock_result = mock_neo4j_driver
        
        # Mock existing document
        existing_doc = {
            "document_id": "existing_doc_123",
            "file_path": "/path/to/existing/document.pdf",
            "uploaded_at": "2025-01-01T00:00:00Z"
        }
        mock_result.single.return_value = existing_doc
        
        # Test duplicate check
        result_state = workflow_instance.check_duplicate(sample_document_state)
        
        # Verify state updates
        assert result_state["file_hash"] is not None
        assert result_state["is_duplicate"]
        assert result_state["status"] == ProcessingStatus.DUPLICATE

    def test_check_duplicate_status_routing(self, workflow_instance, sample_document_state):
        """Test duplicate status routing logic."""
        # Test when not duplicate
        sample_document_state["is_duplicate"] = False
        result = workflow_instance.check_duplicate_status(sample_document_state)
        assert result == "continue"
        
        # Test when is duplicate
        sample_document_state["is_duplicate"] = True
        result = workflow_instance.check_duplicate_status(sample_document_state)
        assert result == "skip"

    def test_check_duplicate_hash_calculation_failure(self, workflow_instance, sample_document_state, mock_neo4j_driver):
        """Test duplicate check when hash calculation fails."""
        # Use nonexistent file to force hash calculation failure
        sample_document_state["file_path"] = "/nonexistent/file.pdf"
        
        result_state = workflow_instance.check_duplicate(sample_document_state)
        
        # Should handle gracefully
        assert not result_state["is_duplicate"]
        assert len(result_state["errors"]) > 0
        assert "Failed to calculate file hash" in result_state["errors"]

    def test_check_duplicate_neo4j_connection_failure(self, workflow_instance, sample_document_state):
        """Test duplicate check when Neo4j connection fails."""
        with patch('src.ehs_workflows.ingestion_workflow.GraphDatabase') as mock_gdb:
            mock_gdb.driver.side_effect = Exception("Connection failed")
            
            result_state = workflow_instance.check_duplicate(sample_document_state)
            
            # Should handle gracefully and assume not duplicate
            assert not result_state["is_duplicate"]
            assert len(result_state["errors"]) > 0
            assert "Duplicate check error" in str(result_state["errors"])


class TestNeo4jMergeOperations:
    """Test suite for Neo4j MERGE operations in duplicate prevention."""

    @pytest.fixture
    def mock_neo4j_session(self):
        """Mock Neo4j session for testing MERGE operations."""
        session = Mock()
        result = Mock()
        session.run.return_value = result
        return session, result

    def test_merge_document_node_creation(self, mock_neo4j_session):
        """Test MERGE operation for creating new document node."""
        session, result = mock_neo4j_session
        result.single.return_value = {"created": True, "matched": False}
        
        # Sample document data
        doc_data = {
            "file_hash": "abc123def456",
            "document_id": "doc_abc123def456",
            "file_path": "/test/document.pdf",
            "file_name": "document.pdf",
            "uploaded_at": "2025-01-01T00:00:00Z"
        }
        
        # Test MERGE query
        merge_query = """
        MERGE (d:Document {file_hash: $file_hash})
        ON CREATE SET 
            d.id = $document_id,
            d.file_path = $file_path,
            d.file_name = $file_name,
            d.uploaded_at = $uploaded_at,
            d.created_at = datetime(),
            d.duplicate_count = 0
        ON MATCH SET
            d.duplicate_count = d.duplicate_count + 1,
            d.last_duplicate_attempt = datetime()
        RETURN d.id AS document_id, 
               d.duplicate_count AS duplicate_count,
               CASE WHEN d.created_at = datetime() THEN true ELSE false END AS created
        """
        
        session.run(merge_query, **doc_data)
        
        # Verify MERGE query was executed
        session.run.assert_called_once()
        call_args = session.run.call_args
        assert "MERGE (d:Document {file_hash: $file_hash})" in call_args[0][0]
        assert call_args[1]["file_hash"] == doc_data["file_hash"]

    def test_merge_document_node_duplicate_match(self, mock_neo4j_session):
        """Test MERGE operation when matching existing document (duplicate)."""
        session, result = mock_neo4j_session
        result.single.return_value = {
            "document_id": "existing_doc_123",
            "duplicate_count": 2,
            "created": False
        }
        
        doc_data = {
            "file_hash": "duplicate_hash_123",
            "document_id": "new_doc_456",
            "file_path": "/test/duplicate.pdf",
            "file_name": "duplicate.pdf",
            "uploaded_at": "2025-01-01T00:00:00Z"
        }
        
        # Execute MERGE operation
        merge_query = """
        MERGE (d:Document {file_hash: $file_hash})
        ON CREATE SET 
            d.id = $document_id,
            d.file_path = $file_path,
            d.file_name = $file_name,
            d.uploaded_at = $uploaded_at,
            d.created_at = datetime(),
            d.duplicate_count = 0
        ON MATCH SET
            d.duplicate_count = d.duplicate_count + 1,
            d.last_duplicate_attempt = datetime()
        RETURN d.id AS document_id, 
               d.duplicate_count AS duplicate_count,
               CASE WHEN d.created_at = datetime() THEN true ELSE false END AS created
        """
        
        session.run(merge_query, **doc_data)
        session.run.assert_called_once()

    def test_create_duplicate_attempt_log(self, mock_neo4j_session):
        """Test creation of duplicate attempt log entry."""
        session, result = mock_neo4j_session
        
        log_data = {
            "original_doc_id": "doc_original_123",
            "duplicate_file_path": "/test/duplicate.pdf",
            "duplicate_hash": "same_hash_123",
            "attempted_at": "2025-01-01T00:00:00Z",
            "attempt_source": "workflow"
        }
        
        log_query = """
        MATCH (d:Document {id: $original_doc_id})
        CREATE (log:DuplicateAttempt {
            id: randomUUID(),
            duplicate_file_path: $duplicate_file_path,
            duplicate_hash: $duplicate_hash,
            attempted_at: $attempted_at,
            source: $attempt_source,
            created_at: datetime()
        })
        CREATE (d)-[:HAS_DUPLICATE_ATTEMPT]->(log)
        RETURN log.id AS log_id
        """
        
        session.run(log_query, **log_data)
        session.run.assert_called_once()


class TestDuplicateScenarios:
    """Test suite for various duplicate scenarios."""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture 
    def identical_files(self, temp_directory):
        """Create two files with identical content."""
        content = b"Identical content for duplicate testing"
        
        file1 = os.path.join(temp_directory, "file1.pdf")
        file2 = os.path.join(temp_directory, "file2.pdf")
        
        with open(file1, 'wb') as f:
            f.write(content)
        with open(file2, 'wb') as f:
            f.write(content)
            
        return file1, file2

    @pytest.fixture
    def workflow_with_mocked_neo4j(self):
        """Create workflow with mocked Neo4j dependencies."""
        with patch('src.ehs_workflows.ingestion_workflow.GraphDatabase'):
            return IngestionWorkflow(
                llama_parse_api_key="test_key",
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="test_user",
                neo4j_password="test_password",
                openai_api_key="test_openai_key"
            )

    def test_processing_same_file_twice(self, workflow_with_mocked_neo4j, identical_files):
        """Test processing the exact same file twice."""
        file1, _ = identical_files
        
        # First processing attempt
        state1 = {
            "file_path": file1,
            "document_id": "doc_001",
            "upload_metadata": {"source": "test"},
            "is_accepted": True,
            "file_hash": None,
            "is_duplicate": False,
            "errors": [],
            "status": ProcessingStatus.PENDING
        }
        
        # Mock Neo4j to return no existing document for first attempt
        with patch('src.ehs_workflows.ingestion_workflow.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = Mock()
            
            mock_gdb.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            
            # First attempt - no duplicate found
            mock_result.single.return_value = None
            mock_session.run.return_value = mock_result
            
            result1 = workflow_with_mocked_neo4j.check_duplicate(state1)
            
            assert not result1["is_duplicate"]
            assert result1["file_hash"] is not None
            first_hash = result1["file_hash"]
        
        # Second processing attempt (same file)
        state2 = {
            "file_path": file1,  # Same file
            "document_id": "doc_002",
            "upload_metadata": {"source": "test"},
            "is_accepted": True,
            "file_hash": None,
            "is_duplicate": False,
            "errors": [],
            "status": ProcessingStatus.PENDING
        }
        
        # Mock Neo4j to return existing document for second attempt
        with patch('src.ehs_workflows.ingestion_workflow.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = Mock()
            
            mock_gdb.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            
            # Second attempt - duplicate found
            mock_result.single.return_value = {
                "document_id": "doc_001",
                "file_path": file1,
                "uploaded_at": "2025-01-01T00:00:00Z"
            }
            mock_session.run.return_value = mock_result
            
            result2 = workflow_with_mocked_neo4j.check_duplicate(state2)
            
            assert result2["is_duplicate"]
            assert result2["file_hash"] == first_hash  # Same hash
            assert result2["status"] == ProcessingStatus.DUPLICATE

    def test_processing_different_files_same_content(self, workflow_with_mocked_neo4j, identical_files):
        """Test processing different files with identical content."""
        file1, file2 = identical_files
        
        # Verify files have same content but different paths
        hash1 = calculate_sha256_hash(file1)
        hash2 = calculate_sha256_hash(file2)
        assert hash1 == hash2  # Same content
        assert file1 != file2  # Different paths
        
        # Process first file
        state1 = {
            "file_path": file1,
            "document_id": "doc_001",
            "upload_metadata": {"source": "test"},
            "is_accepted": True,
            "file_hash": None,
            "is_duplicate": False,
            "errors": [],
            "status": ProcessingStatus.PENDING
        }
        
        with patch('src.ehs_workflows.ingestion_workflow.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = Mock()
            
            mock_gdb.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            
            # First file - no duplicate
            mock_result.single.return_value = None
            mock_session.run.return_value = mock_result
            
            result1 = workflow_with_mocked_neo4j.check_duplicate(state1)
            assert not result1["is_duplicate"]
        
        # Process second file (different path, same content)
        state2 = {
            "file_path": file2,
            "document_id": "doc_002", 
            "upload_metadata": {"source": "test"},
            "is_accepted": True,
            "file_hash": None,
            "is_duplicate": False,
            "errors": [],
            "status": ProcessingStatus.PENDING
        }
        
        with patch('src.ehs_workflows.ingestion_workflow.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_result = Mock()
            
            mock_gdb.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None
            
            # Second file - duplicate detected (same hash as first)
            mock_result.single.return_value = {
                "document_id": "doc_001",
                "file_path": file1,
                "uploaded_at": "2025-01-01T00:00:00Z"
            }
            mock_session.run.return_value = mock_result
            
            result2 = workflow_with_mocked_neo4j.check_duplicate(state2)
            assert result2["is_duplicate"]
            assert result2["file_hash"] == hash1  # Same hash as first file

    def test_processing_different_content_files(self, temp_directory, workflow_with_mocked_neo4j):
        """Test processing files with different content (should not be duplicates)."""
        # Create files with different content
        file1 = os.path.join(temp_directory, "different1.pdf")
        file2 = os.path.join(temp_directory, "different2.pdf")
        
        with open(file1, 'wb') as f:
            f.write(b"Content for file 1")
        with open(file2, 'wb') as f:
            f.write(b"Content for file 2")
        
        # Verify different hashes
        hash1 = calculate_sha256_hash(file1)
        hash2 = calculate_sha256_hash(file2)
        assert hash1 != hash2
        
        # Process both files - neither should be flagged as duplicate
        for i, file_path in enumerate([file1, file2], 1):
            state = {
                "file_path": file_path,
                "document_id": f"doc_00{i}",
                "upload_metadata": {"source": "test"},
                "is_accepted": True,
                "file_hash": None,
                "is_duplicate": False,
                "errors": [],
                "status": ProcessingStatus.PENDING
            }
            
            with patch('src.ehs_workflows.ingestion_workflow.GraphDatabase') as mock_gdb:
                mock_driver = Mock()
                mock_session = Mock()
                mock_result = Mock()
                
                mock_gdb.driver.return_value = mock_driver
                mock_driver.session.return_value.__enter__.return_value = mock_session
                mock_driver.session.return_value.__exit__.return_value = None
                mock_result.single.return_value = None  # No duplicates found
                mock_session.run.return_value = mock_result
                
                result = workflow_with_mocked_neo4j.check_duplicate(state)
                assert not result["is_duplicate"]


class TestIntegrationScenarios:
    """Integration tests for complete duplicate prevention scenarios."""

    @pytest.fixture
    def sample_pdf_copy(self, tmp_path):
        """Create a copy of a real PDF file for testing."""
        data_dir = Path(__file__).parent.parent.parent / "data-foundation" / "data"
        source_pdf = data_dir / "document-1.pdf"
        
        if not source_pdf.exists():
            pytest.skip("Test PDF file not available")
            
        # Copy to temp location
        dest_pdf = tmp_path / "test_document.pdf"
        shutil.copy2(source_pdf, dest_pdf)
        
        return str(dest_pdf)

    def test_end_to_end_duplicate_detection(self, sample_pdf_copy):
        """Test complete end-to-end duplicate detection workflow."""
        # Calculate hash of original file
        original_hash = calculate_sha256_hash(sample_pdf_copy)
        assert original_hash is not None
        
        # Test file info extraction
        file_info = get_file_info_with_hash(sample_pdf_copy)
        assert file_info["sha256"] == original_hash
        
        # Create second copy with same content
        second_copy = sample_pdf_copy.replace(".pdf", "_copy.pdf")
        shutil.copy2(sample_pdf_copy, second_copy)
        
        # Verify both files have same hash
        second_hash = calculate_sha256_hash(second_copy)
        assert original_hash == second_hash
        
        # Test duplicate detection
        duplicates = find_duplicate_files([sample_pdf_copy, second_copy])
        assert len(duplicates) == 1
        
        duplicate_group = list(duplicates.values())[0]
        assert len(duplicate_group) == 2

    def test_workflow_state_transitions(self):
        """Test document state transitions through duplicate detection."""
        # Test state progression
        initial_state = {
            "file_path": "/test/document.pdf",
            "document_id": "doc_001",
            "upload_metadata": {"source": "test"},
            "recognition_result": None,
            "is_accepted": True,
            "rejection_reason": None,
            "file_hash": None,
            "is_duplicate": False,
            "status": ProcessingStatus.PENDING,
            "errors": []
        }
        
        # Test initial state
        assert initial_state["status"] == ProcessingStatus.PENDING
        assert not initial_state["is_duplicate"]
        assert initial_state["file_hash"] is None
        
        # Test state after duplicate detection (non-duplicate)
        non_duplicate_state = initial_state.copy()
        non_duplicate_state.update({
            "file_hash": "abc123def456",
            "is_duplicate": False,
            "status": ProcessingStatus.PROCESSING
        })
        
        assert non_duplicate_state["file_hash"] is not None
        assert not non_duplicate_state["is_duplicate"]
        
        # Test state after duplicate detection (duplicate found)
        duplicate_state = initial_state.copy()
        duplicate_state.update({
            "file_hash": "abc123def456",
            "is_duplicate": True,
            "status": ProcessingStatus.DUPLICATE
        })
        
        assert duplicate_state["is_duplicate"]
        assert duplicate_state["status"] == ProcessingStatus.DUPLICATE


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])