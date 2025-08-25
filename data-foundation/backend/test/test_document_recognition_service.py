"""
Comprehensive test suite for Document Recognition Service
Tests use real PDF files from test_documents directory (no mocks)
Follows pytest framework with 100% coverage goal
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from document_recognition_service import DocumentRecognitionService
except ImportError:
    # Service doesn't exist yet - will be created next
    pass

# Configure logging for test runs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/document_recognition_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestDocumentRecognitionService:
    """Test suite for DocumentRecognitionService with 100% coverage"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment before each test"""
        self.test_docs_path = Path(__file__).parent / "test_documents"
        self.temp_dir = tempfile.mkdtemp()
        
        # Environment variables from .env
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llama_parse_key = os.getenv("LLAMA_PARSE_API_KEY")
        
        if not self.api_key or not self.llama_parse_key:
            pytest.skip("API keys not available in environment")
            
        # Initialize service (when it exists)
        try:
            self.service = DocumentRecognitionService(
                openai_api_key=self.api_key,
                llama_parse_key=self.llama_parse_key
            )
        except NameError:
            # Service class doesn't exist yet
            self.service = None
            
        logger.info(f"Setup complete for test in {self.temp_dir}")
        
    @pytest.fixture(autouse=True) 
    def teardown_method(self):
        """Cleanup after each test"""
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        logger.info("Test cleanup complete")

    def test_analyze_document_type_electricity_bill(self):
        """Test recognition of electricity bill documents"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        electricity_bill_path = self.test_docs_path / "electricity_bills" / "electric_bill.pdf"
        
        if not electricity_bill_path.exists():
            pytest.skip(f"Test document not found: {electricity_bill_path}")
            
        logger.info(f"Testing electricity bill recognition: {electricity_bill_path}")
        
        result = self.service.analyze_document_type(str(electricity_bill_path))
        
        assert result is not None
        assert result["document_type"] == "electricity_bill"
        assert result["confidence"] > 0.8
        assert "features" in result
        
        logger.info(f"Electricity bill recognized with confidence: {result['confidence']}")

    def test_analyze_document_type_water_bill(self):
        """Test recognition of water bill documents"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        water_bill_path = self.test_docs_path / "water_bills" / "water_bill.pdf"
        
        if not water_bill_path.exists():
            pytest.skip(f"Test document not found: {water_bill_path}")
            
        logger.info(f"Testing water bill recognition: {water_bill_path}")
        
        result = self.service.analyze_document_type(str(water_bill_path))
        
        assert result is not None
        assert result["document_type"] == "water_bill"
        assert result["confidence"] > 0.8
        assert "features" in result
        
        logger.info(f"Water bill recognized with confidence: {result['confidence']}")

    def test_analyze_document_type_waste_manifest(self):
        """Test recognition of waste manifest documents"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        waste_manifest_path = self.test_docs_path / "waste_manifests" / "waste_manifest.pdf"
        
        if not waste_manifest_path.exists():
            pytest.skip(f"Test document not found: {waste_manifest_path}")
            
        logger.info(f"Testing waste manifest recognition: {waste_manifest_path}")
        
        result = self.service.analyze_document_type(str(waste_manifest_path))
        
        assert result is not None
        assert result["document_type"] == "waste_manifest"
        assert result["confidence"] > 0.8
        assert "features" in result
        
        logger.info(f"Waste manifest recognized with confidence: {result['confidence']}")

    def test_analyze_document_type_unrecognized(self):
        """Test rejection of unrecognized document types"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        unrecognized_docs = list((self.test_docs_path / "unrecognized").glob("*.pdf"))
        
        if not unrecognized_docs:
            pytest.skip("No unrecognized test documents found")
            
        for doc_path in unrecognized_docs:
            logger.info(f"Testing unrecognized document: {doc_path}")
            
            result = self.service.analyze_document_type(str(doc_path))
            
            assert result is not None
            assert result["document_type"] == "unknown" or result["confidence"] < 0.6
            
            logger.info(f"Document properly rejected with confidence: {result.get('confidence', 0)}")

    def test_extract_document_features_comprehensive(self):
        """Test comprehensive feature extraction from various document types"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        test_files = [
            self.test_docs_path / "electricity_bills" / "electric_bill.pdf",
            self.test_docs_path / "water_bills" / "water_bill.pdf",
            self.test_docs_path / "waste_manifests" / "waste_manifest.pdf"
        ]
        
        for doc_path in test_files:
            if not doc_path.exists():
                continue
                
            logger.info(f"Testing feature extraction: {doc_path}")
            
            features = self.service.extract_document_features(str(doc_path))
            
            assert features is not None
            assert isinstance(features, dict)
            assert "text_content" in features
            assert "structure_indicators" in features
            assert "metadata" in features
            
            # Verify non-empty content
            assert len(features["text_content"]) > 0
            assert len(features["structure_indicators"]) > 0
            
            logger.info(f"Features extracted successfully for {doc_path.name}")

    def test_classify_with_confidence_scoring(self):
        """Test confidence scoring accuracy across document types"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        # Test with known good documents
        known_good_files = [
            (self.test_docs_path / "electricity_bills" / "electric_bill.pdf", "electricity_bill"),
            (self.test_docs_path / "water_bills" / "water_bill.pdf", "water_bill"),
            (self.test_docs_path / "waste_manifests" / "waste_manifest.pdf", "waste_manifest")
        ]
        
        for doc_path, expected_type in known_good_files:
            if not doc_path.exists():
                continue
                
            logger.info(f"Testing confidence scoring: {doc_path}")
            
            features = self.service.extract_document_features(str(doc_path))
            result = self.service.classify_with_confidence(features)
            
            assert result is not None
            assert result["predicted_type"] == expected_type
            assert 0.0 <= result["confidence"] <= 1.0
            assert result["confidence"] > 0.8  # High confidence for known types
            
            logger.info(f"Confidence scoring passed: {result['confidence']:.3f}")

    def test_validate_document_structure_valid(self):
        """Test document structure validation for valid documents"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        valid_docs = [
            self.test_docs_path / "electricity_bills" / "electric_bill.pdf",
            self.test_docs_path / "water_bills" / "water_bill.pdf",
            self.test_docs_path / "waste_manifests" / "waste_manifest.pdf"
        ]
        
        for doc_path in valid_docs:
            if not doc_path.exists():
                continue
                
            logger.info(f"Testing document structure validation: {doc_path}")
            
            is_valid = self.service.validate_document_structure(str(doc_path))
            
            assert is_valid is True
            logger.info(f"Document structure validation passed for {doc_path.name}")

    def test_edge_case_empty_file(self):
        """Test handling of empty files"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        # Create empty file
        empty_file_path = os.path.join(self.temp_dir, "empty.pdf")
        Path(empty_file_path).touch()
        
        logger.info("Testing empty file handling")
        
        with pytest.raises(Exception) as exc_info:
            self.service.analyze_document_type(empty_file_path)
            
        assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        logger.info("Empty file properly rejected")

    def test_edge_case_corrupted_file(self):
        """Test handling of corrupted PDF files"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        # Create corrupted file
        corrupted_file_path = os.path.join(self.temp_dir, "corrupted.pdf")
        with open(corrupted_file_path, "w") as f:
            f.write("This is not a valid PDF file content")
        
        logger.info("Testing corrupted file handling")
        
        with pytest.raises(Exception) as exc_info:
            self.service.analyze_document_type(corrupted_file_path)
            
        assert "corrupt" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        logger.info("Corrupted file properly rejected")

    def test_edge_case_nonexistent_file(self):
        """Test handling of nonexistent files"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pdf")
        
        logger.info("Testing nonexistent file handling")
        
        with pytest.raises(FileNotFoundError):
            self.service.analyze_document_type(nonexistent_path)
            
        logger.info("Nonexistent file properly rejected")

    def test_edge_case_invalid_file_extension(self):
        """Test handling of files with invalid extensions"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        # Create text file with wrong extension
        invalid_file_path = os.path.join(self.temp_dir, "document.pdf")
        with open(invalid_file_path, "w") as f:
            f.write("This is a text file, not a PDF")
        
        logger.info("Testing invalid file extension handling")
        
        with pytest.raises(Exception) as exc_info:
            self.service.analyze_document_type(invalid_file_path)
            
        logger.info("Invalid file extension properly handled")

    def test_confidence_threshold_boundaries(self):
        """Test confidence scoring at various threshold boundaries"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        # Test with borderline documents (if available)
        edge_case_docs = list((self.test_docs_path / "edge_cases").glob("*.pdf"))
        
        for doc_path in edge_case_docs:
            logger.info(f"Testing confidence boundaries: {doc_path}")
            
            result = self.service.analyze_document_type(str(doc_path))
            
            assert result is not None
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0
            
            # Test various confidence thresholds
            if result["confidence"] >= 0.9:
                assert result["document_type"] in ["electricity_bill", "water_bill", "waste_manifest"]
            elif result["confidence"] < 0.6:
                assert result["document_type"] == "unknown"
                
            logger.info(f"Confidence boundary test passed: {result['confidence']:.3f}")

    def test_batch_processing_performance(self):
        """Test batch processing of multiple documents"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        # Collect all available test documents
        all_docs = []
        for doc_type in ["electricity_bills", "water_bills", "waste_manifests"]:
            doc_path = self.test_docs_path / doc_type
            if doc_path.exists():
                all_docs.extend(list(doc_path.glob("*.pdf")))
        
        if len(all_docs) < 2:
            pytest.skip("Not enough test documents for batch processing test")
            
        logger.info(f"Testing batch processing with {len(all_docs)} documents")
        
        results = []
        for doc_path in all_docs:
            result = self.service.analyze_document_type(str(doc_path))
            results.append(result)
        
        assert len(results) == len(all_docs)
        
        # Verify all results are valid
        for result in results:
            assert result is not None
            assert "document_type" in result
            assert "confidence" in result
            
        logger.info(f"Batch processing completed successfully for {len(results)} documents")

    def test_memory_usage_large_documents(self):
        """Test memory efficiency with larger documents"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Find the largest available test document
        all_docs = []
        for doc_type in ["electricity_bills", "water_bills", "waste_manifests"]:
            doc_path = self.test_docs_path / doc_type
            if doc_path.exists():
                all_docs.extend(list(doc_path.glob("*.pdf")))
        
        if not all_docs:
            pytest.skip("No test documents available for memory test")
            
        largest_doc = max(all_docs, key=lambda x: x.stat().st_size)
        
        logger.info(f"Testing memory usage with largest document: {largest_doc.name} ({largest_doc.stat().st_size / 1024:.1f} KB)")
        
        result = self.service.analyze_document_type(str(largest_doc))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory usage: Initial={initial_memory:.1f}MB, Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB")
        
        # Memory increase should be reasonable (less than 100MB for document processing)
        assert memory_increase < 100
        assert result is not None

    def test_concurrent_processing(self):
        """Test concurrent document processing"""
        if self.service is None:
            pytest.skip("DocumentRecognitionService not available")
            
        import threading
        import time
        
        # Collect test documents
        test_docs = []
        for doc_type in ["electricity_bills", "water_bills", "waste_manifests"]:
            doc_path = self.test_docs_path / doc_type
            if doc_path.exists():
                test_docs.extend(list(doc_path.glob("*.pdf"))[:1])  # One from each type
        
        if len(test_docs) < 2:
            pytest.skip("Not enough test documents for concurrent processing test")
            
        results = {}
        errors = {}
        
        def process_document(doc_path, doc_id):
            try:
                result = self.service.analyze_document_type(str(doc_path))
                results[doc_id] = result
                logger.info(f"Thread {doc_id} completed processing {doc_path.name}")
            except Exception as e:
                errors[doc_id] = str(e)
                logger.error(f"Thread {doc_id} failed: {e}")
        
        logger.info(f"Testing concurrent processing with {len(test_docs)} documents")
        
        threads = []
        for i, doc_path in enumerate(test_docs):
            thread = threading.Thread(target=process_document, args=(doc_path, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout per thread
        
        # Verify results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == len(test_docs)
        
        for doc_id, result in results.items():
            assert result is not None
            assert "document_type" in result
            assert "confidence" in result
            
        logger.info("Concurrent processing test completed successfully")


if __name__ == "__main__":
    # Run tests with detailed logging
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--log-cli-level=INFO"
    ])
