"""
Rejection Workflow Service for EHS AI Demo
=========================================

This service manages document rejection workflows, including automated detection,
manual rejection processes, quality validation, duplicate detection, and notification
systems. It integrates with the document processing pipeline to ensure only valid,
relevant, and high-quality documents are processed and stored.

Features:
- Automated rejection detection based on configurable rules
- Manual document rejection workflow with approval chains
- Document quality validation (scan quality, readability)
- Duplicate document detection and handling
- Bulk rejection processing for batch operations
- Appeal process for rejected documents
- Comprehensive notification system
- Integration with document processing pipeline

Author: EHS AI Demo Team
Created: 2025-08-23
"""

import logging
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

from langchain_neo4j import Neo4jGraph
from neo4j.exceptions import TransientError

# Import schema and related modules (to be created)
# from .rejection_schema import RejectionSchema
# from .notification_service import NotificationService


class RejectionReason(Enum):
    """Enumeration of possible rejection reasons."""
    POOR_QUALITY = "poor_quality"
    DUPLICATE = "duplicate"
    IRRELEVANT = "irrelevant"
    CORRUPTED = "corrupted"
    INCOMPLETE = "incomplete"
    WRONG_FORMAT = "wrong_format"
    INSUFFICIENT_DATA = "insufficient_data"
    MANUAL_REVIEW = "manual_review"
    POLICY_VIOLATION = "policy_violation"


class RejectionStatus(Enum):
    """Status of rejection workflow."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPEALED = "appealed"
    RESOLVED = "resolved"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    STRICT = "strict"


@dataclass
class RejectionRule:
    """Configuration for rejection validation rules."""
    rule_id: str
    rule_name: str
    rule_type: str
    parameters: Dict[str, Any]
    enabled: bool
    priority: int
    threshold: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of document validation."""
    document_id: str
    is_valid: bool
    quality_score: float
    rejection_reasons: List[RejectionReason]
    validation_details: Dict[str, Any]
    validation_time: datetime
    rule_violations: List[str]


@dataclass
class RejectionRecord:
    """Record of a document rejection."""
    rejection_id: str
    document_id: str
    rejection_reason: RejectionReason
    rejection_status: RejectionStatus
    initiated_by: str
    created_at: datetime
    updated_at: datetime
    notes: Optional[str] = None
    appeal_deadline: Optional[datetime] = None
    reviewer_id: Optional[str] = None
    appeal_notes: Optional[str] = None


@dataclass
class BulkRejectionResult:
    """Result of bulk rejection processing."""
    total_documents: int
    successful_rejections: int
    failed_rejections: int
    processing_time: float
    rejection_ids: List[str]
    errors: List[Dict[str, Any]]


class RejectionWorkflowService:
    """
    Service for managing document rejection workflows with Neo4j database integration.
    Handles automated detection, manual processes, validation, and notifications.
    """

    def __init__(self, graph: Neo4jGraph, notification_service=None):
        """
        Initialize the rejection workflow service.
        
        Args:
            graph: Neo4j graph database connection
            notification_service: Optional notification service for alerts
        """
        self.graph = graph
        self.notification_service = notification_service
        self.logger = logging.getLogger(__name__)
        
        # Default rejection rules
        self.default_rules = self._initialize_default_rules()
        
        # Ensure schema is initialized
        try:
            self._create_schema_constraints()
            self.logger.info("Rejection workflow service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize rejection workflow service: {str(e)}")
            raise

    async def initialize_schema(self):
        """Initialize rejection workflow database schema."""
        try:
            self._create_schema_constraints()
            self.logger.info("Rejection workflow schema initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize rejection workflow schema: {str(e)}")
            raise

    async def test_connection(self) -> bool:
        """Test service connectivity and basic functionality."""
        try:
            # Test database connectivity
            test_query = "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            result = self.graph.query(test_query)
            
            # Verify we can execute queries and get results
            if result is not None:
                self.logger.debug("Rejection workflow service database connection test successful")
                return True
            else:
                self.logger.error("Rejection workflow service database connection test failed: No result returned")
                return False
                
        except Exception as e:
            self.logger.error(f"Rejection workflow service connection test failed: {str(e)}")
            return False

    async def initiate_rejection_review(self, 
                                document_id: str,
                                rejection_reason: RejectionReason,
                                initiated_by: str,
                                notes: Optional[str] = None,
                                auto_approve: bool = False) -> str:
        """
        Start a rejection review process for a document.
        
        Args:
            document_id: UUID of the document to review
            rejection_reason: Reason for rejection
            initiated_by: User ID who initiated the rejection
            notes: Optional notes about the rejection
            auto_approve: Whether to auto-approve without review
            
        Returns:
            Rejection ID for tracking
            
        Raises:
            Exception: If rejection initiation fails
        """
        try:
            rejection_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            # Determine initial status
            status = RejectionStatus.APPROVED if auto_approve else RejectionStatus.PENDING
            
            # Calculate appeal deadline (7 days from now)
            appeal_deadline = current_time + timedelta(days=7)
            
            # Create rejection record
            query = """
            MATCH (d:Document {id: $document_id})
            CREATE (r:RejectionRecord {
                rejection_id: $rejection_id,
                document_id: $document_id,
                rejection_reason: $rejection_reason,
                rejection_status: $status,
                initiated_by: $initiated_by,
                created_at: $created_at,
                updated_at: $created_at,
                notes: $notes,
                appeal_deadline: $appeal_deadline
            })
            CREATE (d)-[:HAS_REJECTION]->(r)
            SET d.rejection_status = $status,
                d.rejected_at = $created_at
            RETURN r.rejection_id as rejection_id
            """
            
            parameters = {
                "document_id": document_id,
                "rejection_id": rejection_id,
                "rejection_reason": rejection_reason.value,
                "status": status.value,
                "initiated_by": initiated_by,
                "created_at": current_time.isoformat(),
                "notes": notes,
                "appeal_deadline": appeal_deadline.isoformat()
            }
            
            result = self.graph.query(query, parameters)
            
            if not result:
                raise Exception(f"Failed to create rejection record for document {document_id}")
            
            self.logger.info(f"Initiated rejection review {rejection_id} for document {document_id}")
            
            # Send notification if service is available
            if self.notification_service:
                self._notify_rejection_initiated(document_id, rejection_id, rejection_reason, initiated_by)
            
            return rejection_id
            
        except Exception as e:
            error_msg = f"Error initiating rejection review for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def auto_detect_rejections(self, 
                             document_ids: Optional[List[str]] = None,
                             validation_level: ValidationLevel = ValidationLevel.MEDIUM,
                             batch_size: int = 50) -> List[ValidationResult]:
        """
        Automatically detect documents that should be rejected based on rules.
        
        Args:
            document_ids: Optional list of specific documents to check
            validation_level: Strictness of validation
            batch_size: Number of documents to process per batch
            
        Returns:
            List of validation results with rejection recommendations
        """
        try:
            self.logger.info(f"Starting auto-detection of rejections with {validation_level.value} validation")
            
            # Get documents to validate
            if document_ids:
                documents = self._get_documents_by_ids(document_ids)
            else:
                documents = self._get_unvalidated_documents(batch_size)
            
            validation_results = []
            
            for doc in documents:
                try:
                    # Perform comprehensive validation
                    validation_result = await self._validate_document_comprehensive(
                        doc['document_id'], 
                        validation_level
                    )
                    
                    validation_results.append(validation_result)
                    
                    # Auto-reject if validation failed and not valid
                    if not validation_result.is_valid and validation_result.rejection_reasons:
                        primary_reason = validation_result.rejection_reasons[0]
                        await self.initiate_rejection_review(
                            doc['document_id'],
                            primary_reason,
                            "system_auto_detection",
                            f"Auto-detected: {', '.join([r.value for r in validation_result.rejection_reasons])}",
                            auto_approve=True
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error validating document {doc['document_id']}: {str(e)}")
                    
            self.logger.info(f"Auto-detection completed: {len(validation_results)} documents validated")
            return validation_results
            
        except Exception as e:
            error_msg = f"Error in auto-detection process: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def manual_rejection(self, 
                       document_id: str,
                       rejection_reason: RejectionReason,
                       user_id: str,
                       notes: str,
                       require_review: bool = True) -> str:
        """
        Process a manual document rejection initiated by a user.
        
        Args:
            document_id: Document to reject
            rejection_reason: Reason for rejection
            user_id: User initiating the rejection
            notes: Required notes for manual rejection
            require_review: Whether rejection needs approval
            
        Returns:
            Rejection ID
        """
        try:
            # Validate that document exists and is not already rejected
            doc_status = await self._get_document_status(document_id)
            if not doc_status:
                raise Exception(f"Document {document_id} not found")
            
            if doc_status.get('rejection_status') == RejectionStatus.APPROVED.value:
                raise Exception(f"Document {document_id} is already rejected")
            
            # Create rejection with appropriate status
            rejection_id = await self.initiate_rejection_review(
                document_id,
                rejection_reason,
                user_id,
                notes,
                auto_approve=not require_review
            )
            
            # Log manual rejection
            await self._log_rejection_activity(
                rejection_id,
                "manual_rejection_initiated",
                user_id,
                {"require_review": require_review, "notes": notes}
            )
            
            self.logger.info(f"Manual rejection {rejection_id} initiated by user {user_id} for document {document_id}")
            
            return rejection_id
            
        except Exception as e:
            error_msg = f"Error processing manual rejection for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def bulk_rejection(self, 
                     document_ids: List[str],
                     rejection_reason: RejectionReason,
                     user_id: str,
                     notes: str) -> BulkRejectionResult:
        """
        Process bulk rejection of multiple documents.
        
        Args:
            document_ids: List of document IDs to reject
            rejection_reason: Common rejection reason
            user_id: User initiating bulk rejection
            notes: Notes for the bulk rejection
            
        Returns:
            BulkRejectionResult with processing summary
        """
        start_time = datetime.now()
        successful_rejections = 0
        failed_rejections = 0
        rejection_ids = []
        errors = []
        
        try:
            self.logger.info(f"Starting bulk rejection of {len(document_ids)} documents by user {user_id}")
            
            for document_id in document_ids:
                try:
                    rejection_id = await self.manual_rejection(
                        document_id,
                        rejection_reason,
                        user_id,
                        f"Bulk rejection: {notes}",
                        require_review=False  # Bulk rejections are auto-approved
                    )
                    
                    rejection_ids.append(rejection_id)
                    successful_rejections += 1
                    
                except Exception as e:
                    failed_rejections += 1
                    error_info = {
                        "document_id": document_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    errors.append(error_info)
                    self.logger.error(f"Failed to reject document {document_id} in bulk operation: {str(e)}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = BulkRejectionResult(
                total_documents=len(document_ids),
                successful_rejections=successful_rejections,
                failed_rejections=failed_rejections,
                processing_time=processing_time,
                rejection_ids=rejection_ids,
                errors=errors
            )
            
            self.logger.info(f"Bulk rejection completed: {successful_rejections}/{len(document_ids)} successful")
            
            # Send bulk notification
            if self.notification_service and successful_rejections > 0:
                self._notify_bulk_rejection(user_id, result)
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error in bulk rejection process: {str(e)}"
            self.logger.error(error_msg)
            
            return BulkRejectionResult(
                total_documents=len(document_ids),
                successful_rejections=successful_rejections,
                failed_rejections=failed_rejections,
                processing_time=processing_time,
                rejection_ids=rejection_ids,
                errors=errors + [{"error": error_msg, "timestamp": datetime.now().isoformat()}]
            )

    async def appeal_rejection(self, 
                       rejection_id: str,
                       user_id: str,
                       appeal_notes: str) -> bool:
        """
        Process an appeal for a rejected document.
        
        Args:
            rejection_id: ID of the rejection to appeal
            user_id: User submitting the appeal
            appeal_notes: Notes explaining the appeal
            
        Returns:
            True if appeal was successfully submitted
        """
        try:
            # Verify rejection exists and can be appealed
            rejection_info = await self._get_rejection_details(rejection_id)
            if not rejection_info:
                raise Exception(f"Rejection {rejection_id} not found")
            
            # Check if appeal is within deadline
            appeal_deadline = datetime.fromisoformat(rejection_info['appeal_deadline'])
            if datetime.now() > appeal_deadline:
                raise Exception(f"Appeal deadline has passed for rejection {rejection_id}")
            
            # Update rejection record with appeal
            query = """
            MATCH (r:RejectionRecord {rejection_id: $rejection_id})
            SET r.rejection_status = $status,
                r.appeal_notes = $appeal_notes,
                r.appealed_by = $user_id,
                r.appealed_at = $appealed_at,
                r.updated_at = $appealed_at
            RETURN r.rejection_id as rejection_id
            """
            
            parameters = {
                "rejection_id": rejection_id,
                "status": RejectionStatus.APPEALED.value,
                "appeal_notes": appeal_notes,
                "user_id": user_id,
                "appealed_at": datetime.now().isoformat()
            }
            
            result = self.graph.query(query, parameters)
            
            if not result:
                raise Exception(f"Failed to update rejection {rejection_id} with appeal")
            
            # Log appeal activity
            await self._log_rejection_activity(
                rejection_id,
                "appeal_submitted",
                user_id,
                {"appeal_notes": appeal_notes}
            )
            
            # Send notification
            if self.notification_service:
                self._notify_appeal_submitted(rejection_id, user_id, appeal_notes)
            
            self.logger.info(f"Appeal submitted for rejection {rejection_id} by user {user_id}")
            return True
            
        except Exception as e:
            error_msg = f"Error processing appeal for rejection {rejection_id}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def validate_document_quality(self, document_id: str) -> ValidationResult:
        """
        Check document scan quality and readability.
        
        Args:
            document_id: Document to validate
            
        Returns:
            ValidationResult with quality assessment
        """
        try:
            rejection_reasons = []
            validation_details = {}
            quality_score = 100.0
            rule_violations = []
            
            # Get document information
            doc_info = await self._get_document_info(document_id)
            if not doc_info:
                raise Exception(f"Document {document_id} not found")
            
            # Check file size (too small might indicate poor scan)
            if 'file_size' in doc_info and doc_info['file_size'] < 50000:  # 50KB
                quality_score -= 30
                rejection_reasons.append(RejectionReason.POOR_QUALITY)
                rule_violations.append("File size too small")
                validation_details['file_size_issue'] = True
            
            # Check if document has extracted text
            text_length = len(doc_info.get('extracted_text', ''))
            if text_length < 50:  # Very little text extracted
                quality_score -= 40
                rejection_reasons.append(RejectionReason.POOR_QUALITY)
                rule_violations.append("Insufficient text extracted")
                validation_details['text_extraction_issue'] = True
            
            # Check for image quality indicators (if available)
            if 'image_metrics' in doc_info:
                metrics = doc_info['image_metrics']
                if metrics.get('blur_score', 0) > 0.8:  # High blur
                    quality_score -= 25
                    rejection_reasons.append(RejectionReason.POOR_QUALITY)
                    rule_violations.append("Document appears blurred")
                    validation_details['blur_detected'] = True
                
                if metrics.get('resolution_dpi', 300) < 150:  # Low resolution
                    quality_score -= 20
                    rejection_reasons.append(RejectionReason.POOR_QUALITY)
                    rule_violations.append("Low resolution scan")
                    validation_details['low_resolution'] = True
            
            # Determine overall validity
            is_valid = quality_score >= 60 and len(rejection_reasons) == 0
            
            validation_details.update({
                'quality_score': quality_score,
                'text_length': text_length,
                'file_size': doc_info.get('file_size'),
                'file_type': doc_info.get('file_type')
            })
            
            return ValidationResult(
                document_id=document_id,
                is_valid=is_valid,
                quality_score=quality_score,
                rejection_reasons=rejection_reasons,
                validation_details=validation_details,
                validation_time=datetime.now(),
                rule_violations=rule_violations
            )
            
        except Exception as e:
            error_msg = f"Error validating document quality for {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def check_for_duplicates(self, 
                           document_id: str,
                           similarity_threshold: float = 0.85) -> ValidationResult:
        """
        Check if document is a duplicate of existing documents.
        
        Args:
            document_id: Document to check
            similarity_threshold: Threshold for duplicate detection (0.0-1.0)
            
        Returns:
            ValidationResult indicating if document is a duplicate
        """
        try:
            rejection_reasons = []
            validation_details = {}
            quality_score = 100.0
            rule_violations = []
            
            # Get document information
            doc_info = await self._get_document_info(document_id)
            if not doc_info:
                raise Exception(f"Document {document_id} not found")
            
            # Check for exact file hash matches
            file_hash = doc_info.get('file_hash')
            if file_hash:
                duplicates = await self._find_documents_by_hash(file_hash, exclude_id=document_id)
                if duplicates:
                    quality_score = 0.0
                    rejection_reasons.append(RejectionReason.DUPLICATE)
                    rule_violations.append(f"Exact duplicate found: {duplicates[0]['document_id']}")
                    validation_details['exact_duplicates'] = duplicates
            
            # Check for similar content (if no exact duplicates found)
            if not rejection_reasons:
                similar_docs = await self._find_similar_documents(document_id, similarity_threshold)
                if similar_docs:
                    quality_score = max(0, 100 - (len(similar_docs) * 30))
                    if len(similar_docs) > 2 or similar_docs[0]['similarity'] > 0.95:
                        rejection_reasons.append(RejectionReason.DUPLICATE)
                        rule_violations.append(f"High similarity to existing documents")
                    validation_details['similar_documents'] = similar_docs
            
            # Check filename similarity
            filename_matches = await self._find_similar_filenames(
                doc_info.get('file_name', ''), 
                document_id
            )
            if filename_matches:
                validation_details['similar_filenames'] = filename_matches
                if len(filename_matches) > 0:
                    quality_score -= 10
            
            is_valid = len(rejection_reasons) == 0
            
            validation_details.update({
                'similarity_threshold': similarity_threshold,
                'file_hash': file_hash,
                'filename': doc_info.get('file_name')
            })
            
            return ValidationResult(
                document_id=document_id,
                is_valid=is_valid,
                quality_score=quality_score,
                rejection_reasons=rejection_reasons,
                validation_details=validation_details,
                validation_time=datetime.now(),
                rule_violations=rule_violations
            )
            
        except Exception as e:
            error_msg = f"Error checking for duplicates for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def verify_document_relevance(self, 
                                document_id: str,
                                required_keywords: Optional[List[str]] = None,
                                relevance_threshold: float = 0.3) -> ValidationResult:
        """
        Verify if document is relevant to the expected document types and content.
        
        Args:
            document_id: Document to verify
            required_keywords: Optional list of keywords that should be present
            relevance_threshold: Minimum relevance score (0.0-1.0)
            
        Returns:
            ValidationResult indicating document relevance
        """
        try:
            rejection_reasons = []
            validation_details = {}
            quality_score = 100.0
            rule_violations = []
            
            # Get document information
            doc_info = await self._get_document_info(document_id)
            if not doc_info:
                raise Exception(f"Document {document_id} not found")
            
            extracted_text = doc_info.get('extracted_text', '').lower()
            
            # Define utility bill keywords
            utility_keywords = [
                'electric', 'electricity', 'gas', 'water', 'sewer', 'utility',
                'kwh', 'kw', 'therms', 'gallons', 'bill', 'billing',
                'usage', 'consumption', 'meter', 'account', 'service'
            ]
            
            # Check for utility-related keywords
            found_keywords = []
            for keyword in utility_keywords:
                if keyword in extracted_text:
                    found_keywords.append(keyword)
            
            keyword_score = (len(found_keywords) / len(utility_keywords)) * 100
            
            # Check for required keywords if specified
            if required_keywords:
                required_found = []
                for keyword in required_keywords:
                    if keyword.lower() in extracted_text:
                        required_found.append(keyword)
                
                required_score = (len(required_found) / len(required_keywords)) * 100
                validation_details['required_keywords_found'] = required_found
                validation_details['required_keywords_score'] = required_score
                
                if required_score < (relevance_threshold * 100):
                    quality_score -= 50
                    rejection_reasons.append(RejectionReason.IRRELEVANT)
                    rule_violations.append("Required keywords not found")
            
            # Overall relevance scoring
            overall_relevance = keyword_score / 100
            
            if overall_relevance < relevance_threshold:
                quality_score = max(0, quality_score - 60)
                rejection_reasons.append(RejectionReason.IRRELEVANT)
                rule_violations.append(f"Low relevance score: {overall_relevance:.2f}")
            
            # Check document type appropriateness
            file_type = doc_info.get('file_type', '').lower()
            if file_type not in ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'tif']:
                quality_score -= 30
                rejection_reasons.append(RejectionReason.WRONG_FORMAT)
                rule_violations.append(f"Inappropriate file type: {file_type}")
            
            is_valid = len(rejection_reasons) == 0 and overall_relevance >= relevance_threshold
            
            validation_details.update({
                'utility_keywords_found': found_keywords,
                'keyword_score': keyword_score,
                'overall_relevance': overall_relevance,
                'relevance_threshold': relevance_threshold,
                'file_type': file_type,
                'text_length': len(extracted_text)
            })
            
            return ValidationResult(
                document_id=document_id,
                is_valid=is_valid,
                quality_score=quality_score,
                rejection_reasons=rejection_reasons,
                validation_details=validation_details,
                validation_time=datetime.now(),
                rule_violations=rule_violations
            )
            
        except Exception as e:
            error_msg = f"Error verifying document relevance for {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def notify_document_owner(self, 
                            document_id: str,
                            rejection_id: str,
                            notification_type: str = "email") -> bool:
        """
        Send notification to document owner about rejection.
        
        Args:
            document_id: Document that was rejected
            rejection_id: Rejection record ID
            notification_type: Type of notification (email, alert, etc.)
            
        Returns:
            True if notification sent successfully
        """
        try:
            if not self.notification_service:
                self.logger.warning("No notification service configured")
                return False
            
            # Get document and rejection information
            doc_info = await self._get_document_info(document_id)
            rejection_info = await self._get_rejection_details(rejection_id)
            
            if not doc_info or not rejection_info:
                raise Exception("Document or rejection information not found")
            
            # Prepare notification data
            notification_data = {
                'document_id': document_id,
                'document_name': doc_info.get('file_name'),
                'rejection_id': rejection_id,
                'rejection_reason': rejection_info['rejection_reason'],
                'rejection_date': rejection_info['created_at'],
                'appeal_deadline': rejection_info.get('appeal_deadline'),
                'notes': rejection_info.get('notes'),
                'owner_email': doc_info.get('uploaded_by_email', 'unknown@example.com')
            }
            
            # Send notification
            success = self.notification_service.send_rejection_notification(
                notification_data,
                notification_type
            )
            
            if success:
                # Log notification sent
                await self._log_rejection_activity(
                    rejection_id,
                    "notification_sent",
                    "system",
                    {"notification_type": notification_type, "recipient": notification_data['owner_email']}
                )
                
                self.logger.info(f"Rejection notification sent for document {document_id}")
            
            return success
            
        except Exception as e:
            error_msg = f"Error sending notification for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            return False

    async def create_rejection_report(self, 
                              start_date: datetime,
                              end_date: datetime,
                              include_details: bool = True) -> Dict[str, Any]:
        """
        Create a comprehensive rejection report for a date range.
        
        Args:
            start_date: Start of reporting period
            end_date: End of reporting period
            include_details: Whether to include detailed rejection records
            
        Returns:
            Dictionary containing rejection report data
        """
        try:
            # Query for rejection statistics
            stats_query = """
            MATCH (r:RejectionRecord)
            WHERE r.created_at >= $start_date AND r.created_at <= $end_date
            RETURN 
                count(r) as total_rejections,
                collect(DISTINCT r.rejection_reason) as rejection_reasons,
                collect(DISTINCT r.rejection_status) as statuses,
                collect(DISTINCT r.initiated_by) as initiators
            """
            
            stats_result = self.graph.query(stats_query, {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            
            if not stats_result:
                return {"error": "No data found for the specified date range"}
            
            stats = stats_result[0]
            
            # Query for rejection breakdown by reason
            reason_query = """
            MATCH (r:RejectionRecord)
            WHERE r.created_at >= $start_date AND r.created_at <= $end_date
            RETURN r.rejection_reason as reason, count(r) as count
            ORDER BY count DESC
            """
            
            reason_breakdown = self.graph.query(reason_query, {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            
            # Query for status breakdown
            status_query = """
            MATCH (r:RejectionRecord)
            WHERE r.created_at >= $start_date AND r.created_at <= $end_date
            RETURN r.rejection_status as status, count(r) as count
            ORDER BY count DESC
            """
            
            status_breakdown = self.graph.query(status_query, {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            
            report = {
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_rejections": stats["total_rejections"],
                    "unique_reasons": len(stats["rejection_reasons"]),
                    "unique_initiators": len(stats["initiators"])
                },
                "breakdown_by_reason": [
                    {"reason": r["reason"], "count": r["count"]} 
                    for r in reason_breakdown
                ],
                "breakdown_by_status": [
                    {"status": s["status"], "count": s["count"]} 
                    for s in status_breakdown
                ],
                "generated_at": datetime.now().isoformat()
            }
            
            # Include detailed records if requested
            if include_details:
                details_query = """
                MATCH (d:Document)-[:HAS_REJECTION]->(r:RejectionRecord)
                WHERE r.created_at >= $start_date AND r.created_at <= $end_date
                RETURN 
                    d.id as document_id,
                    d.fileName as document_name,
                    r.rejection_id as rejection_id,
                    r.rejection_reason as reason,
                    r.rejection_status as status,
                    r.initiated_by as initiated_by,
                    r.created_at as created_at,
                    r.notes as notes
                ORDER BY r.created_at DESC
                """
                
                details = self.graph.query(details_query, {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                })
                
                report["detailed_records"] = [
                    {
                        "document_id": d["document_id"],
                        "document_name": d["document_name"],
                        "rejection_id": d["rejection_id"],
                        "reason": d["reason"],
                        "status": d["status"],
                        "initiated_by": d["initiated_by"],
                        "created_at": d["created_at"],
                        "notes": d["notes"]
                    }
                    for d in details
                ]
            
            self.logger.info(f"Generated rejection report for period {start_date.date()} to {end_date.date()}")
            return report
            
        except Exception as e:
            error_msg = f"Error creating rejection report: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    # Integration with document processing pipeline
    
    async def integrate_with_processing_pipeline(self, 
                                         pipeline_hook: str,
                                         document_id: str,
                                         pipeline_data: Dict[str, Any]) -> bool:
        """
        Integration point for document processing pipeline.
        
        Args:
            pipeline_hook: Stage in pipeline (pre_processing, post_processing, etc.)
            document_id: Document being processed
            pipeline_data: Data from the processing pipeline
            
        Returns:
            True if document should continue processing, False if rejected
        """
        try:
            self.logger.debug(f"Pipeline integration called for {pipeline_hook} on document {document_id}")
            
            if pipeline_hook == "pre_processing":
                # Validate before processing
                validation_result = await self._validate_document_comprehensive(
                    document_id, 
                    ValidationLevel.MEDIUM
                )
                
                if not validation_result.is_valid:
                    # Auto-reject before processing
                    primary_reason = validation_result.rejection_reasons[0]
                    await self.initiate_rejection_review(
                        document_id,
                        primary_reason,
                        "pipeline_validation",
                        f"Pre-processing validation failed: {', '.join(validation_result.rule_violations)}",
                        auto_approve=True
                    )
                    return False
                
            elif pipeline_hook == "post_processing":
                # Additional validation after processing
                extracted_data = pipeline_data.get('extracted_data', {})
                
                # Check if sufficient data was extracted
                if not extracted_data or len(str(extracted_data)) < 100:
                    await self.initiate_rejection_review(
                        document_id,
                        RejectionReason.INSUFFICIENT_DATA,
                        "pipeline_validation",
                        "Insufficient data extracted during processing",
                        auto_approve=True
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in pipeline integration for document {document_id}: {str(e)}")
            return False

    # Additional API methods
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint for the rejection workflow service.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Test database connectivity
            test_query = "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            result = self.graph.query(test_query)
            
            return {
                "status": "healthy",
                "database_connected": True,
                "node_count": result[0]["node_count"] if result else 0,
                "timestamp": datetime.now().isoformat(),
                "service": "RejectionWorkflowService"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "service": "RejectionWorkflowService"
            }
    
    async def document_exists(self, document_id: str, driver=None) -> bool:
        """
        Check if a document exists in the database.
        
        Args:
            document_id: Document ID to check
            driver: Optional Neo4j driver (unused, kept for API compatibility)
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            doc_info = await self._get_document_info(document_id)
            return doc_info is not None
        except Exception as e:
            self.logger.error(f"Error checking document existence for {document_id}: {str(e)}")
            return False
    
    async def get_document_status(self, document_id: str, driver=None) -> Optional[Dict[str, Any]]:
        """
        Get the status of a document.
        
        Args:
            document_id: Document ID to get status for
            driver: Optional Neo4j driver (unused, kept for API compatibility)
            
        Returns:
            Dictionary containing document status or None if not found
        """
        try:
            return await self._get_document_status(document_id)
        except Exception as e:
            self.logger.error(f"Error getting document status for {document_id}: {str(e)}")
            return None
    
    async def reject_document(self, 
                       document_id: str,
                       rejection_reason: Union[str, RejectionReason],
                       user_id: str,
                       notes: str = "",
                       require_review: bool = False) -> str:
        """
        Reject a document (wrapper around manual_rejection).
        
        Args:
            document_id: Document to reject
            rejection_reason: Reason for rejection (string or RejectionReason enum)
            user_id: User initiating the rejection
            notes: Optional notes for the rejection
            require_review: Whether rejection needs approval
            
        Returns:
            Rejection ID
        """
        try:
            # Convert string reason to enum if needed
            if isinstance(rejection_reason, str):
                rejection_reason = RejectionReason(rejection_reason)
            
            return await self.manual_rejection(
                document_id=document_id,
                rejection_reason=rejection_reason,
                user_id=user_id,
                notes=notes or "Document rejected via API",
                require_review=require_review
            )
        except Exception as e:
            error_msg = f"Error rejecting document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    async def unreject_document(self, 
                         document_id: str,
                         user_id: str,
                         notes: str = "") -> bool:
        """
        Unreject (restore) a previously rejected document.
        
        Args:
            document_id: Document to unreject
            user_id: User performing the unreject action
            notes: Optional notes for the unreject action
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find the most recent rejection for this document
            query = """
            MATCH (d:Document {id: $document_id})-[:HAS_REJECTION]->(r:RejectionRecord)
            WHERE r.rejection_status IN ['approved', 'rejected']
            RETURN r.rejection_id as rejection_id
            ORDER BY r.created_at DESC
            LIMIT 1
            """
            
            result = self.graph.query(query, {"document_id": document_id})
            
            if not result:
                raise Exception(f"No active rejection found for document {document_id}")
            
            rejection_id = result[0]["rejection_id"]
            
            # Update rejection status to resolved
            update_query = """
            MATCH (r:RejectionRecord {rejection_id: $rejection_id})
            MATCH (d:Document {id: $document_id})
            SET r.rejection_status = 'resolved',
                r.resolved_by = $user_id,
                r.resolved_at = $resolved_at,
                r.resolution_notes = $notes,
                r.updated_at = $resolved_at,
                d.rejection_status = null,
                d.rejected_at = null
            RETURN r.rejection_id as rejection_id
            """
            
            resolved_at = datetime.now().isoformat()
            update_result = self.graph.query(update_query, {
                "rejection_id": rejection_id,
                "document_id": document_id,
                "user_id": user_id,
                "resolved_at": resolved_at,
                "notes": notes or "Document unrejected via API"
            })
            
            if update_result:
                await self._log_rejection_activity(
                    rejection_id,
                    "document_unrejected",
                    user_id,
                    {"notes": notes, "resolved_at": resolved_at}
                )
                
                self.logger.info(f"Document {document_id} unrejected by user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            error_msg = f"Error unrejecting document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def get_rejected_documents(self, 
                              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get list of rejected documents.
        
        Args:
            filters: Optional filters containing limit, offset, status
            
        Returns:
            List of rejected document records
        """
        try:
            # Extract filters with defaults
            limit = filters.get('limit', 50) if filters else 50
            offset = filters.get('offset', 0) if filters else 0
            status = filters.get('status') if filters else None
            
            where_clause = ""
            if status:
                where_clause = "AND r.rejection_status = $status"
            
            query = f"""
            MATCH (d:Document)-[:HAS_REJECTION]->(r:RejectionRecord)
            WHERE r.rejection_status IN ['approved', 'rejected'] {where_clause}
            RETURN d.id as document_id,
                   d.fileName as file_name,
                   r.rejection_id as rejection_id,
                   r.rejection_reason as rejection_reason,
                   r.rejection_status as rejection_status,
                   r.initiated_by as initiated_by,
                   r.created_at as created_at,
                   r.notes as notes
            ORDER BY r.created_at DESC
            SKIP $offset
            LIMIT $limit
            """
            
            params = {"limit": limit, "offset": offset}
            if status:
                params["status"] = status
            
            results = self.graph.query(query, params)
            
            return [
                {
                    "document_id": r["document_id"],
                    "file_name": r["file_name"],
                    "rejection_id": r["rejection_id"],
                    "rejection_reason": r["rejection_reason"],
                    "rejection_status": r["rejection_status"],
                    "initiated_by": r["initiated_by"],
                    "created_at": r["created_at"],
                    "notes": r["notes"]
                }
                for r in results
            ]
            
        except Exception as e:
            error_msg = f"Error getting rejected documents: {str(e)}"
            self.logger.error(error_msg)
            return []
    
    async def count_rejected_documents(self, status: Optional[str] = None) -> int:
        """
        Count rejected documents.
        
        Args:
            status: Optional status filter
            
        Returns:
            Number of rejected documents
        """
        try:
            where_clause = ""
            if status:
                where_clause = "AND r.rejection_status = $status"
            
            query = f"""
            MATCH (d:Document)-[:HAS_REJECTION]->(r:RejectionRecord)
            WHERE r.rejection_status IN ['approved', 'rejected'] {where_clause}
            RETURN count(r) as count
            """
            
            params = {}
            if status:
                params["status"] = status
            
            result = self.graph.query(query, params)
            return result[0]["count"] if result else 0
            
        except Exception as e:
            error_msg = f"Error counting rejected documents: {str(e)}"
            self.logger.error(error_msg)
            return 0
    
    async def get_documents_by_status(self, 
                               status: str,
                               limit: int = 50,
                               offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get documents by rejection status.
        
        Args:
            status: Rejection status to filter by
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document records with the specified status
        """
        try:
            query = """
            MATCH (d:Document)-[:HAS_REJECTION]->(r:RejectionRecord)
            WHERE r.rejection_status = $status
            RETURN d.id as document_id,
                   d.fileName as file_name,
                   r.rejection_id as rejection_id,
                   r.rejection_reason as rejection_reason,
                   r.rejection_status as rejection_status,
                   r.initiated_by as initiated_by,
                   r.created_at as created_at,
                   r.notes as notes
            ORDER BY r.created_at DESC
            SKIP $offset
            LIMIT $limit
            """
            
            results = self.graph.query(query, {
                "status": status,
                "limit": limit,
                "offset": offset
            })
            
            return [
                {
                    "document_id": r["document_id"],
                    "file_name": r["file_name"],
                    "rejection_id": r["rejection_id"],
                    "rejection_reason": r["rejection_reason"],
                    "rejection_status": r["rejection_status"],
                    "initiated_by": r["initiated_by"],
                    "created_at": r["created_at"],
                    "notes": r["notes"]
                }
                for r in results
            ]
            
        except Exception as e:
            error_msg = f"Error getting documents by status {status}: {str(e)}"
            self.logger.error(error_msg)
            return []
    
    async def count_documents_by_status(self, status: str) -> int:
        """
        Count documents by rejection status.
        
        Args:
            status: Rejection status to count
            
        Returns:
            Number of documents with the specified status
        """
        try:
            query = """
            MATCH (d:Document)-[:HAS_REJECTION]->(r:RejectionRecord)
            WHERE r.rejection_status = $status
            RETURN count(r) as count
            """
            
            result = self.graph.query(query, {"status": status})
            return result[0]["count"] if result else 0
            
        except Exception as e:
            error_msg = f"Error counting documents by status {status}: {str(e)}"
            self.logger.error(error_msg)
            return 0
    
    async def bulk_reject_documents(self, 
                             document_ids: List[str],
                             rejection_reason: Union[str, RejectionReason],
                             user_id: str,
                             notes: str = "") -> BulkRejectionResult:
        """
        Bulk reject multiple documents (wrapper around bulk_rejection).
        
        Args:
            document_ids: List of document IDs to reject
            rejection_reason: Reason for rejection (string or RejectionReason enum)
            user_id: User initiating bulk rejection
            notes: Optional notes for the bulk rejection
            
        Returns:
            BulkRejectionResult with processing summary
        """
        try:
            # Convert string reason to enum if needed
            if isinstance(rejection_reason, str):
                rejection_reason = RejectionReason(rejection_reason)
            
            return await self.bulk_rejection(
                document_ids=document_ids,
                rejection_reason=rejection_reason,
                user_id=user_id,
                notes=notes or "Bulk rejection via API"
            )
            
        except Exception as e:
            error_msg = f"Error in bulk document rejection: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    async def get_rejection_history(self, 
                             document_id: Optional[str] = None,
                             limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get rejection history for a document or all documents.
        
        Args:
            document_id: Optional document ID to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of rejection history records
        """
        try:
            where_clause = ""
            if document_id:
                where_clause = "WHERE r.document_id = $document_id"
            
            query = f"""
            MATCH (r:RejectionRecord)
            {where_clause}
            OPTIONAL MATCH (r)-[:HAS_ACTIVITY]->(a:RejectionActivity)
            RETURN r.rejection_id as rejection_id,
                   r.document_id as document_id,
                   r.rejection_reason as rejection_reason,
                   r.rejection_status as rejection_status,
                   r.initiated_by as initiated_by,
                   r.created_at as created_at,
                   r.updated_at as updated_at,
                   r.notes as notes,
                   r.appeal_notes as appeal_notes,
                   r.appealed_by as appealed_by,
                   r.appealed_at as appealed_at,
                   collect(a) as activities
            ORDER BY r.created_at DESC
            LIMIT $limit
            """
            
            params = {"limit": limit}
            if document_id:
                params["document_id"] = document_id
            
            results = self.graph.query(query, params)
            
            return [
                {
                    "rejection_id": r["rejection_id"],
                    "document_id": r["document_id"],
                    "rejection_reason": r["rejection_reason"],
                    "rejection_status": r["rejection_status"],
                    "initiated_by": r["initiated_by"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "notes": r["notes"],
                    "appeal_notes": r["appeal_notes"],
                    "appealed_by": r["appealed_by"],
                    "appealed_at": r["appealed_at"],
                    "activities": r["activities"] or []
                }
                for r in results
            ]
            
        except Exception as e:
            error_msg = f"Error getting rejection history: {str(e)}"
            self.logger.error(error_msg)
            return []
    
    async def get_active_appeal(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if there's an active appeal for a document.
        
        Args:
            document_id: Document ID to check for appeals
            
        Returns:
            Active appeal information or None if no active appeal
        """
        try:
            query = """
            MATCH (d:Document {id: $document_id})-[:HAS_REJECTION]->(r:RejectionRecord)
            WHERE r.rejection_status = 'appealed'
            RETURN r.rejection_id as rejection_id,
                   r.appeal_notes as appeal_notes,
                   r.appealed_by as appealed_by,
                   r.appealed_at as appealed_at,
                   r.appeal_deadline as appeal_deadline
            ORDER BY r.appealed_at DESC
            LIMIT 1
            """
            
            result = self.graph.query(query, {"document_id": document_id})
            
            if result:
                appeal = result[0]
                return {
                    "rejection_id": appeal["rejection_id"],
                    "appeal_notes": appeal["appeal_notes"],
                    "appealed_by": appeal["appealed_by"],
                    "appealed_at": appeal["appealed_at"],
                    "appeal_deadline": appeal["appeal_deadline"]
                }
            
            return None
            
        except Exception as e:
            error_msg = f"Error checking for active appeal for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            return None
    
    async def create_appeal(self, 
                     rejection_id: str,
                     user_id: str,
                     appeal_notes: str) -> bool:
        """
        Create an appeal for a rejection (wrapper around appeal_rejection).
        
        Args:
            rejection_id: ID of the rejection to appeal
            user_id: User submitting the appeal
            appeal_notes: Notes explaining the appeal
            
        Returns:
            True if appeal was successfully created
        """
        try:
            return await self.appeal_rejection(
                rejection_id=rejection_id,
                user_id=user_id,
                appeal_notes=appeal_notes
            )
        except Exception as e:
            error_msg = f"Error creating appeal for rejection {rejection_id}: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def get_rejection_statistics(self, 
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get rejection statistics for a date range.
        
        Args:
            start_date: Optional start date for statistics
            end_date: Optional end date for statistics
            
        Returns:
            Dictionary containing rejection statistics
        """
        try:
            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            query = """
            MATCH (r:RejectionRecord)
            WHERE r.created_at >= $start_date AND r.created_at <= $end_date
            RETURN 
                count(r) as total_rejections,
                collect(r.rejection_reason) as reasons,
                collect(r.rejection_status) as statuses,
                collect(r.initiated_by) as initiators
            """
            
            result = self.graph.query(query, {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            
            if result:
                data = result[0]
                
                # Count by reason
                reason_counts = {}
                for reason in data["reasons"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                # Count by status
                status_counts = {}
                for status in data["statuses"]:
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                # Count by initiator
                initiator_counts = {}
                for initiator in data["initiators"]:
                    initiator_counts[initiator] = initiator_counts.get(initiator, 0) + 1
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat()
                    },
                    "total_rejections": data["total_rejections"],
                    "breakdown_by_reason": reason_counts,
                    "breakdown_by_status": status_counts,
                    "breakdown_by_initiator": initiator_counts,
                    "generated_at": datetime.now().isoformat()
                }
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "total_rejections": 0,
                "breakdown_by_reason": {},
                "breakdown_by_status": {},
                "breakdown_by_initiator": {},
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error getting rejection statistics: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    async def get_rejection_trends(self, 
                            days: int = 30,
                            group_by: str = "day") -> List[Dict[str, Any]]:
        """
        Get rejection trends over time.
        
        Args:
            days: Number of days to analyze
            group_by: Grouping interval ('day', 'week', 'month')
            
        Returns:
            List of trend data points
        """
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Simplified trend query - group by day
            query = """
            MATCH (r:RejectionRecord)
            WHERE r.created_at >= $start_date
            WITH date(r.created_at) as rejection_date, count(r) as rejection_count
            RETURN rejection_date, rejection_count
            ORDER BY rejection_date DESC
            """
            
            results = self.graph.query(query, {
                "start_date": start_date.isoformat()
            })
            
            return [
                {
                    "date": str(r["rejection_date"]),
                    "count": r["rejection_count"]
                }
                for r in results
            ]
            
        except Exception as e:
            error_msg = f"Error getting rejection trends: {str(e)}"
            self.logger.error(error_msg)
            return []
    
    async def send_rejection_notification(self, 
                                   document_id: str,
                                   rejection_id: str,
                                   notification_type: str = "email") -> bool:
        """
        Send rejection notification (wrapper around notify_document_owner).
        
        Args:
            document_id: Document that was rejected
            rejection_id: Rejection record ID
            notification_type: Type of notification
            
        Returns:
            True if notification sent successfully
        """
        return await self.notify_document_owner(document_id, rejection_id, notification_type)
    
    async def send_unrerejection_notification(self, 
                                       document_id: str,
                                       user_id: str,
                                       notification_type: str = "email") -> bool:
        """
        Send notification when document is unrejected.
        
        Args:
            document_id: Document that was unrejected
            user_id: User who performed the unreject action
            notification_type: Type of notification
            
        Returns:
            True if notification sent successfully
        """
        try:
            if not self.notification_service:
                self.logger.warning("No notification service configured")
                return False
            
            # Get document information
            doc_info = await self._get_document_info(document_id)
            if not doc_info:
                return False
            
            # Prepare notification data
            notification_data = {
                "document_id": document_id,
                "document_name": doc_info.get("file_name"),
                "unrejected_by": user_id,
                "unrejected_at": datetime.now().isoformat(),
                "owner_email": doc_info.get("uploaded_by_email", "unknown@example.com")
            }
            
            # Send notification
            success = self.notification_service.send_unrerejection_notification(
                notification_data,
                notification_type
            )
            
            self.logger.info(f"Unrerejection notification sent for document {document_id}")
            return success
            
        except Exception as e:
            error_msg = f"Error sending unrerejection notification for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def send_bulk_rejection_notifications(self, 
                                         result: BulkRejectionResult,
                                         user_id: str,
                                         notification_type: str = "email") -> bool:
        """
        Send notifications for bulk rejection (wrapper around existing method).
        
        Args:
            result: Bulk rejection result
            user_id: User who performed bulk rejection
            notification_type: Type of notification
            
        Returns:
            True if notifications sent successfully
        """
        try:
            if self.notification_service and result.successful_rejections > 0:
                self._notify_bulk_rejection(user_id, result)
                return True
            return False
        except Exception as e:
            error_msg = f"Error sending bulk rejection notifications: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def send_appeal_notification(self, 
                                rejection_id: str,
                                user_id: str,
                                appeal_notes: str,
                                notification_type: str = "email") -> bool:
        """
        Send notification for appeal submission (wrapper around existing method).
        
        Args:
            rejection_id: Rejection ID that was appealed
            user_id: User who submitted the appeal
            appeal_notes: Appeal notes
            notification_type: Type of notification
            
        Returns:
            True if notification sent successfully
        """
        try:
            if self.notification_service:
                self._notify_appeal_submitted(rejection_id, user_id, appeal_notes)
                return True
            return False
        except Exception as e:
            error_msg = f"Error sending appeal notification: {str(e)}"
            self.logger.error(error_msg)
            return False

    # Private helper methods
    
    def _initialize_default_rules(self) -> List[RejectionRule]:
        """Initialize default rejection rules."""
        return [
            RejectionRule(
                rule_id="quality_check",
                rule_name="Document Quality Check",
                rule_type="quality",
                parameters={"min_text_length": 50, "min_file_size": 50000},
                enabled=True,
                priority=1,
                threshold=0.6
            ),
            RejectionRule(
                rule_id="duplicate_check",
                rule_name="Duplicate Detection",
                rule_type="duplicate",
                parameters={"similarity_threshold": 0.85},
                enabled=True,
                priority=2,
                threshold=0.85
            ),
            RejectionRule(
                rule_id="relevance_check",
                rule_name="Document Relevance",
                rule_type="relevance",
                parameters={"relevance_threshold": 0.3},
                enabled=True,
                priority=3,
                threshold=0.3
            )
        ]

    def _create_schema_constraints(self):
        """Create Neo4j constraints and indexes for rejection workflow."""
        constraints = [
            "CREATE CONSTRAINT rejection_id_unique IF NOT EXISTS FOR (r:RejectionRecord) REQUIRE r.rejection_id IS UNIQUE",
            "CREATE INDEX rejection_document_id IF NOT EXISTS FOR (r:RejectionRecord) ON (r.document_id)",
            "CREATE INDEX rejection_status IF NOT EXISTS FOR (r:RejectionRecord) ON (r.rejection_status)",
            "CREATE INDEX rejection_created_at IF NOT EXISTS FOR (r:RejectionRecord) ON (r.created_at)"
        ]
        
        for constraint in constraints:
            try:
                self.graph.query(constraint)
            except Exception as e:
                self.logger.warning(f"Could not create constraint/index: {constraint}, error: {str(e)}")

    async def _validate_document_comprehensive(self, 
                                       document_id: str, 
                                       validation_level: ValidationLevel) -> ValidationResult:
        """Perform comprehensive document validation."""
        try:
            # Perform all validation checks
            quality_result = await self.validate_document_quality(document_id)
            duplicate_result = await self.check_for_duplicates(document_id)
            relevance_result = await self.verify_document_relevance(document_id)
            
            # Combine results
            all_reasons = (quality_result.rejection_reasons + 
                         duplicate_result.rejection_reasons + 
                         relevance_result.rejection_reasons)
            
            all_violations = (quality_result.rule_violations + 
                            duplicate_result.rule_violations + 
                            relevance_result.rule_violations)
            
            # Calculate overall score based on validation level
            if validation_level == ValidationLevel.STRICT:
                overall_score = min(quality_result.quality_score, 
                                  duplicate_result.quality_score, 
                                  relevance_result.quality_score)
            else:
                overall_score = (quality_result.quality_score + 
                               duplicate_result.quality_score + 
                               relevance_result.quality_score) / 3
            
            # Combine validation details
            combined_details = {
                "quality": quality_result.validation_details,
                "duplicates": duplicate_result.validation_details,
                "relevance": relevance_result.validation_details,
                "validation_level": validation_level.value
            }
            
            is_valid = len(all_reasons) == 0 and overall_score >= 60
            
            return ValidationResult(
                document_id=document_id,
                is_valid=is_valid,
                quality_score=overall_score,
                rejection_reasons=list(set(all_reasons)),  # Remove duplicates
                validation_details=combined_details,
                validation_time=datetime.now(),
                rule_violations=all_violations
            )
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation for document {document_id}: {str(e)}")
            raise

    def _get_documents_by_ids(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Get document information for specific IDs."""
        query = """
        MATCH (d:Document)
        WHERE d.id IN $document_ids
        RETURN d.id as document_id, d.fileName as file_name, d.fileType as file_type
        """
        
        results = self.graph.query(query, {"document_ids": document_ids})
        return [{"document_id": r["document_id"], "file_name": r["file_name"], "file_type": r["file_type"]} for r in results]

    def _get_unvalidated_documents(self, limit: int) -> List[Dict[str, Any]]:
        """Get documents that haven't been validated yet."""
        query = """
        MATCH (d:Document)
        WHERE NOT EXISTS((d)-[:HAS_REJECTION]->())
        AND NOT d.validation_completed = true
        RETURN d.id as document_id, d.fileName as file_name, d.fileType as file_type
        LIMIT $limit
        """
        
        results = self.graph.query(query, {"limit": limit})
        return [{"document_id": r["document_id"], "file_name": r["file_name"], "file_type": r["file_type"]} for r in results]

    async def _get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive document information."""
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d.id as document_id,
               d.fileName as file_name,
               d.fileType as file_type,
               d.fileSize as file_size,
               d.extractedText as extracted_text,
               d.fileHash as file_hash,
               d.uploadedBy as uploaded_by_email
        """
        
        result = self.graph.query(query, {"document_id": document_id})
        
        if result:
            doc = result[0]
            return {
                "document_id": doc["document_id"],
                "file_name": doc["file_name"],
                "file_type": doc["file_type"],
                "file_size": doc["file_size"],
                "extracted_text": doc["extracted_text"] or "",
                "file_hash": doc["file_hash"],
                "uploaded_by_email": doc["uploaded_by_email"]
            }
        return None

    async def _get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get current document status."""
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d.rejection_status as rejection_status,
               d.rejected_at as rejected_at
        """
        
        result = self.graph.query(query, {"document_id": document_id})
        return result[0] if result else None

    async def _get_rejection_details(self, rejection_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a rejection."""
        query = """
        MATCH (r:RejectionRecord {rejection_id: $rejection_id})
        RETURN r.rejection_id as rejection_id,
               r.document_id as document_id,
               r.rejection_reason as rejection_reason,
               r.rejection_status as rejection_status,
               r.initiated_by as initiated_by,
               r.created_at as created_at,
               r.updated_at as updated_at,
               r.notes as notes,
               r.appeal_deadline as appeal_deadline,
               r.appeal_notes as appeal_notes,
               r.appealed_by as appealed_by,
               r.appealed_at as appealed_at
        """
        
        result = self.graph.query(query, {"rejection_id": rejection_id})
        return result[0] if result else None

    async def _find_documents_by_hash(self, file_hash: str, exclude_id: str) -> List[Dict[str, Any]]:
        """Find documents with matching file hash."""
        query = """
        MATCH (d:Document)
        WHERE d.fileHash = $file_hash AND d.id <> $exclude_id
        RETURN d.id as document_id, d.fileName as file_name, d.createdAt as created_at
        """
        
        results = self.graph.query(query, {"file_hash": file_hash, "exclude_id": exclude_id})
        return [
            {"document_id": r["document_id"], "file_name": r["file_name"], "created_at": r["created_at"]}
            for r in results
        ]

    async def _find_similar_documents(self, document_id: str, similarity_threshold: float) -> List[Dict[str, Any]]:
        """Find documents with similar content (placeholder implementation)."""
        # This would typically use vector similarity or text matching algorithms
        # For now, return empty list as placeholder
        return []

    async def _find_similar_filenames(self, filename: str, exclude_id: str) -> List[Dict[str, Any]]:
        """Find documents with similar filenames."""
        # Simple filename similarity based on partial matching
        if not filename:
            return []
        
        base_name = filename.split('.')[0].lower()
        
        query = """
        MATCH (d:Document)
        WHERE toLower(d.fileName) CONTAINS $base_name AND d.id <> $exclude_id
        RETURN d.id as document_id, d.fileName as file_name
        LIMIT 5
        """
        
        results = self.graph.query(query, {"base_name": base_name, "exclude_id": exclude_id})
        return [
            {"document_id": r["document_id"], "file_name": r["file_name"]}
            for r in results
        ]

    async def _log_rejection_activity(self, 
                              rejection_id: str,
                              activity_type: str,
                              user_id: str,
                              activity_data: Dict[str, Any]):
        """Log rejection workflow activities for audit trail."""
        try:
            query = """
            MATCH (r:RejectionRecord {rejection_id: $rejection_id})
            CREATE (a:RejectionActivity {
                activity_id: $activity_id,
                activity_type: $activity_type,
                user_id: $user_id,
                activity_data: $activity_data,
                created_at: $created_at
            })
            CREATE (r)-[:HAS_ACTIVITY]->(a)
            """
            
            self.graph.query(query, {
                "rejection_id": rejection_id,
                "activity_id": str(uuid.uuid4()),
                "activity_type": activity_type,
                "user_id": user_id,
                "activity_data": str(activity_data),  # Convert to string for Neo4j
                "created_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error logging rejection activity: {str(e)}")

    # Notification helper methods
    
    def _notify_rejection_initiated(self, 
                                  document_id: str,
                                  rejection_id: str,
                                  rejection_reason: RejectionReason,
                                  initiated_by: str):
        """Send notification when rejection is initiated."""
        try:
            if self.notification_service:
                self.notification_service.send_alert(
                    f"Document {document_id} rejected",
                    f"Reason: {rejection_reason.value}, Initiated by: {initiated_by}"
                )
        except Exception as e:
            self.logger.error(f"Error sending rejection notification: {str(e)}")

    def _notify_bulk_rejection(self, user_id: str, result: BulkRejectionResult):
        """Send notification for bulk rejection completion."""
        try:
            if self.notification_service:
                message = (f"Bulk rejection completed by {user_id}: "
                          f"{result.successful_rejections}/{result.total_documents} successful")
                self.notification_service.send_alert("Bulk Rejection Completed", message)
        except Exception as e:
            self.logger.error(f"Error sending bulk rejection notification: {str(e)}")

    def _notify_appeal_submitted(self, rejection_id: str, user_id: str, appeal_notes: str):
        """Send notification when appeal is submitted."""
        try:
            if self.notification_service:
                message = f"Appeal submitted for rejection {rejection_id} by {user_id}"
                self.notification_service.send_alert("Rejection Appeal Submitted", message)
        except Exception as e:
            self.logger.error(f"Error sending appeal notification: {str(e)}")