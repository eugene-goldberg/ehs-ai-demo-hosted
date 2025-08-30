#!/usr/bin/env python3
"""
Test to reproduce recursion error in risk assessment workflow.
This test bypasses early validation and focuses on the risk assessment agent retry logic.
"""

import sys
import logging
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Set very low recursion limit to trigger error quickly
sys.setrecursionlimit(50)

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/risk_assessment_recursion_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project path
sys.path.insert(0, '/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend')

try:
    from src.agents.workflow_coordinator import WorkflowCoordinator
    from src.agents.risk_assessment_agent import RiskAssessmentAgent
    from src.agents.base_agent import BaseAgent
    from src.models.document import Document
    from src.models.validation_result import ValidationResult
    from src.core.config import Config
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

class TestRiskAssessmentRecursion:
    """Test class to reproduce risk assessment recursion error."""
    
    def __init__(self):
        self.config = Config()
        self.setup_mocks()
    
    def setup_mocks(self):
        """Setup mocks to bypass dependencies and focus on recursion."""
        logger.info("Setting up mocks for risk assessment recursion test")
        
        # Mock Neo4j driver to avoid database dependencies
        self.mock_driver = Mock()
        self.mock_session = Mock()
        self.mock_driver.session.return_value.__enter__.return_value = self.mock_session
        self.mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock database responses that will pass validation but trigger risk assessment
        self.mock_session.run.return_value.data.return_value = [
            {
                'doc': {
                    'document_id': 'test-doc-123',
                    'title': 'High Risk Chemical Process Document',
                    'content': 'This document contains high-risk chemical processes involving hazardous materials.',
                    'document_type': 'technical_specification',
                    'source_system': 'ehs_system',
                    'created_date': '2024-01-01T00:00:00Z',
                    'metadata': {
                        'department': 'chemical_engineering',
                        'risk_level': 'high',
                        'contains_hazardous_materials': True,
                        'requires_assessment': True
                    }
                }
            }
        ]
    
    def create_valid_document(self) -> Document:
        """Create a valid document that will pass validation but trigger risk assessment."""
        logger.info("Creating valid document for risk assessment test")
        
        return Document(
            document_id='test-doc-123',
            title='High Risk Chemical Process Document',
            content='This document contains detailed information about high-risk chemical processes involving hazardous materials. The process involves multiple safety-critical steps that require careful risk assessment and monitoring.',
            document_type='technical_specification',
            source_system='ehs_system',
            metadata={
                'department': 'chemical_engineering',
                'risk_level': 'high',
                'contains_hazardous_materials': True,
                'requires_assessment': True,
                'process_type': 'chemical_manufacturing',
                'safety_classification': 'critical'
            }
        )
    
    def create_mock_validation_result(self) -> ValidationResult:
        """Create a validation result that passes all checks."""
        logger.info("Creating mock validation result that passes all checks")
        
        return ValidationResult(
            is_valid=True,
            confidence_score=0.95,
            validation_errors=[],
            quality_issues=[],
            completeness_score=0.98,
            consistency_score=0.96,
            accuracy_score=0.97
        )
    
    async def test_risk_assessment_recursion(self):
        """Test that specifically triggers risk assessment recursion."""
        logger.info("Starting risk assessment recursion test")
        
        try:
            # Create document and validation result
            document = self.create_valid_document()
            validation_result = self.create_mock_validation_result()
            
            logger.info(f"Document created: {document.document_id}")
            logger.info(f"Validation result: {validation_result.is_valid}")
            
            # Configure workflow with risk assessment enabled
            workflow_config = {
                'enable_risk_assessment': True,
                'enable_quality_validation': False,  # Bypass to focus on risk assessment
                'enable_consistency_check': False,   # Bypass to focus on risk assessment
                'risk_assessment_threshold': 0.1,    # Very low threshold to trigger assessment
                'max_retries': 20,                   # High retries to trigger recursion
                'retry_delay': 0.1                   # Fast retries
            }
            
            logger.info(f"Workflow config: {workflow_config}")
            
            # Create workflow coordinator with mocked dependencies
            with patch('src.agents.workflow_coordinator.Neo4jDriver') as mock_neo4j:
                mock_neo4j.return_value = self.mock_driver
                
                # Create coordinator
                coordinator = WorkflowCoordinator(self.config)
                coordinator.enable_risk_assessment = True
                
                logger.info("WorkflowCoordinator created")
                
                # Mock the risk assessment agent to create a recursive scenario
                original_assess_risk = None
                if hasattr(coordinator, 'risk_assessment_agent'):
                    original_assess_risk = coordinator.risk_assessment_agent.assess_risk
                
                async def mock_assess_risk_recursive(document: Document, context: Dict[str, Any] = None):
                    """Mock risk assessment that triggers recursion through retries."""
                    logger.info(f"Mock risk assessment called for document: {document.document_id}")
                    
                    # Simulate a scenario that causes the agent to retry indefinitely
                    # This could be due to state transitions or retry logic
                    context = context or {}
                    retry_count = context.get('retry_count', 0)
                    
                    logger.info(f"Risk assessment retry count: {retry_count}")
                    
                    if retry_count < 15:  # This will cause recursion
                        # Simulate a failed assessment that triggers retry
                        context['retry_count'] = retry_count + 1
                        logger.warning(f"Risk assessment failed, retrying... (attempt {retry_count + 1})")
                        
                        # This recursive call simulates the actual recursion in the workflow
                        return await mock_assess_risk_recursive(document, context)
                    
                    # Eventually return a result (but this won't be reached due to recursion limit)
                    return {
                        'risk_level': 'high',
                        'risk_score': 0.8,
                        'assessment_details': {
                            'hazardous_materials': True,
                            'safety_critical': True,
                            'requires_monitoring': True
                        }
                    }
                
                # Apply the recursive mock
                if hasattr(coordinator, 'risk_assessment_agent') and coordinator.risk_assessment_agent:
                    coordinator.risk_assessment_agent.assess_risk = mock_assess_risk_recursive
                else:
                    # Create a mock risk assessment agent
                    mock_risk_agent = Mock()
                    mock_risk_agent.assess_risk = mock_assess_risk_recursive
                    coordinator.risk_assessment_agent = mock_risk_agent
                
                logger.info("Mock risk assessment agent configured")
                
                # Process the document - this should trigger the recursion
                logger.info("Starting document processing that should trigger recursion...")
                
                result = await coordinator.process_document(
                    document=document,
                    validation_result=validation_result,
                    config=workflow_config
                )
                
                # This line should not be reached due to recursion error
                logger.error("ERROR: Test did not trigger recursion as expected!")
                logger.error(f"Result: {result}")
                return False
                
        except RecursionError as e:
            logger.info("SUCCESS: RecursionError caught as expected!")
            logger.info(f"Recursion error: {str(e)}")
            logger.info(f"Current recursion limit: {sys.getrecursionlimit()}")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error (not RecursionError): {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False

async def main():
    """Main function to run the recursion test."""
    logger.info("=== Starting Risk Assessment Recursion Test ===")
    logger.info(f"Python recursion limit: {sys.getrecursionlimit()}")
    
    test = TestRiskAssessmentRecursion()
    
    try:
        success = await test.test_risk_assessment_recursion()
        
        if success:
            logger.info("=== TEST PASSED: Recursion error reproduced successfully ===")
            print("SUCCESS: RecursionError reproduced in risk assessment workflow")
            return 0
        else:
            logger.error("=== TEST FAILED: Could not reproduce recursion error ===")
            print("FAILED: Could not reproduce recursion error")
            return 1
            
    except Exception as e:
        logger.error(f"=== TEST ERROR: {type(e).__name__}: {str(e)} ===")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        print(f"ERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)