#!/usr/bin/env python3
"""
Audit Trail Integration Examples
===============================

This file provides comprehensive examples for integrating the audit trail feature
into the EHS AI Demo application. Use these examples as a guide for implementation.

Author: Generated for EHS AI Demo
Date: 2025-08-23
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Example 1: Integration into main.py
# =====================================

def integrate_audit_trail_into_main():
    """
    Example showing how to integrate audit trail into the main FastAPI application
    """
    
    # main.py integration example
    main_py_integration = '''
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from .database import get_db
from .audit_trail import AuditTrail, AuditAction
from .models import Document
import json

app = FastAPI(title="EHS AI Demo with Audit Trail")

# Initialize audit trail
audit_trail = AuditTrail()

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = "system",
    db: Session = Depends(get_db)
):
    try:
        # Process document upload
        document = Document(
            filename=file.filename,
            content_type=file.content_type,
            size=file.size,
            uploaded_by=user_id
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Log the upload action
        await audit_trail.log_action(
            action=AuditAction.CREATE,
            entity_type="document",
            entity_id=str(document.id),
            user_id=user_id,
            details={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size
            },
            db=db
        )
        
        return {"document_id": document.id, "status": "uploaded"}
        
    except Exception as e:
        # Log the error
        await audit_trail.log_action(
            action=AuditAction.ERROR,
            entity_type="document",
            entity_id=None,
            user_id=user_id,
            details={
                "error": str(e),
                "filename": file.filename if file else "unknown"
            },
            db=db
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/audit")
async def get_document_audit_trail(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Get audit trail for a specific document"""
    try:
        trail = await audit_trail.get_entity_trail(
            entity_type="document",
            entity_id=document_id,
            db=db
        )
        return {"document_id": document_id, "audit_trail": trail}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    '''
    
    return main_py_integration


# Example 2: Environment-specific Configuration
# ============================================

class AuditTrailConfig:
    """Configuration examples for different environments"""
    
    @staticmethod
    def get_development_config():
        """Development environment configuration"""
        return {
            # Database settings
            "DATABASE_URL": "sqlite:///./audit_dev.db",
            "AUDIT_TABLE_NAME": "audit_logs_dev",
            
            # Logging settings
            "AUDIT_LOG_LEVEL": "DEBUG",
            "AUDIT_LOG_TO_FILE": True,
            "AUDIT_LOG_FILE": "./logs/audit_dev.log",
            
            # Retention settings
            "AUDIT_RETENTION_DAYS": 30,
            "AUDIT_CLEANUP_ENABLED": True,
            
            # Performance settings
            "AUDIT_BATCH_SIZE": 10,
            "AUDIT_ASYNC_ENABLED": True,
            
            # Security settings
            "AUDIT_ENCRYPT_SENSITIVE": False,
            "AUDIT_HASH_USER_IDS": False
        }
    
    @staticmethod
    def get_staging_config():
        """Staging environment configuration"""
        return {
            # Database settings
            "DATABASE_URL": os.getenv("STAGING_DATABASE_URL"),
            "AUDIT_TABLE_NAME": "audit_logs_staging",
            
            # Logging settings
            "AUDIT_LOG_LEVEL": "INFO",
            "AUDIT_LOG_TO_FILE": True,
            "AUDIT_LOG_FILE": "./logs/audit_staging.log",
            
            # Retention settings
            "AUDIT_RETENTION_DAYS": 90,
            "AUDIT_CLEANUP_ENABLED": True,
            
            # Performance settings
            "AUDIT_BATCH_SIZE": 50,
            "AUDIT_ASYNC_ENABLED": True,
            
            # Security settings
            "AUDIT_ENCRYPT_SENSITIVE": True,
            "AUDIT_HASH_USER_IDS": True
        }
    
    @staticmethod
    def get_production_config():
        """Production environment configuration"""
        return {
            # Database settings
            "DATABASE_URL": os.getenv("PROD_DATABASE_URL"),
            "AUDIT_TABLE_NAME": "audit_logs",
            
            # Logging settings
            "AUDIT_LOG_LEVEL": "WARNING",
            "AUDIT_LOG_TO_FILE": True,
            "AUDIT_LOG_FILE": "./logs/audit_prod.log",
            
            # Retention settings
            "AUDIT_RETENTION_DAYS": 365,
            "AUDIT_CLEANUP_ENABLED": True,
            
            # Performance settings
            "AUDIT_BATCH_SIZE": 100,
            "AUDIT_ASYNC_ENABLED": True,
            
            # Security settings
            "AUDIT_ENCRYPT_SENSITIVE": True,
            "AUDIT_HASH_USER_IDS": True,
            "AUDIT_REQUIRE_HTTPS": True
        }


# Example 3: Document Processing with Audit Trail
# ==============================================

class DocumentProcessingWithAudit:
    """Example of document processing with integrated audit trail"""
    
    def __init__(self, audit_trail, db_session):
        self.audit_trail = audit_trail
        self.db = db_session
    
    async def process_document_with_audit(
        self, 
        document_id: str, 
        user_id: str,
        processing_options: Dict[str, Any]
    ):
        """
        Example of processing a document with comprehensive audit logging
        """
        
        # Import AuditAction for type hints (would be imported at module level)
        from enum import Enum
        class AuditAction(Enum):
            CREATE = "CREATE"
            READ = "READ"
            UPDATE = "UPDATE"
            DELETE = "DELETE"
            PROCESS_START = "PROCESS_START"
            PROCESS_COMPLETE = "PROCESS_COMPLETE"
            VALIDATE = "VALIDATE"
            EXTRACT = "EXTRACT"
            ANALYZE = "ANALYZE"
            ERROR = "ERROR"
        
        # Log processing start
        await self.audit_trail.log_action(
            action=AuditAction.PROCESS_START,
            entity_type="document",
            entity_id=document_id,
            user_id=user_id,
            details={
                "processing_options": processing_options,
                "started_at": datetime.utcnow().isoformat()
            },
            db=self.db
        )
        
        try:
            # Example processing steps with audit logging
            
            # Step 1: Document validation
            validation_result = await self._validate_document(document_id)
            await self.audit_trail.log_action(
                action=AuditAction.VALIDATE,
                entity_type="document",
                entity_id=document_id,
                user_id=user_id,
                details={
                    "validation_result": validation_result,
                    "step": "validation"
                },
                db=self.db
            )
            
            if not validation_result["valid"]:
                raise Exception(f"Document validation failed: {validation_result['errors']}")
            
            # Step 2: Content extraction
            content = await self._extract_content(document_id)
            await self.audit_trail.log_action(
                action=AuditAction.EXTRACT,
                entity_type="document",
                entity_id=document_id,
                user_id=user_id,
                details={
                    "content_length": len(content),
                    "extraction_method": "automated",
                    "step": "content_extraction"
                },
                db=self.db
            )
            
            # Step 3: AI Analysis
            analysis_result = await self._analyze_content(content)
            await self.audit_trail.log_action(
                action=AuditAction.ANALYZE,
                entity_type="document",
                entity_id=document_id,
                user_id=user_id,
                details={
                    "analysis_confidence": analysis_result.get("confidence"),
                    "categories_found": analysis_result.get("categories", []),
                    "step": "ai_analysis"
                },
                db=self.db
            )
            
            # Step 4: Results storage
            await self._store_results(document_id, analysis_result)
            await self.audit_trail.log_action(
                action=AuditAction.UPDATE,
                entity_type="document",
                entity_id=document_id,
                user_id=user_id,
                details={
                    "results_stored": True,
                    "step": "results_storage"
                },
                db=self.db
            )
            
            # Log successful completion
            await self.audit_trail.log_action(
                action=AuditAction.PROCESS_COMPLETE,
                entity_type="document",
                entity_id=document_id,
                user_id=user_id,
                details={
                    "status": "success",
                    "completed_at": datetime.utcnow().isoformat(),
                    "processing_time_seconds": 120  # Example
                },
                db=self.db
            )
            
            return {
                "status": "success",
                "document_id": document_id,
                "analysis_result": analysis_result
            }
            
        except Exception as e:
            # Log processing failure
            await self.audit_trail.log_action(
                action=AuditAction.ERROR,
                entity_type="document",
                entity_id=document_id,
                user_id=user_id,
                details={
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat(),
                    "status": "failed"
                },
                db=self.db
            )
            raise
    
    async def _validate_document(self, document_id: str) -> Dict[str, Any]:
        """Mock document validation"""
        return {"valid": True, "errors": []}
    
    async def _extract_content(self, document_id: str) -> str:
        """Mock content extraction"""
        return "Sample document content for processing"
    
    async def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Mock AI analysis"""
        return {
            "confidence": 0.95,
            "categories": ["safety", "compliance"],
            "key_findings": ["No safety violations found"]
        }
    
    async def _store_results(self, document_id: str, results: Dict[str, Any]):
        """Mock results storage"""
        pass


# Example 4: Data Migration Script
# ===============================

class AuditTrailMigrationScript:
    """Example migration script for existing data"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    async def migrate_existing_documents(self):
        """
        Migrate existing documents to include audit trail entries
        """
        migration_script = '''
-- Example SQL migration script
-- File: migrations/001_add_audit_trail.sql

-- Create audit_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255),
    user_id VARCHAR(255) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);

-- Create audit trail entries for existing documents
INSERT INTO audit_logs (action, entity_type, entity_id, user_id, details, timestamp)
SELECT 
    'CREATE' as action,
    'document' as entity_type,
    id::text as entity_id,
    COALESCE(uploaded_by, 'system') as user_id,
    json_build_object(
        'filename', filename,
        'content_type', content_type,
        'size', size,
        'migrated', true
    ) as details,
    COALESCE(created_at, NOW()) as timestamp
FROM documents 
WHERE id NOT IN (
    SELECT DISTINCT entity_id::bigint 
    FROM audit_logs 
    WHERE entity_type = 'document' 
    AND action = 'CREATE'
    AND entity_id IS NOT NULL
);
        '''
        
        print("Migration Script:")
        print(migration_script)
        
        # Python-based migration example
        db = self.SessionLocal()
        try:
            # Get all existing documents without audit trail
            existing_docs = db.execute("""
                SELECT id, filename, content_type, size, 
                       uploaded_by, created_at
                FROM documents 
                WHERE id NOT IN (
                    SELECT DISTINCT entity_id::bigint 
                    FROM audit_logs 
                    WHERE entity_type = 'document' 
                    AND action = 'CREATE'
                )
            """).fetchall()
            
            print(f"Found {len(existing_docs)} documents to migrate")
            
            # Note: In real implementation, would import and use actual AuditTrail class
            print("Migration completed successfully")
            
        except Exception as e:
            db.rollback()
            print(f"Migration failed: {e}")
            raise
        finally:
            db.close()


# Example 5: Docker Compose Configuration
# ======================================

def get_docker_compose_config():
    """
    Example Docker Compose configuration with audit trail support
    """
    
    docker_compose_yaml = '''
version: '3.8'

services:
  # Main application
  ehs-ai-demo:
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/ehsai
      - AUDIT_DATABASE_URL=postgresql://user:password@postgres:5432/ehsai
      - AUDIT_LOG_LEVEL=INFO
      - AUDIT_RETENTION_DAYS=365
      - AUDIT_BATCH_SIZE=50
      - AUDIT_ASYNC_ENABLED=true
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    networks:
      - ehs-network

  # PostgreSQL database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=ehsai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d
    networks:
      - ehs-network

  # Redis for caching and session management
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ehs-network

  # Audit log analyzer (optional service)
  audit-analyzer:
    build:
      context: .
      dockerfile: Dockerfile.audit-analyzer
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/ehsai
      - ANALYSIS_SCHEDULE=0 2 * * *  # Daily at 2 AM
    depends_on:
      - postgres
    volumes:
      - ./audit-reports:/app/reports
    networks:
      - ehs-network

  # Log aggregation service
  filebeat:
    image: docker.elastic.co/beats/filebeat:8.8.0
    user: root
    volumes:
      - ./filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - ./logs:/app/logs:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - ehs-network

  # Elasticsearch for log storage
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - ehs-network

  # Kibana for log visualization
  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - ehs-network

volumes:
  postgres_data:
  redis_data:
  elasticsearch_data:

networks:
  ehs-network:
    driver: bridge

# Example .env file for audit trail configuration
# ================================================
# Copy to .env and customize for your environment

# Database Configuration
# DATABASE_URL=postgresql://user:password@localhost:5432/ehsai
# AUDIT_DATABASE_URL=postgresql://user:password@localhost:5432/ehsai

# Audit Trail Configuration
# AUDIT_LOG_LEVEL=INFO
# AUDIT_LOG_TO_FILE=true
# AUDIT_LOG_FILE=./logs/audit.log
# AUDIT_RETENTION_DAYS=365
# AUDIT_CLEANUP_ENABLED=true
# AUDIT_BATCH_SIZE=50
# AUDIT_ASYNC_ENABLED=true

# Security Configuration
# AUDIT_ENCRYPT_SENSITIVE=true
# AUDIT_HASH_USER_IDS=true
# AUDIT_REQUIRE_HTTPS=false

# Performance Configuration
# REDIS_URL=redis://localhost:6379
# AUDIT_CACHE_TTL=3600

# Monitoring Configuration
# ELASTICSEARCH_HOSTS=http://localhost:9200
# KIBANA_HOST=http://localhost:5601
    '''
    
    return docker_compose_yaml


# Example 6: Usage Examples and Best Practices
# ===========================================

class AuditTrailUsageExamples:
    """Comprehensive usage examples and best practices"""
    
    @staticmethod
    async def example_user_management_audit():
        """Example of auditing user management operations"""
        
        usage_example = '''
# User Management with Audit Trail

from audit_trail import AuditTrail, AuditAction

audit = AuditTrail()

# User login
await audit.log_action(
    action=AuditAction.LOGIN,
    entity_type="user",
    entity_id=user_id,
    user_id=user_id,
    details={
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent"),
        "login_method": "password"
    }
)

# User permission change
await audit.log_action(
    action=AuditAction.UPDATE,
    entity_type="user",
    entity_id=target_user_id,
    user_id=admin_user_id,
    details={
        "field_changed": "permissions",
        "old_value": old_permissions,
        "new_value": new_permissions,
        "reason": "Role update requested by manager"
    }
)

# User logout
await audit.log_action(
    action=AuditAction.LOGOUT,
    entity_type="user",
    entity_id=user_id,
    user_id=user_id,
    details={
        "session_duration_minutes": session_duration,
        "logout_type": "user_initiated"
    }
)
        '''
        return usage_example
    
    @staticmethod
    async def example_compliance_reporting():
        """Example of generating compliance reports from audit trail"""
        
        compliance_example = '''
# Compliance Reporting Examples

# Get all document access in the last 30 days
recent_access = await audit.get_actions_by_criteria(
    entity_type="document",
    actions=[AuditAction.READ, AuditAction.DOWNLOAD],
    start_date=datetime.utcnow() - timedelta(days=30),
    db=db
)

# Get all failed login attempts
failed_logins = await audit.get_actions_by_criteria(
    action=AuditAction.LOGIN_FAILED,
    entity_type="user",
    start_date=datetime.utcnow() - timedelta(days=7),
    db=db
)

# Get all administrative actions
admin_actions = await audit.get_actions_by_criteria(
    actions=[AuditAction.CREATE, AuditAction.UPDATE, AuditAction.DELETE],
    details_filter={"admin_action": True},
    db=db
)

# Generate compliance report
compliance_report = {
    "period": f"{start_date} to {end_date}",
    "total_document_access": len(recent_access),
    "unique_users": len(set(a.user_id for a in recent_access)),
    "security_incidents": len(failed_logins),
    "administrative_changes": len(admin_actions),
    "compliance_score": calculate_compliance_score(recent_access, failed_logins)
}
        '''
        return compliance_example


# Example 7: Testing Examples
# ==========================

class AuditTrailTestExamples:
    """Example test cases for audit trail functionality"""
    
    def test_integration_examples(self):
        """
        Example test cases to verify audit trail integration
        """
        
        test_code = '''
import pytest
from unittest.mock import AsyncMock, MagicMock
from audit_trail import AuditTrail, AuditAction

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def audit_trail():
    return AuditTrail()

@pytest.mark.asyncio
async def test_document_upload_audit(audit_trail, mock_db):
    """Test that document upload creates proper audit log"""
    
    # Arrange
    document_id = "123"
    user_id = "user123"
    file_details = {
        "filename": "test.pdf",
        "content_type": "application/pdf",
        "size": 1024
    }
    
    # Act
    await audit_trail.log_action(
        action=AuditAction.CREATE,
        entity_type="document",
        entity_id=document_id,
        user_id=user_id,
        details=file_details,
        db=mock_db
    )
    
    # Assert
    assert mock_db.add.called
    added_log = mock_db.add.call_args[0][0]
    assert added_log.action == "CREATE"
    assert added_log.entity_type == "document"
    assert added_log.entity_id == document_id
    assert added_log.user_id == user_id

@pytest.mark.asyncio
async def test_audit_trail_query(audit_trail, mock_db):
    """Test querying audit trail for specific entity"""
    
    # Arrange
    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
        MagicMock(action="CREATE", timestamp="2025-01-01"),
        MagicMock(action="READ", timestamp="2025-01-02")
    ]
    
    # Act
    trail = await audit_trail.get_entity_trail(
        entity_type="document",
        entity_id="123",
        db=mock_db
    )
    
    # Assert
    assert len(trail) == 2
    assert trail[0]["action"] == "CREATE"
    assert trail[1]["action"] == "READ"

# Performance test example
@pytest.mark.asyncio
async def test_audit_performance(audit_trail, mock_db):
    """Test audit trail performance with batch operations"""
    import time
    
    start_time = time.time()
    
    # Log 100 actions
    for i in range(100):
        await audit_trail.log_action(
            action=AuditAction.READ,
            entity_type="document",
            entity_id=f"doc_{i}",
            user_id="test_user",
            details={"batch_test": True},
            db=mock_db
        )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Should complete within reasonable time (adjust threshold as needed)
    assert processing_time < 1.0  # Less than 1 second for 100 operations
        '''
        
        return test_code


# Example 8: Monitoring and Alerting
# =================================

def get_monitoring_examples():
    """Examples of monitoring and alerting for audit trail"""
    
    monitoring_config = '''
# Prometheus metrics for audit trail monitoring
# File: monitoring/audit_metrics.py

from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics definitions
audit_actions_total = Counter(
    'audit_actions_total',
    'Total number of audit actions logged',
    ['action_type', 'entity_type']
)

audit_processing_duration = Histogram(
    'audit_processing_duration_seconds',
    'Time spent processing audit logs'
)

audit_queue_size = Gauge(
    'audit_queue_size',
    'Current size of audit processing queue'
)

# Example usage in audit trail
class AuditTrailWithMetrics(AuditTrail):
    async def log_action(self, **kwargs):
        start_time = time.time()
        
        try:
            result = await super().log_action(**kwargs)
            
            # Record metrics
            audit_actions_total.labels(
                action_type=kwargs.get('action'),
                entity_type=kwargs.get('entity_type')
            ).inc()
            
            return result
            
        finally:
            audit_processing_duration.observe(time.time() - start_time)

# Alerting rules (Prometheus AlertManager format)
# File: monitoring/audit_alerts.yml

groups:
  - name: audit_trail_alerts
    rules:
      - alert: HighFailedLoginRate
        expr: rate(audit_actions_total{action_type="LOGIN_FAILED"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High failed login rate detected"
          description: "Failed login rate is {{ $value }} per second"

      - alert: AuditProcessingDelay
        expr: audit_processing_duration_seconds > 1.0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Audit processing is slow"
          description: "Audit processing taking {{ $value }} seconds"

      - alert: AuditQueueBacklog
        expr: audit_queue_size > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Audit queue backlog detected"
          description: "Audit queue has {{ $value }} pending items"
    '''
    
    return monitoring_config


if __name__ == "__main__":
    """
    Example usage and testing of integration examples
    """
    print("EHS AI Demo - Audit Trail Integration Examples")
    print("=" * 50)
    
    # Example 1: Show main.py integration
    print("\n1. Main.py Integration Example:")
    print("-" * 30)
    integration = integrate_audit_trail_into_main()
    print(f"Integration code length: {len(integration)} characters")
    
    # Example 2: Show configuration examples
    print("\n2. Configuration Examples:")
    print("-" * 30)
    config = AuditTrailConfig()
    dev_config = config.get_development_config()
    print(f"Development config keys: {list(dev_config.keys())}")
    
    # Example 3: Show Docker compose
    print("\n3. Docker Compose Configuration:")
    print("-" * 30)
    docker_config = get_docker_compose_config()
    print(f"Docker compose config length: {len(docker_config)} characters")
    
    # Example 4: Show monitoring
    print("\n4. Monitoring Configuration:")
    print("-" * 30)
    monitoring = get_monitoring_examples()
    print(f"Monitoring config length: {len(monitoring)} characters")
    
    print("\n" + "=" * 50)
    print("Integration examples ready for use!")
    print("Refer to the comments and examples above for implementation guidance.")