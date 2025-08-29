#!/usr/bin/env python3
"""
Neo4j Historical Metrics Data Loader

This script loads historical EHS metrics data (electricity, water, waste) into Neo4j
as HistoricalMetric nodes with relationships to existing Facility nodes.

Features:
- Batch processing for efficient loading
- Incremental loading support
- Error handling and retry logic
- Performance indexes creation
- Comprehensive loading reports
- Memory-efficient processing

Author: AI Assistant
Date: 2025-08-28
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd
from contextlib import contextmanager
import traceback
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import TransientError, ServiceUnavailable, DatabaseError
    from dotenv import load_dotenv
    import numpy as np
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script in the virtual environment with required packages installed")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neo4j_loader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class HistoricalMetric:
    """Data class for historical metric records"""
    facility_id: str
    metric_type: str  # 'electricity', 'water', 'waste'
    value: float
    unit: str
    measurement_date: date
    period_type: str  # 'monthly', 'quarterly', 'yearly'
    source_system: str = 'synthetic'
    data_quality: str = 'good'  # 'good', 'fair', 'poor'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j insertion"""
        return {
            'facility_id': self.facility_id,
            'metric_type': self.metric_type,
            'value': float(self.value),
            'unit': self.unit,
            'measurement_date': self.measurement_date.isoformat(),
            'period_type': self.period_type,
            'source_system': self.source_system,
            'data_quality': self.data_quality,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class LoadingReport:
    """Data class for loading operation reports"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_records: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    skipped_records: int = 0
    batch_size: int = 0
    total_batches: int = 0
    facilities_processed: int = 0
    errors: List[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def records_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.successful_loads / self.duration_seconds
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'total_records': self.total_records,
            'successful_loads': self.successful_loads,
            'failed_loads': self.failed_loads,
            'skipped_records': self.skipped_records,
            'success_rate': (self.successful_loads / self.total_records * 100) if self.total_records > 0 else 0,
            'records_per_second': self.records_per_second,
            'batch_size': self.batch_size,
            'total_batches': self.total_batches,
            'facilities_processed': self.facilities_processed,
            'errors': self.errors,
            'performance_metrics': self.performance_metrics
        }


class Neo4jHistoricalMetricsLoader:
    """Neo4j loader for historical EHS metrics data"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j", 
                 batch_size: int = 1000, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize the Neo4j loader"""
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver = None
        self.connection_verified = False
        
        # Performance tracking
        self.stats = {
            'connections_created': 0,
            'queries_executed': 0,
            'total_query_time': 0.0,
            'retries_attempted': 0
        }
        
    def connect(self) -> bool:
        """Establish connection to Neo4j database with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password),
                    max_connection_lifetime=30 * 60,  # 30 minutes
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60  # 60 seconds
                )
                
                # Verify connection
                with self.driver.session(database=self.database) as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    if test_value == 1:
                        self.connection_verified = True
                        self.stats['connections_created'] += 1
                        logger.info(f"Successfully connected to Neo4j at {self.uri} (attempt {attempt + 1})")
                        return True
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to connect to Neo4j after {self.max_retries} attempts: {e}")
                    return False
        
        return False
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    @contextmanager
    def session_scope(self):
        """Context manager for Neo4j sessions with error handling"""
        session = None
        try:
            session = self.driver.session(database=self.database)
            yield session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def execute_query_with_retry(self, query: str, parameters: Dict = None, session=None) -> Any:
        """Execute Neo4j query with retry logic"""
        if parameters is None:
            parameters = {}
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                if session:
                    result = session.run(query, parameters)
                else:
                    with self.session_scope() as session:
                        result = session.run(query, parameters)
                
                # Consume result to ensure query execution
                result_data = list(result)
                
                query_time = time.time() - start_time
                self.stats['queries_executed'] += 1
                self.stats['total_query_time'] += query_time
                
                return result_data
                
            except (TransientError, ServiceUnavailable) as e:
                self.stats['retries_attempted'] += 1
                logger.warning(f"Transient error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Query failed after {self.max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                logger.error(f"Query: {query[:200]}...")
                raise
    
    def create_indexes(self) -> bool:
        """Create performance indexes for HistoricalMetric nodes"""
        indexes = [
            # Primary compound index for efficient lookups
            "CREATE INDEX historical_metric_facility_date IF NOT EXISTS "
            "FOR (m:HistoricalMetric) ON (m.facility_id, m.measurement_date)",
            
            # Index for metric type queries
            "CREATE INDEX historical_metric_type IF NOT EXISTS "
            "FOR (m:HistoricalMetric) ON (m.metric_type)",
            
            # Index for date range queries
            "CREATE INDEX historical_metric_date IF NOT EXISTS "
            "FOR (m:HistoricalMetric) ON (m.measurement_date)",
            
            # Index for facility lookups
            "CREATE INDEX historical_metric_facility IF NOT EXISTS "
            "FOR (m:HistoricalMetric) ON (m.facility_id)",
            
            # Index for period type filtering
            "CREATE INDEX historical_metric_period IF NOT EXISTS "
            "FOR (m:HistoricalMetric) ON (m.period_type)",
            
            # Unique constraint to prevent duplicates
            "CREATE CONSTRAINT historical_metric_unique IF NOT EXISTS "
            "FOR (m:HistoricalMetric) REQUIRE (m.facility_id, m.metric_type, m.measurement_date, m.period_type) IS UNIQUE"
        ]
        
        try:
            for index_query in indexes:
                logger.info(f"Creating index/constraint: {index_query[:60]}...")
                self.execute_query_with_retry(index_query)
                time.sleep(0.1)  # Brief pause between index operations
                
            logger.info("All indexes and constraints created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return False
    
    def check_facility_exists(self, facility_id: str) -> bool:
        """Check if a facility node exists"""
        query = "MATCH (f:Facility {facility_id: $facility_id}) RETURN count(f) as count"
        try:
            result = self.execute_query_with_retry(query, {'facility_id': facility_id})
            return result[0]['count'] > 0 if result else False
        except Exception as e:
            logger.error(f"Error checking facility existence: {e}")
            return False
    
    def get_existing_metrics(self, facility_id: str, metric_type: str, 
                           start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Get existing metrics for incremental loading"""
        query = """
        MATCH (m:HistoricalMetric {facility_id: $facility_id, metric_type: $metric_type})
        WHERE ($start_date IS NULL OR m.measurement_date >= $start_date)
        AND ($end_date IS NULL OR m.measurement_date <= $end_date)
        RETURN m.measurement_date as date, m.period_type as period
        """
        
        try:
            result = self.execute_query_with_retry(query, {
                'facility_id': facility_id,
                'metric_type': metric_type,
                'start_date': start_date,
                'end_date': end_date
            })
            return result
        except Exception as e:
            logger.error(f"Error getting existing metrics: {e}")
            return []
    
    def load_historical_metrics_batch(self, metrics: List[HistoricalMetric], 
                                    create_relationships: bool = True) -> Tuple[int, int, List[str]]:
        """Load a batch of historical metrics with relationships"""
        successful = 0
        failed = 0
        errors = []
        
        if not metrics:
            return successful, failed, errors
        
        # Prepare batch data
        batch_data = [metric.to_neo4j_dict() for metric in metrics]
        
        try:
            with self.session_scope() as session:
                # Create HistoricalMetric nodes
                create_query = """
                UNWIND $batch as row
                MERGE (m:HistoricalMetric {
                    facility_id: row.facility_id,
                    metric_type: row.metric_type,
                    measurement_date: row.measurement_date,
                    period_type: row.period_type
                })
                SET m.value = row.value,
                    m.unit = row.unit,
                    m.source_system = row.source_system,
                    m.data_quality = row.data_quality,
                    m.updated_at = row.updated_at,
                    m.created_at = COALESCE(m.created_at, row.created_at)
                RETURN count(m) as created
                """
                
                result = self.execute_query_with_retry(create_query, {'batch': batch_data}, session)
                created_count = result[0]['created'] if result else 0
                
                if create_relationships:
                    # Create relationships to Facility nodes
                    relationship_query = """
                    UNWIND $batch as row
                    MATCH (f:Facility {facility_id: row.facility_id})
                    MATCH (m:HistoricalMetric {
                        facility_id: row.facility_id,
                        metric_type: row.metric_type,
                        measurement_date: row.measurement_date,
                        period_type: row.period_type
                    })
                    MERGE (f)-[r:HAS_HISTORICAL_METRIC]->(m)
                    SET r.created_at = COALESCE(r.created_at, row.created_at)
                    RETURN count(r) as relationships_created
                    """
                    
                    rel_result = self.execute_query_with_retry(relationship_query, {'batch': batch_data}, session)
                    rel_count = rel_result[0]['relationships_created'] if rel_result else 0
                    
                    logger.debug(f"Created {created_count} metrics and {rel_count} relationships")
                
                successful = len(metrics)
                
        except Exception as e:
            error_msg = f"Batch loading error: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            failed = len(metrics)
        
        return successful, failed, errors
    
    def validate_metrics_data(self, metrics: List[HistoricalMetric]) -> Tuple[List[HistoricalMetric], List[str]]:
        """Validate metrics data before loading"""
        valid_metrics = []
        validation_errors = []
        
        required_fields = ['facility_id', 'metric_type', 'value', 'unit', 'measurement_date', 'period_type']
        valid_metric_types = ['electricity', 'water', 'waste']
        valid_period_types = ['monthly', 'quarterly', 'yearly']
        
        for i, metric in enumerate(metrics):
            errors = []
            
            # Check required fields
            for field in required_fields:
                if not hasattr(metric, field) or getattr(metric, field) is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validate metric type
            if hasattr(metric, 'metric_type') and metric.metric_type not in valid_metric_types:
                errors.append(f"Invalid metric_type: {metric.metric_type}")
            
            # Validate period type
            if hasattr(metric, 'period_type') and metric.period_type not in valid_period_types:
                errors.append(f"Invalid period_type: {metric.period_type}")
            
            # Validate numeric value
            if hasattr(metric, 'value') and metric.value is not None:
                try:
                    float(metric.value)
                    if metric.value < 0:
                        errors.append("Value cannot be negative")
                except (ValueError, TypeError):
                    errors.append("Value must be a valid number")
            
            # Validate date
            if hasattr(metric, 'measurement_date') and metric.measurement_date is not None:
                if not isinstance(metric.measurement_date, date):
                    errors.append("measurement_date must be a date object")
            
            if errors:
                validation_errors.extend([f"Record {i}: {error}" for error in errors])
            else:
                valid_metrics.append(metric)
        
        return valid_metrics, validation_errors
    
    def load_historical_metrics(self, metrics: Union[List[HistoricalMetric], str, Path], 
                              incremental: bool = True, validate: bool = True,
                              create_relationships: bool = True) -> LoadingReport:
        """Load historical metrics data into Neo4j"""
        
        report = LoadingReport(start_time=datetime.now(), batch_size=self.batch_size)
        
        try:
            # Handle different input types
            if isinstance(metrics, (str, Path)):
                # Load from file (CSV, JSON, etc.)
                metrics = self._load_metrics_from_file(metrics)
            
            if not isinstance(metrics, list):
                raise ValueError("Metrics must be a list of HistoricalMetric objects")
            
            report.total_records = len(metrics)
            logger.info(f"Starting to load {report.total_records} historical metrics")
            
            # Validate data if requested
            if validate:
                logger.info("Validating metrics data...")
                valid_metrics, validation_errors = self.validate_metrics_data(metrics)
                
                if validation_errors:
                    logger.warning(f"Found {len(validation_errors)} validation errors")
                    report.errors.extend(validation_errors[:10])  # Limit error list size
                
                metrics = valid_metrics
                report.skipped_records = report.total_records - len(metrics)
            
            if not metrics:
                logger.warning("No valid metrics to load")
                report.end_time = datetime.now()
                return report
            
            # Create indexes for performance
            logger.info("Creating/updating indexes...")
            if not self.create_indexes():
                logger.warning("Some indexes may not have been created properly")
            
            # Filter for incremental loading
            if incremental:
                logger.info("Filtering for incremental loading...")
                metrics = self._filter_for_incremental_loading(metrics)
                logger.info(f"After incremental filtering: {len(metrics)} metrics to load")
            
            # Process in batches
            total_batches = (len(metrics) + self.batch_size - 1) // self.batch_size
            report.total_batches = total_batches
            
            facilities_processed = set()
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(metrics))
                batch_metrics = metrics[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_metrics)} records)")
                
                # Track facilities in this batch
                for metric in batch_metrics:
                    facilities_processed.add(metric.facility_id)
                
                # Load batch
                successful, failed, batch_errors = self.load_historical_metrics_batch(
                    batch_metrics, create_relationships
                )
                
                report.successful_loads += successful
                report.failed_loads += failed
                report.errors.extend(batch_errors)
                
                # Progress logging
                if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
                    progress = (batch_idx + 1) / total_batches * 100
                    logger.info(f"Progress: {progress:.1f}% ({report.successful_loads} successful, {report.failed_loads} failed)")
                
                # Brief pause between batches to avoid overwhelming the database
                if batch_idx < total_batches - 1:
                    time.sleep(0.1)
            
            report.facilities_processed = len(facilities_processed)
            
            # Collect performance metrics
            report.performance_metrics = {
                'avg_query_time': self.stats['total_query_time'] / max(1, self.stats['queries_executed']),
                'total_queries': self.stats['queries_executed'],
                'retries_attempted': self.stats['retries_attempted'],
                'connections_created': self.stats['connections_created']
            }
            
        except Exception as e:
            error_msg = f"Critical loading error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            report.errors.append(error_msg)
        
        finally:
            report.end_time = datetime.now()
            
            # Log final results
            logger.info(f"Loading completed in {report.duration_seconds:.2f} seconds")
            logger.info(f"Total records: {report.total_records}")
            logger.info(f"Successful: {report.successful_loads}")
            logger.info(f"Failed: {report.failed_loads}")
            logger.info(f"Skipped: {report.skipped_records}")
            logger.info(f"Facilities processed: {report.facilities_processed}")
            logger.info(f"Records per second: {report.records_per_second:.2f}")
            
            if report.errors:
                logger.warning(f"Encountered {len(report.errors)} errors during loading")
        
        return report
    
    def _filter_for_incremental_loading(self, metrics: List[HistoricalMetric]) -> List[HistoricalMetric]:
        """Filter metrics for incremental loading (skip existing records)"""
        filtered_metrics = []
        
        # Group metrics by facility and metric type for efficient querying
        grouped_metrics = defaultdict(lambda: defaultdict(list))
        for metric in metrics:
            grouped_metrics[metric.facility_id][metric.metric_type].append(metric)
        
        for facility_id, metric_types in grouped_metrics.items():
            # Check if facility exists
            if not self.check_facility_exists(facility_id):
                logger.warning(f"Facility {facility_id} does not exist, skipping its metrics")
                continue
            
            for metric_type, type_metrics in metric_types.items():
                # Get date range for this group
                dates = [m.measurement_date for m in type_metrics]
                start_date = min(dates).isoformat()
                end_date = max(dates).isoformat()
                
                # Get existing metrics
                existing = self.get_existing_metrics(facility_id, metric_type, start_date, end_date)
                existing_keys = {(e['date'], e['period']) for e in existing}
                
                # Filter out existing metrics
                for metric in type_metrics:
                    key = (metric.measurement_date.isoformat(), metric.period_type)
                    if key not in existing_keys:
                        filtered_metrics.append(metric)
        
        logger.info(f"Incremental filtering: {len(metrics)} -> {len(filtered_metrics)} metrics")
        return filtered_metrics
    
    def _load_metrics_from_file(self, file_path: Union[str, Path]) -> List[HistoricalMetric]:
        """Load metrics from CSV, JSON, or Excel file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading metrics from file: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.json', '.jsonl']:
            if file_path.suffix.lower() == '.jsonl':
                df = pd.read_json(file_path, lines=True)
            else:
                df = pd.read_json(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Convert DataFrame to HistoricalMetric objects
        metrics = []
        for _, row in df.iterrows():
            try:
                metric = HistoricalMetric(
                    facility_id=str(row['facility_id']),
                    metric_type=str(row['metric_type']),
                    value=float(row['value']),
                    unit=str(row['unit']),
                    measurement_date=pd.to_datetime(row['measurement_date']).date(),
                    period_type=str(row['period_type']),
                    source_system=str(row.get('source_system', 'file_import')),
                    data_quality=str(row.get('data_quality', 'good'))
                )
                metrics.append(metric)
            except Exception as e:
                logger.warning(f"Error parsing row {len(metrics)}: {e}")
        
        logger.info(f"Loaded {len(metrics)} metrics from file")
        return metrics
    
    def generate_loading_report(self, report: LoadingReport, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive loading report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NEO4J HISTORICAL METRICS LOADING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Database: {self.uri}/{self.database}")
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("LOADING SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Start Time: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"End Time: {report.end_time.strftime('%Y-%m-%d %H:%M:%S') if report.end_time else 'N/A'}")
        report_lines.append(f"Duration: {report.duration_seconds:.2f} seconds")
        report_lines.append(f"Total Records: {report.total_records:,}")
        report_lines.append(f"Successful Loads: {report.successful_loads:,}")
        report_lines.append(f"Failed Loads: {report.failed_loads:,}")
        report_lines.append(f"Skipped Records: {report.skipped_records:,}")
        report_lines.append(f"Success Rate: {(report.successful_loads / report.total_records * 100):.2f}%" if report.total_records > 0 else "Success Rate: N/A")
        report_lines.append(f"Facilities Processed: {report.facilities_processed:,}")
        report_lines.append("")
        
        # Performance metrics
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Records per Second: {report.records_per_second:.2f}")
        report_lines.append(f"Batch Size: {report.batch_size:,}")
        report_lines.append(f"Total Batches: {report.total_batches:,}")
        
        if report.performance_metrics:
            report_lines.append(f"Average Query Time: {report.performance_metrics.get('avg_query_time', 0):.4f} seconds")
            report_lines.append(f"Total Queries Executed: {report.performance_metrics.get('total_queries', 0):,}")
            report_lines.append(f"Retries Attempted: {report.performance_metrics.get('retries_attempted', 0):,}")
            report_lines.append(f"Connections Created: {report.performance_metrics.get('connections_created', 0):,}")
        
        report_lines.append("")
        
        # Error summary
        if report.errors:
            report_lines.append("ERRORS ENCOUNTERED")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Errors: {len(report.errors)}")
            report_lines.append("")
            report_lines.append("Error Details (first 10):")
            for i, error in enumerate(report.errors[:10], 1):
                report_lines.append(f"  {i}. {error}")
            
            if len(report.errors) > 10:
                report_lines.append(f"  ... and {len(report.errors) - 10} more errors")
        else:
            report_lines.append("No errors encountered during loading.")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Loading report saved to: {output_file}")
        
        return report_text
    
    def get_loading_statistics(self) -> Dict[str, Any]:
        """Get current loading statistics and database state"""
        try:
            stats_query = """
            MATCH (m:HistoricalMetric)
            RETURN 
                count(m) as total_metrics,
                count(DISTINCT m.facility_id) as facilities_with_metrics,
                collect(DISTINCT m.metric_type) as metric_types,
                collect(DISTINCT m.period_type) as period_types,
                min(m.measurement_date) as earliest_date,
                max(m.measurement_date) as latest_date
            """
            
            result = self.execute_query_with_retry(stats_query)
            
            if result:
                stats = result[0]
                
                # Get counts by metric type
                type_counts_query = """
                MATCH (m:HistoricalMetric)
                RETURN m.metric_type as type, count(m) as count
                ORDER BY count DESC
                """
                
                type_result = self.execute_query_with_retry(type_counts_query)
                type_counts = {r['type']: r['count'] for r in type_result}
                
                return {
                    'total_metrics': stats['total_metrics'],
                    'facilities_with_metrics': stats['facilities_with_metrics'],
                    'metric_types': stats['metric_types'],
                    'period_types': stats['period_types'],
                    'earliest_date': stats['earliest_date'],
                    'latest_date': stats['latest_date'],
                    'counts_by_type': type_counts,
                    'generated_at': datetime.now().isoformat()
                }
            else:
                return {'error': 'No data found'}
                
        except Exception as e:
            logger.error(f"Error getting loading statistics: {e}")
            return {'error': str(e)}


def main():
    """Main execution function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load historical EHS metrics into Neo4j')
    parser.add_argument('--input-file', type=str, help='Input file (CSV, JSON, Excel)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for loading')
    parser.add_argument('--incremental', action='store_true', default=True, help='Enable incremental loading')
    parser.add_argument('--no-validate', action='store_true', help='Skip data validation')
    parser.add_argument('--no-relationships', action='store_true', help='Skip relationship creation')
    parser.add_argument('--report-file', type=str, help='Output file for loading report')
    
    args = parser.parse_args()
    
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    load_dotenv(env_path)
    
    # Get Neo4j credentials from environment
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
    database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    logger.info("Starting Neo4j Historical Metrics Loader")
    logger.info(f"Connecting to: {uri}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Incremental loading: {args.incremental}")
    logger.info(f"Data validation: {not args.no_validate}")
    logger.info(f"Create relationships: {not args.no_relationships}")
    
    # Initialize loader
    loader = Neo4jHistoricalMetricsLoader(
        uri=uri,
        username=username,
        password=password,
        database=database,
        batch_size=args.batch_size
    )
    
    try:
        # Connect to database
        if not loader.connect():
            logger.error("Failed to establish connection to Neo4j database")
            sys.exit(1)
        
        # Load data
        if args.input_file:
            # Load from file
            report = loader.load_historical_metrics(
                metrics=args.input_file,
                incremental=args.incremental,
                validate=not args.no_validate,
                create_relationships=not args.no_relationships
            )
        else:
            logger.error("No input file specified. Use --input-file option or call programmatically with data.")
            sys.exit(1)
        
        # Generate and display report
        report_filename = args.report_file or f"neo4j_loading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_text = loader.generate_loading_report(report, report_filename)
        print(report_text)
        
        # Display current statistics
        stats = loader.get_loading_statistics()
        print("\nCURRENT DATABASE STATISTICS:")
        print("-" * 40)
        for key, value in stats.items():
            if key != 'error':
                print(f"{key}: {value}")
        
        logger.info("Neo4j Historical Metrics Loading completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during loading: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # Clean up
        loader.close()


if __name__ == "__main__":
    main()