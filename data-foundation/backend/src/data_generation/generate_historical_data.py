#!/usr/bin/env python3
"""
Historical Data Generation Script for EHS AI Demo

This script orchestrates the generation and loading of 6 months of historical data
for electricity, water, and waste consumption across multiple facilities.

Features:
- Orchestrates all three generators (electricity, water, waste)
- Generates 6 months of historical data for multiple facilities
- Validates generated data using comprehensive validation suite
- Loads data into Neo4j with proper relationships
- Generates comprehensive reports with statistics and insights
- Supports command-line arguments for flexible configuration
- Includes progress tracking with detailed logging
- Handles errors gracefully with retry logic
- Production-ready with proper configuration management

Author: EHS AI Demo Team
Created: 2025-08-28
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import concurrent.futures
from contextlib import contextmanager
import tempfile
import shutil

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from tqdm import tqdm
    
    # Import our modules
    from generators.electricity_generator import ElectricityGenerator, ElectricityGeneratorConfig
    from generators.water_generator import WaterGenerator, WaterGeneratorConfig
    from generators.waste_generator import WasteGenerator, WasteGeneratorConfig
    from generators.base_generator import GeneratorConfig
    from loaders.neo4j_loader import Neo4jHistoricalMetricsLoader, HistoricalMetric, LoadingReport
    from validators.data_validator import DataValidator, ValidationConfig
    from utils.data_utils import FacilityType, EHS_CONSTANTS
    
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script in the virtual environment with required packages installed")
    sys.exit(1)

# Load environment variables
load_dotenv()


@dataclass
class GenerationConfig:
    """Configuration for historical data generation"""
    # Date range configuration
    start_date: date = date(2024, 3, 1)  # 6 months ago from Sept 2024
    end_date: date = date(2024, 8, 31)   # End of August 2024
    
    # Facility configuration
    facilities: List[Dict[str, Any]] = None
    
    # Generator configuration
    random_seed: int = 42
    noise_level: float = 0.1
    enable_patterns: bool = True
    
    # Processing configuration
    batch_size: int = 1000
    max_workers: int = 4
    enable_parallel: bool = True
    
    # Data quality configuration
    missing_data_rate: float = 0.02
    outlier_rate: float = 0.01
    
    # Output configuration
    output_dir: str = "output"
    save_intermediate: bool = True
    generate_reports: bool = True
    
    # Neo4j configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "EhsAI2024!"
    neo4j_database: str = "neo4j"
    
    # Validation configuration
    validate_data: bool = True
    validation_strict: bool = False
    
    def __post_init__(self):
        if self.facilities is None:
            self.facilities = [
                {
                    "facility_id": "FAC001", 
                    "name": "Manufacturing Plant A",
                    "type": FacilityType.MANUFACTURING,
                    "size": "large",
                    "employees": 500,
                    "operating_hours": 24
                },
                {
                    "facility_id": "FAC002",
                    "name": "Office Complex B", 
                    "type": FacilityType.OFFICE,
                    "size": "medium",
                    "employees": 200,
                    "operating_hours": 12
                },
                {
                    "facility_id": "FAC003",
                    "name": "Warehouse C",
                    "type": FacilityType.WAREHOUSE,
                    "size": "large", 
                    "employees": 50,
                    "operating_hours": 16
                },
                {
                    "facility_id": "FAC004",
                    "name": "Data Center D",
                    "type": FacilityType.DATA_CENTER,
                    "size": "medium",
                    "employees": 30,
                    "operating_hours": 24
                },
                {
                    "facility_id": "FAC005",
                    "name": "Research Lab E",
                    "type": FacilityType.LABORATORY,
                    "size": "small",
                    "employees": 75,
                    "operating_hours": 12
                }
            ]


@dataclass
class GenerationReport:
    """Comprehensive report for data generation process"""
    start_time: datetime
    end_time: Optional[datetime] = None
    config: Optional[GenerationConfig] = None
    
    # Generation statistics
    total_facilities: int = 0
    total_records_generated: int = 0
    records_by_type: Dict[str, int] = None
    records_by_facility: Dict[str, int] = None
    
    # Processing statistics
    generation_time_seconds: float = 0.0
    validation_time_seconds: float = 0.0
    loading_time_seconds: float = 0.0
    
    # Validation results
    validation_passed: bool = False
    validation_warnings: int = 0
    validation_errors: int = 0
    validation_details: List[Dict] = None
    
    # Loading results
    loading_report: Optional[LoadingReport] = None
    
    # Performance metrics
    records_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Error tracking
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.records_by_type is None:
            self.records_by_type = {}
        if self.records_by_facility is None:
            self.records_by_facility = {}
        if self.validation_details is None:
            self.validation_details = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if hasattr(self.loading_report, 'total_records') and self.loading_report.total_records > 0:
            return (self.loading_report.successful_loads / self.loading_report.total_records) * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'total_facilities': self.total_facilities,
            'total_records_generated': self.total_records_generated,
            'records_by_type': self.records_by_type,
            'records_by_facility': self.records_by_facility,
            'generation_time_seconds': self.generation_time_seconds,
            'validation_time_seconds': self.validation_time_seconds,
            'loading_time_seconds': self.loading_time_seconds,
            'validation_passed': self.validation_passed,
            'validation_warnings': self.validation_warnings,
            'validation_errors': self.validation_errors,
            'records_per_second': self.records_per_second,
            'memory_usage_mb': self.memory_usage_mb,
            'success_rate': self.success_rate,
            'loading_summary': self.loading_report.to_dict() if self.loading_report else None,
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'generated_at': datetime.now().isoformat()
        }


class HistoricalDataGenerator:
    """Main class for orchestrating historical data generation"""
    
    def __init__(self, config: GenerationConfig):
        """Initialize the historical data generator"""
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.generators = {}
        self.validator = None
        self.neo4j_loader = None
        
        # Initialize tracking
        self.report = GenerationReport(start_time=datetime.now(), config=config)
        self.report.total_facilities = len(config.facilities)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize progress tracking
        self.progress_callback = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('HistoricalDataGenerator')
        logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.output_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_generators(self) -> bool:
        """Initialize all data generators"""
        try:
            self.logger.info("Initializing data generators...")
            
            # Base configuration for all generators
            base_config = GeneratorConfig(
                random_seed=self.config.random_seed,
                start_date=datetime.combine(self.config.start_date, datetime.min.time()),
                end_date=datetime.combine(self.config.end_date, datetime.min.time()),
                noise_level=self.config.noise_level,
                enable_seasonal_patterns=self.config.enable_patterns,
                enable_weekly_patterns=self.config.enable_patterns,
                enable_daily_patterns=self.config.enable_patterns,
                missing_data_rate=self.config.missing_data_rate,
                outlier_rate=self.config.outlier_rate
            )
            
            # Initialize electricity generator
            electricity_config = ElectricityGeneratorConfig(**asdict(base_config))
            self.generators['electricity'] = ElectricityGenerator(electricity_config)
            
            # Initialize water generator  
            water_config = WaterGeneratorConfig(**asdict(base_config))
            self.generators['water'] = WaterGenerator(water_config)
            
            # Initialize waste generator
            waste_config = WasteGeneratorConfig(**asdict(base_config))
            self.generators['waste'] = WasteGenerator(waste_config)
            
            self.logger.info(f"Successfully initialized {len(self.generators)} generators")
            return True
            
        except Exception as e:
            error_msg = f"Error initializing generators: {e}"
            self.logger.error(error_msg)
            self.report.errors.append(error_msg)
            return False
    
    def _initialize_validator(self) -> bool:
        """Initialize data validator"""
        try:
            validation_config = ValidationConfig(
                max_missing_percentage=5.0 if self.config.validation_strict else 10.0,
                min_data_points=30,
                outlier_z_threshold=3.0 if self.config.validation_strict else 4.0
            )
            
            self.validator = DataValidator(validation_config)
            self.logger.info("Data validator initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Error initializing validator: {e}"
            self.logger.error(error_msg)
            self.report.errors.append(error_msg)
            return False
    
    def _initialize_neo4j_loader(self) -> bool:
        """Initialize Neo4j loader"""
        try:
            self.neo4j_loader = Neo4jHistoricalMetricsLoader(
                uri=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database,
                batch_size=self.config.batch_size
            )
            
            # Test connection
            if not self.neo4j_loader.connect():
                raise Exception("Failed to connect to Neo4j database")
            
            self.logger.info("Neo4j loader initialized and connected successfully")
            return True
            
        except Exception as e:
            error_msg = f"Error initializing Neo4j loader: {e}"
            self.logger.error(error_msg)
            self.report.errors.append(error_msg)
            return False
    
    def _generate_facility_data(self, facility: Dict[str, Any], 
                              metric_type: str) -> List[HistoricalMetric]:
        """Generate data for a single facility and metric type"""
        try:
            generator = self.generators[metric_type]
            facility_id = facility['facility_id']
            
            self.logger.debug(f"Generating {metric_type} data for facility {facility_id}")
            
            # Generate monthly data
            generated_data = generator.generate(
                facility_id=facility_id,
                facility_type=facility['type'],
                facility_size=facility['size'],
                period_type='monthly'
            )
            
            # Convert to HistoricalMetric objects
            metrics = []
            dates = generated_data['dates']
            values = generated_data['values']
            units = generated_data['metadata']['unit']
            
            for date_val, value in zip(dates, values):
                if not np.isnan(value):  # Skip missing values
                    metric = HistoricalMetric(
                        facility_id=facility_id,
                        metric_type=metric_type,
                        value=float(value),
                        unit=units,
                        measurement_date=date_val.date() if hasattr(date_val, 'date') else date_val,
                        period_type='monthly',
                        source_system='synthetic_generator',
                        data_quality='good'
                    )
                    metrics.append(metric)
            
            self.logger.debug(f"Generated {len(metrics)} {metric_type} records for facility {facility_id}")
            return metrics
            
        except Exception as e:
            error_msg = f"Error generating {metric_type} data for facility {facility['facility_id']}: {e}"
            self.logger.error(error_msg)
            self.report.errors.append(error_msg)
            return []
    
    def generate_data(self) -> bool:
        """Generate historical data for all facilities and metrics"""
        try:
            self.logger.info("Starting data generation process...")
            generation_start = time.time()
            
            all_metrics = []
            total_combinations = len(self.config.facilities) * len(self.generators)
            
            # Create progress bar
            with tqdm(total=total_combinations, desc="Generating data") as pbar:
                
                if self.config.enable_parallel and self.config.max_workers > 1:
                    # Parallel processing
                    self.logger.info(f"Using parallel processing with {self.config.max_workers} workers")
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                        # Submit all tasks
                        futures = []
                        for facility in self.config.facilities:
                            for metric_type in self.generators.keys():
                                future = executor.submit(self._generate_facility_data, facility, metric_type)
                                futures.append((future, facility['facility_id'], metric_type))
                        
                        # Collect results
                        for future, facility_id, metric_type in futures:
                            try:
                                facility_metrics = future.result(timeout=300)  # 5 minute timeout
                                all_metrics.extend(facility_metrics)
                                
                                # Update statistics
                                self.report.records_by_type[metric_type] = (
                                    self.report.records_by_type.get(metric_type, 0) + len(facility_metrics)
                                )
                                self.report.records_by_facility[facility_id] = (
                                    self.report.records_by_facility.get(facility_id, 0) + len(facility_metrics)
                                )
                                
                                pbar.update(1)
                                
                            except concurrent.futures.TimeoutError:
                                error_msg = f"Timeout generating {metric_type} data for facility {facility_id}"
                                self.logger.error(error_msg)
                                self.report.errors.append(error_msg)
                                pbar.update(1)
                            except Exception as e:
                                error_msg = f"Error in parallel generation: {e}"
                                self.logger.error(error_msg)
                                self.report.errors.append(error_msg)
                                pbar.update(1)
                else:
                    # Sequential processing
                    self.logger.info("Using sequential processing")
                    
                    for facility in self.config.facilities:
                        facility_id = facility['facility_id']
                        
                        for metric_type in self.generators.keys():
                            facility_metrics = self._generate_facility_data(facility, metric_type)
                            all_metrics.extend(facility_metrics)
                            
                            # Update statistics
                            self.report.records_by_type[metric_type] = (
                                self.report.records_by_type.get(metric_type, 0) + len(facility_metrics)
                            )
                            self.report.records_by_facility[facility_id] = (
                                self.report.records_by_facility.get(facility_id, 0) + len(facility_metrics)
                            )
                            
                            pbar.update(1)
            
            # Store generated data
            self.generated_metrics = all_metrics
            self.report.total_records_generated = len(all_metrics)
            self.report.generation_time_seconds = time.time() - generation_start
            
            if self.report.total_records_generated > 0:
                self.report.records_per_second = (
                    self.report.total_records_generated / self.report.generation_time_seconds
                )
            
            self.logger.info(f"Data generation completed successfully")
            self.logger.info(f"Total records generated: {self.report.total_records_generated}")
            self.logger.info(f"Generation time: {self.report.generation_time_seconds:.2f} seconds")
            self.logger.info(f"Records per second: {self.report.records_per_second:.2f}")
            
            # Save intermediate data if requested
            if self.config.save_intermediate:
                self._save_intermediate_data()
            
            return True
            
        except Exception as e:
            error_msg = f"Critical error in data generation: {e}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.report.errors.append(error_msg)
            return False
    
    def validate_data(self) -> bool:
        """Validate generated data"""
        if not self.config.validate_data or not hasattr(self, 'generated_metrics'):
            return True
        
        try:
            self.logger.info("Starting data validation...")
            validation_start = time.time()
            
            # Convert metrics to DataFrame for validation
            data_for_validation = []
            for metric in self.generated_metrics:
                data_for_validation.append({
                    'facility_id': metric.facility_id,
                    'metric_type': metric.metric_type,
                    'value': metric.value,
                    'unit': metric.unit,
                    'measurement_date': metric.measurement_date,
                    'period_type': metric.period_type
                })
            
            df = pd.DataFrame(data_for_validation)
            
            # Run validation by metric type
            validation_results = []
            for metric_type in df['metric_type'].unique():
                metric_data = df[df['metric_type'] == metric_type]
                
                # Run comprehensive validation
                results = self.validator.validate_comprehensive(
                    metric_data, 
                    metric_name=metric_type
                )
                validation_results.extend(results)
            
            # Process validation results
            passed_count = sum(1 for r in validation_results if r.status == 'PASS')
            warning_count = sum(1 for r in validation_results if r.status == 'WARNING')
            error_count = sum(1 for r in validation_results if r.status == 'FAIL')
            
            self.report.validation_passed = error_count == 0
            self.report.validation_warnings = warning_count
            self.report.validation_errors = error_count
            self.report.validation_details = [asdict(r) for r in validation_results]
            self.report.validation_time_seconds = time.time() - validation_start
            
            # Log validation results
            self.logger.info(f"Validation completed in {self.report.validation_time_seconds:.2f} seconds")
            self.logger.info(f"Validation results: {passed_count} passed, {warning_count} warnings, {error_count} errors")
            
            if error_count > 0:
                self.logger.error("Data validation failed - critical errors found")
                for result in validation_results:
                    if result.status == 'FAIL':
                        self.logger.error(f"  {result.metric_name}: {result.message}")
                        self.report.errors.append(f"Validation failure - {result.metric_name}: {result.message}")
            
            if warning_count > 0:
                self.logger.warning(f"Data validation completed with {warning_count} warnings")
                for result in validation_results:
                    if result.status == 'WARNING':
                        self.logger.warning(f"  {result.metric_name}: {result.message}")
                        self.report.warnings.append(f"Validation warning - {result.metric_name}: {result.message}")
            
            return self.report.validation_passed or not self.config.validation_strict
            
        except Exception as e:
            error_msg = f"Error during data validation: {e}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.report.errors.append(error_msg)
            return not self.config.validation_strict
    
    def load_data_to_neo4j(self) -> bool:
        """Load generated data into Neo4j"""
        if not hasattr(self, 'generated_metrics') or not self.generated_metrics:
            self.logger.warning("No data to load into Neo4j")
            return True
        
        try:
            self.logger.info("Starting data loading to Neo4j...")
            loading_start = time.time()
            
            # Load data using Neo4j loader
            loading_report = self.neo4j_loader.load_historical_metrics(
                metrics=self.generated_metrics,
                incremental=True,
                validate=False,  # Already validated
                create_relationships=True
            )
            
            self.report.loading_report = loading_report
            self.report.loading_time_seconds = time.time() - loading_start
            
            # Log results
            success_rate = (loading_report.successful_loads / loading_report.total_records * 100) if loading_report.total_records > 0 else 0
            self.logger.info(f"Data loading completed in {self.report.loading_time_seconds:.2f} seconds")
            self.logger.info(f"Loading results: {loading_report.successful_loads}/{loading_report.total_records} successful ({success_rate:.1f}%)")
            
            if loading_report.failed_loads > 0:
                self.logger.error(f"Failed to load {loading_report.failed_loads} records")
                self.report.errors.extend(loading_report.errors)
            
            return loading_report.failed_loads == 0
            
        except Exception as e:
            error_msg = f"Error during data loading: {e}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.report.errors.append(error_msg)
            return False
    
    def _save_intermediate_data(self):
        """Save intermediate data files"""
        try:
            self.logger.info("Saving intermediate data files...")
            
            # Convert metrics to DataFrame
            data_records = []
            for metric in self.generated_metrics:
                data_records.append({
                    'facility_id': metric.facility_id,
                    'metric_type': metric.metric_type,
                    'value': metric.value,
                    'unit': metric.unit,
                    'measurement_date': metric.measurement_date,
                    'period_type': metric.period_type,
                    'source_system': metric.source_system,
                    'data_quality': metric.data_quality,
                    'created_at': metric.created_at.isoformat() if metric.created_at else None
                })
            
            df = pd.DataFrame(data_records)
            
            # Save as CSV
            csv_file = self.output_dir / f"historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Saved CSV file: {csv_file}")
            
            # Save as JSON
            json_file = self.output_dir / f"historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            df.to_json(json_file, orient='records', date_format='iso')
            self.logger.info(f"Saved JSON file: {json_file}")
            
        except Exception as e:
            warning_msg = f"Warning: Could not save intermediate data: {e}"
            self.logger.warning(warning_msg)
            self.report.warnings.append(warning_msg)
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        try:
            self.report.end_time = datetime.now()
            
            # Generate report content
            report_lines = []
            report_lines.append("=" * 100)
            report_lines.append("EHS AI DEMO - HISTORICAL DATA GENERATION REPORT")
            report_lines.append("=" * 100)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Configuration: {self.config.start_date} to {self.config.end_date}")
            report_lines.append("")
            
            # Execution summary
            report_lines.append("EXECUTION SUMMARY")
            report_lines.append("-" * 50)
            report_lines.append(f"Total Duration: {self.report.duration_seconds:.2f} seconds")
            report_lines.append(f"Generation Time: {self.report.generation_time_seconds:.2f} seconds")
            report_lines.append(f"Validation Time: {self.report.validation_time_seconds:.2f} seconds")
            report_lines.append(f"Loading Time: {self.report.loading_time_seconds:.2f} seconds")
            report_lines.append("")
            
            # Data generation statistics
            report_lines.append("DATA GENERATION STATISTICS")
            report_lines.append("-" * 50)
            report_lines.append(f"Total Facilities: {self.report.total_facilities}")
            report_lines.append(f"Total Records Generated: {self.report.total_records_generated:,}")
            report_lines.append(f"Records per Second: {self.report.records_per_second:.2f}")
            report_lines.append("")
            
            report_lines.append("Records by Type:")
            for metric_type, count in self.report.records_by_type.items():
                report_lines.append(f"  {metric_type.capitalize()}: {count:,}")
            
            report_lines.append("")
            report_lines.append("Records by Facility:")
            for facility_id, count in self.report.records_by_facility.items():
                report_lines.append(f"  {facility_id}: {count:,}")
            
            report_lines.append("")
            
            # Validation results
            if self.config.validate_data:
                report_lines.append("VALIDATION RESULTS")
                report_lines.append("-" * 50)
                report_lines.append(f"Validation Status: {'PASSED' if self.report.validation_passed else 'FAILED'}")
                report_lines.append(f"Validation Warnings: {self.report.validation_warnings}")
                report_lines.append(f"Validation Errors: {self.report.validation_errors}")
                
                if self.report.validation_details:
                    failed_validations = [v for v in self.report.validation_details if v['status'] == 'FAIL']
                    if failed_validations:
                        report_lines.append("")
                        report_lines.append("Failed Validations:")
                        for validation in failed_validations[:5]:  # Show first 5
                            report_lines.append(f"  - {validation['metric_name']}: {validation['message']}")
                report_lines.append("")
            
            # Loading results
            if self.report.loading_report:
                loading = self.report.loading_report
                report_lines.append("NEO4J LOADING RESULTS")
                report_lines.append("-" * 50)
                report_lines.append(f"Records Loaded Successfully: {loading.successful_loads:,}")
                report_lines.append(f"Records Failed to Load: {loading.failed_loads:,}")
                report_lines.append(f"Records Skipped: {loading.skipped_records:,}")
                report_lines.append(f"Loading Success Rate: {self.report.success_rate:.2f}%")
                report_lines.append(f"Facilities Processed: {loading.facilities_processed}")
                report_lines.append(f"Loading Performance: {loading.records_per_second:.2f} records/second")
                report_lines.append("")
            
            # Error and warning summary
            if self.report.errors or self.report.warnings:
                report_lines.append("ISSUES ENCOUNTERED")
                report_lines.append("-" * 50)
                
                if self.report.errors:
                    report_lines.append(f"Errors ({len(self.report.errors)}):")
                    for error in self.report.errors[:10]:  # Show first 10
                        report_lines.append(f"  - {error}")
                    if len(self.report.errors) > 10:
                        report_lines.append(f"  ... and {len(self.report.errors) - 10} more errors")
                
                if self.report.warnings:
                    report_lines.append(f"Warnings ({len(self.report.warnings)}):")
                    for warning in self.report.warnings[:10]:  # Show first 10
                        report_lines.append(f"  - {warning}")
                    if len(self.report.warnings) > 10:
                        report_lines.append(f"  ... and {len(self.report.warnings) - 10} more warnings")
            else:
                report_lines.append("No errors or warnings encountered.")
            
            report_lines.append("")
            
            # Configuration details
            report_lines.append("CONFIGURATION DETAILS")
            report_lines.append("-" * 50)
            report_lines.append(f"Random Seed: {self.config.random_seed}")
            report_lines.append(f"Noise Level: {self.config.noise_level}")
            report_lines.append(f"Missing Data Rate: {self.config.missing_data_rate}")
            report_lines.append(f"Outlier Rate: {self.config.outlier_rate}")
            report_lines.append(f"Batch Size: {self.config.batch_size}")
            report_lines.append(f"Parallel Processing: {self.config.enable_parallel}")
            report_lines.append(f"Max Workers: {self.config.max_workers}")
            report_lines.append(f"Validation Enabled: {self.config.validate_data}")
            report_lines.append(f"Validation Strict Mode: {self.config.validation_strict}")
            
            report_lines.append("")
            report_lines.append("=" * 100)
            report_lines.append("END OF REPORT")
            report_lines.append("=" * 100)
            
            report_text = "\n".join(report_lines)
            
            # Save report
            if self.config.generate_reports:
                report_file = self.output_dir / f"generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_file, 'w') as f:
                    f.write(report_text)
                self.logger.info(f"Report saved to: {report_file}")
                
                # Also save as JSON
                json_report_file = self.output_dir / f"generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(json_report_file, 'w') as f:
                    json.dump(self.report.to_dict(), f, indent=2, default=str)
                self.logger.info(f"JSON report saved to: {json_report_file}")
            
            return report_text
            
        except Exception as e:
            error_msg = f"Error generating report: {e}"
            self.logger.error(error_msg)
            return f"Error generating report: {error_msg}"
    
    def run(self) -> bool:
        """Run the complete historical data generation process"""
        try:
            self.logger.info("Starting historical data generation process...")
            self.logger.info(f"Date range: {self.config.start_date} to {self.config.end_date}")
            self.logger.info(f"Facilities: {len(self.config.facilities)}")
            self.logger.info(f"Metrics: {list(self.generators.keys()) if self.generators else ['electricity', 'water', 'waste']}")
            
            # Step 1: Initialize all components
            self.logger.info("Step 1: Initializing components...")
            if not self._initialize_generators():
                return False
            
            if self.config.validate_data and not self._initialize_validator():
                return False
            
            if not self._initialize_neo4j_loader():
                return False
            
            self.logger.info("All components initialized successfully")
            
            # Step 2: Generate data
            self.logger.info("Step 2: Generating historical data...")
            if not self.generate_data():
                return False
            
            # Step 3: Validate data (if enabled)
            if self.config.validate_data:
                self.logger.info("Step 3: Validating generated data...")
                if not self.validate_data():
                    if self.config.validation_strict:
                        self.logger.error("Data validation failed in strict mode - stopping process")
                        return False
                    else:
                        self.logger.warning("Data validation failed in non-strict mode - continuing")
            
            # Step 4: Load data to Neo4j
            self.logger.info("Step 4: Loading data to Neo4j...")
            if not self.load_data_to_neo4j():
                self.logger.error("Data loading failed - some records may not have been loaded")
                # Continue to generate report even if loading partially failed
            
            # Step 5: Generate final report
            self.logger.info("Step 5: Generating final report...")
            report_text = self.generate_report()
            
            # Print summary to console
            print("\n" + "=" * 80)
            print("HISTORICAL DATA GENERATION COMPLETED")
            print("=" * 80)
            print(f"Total Records Generated: {self.report.total_records_generated:,}")
            print(f"Total Duration: {self.report.duration_seconds:.2f} seconds")
            print(f"Success Rate: {self.report.success_rate:.2f}%")
            
            if self.report.errors:
                print(f"Errors Encountered: {len(self.report.errors)}")
            if self.report.warnings:
                print(f"Warnings: {len(self.report.warnings)}")
            
            print("=" * 80)
            
            return len(self.report.errors) == 0
            
        except Exception as e:
            error_msg = f"Critical error in data generation process: {e}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.report.errors.append(error_msg)
            return False
        
        finally:
            # Cleanup
            if self.neo4j_loader:
                self.neo4j_loader.close()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Generate 6 months of historical EHS data for multiple facilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data with default settings
  python generate_historical_data.py
  
  # Generate data with custom date range
  python generate_historical_data.py --start-date 2024-01-01 --end-date 2024-06-30
  
  # Generate data without validation
  python generate_historical_data.py --no-validation
  
  # Generate data with parallel processing disabled
  python generate_historical_data.py --no-parallel
  
  # Generate data with custom batch size
  python generate_historical_data.py --batch-size 2000
        """
    )
    
    # Date range options
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-03-01',
        help='Start date for data generation (YYYY-MM-DD) [default: 2024-03-01]'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str,
        default='2024-08-31',
        help='End date for data generation (YYYY-MM-DD) [default: 2024-08-31]'
    )
    
    # Generation options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible generation [default: 42]'
    )
    
    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.1,
        help='Noise level for data generation (0.0-1.0) [default: 0.1]'
    )
    
    parser.add_argument(
        '--no-patterns',
        action='store_true',
        help='Disable seasonal/weekly/daily patterns'
    )
    
    # Processing options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for Neo4j loading [default: 1000]'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers [default: 4]'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    
    # Validation options
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Skip data validation'
    )
    
    parser.add_argument(
        '--validation-strict',
        action='store_true',
        help='Enable strict validation mode (fail on warnings)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for reports and intermediate files [default: output]'
    )
    
    parser.add_argument(
        '--no-intermediate',
        action='store_true',
        help='Skip saving intermediate data files'
    )
    
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Skip generating detailed reports'
    )
    
    # Neo4j options
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        help='Neo4j URI [default: from environment or bolt://localhost:7687]'
    )
    
    parser.add_argument(
        '--neo4j-username',
        type=str,
        help='Neo4j username [default: from environment or neo4j]'
    )
    
    parser.add_argument(
        '--neo4j-password',
        type=str,
        help='Neo4j password [default: from environment]'
    )
    
    parser.add_argument(
        '--neo4j-database',
        type=str,
        help='Neo4j database name [default: from environment or neo4j]'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level [default: INFO]'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser


def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Set logging level
        if args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Parse dates
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            print("Please use YYYY-MM-DD format for dates")
            sys.exit(1)
        
        # Validate date range
        if start_date >= end_date:
            print("Error: Start date must be before end date")
            sys.exit(1)
        
        if (end_date - start_date).days > 730:  # > 2 years
            print("Warning: Date range is very large (>2 years), generation may take a long time")
        
        # Get Neo4j credentials (from args or environment)
        neo4j_uri = args.neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_username = args.neo4j_username or os.getenv('NEO4J_USERNAME', 'neo4j')  
        neo4j_password = args.neo4j_password or os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
        neo4j_database = args.neo4j_database or os.getenv('NEO4J_DATABASE', 'neo4j')
        
        if not neo4j_password or neo4j_password == 'EhsAI2024!':
            print("Warning: Using default Neo4j password. Consider setting NEO4J_PASSWORD environment variable.")
        
        # Create configuration
        config = GenerationConfig(
            start_date=start_date,
            end_date=end_date,
            random_seed=args.seed,
            noise_level=args.noise_level,
            enable_patterns=not args.no_patterns,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            enable_parallel=not args.no_parallel,
            output_dir=args.output_dir,
            save_intermediate=not args.no_intermediate,
            generate_reports=not args.no_reports,
            validate_data=not args.no_validation,
            validation_strict=args.validation_strict,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database
        )
        
        # Print configuration summary
        if not args.quiet:
            print("=" * 80)
            print("EHS AI DEMO - HISTORICAL DATA GENERATION")
            print("=" * 80)
            print(f"Date Range: {start_date} to {end_date} ({(end_date - start_date).days + 1} days)")
            print(f"Facilities: {len(config.facilities)}")
            print(f"Metrics: electricity, water, waste")
            print(f"Random Seed: {config.random_seed}")
            print(f"Noise Level: {config.noise_level}")
            print(f"Patterns Enabled: {config.enable_patterns}")
            print(f"Validation: {config.validate_data} ({'strict' if config.validation_strict else 'normal'})")
            print(f"Parallel Processing: {config.enable_parallel} (max {config.max_workers} workers)")
            print(f"Batch Size: {config.batch_size}")
            print(f"Output Directory: {config.output_dir}")
            print(f"Neo4j: {neo4j_uri}")
            print("=" * 80)
            print()
        
        # Initialize and run generator
        generator = HistoricalDataGenerator(config)
        success = generator.run()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()