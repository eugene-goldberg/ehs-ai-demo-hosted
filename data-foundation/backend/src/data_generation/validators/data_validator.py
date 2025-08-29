"""
Data Validator for EHS AI Demo
Comprehensive validation suite for generated environmental, health, and safety data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""
    metric_name: str
    validation_type: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    message: str
    details: Optional[Dict] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ValidationConfig:
    """Configuration for data validation rules"""
    # Data quality thresholds
    max_missing_percentage: float = 5.0
    min_data_points: int = 30
    
    # Statistical thresholds
    outlier_z_threshold: float = 3.0
    correlation_threshold: float = 0.95  # For detecting potential data generation issues
    
    # Business rule ranges
    percentage_ranges: Dict[str, Tuple[float, float]] = None
    rate_ranges: Dict[str, Tuple[float, float]] = None
    count_ranges: Dict[str, Tuple[int, int]] = None
    
    # Temporal validation
    max_date_gap_days: int = 7
    future_date_tolerance_days: int = 1
    
    def __post_init__(self):
        if self.percentage_ranges is None:
            self.percentage_ranges = {
                'recycling_rate': (0.0, 100.0),
                'waste_diversion_rate': (0.0, 100.0),
                'energy_efficiency': (0.0, 100.0),
                'compliance_rate': (0.0, 100.0)
            }
        
        if self.rate_ranges is None:
            self.rate_ranges = {
                'incident_rate': (0.0, 50.0),  # per 100 employees
                'emission_rate': (0.0, 1000.0),  # tons CO2e
                'water_usage_rate': (0.0, 10000.0)  # gallons per unit
            }
        
        if self.count_ranges is None:
            self.count_ranges = {
                'employee_count': (1, 50000),
                'incident_count': (0, 1000),
                'audit_count': (0, 100)
            }


class DataValidator:
    """Comprehensive data validation suite for EHS metrics"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.validation_results: List[ValidationResult] = []
        
    def validate_dataset(self, data: pd.DataFrame, 
                        dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Comprehensive validation of entire dataset
        
        Args:
            data: DataFrame to validate
            dataset_name: Name identifier for the dataset
            
        Returns:
            Dictionary containing validation summary and detailed results
        """
        logger.info(f"Starting validation for dataset: {dataset_name}")
        self.validation_results.clear()
        
        # Core data integrity checks
        self._validate_data_integrity(data, dataset_name)
        
        # Data quality checks
        self._validate_data_quality(data, dataset_name)
        
        # Business rule validation
        self._validate_business_rules(data, dataset_name)
        
        # Temporal consistency checks
        self._validate_temporal_consistency(data, dataset_name)
        
        # Statistical validation
        self._validate_statistical_properties(data, dataset_name)
        
        # Relationship validation
        self._validate_relationships(data, dataset_name)
        
        # Generate summary report
        return self._generate_validation_report(dataset_name)
    
    def _validate_data_integrity(self, data: pd.DataFrame, dataset_name: str):
        """Validate basic data integrity"""
        
        # Check if dataset is empty
        if data.empty:
            self.validation_results.append(ValidationResult(
                metric_name="dataset_size",
                validation_type="integrity",
                status="FAIL",
                message=f"Dataset {dataset_name} is empty"
            ))
            return
        
        # Check minimum data points
        if len(data) < self.config.min_data_points:
            self.validation_results.append(ValidationResult(
                metric_name="dataset_size",
                validation_type="integrity",
                status="WARNING",
                message=f"Dataset has only {len(data)} rows, minimum recommended: {self.config.min_data_points}",
                details={"actual_count": len(data), "minimum_required": self.config.min_data_points}
            ))
        else:
            self.validation_results.append(ValidationResult(
                metric_name="dataset_size",
                validation_type="integrity",
                status="PASS",
                message=f"Dataset size validation passed with {len(data)} rows"
            ))
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            self.validation_results.append(ValidationResult(
                metric_name="duplicate_rows",
                validation_type="integrity",
                status="WARNING",
                message=f"Found {duplicate_count} duplicate rows",
                details={"duplicate_count": duplicate_count, "total_rows": len(data)}
            ))
        else:
            self.validation_results.append(ValidationResult(
                metric_name="duplicate_rows",
                validation_type="integrity",
                status="PASS",
                message="No duplicate rows found"
            ))
    
    def _validate_data_quality(self, data: pd.DataFrame, dataset_name: str):
        """Validate data quality metrics"""
        
        for column in data.columns:
            # Skip non-numeric columns for certain checks
            if data[column].dtype in ['object', 'string']:
                continue
                
            # Check for missing values
            missing_count = data[column].isnull().sum()
            missing_percentage = (missing_count / len(data)) * 100
            
            if missing_percentage > self.config.max_missing_percentage:
                self.validation_results.append(ValidationResult(
                    metric_name=f"{column}_missing_values",
                    validation_type="data_quality",
                    status="FAIL",
                    message=f"Column {column} has {missing_percentage:.1f}% missing values (threshold: {self.config.max_missing_percentage}%)",
                    details={"missing_count": missing_count, "missing_percentage": missing_percentage}
                ))
            elif missing_count > 0:
                self.validation_results.append(ValidationResult(
                    metric_name=f"{column}_missing_values",
                    validation_type="data_quality",
                    status="WARNING",
                    message=f"Column {column} has {missing_count} missing values ({missing_percentage:.1f}%)",
                    details={"missing_count": missing_count, "missing_percentage": missing_percentage}
                ))
            else:
                self.validation_results.append(ValidationResult(
                    metric_name=f"{column}_missing_values",
                    validation_type="data_quality",
                    status="PASS",
                    message=f"Column {column} has no missing values"
                ))
            
            # Check for infinite values
            if np.isfinite(data[column]).sum() < len(data[column].dropna()):
                inf_count = len(data[column]) - np.isfinite(data[column]).sum()
                self.validation_results.append(ValidationResult(
                    metric_name=f"{column}_infinite_values",
                    validation_type="data_quality",
                    status="FAIL",
                    message=f"Column {column} contains {inf_count} infinite values",
                    details={"infinite_count": inf_count}
                ))
    
    def _validate_business_rules(self, data: pd.DataFrame, dataset_name: str):
        """Validate business-specific rules and constraints"""
        
        for column in data.columns:
            if data[column].dtype not in ['int64', 'float64']:
                continue
            
            column_lower = column.lower()
            
            # Validate percentage ranges
            for metric_name, (min_val, max_val) in self.config.percentage_ranges.items():
                if metric_name in column_lower:
                    self._validate_range(data[column], column, min_val, max_val, "percentage")
            
            # Validate rate ranges
            for metric_name, (min_val, max_val) in self.config.rate_ranges.items():
                if metric_name in column_lower:
                    self._validate_range(data[column], column, min_val, max_val, "rate")
            
            # Validate count ranges
            for metric_name, (min_val, max_val) in self.config.count_ranges.items():
                if metric_name in column_lower:
                    self._validate_range(data[column], column, min_val, max_val, "count")
            
            # Negative values check for metrics that should be non-negative
            negative_metrics = ['count', 'rate', 'percentage', 'amount', 'volume', 'weight']
            if any(neg_metric in column_lower for neg_metric in negative_metrics):
                negative_count = (data[column] < 0).sum()
                if negative_count > 0:
                    self.validation_results.append(ValidationResult(
                        metric_name=f"{column}_negative_values",
                        validation_type="business_rules",
                        status="FAIL",
                        message=f"Column {column} has {negative_count} negative values, but should be non-negative",
                        details={"negative_count": negative_count}
                    ))
    
    def _validate_range(self, series: pd.Series, column_name: str, 
                       min_val: float, max_val: float, range_type: str):
        """Helper method to validate value ranges"""
        
        out_of_range_count = ((series < min_val) | (series > max_val)).sum()
        
        if out_of_range_count > 0:
            self.validation_results.append(ValidationResult(
                metric_name=f"{column_name}_range_validation",
                validation_type="business_rules",
                status="FAIL",
                message=f"Column {column_name} has {out_of_range_count} values outside expected {range_type} range [{min_val}, {max_val}]",
                details={
                    "out_of_range_count": out_of_range_count,
                    "expected_range": [min_val, max_val],
                    "actual_range": [series.min(), series.max()]
                }
            ))
        else:
            self.validation_results.append(ValidationResult(
                metric_name=f"{column_name}_range_validation",
                validation_type="business_rules",
                status="PASS",
                message=f"Column {column_name} values within expected {range_type} range"
            ))
    
    def _validate_temporal_consistency(self, data: pd.DataFrame, dataset_name: str):
        """Validate temporal aspects of the data"""
        
        date_columns = []
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
        
        for date_col in date_columns:
            try:
                # Convert to datetime if not already
                if data[date_col].dtype == 'object':
                    date_series = pd.to_datetime(data[date_col])
                else:
                    date_series = data[date_col]
                
                # Check for future dates (beyond tolerance)
                today = datetime.now()
                future_tolerance = today + timedelta(days=self.config.future_date_tolerance_days)
                future_dates = (date_series > future_tolerance).sum()
                
                if future_dates > 0:
                    self.validation_results.append(ValidationResult(
                        metric_name=f"{date_col}_future_dates",
                        validation_type="temporal",
                        status="WARNING",
                        message=f"Column {date_col} has {future_dates} future dates beyond tolerance",
                        details={"future_date_count": future_dates}
                    ))
                
                # Check for large gaps in date sequence
                if len(date_series.dropna()) > 1:
                    date_sorted = date_series.dropna().sort_values()
                    max_gap = (date_sorted.diff().max()).days
                    
                    if max_gap > self.config.max_date_gap_days:
                        self.validation_results.append(ValidationResult(
                            metric_name=f"{date_col}_date_gaps",
                            validation_type="temporal",
                            status="WARNING",
                            message=f"Column {date_col} has maximum gap of {max_gap} days (threshold: {self.config.max_date_gap_days})",
                            details={"max_gap_days": max_gap}
                        ))
                
            except Exception as e:
                self.validation_results.append(ValidationResult(
                    metric_name=f"{date_col}_temporal_validation",
                    validation_type="temporal",
                    status="FAIL",
                    message=f"Failed to validate temporal consistency for {date_col}: {str(e)}"
                ))
    
    def _validate_statistical_properties(self, data: pd.DataFrame, dataset_name: str):
        """Validate statistical properties of the data"""
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = data[column].dropna()
            
            if len(series) < 3:
                continue
            
            # Outlier detection using Z-score
            z_scores = np.abs(stats.zscore(series))
            outliers = (z_scores > self.config.outlier_z_threshold).sum()
            
            if outliers > len(series) * 0.1:  # More than 10% outliers
                self.validation_results.append(ValidationResult(
                    metric_name=f"{column}_outliers",
                    validation_type="statistical",
                    status="WARNING",
                    message=f"Column {column} has {outliers} potential outliers ({outliers/len(series)*100:.1f}%)",
                    details={"outlier_count": outliers, "outlier_percentage": outliers/len(series)*100}
                ))
            
            # Check for constant values (no variation)
            if series.std() == 0:
                self.validation_results.append(ValidationResult(
                    metric_name=f"{column}_variance",
                    validation_type="statistical",
                    status="WARNING",
                    message=f"Column {column} has zero variance (all values are identical)",
                    details={"unique_values": series.nunique()}
                ))
    
    def _validate_relationships(self, data: pd.DataFrame, dataset_name: str):
        """Validate relationships and correlations between variables"""
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return
        
        # Calculate correlation matrix
        try:
            corr_matrix = data[numeric_columns].corr()
            
            # Check for suspiciously high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > self.config.correlation_threshold:
                        high_corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if high_corr_pairs:
                self.validation_results.append(ValidationResult(
                    metric_name="high_correlations",
                    validation_type="relationships",
                    status="WARNING",
                    message=f"Found {len(high_corr_pairs)} variable pairs with suspiciously high correlation (>{self.config.correlation_threshold})",
                    details={"high_correlation_pairs": high_corr_pairs}
                ))
            
        except Exception as e:
            self.validation_results.append(ValidationResult(
                metric_name="correlation_analysis",
                validation_type="relationships",
                status="WARNING",
                message=f"Could not perform correlation analysis: {str(e)}"
            ))
    
    def _generate_validation_report(self, dataset_name: str) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        # Summary statistics
        total_validations = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.status == "PASS")
        warnings = sum(1 for r in self.validation_results if r.status == "WARNING")
        failed = sum(1 for r in self.validation_results if r.status == "FAIL")
        
        # Categorize results by validation type
        by_type = {}
        for result in self.validation_results:
            if result.validation_type not in by_type:
                by_type[result.validation_type] = {"PASS": 0, "WARNING": 0, "FAIL": 0}
            by_type[result.validation_type][result.status] += 1
        
        # Overall status
        overall_status = "PASS"
        if failed > 0:
            overall_status = "FAIL"
        elif warnings > 0:
            overall_status = "WARNING"
        
        report = {
            "dataset_name": dataset_name,
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_validations": total_validations,
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "success_rate": (passed / total_validations * 100) if total_validations > 0 else 0
            },
            "by_validation_type": by_type,
            "detailed_results": [
                {
                    "metric": r.metric_name,
                    "type": r.validation_type,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.validation_results
            ]
        }
        
        return report
    
    def save_validation_report(self, report: Dict[str, Any], 
                             output_path: Union[str, Path]):
        """Save validation report to JSON file"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {output_path}")
    
    def validate_metric_specific_rules(self, data: pd.DataFrame, 
                                     metric_type: str) -> List[ValidationResult]:
        """
        Apply metric-specific validation rules
        
        Args:
            data: DataFrame containing the metric data
            metric_type: Type of metric (e.g., 'environmental', 'safety', 'health')
            
        Returns:
            List of validation results specific to the metric type
        """
        
        metric_results = []
        
        if metric_type.lower() == 'environmental':
            metric_results.extend(self._validate_environmental_metrics(data))
        elif metric_type.lower() == 'safety':
            metric_results.extend(self._validate_safety_metrics(data))
        elif metric_type.lower() == 'health':
            metric_results.extend(self._validate_health_metrics(data))
        
        return metric_results
    
    def _validate_environmental_metrics(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate environmental-specific metrics"""
        results = []
        
        # Check for logical relationships in environmental data
        if 'total_waste' in data.columns and 'recycled_waste' in data.columns:
            invalid_recycling = (data['recycled_waste'] > data['total_waste']).sum()
            if invalid_recycling > 0:
                results.append(ValidationResult(
                    metric_name="waste_recycling_logic",
                    validation_type="environmental",
                    status="FAIL",
                    message=f"{invalid_recycling} records have recycled waste exceeding total waste"
                ))
        
        # Validate energy metrics
        energy_cols = [col for col in data.columns if 'energy' in col.lower()]
        for col in energy_cols:
            if data[col].dtype in ['int64', 'float64']:
                if (data[col] < 0).any():
                    results.append(ValidationResult(
                        metric_name=f"{col}_negative_energy",
                        validation_type="environmental",
                        status="FAIL",
                        message=f"Energy metric {col} contains negative values"
                    ))
        
        return results
    
    def _validate_safety_metrics(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate safety-specific metrics"""
        results = []
        
        # Validate incident rates
        if 'incident_count' in data.columns and 'employee_count' in data.columns:
            # Check for incidents without employees
            no_employees_with_incidents = ((data['employee_count'] == 0) & 
                                         (data['incident_count'] > 0)).sum()
            if no_employees_with_incidents > 0:
                results.append(ValidationResult(
                    metric_name="incidents_without_employees",
                    validation_type="safety",
                    status="FAIL",
                    message=f"{no_employees_with_incidents} records show incidents with zero employees"
                ))
        
        return results
    
    def _validate_health_metrics(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate health-specific metrics"""
        results = []
        
        # Add health-specific validation logic here
        # Example: validate air quality indices, exposure limits, etc.
        
        return results


def main():
    """Example usage of the DataValidator"""
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'recycling_rate': np.random.uniform(60, 95, 100),
        'energy_consumption': np.random.uniform(1000, 5000, 100),
        'incident_count': np.random.poisson(2, 100),
        'employee_count': np.random.randint(50, 200, 100),
        'water_usage': np.random.uniform(500, 2000, 100)
    })
    
    # Add some data quality issues for testing
    sample_data.loc[5, 'recycling_rate'] = np.nan  # Missing value
    sample_data.loc[10, 'recycling_rate'] = 150  # Out of range
    sample_data.loc[15, 'energy_consumption'] = -100  # Negative value
    
    # Initialize validator
    validator = DataValidator()
    
    # Run validation
    report = validator.validate_dataset(sample_data, "sample_ehs_data")
    
    # Print summary
    print(f"Validation Status: {report['overall_status']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Total Validations: {report['summary']['total_validations']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Warnings: {report['summary']['warnings']}")
    
    # Save report
    validator.save_validation_report(report, "/tmp/validation_report.json")


if __name__ == "__main__":
    main()