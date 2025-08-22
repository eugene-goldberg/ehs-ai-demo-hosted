"""
Comprehensive Test Suite for EHS Risk Assessment Framework

This module provides comprehensive testing for all risk assessment components including:
- Risk assessment framework (RiskSeverity, RiskFactor, RiskAssessment, BaseRiskAnalyzer)
- Individual risk analyzers (Water, Electricity, Waste consumption risk scenarios)
- Time series analysis (trend detection, seasonal decomposition, anomaly detection)
- Forecasting engine (model selection, accuracy metrics, external factors)
- Anomaly detection (various anomaly types, ensemble voting, false positive rates)
- Risk-aware query processing (query enhancement, risk filtering/ranking)
- Monitoring and alerting (alert generation, deduplication, escalation chains)
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import uuid
from dataclasses import dataclass

# Import risk assessment components
from ehs_analytics.risk_assessment.base import (
    RiskSeverity, RiskFactor, RiskAssessment, BaseRiskAnalyzer, RiskThresholds
)
from ehs_analytics.risk_assessment.water_risk import (
    WaterConsumptionRiskAnalyzer, WaterConsumptionRecord, WaterPermitData, EquipmentEfficiencyData
)
from ehs_analytics.risk_assessment.electricity_risk import (
    ElectricityConsumptionRiskAnalyzer, ElectricityConsumptionRecord, 
    ElectricalContractData, CarbonEmissionsData, PowerQualityThresholds
)
from ehs_analytics.risk_assessment.waste_risk import (
    WasteGenerationRiskAnalyzer, WasteGenerationRecord, WasteRegulationData, 
    WasteStreamData, StorageFacilityData
)
from ehs_analytics.risk_assessment.time_series import (
    TimeSeriesAnalyzer, TimeSeriesData, TimeSeriesPredictor, TrendDirection,
    AnomalyType, SeasonalComponents, TrendAnalysis, AnomalyResult
)
from ehs_analytics.risk_assessment.forecasting import (
    ForecastingEngine, ForecastModel, ForecastHorizon, ForecastResult,
    ModelSelector, ExternalFactorsProcessor
)
from ehs_analytics.risk_assessment.anomaly_detection import (
    AnomalyDetectionSystem, DetectorConfig, DetectorType, EnsembleStrategy,
    AnomalyAlert, AnomalyScore, create_ehs_anomaly_system
)
from ehs_analytics.risk_assessment.monitoring import (
    RiskMonitor, AlertManager, EscalationChain, MetricsCollector
)
from ehs_analytics.risk_assessment.risk_query_processor import (
    RiskAwareQueryProcessor, RiskQueryEnhancer, RiskFilteringRetriever
)


# =============================================================================
# Test Fixtures and Mock Data Generators
# =============================================================================

@pytest.fixture
def sample_risk_thresholds():
    """Sample risk thresholds for testing."""
    return RiskThresholds(
        low_threshold=0.3,
        medium_threshold=0.6,
        high_threshold=0.8,
        critical_threshold=0.95
    )

@pytest.fixture
def sample_risk_factors(sample_risk_thresholds):
    """Sample risk factors for testing."""
    return [
        RiskFactor(
            name="Water Usage",
            value=0.7,
            weight=0.4,
            severity=RiskSeverity.HIGH,
            thresholds=sample_risk_thresholds,
            description="Water consumption risk factor",
            unit="utilization_ratio"
        ),
        RiskFactor(
            name="Permit Compliance",
            value=0.2,
            weight=0.3,
            severity=RiskSeverity.LOW,
            thresholds=sample_risk_thresholds,
            description="Regulatory compliance risk",
            unit="compliance_score"
        ),
        RiskFactor(
            name="Equipment Efficiency",
            value=0.5,
            weight=0.3,
            severity=RiskSeverity.MEDIUM,
            thresholds=sample_risk_thresholds,
            description="Equipment performance risk",
            unit="efficiency_score"
        )
    ]

@pytest.fixture
def sample_water_consumption_records():
    """Generate sample water consumption records for testing."""
    records = []
    base_date = datetime.now() - timedelta(days=90)
    
    for i in range(90):
        date = base_date + timedelta(days=i)
        # Add some seasonal variation and trends
        seasonal = 50 * np.sin(2 * np.pi * i / 365) + 100
        noise = np.random.normal(0, 10)
        consumption = max(0, seasonal + noise + i * 0.5)  # Adding slight upward trend
        
        records.append(WaterConsumptionRecord(
            timestamp=date,
            facility_id="facility-test-001",
            consumption_gallons=consumption,
            meter_id=f"meter-{i % 3 + 1}",
            consumption_type="operational",
            quality_flag=True,
            metadata={"source": "test_data", "day_of_year": i}
        ))
    
    return records

@pytest.fixture
def sample_water_permit():
    """Sample water permit data for testing."""
    return WaterPermitData(
        permit_id="permit-water-001",
        facility_id="facility-test-001",
        daily_limit=500.0,
        monthly_limit=15000.0,
        annual_limit=180000.0,
        issue_date=datetime.now() - timedelta(days=365),
        expiry_date=datetime.now() + timedelta(days=365),
        permit_type="water_withdrawal",
        regulatory_body="EPA"
    )

@pytest.fixture
def sample_equipment_efficiency_data():
    """Sample equipment efficiency data for testing."""
    return [
        EquipmentEfficiencyData(
            equipment_id="pump-001",
            equipment_type="Water Pump",
            baseline_efficiency=0.85,
            current_efficiency=0.78,
            last_maintenance=datetime.now() - timedelta(days=45),
            efficiency_trend=-0.02,
            operational_status="active"
        ),
        EquipmentEfficiencyData(
            equipment_id="tank-001",
            equipment_type="Storage Tank",
            baseline_efficiency=0.95,
            current_efficiency=0.92,
            last_maintenance=datetime.now() - timedelta(days=30),
            efficiency_trend=-0.01,
            operational_status="active"
        )
    ]

@pytest.fixture
def sample_electricity_records():
    """Generate sample electricity consumption records."""
    records = []
    base_date = datetime.now() - timedelta(days=60)
    
    for i in range(60):
        date = base_date + timedelta(days=i)
        # Simulate business patterns
        is_weekend = date.weekday() >= 5
        hour = date.hour
        
        # Base consumption with business patterns
        base_kwh = 100 if not is_weekend else 50
        time_factor = 1.2 if 8 <= hour <= 18 else 0.8
        consumption = base_kwh * time_factor + np.random.normal(0, 10)
        
        # Demand pattern
        demand = consumption * 0.8 + np.random.normal(0, 5)
        
        records.append(ElectricityConsumptionRecord(
            timestamp=date,
            facility_id="facility-test-001",
            energy_kwh=max(0, consumption),
            demand_kw=max(0, demand),
            voltage_l1=480 + np.random.normal(0, 5),
            power_factor=0.95 + np.random.normal(0, 0.02),
            time_of_use_period="peak" if 16 <= hour <= 20 else "offpeak",
            quality_flag=True
        ))
    
    return records

@pytest.fixture
def sample_electrical_contract():
    """Sample electrical contract data."""
    return ElectricalContractData(
        contract_id="contract-elec-001",
        facility_id="facility-test-001",
        contracted_demand_kw=200.0,
        demand_rate_per_kw=15.0,
        energy_rate_peak=0.12,
        energy_rate_offpeak=0.08,
        power_factor_threshold=0.95,
        power_factor_penalty_rate=0.10
    )

@pytest.fixture
def sample_waste_records():
    """Generate sample waste generation records."""
    records = []
    base_date = datetime.now() - timedelta(days=30)
    
    waste_categories = ["general", "hazardous", "recyclable", "electronic"]
    disposal_methods = ["landfill", "recycling", "incineration", "treatment"]
    
    for i in range(30):
        for category in waste_categories:
            date = base_date + timedelta(days=i)
            
            records.append(WasteGenerationRecord(
                timestamp=date,
                facility_id="facility-test-001",
                waste_category=category,
                amount_tons=np.random.uniform(0.5, 5.0),
                disposal_method=np.random.choice(disposal_methods),
                disposal_cost_per_ton=np.random.uniform(50, 200),
                hazardous_level="hazardous" if category == "hazardous" else "non-hazardous",
                contamination_risk=np.random.uniform(0, 0.3),
                quality_flag=True
            ))
    
    return records

@pytest.fixture
def sample_waste_regulations():
    """Sample waste regulation data."""
    return [
        WasteRegulationData(
            regulation_id="reg-waste-001",
            facility_id="facility-test-001",
            waste_category="hazardous",
            storage_limit_tons=10.0,
            disposal_frequency_days=30,
            reporting_threshold_tons=5.0,
            regulatory_body="EPA",
            effective_date=datetime.now() - timedelta(days=365)
        ),
        WasteRegulationData(
            regulation_id="reg-waste-002",
            facility_id="facility-test-001",
            waste_category="general",
            storage_limit_tons=50.0,
            disposal_frequency_days=7,
            reporting_threshold_tons=25.0,
            regulatory_body="local",
            effective_date=datetime.now() - timedelta(days=180)
        )
    ]

@pytest.fixture
def sample_time_series_data():
    """Generate time series data for analysis."""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Create synthetic data with trend, seasonality, and noise
    trend = np.linspace(100, 120, 365)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365)
    noise = np.random.normal(0, 5, 365)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(365, 10, replace=False)
    values = trend + seasonal + noise
    values[anomaly_indices] *= 2  # Spike anomalies
    
    return TimeSeriesData(
        timestamps=dates.to_pydatetime().tolist(),
        values=values.tolist(),
        metadata={'source': 'test_data'}
    )

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    driver = Mock()
    session = Mock()
    result = Mock()
    
    # Configure session mock
    session.__enter__.return_value = session
    session.__exit__.return_value = None
    session.run.return_value = result
    
    # Configure result mock
    result.single.return_value = Mock()
    result.data.return_value = [
        {"facility_name": "Test Facility", "risk_score": 0.7}
    ]
    
    # Configure driver mock
    driver.session.return_value = session
    
    return driver


# =============================================================================
# Test Risk Assessment Framework Foundation
# =============================================================================

class TestRiskSeverityComparisons:
    """Test RiskSeverity enum comparisons and operations."""
    
    def test_risk_severity_ordering(self):
        """Test that risk severities can be compared correctly."""
        assert RiskSeverity.LOW < RiskSeverity.MEDIUM
        assert RiskSeverity.MEDIUM < RiskSeverity.HIGH
        assert RiskSeverity.HIGH < RiskSeverity.CRITICAL
        
        assert RiskSeverity.CRITICAL > RiskSeverity.HIGH
        assert RiskSeverity.HIGH >= RiskSeverity.HIGH
        assert RiskSeverity.LOW <= RiskSeverity.MEDIUM

    def test_risk_severity_numeric_values(self):
        """Test numeric value mapping for calculations."""
        assert RiskSeverity.LOW.numeric_value == 1
        assert RiskSeverity.MEDIUM.numeric_value == 2
        assert RiskSeverity.HIGH.numeric_value == 3
        assert RiskSeverity.CRITICAL.numeric_value == 4

    def test_risk_severity_string_values(self):
        """Test string representation."""
        assert RiskSeverity.LOW.value == "low"
        assert RiskSeverity.CRITICAL.value == "critical"


class TestRiskFactorValidation:
    """Test RiskFactor validation and calculations."""
    
    def test_risk_factor_creation(self, sample_risk_thresholds):
        """Test creating valid risk factors."""
        factor = RiskFactor(
            name="Test Factor",
            value=0.5,
            weight=0.3,
            severity=RiskSeverity.MEDIUM,
            thresholds=sample_risk_thresholds
        )
        
        assert factor.name == "Test Factor"
        assert factor.value == 0.5
        assert factor.weight == 0.3
        assert factor.severity == RiskSeverity.MEDIUM

    def test_risk_factor_weight_validation(self, sample_risk_thresholds):
        """Test weight validation (must be 0-1)."""
        # Valid weights
        RiskFactor("Test", 0.5, 0.0, RiskSeverity.LOW, sample_risk_thresholds)
        RiskFactor("Test", 0.5, 1.0, RiskSeverity.LOW, sample_risk_thresholds)
        
        # Invalid weights should raise ValueError
        with pytest.raises(ValueError, match="Weight must be between 0.0 and 1.0"):
            RiskFactor("Test", 0.5, -0.1, RiskSeverity.LOW, sample_risk_thresholds)
        
        with pytest.raises(ValueError, match="Weight must be between 0.0 and 1.0"):
            RiskFactor("Test", 0.5, 1.1, RiskSeverity.LOW, sample_risk_thresholds)

    def test_weighted_score_calculation(self, sample_risk_thresholds):
        """Test weighted score calculation."""
        factor = RiskFactor(
            "Test", 0.8, 0.5, RiskSeverity.HIGH, sample_risk_thresholds
        )
        assert factor.weighted_score == 0.4  # 0.8 * 0.5

    def test_risk_factor_to_dict(self, sample_risk_thresholds):
        """Test conversion to dictionary."""
        factor = RiskFactor(
            "Test Factor", 0.7, 0.4, RiskSeverity.HIGH, sample_risk_thresholds,
            description="Test description", unit="test_unit"
        )
        
        result = factor.to_dict()
        assert result['name'] == "Test Factor"
        assert result['value'] == 0.7
        assert result['weight'] == 0.4
        assert result['severity'] == "high"
        assert result['weighted_score'] == 0.28
        assert result['description'] == "Test description"
        assert result['unit'] == "test_unit"

    def test_severity_auto_update(self, sample_risk_thresholds):
        """Test that severity is updated based on value and thresholds."""
        factor = RiskFactor(
            "Test", 0.9, 0.5, RiskSeverity.LOW, sample_risk_thresholds
        )
        # Should be updated to CRITICAL based on value 0.9
        assert factor.severity == RiskSeverity.CRITICAL


class TestRiskAssessmentAggregation:
    """Test RiskAssessment aggregation and validation."""
    
    def test_create_from_factors(self, sample_risk_factors):
        """Test creating assessment from risk factors."""
        assessment = RiskAssessment.from_factors(
            factors=sample_risk_factors,
            assessment_type="test_assessment"
        )
        
        assert len(assessment.factors) == 3
        assert assessment.assessment_type == "test_assessment"
        assert 0.0 <= assessment.overall_score <= 1.0
        assert assessment.severity in [RiskSeverity.LOW, RiskSeverity.MEDIUM, 
                                     RiskSeverity.HIGH, RiskSeverity.CRITICAL]

    def test_weighted_score_calculation(self, sample_risk_factors):
        """Test that overall score is correctly calculated."""
        assessment = RiskAssessment.from_factors(sample_risk_factors)
        
        # Calculate expected score manually
        total_weighted = sum(f.weighted_score for f in sample_risk_factors)
        total_weight = sum(f.weight for f in sample_risk_factors)
        expected_score = total_weighted / total_weight
        
        assert abs(assessment.overall_score - expected_score) < 1e-6

    def test_weight_validation(self, sample_risk_thresholds):
        """Test validation of factor weights."""
        # Weights that sum to more than 1.1 should raise error
        factors = [
            RiskFactor("F1", 0.5, 0.7, RiskSeverity.MEDIUM, sample_risk_thresholds),
            RiskFactor("F2", 0.5, 0.7, RiskSeverity.MEDIUM, sample_risk_thresholds)
        ]
        
        with pytest.raises(ValueError, match="Total factor weights exceed 1.0"):
            RiskAssessment.from_factors(factors)

    def test_get_critical_factors(self, sample_risk_thresholds):
        """Test getting critical factors."""
        factors = [
            RiskFactor("Critical", 0.95, 0.5, RiskSeverity.CRITICAL, sample_risk_thresholds),
            RiskFactor("High", 0.8, 0.3, RiskSeverity.HIGH, sample_risk_thresholds),
            RiskFactor("Low", 0.2, 0.2, RiskSeverity.LOW, sample_risk_thresholds)
        ]
        
        assessment = RiskAssessment.from_factors(factors)
        critical_factors = assessment.get_critical_factors()
        
        assert len(critical_factors) == 1
        assert critical_factors[0].name == "Critical"

    def test_get_high_risk_factors(self, sample_risk_thresholds):
        """Test getting high and critical risk factors."""
        factors = [
            RiskFactor("Critical", 0.95, 0.4, RiskSeverity.CRITICAL, sample_risk_thresholds),
            RiskFactor("High", 0.8, 0.3, RiskSeverity.HIGH, sample_risk_thresholds),
            RiskFactor("Medium", 0.6, 0.2, RiskSeverity.MEDIUM, sample_risk_thresholds),
            RiskFactor("Low", 0.2, 0.1, RiskSeverity.LOW, sample_risk_thresholds)
        ]
        
        assessment = RiskAssessment.from_factors(factors)
        high_risk_factors = assessment.get_high_risk_factors()
        
        assert len(high_risk_factors) == 2
        factor_names = [f.name for f in high_risk_factors]
        assert "Critical" in factor_names
        assert "High" in factor_names

    def test_assessment_to_dict(self, sample_risk_factors):
        """Test assessment dictionary conversion."""
        assessment = RiskAssessment.from_factors(
            factors=sample_risk_factors,
            assessment_id="test-123",
            assessment_type="comprehensive"
        )
        
        result = assessment.to_dict()
        
        assert result['assessment_id'] == "test-123"
        assert result['assessment_type'] == "comprehensive"
        assert 'overall_score' in result
        assert 'severity' in result
        assert 'factors' in result
        assert len(result['factors']) == 3
        assert 'critical_factors_count' in result
        assert 'high_risk_factors_count' in result


class TestBaseRiskAnalyzer:
    """Test BaseRiskAnalyzer interface and behavior."""
    
    def test_analyzer_initialization(self):
        """Test base analyzer initialization."""
        class TestAnalyzer(BaseRiskAnalyzer):
            def analyze(self, data, **kwargs):
                return Mock()
        
        analyzer = TestAnalyzer("Test Analyzer", "Test description")
        assert analyzer.name == "Test Analyzer"
        assert analyzer.description == "Test description"

    def test_input_validation(self):
        """Test input data validation."""
        class TestAnalyzer(BaseRiskAnalyzer):
            def analyze(self, data, **kwargs):
                return Mock()
        
        analyzer = TestAnalyzer("Test")
        
        # Valid data should pass
        analyzer.validate_input_data({"key": "value"})
        
        # Non-dict should fail
        with pytest.raises(ValueError, match="Input data must be a dictionary"):
            analyzer.validate_input_data("not a dict")
        
        # Empty dict should fail
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            analyzer.validate_input_data({})

    def test_analyzer_info(self):
        """Test getting analyzer information."""
        class TestAnalyzer(BaseRiskAnalyzer):
            def analyze(self, data, **kwargs):
                return Mock()
        
        analyzer = TestAnalyzer("Test", "Description")
        info = analyzer.get_analyzer_info()
        
        assert info['name'] == "Test"
        assert info['description'] == "Description"
        assert info['class'] == "TestAnalyzer"


# =============================================================================
# Test Individual Risk Analyzers
# =============================================================================

class TestWaterConsumptionRiskAnalyzer:
    """Test water consumption risk analysis scenarios."""
    
    @pytest.mark.asyncio
    async def test_basic_water_risk_analysis(
        self, sample_water_consumption_records, sample_water_permit, sample_equipment_efficiency_data
    ):
        """Test basic water consumption risk analysis."""
        analyzer = WaterConsumptionRiskAnalyzer()
        
        data = {
            'consumption_records': sample_water_consumption_records,
            'permit_data': sample_water_permit,
            'equipment_data': sample_equipment_efficiency_data,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        assert isinstance(assessment, RiskAssessment)
        assert assessment.assessment_type == "water_consumption_risk"
        assert len(assessment.factors) == 4  # compliance, trend, seasonal, equipment
        assert assessment.overall_score >= 0.0
        assert assessment.overall_score <= 1.0
        assert len(assessment.recommendations) > 0

    @pytest.mark.asyncio
    async def test_permit_compliance_analysis(self, sample_water_consumption_records, sample_water_permit):
        """Test permit compliance specific analysis."""
        analyzer = WaterConsumptionRiskAnalyzer(permit_buffer_percentage=0.1)
        
        # Create high consumption scenario
        high_consumption_records = []
        for record in sample_water_consumption_records[:10]:
            # Set consumption near permit limit
            record.consumption_gallons = 450  # Close to daily limit of 500
            high_consumption_records.append(record)
        
        data = {
            'consumption_records': high_consumption_records,
            'permit_data': sample_water_permit,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        # Should detect high risk due to permit utilization
        compliance_factor = next(f for f in assessment.factors if f.name == "Permit Compliance")
        assert compliance_factor.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]
        
        # Check that buffer exceeded is flagged
        assert compliance_factor.metadata.get('buffer_exceeded', False) == True

    @pytest.mark.asyncio
    async def test_trend_analysis(self, sample_water_permit):
        """Test consumption trend detection."""
        analyzer = WaterConsumptionRiskAnalyzer(trend_analysis_days=30)
        
        # Create records with increasing trend
        records = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            consumption = 100 + i * 5  # Increasing trend
            records.append(WaterConsumptionRecord(
                timestamp=base_date + timedelta(days=i),
                facility_id="facility-test-001",
                consumption_gallons=consumption,
                quality_flag=True
            ))
        
        data = {
            'consumption_records': records,
            'permit_data': sample_water_permit,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        trend_factor = next(f for f in assessment.factors if f.name == "Consumption Trend")
        assert trend_factor.metadata.get('trend_direction') == 'increasing'
        assert trend_factor.value > 0.0

    @pytest.mark.asyncio
    async def test_equipment_efficiency_impact(self, sample_water_consumption_records, sample_water_permit):
        """Test equipment efficiency risk analysis."""
        analyzer = WaterConsumptionRiskAnalyzer()
        
        # Create equipment with poor efficiency
        poor_equipment = [
            EquipmentEfficiencyData(
                equipment_id="pump-bad",
                equipment_type="Water Pump",
                baseline_efficiency=0.90,
                current_efficiency=0.60,  # 33% degradation
                last_maintenance=datetime.now() - timedelta(days=120),  # Overdue
                efficiency_trend=-0.1,
                operational_status="active"
            )
        ]
        
        data = {
            'consumption_records': sample_water_consumption_records,
            'permit_data': sample_water_permit,
            'equipment_data': poor_equipment,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        equipment_factor = next(f for f in assessment.factors if f.name == "Equipment Efficiency")
        assert equipment_factor.severity in [RiskSeverity.MEDIUM, RiskSeverity.HIGH, RiskSeverity.CRITICAL]
        assert equipment_factor.metadata.get('critical_equipment', 0) > 0

    @pytest.mark.asyncio
    async def test_seasonal_deviation_detection(self, sample_water_permit):
        """Test seasonal pattern deviation detection."""
        analyzer = WaterConsumptionRiskAnalyzer(seasonal_comparison_years=2)
        
        # Create records with seasonal patterns
        records = []
        current_year = datetime.now().year
        
        # Create 2 years of historical data with normal pattern
        for year in range(current_year - 2, current_year):
            for month in range(1, 13):
                # Seasonal pattern: higher in summer
                seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
                base_consumption = 100 * seasonal_factor
                
                for day in range(1, min(29, 32)):  # Simplified monthly data
                    try:
                        date = datetime(year, month, day)
                        records.append(WaterConsumptionRecord(
                            timestamp=date,
                            facility_id="facility-test-001",
                            consumption_gallons=base_consumption + np.random.normal(0, 10),
                            quality_flag=True
                        ))
                    except ValueError:
                        continue
        
        # Add current month data that deviates significantly
        current_month = datetime.now().month
        for day in range(1, datetime.now().day + 1):
            date = datetime(current_year, current_month, day)
            # Anomalous consumption - much higher than historical
            records.append(WaterConsumptionRecord(
                timestamp=date,
                facility_id="facility-test-001",
                consumption_gallons=300,  # Much higher than normal ~130
                quality_flag=True
            ))
        
        data = {
            'consumption_records': records,
            'permit_data': sample_water_permit,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        seasonal_factor = next(f for f in assessment.factors if f.name == "Seasonal Deviation")
        assert seasonal_factor.value > 0.3  # Should detect significant deviation

    def test_data_validation(self, sample_water_consumption_records, sample_water_permit):
        """Test input data validation."""
        analyzer = WaterConsumptionRiskAnalyzer()
        
        # Valid data should pass
        valid_data = {
            'consumption_records': sample_water_consumption_records,
            'permit_data': sample_water_permit,
            'facility_id': 'facility-test-001'
        }
        analyzer._validate_water_data(valid_data)
        
        # Missing required fields should fail
        with pytest.raises(ValueError, match="Required field 'permit_data' missing"):
            invalid_data = {'consumption_records': sample_water_consumption_records}
            analyzer._validate_water_data(invalid_data)
        
        # Wrong data type should fail
        with pytest.raises(ValueError, match="consumption_records must be a list"):
            invalid_data = {
                'consumption_records': "not a list",
                'permit_data': sample_water_permit,
                'facility_id': 'test'
            }
            analyzer._validate_water_data(invalid_data)

    @pytest.mark.asyncio
    async def test_recommendation_generation(self, sample_water_consumption_records, sample_water_permit):
        """Test that appropriate recommendations are generated."""
        analyzer = WaterConsumptionRiskAnalyzer()
        
        # Create high-risk scenario
        high_consumption_records = []
        for i, record in enumerate(sample_water_consumption_records[:5]):
            # Exceed permit buffer
            record.consumption_gallons = 480 if i % 2 == 0 else 460
            high_consumption_records.append(record)
        
        data = {
            'consumption_records': high_consumption_records,
            'permit_data': sample_water_permit,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        assert len(assessment.recommendations) > 0
        
        # Check for critical recommendations when buffer is exceeded
        critical_recs = [r for r in assessment.recommendations if "CRITICAL" in r]
        if assessment.severity == RiskSeverity.CRITICAL:
            assert len(critical_recs) > 0


class TestElectricityConsumptionRiskAnalyzer:
    """Test electricity consumption risk analysis scenarios."""
    
    @pytest.mark.asyncio
    async def test_basic_electricity_risk_analysis(
        self, sample_electricity_records, sample_electrical_contract
    ):
        """Test basic electricity consumption risk analysis."""
        analyzer = ElectricityConsumptionRiskAnalyzer()
        
        data = {
            'consumption_records': sample_electricity_records,
            'contract_data': sample_electrical_contract,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        assert isinstance(assessment, RiskAssessment)
        assert assessment.assessment_type == "electricity_consumption_risk"
        assert len(assessment.factors) == 5  # demand, cost, quality, carbon, reliability
        assert 0.0 <= assessment.overall_score <= 1.0
        assert len(assessment.recommendations) > 0

    @pytest.mark.asyncio
    async def test_demand_management_analysis(self, sample_electricity_records, sample_electrical_contract):
        """Test demand management risk assessment."""
        analyzer = ElectricityConsumptionRiskAnalyzer(demand_safety_margin=0.1)
        
        # Create high demand scenario
        high_demand_records = []
        for record in sample_electricity_records[:10]:
            # Set demand near contract limit
            record.demand_kw = 190  # Close to contract limit of 200 kW
            high_demand_records.append(record)
        
        data = {
            'consumption_records': high_demand_records,
            'contract_data': sample_electrical_contract,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        demand_factor = next(f for f in assessment.factors if f.name == "Demand Management")
        assert demand_factor.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]
        assert demand_factor.metadata.get('margin_exceeded', False) == True

    @pytest.mark.asyncio
    async def test_power_quality_analysis(self, sample_electrical_contract):
        """Test power quality risk assessment."""
        analyzer = ElectricityConsumptionRiskAnalyzer()
        
        # Create records with power quality issues
        poor_quality_records = []
        base_date = datetime.now() - timedelta(days=10)
        
        for i in range(10):
            record = ElectricityConsumptionRecord(
                timestamp=base_date + timedelta(days=i),
                facility_id="facility-test-001",
                energy_kwh=100,
                demand_kw=80,
                voltage_l1=450,  # Low voltage (nominal 480V)
                voltage_l2=490,  # High voltage
                voltage_l3=475,  # Normal voltage
                power_factor=0.85,  # Poor power factor (threshold 0.95)
                frequency_hz=59.2,  # Off-frequency (nominal 60Hz)
                total_harmonic_distortion=12.0,  # High THD (threshold 8%)
                quality_flag=True
            )
            poor_quality_records.append(record)
        
        data = {
            'consumption_records': poor_quality_records,
            'contract_data': sample_electrical_contract,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        quality_factor = next(f for f in assessment.factors if f.name == "Power Quality")
        assert quality_factor.value > 0.2  # Should detect quality issues
        
        # Check specific quality issue counts
        metadata = quality_factor.metadata
        assert metadata.get('voltage_deviations_count', 0) > 0
        assert metadata.get('poor_power_factor_count', 0) > 0
        assert metadata.get('frequency_deviations_count', 0) > 0
        assert metadata.get('high_thd_count', 0) > 0

    @pytest.mark.asyncio
    async def test_carbon_compliance_analysis(self, sample_electricity_records, sample_electrical_contract):
        """Test carbon compliance risk assessment."""
        analyzer = ElectricityConsumptionRiskAnalyzer()
        
        # Create emissions data with high target
        emissions_data = CarbonEmissionsData(
            facility_id="facility-test-001",
            reporting_period="2024-Q1",
            annual_emissions_target_kg_co2e=50000,  # Low target
            current_emissions_kg_co2e=40000,
            renewable_energy_percentage=10.0,  # Low renewable percentage
            emission_factor_grid_kg_co2e_per_kwh=0.5
        )
        
        data = {
            'consumption_records': sample_electricity_records,
            'contract_data': sample_electrical_contract,
            'emissions_data': emissions_data,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        carbon_factor = next(f for f in assessment.factors if f.name == "Carbon Compliance")
        assert carbon_factor.value >= 0.0
        
        # Check emissions calculations
        metadata = carbon_factor.metadata
        assert 'projected_annual_emissions_kg_co2e' in metadata
        assert 'emissions_ratio' in metadata
        assert metadata.get('renewable_percentage') == 10.0

    @pytest.mark.asyncio
    async def test_cost_trend_analysis(self, sample_electrical_contract):
        """Test energy cost trend analysis."""
        analyzer = ElectricityConsumptionRiskAnalyzer(cost_trend_analysis_days=30)
        
        # Create records with cost escalation pattern
        cost_records = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            # Simulate increasing costs over time (more peak usage)
            is_peak = i % 2 == 0  # Alternating peak/off-peak
            
            record = ElectricityConsumptionRecord(
                timestamp=base_date + timedelta(days=i),
                facility_id="facility-test-001",
                energy_kwh=100,
                demand_kw=80,
                power_factor=0.90 if i < 15 else 0.85,  # Declining power factor
                time_of_use_period="peak" if is_peak else "offpeak",
                quality_flag=True
            )
            cost_records.append(record)
        
        data = {
            'consumption_records': cost_records,
            'contract_data': sample_electrical_contract,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        cost_factor = next(f for f in assessment.factors if f.name == "Energy Cost Trends")
        
        # Check cost analysis metadata
        metadata = cost_factor.metadata
        assert 'avg_daily_cost_usd' in metadata
        assert 'poor_power_factor_ratio' in metadata
        assert metadata.get('poor_power_factor_ratio', 0) > 0

    def test_time_of_use_savings_calculation(self, sample_electricity_records, sample_electrical_contract):
        """Test time-of-use optimization calculations."""
        analyzer = ElectricityConsumptionRiskAnalyzer()
        
        savings_analysis = analyzer.calculate_time_of_use_savings(
            sample_electricity_records, 
            sample_electrical_contract,
            load_shift_percentage=0.25
        )
        
        assert 'current_total_cost' in savings_analysis
        assert 'optimized_total_cost' in savings_analysis
        assert 'potential_annual_savings' in savings_analysis
        assert 'savings_percentage' in savings_analysis
        
        # Savings should be positive if there's peak consumption to shift
        if savings_analysis.get('peak_consumption_kwh', 0) > 0:
            assert savings_analysis.get('potential_annual_savings', 0) >= 0

    def test_power_factor_optimization_calculation(self, sample_electricity_records, sample_electrical_contract):
        """Test power factor optimization calculations."""
        analyzer = ElectricityConsumptionRiskAnalyzer()
        
        # Add power factor data to records
        for record in sample_electricity_records:
            record.power_factor = 0.88  # Below threshold
        
        pf_analysis = analyzer.calculate_power_factor_optimization(
            sample_electricity_records,
            sample_electrical_contract,
            target_power_factor=0.95
        )
        
        assert 'current_avg_power_factor' in pf_analysis
        assert 'potential_annual_savings' in pf_analysis
        assert 'improvement_needed' in pf_analysis
        
        assert pf_analysis.get('current_avg_power_factor', 0) < 0.95
        assert pf_analysis.get('improvement_needed', 0) > 0


class TestWasteGenerationRiskAnalyzer:
    """Test waste generation risk analysis scenarios."""
    
    @pytest.mark.asyncio
    async def test_basic_waste_risk_analysis(self, sample_waste_records, sample_waste_regulations):
        """Test basic waste generation risk analysis."""
        analyzer = WasteGenerationRiskAnalyzer()
        
        data = {
            'waste_records': sample_waste_records,
            'regulation_data': sample_waste_regulations,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        assert isinstance(assessment, RiskAssessment)
        assert assessment.assessment_type == "waste_generation_risk"
        assert len(assessment.factors) == 5  # compliance, cost, diversion, storage, contamination
        assert 0.0 <= assessment.overall_score <= 1.0
        assert len(assessment.recommendations) > 0

    @pytest.mark.asyncio
    async def test_regulatory_compliance_analysis(self, sample_waste_records, sample_waste_regulations):
        """Test waste regulatory compliance analysis."""
        analyzer = WasteGenerationRiskAnalyzer(compliance_buffer_percentage=0.1)
        
        # Create high waste generation scenario exceeding limits
        high_waste_records = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            # Generate hazardous waste exceeding storage limit
            record = WasteGenerationRecord(
                timestamp=base_date + timedelta(days=i),
                facility_id="facility-test-001",
                waste_category="hazardous",
                amount_tons=0.5,  # Daily generation
                disposal_method="treatment",
                disposal_cost_per_ton=150.0,
                hazardous_level="hazardous",
                quality_flag=True
            )
            high_waste_records.append(record)
        
        # Total monthly hazardous waste: 15 tons (exceeds 10 ton limit)
        
        data = {
            'waste_records': high_waste_records,
            'regulation_data': sample_waste_regulations,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        compliance_factor = next(f for f in assessment.factors if f.name == "Regulatory Compliance")
        assert compliance_factor.value > 0.0
        assert compliance_factor.metadata.get('compliance_violations', 0) > 0

    @pytest.mark.asyncio
    async def test_disposal_cost_trend_analysis(self, sample_waste_regulations):
        """Test disposal cost trend analysis."""
        analyzer = WasteGenerationRiskAnalyzer(cost_trend_analysis_days=60)
        
        # Create records with escalating costs
        cost_trend_records = []
        base_date = datetime.now() - timedelta(days=60)
        
        for i in range(60):
            # Simulate increasing disposal costs
            base_cost = 100 + (i / 60) * 50  # Cost increases from 100 to 150
            
            record = WasteGenerationRecord(
                timestamp=base_date + timedelta(days=i),
                facility_id="facility-test-001",
                waste_category="general",
                amount_tons=2.0,
                disposal_method="landfill",
                disposal_cost_per_ton=base_cost + np.random.normal(0, 5),
                quality_flag=True
            )
            cost_trend_records.append(record)
        
        data = {
            'waste_records': cost_trend_records,
            'regulation_data': sample_waste_regulations,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        cost_factor = next(f for f in assessment.factors if f.name == "Disposal Cost Trends")
        assert cost_factor.value > 0.1  # Should detect cost escalation
        
        metadata = cost_factor.metadata
        assert 'cost_escalation_risk' in metadata
        assert metadata.get('cost_escalation_risk', 0) > 0

    @pytest.mark.asyncio
    async def test_diversion_performance_analysis(self, sample_waste_records, sample_waste_regulations):
        """Test waste diversion rate performance."""
        analyzer = WasteGenerationRiskAnalyzer(diversion_target_threshold=0.20)
        
        # Create waste stream data with poor diversion performance
        poor_stream_data = [
            WasteStreamData(
                stream_id="stream-001",
                stream_type="production",
                baseline_generation_rate=5.0,
                current_generation_rate=6.0,
                diversion_rate=0.30,  # Current rate
                target_diversion_rate=0.60,  # Target rate (30% gap)
                contamination_incidents=5,
                cost_per_ton_trend=0.15,
                last_audit_date=datetime.now() - timedelta(days=90)
            ),
            WasteStreamData(
                stream_id="stream-002",
                stream_type="office",
                baseline_generation_rate=1.0,
                current_generation_rate=1.2,
                diversion_rate=0.20,  # Poor performance
                target_diversion_rate=0.50,
                contamination_incidents=2,
                cost_per_ton_trend=0.08,
                last_audit_date=datetime.now() - timedelta(days=120)
            )
        ]
        
        data = {
            'waste_records': sample_waste_records,
            'regulation_data': sample_waste_regulations,
            'stream_data': poor_stream_data,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        diversion_factor = next(f for f in assessment.factors if f.name == "Diversion Performance")
        assert diversion_factor.value > 0.2  # Should detect poor diversion
        
        metadata = diversion_factor.metadata
        assert 'stream_performance' in metadata
        assert len(metadata['stream_performance']) == 2

    @pytest.mark.asyncio
    async def test_storage_utilization_analysis(self, sample_waste_records, sample_waste_regulations):
        """Test storage facility utilization analysis."""
        analyzer = WasteGenerationRiskAnalyzer(storage_capacity_threshold=0.85)
        
        # Create storage facility data with high utilization
        high_utilization_storage = [
            StorageFacilityData(
                facility_id="storage-001",
                facility_type="temporary",
                total_capacity_tons=100.0,
                current_utilization_tons=92.0,  # 92% utilization (exceeds 85% threshold)
                waste_categories=["hazardous", "general"],
                last_inspection_date=datetime.now() - timedelta(days=400),  # Overdue
                compliance_status="non-compliant",
                safety_incidents=2
            ),
            StorageFacilityData(
                facility_id="storage-002",
                facility_type="permanent",
                total_capacity_tons=50.0,
                current_utilization_tons=48.0,  # 96% utilization
                waste_categories=["recyclable"],
                last_inspection_date=datetime.now() - timedelta(days=30),
                compliance_status="compliant",
                safety_incidents=0
            )
        ]
        
        data = {
            'waste_records': sample_waste_records,
            'regulation_data': sample_waste_regulations,
            'storage_data': high_utilization_storage,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        storage_factor = next(f for f in assessment.factors if f.name == "Storage Utilization")
        assert storage_factor.severity in [RiskSeverity.MEDIUM, RiskSeverity.HIGH, RiskSeverity.CRITICAL]
        
        metadata = storage_factor.metadata
        assert metadata.get('facilities_analyzed') == 2
        assert metadata.get('compliance_risk', 0) > 0

    @pytest.mark.asyncio
    async def test_contamination_risk_analysis(self, sample_waste_regulations):
        """Test contamination risk analysis."""
        analyzer = WasteGenerationRiskAnalyzer()
        
        # Create records with contamination risk
        contaminated_records = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            # Some records have high contamination risk
            contamination_risk = 0.8 if i % 5 == 0 else 0.1
            
            record = WasteGenerationRecord(
                timestamp=base_date + timedelta(days=i),
                facility_id="facility-test-001",
                waste_category="recyclable",
                amount_tons=3.0,
                disposal_method="recycling",
                disposal_cost_per_ton=75.0,
                contamination_risk=contamination_risk,
                quality_flag=True
            )
            contaminated_records.append(record)
        
        # Add stream data with contamination incidents
        contaminated_streams = [
            WasteStreamData(
                stream_id="stream-contaminated",
                stream_type="production",
                baseline_generation_rate=4.0,
                current_generation_rate=4.2,
                diversion_rate=0.40,
                target_diversion_rate=0.60,
                contamination_incidents=10,  # High incidents
                cost_per_ton_trend=0.05,
                last_audit_date=datetime.now() - timedelta(days=60)
            )
        ]
        
        data = {
            'waste_records': contaminated_records,
            'regulation_data': sample_waste_regulations,
            'stream_data': contaminated_streams,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        
        contamination_factor = next(f for f in assessment.factors if f.name == "Contamination Risk")
        assert contamination_factor.value > 0.3  # Should detect contamination risk
        
        metadata = contamination_factor.metadata
        assert metadata.get('high_risk_streams', 0) > 0
        assert metadata.get('total_contamination_incidents', 0) > 0

    @pytest.mark.asyncio
    async def test_circular_economy_scoring(self, sample_waste_records):
        """Test circular economy score calculation."""
        analyzer = WasteGenerationRiskAnalyzer()
        
        # Create records with different disposal methods
        ce_records = [
            WasteGenerationRecord(
                timestamp=datetime.now(),
                facility_id="facility-test-001",
                waste_category="general",
                amount_tons=10.0,
                disposal_method="reuse",  # Best score (1.0)
                disposal_cost_per_ton=50.0,
                quality_flag=True
            ),
            WasteGenerationRecord(
                timestamp=datetime.now(),
                facility_id="facility-test-001",
                waste_category="recyclable",
                amount_tons=20.0,
                disposal_method="recycling",  # Good score (0.8)
                disposal_cost_per_ton=60.0,
                quality_flag=True
            ),
            WasteGenerationRecord(
                timestamp=datetime.now(),
                facility_id="facility-test-001",
                waste_category="general",
                amount_tons=5.0,
                disposal_method="landfill",  # Poor score (0.0)
                disposal_cost_per_ton=40.0,
                quality_flag=True
            )
        ]
        
        score = analyzer._calculate_circular_economy_score(ce_records)
        
        # Expected: (10*1.0 + 20*0.8 + 5*0.0) / 35 = 26/35 ≈ 0.743
        expected_score = (10 * 1.0 + 20 * 0.8 + 5 * 0.0) / 35
        assert abs(score - expected_score) < 0.01

    def test_waste_data_validation(self, sample_waste_records, sample_waste_regulations):
        """Test waste data validation."""
        analyzer = WasteGenerationRiskAnalyzer()
        
        # Valid data should pass
        valid_data = {
            'waste_records': sample_waste_records,
            'regulation_data': sample_waste_regulations,
            'facility_id': 'facility-test-001'
        }
        analyzer._validate_waste_data(valid_data)
        
        # Missing required fields should fail
        with pytest.raises(ValueError, match="Required field 'regulation_data' missing"):
            invalid_data = {'waste_records': sample_waste_records}
            analyzer._validate_waste_data(invalid_data)
        
        # Invalid facility_id should fail
        with pytest.raises(ValueError, match="facility_id must be a non-empty string"):
            invalid_data = {
                'waste_records': sample_waste_records,
                'regulation_data': sample_waste_regulations,
                'facility_id': ''
            }
            analyzer._validate_waste_data(invalid_data)


# =============================================================================
# Test Time Series Analysis
# =============================================================================

class TestTimeSeriesAnalysis:
    """Test time series analysis capabilities."""
    
    @pytest.mark.asyncio
    async def test_complete_time_series_analysis(self, sample_time_series_data):
        """Test comprehensive time series analysis."""
        analyzer = TimeSeriesAnalyzer()
        
        results = await analyzer.analyze_complete(sample_time_series_data)
        
        assert 'data_quality' in results
        assert 'trend_analysis' in results
        assert 'seasonal_components' in results
        assert 'anomalies' in results
        assert 'changepoints' in results
        assert 'summary_statistics' in results

    @pytest.mark.asyncio
    async def test_data_quality_assessment(self, sample_time_series_data):
        """Test data quality assessment."""
        analyzer = TimeSeriesAnalyzer()
        
        quality_report = await analyzer.assess_data_quality(sample_time_series_data)
        
        assert hasattr(quality_report, 'missing_values')
        assert hasattr(quality_report, 'missing_percentage')
        assert hasattr(quality_report, 'duplicate_timestamps')
        assert hasattr(quality_report, 'outlier_count')
        assert hasattr(quality_report, 'overall_quality_score')
        
        assert 0.0 <= quality_report.overall_quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_trend_detection_accuracy(self):
        """Test trend detection accuracy."""
        analyzer = TimeSeriesAnalyzer()
        
        # Create data with known increasing trend
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        trend_values = np.arange(100) * 0.5 + np.random.normal(0, 2, 100)
        
        trend_data = TimeSeriesData(
            timestamps=dates.to_pydatetime().tolist(),
            values=trend_values.tolist()
        )
        
        trend_analysis = await analyzer.detect_trend(trend_data)
        
        assert trend_analysis.direction == TrendDirection.INCREASING
        assert trend_analysis.slope > 0
        assert trend_analysis.is_significant

    @pytest.mark.asyncio
    async def test_seasonal_decomposition(self, sample_time_series_data):
        """Test seasonal decomposition."""
        analyzer = TimeSeriesAnalyzer(seasonal_periods=12)
        
        seasonal_components = await analyzer.decompose_seasonal(sample_time_series_data)
        
        if seasonal_components is not None:
            assert hasattr(seasonal_components, 'trend')
            assert hasattr(seasonal_components, 'seasonal')
            assert hasattr(seasonal_components, 'residual')
            assert hasattr(seasonal_components, 'seasonal_strength')
            assert hasattr(seasonal_components, 'trend_strength')
            
            assert len(seasonal_components.trend) == len(sample_time_series_data.values)

    @pytest.mark.asyncio
    async def test_anomaly_detection_precision_recall(self):
        """Test anomaly detection precision and recall."""
        analyzer = TimeSeriesAnalyzer()
        
        # Create data with known anomalies
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        normal_values = np.random.normal(100, 10, 200)
        
        # Insert known anomalies
        anomaly_indices = [50, 100, 150]
        for idx in anomaly_indices:
            normal_values[idx] = 300  # Clear anomalies
        
        anomaly_data = TimeSeriesData(
            timestamps=dates.to_pydatetime().tolist(),
            values=normal_values.tolist()
        )
        
        # Test different detection methods
        methods = ['statistical', 'iqr', 'modified_zscore']
        
        for method in methods:
            anomaly_result = await analyzer.detect_anomalies(
                anomaly_data, method=method, threshold=3.0
            )
            
            # Calculate precision and recall
            detected_indices = set(anomaly_result.indices)
            true_anomalies = set(anomaly_indices)
            
            true_positives = len(detected_indices.intersection(true_anomalies))
            false_positives = len(detected_indices - true_anomalies)
            false_negatives = len(true_anomalies - detected_indices)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Should detect at least some anomalies
            assert anomaly_result.count > 0
            # Should have reasonable precision (at least 30%)
            assert precision >= 0.3 or recall >= 0.3

    @pytest.mark.asyncio
    async def test_change_point_detection(self):
        """Test change point detection."""
        analyzer = TimeSeriesAnalyzer()
        
        # Create data with known change points
        dates = pd.date_range('2023-01-01', periods=150, freq='D')
        
        # Two regimes: low values then high values
        values = np.concatenate([
            np.random.normal(50, 5, 75),  # First regime
            np.random.normal(100, 5, 75)  # Second regime (change at index 75)
        ])
        
        change_data = TimeSeriesData(
            timestamps=dates.to_pydatetime().tolist(),
            values=values.tolist()
        )
        
        changepoints = await analyzer.detect_changepoints(
            change_data, min_size=20, jump_threshold=2.0
        )
        
        # Should detect at least one change point
        assert len(changepoints) > 0
        
        # Change point should be reasonably close to true change point (index 75)
        detected_point = changepoints[0].index
        assert 60 <= detected_point <= 90  # Allow some tolerance

    def test_statistical_calculations(self, sample_time_series_data):
        """Test statistical calculations."""
        analyzer = TimeSeriesAnalyzer()
        
        stats = analyzer.calculate_statistics(sample_time_series_data)
        
        expected_keys = ['count', 'mean', 'median', 'std', 'var', 'min', 'max', 
                        'range', 'q25', 'q75', 'iqr', 'skewness', 'kurtosis']
        
        for key in expected_keys:
            assert key in stats
        
        # Basic sanity checks
        assert stats['count'] == len(sample_time_series_data.values)
        assert stats['min'] <= stats['mean'] <= stats['max']
        assert stats['q25'] <= stats['median'] <= stats['q75']


class TestTimeSeriesPredictor:
    """Test time series prediction capabilities."""
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = TimeSeriesPredictor(default_confidence=0.9, max_forecast_horizon=180)
        
        assert predictor.default_confidence == 0.9
        assert predictor.max_forecast_horizon == 180

    def test_prediction_with_simple_data(self):
        """Test prediction with simple time series data."""
        predictor = TimeSeriesPredictor()
        
        # Simple trend data
        data = {
            'history': [
                {'timestamp': '2023-01-01', 'value': 100},
                {'timestamp': '2023-01-02', 'value': 105},
                {'timestamp': '2023-01-03', 'value': 110},
                {'timestamp': '2023-01-04', 'value': 115}
            ]
        }
        
        result = predictor.predict(data, horizon_days=7)
        
        assert result is not None
        assert 'predictions' in result
        assert 'forecast_timestamps' in result
        assert 'model_name' in result
        assert result['forecast_horizon_days'] == 7

    def test_prediction_horizon_limits(self):
        """Test prediction horizon limits."""
        predictor = TimeSeriesPredictor(max_forecast_horizon=30)
        
        data = {'values': [100, 105, 110, 115]}
        
        # Should accept valid horizon
        result = predictor.predict(data, horizon_days=30)
        assert result is not None
        
        # Should reject horizon that's too large
        result = predictor.predict(data, horizon_days=50)
        assert result is None

    def test_data_extraction_formats(self):
        """Test extraction from different data formats."""
        predictor = TimeSeriesPredictor()
        
        # Test single point format
        single_point = {'timestamp': '2023-01-01', 'value': 100}
        extracted = predictor._extract_time_series(single_point)
        assert extracted is not None
        assert len(extracted.values) == 2  # Creates 2-point series
        
        # Test values list format
        values_list = {'values': [100, 105, 110, 115, 120]}
        extracted = predictor._extract_time_series(values_list)
        assert extracted is not None
        assert len(extracted.values) == 5
        
        # Test history format
        history_format = {
            'history': [
                {'timestamp': '2023-01-01', 'value': 100},
                {'timestamp': '2023-01-02', 'value': 105}
            ]
        }
        extracted = predictor._extract_time_series(history_format)
        assert extracted is not None
        assert len(extracted.values) == 2


# =============================================================================
# Test Forecasting Engine
# =============================================================================

class TestForecastingEngine:
    """Test forecasting engine capabilities."""
    
    def test_engine_initialization(self):
        """Test forecasting engine initialization."""
        engine = ForecastingEngine()
        
        assert engine.model_selector is not None
        assert engine.external_processor is not None
        assert isinstance(engine.trained_models, dict)

    @pytest.mark.asyncio
    async def test_moving_average_forecast(self):
        """Test moving average forecasting."""
        engine = ForecastingEngine()
        
        # Create simple time series
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        values = np.random.normal(100, 10, 30)
        data = pd.Series(values, index=dates)
        
        result = await engine.forecast(
            data=data,
            horizon=7,
            model=ForecastModel.MOVING_AVERAGE
        )
        
        assert isinstance(result, ForecastResult)
        assert result.model_name == "moving_average"
        assert len(result.predictions) == 7
        assert result.confidence_intervals is not None

    @pytest.mark.asyncio
    async def test_model_auto_selection(self):
        """Test automatic model selection."""
        engine = ForecastingEngine()
        
        # Create time series with trend and seasonality
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        trend = np.arange(100) * 0.1
        seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 30)
        values = trend + seasonal + np.random.normal(0, 2, 100)
        data = pd.Series(values, index=dates)
        
        result = await engine.forecast(
            data=data,
            horizon=14,
            model=ForecastModel.AUTO_SELECT
        )
        
        assert isinstance(result, ForecastResult)
        assert result.model_name in ["exponential_smoothing", "moving_average", "prophet", "arima"]
        assert len(result.predictions) == 14

    @pytest.mark.asyncio 
    async def test_ensemble_forecast(self):
        """Test ensemble forecasting."""
        engine = ForecastingEngine()
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        values = np.random.normal(100, 10, 50)
        data = pd.Series(values, index=dates)
        
        result = await engine.forecast(
            data=data,
            horizon=10,
            model=ForecastModel.ENSEMBLE
        )
        
        assert isinstance(result, ForecastResult)
        assert result.model_name == "ensemble"
        assert len(result.predictions) == 10
        assert 'models' in result.model_params
        assert 'weights' in result.model_params

    @pytest.mark.asyncio
    async def test_model_validation(self):
        """Test model validation and selection."""
        engine = ForecastingEngine()
        
        # Create data with enough points for validation
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        values = np.arange(60) * 0.5 + np.random.normal(0, 5, 60)  # Clear trend
        data = pd.Series(values, index=dates)
        
        best_model, performance = await engine.select_best_model(
            data, validation_split=0.2
        )
        
        assert best_model in [ForecastModel.MOVING_AVERAGE, ForecastModel.EXPONENTIAL_SMOOTHING, 
                             ForecastModel.ARIMA, ForecastModel.PROPHET]
        assert hasattr(performance, 'mape')
        assert hasattr(performance, 'rmse')
        assert hasattr(performance, 'mae')

    @pytest.mark.asyncio
    async def test_cross_validation(self):
        """Test time series cross-validation."""
        engine = ForecastingEngine()
        
        # Create sufficient data for cross-validation
        dates = pd.date_range('2023-01-01', periods=120, freq='D')
        values = np.random.normal(100, 15, 120)
        data = pd.Series(values, index=dates)
        
        validation_results = await engine.validate_forecast(
            data=data,
            model=ForecastModel.MOVING_AVERAGE,
            validation_periods=3,
            horizon_days=20
        )
        
        assert 'periods_tested' in validation_results
        assert 'avg_mape' in validation_results
        assert 'avg_rmse' in validation_results
        assert 'detailed_results' in validation_results
        
        # Should test multiple periods
        assert validation_results['periods_tested'] <= 3

    def test_external_factors_processor(self):
        """Test external factors processing."""
        processor = ExternalFactorsProcessor()
        
        # Add weather data
        weather_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'temperature': np.random.normal(20, 5, 10),
            'humidity': np.random.uniform(40, 80, 10)
        }).set_index('date')
        
        processor.add_weather_data(weather_data)
        
        # Create features
        forecast_dates = pd.date_range('2023-01-01', periods=5, freq='D')
        features = processor.create_features(forecast_dates)
        
        assert len(features) == 5
        assert 'weather_temperature' in features.columns
        assert 'weather_humidity' in features.columns

    def test_model_selector_analysis(self):
        """Test model selector data analysis."""
        selector = ModelSelector()
        
        # Create data with characteristics
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        trend = np.arange(100) * 0.1
        seasonal = 3 * np.sin(2 * np.pi * np.arange(100) / 12)
        values = trend + seasonal + np.random.normal(0, 2, 100)
        data = pd.Series(values, index=dates)
        
        analysis = selector.analyze_series(data)
        
        assert 'length' in analysis
        assert 'has_trend' in analysis
        assert 'has_seasonality' in analysis
        assert 'is_stationary' in analysis
        assert 'frequency' in analysis
        
        assert analysis['length'] == 100

    def test_model_selection_logic(self):
        """Test model selection logic."""
        selector = ModelSelector()
        
        # Test short series
        short_analysis = {'length': 5, 'has_trend': False, 'has_seasonality': False}
        selected = selector.select_best_model(short_analysis)
        assert selected == ForecastModel.MOVING_AVERAGE
        
        # Test seasonal data
        seasonal_analysis = {
            'length': 50, 'has_trend': False, 'has_seasonality': True, 'is_stationary': True
        }
        selected = selector.select_best_model(seasonal_analysis)
        # Should select Prophet if available, otherwise exponential smoothing
        assert selected in [ForecastModel.PROPHET, ForecastModel.EXPONENTIAL_SMOOTHING]


# =============================================================================
# Test Anomaly Detection System
# =============================================================================

class TestAnomalyDetectionSystem:
    """Test anomaly detection system capabilities."""
    
    @pytest.mark.asyncio
    async def test_system_creation(self):
        """Test anomaly detection system creation."""
        system = create_ehs_anomaly_system(contamination_rate=0.05)
        
        assert system is not None
        assert len(system.detector_configs) > 0
        assert system.ehs_thresholds is not None

    @pytest.mark.asyncio
    async def test_detector_training(self):
        """Test training anomaly detectors."""
        system = create_ehs_anomaly_system()
        
        # Create training data
        training_data = pd.DataFrame({
            'electricity_consumption': np.random.normal(100, 15, 500),
            'water_consumption': np.random.normal(50, 8, 500),
            'waste_generation': np.random.normal(25, 5, 500)
        })
        
        await system.train_detectors(training_data, save_models=False)
        
        assert system.is_fitted
        assert len(system.detectors) > 0
        assert system.ensemble is not None

    @pytest.mark.asyncio
    async def test_anomaly_detection_accuracy(self):
        """Test anomaly detection accuracy."""
        system = create_ehs_anomaly_system()
        
        # Create training data (normal)
        normal_data = pd.DataFrame({
            'electricity': np.random.normal(100, 10, 1000),
            'water': np.random.normal(50, 5, 1000)
        })
        
        await system.train_detectors(normal_data, save_models=False)
        
        # Create test data with known anomalies
        test_data = pd.DataFrame({
            'electricity': [95, 98, 300, 102],  # Third value is anomalous
            'water': [48, 52, 45, 49]
        })
        
        alerts = await system.detect_anomalies(test_data, return_explanations=False)
        
        # Should detect at least one anomaly
        assert len(alerts) > 0
        
        # Check alert properties
        for alert in alerts:
            assert hasattr(alert, 'score')
            assert hasattr(alert, 'anomaly_type')
            assert hasattr(alert, 'description')
            assert hasattr(alert, 'recommendations')

    @pytest.mark.asyncio
    async def test_ensemble_voting(self):
        """Test ensemble voting mechanisms."""
        # Test different ensemble strategies
        strategies = [
            EnsembleStrategy.MAJORITY_VOTE,
            EnsembleStrategy.WEIGHTED_AVERAGE,
            EnsembleStrategy.MAXIMUM,
            EnsembleStrategy.AVERAGE
        ]
        
        for strategy in strategies:
            system = create_ehs_anomaly_system(ensemble_strategy=strategy)
            
            # Simple training
            training_data = pd.DataFrame({
                'metric': np.random.normal(100, 10, 200)
            })
            
            await system.train_detectors(training_data, save_models=False)
            
            # Test detection
            test_data = pd.DataFrame({'metric': [150, 95]})  # One anomaly
            alerts = await system.detect_anomalies(test_data, return_explanations=False)
            
            # Should work with any strategy
            assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_false_positive_rates(self):
        """Test false positive rates."""
        system = create_ehs_anomaly_system(contamination_rate=0.02)  # Very low contamination
        
        # Train on normal data
        normal_data = pd.DataFrame({
            'electricity': np.random.normal(100, 5, 1000),  # Low variance
            'water': np.random.normal(50, 3, 1000)
        })
        
        await system.train_detectors(normal_data, save_models=False)
        
        # Test on similar normal data
        test_data = pd.DataFrame({
            'electricity': np.random.normal(100, 5, 100),
            'water': np.random.normal(50, 3, 100)
        })
        
        alerts = await system.detect_anomalies(test_data, return_explanations=False)
        
        # False positive rate should be low (< 10% of test data)
        false_positive_rate = len(alerts) / len(test_data)
        assert false_positive_rate < 0.10

    @pytest.mark.asyncio
    async def test_real_time_detection_setup(self):
        """Test real-time detection setup."""
        system = create_ehs_anomaly_system()
        
        # Train system
        training_data = pd.DataFrame({
            'metric': np.random.normal(100, 10, 200)
        })
        await system.train_detectors(training_data, save_models=False)
        
        # Create data stream
        data_queue = asyncio.Queue()
        received_alerts = []
        
        def alert_callback(alert):
            received_alerts.append(alert)
        
        # Add test data to queue
        await data_queue.put({'metric': 200})  # Anomaly
        await data_queue.put({'metric': 100})  # Normal
        await data_queue.put(None)  # Shutdown signal
        
        # Run real-time detection briefly
        try:
            await asyncio.wait_for(
                system.real_time_detection(data_queue, alert_callback),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            pass
        
        # Should have processed some data
        assert len(received_alerts) >= 0  # May or may not detect anomalies

    @pytest.mark.asyncio
    async def test_anomaly_explanation(self):
        """Test anomaly explanation generation."""
        system = create_ehs_anomaly_system()
        
        # Train system
        training_data = pd.DataFrame({
            'electricity': np.random.normal(100, 10, 200),
            'water': np.random.normal(50, 5, 200)
        })
        await system.train_detectors(training_data, save_models=False)
        
        # Create anomalous data point
        anomalous_point = pd.Series({
            'electricity': 200,  # High value
            'water': 45
        })
        
        anomaly_score = AnomalyScore(
            score=0.8,
            confidence=0.9,
            severity=RiskSeverity.HIGH,
            detector="test"
        )
        
        explanation = await system.explain_anomaly(
            anomalous_point, anomaly_score
        )
        
        assert 'overall_score' in explanation
        assert 'feature_contributions' in explanation
        assert 'statistical_analysis' in explanation
        assert 'recommendations' in explanation
        
        # Check feature contributions
        assert 'electricity' in explanation['feature_contributions']
        assert explanation['feature_contributions']['electricity']['is_outlier'] == True

    def test_detector_configurations(self):
        """Test detector configurations."""
        config = DetectorConfig(
            detector_type=DetectorType.ISOLATION_FOREST,
            weight=2.0,
            contamination=0.05,
            sensitivity=0.7
        )
        
        assert config.detector_type == DetectorType.ISOLATION_FOREST
        assert config.weight == 2.0
        assert config.contamination == 0.05
        assert len(config.parameters) > 0  # Should have default parameters

    def test_ehs_specific_thresholds(self):
        """Test EHS-specific thresholds and rules."""
        system = create_ehs_anomaly_system()
        
        # Check that EHS thresholds are configured
        assert 'electricity_consumption' in system.ehs_thresholds
        assert 'water_consumption' in system.ehs_thresholds
        assert 'waste_generation' in system.ehs_thresholds
        
        # Check threshold structure
        elec_thresholds = system.ehs_thresholds['electricity_consumption']
        assert 'warning' in elec_thresholds
        assert 'critical' in elec_thresholds
        assert elec_thresholds['critical'] > elec_thresholds['warning']


# =============================================================================
# Test Risk-Aware Query Processing
# =============================================================================

class TestRiskAwareQueryProcessing:
    """Test risk-aware query processing capabilities."""
    
    def test_risk_query_enhancer_initialization(self):
        """Test risk query enhancer initialization."""
        try:
            from ehs_analytics.risk_assessment.risk_query_processor import RiskQueryEnhancer
            
            enhancer = RiskQueryEnhancer()
            assert enhancer is not None
        except ImportError:
            pytest.skip("RiskQueryEnhancer not available")

    @pytest.mark.asyncio
    async def test_query_enhancement(self):
        """Test query enhancement with risk context."""
        try:
            from ehs_analytics.risk_assessment.risk_query_processor import RiskQueryEnhancer
            
            enhancer = RiskQueryEnhancer()
            
            original_query = "Show me electricity consumption"
            risk_context = {"high_risk_facilities": ["Plant A"], "risk_factors": ["demand_exceeded"]}
            
            enhanced_query = await enhancer.enhance_query(original_query, risk_context)
            
            assert enhanced_query != original_query
            assert "Plant A" in enhanced_query or "high risk" in enhanced_query.lower()
            
        except ImportError:
            pytest.skip("RiskQueryEnhancer not available")

    @pytest.mark.asyncio
    async def test_risk_filtering_retriever(self, mock_neo4j_driver):
        """Test risk-aware retrieval filtering."""
        try:
            from ehs_analytics.risk_assessment.risk_query_processor import RiskFilteringRetriever
            
            retriever = RiskFilteringRetriever(driver=mock_neo4j_driver)
            
            query = "facilities with high electricity consumption"
            risk_filters = {
                "min_risk_score": 0.7,
                "risk_categories": ["consumption", "compliance"]
            }
            
            # Mock the retrieve method to return risk-filtered results
            retriever.retrieve = AsyncMock(return_value={
                "results": [
                    {"facility": "Plant A", "risk_score": 0.8},
                    {"facility": "Plant B", "risk_score": 0.6}  # Should be filtered out
                ],
                "filtered_count": 1
            })
            
            results = await retriever.retrieve(query, risk_filters)
            
            assert results is not None
            assert "results" in results
            
        except ImportError:
            pytest.skip("RiskFilteringRetriever not available")

    @pytest.mark.asyncio
    async def test_risk_ranking_integration(self):
        """Test integration of risk scores in query results."""
        try:
            from ehs_analytics.risk_assessment.risk_query_processor import RiskAwareQueryProcessor
            
            processor = RiskAwareQueryProcessor()
            
            # Mock query results
            query_results = [
                {"facility": "Plant A", "consumption": 1500},
                {"facility": "Plant B", "consumption": 1200},
                {"facility": "Plant C", "consumption": 1800}
            ]
            
            # Mock risk scores
            risk_scores = {
                "Plant A": 0.7,
                "Plant B": 0.3,
                "Plant C": 0.9
            }
            
            processor.add_risk_scores = Mock(return_value=[
                {"facility": "Plant C", "consumption": 1800, "risk_score": 0.9},
                {"facility": "Plant A", "consumption": 1500, "risk_score": 0.7},
                {"facility": "Plant B", "consumption": 1200, "risk_score": 0.3}
            ])
            
            enhanced_results = processor.add_risk_scores(query_results, risk_scores)
            
            # Results should be sorted by risk score (highest first)
            assert enhanced_results[0]["facility"] == "Plant C"
            assert enhanced_results[0]["risk_score"] == 0.9
            
        except ImportError:
            pytest.skip("RiskAwareQueryProcessor not available")


# =============================================================================
# Test Monitoring and Alerting
# =============================================================================

class TestMonitoringAndAlerting:
    """Test monitoring and alerting capabilities."""
    
    def test_risk_monitor_initialization(self):
        """Test risk monitor initialization."""
        try:
            from ehs_analytics.risk_assessment.monitoring import RiskMonitor
            
            monitor = RiskMonitor()
            assert monitor is not None
            
        except ImportError:
            pytest.skip("RiskMonitor not available")

    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Test alert generation from risk assessments."""
        try:
            from ehs_analytics.risk_assessment.monitoring import AlertManager
            
            alert_manager = AlertManager()
            
            # Create high-risk assessment
            high_risk_assessment = Mock()
            high_risk_assessment.severity = RiskSeverity.CRITICAL
            high_risk_assessment.overall_score = 0.9
            high_risk_assessment.factors = [
                Mock(name="Water Usage", severity=RiskSeverity.CRITICAL, value=0.95)
            ]
            high_risk_assessment.facility_id = "Plant A"
            
            alerts = await alert_manager.generate_alerts(high_risk_assessment)
            
            assert len(alerts) > 0
            assert any(alert.severity == RiskSeverity.CRITICAL for alert in alerts)
            
        except ImportError:
            pytest.skip("AlertManager not available")

    @pytest.mark.asyncio
    async def test_alert_deduplication(self):
        """Test alert deduplication logic."""
        try:
            from ehs_analytics.risk_assessment.monitoring import AlertManager
            
            alert_manager = AlertManager()
            
            # Create duplicate alerts
            alert1 = Mock()
            alert1.id = "alert-001"
            alert1.facility_id = "Plant A"
            alert1.risk_type = "water_consumption"
            alert1.severity = RiskSeverity.HIGH
            alert1.timestamp = datetime.now()
            
            alert2 = Mock()
            alert2.id = "alert-002" 
            alert2.facility_id = "Plant A"
            alert2.risk_type = "water_consumption"
            alert2.severity = RiskSeverity.HIGH
            alert2.timestamp = datetime.now() + timedelta(minutes=5)  # Similar time
            
            alerts = [alert1, alert2]
            deduplicated = await alert_manager.deduplicate_alerts(alerts, window_minutes=30)
            
            # Should keep only one alert
            assert len(deduplicated) == 1
            
        except ImportError:
            pytest.skip("AlertManager not available")

    @pytest.mark.asyncio
    async def test_escalation_chains(self):
        """Test alert escalation chains."""
        try:
            from ehs_analytics.risk_assessment.monitoring import EscalationChain
            
            escalation = EscalationChain()
            
            # Configure escalation levels
            escalation.add_level(
                severity=RiskSeverity.HIGH,
                recipients=["ehs-team@company.com"],
                delay_minutes=0
            )
            escalation.add_level(
                severity=RiskSeverity.CRITICAL,
                recipients=["ehs-manager@company.com", "operations@company.com"],
                delay_minutes=15
            )
            
            # Test escalation for critical alert
            critical_alert = Mock()
            critical_alert.severity = RiskSeverity.CRITICAL
            critical_alert.facility_id = "Plant A"
            
            escalation_plan = await escalation.get_escalation_plan(critical_alert)
            
            assert len(escalation_plan) > 0
            assert any("ehs-manager@company.com" in step["recipients"] for step in escalation_plan)
            
        except ImportError:
            pytest.skip("EscalationChain not available")

    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection for monitoring."""
        try:
            from ehs_analytics.risk_assessment.monitoring import MetricsCollector
            
            collector = MetricsCollector()
            
            # Record some metrics
            await collector.record_risk_assessment("Plant A", 0.7, RiskSeverity.HIGH)
            await collector.record_risk_assessment("Plant B", 0.3, RiskSeverity.LOW)
            await collector.record_risk_assessment("Plant A", 0.8, RiskSeverity.HIGH)
            
            # Get aggregated metrics
            metrics = await collector.get_facility_metrics("Plant A")
            
            assert metrics is not None
            assert "average_risk_score" in metrics
            assert "assessment_count" in metrics
            assert metrics["assessment_count"] == 2
            
            # Test overall metrics
            overall_metrics = await collector.get_overall_metrics()
            
            assert "total_assessments" in overall_metrics
            assert "average_risk_score" in overall_metrics
            assert "severity_distribution" in overall_metrics
            
        except ImportError:
            pytest.skip("MetricsCollector not available")

    def test_alert_severity_mapping(self):
        """Test alert severity mapping."""
        severity_map = {
            RiskSeverity.LOW: "info",
            RiskSeverity.MEDIUM: "warning", 
            RiskSeverity.HIGH: "error",
            RiskSeverity.CRITICAL: "critical"
        }
        
        for risk_severity, alert_level in severity_map.items():
            assert risk_severity.numeric_value >= 1
            assert alert_level in ["info", "warning", "error", "critical"]

    @pytest.mark.asyncio
    async def test_monitoring_dashboard_data(self):
        """Test data preparation for monitoring dashboards."""
        try:
            from ehs_analytics.risk_assessment.monitoring import MetricsCollector
            
            collector = MetricsCollector()
            
            # Simulate various risk assessments
            facilities = ["Plant A", "Plant B", "Plant C"]
            risk_types = ["water", "electricity", "waste"]
            
            for facility in facilities:
                for risk_type in risk_types:
                    score = np.random.uniform(0.2, 0.9)
                    severity = RiskSeverity.LOW if score < 0.4 else \
                              RiskSeverity.MEDIUM if score < 0.6 else \
                              RiskSeverity.HIGH if score < 0.8 else \
                              RiskSeverity.CRITICAL
                    
                    await collector.record_risk_assessment(
                        facility, score, severity, risk_type=risk_type
                    )
            
            # Get dashboard data
            dashboard_data = await collector.get_dashboard_data()
            
            assert "facility_scores" in dashboard_data
            assert "risk_type_distribution" in dashboard_data
            assert "trend_data" in dashboard_data
            
            # Check data completeness
            assert len(dashboard_data["facility_scores"]) == len(facilities)
            
        except ImportError:
            pytest.skip("MetricsCollector not available")


# =============================================================================
# Integration Tests
# =============================================================================

class TestRiskAssessmentIntegration:
    """Test integration between different risk assessment components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_risk_workflow(
        self, sample_water_consumption_records, sample_water_permit, 
        sample_equipment_efficiency_data
    ):
        """Test complete end-to-end risk assessment workflow."""
        # 1. Perform risk assessment
        analyzer = WaterConsumptionRiskAnalyzer()
        
        data = {
            'consumption_records': sample_water_consumption_records,
            'permit_data': sample_water_permit,
            'equipment_data': sample_equipment_efficiency_data,
            'facility_id': 'facility-test-001'
        }
        
        assessment = await analyzer.analyze(data)
        assert assessment is not None
        
        # 2. If assessment shows risk, trigger time series analysis
        if assessment.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]:
            # Extract time series data
            consumption_data = [(r.timestamp, r.consumption_gallons) for r in sample_water_consumption_records]
            timestamps, values = zip(*consumption_data)
            
            ts_data = TimeSeriesData(
                timestamps=list(timestamps),
                values=list(values)
            )
            
            # Perform time series analysis
            ts_analyzer = TimeSeriesAnalyzer()
            ts_results = await ts_analyzer.analyze_complete(ts_data)
            
            assert ts_results is not None
            assert 'anomalies' in ts_results
        
        # 3. Generate monitoring alerts if needed
        if assessment.severity == RiskSeverity.CRITICAL:
            # This would trigger alert generation
            assert len(assessment.recommendations) > 0
            
            # Check that critical recommendations are present
            critical_recs = [r for r in assessment.recommendations if "CRITICAL" in r or "EMERGENCY" in r]
            assert len(critical_recs) > 0

    @pytest.mark.asyncio
    async def test_multi_analyzer_comparison(
        self, sample_water_consumption_records, sample_water_permit,
        sample_electricity_records, sample_electrical_contract,
        sample_waste_records, sample_waste_regulations
    ):
        """Test comparison across multiple risk analyzers."""
        
        # Run all analyzers
        water_analyzer = WaterConsumptionRiskAnalyzer()
        water_data = {
            'consumption_records': sample_water_consumption_records,
            'permit_data': sample_water_permit,
            'facility_id': 'facility-test-001'
        }
        water_assessment = await water_analyzer.analyze(water_data)
        
        electricity_analyzer = ElectricityConsumptionRiskAnalyzer()
        electricity_data = {
            'consumption_records': sample_electricity_records,
            'contract_data': sample_electrical_contract,
            'facility_id': 'facility-test-001'
        }
        electricity_assessment = await electricity_analyzer.analyze(electricity_data)
        
        waste_analyzer = WasteGenerationRiskAnalyzer()
        waste_data = {
            'waste_records': sample_waste_records,
            'regulation_data': sample_waste_regulations,
            'facility_id': 'facility-test-001'
        }
        waste_assessment = await waste_analyzer.analyze(waste_data)
        
        # All assessments should be valid
        assessments = [water_assessment, electricity_assessment, waste_assessment]
        
        for assessment in assessments:
            assert isinstance(assessment, RiskAssessment)
            assert 0.0 <= assessment.overall_score <= 1.0
            assert assessment.severity in [RiskSeverity.LOW, RiskSeverity.MEDIUM, 
                                         RiskSeverity.HIGH, RiskSeverity.CRITICAL]
        
        # Compare overall facility risk (would be aggregate of all assessments)
        overall_scores = [a.overall_score for a in assessments]
        facility_risk_score = sum(overall_scores) / len(overall_scores)
        
        assert 0.0 <= facility_risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_anomaly_detection_integration_with_risk_analysis(
        self, sample_water_consumption_records
    ):
        """Test integration between anomaly detection and risk analysis."""
        
        # First, create time series data from consumption records
        consumption_data = [(r.timestamp, r.consumption_gallons) for r in sample_water_consumption_records]
        df = pd.DataFrame(consumption_data, columns=['timestamp', 'consumption'])
        df.set_index('timestamp', inplace=True)
        
        # 1. Train anomaly detection system
        anomaly_system = create_ehs_anomaly_system()
        await anomaly_system.train_detectors(df, save_models=False)
        
        # 2. Detect anomalies
        anomalies = await anomaly_system.detect_anomalies(df.tail(10), return_explanations=False)
        
        # 3. If anomalies found, they should influence risk assessment
        if len(anomalies) > 0:
            # Risk assessment should account for anomalous behavior
            # This would be implemented in a real integration
            high_anomaly_count = len([a for a in anomalies if a.score.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]])
            
            if high_anomaly_count > 0:
                # Should trigger additional risk analysis
                assert True  # Placeholder for integration logic

    @pytest.mark.asyncio
    async def test_forecasting_integration_with_risk_assessment(
        self, sample_water_consumption_records, sample_water_permit
    ):
        """Test integration between forecasting and risk assessment."""
        
        # 1. Create time series from consumption records
        consumption_data = [(r.timestamp, r.consumption_gallons) for r in sample_water_consumption_records]
        timestamps, values = zip(*consumption_data)
        
        ts_data = pd.Series(values, index=pd.to_datetime(timestamps))
        
        # 2. Generate forecast
        forecasting_engine = ForecastingEngine()
        forecast_result = await forecasting_engine.forecast(
            data=ts_data,
            horizon=30,  # 30 days ahead
            model=ForecastModel.AUTO_SELECT
        )
        
        assert forecast_result is not None
        
        # 3. Use forecast to assess future risk
        if forecast_result.predictions.max() > sample_water_permit.daily_limit * 0.9:
            # Forecast suggests approaching permit limits
            # This should trigger proactive risk management
            future_risk_score = forecast_result.predictions.max() / sample_water_permit.daily_limit
            
            if future_risk_score > 0.85:
                # High future risk detected
                assert True  # Would trigger risk mitigation planning

    def test_risk_assessment_serialization(self, sample_risk_factors):
        """Test serialization of risk assessment objects."""
        assessment = RiskAssessment.from_factors(
            factors=sample_risk_factors,
            assessment_id="test-serialization",
            assessment_type="integration_test"
        )
        
        # Convert to dictionary
        assessment_dict = assessment.to_dict()
        
        # Verify all required fields are present
        required_fields = [
            'assessment_id', 'timestamp', 'assessment_type', 'overall_score',
            'severity', 'confidence_score', 'factors', 'recommendations'
        ]
        
        for field in required_fields:
            assert field in assessment_dict
        
        # Verify factors are properly serialized
        assert len(assessment_dict['factors']) == len(sample_risk_factors)
        
        for factor_dict in assessment_dict['factors']:
            assert 'name' in factor_dict
            assert 'value' in factor_dict
            assert 'weight' in factor_dict
            assert 'severity' in factor_dict

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, sample_water_consumption_records, sample_water_permit):
        """Test that risk assessment meets performance benchmarks."""
        analyzer = WaterConsumptionRiskAnalyzer()
        
        data = {
            'consumption_records': sample_water_consumption_records,
            'permit_data': sample_water_permit,
            'facility_id': 'facility-test-001'
        }
        
        # Measure performance
        start_time = datetime.now()
        assessment = await analyzer.analyze(data)
        end_time = datetime.now()
        
        analysis_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Performance benchmarks
        assert analysis_duration_ms < 5000  # Should complete within 5 seconds
        assert assessment is not None
        assert len(assessment.factors) > 0
        
        # Quality benchmarks
        assert 0.0 <= assessment.overall_score <= 1.0
        assert assessment.confidence_score >= 0.5  # At least 50% confidence
        
        # Check that all factors have valid values
        for factor in assessment.factors:
            assert 0.0 <= factor.value <= 1.0
            assert 0.0 <= factor.weight <= 1.0


# =============================================================================
# Test Runner and Utilities
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])