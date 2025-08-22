#!/usr/bin/env python3
"""
Debug Script for Time Series Test Failures

This script identifies issues with time series analysis by testing all components
step by step with detailed error reporting and debugging information.
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'debug_time_series_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print(f"🔍 Time Series Debug Script Started")
print(f"📁 Project root: {project_root}")
print(f"📁 Source path: {src_path}")
print(f"🐍 Python version: {sys.version}")

def debug_dependencies():
    """Check all required dependencies."""
    print("\n" + "="*60)
    print("🔍 CHECKING DEPENDENCIES")
    print("="*60)
    
    required_deps = ['numpy', 'pandas', 'scipy', 'statsmodels']
    optional_deps = ['prophet', 'scikit-learn']
    
    for dep in required_deps:
        try:
            module = __import__(dep.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep}: {version}")
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            return False
    
    for dep in optional_deps:
        try:
            module = __import__(dep.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            print(f"🟡 {dep}: {version}")
        except ImportError:
            print(f"⚠️ {dep}: not available (optional)")
    
    return True

def debug_imports():
    """Test importing time series components."""
    print("\n" + "="*60)
    print("🔍 TESTING IMPORTS")
    print("="*60)
    
    import_results = {}
    
    # Test base imports
    try:
        from ehs_analytics.risk_assessment.base import (
            RiskSeverity, RiskFactor, RiskAssessment, BaseRiskAnalyzer, RiskThresholds
        )
        print("✅ Base components imported successfully")
        import_results['base'] = True
    except ImportError as e:
        print(f"❌ Base components import failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        import_results['base'] = False
    
    # Test time series imports
    try:
        from ehs_analytics.risk_assessment.time_series import (
            TimeSeriesAnalyzer, TimeSeriesData, TimeSeriesPredictor
        )
        print("✅ Time series components imported successfully")
        import_results['time_series'] = True
    except ImportError as e:
        print(f"❌ Time series components import failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        import_results['time_series'] = False
        return import_results
    
    # Test specific classes
    try:
        # Test TimeSeriesData
        test_data = TimeSeriesData(
            timestamps=[datetime.now() - timedelta(days=1), datetime.now()],
            values=[100.0, 105.0]
        )
        print("✅ TimeSeriesData instantiation successful")
        
        # Test TimeSeriesAnalyzer
        analyzer = TimeSeriesAnalyzer()
        print("✅ TimeSeriesAnalyzer instantiation successful")
        
        import_results['instantiation'] = True
        
    except Exception as e:
        print(f"❌ Component instantiation failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        import_results['instantiation'] = False
    
    return import_results

def generate_test_data(length: int = 365) -> 'TimeSeriesData':
    """Generate test time series data with same logic as validation script."""
    print(f"\n🔍 Generating test data with {length} points...")
    
    from ehs_analytics.risk_assessment.time_series import TimeSeriesData
    
    base_date = datetime.now() - timedelta(days=length)
    timestamps = [base_date + timedelta(days=i) for i in range(length)]
    
    # Generate realistic time series with trend, seasonality, and noise
    values = []
    for i in range(length):
        # Trend component
        trend = 1000 + (i / length) * 500
        
        # Seasonal component (annual cycle)
        seasonal = 200 * np.sin(2 * np.pi * i / 365.25)
        
        # Weekly seasonality
        weekly = 50 * np.sin(2 * np.pi * i / 7)
        
        # Random noise
        noise = np.random.normal(0, 50)
        
        value = trend + seasonal + weekly + noise
        
        # Add some anomalies at specific points
        if i in [100, 200, 300]:
            value *= 1.5  # Spike anomalies
        elif i in [150, 250]:
            value *= 0.5  # Dip anomalies
            
        values.append(max(0, value))  # Ensure non-negative
    
    print(f"✅ Generated {len(values)} data points")
    print(f"📊 Value range: {min(values):.2f} to {max(values):.2f}")
    
    return TimeSeriesData(
        timestamps=timestamps,
        values=values,
        metadata={'generator': 'debug_script', 'anomalies': [100, 150, 200, 250, 300]}
    )

async def debug_analyzer_methods(analyzer: 'TimeSeriesAnalyzer', data: 'TimeSeriesData'):
    """Test individual analyzer methods."""
    print("\n" + "="*60)
    print("🔍 TESTING ANALYZER METHODS")
    print("="*60)
    
    test_results = {}
    
    # Test data quality assessment
    try:
        print("🔍 Testing data quality assessment...")
        quality_report = await analyzer.assess_data_quality(data)
        print(f"✅ Data quality assessment completed")
        print(f"📊 Overall quality score: {quality_report.overall_quality_score:.3f}")
        print(f"📊 Missing values: {quality_report.missing_values}")
        print(f"📊 Outlier count: {quality_report.outlier_count}")
        test_results['data_quality'] = True
    except Exception as e:
        print(f"❌ Data quality assessment failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        test_results['data_quality'] = False
    
    # Test trend analysis - this is the method called in the validation script
    try:
        print("\n🔍 Testing analyze_trend method (main test)...")
        trend_analysis = await analyzer.analyze_trend(data)
        print(f"✅ Trend analysis completed")
        print(f"📊 Direction: {trend_analysis.direction.value}")
        print(f"📊 Slope: {trend_analysis.slope:.6f}")
        print(f"📊 P-value: {trend_analysis.p_value:.6f}")
        print(f"📊 R-squared: {trend_analysis.r_squared:.6f}")
        print(f"📊 Is significant: {trend_analysis.is_significant}")
        test_results['analyze_trend'] = True
    except Exception as e:
        print(f"❌ Trend analysis failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        test_results['analyze_trend'] = False
    
    # Test alternative trend detection method
    try:
        print("\n🔍 Testing detect_trend method (alternative)...")
        trend_analysis2 = await analyzer.detect_trend(data)
        print(f"✅ Alternative trend detection completed")
        print(f"📊 Direction: {trend_analysis2.direction.value}")
        test_results['detect_trend'] = True
    except Exception as e:
        print(f"❌ Alternative trend detection failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        test_results['detect_trend'] = False
    
    # Test seasonal decomposition
    try:
        print("\n🔍 Testing seasonal decomposition...")
        seasonal_components = await analyzer.decompose_seasonal(data)
        if seasonal_components:
            print(f"✅ Seasonal decomposition completed")
            print(f"📊 Seasonal strength: {seasonal_components.seasonal_strength:.3f}")
            print(f"📊 Trend strength: {seasonal_components.trend_strength:.3f}")
            print(f"📊 Has strong seasonality: {seasonal_components.has_strong_seasonality}")
        else:
            print("⚠️ Seasonal decomposition returned None (insufficient data)")
        test_results['seasonal_decomposition'] = True
    except Exception as e:
        print(f"❌ Seasonal decomposition failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        test_results['seasonal_decomposition'] = False
    
    # Test anomaly detection
    try:
        print("\n🔍 Testing anomaly detection...")
        anomalies = await analyzer.detect_anomalies(data, method='statistical')
        print(f"✅ Anomaly detection completed")
        print(f"📊 Anomalies found: {anomalies.count}")
        print(f"📊 Anomaly rate: {anomalies.anomaly_rate:.2f}%")
        if anomalies.indices:
            print(f"📊 First few anomaly indices: {anomalies.indices[:5]}")
        test_results['anomaly_detection'] = True
    except Exception as e:
        print(f"❌ Anomaly detection failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        test_results['anomaly_detection'] = False
    
    # Test change point detection
    try:
        print("\n🔍 Testing change point detection...")
        changepoints = await analyzer.detect_changepoints(data)
        print(f"✅ Change point detection completed")
        print(f"📊 Change points found: {len(changepoints)}")
        if changepoints:
            print(f"📊 Top change point confidence: {changepoints[0].confidence:.3f}")
        test_results['changepoint_detection'] = True
    except Exception as e:
        print(f"❌ Change point detection failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        test_results['changepoint_detection'] = False
    
    # Test complete analysis
    try:
        print("\n🔍 Testing complete analysis...")
        complete_results = await analyzer.analyze_complete(data)
        print(f"✅ Complete analysis completed")
        print(f"📊 Results keys: {list(complete_results.keys())}")
        test_results['complete_analysis'] = True
    except Exception as e:
        print(f"❌ Complete analysis failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        test_results['complete_analysis'] = False
    
    return test_results

def debug_data_validation(data: 'TimeSeriesData'):
    """Test data validation and properties."""
    print("\n" + "="*60)
    print("🔍 TESTING DATA VALIDATION")
    print("="*60)
    
    try:
        print(f"📊 Data length: {data.length}")
        print(f"📊 First timestamp: {data.timestamps[0]}")
        print(f"📊 Last timestamp: {data.timestamps[-1]}")
        print(f"📊 First value: {data.values[0]:.2f}")
        print(f"📊 Last value: {data.values[-1]:.2f}")
        print(f"📊 Metadata: {data.metadata}")
        
        # Test DataFrame conversion
        df = data.df
        print(f"✅ DataFrame conversion successful")
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📊 DataFrame columns: {list(df.columns)}")
        
        # Test basic statistics
        print(f"📊 Mean value: {df['value'].mean():.2f}")
        print(f"📊 Std deviation: {df['value'].std():.2f}")
        print(f"📊 Min value: {df['value'].min():.2f}")
        print(f"📊 Max value: {df['value'].max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main debug function."""
    print("🚀 Starting comprehensive time series debug...")
    
    try:
        # Check dependencies
        if not debug_dependencies():
            print("❌ Dependency check failed - aborting")
            return
        
        # Import numpy here after dependency check
        global np
        import numpy as np
        
        # Test imports
        import_results = debug_imports()
        if not import_results.get('time_series', False):
            print("❌ Time series imports failed - aborting")
            return
        
        # Import components after successful import test
        from ehs_analytics.risk_assessment.time_series import TimeSeriesAnalyzer, TimeSeriesData
        
        # Generate test data
        print("\n🔍 Generating test data...")
        test_data = generate_test_data(365)
        
        # Validate data
        if not debug_data_validation(test_data):
            print("❌ Data validation failed - aborting")
            return
        
        # Initialize analyzer
        print("\n🔍 Initializing TimeSeriesAnalyzer...")
        analyzer = TimeSeriesAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Test analyzer methods
        test_results = await debug_analyzer_methods(analyzer, test_data)
        
        # Summary
        print("\n" + "="*60)
        print("📋 SUMMARY")
        print("="*60)
        
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        print(f"🧪 Total tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}")
        
        for test_name, result in test_results.items():
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")
        
        if not test_results.get('analyze_trend', False):
            print("\n🚨 CRITICAL: analyze_trend method failed!")
            print("🔍 This is the method called in the validation script.")
            print("🔧 Check the error details above for the root cause.")
        else:
            print("\n🎉 SUCCESS: analyze_trend method works correctly!")
            print("🔍 The time series analysis should work in the validation script.")
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR in debug script: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())