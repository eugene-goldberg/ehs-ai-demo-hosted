#!/usr/bin/env python3
"""
Debug script for the risk assessment framework test.

This script isolates and runs only the framework test portion of the validation script
to help identify specific issues with the risk_framework.framework_test.

ISSUE IDENTIFIED: 
The RiskThresholds.critical_threshold is set to 1.0, which means only a perfect score
of 100% is considered CRITICAL. The test expects 0.95 (95%) to be CRITICAL, which is
more realistic for risk assessment frameworks.

RECOMMENDED FIX:
Change critical_threshold from 1.0 to 0.9 in the RiskThresholds class.
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Add src to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print("🔍 Debug: Risk Assessment Framework Test")
print("="*60)
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"Source path: {src_path}")
print("")

# Test imports one by one with detailed error reporting
print("📦 Testing imports...")

try:
    print("  → Importing base components...")
    from ehs_analytics.risk_assessment.base import (
        RiskSeverity, RiskFactor, RiskAssessment, BaseRiskAnalyzer, RiskThresholds
    )
    print("  ✅ Base components imported successfully")
except ImportError as e:
    print(f"  ❌ Base components import failed: {e}")
    print(f"  📋 Full traceback:")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Unexpected error importing base components: {e}")
    print(f"  📋 Full traceback:")
    traceback.print_exc()
    sys.exit(1)

print("")

def analyze_threshold_issue():
    """Analyze the threshold configuration issue."""
    print("🔍 Analyzing Threshold Configuration Issue")
    print("-" * 50)
    
    # Show current thresholds
    thresholds = RiskThresholds()
    print(f"Current thresholds:")
    print(f"  → Low threshold:      {thresholds.low_threshold}")
    print(f"  → Medium threshold:   {thresholds.medium_threshold}")
    print(f"  → High threshold:     {thresholds.high_threshold}")
    print(f"  → Critical threshold: {thresholds.critical_threshold}")
    print("")
    
    # Test various scores with current thresholds
    test_scores = [0.1, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    print("Score mappings with current thresholds:")
    for score in test_scores:
        severity = thresholds.get_severity(score)
        print(f"  → Score {score:4.2f}: {severity.value.upper():8}")
    print("")
    
    # Show what thresholds would work better
    print("ISSUE: Current critical_threshold=1.0 means only perfect scores are critical!")
    print("RECOMMENDATION: Use critical_threshold=0.9 for more realistic assessment.")
    print("")
    
    # Test with recommended thresholds
    recommended_thresholds = RiskThresholds(
        low_threshold=0.25,
        medium_threshold=0.5,
        high_threshold=0.75,
        critical_threshold=0.9  # Changed from 1.0 to 0.9
    )
    
    print(f"Recommended thresholds:")
    print(f"  → Low threshold:      {recommended_thresholds.low_threshold}")
    print(f"  → Medium threshold:   {recommended_thresholds.medium_threshold}")
    print(f"  → High threshold:     {recommended_thresholds.high_threshold}")
    print(f"  → Critical threshold: {recommended_thresholds.critical_threshold}")
    print("")
    
    print("Score mappings with recommended thresholds:")
    for score in test_scores:
        severity = recommended_thresholds.get_severity(score)
        print(f"  → Score {score:4.2f}: {severity.value.upper():8}")
    print("")

def run_framework_test():
    """Run the exact framework test logic from the validation script."""
    print("🧪 Running Framework Tests...")
    print("-" * 40)
    
    errors = []
    test_results = {}
    
    try:
        # Test 1: RiskSeverity enum
        print("Test 1: RiskSeverity enum...")
        start_time = time.time()
        
        try:
            severity_low = RiskSeverity.LOW
            severity_high = RiskSeverity.HIGH
            
            print(f"  → LOW severity: {severity_low}")
            print(f"  → HIGH severity: {severity_high}")
            print(f"  → LOW < HIGH: {severity_low < severity_high}")
            print(f"  → LOW numeric value: {severity_low.numeric_value}")
            print(f"  → HIGH numeric value: {severity_high.numeric_value}")
            
            assert severity_low < severity_high, "LOW should be less than HIGH"
            assert severity_high.numeric_value > severity_low.numeric_value, "HIGH numeric value should be greater than LOW"
            
            execution_time = time.time() - start_time
            test_results['risk_severity_enum'] = {
                'success': True,
                'execution_time': execution_time,
                'details': {
                    'severity_levels': len(RiskSeverity),
                    'comparison_test': 'passed'
                }
            }
            print(f"  ✅ RiskSeverity enum test passed ({execution_time*1000:.1f}ms)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"RiskSeverity enum test failed: {e}"
            errors.append(error_msg)
            test_results['risk_severity_enum'] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"  ❌ {error_msg}")
            traceback.print_exc()
        
        print("")
        
        # Test 2: RiskThresholds (with current implementation)
        print("Test 2: RiskThresholds (current implementation)...")
        start_time = time.time()
        
        try:
            thresholds = RiskThresholds()
            print(f"  → RiskThresholds instance: {thresholds}")
            
            # Test threshold mappings
            low_result = thresholds.get_severity(0.1)
            medium_result = thresholds.get_severity(0.6)
            high_result = thresholds.get_severity(0.8)
            critical_result = thresholds.get_severity(0.95)
            
            print(f"  → get_severity(0.1): {low_result}")
            print(f"  → get_severity(0.6): {medium_result}")
            print(f"  → get_severity(0.8): {high_result}")
            print(f"  → get_severity(0.95): {critical_result}")
            
            assert low_result == RiskSeverity.LOW, f"Expected LOW, got {low_result}"
            assert medium_result == RiskSeverity.MEDIUM, f"Expected MEDIUM, got {medium_result}"
            assert high_result == RiskSeverity.HIGH, f"Expected HIGH, got {high_result}"
            
            # This is the failing assertion - expecting CRITICAL but getting HIGH
            print(f"  ⚠️  Expected CRITICAL for 0.95, but got {critical_result}")
            print(f"  ⚠️  This fails because critical_threshold={thresholds.critical_threshold}")
            
            # Don't assert this one since we know it fails
            # assert critical_result == RiskSeverity.CRITICAL, f"Expected CRITICAL, got {critical_result}"
            
            execution_time = time.time() - start_time
            test_results['risk_thresholds'] = {
                'success': False,  # Mark as failed due to threshold issue
                'execution_time': execution_time,
                'details': {
                    'threshold_mapping': 'fails on critical threshold',
                    'issue': f'critical_threshold={thresholds.critical_threshold} too high'
                }
            }
            print(f"  ❌ RiskThresholds test failed due to threshold configuration ({execution_time*1000:.1f}ms)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"RiskThresholds test failed: {e}"
            errors.append(error_msg)
            test_results['risk_thresholds'] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"  ❌ {error_msg}")
            traceback.print_exc()
        
        print("")
        
        # Test 2b: RiskThresholds (with recommended fix)
        print("Test 2b: RiskThresholds (with recommended fix)...")
        start_time = time.time()
        
        try:
            # Test with corrected thresholds
            fixed_thresholds = RiskThresholds(
                low_threshold=0.25,
                medium_threshold=0.5,
                high_threshold=0.75,
                critical_threshold=0.9  # Fixed threshold
            )
            print(f"  → Fixed RiskThresholds instance: {fixed_thresholds}")
            
            # Test threshold mappings with fixed thresholds
            low_result = fixed_thresholds.get_severity(0.1)
            medium_result = fixed_thresholds.get_severity(0.6)
            high_result = fixed_thresholds.get_severity(0.8)
            critical_result = fixed_thresholds.get_severity(0.95)
            
            print(f"  → get_severity(0.1): {low_result}")
            print(f"  → get_severity(0.6): {medium_result}")
            print(f"  → get_severity(0.8): {high_result}")
            print(f"  → get_severity(0.95): {critical_result}")
            
            assert low_result == RiskSeverity.LOW, f"Expected LOW, got {low_result}"
            assert medium_result == RiskSeverity.MEDIUM, f"Expected MEDIUM, got {medium_result}"
            assert high_result == RiskSeverity.HIGH, f"Expected HIGH, got {high_result}"
            assert critical_result == RiskSeverity.CRITICAL, f"Expected CRITICAL, got {critical_result}"
            
            execution_time = time.time() - start_time
            test_results['risk_thresholds_fixed'] = {
                'success': True,
                'execution_time': execution_time,
                'details': {
                    'threshold_mapping': 'correct with fixed critical_threshold=0.9'
                }
            }
            print(f"  ✅ Fixed RiskThresholds test passed ({execution_time*1000:.1f}ms)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Fixed RiskThresholds test failed: {e}"
            errors.append(error_msg)
            test_results['risk_thresholds_fixed'] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"  ❌ {error_msg}")
            traceback.print_exc()
        
        print("")
        
        # Test 3: RiskFactor
        print("Test 3: RiskFactor...")
        start_time = time.time()
        
        try:
            factor = RiskFactor(
                name="Test Factor",
                value=0.7,
                weight=0.5,
                severity=RiskSeverity.HIGH,
                description="Test risk factor"
            )
            print(f"  → RiskFactor instance: {factor}")
            print(f"  → Weighted score: {factor.weighted_score}")
            print(f"  → Expected weighted score: 0.35 (0.7 * 0.5)")
            
            assert factor.weighted_score == 0.35, f"Expected 0.35, got {factor.weighted_score}"
            
            factor_dict = factor.to_dict()
            print(f"  → to_dict() keys: {list(factor_dict.keys())}")
            
            assert 'name' in factor_dict, "Missing 'name' in dict"
            assert 'weighted_score' in factor_dict, "Missing 'weighted_score' in dict"
            
            execution_time = time.time() - start_time
            test_results['risk_factor'] = {
                'success': True,
                'execution_time': execution_time,
                'details': {
                    'weighted_score': factor.weighted_score,
                    'serialization': 'successful'
                }
            }
            print(f"  ✅ RiskFactor test passed ({execution_time*1000:.1f}ms)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"RiskFactor test failed: {e}"
            errors.append(error_msg)
            test_results['risk_factor'] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"  ❌ {error_msg}")
            traceback.print_exc()
        
        print("")
        
        # Test 4: RiskAssessment
        print("Test 4: RiskAssessment...")
        start_time = time.time()
        
        try:
            factors = [
                RiskFactor("Factor1", 0.6, 0.4, RiskSeverity.MEDIUM),
                RiskFactor("Factor2", 0.8, 0.6, RiskSeverity.HIGH)
            ]
            print(f"  → Created {len(factors)} test factors")
            
            assessment = RiskAssessment.from_factors(factors, assessment_type="test")
            print(f"  → RiskAssessment instance: {assessment}")
            print(f"  → Overall score: {assessment.overall_score}")
            print(f"  → Number of factors: {len(assessment.factors)}")
            
            assert assessment.overall_score > 0, f"Overall score should be > 0, got {assessment.overall_score}"
            assert len(assessment.factors) == 2, f"Expected 2 factors, got {len(assessment.factors)}"
            
            critical_factors = assessment.get_critical_factors()
            high_risk_factors = assessment.get_high_risk_factors()
            print(f"  → Critical factors: {len(critical_factors)}")
            print(f"  → High risk factors: {len(high_risk_factors)}")
            
            execution_time = time.time() - start_time
            test_results['risk_assessment'] = {
                'success': True,
                'execution_time': execution_time,
                'details': {
                    'overall_score': assessment.overall_score,
                    'factor_count': len(assessment.factors),
                    'high_risk_count': len(high_risk_factors)
                }
            }
            print(f"  ✅ RiskAssessment test passed ({execution_time*1000:.1f}ms)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"RiskAssessment test failed: {e}"
            errors.append(error_msg)
            test_results['risk_assessment'] = {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
            print(f"  ❌ {error_msg}")
            traceback.print_exc()
        
    except Exception as e:
        error_msg = f"Framework test execution failed: {e}"
        errors.append(error_msg)
        print(f"❌ {error_msg}")
        traceback.print_exc()
    
    return test_results, errors

def main():
    """Main debug function."""
    print("🚀 Starting framework test debug...")
    print("")
    
    # First, analyze the threshold issue
    analyze_threshold_issue()
    
    # Then run the framework tests
    test_results, errors = run_framework_test()
    
    print("")
    print("📊 Test Results Summary")
    print("="*60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result['success'])
    
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Failed tests: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
    print("")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        time_str = f"{result['execution_time']*1000:.1f}ms" if result['execution_time'] < 1 else f"{result['execution_time']:.2f}s"
        print(f"{status} {test_name:25} ({time_str})")
        
        if not result['success'] and 'error' in result:
            print(f"     Error: {result['error']}")
    
    print("")
    print("🔧 DIAGNOSIS:")
    print("-" * 40)
    print("The framework test fails because:")
    print("1. RiskThresholds.critical_threshold is set to 1.0 (100%)")
    print("2. The test expects score 0.95 (95%) to be CRITICAL")
    print("3. With current thresholds, 0.95 maps to HIGH, not CRITICAL")
    print("")
    print("🔧 RECOMMENDED FIX:")
    print("-" * 40)
    print("In src/ehs_analytics/risk_assessment/base.py, line 60:")
    print("Change:    critical_threshold: float = 1.0")
    print("To:        critical_threshold: float = 0.9")
    print("")
    print("This makes scores ≥90% critical, which is more realistic for risk assessment.")
    
    if errors:
        print("")
        print("❌ Errors Found:")
        print("-" * 40)
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
    
    print("")
    
    # Check if core functionality works (excluding the threshold config issue)
    core_tests = ['risk_severity_enum', 'risk_factor', 'risk_assessment', 'risk_thresholds_fixed']
    core_passed = sum(1 for test in core_tests if test_results.get(test, {}).get('success', False))
    
    if core_passed == len(core_tests):
        print("🎉 Core framework functionality works! Only threshold configuration needs fixing.")
        return 0
    else:
        print("⚠️  Some core framework functionality failed. See detailed output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)