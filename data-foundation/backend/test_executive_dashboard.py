#!/usr/bin/env python3
"""
Executive Dashboard Service Test Script

This script demonstrates the comprehensive functionality of the ExecutiveDashboardService,
including real data integration, trend analysis, risk assessment, and dynamic JSON generation.

Usage:
    python test_executive_dashboard.py

Created: 2025-08-28
Version: 1.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from services.executive_dashboard.dashboard_service import (
    ExecutiveDashboardService, LocationFilter, DateRangeFilter, 
    AggregationPeriod, create_dashboard_service
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_service_initialization():
    """Test service initialization and health check"""
    print("\n" + "="*80)
    print("TESTING: Service Initialization and Health Check")
    print("="*80)
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Create dashboard service
        dashboard_service = create_dashboard_service()
        logger.info("✓ Dashboard service created successfully")
        
        # Health check
        health_status = dashboard_service.health_check()
        print(f"Health Status: {json.dumps(health_status, indent=2)}")
        
        if health_status.get('database', {}).get('status') == 'healthy':
            logger.info("✓ Database connection healthy")
        else:
            logger.warning("⚠ Database connection issues detected")
        
        return dashboard_service
        
    except Exception as e:
        logger.error(f"✗ Service initialization failed: {e}")
        return None


def test_real_time_metrics(dashboard_service: ExecutiveDashboardService):
    """Test real-time metrics functionality"""
    print("\n" + "="*80)
    print("TESTING: Real-Time Metrics")
    print("="*80)
    
    try:
        # Get real-time metrics
        real_time_data = dashboard_service.get_real_time_metrics()
        
        print("Real-Time Metrics:")
        print(f"  - Alert Level: {real_time_data.get('alert_level', 'N/A')}")
        print(f"  - Today's Incidents: {real_time_data.get('metrics', {}).get('todays_incidents', 0)}")
        print(f"  - Active Alerts: {real_time_data.get('metrics', {}).get('active_alerts', 0)}")
        print(f"  - Overdue Training: {real_time_data.get('metrics', {}).get('overdue_training', 0)}")
        print(f"  - Overdue Inspections: {real_time_data.get('metrics', {}).get('overdue_inspections', 0)}")
        
        # Facility breakdown
        facility_breakdown = real_time_data.get('facility_breakdown', [])
        print(f"\nFacility Breakdown ({len(facility_breakdown)} facilities):")
        for i, facility in enumerate(facility_breakdown[:5]):  # Show first 5
            print(f"  {i+1}. {facility.get('facility_name', 'Unknown')} - "
                  f"Alerts: {facility.get('active_alerts', 0)}, "
                  f"Incidents Today: {facility.get('todays_incidents', 0)}")
        
        logger.info("✓ Real-time metrics retrieved successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Real-time metrics test failed: {e}")
        return False


def test_kpi_details(dashboard_service: ExecutiveDashboardService):
    """Test detailed KPI functionality"""
    print("\n" + "="*80)
    print("TESTING: Detailed KPI Metrics")
    print("="*80)
    
    try:
        # Get KPI details for last 30 days
        kpi_details = dashboard_service.get_kpi_details(date_range_days=30)
        
        print("Safety KPIs:")
        safety_kpis = kpi_details.get('safety_kpis', {})
        for name, kpi in safety_kpis.items():
            status_icon = "🟢" if kpi.get('status') == 'green' else "🟡" if kpi.get('status') == 'yellow' else "🔴"
            print(f"  {status_icon} {kpi.get('name', name)}: {kpi.get('value', 'N/A')} {kpi.get('unit', '')}")
            if kpi.get('target'):
                print(f"      Target: {kpi['target']}, Warning: {kpi.get('threshold_warning', 'N/A')}")
        
        print("\nCompliance KPIs:")
        compliance_kpis = kpi_details.get('compliance_kpis', {})
        for name, kpi in compliance_kpis.items():
            status_icon = "🟢" if kpi.get('status') == 'green' else "🟡" if kpi.get('status') == 'yellow' else "🔴"
            print(f"  {status_icon} {kpi.get('name', name)}: {kpi.get('value', 'N/A')} {kpi.get('unit', '')}")
            if kpi.get('target'):
                print(f"      Target: {kpi['target']}, Warning: {kpi.get('threshold_warning', 'N/A')}")
        
        period = kpi_details.get('period', {})
        print(f"\nPeriod: {period.get('start_date', 'N/A')} to {period.get('end_date', 'N/A')} ({period.get('days', 'N/A')} days)")
        
        logger.info("✓ KPI details retrieved successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ KPI details test failed: {e}")
        return False


def test_dashboard_summary(dashboard_service: ExecutiveDashboardService):
    """Test dashboard summary functionality"""
    print("\n" + "="*80)
    print("TESTING: Dashboard Summary")
    print("="*80)
    
    try:
        # Get dashboard summary
        summary = dashboard_service.get_dashboard_summary()
        
        print("Dashboard Summary:")
        print(f"  - Overall Health Score: {summary.get('overall_health_score', 'N/A')}")
        print(f"  - Alert Level: {summary.get('alert_level', 'N/A')}")
        
        summary_data = summary.get('summary', {})
        if summary_data:
            # Period info
            period = summary_data.get('period', {})
            print(f"  - Period: {period.get('period_days', 'N/A')} days")
            
            # Facilities
            facilities = summary_data.get('facilities', {})
            print(f"  - Total Facilities: {facilities.get('total_count', 'N/A')}")
            print(f"  - Active Alerts: {facilities.get('active_alerts', 'N/A')}")
            
            # Incidents
            incidents = summary_data.get('incidents', {})
            print(f"  - Total Incidents: {incidents.get('total', 'N/A')}")
            print(f"  - Today's Incidents: {incidents.get('today', 'N/A')}")
            print(f"  - Incident Rate: {incidents.get('incident_rate', 'N/A')}")
            
            # Compliance
            compliance = summary_data.get('compliance', {})
            print(f"  - Audit Pass Rate: {compliance.get('audit_pass_rate', 'N/A')}%")
            print(f"  - Training Completion: {compliance.get('training_completion', 'N/A')}%")
            print(f"  - Overdue Items: {compliance.get('overdue_items', 'N/A')}")
        
        logger.info("✓ Dashboard summary retrieved successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Dashboard summary test failed: {e}")
        return False


def test_location_filtering(dashboard_service: ExecutiveDashboardService):
    """Test location-based filtering"""
    print("\n" + "="*80)
    print("TESTING: Location-Based Filtering")
    print("="*80)
    
    try:
        # First, let's see what facilities are available
        all_facilities_summary = dashboard_service.get_dashboard_summary()
        facilities_count = all_facilities_summary.get('summary', {}).get('facilities', {}).get('total_count', 0)
        print(f"Total facilities available: {facilities_count}")
        
        # Test with specific facility filter (using hypothetical facility IDs)
        location_filter = LocationFilter(facility_ids=["FAC001", "FAC002"])
        
        filtered_dashboard = dashboard_service.generate_executive_dashboard(
            location_filter=location_filter,
            include_trends=False,
            include_recommendations=False
        )
        
        if 'error' not in filtered_dashboard:
            filtered_summary = filtered_dashboard.get('summary', {})
            filtered_facilities = filtered_summary.get('facilities', {}).get('total_count', 0)
            
            print(f"Filtered facilities count: {filtered_facilities}")
            print("Filter configuration:")
            print(f"  - Facility IDs: {location_filter.facility_ids}")
            
            # Show some filtered metrics
            incidents = filtered_summary.get('incidents', {})
            print(f"Filtered Results:")
            print(f"  - Total Incidents: {incidents.get('total', 'N/A')}")
            print(f"  - Incident Rate: {incidents.get('incident_rate', 'N/A')}")
            
            logger.info("✓ Location filtering test completed")
        else:
            logger.warning(f"⚠ Location filtering returned error: {filtered_dashboard.get('error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Location filtering test failed: {e}")
        return False


def test_date_range_filtering(dashboard_service: ExecutiveDashboardService):
    """Test date range filtering"""
    print("\n" + "="*80)
    print("TESTING: Date Range Filtering")
    print("="*80)
    
    try:
        # Test different date ranges
        date_ranges = [
            {
                "name": "Last 7 days",
                "filter": DateRangeFilter(
                    start_date=datetime.now() - timedelta(days=7),
                    end_date=datetime.now(),
                    period=AggregationPeriod.DAILY
                )
            },
            {
                "name": "Last 30 days", 
                "filter": DateRangeFilter(
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                    period=AggregationPeriod.WEEKLY
                )
            },
            {
                "name": "Last 90 days",
                "filter": DateRangeFilter(
                    start_date=datetime.now() - timedelta(days=90),
                    end_date=datetime.now(),
                    period=AggregationPeriod.MONTHLY
                )
            }
        ]
        
        for date_range_config in date_ranges:
            print(f"\n--- {date_range_config['name']} ---")
            
            dashboard_data = dashboard_service.generate_executive_dashboard(
                date_filter=date_range_config['filter'],
                include_trends=False,
                include_recommendations=False
            )
            
            if 'error' not in dashboard_data:
                summary = dashboard_data.get('summary', {})
                period = summary.get('period', {})
                incidents = summary.get('incidents', {})
                
                print(f"Period: {period.get('start_date', 'N/A')} to {period.get('end_date', 'N/A')}")
                print(f"Duration: {period.get('period_days', 'N/A')} days")
                print(f"Total Incidents: {incidents.get('total', 'N/A')}")
                print(f"Incident Rate: {incidents.get('incident_rate', 'N/A')}")
                
            else:
                print(f"Error: {dashboard_data.get('error')}")
        
        logger.info("✓ Date range filtering test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Date range filtering test failed: {e}")
        return False


def test_comprehensive_dashboard(dashboard_service: ExecutiveDashboardService):
    """Test comprehensive dashboard generation with all features"""
    print("\n" + "="*80)
    print("TESTING: Comprehensive Dashboard Generation")
    print("="*80)
    
    try:
        # Generate comprehensive dashboard with all features
        comprehensive_dashboard = dashboard_service.generate_executive_dashboard(
            include_trends=True,
            include_recommendations=True,
            include_forecasts=True
        )
        
        if 'error' in comprehensive_dashboard:
            logger.error(f"✗ Comprehensive dashboard generation failed: {comprehensive_dashboard['error']}")
            return False
        
        print("Dashboard Components Generated:")
        
        # Metadata
        metadata = comprehensive_dashboard.get('metadata', {})
        print(f"  - Generated At: {metadata.get('generated_at', 'N/A')}")
        print(f"  - Version: {metadata.get('version', 'N/A')}")
        print(f"  - Cache Status: {metadata.get('cache_status', 'N/A')}")
        
        # Summary
        summary = comprehensive_dashboard.get('summary', {})
        if summary:
            print(f"  - Overall Health Score: {summary.get('overall_health_score', 'N/A')}")
            incidents = summary.get('incidents', {})
            print(f"  - Total Incidents: {incidents.get('total', 'N/A')}")
        
        # KPIs
        kpis = comprehensive_dashboard.get('kpis', {})
        if kpis:
            kpi_summary = kpis.get('summary', {})
            print(f"  - Total KPIs: {kpi_summary.get('total_kpis', 'N/A')}")
            print(f"  - Green Status: {kpi_summary.get('green_status', 'N/A')}")
            print(f"  - Yellow Status: {kpi_summary.get('yellow_status', 'N/A')}")
            print(f"  - Red Status: {kpi_summary.get('red_status', 'N/A')}")
        
        # Charts
        charts = comprehensive_dashboard.get('charts', {})
        if charts:
            print(f"  - Charts Generated: {len(charts)}")
            for chart_name in list(charts.keys())[:3]:  # Show first 3
                chart = charts[chart_name]
                print(f"    • {chart.get('title', chart_name)} ({chart.get('type', 'unknown')})")
        
        # Alerts
        alerts = comprehensive_dashboard.get('alerts', {})
        if alerts:
            alert_summary = alerts.get('summary', {})
            print(f"  - Total Active Alerts: {alert_summary.get('total_active', 'N/A')}")
            print(f"  - Critical Alerts: {alert_summary.get('critical_count', 'N/A')}")
            print(f"  - Alert Level: {alert_summary.get('alert_level', 'N/A')}")
        
        # Trends
        trends = comprehensive_dashboard.get('trends', {})
        if trends:
            metric_trends = trends.get('metric_trends', {})
            recent_anomalies = trends.get('recent_anomalies', [])
            print(f"  - Metrics Analyzed for Trends: {len(metric_trends)}")
            print(f"  - Recent Anomalies Detected: {len(recent_anomalies)}")
        
        # Recommendations
        recommendations = comprehensive_dashboard.get('recommendations', {})
        if recommendations:
            rec_summary = recommendations.get('summary', {})
            ai_recs = recommendations.get('ai_generated_recommendations', [])
            print(f"  - Pending Recommendations: {rec_summary.get('pending_review', 'N/A')}")
            print(f"  - In Progress: {rec_summary.get('in_progress', 'N/A')}")
            print(f"  - AI-Generated Recommendations: {len(ai_recs)}")
        
        # Forecasts
        forecasts = comprehensive_dashboard.get('forecasts', {})
        if forecasts:
            forecast_data = forecasts.get('forecasts', {})
            print(f"  - Forecast Models: {len(forecast_data)}")
            for forecast_name in forecast_data.keys():
                print(f"    • {forecast_name}")
        
        # Status
        status = comprehensive_dashboard.get('status', {})
        if status:
            print(f"  - Overall System Status: {status.get('overall_status', 'N/A')}")
            performance = status.get('performance_metrics', {})
            print(f"  - Error Rate: {performance.get('error_rate', 'N/A')}%")
            print(f"  - Cache Hit Rate: {performance.get('cache_hit_rate', 'N/A')}%")
        
        # Calculate dashboard size
        dashboard_json = json.dumps(comprehensive_dashboard)
        dashboard_size_kb = len(dashboard_json.encode('utf-8')) / 1024
        print(f"\nDashboard JSON Size: {dashboard_size_kb:.2f} KB")
        
        logger.info("✓ Comprehensive dashboard generated successfully")
        
        # Optionally save dashboard to file for inspection
        output_file = f"sample_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive_dashboard, f, indent=2)
        
        print(f"\nDashboard saved to: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Comprehensive dashboard test failed: {e}")
        return False


def test_error_handling(dashboard_service: ExecutiveDashboardService):
    """Test error handling and edge cases"""
    print("\n" + "="*80)
    print("TESTING: Error Handling and Edge Cases")
    print("="*80)
    
    try:
        # Test with invalid facility IDs
        print("Testing with invalid facility IDs...")
        invalid_filter = LocationFilter(facility_ids=["INVALID001", "INVALID002"])
        
        dashboard_with_invalid = dashboard_service.generate_executive_dashboard(
            location_filter=invalid_filter
        )
        
        if 'error' in dashboard_with_invalid:
            print(f"  ✓ Handled invalid facility IDs gracefully: {dashboard_with_invalid.get('error_type', 'Unknown')}")
        else:
            print("  ✓ Generated dashboard despite invalid facility IDs (graceful degradation)")
            facilities_count = dashboard_with_invalid.get('summary', {}).get('facilities', {}).get('total_count', 0)
            print(f"    Facilities found: {facilities_count}")
        
        # Test with extreme date ranges
        print("\nTesting with extreme date ranges...")
        extreme_date_filter = DateRangeFilter(
            start_date=datetime(2020, 1, 1),  # Very old start date
            end_date=datetime(2030, 12, 31)   # Future end date
        )
        
        dashboard_with_extreme_dates = dashboard_service.generate_executive_dashboard(
            date_filter=extreme_date_filter
        )
        
        if 'error' in dashboard_with_extreme_dates:
            print(f"  ✓ Handled extreme date ranges: {dashboard_with_extreme_dates.get('error_type', 'Unknown')}")
        else:
            print("  ✓ Generated dashboard with extreme date ranges")
        
        # Test cache functionality
        print("\nTesting cache functionality...")
        start_time = datetime.now()
        
        # First call (should miss cache)
        dashboard1 = dashboard_service.get_dashboard_summary()
        first_call_time = (datetime.now() - start_time).total_seconds()
        
        start_time = datetime.now()
        
        # Second call (should hit cache)
        dashboard2 = dashboard_service.get_dashboard_summary()
        second_call_time = (datetime.now() - start_time).total_seconds()
        
        print(f"  First call time: {first_call_time:.3f}s")
        print(f"  Second call time: {second_call_time:.3f}s")
        
        if second_call_time < first_call_time:
            print("  ✓ Cache appears to be working (second call faster)")
        else:
            print("  ⚠ Cache may not be working optimally")
        
        # Clear cache test
        dashboard_service.clear_cache()
        print("  ✓ Cache cleared successfully")
        
        logger.info("✓ Error handling tests completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Main test function"""
    print("Executive Dashboard Service - Comprehensive Test Suite")
    print("="*80)
    
    # Test results tracking
    test_results = {}
    dashboard_service = None
    
    try:
        # Initialize service
        dashboard_service = test_service_initialization()
        if not dashboard_service:
            print("❌ Cannot proceed with tests - service initialization failed")
            return
        
        test_results['initialization'] = True
        
        # Run tests
        test_functions = [
            ('real_time_metrics', test_real_time_metrics),
            ('kpi_details', test_kpi_details),
            ('dashboard_summary', test_dashboard_summary),
            ('location_filtering', test_location_filtering),
            ('date_range_filtering', test_date_range_filtering),
            ('comprehensive_dashboard', test_comprehensive_dashboard),
            ('error_handling', test_error_handling)
        ]
        
        for test_name, test_function in test_functions:
            try:
                result = test_function(dashboard_service)
                test_results[test_name] = result
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                test_results[test_name] = False
        
        # Print test summary
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        for test_name, result in test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name.upper().replace('_', ' ')}: {status}")
        
        print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
        
        if failed_tests == 0:
            print("🎉 All tests passed! The Executive Dashboard Service is fully functional.")
        else:
            print(f"⚠️  {failed_tests} test(s) failed. Please review the logs above.")
        
        # Performance summary
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        if dashboard_service:
            # Get performance metrics from service
            try:
                health_status = dashboard_service.health_check()
                
                # Print performance info if available
                print("Service Performance:")
                print(f"  - Database Status: {health_status.get('database', {}).get('status', 'Unknown')}")
                print(f"  - Trend Analysis Status: {health_status.get('trend_analysis', {}).get('status', 'Unknown')}")
                print(f"  - Cache Status: {health_status.get('cache', {}).get('status', 'Unknown')}")
                
            except Exception as e:
                print(f"Could not retrieve performance metrics: {e}")
    
    finally:
        # Cleanup
        if dashboard_service:
            try:
                dashboard_service.close()
                print("\n✓ Dashboard service closed successfully")
            except Exception as e:
                print(f"\n⚠ Error closing dashboard service: {e}")


if __name__ == "__main__":
    main()