#!/usr/bin/env python3
"""
Executive Dashboard Service - Usage Examples

This script provides practical examples of how to use the ExecutiveDashboardService
for various executive dashboard scenarios.

Created: 2025-08-28
Version: 1.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pprint import pprint

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from services.executive_dashboard.dashboard_service import (
    ExecutiveDashboardService, LocationFilter, DateRangeFilter, 
    AggregationPeriod, create_dashboard_service
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_dashboard():
    """Example 1: Basic executive dashboard for all facilities"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Executive Dashboard")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Create dashboard service
    dashboard_service = create_dashboard_service()
    
    try:
        # Generate basic dashboard
        dashboard_data = dashboard_service.generate_executive_dashboard()
        
        if 'error' in dashboard_data:
            print(f"Error generating dashboard: {dashboard_data['error']}")
            return
        
        # Extract key information
        summary = dashboard_data.get('summary', {})
        alerts = dashboard_data.get('alerts', {})
        
        print("📊 EXECUTIVE DASHBOARD SUMMARY")
        print("-" * 40)
        
        # Overall health
        health_score = summary.get('overall_health_score', 'N/A')
        alert_level = alerts.get('summary', {}).get('alert_level', 'N/A')
        
        print(f"Overall Health Score: {health_score}")
        print(f"Alert Level: {alert_level}")
        
        # Key metrics
        incidents = summary.get('incidents', {})
        compliance = summary.get('compliance', {})
        facilities = summary.get('facilities', {})
        
        print(f"\nKey Metrics:")
        print(f"  - Total Facilities: {facilities.get('total_count', 'N/A')}")
        print(f"  - Total Incidents: {incidents.get('total', 'N/A')}")
        print(f"  - Today's Incidents: {incidents.get('today', 'N/A')}")
        print(f"  - Incident Rate: {incidents.get('incident_rate', 'N/A')}")
        print(f"  - Audit Pass Rate: {compliance.get('audit_pass_rate', 'N/A')}%")
        print(f"  - Training Completion: {compliance.get('training_completion', 'N/A')}%")
        
        # Alert summary
        alert_summary = alerts.get('summary', {})
        print(f"\nAlert Summary:")
        print(f"  - Total Active: {alert_summary.get('total_active', 'N/A')}")
        print(f"  - Critical: {alert_summary.get('critical_count', 'N/A')}")
        print(f"  - High Priority: {alert_summary.get('high_count', 'N/A')}")
        
    finally:
        dashboard_service.close()


def example_2_filtered_dashboard():
    """Example 2: Filtered dashboard for specific facilities and date range"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Filtered Dashboard (Location + Date)")
    print("="*60)
    
    load_dotenv()
    dashboard_service = create_dashboard_service()
    
    try:
        # Define filters
        location_filter = LocationFilter(
            facility_ids=["FAC001", "FAC002", "FAC003"]  # Example facility IDs
        )
        
        date_filter = DateRangeFilter(
            start_date=datetime.now() - timedelta(days=7),  # Last 7 days
            end_date=datetime.now(),
            period=AggregationPeriod.DAILY
        )
        
        # Generate filtered dashboard
        dashboard_data = dashboard_service.generate_executive_dashboard(
            location_filter=location_filter,
            date_filter=date_filter,
            include_trends=True
        )
        
        if 'error' in dashboard_data:
            print(f"Error: {dashboard_data['error']}")
            return
        
        print("📍 FILTERED DASHBOARD RESULTS")
        print("-" * 40)
        
        # Show filter information
        metadata = dashboard_data.get('metadata', {})
        filters = metadata.get('filters', {})
        
        location_info = filters.get('location', {})
        date_info = filters.get('date_range', {})
        
        print(f"Filters Applied:")
        print(f"  - Facility IDs: {location_info.get('facility_ids', 'All')}")
        print(f"  - Date Range: {date_info.get('start_date', 'N/A')} to {date_info.get('end_date', 'N/A')}")
        print(f"  - Aggregation Period: {date_info.get('period', 'N/A')}")
        
        # Show results
        summary = dashboard_data.get('summary', {})
        period = summary.get('period', {})
        facilities = summary.get('facilities', {})
        incidents = summary.get('incidents', {})
        
        print(f"\nFiltered Results:")
        print(f"  - Period: {period.get('period_days', 'N/A')} days")
        print(f"  - Facilities Included: {facilities.get('total_count', 'N/A')}")
        print(f"  - Total Incidents: {incidents.get('total', 'N/A')}")
        print(f"  - Incident Rate: {incidents.get('incident_rate', 'N/A')}")
        
        # Show trends if available
        trends = dashboard_data.get('trends', {})
        if trends:
            metric_trends = trends.get('metric_trends', {})
            print(f"\nTrend Analysis:")
            print(f"  - Metrics Analyzed: {len(metric_trends)}")
            
            for metric_name, trend_data in metric_trends.items():
                if 'key_findings' in trend_data:
                    findings = trend_data['key_findings']
                    if findings:
                        print(f"  - {metric_name}: {findings[0].get('finding', 'No trend detected')}")
        
    finally:
        dashboard_service.close()


def example_3_real_time_monitoring():
    """Example 3: Real-time monitoring dashboard"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Real-Time Monitoring")
    print("="*60)
    
    load_dotenv()
    dashboard_service = create_dashboard_service()
    
    try:
        # Get real-time metrics
        real_time_data = dashboard_service.get_real_time_metrics()
        
        print("🔴 REAL-TIME MONITORING DASHBOARD")
        print("-" * 40)
        
        # Current status
        alert_level = real_time_data.get('alert_level', 'UNKNOWN')
        timestamp = real_time_data.get('timestamp', 'N/A')
        
        # Set emoji based on alert level
        status_emoji = {
            'GREEN': '🟢',
            'YELLOW': '🟡', 
            'ORANGE': '🟠',
            'RED': '🔴'
        }.get(alert_level, '⚪')
        
        print(f"Status: {status_emoji} {alert_level}")
        print(f"Last Updated: {timestamp}")
        
        # Key metrics
        metrics = real_time_data.get('metrics', {})
        print(f"\nCurrent Metrics:")
        print(f"  - Today's Incidents: {metrics.get('todays_incidents', 0)}")
        print(f"  - Active Alerts: {metrics.get('active_alerts', 0)}")
        print(f"  - Overdue Training: {metrics.get('overdue_training', 0)}")
        print(f"  - Overdue Inspections: {metrics.get('overdue_inspections', 0)}")
        
        # Facility breakdown
        facility_breakdown = real_time_data.get('facility_breakdown', [])
        print(f"\nFacility Status ({len(facility_breakdown)} facilities):")
        
        # Show facilities with issues first
        facilities_with_issues = [
            f for f in facility_breakdown 
            if f.get('active_alerts', 0) > 0 or f.get('todays_incidents', 0) > 0
        ]
        
        if facilities_with_issues:
            print("  🚨 Facilities Requiring Attention:")
            for facility in facilities_with_issues[:5]:  # Top 5
                name = facility.get('facility_name', 'Unknown')
                alerts = facility.get('active_alerts', 0)
                incidents = facility.get('todays_incidents', 0)
                print(f"     • {name}: {alerts} alerts, {incidents} incidents today")
        else:
            print("  ✅ All facilities operating normally")
        
        # Show total summary
        total_alerts = sum(f.get('active_alerts', 0) for f in facility_breakdown)
        total_incidents_today = sum(f.get('todays_incidents', 0) for f in facility_breakdown)
        
        print(f"\nTotals Across All Facilities:")
        print(f"  - Total Active Alerts: {total_alerts}")
        print(f"  - Total Incidents Today: {total_incidents_today}")
        
    finally:
        dashboard_service.close()


def example_4_kpi_analysis():
    """Example 4: Detailed KPI analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Detailed KPI Analysis")
    print("="*60)
    
    load_dotenv()
    dashboard_service = create_dashboard_service()
    
    try:
        # Get detailed KPIs for last 30 days
        kpi_details = dashboard_service.get_kpi_details(date_range_days=30)
        
        print("📈 DETAILED KPI ANALYSIS")
        print("-" * 40)
        
        # Period information
        period = kpi_details.get('period', {})
        print(f"Analysis Period: {period.get('days', 'N/A')} days")
        print(f"From: {period.get('start_date', 'N/A')}")
        print(f"To: {period.get('end_date', 'N/A')}")
        
        # Safety KPIs
        print(f"\n🦺 SAFETY KPIs:")
        safety_kpis = kpi_details.get('safety_kpis', {})
        
        for name, kpi in safety_kpis.items():
            # Determine status icon
            status_icons = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
            status = self._determine_kpi_status_example(kpi)
            icon = status_icons.get(status, '⚪')
            
            print(f"  {icon} {kpi.get('name', name)}")
            print(f"     Value: {kpi.get('value', 'N/A')} {kpi.get('unit', '')}")
            
            if kpi.get('target') is not None:
                target = kpi['target']
                value = kpi.get('value', 0)
                
                if isinstance(value, (int, float)) and isinstance(target, (int, float)):
                    if target == 0:
                        performance = "Target is zero-tolerance"
                    else:
                        performance_pct = (value / target) * 100
                        performance = f"{performance_pct:.1f}% of target"
                else:
                    performance = f"Target: {target}"
                
                print(f"     Performance: {performance}")
            
            if kpi.get('threshold_warning') is not None:
                print(f"     Warning Threshold: {kpi['threshold_warning']}")
        
        # Compliance KPIs
        print(f"\n📋 COMPLIANCE KPIs:")
        compliance_kpis = kpi_details.get('compliance_kpis', {})
        
        for name, kpi in compliance_kpis.items():
            status = self._determine_kpi_status_example(kpi)
            icon = status_icons.get(status, '⚪')
            
            print(f"  {icon} {kpi.get('name', name)}")
            print(f"     Value: {kpi.get('value', 'N/A')} {kpi.get('unit', '')}")
            
            if kpi.get('target') is not None:
                target = kpi['target']
                value = kpi.get('value', 0)
                
                if isinstance(value, (int, float)) and isinstance(target, (int, float)):
                    if target > 0:
                        performance_pct = (value / target) * 100
                        performance = f"{performance_pct:.1f}% of target"
                    else:
                        performance = f"Target: {target}"
                else:
                    performance = f"Target: {target}"
                
                print(f"     Performance: {performance}")
        
        # KPI summary
        all_kpis = {**safety_kpis, **compliance_kpis}
        green_count = sum(1 for kpi in all_kpis.values() if self._determine_kpi_status_example(kpi) == 'green')
        yellow_count = sum(1 for kpi in all_kpis.values() if self._determine_kpi_status_example(kpi) == 'yellow')
        red_count = sum(1 for kpi in all_kpis.values() if self._determine_kpi_status_example(kpi) == 'red')
        
        print(f"\n📊 KPI SUMMARY:")
        print(f"  🟢 Green (On Target): {green_count}")
        print(f"  🟡 Yellow (Warning): {yellow_count}")
        print(f"  🔴 Red (Critical): {red_count}")
        print(f"  Total KPIs: {len(all_kpis)}")
        
    finally:
        dashboard_service.close()


def example_5_comprehensive_export():
    """Example 5: Generate comprehensive dashboard and export to JSON"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Comprehensive Dashboard Export")
    print("="*60)
    
    load_dotenv()
    dashboard_service = create_dashboard_service()
    
    try:
        # Generate comprehensive dashboard with all features
        comprehensive_dashboard = dashboard_service.generate_executive_dashboard(
            include_trends=True,
            include_recommendations=True,
            include_forecasts=True
        )
        
        if 'error' in comprehensive_dashboard:
            print(f"Error: {comprehensive_dashboard['error']}")
            return
        
        print("📁 COMPREHENSIVE DASHBOARD EXPORT")
        print("-" * 40)
        
        # Show what was generated
        components = []
        if 'summary' in comprehensive_dashboard:
            components.append("Summary Metrics")
        if 'kpis' in comprehensive_dashboard:
            components.append("KPI Analysis")
        if 'charts' in comprehensive_dashboard:
            components.append(f"Charts ({len(comprehensive_dashboard['charts'])})")
        if 'alerts' in comprehensive_dashboard:
            components.append("Alerts & Notifications")
        if 'trends' in comprehensive_dashboard:
            components.append("Trend Analysis")
        if 'recommendations' in comprehensive_dashboard:
            components.append("Recommendations")
        if 'forecasts' in comprehensive_dashboard:
            components.append("Forecasts")
        if 'status' in comprehensive_dashboard:
            components.append("System Status")
        
        print(f"Generated Components:")
        for i, component in enumerate(components, 1):
            print(f"  {i}. {component}")
        
        # Calculate dashboard statistics
        dashboard_json = json.dumps(comprehensive_dashboard)
        dashboard_size = len(dashboard_json.encode('utf-8'))
        
        print(f"\nDashboard Statistics:")
        print(f"  - JSON Size: {dashboard_size:,} bytes ({dashboard_size/1024:.2f} KB)")
        print(f"  - Components: {len(components)}")
        
        # Show key insights
        summary = comprehensive_dashboard.get('summary', {})
        health_score = summary.get('overall_health_score', 'N/A')
        alert_level = comprehensive_dashboard.get('alerts', {}).get('summary', {}).get('alert_level', 'N/A')
        
        print(f"\nKey Insights:")
        print(f"  - Overall Health: {health_score}")
        print(f"  - Alert Level: {alert_level}")
        
        # Export to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"executive_dashboard_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(comprehensive_dashboard, f, indent=2)
        
        print(f"\n💾 Dashboard exported to: {filename}")
        
        # Show sample of trends if available
        trends = comprehensive_dashboard.get('trends', {})
        if trends:
            trend_summary = trends.get('trend_summary', {})
            print(f"\nTrend Analysis Summary:")
            print(f"  - Metrics Analyzed: {trend_summary.get('total_metrics_analyzed', 'N/A')}")
            print(f"  - Trends Detected: {trend_summary.get('trends_detected', 'N/A')}")
            print(f"  - Anomalies Found: {trend_summary.get('anomalies_detected', 'N/A')}")
            
            concerning_trends = trend_summary.get('concerning_trends', [])
            if concerning_trends:
                print(f"  - Concerning Trends: {len(concerning_trends)}")
                for trend in concerning_trends[:2]:  # Show first 2
                    print(f"    • {trend}")
        
        # Show sample recommendations
        recommendations = comprehensive_dashboard.get('recommendations', {})
        if recommendations:
            ai_recs = recommendations.get('ai_generated_recommendations', [])
            if ai_recs:
                print(f"\nAI-Generated Recommendations ({len(ai_recs)}):")
                for i, rec in enumerate(ai_recs[:2], 1):  # Show first 2
                    print(f"  {i}. {rec.get('title', 'No title')}")
                    print(f"     Priority: {rec.get('priority', 'N/A')}")
                    print(f"     Impact: {rec.get('estimated_impact', 'N/A')}")
        
    finally:
        dashboard_service.close()


def _determine_kpi_status_example(kpi: dict) -> str:
    """Helper function to determine KPI status"""
    value = kpi.get('value', 0)
    threshold_critical = kpi.get('threshold_critical')
    threshold_warning = kpi.get('threshold_warning')
    
    if threshold_critical is not None and value >= threshold_critical:
        return 'red'
    elif threshold_warning is not None and value >= threshold_warning:
        return 'yellow'
    else:
        return 'green'


def main():
    """Run all examples"""
    print("Executive Dashboard Service - Usage Examples")
    print("=" * 80)
    
    examples = [
        example_1_basic_dashboard,
        example_2_filtered_dashboard,
        example_3_real_time_monitoring,
        example_4_kpi_analysis,
        example_5_comprehensive_export
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\nRunning Example {i}...")
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            logger.error(f"Example {i} failed", exc_info=True)
    
    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("\nThe Executive Dashboard Service provides comprehensive")
    print("functionality for executive-level EHS monitoring and reporting.")


if __name__ == "__main__":
    main()