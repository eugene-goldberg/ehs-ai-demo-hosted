"""
Analytics Aggregation Layer for Executive Dashboard

This module provides comprehensive analytics aggregation functionality including:
- KPI calculations and monitoring
- Dashboard data aggregation
- Time-series data processing
- Cross-facility benchmarking
- Real-time metric calculations
- Executive summary generation
- Export formats for reporting
- Cache management for performance optimization

Created: 2025-08-28
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd
from decimal import Decimal
import hashlib
import time
from functools import wraps

from neo4j import GraphDatabase
from ..base_query import BaseQuery


class AggregationPeriod(Enum):
    """Enumeration for aggregation time periods"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    REAL_TIME = "real_time"


class ExportFormat(Enum):
    """Enumeration for export formats"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    XML = "xml"


@dataclass
class KPIMetric:
    """Data class for KPI metrics"""
    name: str
    value: Union[int, float, Decimal]
    unit: str
    target: Optional[Union[int, float, Decimal]] = None
    threshold_warning: Optional[Union[int, float, Decimal]] = None
    threshold_critical: Optional[Union[int, float, Decimal]] = None
    trend: Optional[str] = None  # 'up', 'down', 'stable'
    change_percent: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class FacilityBenchmark:
    """Data class for facility benchmarking"""
    facility_id: str
    facility_name: str
    metric_name: str
    value: Union[int, float, Decimal]
    rank: int
    percentile: float
    industry_average: Optional[Union[int, float, Decimal]] = None
    best_practice: Optional[Union[int, float, Decimal]] = None


def cache_result(expiry_minutes: int = 15):
    """Decorator for caching query results"""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = hashlib.md5(
                f"{func.__name__}_{str(args)}_{str(kwargs)}".encode()
            ).hexdigest()
            
            current_time = time.time()
            
            # Check if cached result exists and is still valid
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp < (expiry_minutes * 60):
                    return result
            
            # Execute function and cache result
            result = func(self, *args, **kwargs)
            cache[cache_key] = (result, current_time)
            
            return result
        
        return wrapper
    return decorator


class AnalyticsAggregationLayer(BaseQuery):
    """
    Comprehensive analytics aggregation layer for executive dashboard
    """
    
    def __init__(self, driver: GraphDatabase):
        super().__init__(driver)
        self.cache_enabled = True
        self.default_cache_ttl = 15  # minutes
    
    # KPI Calculation Methods
    
    @cache_result(expiry_minutes=5)
    def calculate_safety_kpis(self, 
                            facility_ids: Optional[List[str]] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[str, KPIMetric]:
        """Calculate comprehensive safety KPIs"""
        
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        facility_filter = ""
        if facility_ids:
            facility_filter = "AND f.facility_id IN $facility_ids"
        
        query = """
        MATCH (f:Facility)
        WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
        
        OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
        WHERE i.incident_date >= $start_date AND i.incident_date <= $end_date
        
        OPTIONAL MATCH (f)-[:HAS_INJURY]->(inj:Injury)
        WHERE inj.injury_date >= $start_date AND inj.injury_date <= $end_date
        
        OPTIONAL MATCH (f)-[:HAS_EMPLOYEE]->(e:Employee)
        
        WITH f, 
             count(DISTINCT i) as total_incidents,
             count(DISTINCT CASE WHEN i.severity = 'Critical' THEN i END) as critical_incidents,
             count(DISTINCT inj) as total_injuries,
             count(DISTINCT CASE WHEN inj.lost_time = true THEN inj END) as lost_time_injuries,
             count(DISTINCT e) as total_employees
        
        RETURN f.facility_id as facility_id,
               f.facility_name as facility_name,
               total_incidents,
               critical_incidents,
               total_injuries,
               lost_time_injuries,
               total_employees,
               CASE 
                   WHEN total_employees > 0 
                   THEN (total_incidents * 200000.0) / total_employees 
                   ELSE 0 
               END as incident_rate,
               CASE 
                   WHEN total_employees > 0 
                   THEN (lost_time_injuries * 200000.0) / total_employees 
                   ELSE 0 
               END as ltir
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'facility_ids': facility_ids,
                'start_date': start_date,
                'end_date': end_date
            })
            
            records = list(result)
        
        # Calculate aggregated KPIs
        total_incidents = sum(r['total_incidents'] for r in records)
        total_critical = sum(r['critical_incidents'] for r in records)
        total_injuries = sum(r['total_injuries'] for r in records)
        total_lti = sum(r['lost_time_injuries'] for r in records)
        total_employees = sum(r['total_employees'] for r in records)
        
        # Calculate rates
        incident_rate = (total_incidents * 200000 / total_employees) if total_employees > 0 else 0
        ltir = (total_lti * 200000 / total_employees) if total_employees > 0 else 0
        severity_rate = (total_critical / total_incidents * 100) if total_incidents > 0 else 0
        
        kpis = {
            'total_incidents': KPIMetric(
                name='Total Incidents',
                value=total_incidents,
                unit='count',
                target=0,
                threshold_warning=5,
                threshold_critical=10,
                timestamp=datetime.now()
            ),
            'incident_rate': KPIMetric(
                name='Total Incident Rate',
                value=round(incident_rate, 2),
                unit='per 200K hours',
                target=2.0,
                threshold_warning=3.0,
                threshold_critical=5.0,
                timestamp=datetime.now()
            ),
            'ltir': KPIMetric(
                name='Lost Time Injury Rate',
                value=round(ltir, 2),
                unit='per 200K hours',
                target=1.0,
                threshold_warning=2.0,
                threshold_critical=3.0,
                timestamp=datetime.now()
            ),
            'severity_rate': KPIMetric(
                name='Critical Incident Severity Rate',
                value=round(severity_rate, 2),
                unit='percentage',
                target=0,
                threshold_warning=10,
                threshold_critical=20,
                timestamp=datetime.now()
            )
        }
        
        return kpis
    
    @cache_result(expiry_minutes=10)
    def calculate_compliance_kpis(self,
                                facility_ids: Optional[List[str]] = None,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> Dict[str, KPIMetric]:
        """Calculate compliance KPIs"""
        
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        query = """
        MATCH (f:Facility)
        WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
        
        OPTIONAL MATCH (f)-[:HAS_AUDIT]->(a:Audit)
        WHERE a.audit_date >= $start_date AND a.audit_date <= $end_date
        
        OPTIONAL MATCH (f)-[:HAS_VIOLATION]->(v:Violation)
        WHERE v.violation_date >= $start_date AND v.violation_date <= $end_date
        
        OPTIONAL MATCH (f)-[:REQUIRES_TRAINING]->(t:Training)
        WHERE t.due_date >= $start_date AND t.due_date <= $end_date
        
        WITH f,
             count(DISTINCT a) as total_audits,
             count(DISTINCT CASE WHEN a.status = 'Passed' THEN a END) as passed_audits,
             count(DISTINCT v) as total_violations,
             count(DISTINCT CASE WHEN v.severity = 'Critical' THEN v END) as critical_violations,
             count(DISTINCT t) as total_training,
             count(DISTINCT CASE WHEN t.status = 'Completed' THEN t END) as completed_training
        
        RETURN f.facility_id as facility_id,
               total_audits,
               passed_audits,
               total_violations,
               critical_violations,
               total_training,
               completed_training,
               CASE 
                   WHEN total_audits > 0 
                   THEN (passed_audits * 100.0) / total_audits 
                   ELSE 100.0 
               END as audit_pass_rate,
               CASE 
                   WHEN total_training > 0 
                   THEN (completed_training * 100.0) / total_training 
                   ELSE 100.0 
               END as training_completion_rate
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'facility_ids': facility_ids,
                'start_date': start_date,
                'end_date': end_date
            })
            
            records = list(result)
        
        # Calculate aggregated compliance KPIs
        total_audits = sum(r['total_audits'] for r in records)
        passed_audits = sum(r['passed_audits'] for r in records)
        total_violations = sum(r['total_violations'] for r in records)
        critical_violations = sum(r['critical_violations'] for r in records)
        total_training = sum(r['total_training'] for r in records)
        completed_training = sum(r['completed_training'] for r in records)
        
        audit_pass_rate = (passed_audits / total_audits * 100) if total_audits > 0 else 100
        training_completion_rate = (completed_training / total_training * 100) if total_training > 0 else 100
        violation_rate = (total_violations / len(records)) if records else 0
        
        kpis = {
            'audit_pass_rate': KPIMetric(
                name='Audit Pass Rate',
                value=round(audit_pass_rate, 2),
                unit='percentage',
                target=95.0,
                threshold_warning=85.0,
                threshold_critical=75.0,
                timestamp=datetime.now()
            ),
            'training_completion_rate': KPIMetric(
                name='Training Completion Rate',
                value=round(training_completion_rate, 2),
                unit='percentage',
                target=100.0,
                threshold_warning=90.0,
                threshold_critical=80.0,
                timestamp=datetime.now()
            ),
            'total_violations': KPIMetric(
                name='Total Violations',
                value=total_violations,
                unit='count',
                target=0,
                threshold_warning=3,
                threshold_critical=5,
                timestamp=datetime.now()
            ),
            'critical_violations': KPIMetric(
                name='Critical Violations',
                value=critical_violations,
                unit='count',
                target=0,
                threshold_warning=1,
                threshold_critical=2,
                timestamp=datetime.now()
            )
        }
        
        return kpis
    
    # Dashboard Data Aggregation
    
    @cache_result(expiry_minutes=5)
    def get_dashboard_summary(self,
                            facility_ids: Optional[List[str]] = None,
                            period: AggregationPeriod = AggregationPeriod.DAILY) -> Dict[str, Any]:
        """Get comprehensive dashboard summary data"""
        
        end_date = datetime.now()
        
        # Determine date range based on period
        if period == AggregationPeriod.DAILY:
            start_date = end_date - timedelta(days=1)
        elif period == AggregationPeriod.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif period == AggregationPeriod.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif period == AggregationPeriod.QUARTERLY:
            start_date = end_date - timedelta(days=90)
        elif period == AggregationPeriod.YEARLY:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=7)
        
        # Get all KPIs
        safety_kpis = self.calculate_safety_kpis(facility_ids, start_date, end_date)
        compliance_kpis = self.calculate_compliance_kpis(facility_ids, start_date, end_date)
        
        # Get facility overview
        facility_overview = self.get_facility_overview(facility_ids)
        
        # Get trending data
        trending_data = self.get_trending_metrics(facility_ids, start_date, end_date, period)
        
        # Get top risks
        top_risks = self.get_top_risks(facility_ids, limit=5)
        
        # Get recent alerts
        recent_alerts = self.get_recent_alerts(facility_ids, limit=10)
        
        return {
            'summary': {
                'generated_at': datetime.now().isoformat(),
                'period': period.value,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'facilities_included': len(facility_overview) if facility_overview else 0
            },
            'kpis': {
                **safety_kpis,
                **compliance_kpis
            },
            'facility_overview': facility_overview,
            'trending_data': trending_data,
            'top_risks': top_risks,
            'recent_alerts': recent_alerts
        }
    
    def get_facility_overview(self, facility_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get overview of facilities"""
        
        query = """
        MATCH (f:Facility)
        WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
        
        OPTIONAL MATCH (f)-[:HAS_EMPLOYEE]->(e:Employee)
        OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
        WHERE i.incident_date >= date() - duration('P30D')
        
        RETURN f.facility_id as facility_id,
               f.facility_name as facility_name,
               f.location as location,
               f.facility_type as facility_type,
               count(DISTINCT e) as employee_count,
               count(DISTINCT i) as recent_incidents
        ORDER BY f.facility_name
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'facility_ids': facility_ids})
            return [dict(record) for record in result]
    
    # Time-series Aggregation Functions
    
    def get_trending_metrics(self,
                           facility_ids: Optional[List[str]] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           period: AggregationPeriod = AggregationPeriod.DAILY) -> Dict[str, List[Dict]]:
        """Get time-series trending data for key metrics"""
        
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Determine grouping based on period
        if period == AggregationPeriod.HOURLY:
            group_format = "datetime({year: date.year, month: date.month, day: date.day, hour: date.hour})"
        elif period == AggregationPeriod.DAILY:
            group_format = "date({year: date.year, month: date.month, day: date.day})"
        elif period == AggregationPeriod.WEEKLY:
            group_format = "date({year: date.year, week: date.week})"
        elif period == AggregationPeriod.MONTHLY:
            group_format = "date({year: date.year, month: date.month})"
        else:
            group_format = "date({year: date.year, month: date.month, day: date.day})"
        
        query = f"""
        MATCH (f:Facility)
        WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
        
        OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
        WHERE i.incident_date >= $start_date AND i.incident_date <= $end_date
        
        OPTIONAL MATCH (f)-[:HAS_VIOLATION]->(v:Violation)
        WHERE v.violation_date >= $start_date AND v.violation_date <= $end_date
        
        WITH {group_format} as period_date,
             count(DISTINCT i) as incidents,
             count(DISTINCT CASE WHEN i.severity = 'Critical' THEN i END) as critical_incidents,
             count(DISTINCT v) as violations,
             count(DISTINCT CASE WHEN v.severity = 'Critical' THEN v END) as critical_violations
        
        ORDER BY period_date
        
        RETURN period_date,
               incidents,
               critical_incidents,
               violations,
               critical_violations
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'facility_ids': facility_ids,
                'start_date': start_date,
                'end_date': end_date
            })
            
            records = list(result)
        
        # Convert to time series format
        incidents_trend = []
        violations_trend = []
        
        for record in records:
            date_str = record['period_date'].isoformat() if hasattr(record['period_date'], 'isoformat') else str(record['period_date'])
            
            incidents_trend.append({
                'date': date_str,
                'total': record['incidents'],
                'critical': record['critical_incidents']
            })
            
            violations_trend.append({
                'date': date_str,
                'total': record['violations'],
                'critical': record['critical_violations']
            })
        
        return {
            'incidents': incidents_trend,
            'violations': violations_trend
        }
    
    # Cross-facility Benchmarking
    
    @cache_result(expiry_minutes=30)
    def get_facility_benchmarks(self,
                              metric_names: List[str] = None,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> List[FacilityBenchmark]:
        """Generate cross-facility benchmarking data"""
        
        if not metric_names:
            metric_names = ['incident_rate', 'ltir', 'audit_pass_rate', 'training_completion_rate']
        
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=90)
        
        benchmarks = []
        
        # Get all facilities
        facility_query = """
        MATCH (f:Facility)
        RETURN f.facility_id as facility_id, f.facility_name as facility_name
        """
        
        with self.driver.session() as session:
            facilities = session.run(facility_query).data()
        
        for metric_name in metric_names:
            facility_values = []
            
            for facility in facilities:
                facility_id = facility['facility_id']
                
                if metric_name in ['incident_rate', 'ltir']:
                    kpis = self.calculate_safety_kpis([facility_id], start_date, end_date)
                    value = kpis.get(metric_name, KPIMetric(metric_name, 0, 'rate')).value
                elif metric_name in ['audit_pass_rate', 'training_completion_rate']:
                    kpis = self.calculate_compliance_kpis([facility_id], start_date, end_date)
                    value = kpis.get(metric_name, KPIMetric(metric_name, 0, 'percentage')).value
                else:
                    continue
                
                facility_values.append({
                    'facility_id': facility_id,
                    'facility_name': facility['facility_name'],
                    'value': float(value)
                })
            
            # Sort by value (ascending for rates, descending for percentages)
            if 'rate' in metric_name and metric_name != 'audit_pass_rate':
                facility_values.sort(key=lambda x: x['value'])
            else:
                facility_values.sort(key=lambda x: x['value'], reverse=True)
            
            # Calculate percentiles and rankings
            total_facilities = len(facility_values)
            values_only = [f['value'] for f in facility_values]
            industry_avg = sum(values_only) / total_facilities if total_facilities > 0 else 0
            best_practice = values_only[0] if values_only else 0
            
            for i, facility_data in enumerate(facility_values):
                rank = i + 1
                percentile = ((total_facilities - rank) / total_facilities) * 100
                
                benchmark = FacilityBenchmark(
                    facility_id=facility_data['facility_id'],
                    facility_name=facility_data['facility_name'],
                    metric_name=metric_name,
                    value=facility_data['value'],
                    rank=rank,
                    percentile=round(percentile, 2),
                    industry_average=round(industry_avg, 2),
                    best_practice=best_practice
                )
                benchmarks.append(benchmark)
        
        return benchmarks
    
    # Real-time Metric Calculations
    
    def get_real_time_metrics(self, facility_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate real-time metrics for live dashboard"""
        
        current_time = datetime.now()
        today = current_time.date()
        
        query = """
        MATCH (f:Facility)
        WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
        
        // Today's incidents
        OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
        WHERE date(i.incident_date) = date()
        
        // Active alerts
        OPTIONAL MATCH (f)-[:HAS_ALERT]->(a:Alert)
        WHERE a.status = 'Active'
        
        // Overdue items
        OPTIONAL MATCH (f)-[:HAS_TRAINING]->(t:Training)
        WHERE t.due_date < date() AND t.status <> 'Completed'
        
        OPTIONAL MATCH (f)-[:HAS_INSPECTION]->(ins:Inspection)
        WHERE ins.due_date < date() AND ins.status <> 'Completed'
        
        RETURN f.facility_id as facility_id,
               f.facility_name as facility_name,
               count(DISTINCT i) as todays_incidents,
               count(DISTINCT a) as active_alerts,
               count(DISTINCT t) as overdue_training,
               count(DISTINCT ins) as overdue_inspections
        """
        
        with self.driver.session() as session:
            result = session.run(query, {'facility_ids': facility_ids})
            records = list(result)
        
        # Aggregate real-time metrics
        total_incidents_today = sum(r['todays_incidents'] for r in records)
        total_active_alerts = sum(r['active_alerts'] for r in records)
        total_overdue_training = sum(r['overdue_training'] for r in records)
        total_overdue_inspections = sum(r['overdue_inspections'] for r in records)
        
        # Calculate status indicators
        alert_level = "GREEN"
        if total_active_alerts > 0 or total_incidents_today > 0:
            alert_level = "YELLOW"
        if total_active_alerts > 5 or total_incidents_today > 2:
            alert_level = "RED"
        
        return {
            'timestamp': current_time.isoformat(),
            'alert_level': alert_level,
            'metrics': {
                'todays_incidents': total_incidents_today,
                'active_alerts': total_active_alerts,
                'overdue_training': total_overdue_training,
                'overdue_inspections': total_overdue_inspections
            },
            'facility_breakdown': [dict(record) for record in records]
        }
    
    # Executive Summary Generation
    
    def generate_executive_summary(self,
                                 facility_ids: Optional[List[str]] = None,
                                 period: AggregationPeriod = AggregationPeriod.MONTHLY) -> Dict[str, Any]:
        """Generate comprehensive executive summary"""
        
        end_date = datetime.now()
        if period == AggregationPeriod.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif period == AggregationPeriod.QUARTERLY:
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=30)
        
        # Get all data components
        safety_kpis = self.calculate_safety_kpis(facility_ids, start_date, end_date)
        compliance_kpis = self.calculate_compliance_kpis(facility_ids, start_date, end_date)
        benchmarks = self.get_facility_benchmarks(start_date=start_date, end_date=end_date)
        top_risks = self.get_top_risks(facility_ids, limit=3)
        trending_data = self.get_trending_metrics(facility_ids, start_date, end_date, period)
        
        # Generate insights
        insights = self._generate_insights(safety_kpis, compliance_kpis, benchmarks, trending_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(safety_kpis, compliance_kpis, top_risks)
        
        return {
            'executive_summary': {
                'generated_at': datetime.now().isoformat(),
                'period': period.value,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            },
            'key_metrics': {
                **{k: v.__dict__ for k, v in safety_kpis.items()},
                **{k: v.__dict__ for k, v in compliance_kpis.items()}
            },
            'insights': insights,
            'recommendations': recommendations,
            'top_risks': top_risks,
            'performance_trends': trending_data
        }
    
    def _generate_insights(self, safety_kpis, compliance_kpis, benchmarks, trending_data) -> List[str]:
        """Generate automated insights from data"""
        insights = []
        
        # Safety insights
        incident_rate = safety_kpis.get('incident_rate')
        if incident_rate and incident_rate.value > incident_rate.threshold_critical:
            insights.append(f"Critical: Incident rate ({incident_rate.value}) exceeds critical threshold ({incident_rate.threshold_critical})")
        
        ltir = safety_kpis.get('ltir')
        if ltir and ltir.target and ltir.value <= ltir.target:
            insights.append(f"Positive: Lost Time Injury Rate ({ltir.value}) is meeting target ({ltir.target})")
        
        # Compliance insights
        audit_rate = compliance_kpis.get('audit_pass_rate')
        if audit_rate and audit_rate.target and audit_rate.value >= audit_rate.target:
            insights.append(f"Positive: Audit pass rate ({audit_rate.value}%) meets target ({audit_rate.target}%)")
        
        # Trending insights
        if trending_data.get('incidents'):
            recent_incidents = trending_data['incidents'][-7:]  # Last 7 periods
            if len(recent_incidents) >= 2:
                if recent_incidents[-1]['total'] > recent_incidents[-2]['total']:
                    insights.append("Concern: Recent upward trend in incident reporting")
                elif recent_incidents[-1]['total'] < recent_incidents[-2]['total']:
                    insights.append("Positive: Recent downward trend in incident reporting")
        
        return insights
    
    def _generate_recommendations(self, safety_kpis, compliance_kpis, top_risks) -> List[str]:
        """Generate automated recommendations"""
        recommendations = []
        
        # Safety recommendations
        incident_rate = safety_kpis.get('incident_rate')
        if incident_rate and incident_rate.value > incident_rate.threshold_warning:
            recommendations.append("Implement enhanced safety training programs and increase supervisor presence")
        
        severity_rate = safety_kpis.get('severity_rate')
        if severity_rate and severity_rate.value > severity_rate.threshold_warning:
            recommendations.append("Review and strengthen critical safety procedures and emergency response protocols")
        
        # Compliance recommendations
        training_rate = compliance_kpis.get('training_completion_rate')
        if training_rate and training_rate.value < training_rate.target:
            recommendations.append("Accelerate training completion through automated reminders and management escalation")
        
        violations = compliance_kpis.get('total_violations')
        if violations and violations.value > violations.threshold_warning:
            recommendations.append("Conduct comprehensive compliance audit and implement corrective action plan")
        
        # Risk-based recommendations
        if top_risks:
            high_risk_count = sum(1 for risk in top_risks if risk.get('risk_level') == 'High')
            if high_risk_count > 0:
                recommendations.append(f"Address {high_risk_count} high-priority risks through targeted mitigation strategies")
        
        return recommendations
    
    # Support Methods
    
    def get_top_risks(self, facility_ids: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top risks across facilities"""
        
        query = """
        MATCH (f:Facility)-[:HAS_RISK]->(r:Risk)
        WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
        AND r.status = 'Active'
        
        RETURN r.risk_id as risk_id,
               r.description as description,
               r.risk_level as risk_level,
               r.probability as probability,
               r.impact as impact,
               r.risk_score as risk_score,
               f.facility_name as facility_name
        ORDER BY r.risk_score DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'facility_ids': facility_ids,
                'limit': limit
            })
            return [dict(record) for record in result]
    
    def get_recent_alerts(self, facility_ids: Optional[List[str]] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts and notifications"""
        
        query = """
        MATCH (f:Facility)-[:HAS_ALERT]->(a:Alert)
        WHERE ($facility_ids IS NULL OR f.facility_id IN $facility_ids)
        AND a.created_date >= date() - duration('P7D')
        
        RETURN a.alert_id as alert_id,
               a.title as title,
               a.description as description,
               a.severity as severity,
               a.status as status,
               a.created_date as created_date,
               f.facility_name as facility_name
        ORDER BY a.created_date DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'facility_ids': facility_ids,
                'limit': limit
            })
            return [dict(record) for record in result]
    
    # Export Functions
    
    def export_dashboard_data(self,
                            data: Dict[str, Any],
                            format_type: ExportFormat = ExportFormat.JSON,
                            filename: Optional[str] = None) -> str:
        """Export dashboard data in specified format"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_export_{timestamp}"
        
        if format_type == ExportFormat.JSON:
            return self._export_to_json(data, filename)
        elif format_type == ExportFormat.CSV:
            return self._export_to_csv(data, filename)
        elif format_type == ExportFormat.EXCEL:
            return self._export_to_excel(data, filename)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """Export data to JSON format"""
        filepath = f"/tmp/{filename}.json"
        
        # Convert KPIMetric objects to dictionaries for JSON serialization
        json_data = self._prepare_data_for_export(data)
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        return filepath
    
    def _export_to_csv(self, data: Dict[str, Any], filename: str) -> str:
        """Export KPI data to CSV format"""
        filepath = f"/tmp/{filename}.csv"
        
        # Extract KPI data for CSV export
        kpi_data = []
        if 'kpis' in data:
            for kpi_name, kpi_metric in data['kpis'].items():
                kpi_data.append({
                    'metric_name': kpi_name,
                    'value': kpi_metric.value if hasattr(kpi_metric, 'value') else kpi_metric.get('value'),
                    'unit': kpi_metric.unit if hasattr(kpi_metric, 'unit') else kpi_metric.get('unit'),
                    'target': kpi_metric.target if hasattr(kpi_metric, 'target') else kpi_metric.get('target'),
                    'threshold_warning': kpi_metric.threshold_warning if hasattr(kpi_metric, 'threshold_warning') else kpi_metric.get('threshold_warning'),
                    'threshold_critical': kpi_metric.threshold_critical if hasattr(kpi_metric, 'threshold_critical') else kpi_metric.get('threshold_critical')
                })
        
        df = pd.DataFrame(kpi_data)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def _export_to_excel(self, data: Dict[str, Any], filename: str) -> str:
        """Export data to Excel format with multiple sheets"""
        filepath = f"/tmp/{filename}.xlsx"
        
        with pd.ExcelWriter(filepath) as writer:
            # KPIs sheet
            if 'kpis' in data:
                kpi_data = []
                for kpi_name, kpi_metric in data['kpis'].items():
                    kpi_data.append({
                        'metric_name': kpi_name,
                        'value': kpi_metric.value if hasattr(kpi_metric, 'value') else kpi_metric.get('value'),
                        'unit': kpi_metric.unit if hasattr(kpi_metric, 'unit') else kpi_metric.get('unit'),
                        'target': kpi_metric.target if hasattr(kpi_metric, 'target') else kpi_metric.get('target')
                    })
                
                kpi_df = pd.DataFrame(kpi_data)
                kpi_df.to_excel(writer, sheet_name='KPIs', index=False)
            
            # Facility Overview sheet
            if 'facility_overview' in data and data['facility_overview']:
                facility_df = pd.DataFrame(data['facility_overview'])
                facility_df.to_excel(writer, sheet_name='Facilities', index=False)
            
            # Trending Data sheet
            if 'trending_data' in data and data['trending_data'].get('incidents'):
                trend_df = pd.DataFrame(data['trending_data']['incidents'])
                trend_df.to_excel(writer, sheet_name='Trends', index=False)
        
        return filepath
    
    def _prepare_data_for_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JSON export by converting objects to dictionaries"""
        
        def convert_object(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_object(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_object(item) for item in obj]
            else:
                return obj
        
        return convert_object(data)
    
    # Cache Management
    
    def clear_cache(self):
        """Clear all cached results"""
        # This would clear the cache for all decorated methods
        # Implementation depends on the specific caching strategy
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_enabled': self.cache_enabled,
            'default_ttl_minutes': self.default_cache_ttl,
            'cached_methods': [
                'calculate_safety_kpis',
                'calculate_compliance_kpis',
                'get_dashboard_summary',
                'get_facility_benchmarks'
            ]
        }