"""
EHS-specific query examples for Text2Cypher retrieval.

This module provides comprehensive examples for each EHS intent type,
helping the LLM generate better Cypher queries for common EHS use cases.
"""

from typing import Dict, List, Any
from enum import Enum


class EHSExampleType(Enum):
    """Types of EHS examples."""
    CONSUMPTION_ANALYSIS = "consumption_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    RISK_ASSESSMENT = "risk_assessment"
    EMISSION_TRACKING = "emission_tracking"
    EQUIPMENT_EFFICIENCY = "equipment_efficiency"
    PERMIT_STATUS = "permit_status"
    GENERAL_INQUIRY = "general_inquiry"


def get_consumption_analysis_examples() -> List[Dict[str, str]]:
    """
    Get examples for consumption analysis queries.
    
    Returns:
        List of example query-cypher pairs for consumption analysis
    """
    return [
        {
            "question": "What is the total water consumption for all facilities last month?",
            "cypher": """
                MATCH (f:Facility)-[:RECORDED_AT]-(u:UtilityBill)
                WHERE u.utility_type = 'water' 
                  AND u.billing_period >= date('2024-07-01') 
                  AND u.billing_period < date('2024-08-01')
                RETURN f.name as facility_name, 
                       SUM(u.amount) as total_consumption,
                       u.unit as unit
                ORDER BY total_consumption DESC
                LIMIT 10
            """,
            "intent": "consumption_analysis",
            "description": "Aggregate water consumption by facility for a specific time period"
        },
        {
            "question": "Show electricity usage trends for Manufacturing Plant A over the past 6 months",
            "cypher": """
                MATCH (f:Facility {name: 'Manufacturing Plant A'})-[:RECORDED_AT]-(u:UtilityBill)
                WHERE u.utility_type = 'electricity' 
                  AND u.billing_period >= date() - duration({months: 6})
                RETURN u.billing_period as month, 
                       u.amount as consumption,
                       u.unit,
                       u.cost
                ORDER BY u.billing_period ASC
            """,
            "intent": "consumption_analysis",
            "description": "Track electricity consumption trends for a specific facility over time"
        },
        {
            "question": "Which facility has the highest gas consumption per square foot?",
            "cypher": """
                MATCH (f:Facility)-[:RECORDED_AT]-(u:UtilityBill)
                WHERE u.utility_type = 'natural_gas' 
                  AND f.square_footage IS NOT NULL
                  AND u.billing_period >= date() - duration({months: 1})
                WITH f, SUM(u.amount) as total_gas, f.square_footage as sqft
                RETURN f.name as facility_name,
                       total_gas,
                       sqft,
                       round(total_gas / sqft, 2) as consumption_per_sqft
                ORDER BY consumption_per_sqft DESC
                LIMIT 5
            """,
            "intent": "consumption_analysis",
            "description": "Calculate normalized gas consumption per facility area"
        },
        {
            "question": "Compare water usage between production and office facilities",
            "cypher": """
                MATCH (f:Facility)-[:RECORDED_AT]-(u:UtilityBill)
                WHERE u.utility_type = 'water' 
                  AND u.billing_period >= date() - duration({months: 3})
                  AND f.facility_type IN ['production', 'office']
                WITH f.facility_type as type, 
                     AVG(u.amount) as avg_consumption,
                     COUNT(u) as bill_count
                RETURN type,
                       round(avg_consumption, 2) as average_monthly_consumption,
                       bill_count as total_bills
                ORDER BY avg_consumption DESC
            """,
            "intent": "consumption_analysis",
            "description": "Compare consumption patterns between different facility types"
        },
        {
            "question": "Show energy consumption breakdown by utility type for Q1 2024",
            "cypher": """
                MATCH (f:Facility)-[:RECORDED_AT]-(u:UtilityBill)
                WHERE u.billing_period >= date('2024-01-01') 
                  AND u.billing_period < date('2024-04-01')
                  AND u.utility_type IN ['electricity', 'natural_gas', 'heating_oil']
                RETURN u.utility_type,
                       SUM(u.amount) as total_consumption,
                       u.unit,
                       SUM(u.cost) as total_cost,
                       COUNT(DISTINCT f) as facilities_count
                ORDER BY total_cost DESC
            """,
            "intent": "consumption_analysis",
            "description": "Analyze energy consumption by type across all facilities for a quarter"
        }
    ]


def get_compliance_check_examples() -> List[Dict[str, str]]:
    """
    Get examples for compliance check queries.
    
    Returns:
        List of example query-cypher pairs for compliance checks
    """
    return [
        {
            "question": "Which permits are expiring in the next 30 days?",
            "cypher": """
                MATCH (p:Permit)
                WHERE p.expiry_date <= date() + duration({days: 30})
                  AND p.expiry_date >= date()
                  AND p.status = 'active'
                RETURN p.permit_number,
                       p.type as permit_type,
                       p.expiry_date,
                       p.issuing_authority,
                       (p.expiry_date - date()).days as days_until_expiry
                ORDER BY p.expiry_date ASC
                LIMIT 20
            """,
            "intent": "compliance_check",
            "description": "Identify permits requiring immediate renewal attention"
        },
        {
            "question": "Show all expired permits that need immediate action",
            "cypher": """
                MATCH (p:Permit)
                WHERE p.expiry_date < date()
                  AND p.status IN ['active', 'pending_renewal']
                RETURN p.permit_number,
                       p.type,
                       p.expiry_date,
                       p.issuing_authority,
                       (date() - p.expiry_date).days as days_overdue,
                       p.status
                ORDER BY days_overdue DESC
            """,
            "intent": "compliance_check",
            "description": "Find overdue permits requiring immediate compliance action"
        },
        {
            "question": "What is the compliance status for Environmental permits at Facility X?",
            "cypher": """
                MATCH (f:Facility {name: 'Facility X'})-[:HOLDS]-(p:Permit)
                WHERE p.type CONTAINS 'Environmental' OR p.category = 'environmental'
                RETURN p.permit_number,
                       p.type,
                       p.status,
                       p.expiry_date,
                       p.issuing_authority,
                       CASE 
                         WHEN p.expiry_date < date() THEN 'EXPIRED'
                         WHEN p.expiry_date <= date() + duration({days: 90}) THEN 'EXPIRING_SOON'
                         ELSE 'VALID'
                       END as compliance_status
                ORDER BY p.expiry_date ASC
            """,
            "intent": "compliance_check",
            "description": "Check environmental permit compliance for a specific facility"
        },
        {
            "question": "List all permits by authority and their renewal requirements",
            "cypher": """
                MATCH (p:Permit)
                WHERE p.status = 'active'
                WITH p.issuing_authority as authority,
                     COUNT(p) as total_permits,
                     SUM(CASE WHEN p.expiry_date <= date() + duration({days: 90}) THEN 1 ELSE 0 END) as expiring_soon,
                     SUM(CASE WHEN p.expiry_date < date() THEN 1 ELSE 0 END) as expired
                RETURN authority,
                       total_permits,
                       expiring_soon,
                       expired,
                       round(100.0 * (expiring_soon + expired) / total_permits, 1) as compliance_risk_percentage
                ORDER BY compliance_risk_percentage DESC
            """,
            "intent": "compliance_check",
            "description": "Analyze compliance risk by regulatory authority"
        },
        {
            "question": "Show permit renewal history and patterns for the past year",
            "cypher": """
                MATCH (p:Permit)
                WHERE p.issue_date >= date() - duration({years: 1})
                  OR p.renewal_date >= date() - duration({years: 1})
                WITH p, 
                     COALESCE(p.renewal_date, p.issue_date) as activity_date
                RETURN extract(month from activity_date) as month,
                       COUNT(p) as permits_processed,
                       SUM(CASE WHEN p.renewal_date IS NOT NULL THEN 1 ELSE 0 END) as renewals,
                       SUM(CASE WHEN p.renewal_date IS NULL THEN 1 ELSE 0 END) as new_permits
                ORDER BY month ASC
            """,
            "intent": "compliance_check",
            "description": "Analyze permit activity patterns and renewal trends"
        }
    ]


def get_risk_assessment_examples() -> List[Dict[str, str]]:
    """
    Get examples for risk assessment queries.
    
    Returns:
        List of example query-cypher pairs for risk assessments
    """
    return [
        {
            "question": "What are the high-risk incidents in the past 6 months?",
            "cypher": """
                MATCH (i:Incident)
                WHERE i.incident_date >= date() - duration({months: 6})
                  AND i.severity_level IN ['high', 'critical']
                RETURN i.incident_id,
                       i.incident_date,
                       i.incident_type,
                       i.severity_level,
                       i.description,
                       i.location,
                       i.investigation_status
                ORDER BY i.incident_date DESC
                LIMIT 20
            """,
            "intent": "risk_assessment",
            "description": "Identify recent high-severity safety incidents"
        },
        {
            "question": "Which facilities have the highest incident rates?",
            "cypher": """
                MATCH (f:Facility)-[:LOCATION_OF]-(i:Incident)
                WHERE i.incident_date >= date() - duration({years: 1})
                WITH f, COUNT(i) as incident_count, f.employee_count as employees
                WHERE employees > 0
                RETURN f.name as facility_name,
                       incident_count,
                       employees,
                       round(1000.0 * incident_count / employees, 2) as incidents_per_1000_employees,
                       f.facility_type
                ORDER BY incidents_per_1000_employees DESC
                LIMIT 10
            """,
            "intent": "risk_assessment",
            "description": "Calculate and rank facility incident rates per employee"
        },
        {
            "question": "Show safety trends by incident type over time",
            "cypher": """
                MATCH (i:Incident)
                WHERE i.incident_date >= date() - duration({years: 2})
                WITH i.incident_type as type,
                     extract(year from i.incident_date) as year,
                     extract(quarter from i.incident_date) as quarter,
                     COUNT(i) as incident_count
                RETURN type,
                       year,
                       quarter,
                       incident_count
                ORDER BY year DESC, quarter DESC, incident_count DESC
            """,
            "intent": "risk_assessment",
            "description": "Analyze incident trends by type and time period"
        },
        {
            "question": "Identify equipment with recurring safety issues",
            "cypher": """
                MATCH (e:Equipment)-[:INVOLVED_IN]-(i:Incident)
                WHERE i.incident_date >= date() - duration({years: 1})
                  AND i.category IN ['safety', 'equipment_failure']
                WITH e, COUNT(i) as incident_count, 
                     COLLECT(i.incident_type) as incident_types
                WHERE incident_count >= 2
                RETURN e.equipment_id,
                       e.equipment_type,
                       e.manufacturer,
                       e.installation_date,
                       incident_count,
                       incident_types
                ORDER BY incident_count DESC
            """,
            "intent": "risk_assessment",
            "description": "Find equipment with multiple safety-related incidents"
        },
        {
            "question": "What are the root causes of incidents by facility type?",
            "cypher": """
                MATCH (f:Facility)-[:LOCATION_OF]-(i:Incident)
                WHERE i.incident_date >= date() - duration({years: 1})
                  AND i.root_cause IS NOT NULL
                WITH f.facility_type as facility_type,
                     i.root_cause as cause,
                     COUNT(i) as incident_count
                RETURN facility_type,
                       cause,
                       incident_count,
                       round(100.0 * incident_count / SUM(incident_count), 1) as percentage
                ORDER BY facility_type, incident_count DESC
            """,
            "intent": "risk_assessment",
            "description": "Analyze incident root causes by facility type"
        }
    ]


def get_emission_tracking_examples() -> List[Dict[str, str]]:
    """
    Get examples for emission tracking queries.
    
    Returns:
        List of example query-cypher pairs for emission tracking
    """
    return [
        {
            "question": "What are the total CO2 emissions for all facilities this year?",
            "cypher": """
                MATCH (f:Facility)-[:EMITS]-(e:Emission)
                WHERE e.measurement_date >= date('2024-01-01')
                  AND e.emission_type = 'CO2'
                RETURN f.name as facility_name,
                       SUM(e.quantity) as total_co2_emissions,
                       e.unit,
                       COUNT(e) as measurement_count
                ORDER BY total_co2_emissions DESC
            """,
            "intent": "emission_tracking",
            "description": "Calculate total CO2 emissions by facility for the current year"
        },
        {
            "question": "Show emission trends for NOx over the past two years",
            "cypher": """
                MATCH (e:Emission)
                WHERE e.emission_type = 'NOx'
                  AND e.measurement_date >= date() - duration({years: 2})
                WITH extract(year from e.measurement_date) as year,
                     extract(month from e.measurement_date) as month,
                     SUM(e.quantity) as monthly_emissions
                RETURN year, month, monthly_emissions
                ORDER BY year ASC, month ASC
            """,
            "intent": "emission_tracking",
            "description": "Track NOx emission trends over time"
        },
        {
            "question": "Which emission sources exceed regulatory limits?",
            "cypher": """
                MATCH (s:EmissionSource)-[:PRODUCES]-(e:Emission)
                WHERE e.measurement_date >= date() - duration({months: 3})
                WITH s, e.emission_type as type, 
                     AVG(e.quantity) as avg_emission,
                     s.regulatory_limit as limit
                WHERE limit IS NOT NULL AND avg_emission > limit
                RETURN s.source_id,
                       s.source_type,
                       type,
                       round(avg_emission, 2) as average_emission,
                       limit as regulatory_limit,
                       round(100.0 * (avg_emission - limit) / limit, 1) as percentage_over_limit
                ORDER BY percentage_over_limit DESC
            """,
            "intent": "emission_tracking",
            "description": "Identify emission sources exceeding regulatory compliance limits"
        },
        {
            "question": "Calculate total greenhouse gas emissions by scope",
            "cypher": """
                MATCH (e:Emission)
                WHERE e.measurement_date >= date() - duration({years: 1})
                  AND e.ghg_scope IS NOT NULL
                WITH e.ghg_scope as scope,
                     SUM(e.co2_equivalent) as total_co2_eq
                RETURN scope,
                       round(total_co2_eq, 2) as total_co2_equivalent_tons,
                       round(100.0 * total_co2_eq / SUM(total_co2_eq), 1) as percentage_of_total
                ORDER BY total_co2_eq DESC
            """,
            "intent": "emission_tracking",
            "description": "Break down greenhouse gas emissions by scope classification"
        },
        {
            "question": "Show particulate matter emissions by monitoring location",
            "cypher": """
                MATCH (l:MonitoringLocation)-[:MEASURES]-(e:Emission)
                WHERE e.emission_type CONTAINS 'PM'
                  AND e.measurement_date >= date() - duration({months: 6})
                RETURN l.location_id,
                       l.location_name,
                       e.emission_type,
                       AVG(e.quantity) as average_concentration,
                       MAX(e.quantity) as peak_concentration,
                       e.unit,
                       COUNT(e) as measurement_count
                ORDER BY average_concentration DESC
            """,
            "intent": "emission_tracking",
            "description": "Analyze particulate matter emissions by monitoring location"
        }
    ]


def get_equipment_efficiency_examples() -> List[Dict[str, str]]:
    """
    Get examples for equipment efficiency queries.
    
    Returns:
        List of example query-cypher pairs for equipment efficiency
    """
    return [
        {
            "question": "Which equipment has the lowest energy efficiency rating?",
            "cypher": """
                MATCH (e:Equipment)
                WHERE e.efficiency_rating IS NOT NULL
                  AND e.status = 'operational'
                RETURN e.equipment_id,
                       e.equipment_type,
                       e.manufacturer,
                       e.efficiency_rating,
                       e.installation_date,
                       e.last_maintenance_date
                ORDER BY e.efficiency_rating ASC
                LIMIT 10
            """,
            "intent": "equipment_efficiency",
            "description": "Identify equipment with poor efficiency ratings"
        },
        {
            "question": "Calculate energy consumption per unit of output for all motors",
            "cypher": """
                MATCH (e:Equipment {equipment_type: 'motor'})-[:CONSUMES]-(u:UtilityBill)
                MATCH (e)-[:PRODUCES]-(p:ProductionRecord)
                WHERE u.billing_period = p.production_period
                  AND u.utility_type = 'electricity'
                  AND p.output_quantity > 0
                WITH e, 
                     SUM(u.amount) as total_energy_consumed,
                     SUM(p.output_quantity) as total_output
                RETURN e.equipment_id,
                       e.manufacturer,
                       e.model,
                       round(total_energy_consumed / total_output, 3) as energy_per_unit,
                       total_energy_consumed,
                       total_output
                ORDER BY energy_per_unit DESC
            """,
            "intent": "equipment_efficiency",
            "description": "Calculate energy efficiency ratios for production equipment"
        },
        {
            "question": "Show equipment performance degradation over time",
            "cypher": """
                MATCH (e:Equipment)-[:HAS_MEASUREMENT]-(m:PerformanceMeasurement)
                WHERE m.measurement_date >= date() - duration({years: 1})
                  AND m.efficiency_percentage IS NOT NULL
                WITH e, 
                     extract(month from m.measurement_date) as month,
                     AVG(m.efficiency_percentage) as monthly_efficiency
                RETURN e.equipment_id,
                       e.equipment_type,
                       month,
                       round(monthly_efficiency, 2) as efficiency_percentage
                ORDER BY e.equipment_id, month
            """,
            "intent": "equipment_efficiency",
            "description": "Track equipment efficiency trends over time"
        },
        {
            "question": "Identify equipment due for efficiency optimization",
            "cypher": """
                MATCH (e:Equipment)
                WHERE e.last_efficiency_check < date() - duration({months: 6})
                  OR e.efficiency_rating < 0.75
                  OR e.maintenance_score < 3
                OPTIONAL MATCH (e)-[:HAS_INCIDENT]-(i:Incident)
                WHERE i.incident_date >= date() - duration({months: 12})
                  AND i.category = 'equipment_failure'
                WITH e, COUNT(i) as recent_failures
                RETURN e.equipment_id,
                       e.equipment_type,
                       e.efficiency_rating,
                       e.last_efficiency_check,
                       e.maintenance_score,
                       recent_failures,
                       CASE 
                         WHEN recent_failures > 2 THEN 'HIGH'
                         WHEN e.efficiency_rating < 0.6 THEN 'HIGH'
                         WHEN e.maintenance_score < 2 THEN 'MEDIUM'
                         ELSE 'LOW'
                       END as optimization_priority
                ORDER BY optimization_priority DESC, e.efficiency_rating ASC
            """,
            "intent": "equipment_efficiency",
            "description": "Prioritize equipment for efficiency optimization"
        },
        {
            "question": "Compare efficiency between different equipment manufacturers",
            "cypher": """
                MATCH (e:Equipment)
                WHERE e.efficiency_rating IS NOT NULL
                  AND e.installation_date >= date() - duration({years: 5})
                WITH e.manufacturer as manufacturer,
                     e.equipment_type as type,
                     AVG(e.efficiency_rating) as avg_efficiency,
                     COUNT(e) as equipment_count,
                     STDEV(e.efficiency_rating) as efficiency_stddev
                WHERE equipment_count >= 3
                RETURN manufacturer,
                       type,
                       round(avg_efficiency, 3) as average_efficiency,
                       equipment_count,
                       round(efficiency_stddev, 3) as efficiency_variation
                ORDER BY average_efficiency DESC
            """,
            "intent": "equipment_efficiency",
            "description": "Benchmark equipment efficiency by manufacturer and type"
        }
    ]


def get_permit_status_examples() -> List[Dict[str, str]]:
    """
    Get examples for permit status queries.
    
    Returns:
        List of example query-cypher pairs for permit status
    """
    return [
        {
            "question": "Show all active permits and their expiration dates",
            "cypher": """
                MATCH (p:Permit)
                WHERE p.status = 'active'
                RETURN p.permit_number,
                       p.type,
                       p.issuing_authority,
                       p.issue_date,
                       p.expiry_date,
                       (p.expiry_date - date()).days as days_until_expiry
                ORDER BY p.expiry_date ASC
            """,
            "intent": "permit_status",
            "description": "List all active permits with time to expiration"
        },
        {
            "question": "What permits does Facility Y currently hold?",
            "cypher": """
                MATCH (f:Facility {name: 'Facility Y'})-[:HOLDS]-(p:Permit)
                WHERE p.status IN ['active', 'pending_renewal']
                RETURN p.permit_number,
                       p.type,
                       p.category,
                       p.status,
                       p.issue_date,
                       p.expiry_date,
                       p.issuing_authority
                ORDER BY p.expiry_date ASC
            """,
            "intent": "permit_status",
            "description": "Show all permits held by a specific facility"
        },
        {
            "question": "List permits by status and authority",
            "cypher": """
                MATCH (p:Permit)
                WITH p.issuing_authority as authority,
                     p.status as status,
                     COUNT(p) as permit_count
                RETURN authority,
                       status,
                       permit_count
                ORDER BY authority, status
            """,
            "intent": "permit_status",
            "description": "Summarize permit counts by authority and status"
        },
        {
            "question": "Show permit renewal schedule for the next quarter",
            "cypher": """
                MATCH (p:Permit)
                WHERE p.expiry_date >= date()
                  AND p.expiry_date <= date() + duration({months: 3})
                  AND p.status = 'active'
                WITH p, 
                     extract(month from p.expiry_date) as expiry_month,
                     extract(week from p.expiry_date) as expiry_week
                RETURN expiry_month,
                       expiry_week,
                       COUNT(p) as permits_expiring,
                       COLLECT(p.permit_number) as permit_numbers,
                       COLLECT(DISTINCT p.issuing_authority) as authorities
                ORDER BY expiry_month, expiry_week
            """,
            "intent": "permit_status",
            "description": "Plan permit renewals by time period"
        },
        {
            "question": "Identify permits with compliance violations",
            "cypher": """
                MATCH (p:Permit)-[:HAS_VIOLATION]-(v:Violation)
                WHERE v.violation_date >= date() - duration({years: 1})
                  AND v.status IN ['open', 'pending']
                RETURN p.permit_number,
                       p.type,
                       p.issuing_authority,
                       v.violation_type,
                       v.violation_date,
                       v.severity,
                       v.status as violation_status,
                       v.resolution_deadline
                ORDER BY v.violation_date DESC
            """,
            "intent": "permit_status",
            "description": "Track permits with outstanding compliance violations"
        }
    ]


def get_general_inquiry_examples() -> List[Dict[str, str]]:
    """
    Get examples for general EHS inquiries.
    
    Returns:
        List of example query-cypher pairs for general inquiries
    """
    return [
        {
            "question": "What facilities do we operate and what types are they?",
            "cypher": """
                MATCH (f:Facility)
                RETURN f.name,
                       f.facility_type,
                       f.location,
                       f.square_footage,
                       f.employee_count,
                       f.operational_status
                ORDER BY f.name
            """,
            "intent": "general_inquiry",
            "description": "Overview of all facilities and their basic information"
        },
        {
            "question": "Show me a summary of our EHS data for the dashboard",
            "cypher": """
                MATCH (f:Facility)
                OPTIONAL MATCH (f)-[:RECORDED_AT]-(u:UtilityBill)
                WHERE u.billing_period >= date() - duration({months: 1})
                OPTIONAL MATCH (f)-[:HOLDS]-(p:Permit)
                WHERE p.status = 'active'
                OPTIONAL MATCH (f)-[:LOCATION_OF]-(i:Incident)
                WHERE i.incident_date >= date() - duration({months: 1})
                RETURN f.name as facility_name,
                       COUNT(DISTINCT u) as utility_bills_count,
                       COUNT(DISTINCT p) as active_permits_count,
                       COUNT(DISTINCT i) as recent_incidents_count,
                       f.facility_type
                ORDER BY f.name
            """,
            "intent": "general_inquiry",
            "description": "High-level EHS metrics summary for dashboard display"
        },
        {
            "question": "What types of equipment do we have across all facilities?",
            "cypher": """
                MATCH (e:Equipment)
                WITH e.equipment_type as type,
                     COUNT(e) as equipment_count,
                     COUNT(DISTINCT e.manufacturer) as manufacturer_count,
                     AVG(e.efficiency_rating) as avg_efficiency
                WHERE equipment_count > 0
                RETURN type,
                       equipment_count,
                       manufacturer_count,
                       round(avg_efficiency, 2) as average_efficiency_rating
                ORDER BY equipment_count DESC
            """,
            "intent": "general_inquiry",
            "description": "Inventory summary of equipment types and their characteristics"
        },
        {
            "question": "Show me the relationship between facilities and their utilities",
            "cypher": """
                MATCH (f:Facility)-[:RECORDED_AT]-(u:UtilityBill)
                WHERE u.billing_period >= date() - duration({months: 3})
                WITH f, 
                     COLLECT(DISTINCT u.utility_type) as utility_types,
                     COUNT(u) as total_bills,
                     SUM(u.cost) as total_cost
                RETURN f.name as facility_name,
                       f.facility_type,
                       utility_types,
                       total_bills,
                       round(total_cost, 2) as total_utility_cost
                ORDER BY total_cost DESC
            """,
            "intent": "general_inquiry",
            "description": "Overview of facility-utility relationships and costs"
        },
        {
            "question": "What is our overall environmental footprint?",
            "cypher": """
                MATCH (e:Emission)
                WHERE e.measurement_date >= date() - duration({years: 1})
                WITH e.emission_type as type,
                     SUM(e.quantity) as total_quantity,
                     e.unit as unit,
                     COUNT(DISTINCT e.facility_id) as facilities_with_emissions
                RETURN type,
                       round(total_quantity, 2) as total_emissions,
                       unit,
                       facilities_with_emissions
                ORDER BY total_quantity DESC
            """,
            "intent": "general_inquiry",
            "description": "Environmental footprint summary across all emission types"
        }
    ]


def get_examples_by_intent(intent: EHSExampleType) -> List[Dict[str, str]]:
    """
    Get examples for a specific EHS intent.
    
    Args:
        intent: The EHS intent type
        
    Returns:
        List of examples for the specified intent
    """
    example_functions = {
        EHSExampleType.CONSUMPTION_ANALYSIS: get_consumption_analysis_examples,
        EHSExampleType.COMPLIANCE_CHECK: get_compliance_check_examples,
        EHSExampleType.RISK_ASSESSMENT: get_risk_assessment_examples,
        EHSExampleType.EMISSION_TRACKING: get_emission_tracking_examples,
        EHSExampleType.EQUIPMENT_EFFICIENCY: get_equipment_efficiency_examples,
        EHSExampleType.PERMIT_STATUS: get_permit_status_examples,
        EHSExampleType.GENERAL_INQUIRY: get_general_inquiry_examples,
    }
    
    function = example_functions.get(intent, get_general_inquiry_examples)
    return function()


def get_all_examples() -> List[Dict[str, str]]:
    """
    Get all EHS examples across all intent types.
    
    Returns:
        Combined list of all examples
    """
    all_examples = []
    
    for intent in EHSExampleType:
        examples = get_examples_by_intent(intent)
        all_examples.extend(examples)
    
    return all_examples


def get_example_summary() -> Dict[str, Any]:
    """
    Get a summary of available examples.
    
    Returns:
        Dictionary with example counts and categories
    """
    summary = {
        "total_examples": 0,
        "examples_by_intent": {},
        "intent_types": [intent.value for intent in EHSExampleType]
    }
    
    for intent in EHSExampleType:
        examples = get_examples_by_intent(intent)
        count = len(examples)
        summary["examples_by_intent"][intent.value] = count
        summary["total_examples"] += count
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    # Print summary of available examples
    summary = get_example_summary()
    print("EHS Query Examples Summary:")
    print(f"Total examples: {summary['total_examples']}")
    print("\nExamples by intent:")
    for intent, count in summary["examples_by_intent"].items():
        print(f"  {intent}: {count} examples")
    
    # Show a sample example
    consumption_examples = get_consumption_analysis_examples()
    if consumption_examples:
        print(f"\nSample consumption analysis example:")
        example = consumption_examples[0]
        print(f"Question: {example['question']}")
        print(f"Intent: {example['intent']}")
        print(f"Description: {example['description']}")
        print(f"Cypher: {example['cypher']}")