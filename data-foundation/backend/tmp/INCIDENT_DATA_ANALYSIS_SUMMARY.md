# Incident Data Analysis Summary

## Overview
**Date:** 2025-08-30  
**Test ID:** incident_query_20250830_113031  
**Status:** ✅ Completed Successfully  

## Key Findings

### Database Statistics
- **Total Incident Nodes:** 114
- **Total Outgoing Relationships:** 110 
- **Total Incoming Relationships:** 4
- **Total Location Connections:** 258

### Incident Classifications

#### Incident Types Found:
1. **Environmental Release** - Environmental incidents
2. **Equipment Injury** - Equipment-related injuries
3. **Chemical Exposure** - Chemical exposure incidents  
4. **Slip/Trip/Fall** - Physical injury incidents

#### Severity Levels:
1. **Minor/low** - Low severity incidents
2. **Moderate/medium** - Medium severity incidents

### Data Structure Analysis

#### Two Different Incident Data Models:
The database contains incidents with two different data structures:

**Model 1 - Legacy/Demo Format:**
```
Properties: [status, facility_id, description, reported_date, severity, type, incident_id]
- incident_id: "DEMO_FACILITY_001_incident_001", "DEMO_FACILITY_001_incident_002"
- facility_id: "DEMO_FACILITY_001"
- Simple type classification: "environmental", "safety"
```

**Model 2 - Enhanced Format:**
```
Properties: [incident_type, updated_at, building_name, days_lost, created_at, 
           incident_date, source, floor_name, area_name, injured_person_count, 
           incident_id, description, status, severity, root_cause, 
           reported_date, area_type, site_code, corrective_action, cost_estimate]
- incident_id: "INC-202508-[UUID]" format
- Rich location data: site_code, building_name, floor_name, area_name
- Detailed tracking: injured_person_count, days_lost, cost_estimate
- Process data: root_cause, corrective_action
- Source tracking: source = 'test_data'
```

### Location Relationships

#### Direct Location Properties in Incidents:
Many incidents contain embedded location information:
- **site_code**: "ALG001", "HOU001"
- **building_name**: Various building names
- **floor_name**: Floor identification
- **area_name**: Specific area where incident occurred
- **area_type**: Type classification of the area

#### Associated Locations (31 unique areas):
- Production Line A & B
- Quality Control Lab
- Raw Materials Storage
- Finished Goods Storage
- Shipping & Receiving
- Chemical Storage
- Plant Office
- Conference Rooms
- Training Center
- Data Center
- Executive Assistant Area
- And 20+ more locations

### Location Hierarchy Structure

#### Sites (2 sites):
1. **Algonquin Manufacturing Site** (ALG001)
2. **Houston Corporate Campus** (HOU001)

#### Buildings (5 buildings):
- Main Manufacturing Building
- Warehouse
- Corporate Tower
- Utility Building
- Parking Garage

#### Floors (8 floors):
- Production Floor
- Mezzanine Level
- Main Warehouse Floor
- Ground Level/Ground Floor Lobby
- Second Floor
- Executive Floor

#### Areas (31 areas):
Comprehensive coverage of operational areas from production to office spaces.

## Relationship Patterns

### Primary Relationship: OCCURRED_AT
Most incidents use the `OCCURRED_AT` relationship to connect to Area nodes:
```
(Incident)-[:OCCURRED_AT]->(Area)
```

### Location Connections Found: 258 total
The system shows strong connectivity between incidents and location hierarchy through multiple patterns:
1. Direct area relationships via `OCCURRED_AT`
2. Embedded location properties within incident nodes
3. Hierarchical structure: Site → Building → Floor → Area

## Data Model Recommendations

### Strengths:
1. **Rich Location Context** - Incidents are well-connected to specific locations
2. **Comprehensive Metadata** - Enhanced model includes cost, injuries, root cause analysis
3. **Hierarchical Structure** - Clear site/building/floor/area organization
4. **Process Tracking** - Status, corrective actions, and follow-up data

### Observations:
1. **Mixed Data Models** - Two different incident formats exist (legacy vs enhanced)
2. **Test Data Dominance** - Most recent incidents (110+) are test data with source='test_data'
3. **Strong Location Embedding** - Location data is both referenced and embedded for redundancy

## Testing Results
✅ **Query All Incidents**: Successfully retrieved 114 incidents  
✅ **Analyze Relationships**: Found 114 total relationships  
✅ **Check Facility Connections**: Identified 258 location connections  
✅ **Analyze Data Model**: Documented property structures  
✅ **Check Location Hierarchy**: Confirmed complete hierarchy  

## Files Generated
- **Detailed JSON Report**: `incident_analysis_report_incident_query_20250830_113031.json`
- **Complete Log**: `incident_query_test_incident_query_20250830_113031.log`
- **Test Script**: `test_incident_query.py`

## Conclusion
The incident data is well-structured and properly connected to the location hierarchy. The system supports comprehensive incident tracking with rich metadata, location context, and process management capabilities. The presence of both legacy and enhanced data models suggests the system has evolved over time to support more detailed incident analysis.
