# Neo4j Database Catalog

**Generated:** 2025-09-08T21:52:56.259620
**Neo4j URI:** bolt://localhost:7687
**Database:** neo4j

## Table of Contents

- [Database Information](#database-information)
- [Summary Statistics](#summary-statistics)
- [Node Labels](#node-labels)
- [Relationship Types](#relationship-types)
- [Constraints and Indexes](#constraints-and-indexes)
- [Sample Data](#sample-data)

## Database Information

### Neo4j Components

- **Neo4j Kernel:** ['5.23.0'] (community)

## Summary Statistics

- **Total Node Labels:** 53
- **Total Relationship Types:** 33
- **Total Nodes:** 6393
- **Total Relationships:** 12672

## Node Labels

### Node Label Counts

| Label | Count |
|-------|-------|
| Anomaly | 0 |
| ApprovalWorkflow | 0 |
| Area | 39 |
| AssessmentReport | 1 |
| Building | 6 |
| ChangePoint | 0 |
| ComplianceRecord | 10 |
| Customer | 4 |
| DataPoint | 0 |
| DisposalFacility | 2 |
| Document | 3 |
| DocumentChunk | 0 |
| EHSMetric | 5031 |
| EffectivenessMeasurement | 0 |
| ElectricityConsumption | 309 |
| Electricitybill | 1 |
| Emission | 65 |
| EmissionFactor | 1 |
| EnvironmentalKPI | 20 |
| EnvironmentalRecommendation | 8 |
| EnvironmentalRisk | 12 |
| EnvironmentalTarget | 6 |
| Equipment | 0 |
| Facility | 15 |
| Floor | 12 |
| Goal | 6 |
| Incident | 116 |
| Meter | 6 |
| Migration | 0 |
| MonthlyUsageAllocation | 0 |
| Permit | 0 |
| Recommendation | 19 |
| RejectedDocument | 1 |
| RejectionRecord | 0 |
| RiskAssessment | 2 |
| SeasonalDecomposition | 0 |
| Site | 4 |
| Transporter | 2 |
| TrendAnalysis | 0 |
| User | 0 |
| UtilityBill | 2 |
| UtilityProvider | 2 |
| WasteGeneration | 248 |
| WasteGenerator | 2 |
| WasteItem | 2 |
| WasteManifest | 2 |
| WasteShipment | 2 |
| Wastemanifest | 1 |
| WaterBill | 2 |
| WaterConsumption | 428 |
| Waterbill | 1 |
| __Entity__ | 0 |
| __Node__ | 0 |

### Node Properties

#### Anomaly
No properties found.

#### ApprovalWorkflow
No properties found.

#### Area
Properties:
- air_pressure_psi
- area_sqft
- backup_generator
- capacity
- capacity_units_per_hour
- certification
- charging_ports
- created_at
- dock_doors
- equipment_count
- floor_id
- hazmat_certified
- monitoring_stations
- name
- parking_spaces
- primary_process
- private_offices
- reception_desks
- rooms
- server_racks
- source
- storage_capacity_tons
- temperature_controlled
- testing_capacity
- treatment_capacity_gpd
- trucks_per_day
- type
- updated_at
- ventilation_type
- video_conferencing
- visitor_capacity
- voltage_primary
- workstations

#### AssessmentReport
Properties:
- assessment_id
- data_sources
- executive_summary
- facility_id
- id
- metadata
- report_date

#### Building
Properties:
- area_sqft
- built_year
- code
- created_at
- name
- site_code
- source
- type
- updated_at

#### ChangePoint
No properties found.

#### ComplianceRecord
Properties:
- compliance_date
- corrective_actions
- created_at
- expiry_date
- facility_id
- findings
- fine_amount_usd
- id
- inspection_date
- inspector_name
- regulation_code
- regulation_name
- status
- updated_at

#### Customer
Properties:
- id
- name

#### DataPoint
No properties found.

#### DisposalFacility
Properties:
- address
- disposal_methods
- epa_id
- id
- name
- permit_number

#### Document
Properties:
- batch_processing
- document_type
- file_hash
- file_path
- id
- original_filename
- output_dir
- recognition_confidence
- statement_date
- type
- uploaded_at

#### DocumentChunk
No properties found.

#### EHSMetric
Properties:
- area_name
- building_name
- calculation
- category
- created_at
- floor_name
- frequency
- metric_code
- metric_name
- recorded_date
- site_code
- source
- target_value
- unit
- value

#### EffectivenessMeasurement
No properties found.

#### ElectricityConsumption
Properties:
- co2_emissions
- consumption_kwh
- cost_per_kwh
- cost_usd
- created_at
- data_source
- date
- equipment_notes
- facility_id
- id
- meter_reading
- month
- off_peak_consumption_kwh
- peak_demand_kw
- site_id
- unit
- updated_at
- weather_factor
- year

#### Electricitybill
Properties:
- batch_processing
- document_type
- file_hash
- file_path
- id
- original_filename
- output_dir
- recognition_confidence
- statement_date
- type
- uploaded_at

#### Emission
Properties:
- amount
- calculation_method
- consumption_id
- created_at
- date
- disposal_method
- emission_factor
- facility_id
- id
- pollutant
- source_type
- unit

#### EmissionFactor
Properties:
- created_at
- grid_subregion
- id
- methodology
- region
- source
- unit
- updated_at
- value
- year

#### EnvironmentalKPI
Properties:
- created_at
- date
- facility_id
- id
- performance_status
- target_value
- type
- unit
- updated_at
- value
- variance_percentage

#### EnvironmentalRecommendation
Properties:
- created_at
- created_date
- description
- estimated_cost_usd
- estimated_savings_usd
- facility_id
- id
- implementation_timeline_days
- priority
- status
- target_completion_date
- title
- type
- updated_at

#### EnvironmentalRisk
Properties:
- created_at
- description
- facility_id
- id
- identified_date
- impact_score
- last_assessment_date
- mitigation_status
- probability
- risk_type
- severity
- updated_at

#### EnvironmentalTarget
Properties:
- created_at
- deadline
- description
- id
- site_id
- status
- target_type
- target_unit
- target_value
- updated_at

#### Equipment
No properties found.

#### Facility
Properties:
- address
- capacity
- created_at
- created_date
- employee_count
- established_date
- facility_id
- id
- industry_sector
- location
- name
- operational_status
- source
- test_facility
- type
- updated_at

#### Floor
Properties:
- area_sqft
- building_id
- ceiling_height
- created_at
- level
- name
- source
- updated_at

#### Goal
Properties:
- baseline_year
- category
- created_at
- description
- id
- period
- site_id
- target_date
- target_value
- target_year
- unit
- updated_at

#### Incident
Properties:
- area_name
- area_type
- building_name
- corrective_action
- cost_estimate
- created_at
- date
- days_lost
- description
- facility_id
- floor_name
- id
- incident_date
- incident_id
- incident_type
- injured_person_count
- reported_date
- root_cause
- severity
- site_code
- source
- status
- type
- updated_at

#### Meter
Properties:
- current_reading
- id
- previous_reading
- service_type
- type
- unit
- usage

#### Migration
No properties found.

#### MonthlyUsageAllocation
No properties found.

#### Permit
No properties found.

#### Recommendation
Properties:
- category
- created_date
- description
- estimated_impact
- id
- priority
- site_id
- title

#### RejectedDocument
Properties:
- attempted_type
- confidence
- content_length
- file_path
- file_size
- id
- key_terms
- original_filename
- page_count
- rejected_at
- rejection_reason
- tables_found
- upload_source
- upload_timestamp

#### RejectionRecord
No properties found.

#### RiskAssessment
Properties:
- assessment_date
- category
- confidence_score
- description
- factors
- id
- risk_level
- site_id

#### SeasonalDecomposition
No properties found.

#### Site
Properties:
- address
- certifications
- code
- country
- created_at
- electricity_baseline_kwh
- electricity_target_change
- employee_count
- employees
- established
- facility_type
- id
- location
- name
- operating_hours
- operating_hours_per_day
- recycling_target_rate
- risk_profile
- source
- state
- type
- updated_at
- waste_baseline_lbs
- water_baseline_gallons
- zip_code

#### Transporter
Properties:
- address
- id
- license_number
- name
- phone

#### TrendAnalysis
No properties found.

#### User
No properties found.

#### UtilityBill
Properties:
- base_service_charge
- billing_period_end
- billing_period_start
- due_date
- grid_infrastructure_fee
- id
- off_peak_kwh
- peak_kwh
- state_environmental_surcharge
- total_cost
- total_kwh

#### UtilityProvider
Properties:
- address
- id
- name
- phone
- website

#### WasteGeneration
Properties:
- amount_pounds
- contractor
- created_at
- date
- disposal_cost_usd
- disposal_method
- facility_id
- id
- performance_notes
- recycling_rate_achieved
- recycling_target
- site_id
- updated_at
- waste_type

#### WasteGenerator
Properties:
- address
- contact
- epa_id
- id
- name
- phone

#### WasteItem
Properties:
- container_count
- container_quantity
- container_type
- description
- hazard_class
- id
- proper_shipping_name
- quantity
- waste_type
- weight_unit

#### WasteManifest
Properties:
- id
- issue_date
- manifest_tracking_number
- status
- total_quantity
- total_weight
- weight_unit

#### WasteShipment
Properties:
- id
- shipment_date
- status
- total_weight
- transport_method
- weight_unit

#### Wastemanifest
Properties:
- batch_processing
- document_type
- file_hash
- file_path
- id
- original_filename
- output_dir
- recognition_confidence
- type
- uploaded_at

#### WaterBill
Properties:
- billing_period_end
- billing_period_start
- conservation_tax
- due_date
- id
- infrastructure_surcharge
- sewer_service_charge
- stormwater_fee
- total_cost
- total_gallons
- water_consumption_cost

#### WaterConsumption
Properties:
- consumption_gallons
- cooling_usage_gallons
- cost_per_gallon
- cost_usd
- created_at
- date
- domestic_usage_gallons
- facility_id
- id
- process_usage_gallons
- quality_rating
- seasonal_notes
- site_id
- source_type
- updated_at

#### Waterbill
Properties:
- batch_processing
- document_type
- file_hash
- file_path
- id
- original_filename
- output_dir
- recognition_confidence
- type
- uploaded_at

#### __Entity__
No properties found.

#### __Node__
No properties found.

## Relationship Types

### Relationship Type Counts

| Type | Count |
|------|-------|
| AFFECTS_CONSUMPTION | 0 |
| APPLIES_TO | 12 |
| BILLED_FOR | 22 |
| BILLED_TO | 6 |
| CONTAINS | 114 |
| CONTAINS_WASTE | 4 |
| DISPOSED_AT | 6 |
| DOCUMENTS | 4 |
| EXTRACTED_TO | 4 |
| GENERATED_BY | 6 |
| GENERATES_EMISSION | 118 |
| GENERATES_WASTE | 496 |
| HAS_COMPLIANCE_RECORD | 20 |
| HAS_ELECTRICITY_CONSUMPTION | 500 |
| HAS_EMISSION_FACTOR | 4 |
| HAS_ENVIRONMENTAL_KPI | 40 |
| HAS_ENVIRONMENTAL_RISK | 24 |
| HAS_INCIDENT | 12 |
| HAS_RECOMMENDATION | 54 |
| HAS_RISK | 4 |
| HAS_TARGET | 12 |
| HAS_WATER_CONSUMPTION | 856 |
| LOCATED_AT | 0 |
| LOCATED_IN | 18 |
| MEASURED_AT | 10062 |
| MONITORS | 10 |
| OCCURRED_AT | 220 |
| PERMITS | 0 |
| PROVIDED_BY | 6 |
| RECORDED_IN | 18 |
| RESULTED_IN | 12 |
| TRACKS | 2 |
| TRANSPORTED_BY | 6 |

### Relationship Properties

#### AFFECTS_CONSUMPTION
No properties found.

#### APPLIES_TO
No properties found.

#### BILLED_FOR
No properties found.

#### BILLED_TO
No properties found.

#### CONTAINS
No properties found.

#### CONTAINS_WASTE
Properties:
- quantity
- unit

#### DISPOSED_AT
Properties:
- disposal_method

#### DOCUMENTS
No properties found.

#### EXTRACTED_TO
Properties:
- extraction_date

#### GENERATED_BY
No properties found.

#### GENERATES_EMISSION
No properties found.

#### GENERATES_WASTE
No properties found.

#### HAS_COMPLIANCE_RECORD
No properties found.

#### HAS_ELECTRICITY_CONSUMPTION
No properties found.

#### HAS_EMISSION_FACTOR
No properties found.

#### HAS_ENVIRONMENTAL_KPI
No properties found.

#### HAS_ENVIRONMENTAL_RISK
No properties found.

#### HAS_INCIDENT
No properties found.

#### HAS_RECOMMENDATION
No properties found.

#### HAS_RISK
No properties found.

#### HAS_TARGET
No properties found.

#### HAS_WATER_CONSUMPTION
No properties found.

#### LOCATED_AT
No properties found.

#### LOCATED_IN
No properties found.

#### MEASURED_AT
No properties found.

#### MONITORS
No properties found.

#### OCCURRED_AT
No properties found.

#### PERMITS
No properties found.

#### PROVIDED_BY
No properties found.

#### RECORDED_IN
Properties:
- reading_date

#### RESULTED_IN
Properties:
- calculation_method

#### TRACKS
Properties:
- tracking_date

#### TRANSPORTED_BY
Properties:
- transport_date

## Constraints and Indexes

### Constraints

- **anomaly_id_unique:** No description
- **approval_workflow_id_unique:** No description
- **area_floor_name_unique:** No description
- **building_site_name_unique:** No description
- **change_point_id_unique:** No description
- **compliance_record_id:** No description
- **constraint_907a464e:** No description
- **constraint_ec67c859:** No description
- **decomposition_id_unique:** No description
- **effectiveness_measurement_id_unique:** No description
- **electricity_consumption_id:** No description
- **environmental_kpi_id:** No description
- **environmental_recommendation_id:** No description
- **environmental_risk_id:** No description
- **equipment_id_unique:** No description
- **floor_building_name_unique:** No description
- **goal_id_unique:** No description
- **migration_name_unique:** No description
- **permit_id_unique:** No description
- **recommendation_id_unique:** No description
- **rejection_id_unique:** No description
- **site_code_unique:** No description
- **site_name_unique:** No description
- **trend_analysis_id_unique:** No description
- **unique_allocation_id:** No description
- **unique_original_filename:** No description
- **unique_user_id:** No description
- **waste_generation_id:** No description
- **water_consumption_id:** No description

### Indexes

- **anomaly_id_unique:** No description
- **anomaly_metric_index:** No description
- **anomaly_severity_index:** No description
- **anomaly_timestamp_index:** No description
- **approval_workflow_id_unique:** No description
- **area_floor_id_index:** No description
- **area_floor_name_unique:** No description
- **building_site_code_index:** No description
- **building_site_name_unique:** No description
- **change_point_id_unique:** No description
- **change_point_metric_index:** No description
- **change_point_timestamp_index:** No description
- **compliance_record_id:** No description
- **compliance_status_idx:** No description
- **constraint_907a464e:** No description
- **constraint_ec67c859:** No description
- **data_point_metric_index:** No description
- **data_point_timestamp_index:** No description
- **decomposition_id_unique:** No description
- **effectiveness_measurement_id_unique:** No description
- **ehs_chunk_content_fulltext_idx:** No description
- **ehs_chunk_content_vector_idx:** No description
- **ehs_compliance_terms_fulltext_idx:** No description
- **ehs_document_content_fulltext_idx:** No description
- **ehs_document_content_vector_idx:** No description
- **ehs_document_metadata_fulltext_idx:** No description
- **ehs_document_summary_vector_idx:** No description
- **ehs_document_title_vector_idx:** No description
- **ehs_documents:** No description
- **ehs_environmental_terms_fulltext_idx:** No description
- **ehs_equipment_desc_vector_idx:** No description
- **ehs_equipment_model_fulltext_idx:** No description
- **ehs_equipment_specs_fulltext_idx:** No description
- **ehs_facility_desc_vector_idx:** No description
- **ehs_facility_description_fulltext_idx:** No description
- **ehs_facility_name_fulltext_idx:** No description
- **ehs_permit_compliance_fulltext_idx:** No description
- **ehs_permit_desc_vector_idx:** No description
- **ehs_permit_description_fulltext_idx:** No description
- **electricity_consumption_id:** No description
- **electricity_date_idx:** No description
- **electricity_facility_idx:** No description
- **emission_date:** No description
- **emission_id:** No description
- **entity:** No description
- **environmental_kpi_id:** No description
- **environmental_recommendation_id:** No description
- **environmental_risk_id:** No description
- **equipment_affects_consumption_index:** No description
- **equipment_id_unique:** No description
- **equipment_installation_date_index:** No description
- **equipment_located_at_index:** No description
- **equipment_model_index:** No description
- **equipment_type_index:** No description
- **facility_id:** No description
- **facility_name:** No description
- **floor_building_id_index:** No description
- **floor_building_name_unique:** No description
- **goal_category_index:** No description
- **goal_id_unique:** No description
- **goal_period_index:** No description
- **goal_target_date_index:** No description
- **idx_allocated_usage:** No description
- **idx_allocation_percentage:** No description
- **idx_allocation_year_month:** No description
- **idx_document_status:** No description
- **idx_rejected_at:** No description
- **idx_rejected_by_user_id:** No description
- **idx_rejection_reason:** No description
- **idx_source_file_path:** No description
- **index_343aff4e:** No description
- **index_f7700477:** No description
- **kpi_date_idx:** No description
- **kpi_type_idx:** No description
- **migration_name_unique:** No description
- **permit_authority_index:** No description
- **permit_expiration_date_index:** No description
- **permit_id_unique:** No description
- **permit_permits_index:** No description
- **permit_type_index:** No description
- **permit_unit_index:** No description
- **recommendation_assigned_to_idx:** No description
- **recommendation_created_date_idx:** No description
- **recommendation_department_idx:** No description
- **recommendation_due_date_idx:** No description
- **recommendation_id_unique:** No description
- **recommendation_priority_idx:** No description
- **recommendation_status_idx:** No description
- **recommendation_type_idx:** No description
- **rejection_created_at:** No description
- **rejection_document_id:** No description
- **rejection_id_unique:** No description
- **rejection_status:** No description
- **risk_severity_idx:** No description
- **site_code_unique:** No description
- **site_name_unique:** No description
- **trend_analysis_date_index:** No description
- **trend_analysis_id_unique:** No description
- **trend_analysis_metric_index:** No description
- **unique_allocation_id:** No description
- **unique_original_filename:** No description
- **unique_user_id:** No description
- **utility_bill_id:** No description
- **utility_bill_time:** No description
- **waste_date_idx:** No description
- **waste_facility_idx:** No description
- **waste_generation_id:** No description
- **water_consumption_id:** No description
- **water_date_idx:** No description
- **water_facility_idx:** No description

## Sample Data

### Sample Nodes

#### Anomaly (Sample)

No sample data available.

#### ApprovalWorkflow (Sample)

No sample data available.

#### Area (Sample)

**Sample 1:**
```json
{
  "parking_spaces": 100,
  "updated_at": "2025-08-30T15:07:36.173000000+00:00",
  "name": "Visitor Parking",
  "created_at": "2025-08-30T15:07:36.173000000+00:00",
  "floor_id": "4:d7c9ada1-1dea-413c-b7d3-39b8ae6715fc:0",
  "source": "test_data",
  "type": "Parking",
  "area_sqft": 15000
}
```

**Sample 2:**
```json
{
  "updated_at": "2025-08-30T15:07:36.220000000+00:00",
  "charging_ports": 20,
  "name": "EV Charging Station",
  "created_at": "2025-08-30T15:07:36.220000000+00:00",
  "floor_id": "4:d7c9ada1-1dea-413c-b7d3-39b8ae6715fc:0",
  "source": "test_data",
  "type": "Utility",
  "area_sqft": 2000
}
```

**Sample 3:**
```json
{
  "updated_at": "2025-08-30T15:02:56.988000000+00:00",
  "name": "Production Line A",
  "created_at": "2025-08-30T15:02:56.988000000+00:00",
  "floor_id": "4:d7c9ada1-1dea-413c-b7d3-39b8ae6715fc:93",
  "type": "Production"
}
```

#### AssessmentReport (Sample)

**Sample 1:**
```json
{
  "metadata": "{\"processing_time\": 15.5, \"methodology\": \"comprehensive\", \"generated_by\": \"Mock Risk Assessment Demo\", \"version\": \"1.0.0\"}",
  "executive_summary": "{\"overall_risk_level\": \"medium\", \"risk_score\": 35.7, \"total_risk_factors\": 3, \"total_recommendations\": 3}",
  "facility_id": "DEMO_FACILITY_001",
  "assessment_id": "mock_assessment_DEMO_FACILITY_001_20250829_175412",
  "id": "mock_assessment_DEMO_FACILITY_001_20250829_175412_report",
  "data_sources": "{\"environmental_records\": 1, \"health_records\": 0, \"safety_records\": 1, \"compliance_records\": 0}",
  "report_date": "2025-08-29T17:54:12.534702000+00:00"
}
```

#### Building (Sample)

**Sample 1:**
```json
{
  "code": "ALG001-MFG",
  "updated_at": "2025-08-30T15:02:56.847000000+00:00",
  "name": "Main Manufacturing",
  "created_at": "2025-08-30T15:02:56.847000000+00:00",
  "site_code": "ALG001",
  "type": "Manufacturing"
}
```

**Sample 2:**
```json
{
  "code": "ALG001-WH",
  "updated_at": "2025-08-30T15:07:35.532000000+00:00",
  "name": "Warehouse",
  "created_at": "2025-08-30T15:07:35.532000000+00:00",
  "source": "test_data",
  "site_code": "ALG001",
  "built_year": 2019,
  "type": "Warehouse",
  "area_sqft": 95000
}
```

**Sample 3:**
```json
{
  "code": "HOU001-CT",
  "updated_at": "2025-08-30T15:07:35.870000000+00:00",
  "name": "Corporate Tower",
  "created_at": "2025-08-30T15:07:35.870000000+00:00",
  "site_code": "HOU001",
  "source": "test_data",
  "built_year": 2020,
  "type": "Office",
  "area_sqft": 180000
}
```

#### ChangePoint (Sample)

No sample data available.

#### ComplianceRecord (Sample)

**Sample 1:**
```json
{
  "regulation_name": "Clean Air Act",
  "findings": "Routine inspection findings for Clean Air Act",
  "expiry_date": "2026-08-30",
  "inspection_date": "2025-08-15",
  "created_at": "2025-08-30T18:13:30.349000000+00:00",
  "compliance_date": "2025-07-31",
  "inspector_name": "Inspector A. Smith",
  "corrective_actions": "None required",
  "updated_at": "2025-08-30T18:13:30.349000000+00:00",
  "fine_amount_usd": 0,
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "comp_facility_apex_manufacturing___plant_a_0",
  "status": "Compliant",
  "regulation_code": "CLEAN-2024-001"
}
```

**Sample 2:**
```json
{
  "regulation_name": "Clean Water Act",
  "findings": "Violation found in Clean Water Act compliance",
  "expiry_date": "2026-07-31",
  "inspection_date": "2025-08-10",
  "created_at": "2025-08-30T18:13:30.373000000+00:00",
  "compliance_date": "2025-07-21",
  "inspector_name": "Inspector B. Smith",
  "corrective_actions": "Must address clean violations within 30 days",
  "updated_at": "2025-08-30T18:13:30.373000000+00:00",
  "fine_amount_usd": 5000,
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "comp_facility_apex_manufacturing___plant_a_1",
  "regulation_code": "CLEAN-2024-002",
  "status": "Non-Compliant"
}
```

**Sample 3:**
```json
{
  "regulation_name": "Resource Conservation and Recovery Act",
  "findings": "Violation found in Resource Conservation and Recovery Act compliance",
  "expiry_date": "2026-07-01",
  "inspection_date": "2025-08-05",
  "created_at": "2025-08-30T18:13:30.390000000+00:00",
  "compliance_date": "2025-07-11",
  "inspector_name": "Inspector C. Smith",
  "corrective_actions": "Must address resource violations within 30 days",
  "updated_at": "2025-08-30T18:13:30.390000000+00:00",
  "fine_amount_usd": 0,
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "comp_facility_apex_manufacturing___plant_a_2",
  "status": "Under Review",
  "regulation_code": "RESOURCE-2024-003"
}
```

#### Customer (Sample)

**Sample 1:**
```json
{
  "name": "Apex Manufacturing Inc.",
  "id": "customer_apex_manufacturing_inc"
}
```

**Sample 2:**
```json
{
  "name": "Apex Manufacturing Inc.",
  "id": "customer_apex_manufacturing_inc"
}
```

**Sample 3:**
```json
{
  "name": "Apex Manufacturing Inc.",
  "id": "customer_apex_manufacturing_inc"
}
```

#### DataPoint (Sample)

No sample data available.

#### DisposalFacility (Sample)

**Sample 1:**
```json
{
  "disposal_methods": "[]",
  "address": "100 Landfill Lane, Greenville, CA 91102",
  "epa_id": "CAL000654321",
  "permit_number": "",
  "name": "Green Valley Landfill",
  "id": "disposal_facility_green_valley_landfill"
}
```

**Sample 2:**
```json
{
  "disposal_methods": "[]",
  "address": "100 Landfill Lane, Greenville, CA 91102",
  "epa_id": "CAL000654321",
  "permit_number": "",
  "name": "Green Valley Landfill",
  "id": "disposal_facility_green_valley_landfill"
}
```

#### Document (Sample)

**Sample 1:**
```json
{
  "batch_processing": true,
  "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/document-3.pdf",
  "original_filename": "document-3.pdf",
  "file_hash": "503dbc341f96dcb27aa73c9390992d4095b322a3af85d458c145937a38504d69",
  "uploaded_at": "2025-09-03T20:22:38.854464",
  "recognition_confidence": 1.0,
  "output_dir": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/data/processed",
  "id": "document-3",
  "type": "water_bill",
  "document_type": "water_bill"
}
```

**Sample 2:**
```json
{
  "batch_processing": true,
  "statement_date": "2025-07-05",
  "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/document-1.pdf",
  "original_filename": "document-1.pdf",
  "file_hash": "fb388804f18eba89e7f11ab9062a50e7787be0fe87dab43e9780a6912e21850d",
  "uploaded_at": "2025-09-03T20:23:47.257184",
  "recognition_confidence": 1.0,
  "output_dir": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/data/processed",
  "id": "document-1",
  "type": "electricity_bill",
  "document_type": "electricity_bill"
}
```

**Sample 3:**
```json
{
  "batch_processing": true,
  "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/document-2.pdf",
  "original_filename": "document-2.pdf",
  "file_hash": "1d5e08e6486995c0f0accd57613c4af5b1941f269f2f3dc80f674456087e23ce",
  "uploaded_at": "2025-09-03T20:21:43.276265",
  "recognition_confidence": 1.0,
  "output_dir": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/data/processed",
  "id": "document-2",
  "type": "waste_manifest",
  "document_type": "waste_manifest"
}
```

#### DocumentChunk (Sample)

No sample data available.

#### EHSMetric (Sample)

**Sample 1:**
```json
{
  "area_name": "Water Treatment",
  "calculation": "total water consumed / units produced",
  "building_name": "Utility Building",
  "created_at": "2025-08-30T15:07:36.494000000+00:00",
  "site_code": "ALG001",
  "source": "test_data",
  "frequency": "monthly",
  "floor_name": "Utility Floor",
  "unit": "gallons per unit produced",
  "metric_name": "Water Usage",
  "target_value": 50.0,
  "metric_code": "WU",
  "recorded_date": "2025-08-30",
  "category": "Environmental",
  "value": 50.98
}
```

**Sample 2:**
```json
{
  "area_name": "Water Treatment",
  "calculation": "total energy used / units produced",
  "building_name": "Utility Building",
  "created_at": "2025-08-30T15:07:36.496000000+00:00",
  "source": "test_data",
  "site_code": "ALG001",
  "frequency": "monthly",
  "floor_name": "Utility Floor",
  "unit": "kWh per unit produced",
  "metric_name": "Energy Consumption",
  "target_value": 2.5,
  "metric_code": "EC",
  "recorded_date": "2025-06-01",
  "category": "Environmental",
  "value": 2.78
}
```

**Sample 3:**
```json
{
  "area_name": "Water Treatment",
  "calculation": "total energy used / units produced",
  "building_name": "Utility Building",
  "created_at": "2025-08-30T15:07:36.498000000+00:00",
  "source": "test_data",
  "site_code": "ALG001",
  "frequency": "monthly",
  "floor_name": "Utility Floor",
  "unit": "kWh per unit produced",
  "metric_name": "Energy Consumption",
  "target_value": 2.5,
  "metric_code": "EC",
  "recorded_date": "2025-07-01",
  "category": "Environmental",
  "value": 2.65
}
```

#### EffectivenessMeasurement (Sample)

No sample data available.

#### ElectricityConsumption (Sample)

**Sample 1:**
```json
{
  "date": "2025-01-01",
  "cost_usd": 174.7264,
  "updated_at": "2025-09-04T19:25:19.754000000+00:00",
  "created_at": "2025-09-04T19:25:19.754000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "elec_facility_apex_manufacturing___plant_a_20250101",
  "co2_emissions": 325.14304000000004,
  "consumption_kwh": 1519.3600000000001,
  "peak_demand_kw": 341,
  "meter_reading": 2374
}
```

**Sample 2:**
```json
{
  "date": "2025-01-02",
  "cost_usd": 208.50560000000002,
  "updated_at": "2025-09-04T19:25:19.770000000+00:00",
  "created_at": "2025-09-04T19:25:19.770000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "elec_facility_apex_manufacturing___plant_a_20250102",
  "co2_emissions": 388.00768,
  "consumption_kwh": 1813.1200000000001,
  "peak_demand_kw": 363,
  "meter_reading": 5666
}
```

**Sample 3:**
```json
{
  "date": "2025-07-31",
  "cost_usd": 329.28,
  "updated_at": "2025-08-30T18:13:29.819000000+00:00",
  "created_at": "2025-08-30T18:13:29.819000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "elec_facility_apex_manufacturing___plant_a_20250731",
  "co2_emissions": 587.2159999999999,
  "consumption_kwh": 2743.9999999999995,
  "peak_demand_kw": 800.0,
  "meter_reading": 1000
}
```

#### Electricitybill (Sample)

**Sample 1:**
```json
{
  "batch_processing": true,
  "statement_date": "2025-07-05",
  "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/document-1.pdf",
  "original_filename": "document-1.pdf",
  "file_hash": "fb388804f18eba89e7f11ab9062a50e7787be0fe87dab43e9780a6912e21850d",
  "uploaded_at": "2025-09-03T20:23:47.257184",
  "recognition_confidence": 1.0,
  "output_dir": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/data/processed",
  "id": "document-1",
  "type": "electricity_bill",
  "document_type": "electricity_bill"
}
```

#### Emission (Sample)

**Sample 1:**
```json
{
  "emission_factor": 0.5,
  "unit": "metric_tons_CO2e",
  "amount": 7.5,
  "disposal_method": "landfill",
  "calculation_method": "waste_disposal_landfill",
  "source_type": "waste_disposal",
  "id": "emission_unknown_document_20250828_173206_629"
}
```

**Sample 2:**
```json
{
  "emission_factor": 0.0002,
  "amount": 50.0,
  "unit": "kg_CO2",
  "calculation_method": "water_treatment_distribution_factor",
  "source_type": "water",
  "id": "emission_unknown_document_20250828_173247_463"
}
```

**Sample 3:**
```json
{
  "emission_factor": 0.4,
  "amount": 52000.0,
  "unit": "kg_CO2",
  "calculation_method": "grid_average_factor",
  "id": "emission_unknown_document_20250828_173333_450"
}
```

#### EmissionFactor (Sample)

**Sample 1:**
```json
{
  "grid_subregion": "SRCE (SERC Central)",
  "unit": "kg_CO2_per_kWh",
  "updated_at": "2025-09-04T19:25:19.613000000+00:00",
  "year": 2023,
  "created_at": "2025-09-04T19:25:19.613000000+00:00",
  "source": "EPA eGRID 2023",
  "id": "emission_factor_None_electricity_2023",
  "region": "Illinois",
  "value": 0.214,
  "methodology": "Grid average emission factor"
}
```

#### EnvironmentalKPI (Sample)

**Sample 1:**
```json
{
  "date": "2025-06-01",
  "performance_status": "Below Target",
  "unit": "percentage",
  "updated_at": "2025-08-30T18:13:30.264000000+00:00",
  "target_value": 90,
  "created_at": "2025-08-30T18:13:30.264000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "kpi_facility_apex_manufacturing___plant_a_Energy_Efficiency_202506_1756577610",
  "type": "Energy Efficiency",
  "value": 85,
  "variance_percentage": -5.56
}
```

**Sample 2:**
```json
{
  "performance_status": "Below Target",
  "date": "2025-07-31",
  "unit": "percentage",
  "updated_at": "2025-08-30T18:13:30.289000000+00:00",
  "target_value": 90,
  "created_at": "2025-08-30T18:13:30.289000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "kpi_facility_apex_manufacturing___plant_a_Energy_Efficiency_202507_1756577610",
  "type": "Energy Efficiency",
  "value": 87,
  "variance_percentage": -3.33
}
```

**Sample 3:**
```json
{
  "performance_status": "Above Target",
  "date": "2025-06-01",
  "unit": "gallons_per_day",
  "updated_at": "2025-08-30T18:13:30.291000000+00:00",
  "target_value": 800,
  "created_at": "2025-08-30T18:13:30.291000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "id": "kpi_facility_apex_manufacturing___plant_a_Water_Usage_202506_1756577610",
  "type": "Water Usage",
  "value": 1000,
  "variance_percentage": 25.0
}
```

#### EnvironmentalRecommendation (Sample)

**Sample 1:**
```json
{
  "estimated_savings_usd": 1500.0,
  "target_completion_date": "2025-10-29",
  "implementation_timeline_days": 30,
  "created_at": "2025-08-30T18:13:30.130000000+00:00",
  "description": "Replace fluorescent lights with LED fixtures to reduce energy consumption by 30%",
  "type": "Energy Efficiency",
  "title": "Install LED Lighting",
  "priority": "Low",
  "updated_at": "2025-08-30T18:13:30.130000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "estimated_cost_usd": 5000,
  "created_date": "2025-08-15",
  "id": "rec_facility_apex_manufacturing___plant_a_0",
  "status": "Proposed"
}
```

**Sample 2:**
```json
{
  "estimated_savings_usd": 4200.0,
  "target_completion_date": "2025-11-13",
  "implementation_timeline_days": 45,
  "created_at": "2025-08-30T18:13:30.150000000+00:00",
  "description": "Upgrade to low-flow faucets and toilets to reduce water consumption",
  "type": "Water Conservation",
  "title": "Install Low-Flow Fixtures",
  "priority": "Medium",
  "updated_at": "2025-08-30T18:13:30.150000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "created_date": "2025-08-14",
  "estimated_cost_usd": 7000,
  "id": "rec_facility_apex_manufacturing___plant_a_1",
  "status": "Proposed"
}
```

**Sample 3:**
```json
{
  "estimated_savings_usd": 8100.0,
  "target_completion_date": "2025-11-28",
  "implementation_timeline_days": 60,
  "created_at": "2025-08-30T18:13:30.152000000+00:00",
  "description": "Establish comprehensive recycling program to divert 50% of waste from landfills",
  "type": "Waste Reduction",
  "title": "Implement Recycling Program",
  "priority": "High",
  "updated_at": "2025-08-30T18:13:30.152000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "created_date": "2025-08-13",
  "estimated_cost_usd": 9000,
  "id": "rec_facility_apex_manufacturing___plant_a_2",
  "status": "Proposed"
}
```

#### EnvironmentalRisk (Sample)

**Sample 1:**
```json
{
  "severity": "Low",
  "risk_type": "Air Quality Violation",
  "updated_at": "2025-08-30T18:13:30.076000000+00:00",
  "probability": "Low",
  "created_at": "2025-08-30T18:13:30.076000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "description": "Potential air quality violation at facility facility_apex_manufacturing___plant_a",
  "identified_date": "2025-07-31",
  "impact_score": 2,
  "id": "risk_facility_apex_manufacturing___plant_a_0",
  "mitigation_status": "Planned",
  "last_assessment_date": "2025-08-30"
}
```

**Sample 2:**
```json
{
  "severity": "Medium",
  "risk_type": "Water Contamination",
  "updated_at": "2025-08-30T18:13:30.100000000+00:00",
  "probability": "Low",
  "created_at": "2025-08-30T18:13:30.100000000+00:00",
  "identified_date": "2025-07-30",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "description": "Potential water contamination at facility facility_apex_manufacturing___plant_a",
  "impact_score": 5,
  "id": "risk_facility_apex_manufacturing___plant_a_1",
  "mitigation_status": "Planned",
  "last_assessment_date": "2025-08-29"
}
```

**Sample 3:**
```json
{
  "severity": "High",
  "risk_type": "Soil Contamination",
  "updated_at": "2025-08-30T18:13:30.101000000+00:00",
  "probability": "Low",
  "created_at": "2025-08-30T18:13:30.101000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "description": "Potential soil contamination at facility facility_apex_manufacturing___plant_a",
  "identified_date": "2025-07-29",
  "impact_score": 8,
  "id": "risk_facility_apex_manufacturing___plant_a_2",
  "mitigation_status": "Planned",
  "last_assessment_date": "2025-08-28"
}
```

#### EnvironmentalTarget (Sample)

**Sample 1:**
```json
{
  "target_unit": "percent_reduction",
  "updated_at": "2025-08-31T23:43:54.691000000+00:00",
  "target_value": 0.15,
  "target_type": "electricity_reduction",
  "site_id": "algonquin_il",
  "created_at": "2025-08-31T23:43:54.691000000+00:00",
  "description": "Reduce electricity consumption by 15% through efficiency improvements",
  "id": "target_algonquin_il_0",
  "deadline": "2025-12-31",
  "status": "in_progress"
}
```

**Sample 2:**
```json
{
  "target_unit": "percentage",
  "updated_at": "2025-08-31T23:43:54.751000000+00:00",
  "target_value": 0.5,
  "target_type": "recycling_rate",
  "site_id": "algonquin_il",
  "created_at": "2025-08-31T23:43:54.751000000+00:00",
  "description": "Achieve 50% recycling rate for all waste streams",
  "id": "target_algonquin_il_1",
  "deadline": "2025-12-31",
  "status": "in_progress"
}
```

**Sample 3:**
```json
{
  "target_unit": "percent_reduction",
  "updated_at": "2025-08-31T23:43:54.800000000+00:00",
  "target_value": 0.1,
  "target_type": "water_efficiency",
  "site_id": "algonquin_il",
  "created_at": "2025-08-31T23:43:54.800000000+00:00",
  "description": "Reduce water consumption per unit of production by 10%",
  "id": "target_algonquin_il_2",
  "deadline": "2025-12-31",
  "status": "planning"
}
```

#### Equipment (Sample)

No sample data available.

#### Facility (Sample)

**Sample 1:**
```json
{
  "address": "789 Production Way, Mechanicsburg, CA 93011",
  "name": "Apex Manufacturing - Plant A",
  "id": "facility_apex_manufacturing___plant_a"
}
```

**Sample 2:**
```json
{
  "employee_count": 250,
  "address": "Demo Location",
  "operational_status": "Active",
  "industry_sector": "Chemical Manufacturing",
  "name": "DEMO_FACILITY_001",
  "id": "DEMO_FACILITY_001",
  "established_date": "2020-01-01",
  "type": "Manufacturing"
}
```

**Sample 3:**
```json
{
  "address": "789 Production Way, Mechanicsburg, CA 93011",
  "name": "Apex Manufacturing - Plant A",
  "id": "facility_apex_manufacturing___plant_a"
}
```

#### Floor (Sample)

**Sample 1:**
```json
{
  "building_id": "4:d7c9ada1-1dea-413c-b7d3-39b8ae6715fc:178",
  "updated_at": "2025-08-30T15:07:36.170000000+00:00",
  "level": 0,
  "name": "Ground Level",
  "created_at": "2025-08-30T15:07:36.170000000+00:00",
  "source": "test_data",
  "ceiling_height": 10,
  "area_sqft": 50000
}
```

**Sample 2:**
```json
{
  "building_id": "4:d7c9ada1-1dea-413c-b7d3-39b8ae6715fc:92",
  "updated_at": "2025-08-30T15:02:56.906000000+00:00",
  "level": 0,
  "name": "Ground Floor",
  "created_at": "2025-08-30T15:02:56.906000000+00:00"
}
```

**Sample 3:**
```json
{
  "building_id": "4:d7c9ada1-1dea-413c-b7d3-39b8ae6715fc:92",
  "updated_at": "2025-08-30T15:02:57.104000000+00:00",
  "level": 2,
  "name": "Second Floor",
  "created_at": "2025-08-30T15:02:57.104000000+00:00"
}
```

#### Goal (Sample)

**Sample 1:**
```json
{
  "unit": "tonnes CO2e",
  "period": "yearly",
  "updated_at": "2025-09-01T01:15:01.699000000+00:00",
  "target_date": "2025-12-31",
  "target_value": 15.0,
  "baseline_year": 2024,
  "site_id": "algonquin_il",
  "target_year": 2025,
  "created_at": "2025-09-01T01:15:01.699000000+00:00",
  "description": "CO2 emissions reduction from electricity consumption",
  "id": "goal_algonquin_il_electricity_2025",
  "category": "electricity"
}
```

**Sample 2:**
```json
{
  "unit": "gallons",
  "period": "yearly",
  "updated_at": "2025-09-01T01:15:01.756000000+00:00",
  "target_date": "2025-12-31",
  "target_value": 12.0,
  "baseline_year": 2024,
  "site_id": "algonquin_il",
  "target_year": 2025,
  "created_at": "2025-09-01T01:15:01.756000000+00:00",
  "description": "Water consumption reduction",
  "id": "goal_algonquin_il_water_2025",
  "category": "water"
}
```

**Sample 3:**
```json
{
  "unit": "pounds",
  "period": "yearly",
  "updated_at": "2025-09-01T01:15:01.771000000+00:00",
  "target_date": "2025-12-31",
  "target_value": 10.0,
  "baseline_year": 2024,
  "site_id": "algonquin_il",
  "target_year": 2025,
  "created_at": "2025-09-01T01:15:01.771000000+00:00",
  "description": "Waste generation reduction",
  "id": "goal_algonquin_il_waste_2025",
  "category": "waste"
}
```

#### Incident (Sample)

**Sample 1:**
```json
{
  "reported_date": "2025-06-30T14:28:41.540000000+00:00",
  "severity": "low",
  "incident_id": "DEMO_FACILITY_001_incident_002",
  "facility_id": "DEMO_FACILITY_001",
  "description": "Temporary air quality reading spike",
  "type": "environmental",
  "status": "resolved"
}
```

**Sample 2:**
```json
{
  "date": "2025-07-30",
  "severity": "medium",
  "description": "Minor equipment malfunction in production line 2",
  "id": "DEMO_FACILITY_001_incident_001",
  "type": "safety",
  "status": "resolved"
}
```

**Sample 3:**
```json
{
  "date": "2025-06-30",
  "severity": "low",
  "description": "Temporary air quality reading spike due to equipment calibration",
  "id": "DEMO_FACILITY_001_incident_002",
  "type": "environmental",
  "status": "resolved"
}
```

#### Meter (Sample)

**Sample 1:**
```json
{
  "unit": "gallons",
  "service_type": "Commercial",
  "previous_reading": 8765400,
  "usage": 250000,
  "current_reading": 9015400,
  "id": "WTR-MAIN-01",
  "type": "water"
}
```

**Sample 2:**
```json
{
  "previous_reading": 543210,
  "service_type": "Commercial - Peak",
  "unit": "kWh",
  "usage": 70000,
  "current_reading": 613210,
  "id": "MTR-7743-A",
  "type": "electricity"
}
```

**Sample 3:**
```json
{
  "previous_reading": 891234,
  "service_type": "Commercial - Off-Peak",
  "unit": "kWh",
  "usage": 60000,
  "current_reading": 951234,
  "id": "MTR-7743-B",
  "type": "electricity"
}
```

#### Migration (Sample)

No sample data available.

#### MonthlyUsageAllocation (Sample)

No sample data available.

#### Permit (Sample)

No sample data available.

#### Recommendation (Sample)

**Sample 1:**
```json
{
  "site_id": "algonquin_il",
  "description": "{'priorityLevel': 'High', 'actionDescription': 'Implement advanced lighting systems such as LED upgrades and smart controls', 'bestPracticeCategory': 'Energy Efficiency Measures', 'estimatedMonthlyImpact': '5% units reduction', 'implementationEffort': 'Medium', 'timeline': 'Short-term', 'resourceRequirements': 'Investment in LED lighting and smart control systems, technical expertise for installation and maintenance', 'supportingEvidence': 'General Motors achieved 35% energy reduction across facilities through comprehensive efficiency programs'}",
  "estimated_impact": "TBD",
  "created_date": "2025-09-05T22:35:46.254530000+00:00",
  "id": "algonquin_il_electricity_rec_1_20250905_223546",
  "category": "electricity",
  "priority": "medium",
  "title": "Recommendation 1"
}
```

**Sample 2:**
```json
{
  "site_id": "algonquin_il",
  "description": "{'priorityLevel': 'High', 'actionDescription': 'Convert combustion-based processes to electric alternatives', 'bestPracticeCategory': 'Process Electrification', 'estimatedMonthlyImpact': '10% units reduction', 'implementationEffort': 'High', 'timeline': 'Long-term', 'resourceRequirements': 'Investment in electric equipment, technical expertise for installation and maintenance, potential downtime during conversion', 'supportingEvidence': 'Cleveland-Cliffs implemented electric arc furnace technology, reducing CO2 emissions by 75% compared to traditional blast furnaces'}",
  "estimated_impact": "TBD",
  "created_date": "2025-09-05T22:35:46.262837000+00:00",
  "id": "algonquin_il_electricity_rec_2_20250905_223546",
  "category": "electricity",
  "priority": "medium",
  "title": "Recommendation 2"
}
```

**Sample 3:**
```json
{
  "site_id": "algonquin_il",
  "description": "{'priorityLevel': 'High', 'actionDescription': 'Procure renewable energy through on-site solar installations or Power Purchase Agreements (PPAs) with renewable energy providers', 'bestPracticeCategory': 'Renewable Energy Procurement', 'estimatedMonthlyImpact': '8% units reduction', 'implementationEffort': 'High', 'timeline': 'Long-term', 'resourceRequirements': 'Investment in solar installations or contracts with renewable energy providers, technical expertise for installation and maintenance', 'supportingEvidence': 'RMS Company achieved 100% renewable electricity through a combination of on-site solar and renewable energy contracts'}",
  "estimated_impact": "TBD",
  "created_date": "2025-09-05T22:35:46.265727000+00:00",
  "id": "algonquin_il_electricity_rec_3_20250905_223546",
  "category": "electricity",
  "priority": "medium",
  "title": "Recommendation 3"
}
```

#### RejectedDocument (Sample)

**Sample 1:**
```json
{
  "file_path": "/tmp/rejected_documents/20250905_233349_document-4_document-4.pdf",
  "upload_timestamp": "",
  "rejected_at": "2025-09-05T23:33:49.982092",
  "attempted_type": "unknown",
  "confidence": 0.8625,
  "upload_source": "",
  "file_size": 381709,
  "original_filename": "document-4.pdf",
  "key_terms": "[\"account\"]",
  "rejection_reason": "Document type could not be determined (confidence: 0.863)",
  "id": "document-4",
  "tables_found": true,
  "page_count": 1,
  "content_length": 7523
}
```

#### RejectionRecord (Sample)

No sample data available.

#### RiskAssessment (Sample)

**Sample 1:**
```json
{
  "risk_level": "HIGH",
  "confidence_score": 0.5,
  "site_id": "algonquin_il",
  "description": "{\n  \"1. Projected annual consumption based on current trend\": {\n    \"projected_annual_consumption\": \"10.7461 tonnes CO2e\"\n  },\n  \"2. Gap analysis\": {\n    \"absolute_difference\": \"-4.2539 tonnes CO2e\",\n    \"percentage_difference\": \"-28.36%\"\n  },\n  \"3. Risk level assignment with justification\": {\n    \"risk_level\": \"HIGH\",\n    \"justification\": \"Based on the current trend, there is a 28.36% deviation from the goal, which falls into the 'HIGH' risk category (25-50% deviation from goal).\"\n  },\n  \"4. Key risk factors considering baseline vs current performance\": {\n    \"key_risk_factors\": [\n      \"Increasing trend in electricity consumption\",\n      \"Seasonal patterns indicating higher consumption during summer months\",\n      \"Current performance already exceeding baseline consumption\"\n    ]\n  },\n  \"5. Assessment of whether goal is achievable given current trend\": {\n    \"goal_achievable\": \"No\",\n    \"justification\": \"Given the current increasing trend in electricity consumption and the fact that the current performance is already exceeding the baseline, it is unlikely that the annual goal of 15.0 tonnes CO2e will be met.\"\n  }\n}",
  "id": "algonquin_il_electricity_20250905_223546",
  "category": "electricity",
  "factors": [
    "Gap: 28.36%"
  ],
  "assessment_date": "2025-09-05T22:35:34.755670000+00:00"
}
```

**Sample 2:**
```json
{
  "risk_level": "CRITICAL",
  "confidence_score": 0.5,
  "site_id": "houston_tx",
  "description": "{\n  \"projected_annual_consumption\": {\n    \"amount\": 208.99,\n    \"unit\": \"tonnes CO2e\"\n  },\n  \"gap_analysis\": {\n    \"absolute_difference\": 190.99,\n    \"percentage_difference\": 1061.06\n  },\n  \"risk_level_assignment\": {\n    \"risk_level\": \"CRITICAL\",\n    \"justification\": \"The projected annual consumption is more than 50% above the goal, indicating a less than 10% chance of meeting the goal.\"\n  },\n  \"key_risk_factors\": {\n    \"increasing_trend\": \"The overall trend of electricity consumption, cost, and CO2e emissions is increasing, which could potentially impact the annual goal achievement if the trend continues.\",\n    \"baseline_vs_current_performance\": \"The current performance is significantly higher than the baseline consumption, indicating a high risk of not meeting the reduction goal.\"\n  },\n  \"goal_achievability_assessment\": {\n    \"is_goal_achievable\": false,\n    \"justification\": \"Given the current trend of increasing consumption and emissions, and the significant gap between the current performance and the goal, it is highly unlikely that the goal will be achieved.\"\n  }\n}",
  "id": "houston_tx_electricity_20250905_224501",
  "category": "electricity",
  "factors": [
    "Gap: 1061.06%"
  ],
  "assessment_date": "2025-09-05T22:44:45.615343000+00:00"
}
```

#### SeasonalDecomposition (Sample)

No sample data available.

#### Site (Sample)

**Sample 1:**
```json
{
  "established": "2018-03-15",
  "code": "ALG001",
  "address": "2350 Millennium Drive, Algonquin, IL 60102",
  "updated_at": "2025-08-30T15:07:34.832000000+00:00",
  "operating_hours": "24/7",
  "name": "Algonquin Manufacturing Site",
  "created_at": "2025-08-30T15:07:34.832000000+00:00",
  "source": "test_data",
  "certifications": [
    "ISO 14001",
    "OSHA VPP",
    "ISO 45001"
  ],
  "employees": 485,
  "type": "Manufacturing"
}
```

**Sample 2:**
```json
{
  "established": "2020-01-10",
  "code": "HOU001",
  "address": "1200 Smith Street, Houston, TX 77002",
  "updated_at": "2025-08-30T15:07:35.850000000+00:00",
  "operating_hours": "8AM-6PM",
  "name": "Houston Corporate Campus",
  "created_at": "2025-08-30T15:07:35.850000000+00:00",
  "source": "test_data",
  "certifications": [
    "LEED Gold",
    "Energy Star"
  ],
  "type": "Corporate",
  "employees": 280
}
```

**Sample 3:**
```json
{
  "employee_count": 245,
  "country": "USA",
  "electricity_target_change": -0.15,
  "risk_profile": "HIGH",
  "recycling_target_rate": 0.5,
  "created_at": "2025-08-31T23:47:53.676000000+00:00",
  "water_baseline_gallons": 150000,
  "waste_baseline_lbs": 8000,
  "zip_code": "60102",
  "updated_at": "2025-09-05T21:08:15.729000000+00:00",
  "electricity_baseline_kwh": 45000,
  "name": "Algonquin Manufacturing",
  "operating_hours_per_day": 16,
  "location": "Algonquin, IL",
  "state": "Illinois",
  "facility_type": "Manufacturing",
  "id": "algonquin_il"
}
```

#### Transporter (Sample)

**Sample 1:**
```json
{
  "address": "",
  "phone": "",
  "name": "Evergreen Environmental",
  "id": "transporter_evergreen_environmental",
  "license_number": ""
}
```

**Sample 2:**
```json
{
  "address": "",
  "phone": "",
  "name": "Evergreen Environmental",
  "id": "transporter_evergreen_environmental",
  "license_number": ""
}
```

#### TrendAnalysis (Sample)

No sample data available.

#### User (Sample)

No sample data available.

#### UtilityBill (Sample)

**Sample 1:**
```json
{
  "off_peak_kwh": 60000,
  "peak_kwh": 70000,
  "grid_infrastructure_fee": 182.39,
  "total_cost": 15432.89,
  "due_date": "2025-08-01",
  "billing_period_end": "2025-06-30",
  "base_service_charge": 250.0,
  "state_environmental_surcharge": 875.5,
  "billing_period_start": "2025-06-01",
  "id": "bill_unknown_document_20250828_173333_450",
  "total_kwh": 130000
}
```

**Sample 2:**
```json
{
  "off_peak_kwh": 60000,
  "peak_kwh": 70000,
  "grid_infrastructure_fee": 182.39,
  "total_cost": 15432.89,
  "due_date": "2025-08-01",
  "billing_period_end": "2025-06-30",
  "base_service_charge": 250.0,
  "state_environmental_surcharge": 875.5,
  "billing_period_start": "2025-06-01",
  "id": "bill_document-1",
  "total_kwh": 130000
}
```

#### UtilityProvider (Sample)

**Sample 1:**
```json
{
  "website": "www.aquaflow.gov",
  "address": "789 Reservoir Road, Clearwater, CA 90330",
  "phone": "(800) 555-0211",
  "name": "Aquaflow Municipal Water",
  "id": "provider_aquaflow_municipal_water"
}
```

**Sample 2:**
```json
{
  "website": "www.aquaflow.gov",
  "address": "789 Reservoir Road, Clearwater, CA 90330",
  "phone": "(800) 555-0211",
  "name": "Aquaflow Municipal Water",
  "id": "provider_aquaflow_municipal_water"
}
```

#### WasteGeneration (Sample)

**Sample 1:**
```json
{
  "date": "2025-07-31",
  "contractor": "Waste Management Co. 1",
  "disposal_cost_usd": 272.5,
  "updated_at": "2025-08-30T18:13:29.973000000+00:00",
  "disposal_method": "Incineration",
  "created_at": "2025-08-30T18:13:29.973000000+00:00",
  "waste_type": "Hazardous",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "amount_pounds": 109,
  "id": "waste_facility_apex_manufacturing___plant_a_Hazardous_20250731"
}
```

**Sample 2:**
```json
{
  "date": "2025-07-31",
  "contractor": "Waste Management Co. 1",
  "disposal_cost_usd": 74.0,
  "updated_at": "2025-08-30T18:13:29.998000000+00:00",
  "disposal_method": "Landfill",
  "created_at": "2025-08-30T18:13:29.998000000+00:00",
  "waste_type": "Non-Hazardous",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "amount_pounds": 148,
  "id": "waste_facility_apex_manufacturing___plant_a_Non-Hazardous_20250731"
}
```

**Sample 3:**
```json
{
  "date": "2025-07-31",
  "contractor": "Waste Management Co. 1",
  "disposal_cost_usd": 72.0,
  "updated_at": "2025-08-30T18:13:30.018000000+00:00",
  "disposal_method": "Landfill",
  "created_at": "2025-08-30T18:13:30.018000000+00:00",
  "waste_type": "Recyclable",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "amount_pounds": 144,
  "id": "waste_facility_apex_manufacturing___plant_a_Recyclable_20250731"
}
```

#### WasteGenerator (Sample)

**Sample 1:**
```json
{
  "address": "789 Production Way, Mechanicsburg, CA 93011",
  "epa_id": "CAL000123456",
  "phone": "(555) 123-7890",
  "contact": "John Doe, EHS Manager",
  "name": "Apex Manufacturing Inc.",
  "id": "generator_apex_manufacturing_inc."
}
```

**Sample 2:**
```json
{
  "address": "789 Production Way, Mechanicsburg, CA 93011",
  "epa_id": "CAL000123456",
  "phone": "(555) 123-7890",
  "contact": "John Doe, EHS Manager",
  "name": "Apex Manufacturing Inc.",
  "id": "generator_apex_manufacturing_inc."
}
```

#### WasteItem (Sample)

**Sample 1:**
```json
{
  "proper_shipping_name": "",
  "quantity": 15,
  "hazard_class": "",
  "weight_unit": "tons",
  "container_type": "Open Top Roll-off",
  "waste_type": "Industrial Solid Waste - Mixed non-recyclable production scrap (plastic, wood, fabric)",
  "description": "Industrial Solid Waste - Mixed non-recyclable production scrap (plastic, wood, fabric)",
  "container_quantity": 1,
  "id": "waste_item_unknown_document_20250828_173206_629_6",
  "container_count": 0
}
```

**Sample 2:**
```json
{
  "proper_shipping_name": "",
  "quantity": 15,
  "hazard_class": "",
  "weight_unit": "tons",
  "container_type": "Open Top Roll-off",
  "waste_type": "Industrial Solid Waste - Mixed non-recyclable production scrap (plastic, wood, fabric)",
  "description": "Industrial Solid Waste - Mixed non-recyclable production scrap (plastic, wood, fabric)",
  "container_quantity": 1,
  "id": "waste_item_document-2_6",
  "container_count": 0
}
```

#### WasteManifest (Sample)

**Sample 1:**
```json
{
  "weight_unit": "tons",
  "issue_date": "2025-07-15",
  "manifest_tracking_number": "EES-2025-0715-A45",
  "total_quantity": 15,
  "id": "manifest_document-2",
  "total_weight": 15,
  "status": "Not specified"
}
```

**Sample 2:**
```json
{
  "weight_unit": "tons",
  "issue_date": "2025-07-15",
  "manifest_tracking_number": "EES-2025-0715-A45",
  "total_quantity": 15,
  "id": "manifest_unknown_document_20250828_173206_629",
  "total_weight": 15
}
```

#### WasteShipment (Sample)

**Sample 1:**
```json
{
  "weight_unit": "tons",
  "transport_method": "truck",
  "shipment_date": "2025-07-15",
  "id": "shipment_document-2",
  "total_weight": 15,
  "status": "Not specified"
}
```

**Sample 2:**
```json
{
  "weight_unit": "tons",
  "transport_method": "truck",
  "shipment_date": "2025-07-15",
  "id": "shipment_unknown_document_20250828_173206_629",
  "total_weight": 15
}
```

#### Wastemanifest (Sample)

**Sample 1:**
```json
{
  "batch_processing": true,
  "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/document-2.pdf",
  "original_filename": "document-2.pdf",
  "file_hash": "1d5e08e6486995c0f0accd57613c4af5b1941f269f2f3dc80f674456087e23ce",
  "uploaded_at": "2025-09-03T20:21:43.276265",
  "recognition_confidence": 1.0,
  "output_dir": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/data/processed",
  "id": "document-2",
  "type": "waste_manifest",
  "document_type": "waste_manifest"
}
```

#### WaterBill (Sample)

**Sample 1:**
```json
{
  "stormwater_fee": 125.0,
  "total_cost": 4891.5,
  "sewer_service_charge": 950.0,
  "due_date": "2025-08-05",
  "billing_period_end": "2025-06-30",
  "water_consumption_cost": 3750.0,
  "total_gallons": 250000,
  "billing_period_start": "2025-06-01",
  "id": "water_bill_unknown_document_20250828_173247_463",
  "infrastructure_surcharge": 5.0,
  "conservation_tax": 61.5
}
```

**Sample 2:**
```json
{
  "stormwater_fee": 125.0,
  "total_cost": 4891.5,
  "sewer_service_charge": 950.0,
  "due_date": "2025-08-05",
  "billing_period_end": "2025-06-30",
  "water_consumption_cost": 3750.0,
  "total_gallons": 250000,
  "billing_period_start": "2025-06-01",
  "id": "water_bill_document-3",
  "infrastructure_surcharge": 5.0,
  "conservation_tax": 61.5
}
```

#### WaterConsumption (Sample)

**Sample 1:**
```json
{
  "date": "2025-07-31",
  "cost_usd": 1.5,
  "updated_at": "2025-08-30T18:13:29.907000000+00:00",
  "quality_rating": "A",
  "created_at": "2025-08-30T18:13:29.907000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "source_type": "Municipal",
  "id": "water_facility_apex_manufacturing___plant_a_20250731",
  "consumption_gallons": 500
}
```

**Sample 2:**
```json
{
  "date": "2025-08-01",
  "cost_usd": 1.59,
  "updated_at": "2025-08-30T18:13:29.925000000+00:00",
  "quality_rating": "B",
  "created_at": "2025-08-30T18:13:29.925000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "source_type": "Municipal",
  "id": "water_facility_apex_manufacturing___plant_a_20250801",
  "consumption_gallons": 530
}
```

**Sample 3:**
```json
{
  "date": "2025-08-02",
  "cost_usd": 1.68,
  "updated_at": "2025-08-30T18:13:29.926000000+00:00",
  "quality_rating": "B",
  "created_at": "2025-08-30T18:13:29.926000000+00:00",
  "facility_id": "facility_apex_manufacturing___plant_a",
  "source_type": "Municipal",
  "id": "water_facility_apex_manufacturing___plant_a_20250802",
  "consumption_gallons": 560
}
```

#### Waterbill (Sample)

**Sample 1:**
```json
{
  "batch_processing": true,
  "file_path": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/document-3.pdf",
  "original_filename": "document-3.pdf",
  "file_hash": "503dbc341f96dcb27aa73c9390992d4095b322a3af85d458c145937a38504d69",
  "uploaded_at": "2025-09-03T20:22:38.854464",
  "recognition_confidence": 1.0,
  "output_dir": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/data/processed",
  "id": "document-3",
  "type": "water_bill",
  "document_type": "water_bill"
}
```

#### __Entity__ (Sample)

No sample data available.

#### __Node__ (Sample)

No sample data available.

### Sample Relationships

#### AFFECTS_CONSUMPTION (Sample)

No sample data available.

#### APPLIES_TO (Sample)

**Sample 1:**
- **Start Node:** ['Goal'] (ID: 6413)
- **End Node:** ['Site'] (ID: 5461)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Goal'] (ID: 6414)
- **End Node:** ['Site'] (ID: 5461)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Goal'] (ID: 6415)
- **End Node:** ['Site'] (ID: 5461)
- **Properties:**
```json
{}
```

#### BILLED_FOR (Sample)

**Sample 1:**
- **Start Node:** ['WaterBill'] (ID: 65)
- **End Node:** ['Customer'] (ID: 67)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['WaterBill'] (ID: 65)
- **End Node:** ['Customer'] (ID: 76)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['WaterBill'] (ID: 59)
- **End Node:** ['Customer'] (ID: 61)
- **Properties:**
```json
{}
```

#### BILLED_TO (Sample)

**Sample 1:**
- **Start Node:** ['WaterBill'] (ID: 59)
- **End Node:** ['Facility'] (ID: 60)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['WaterBill'] (ID: 65)
- **End Node:** ['Facility'] (ID: 60)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['WaterBill'] (ID: 65)
- **End Node:** ['Facility'] (ID: 75)
- **Properties:**
```json
{}
```

#### CONTAINS (Sample)

**Sample 1:**
- **Start Node:** ['Site'] (ID: 91)
- **End Node:** ['Building'] (ID: 92)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Building'] (ID: 92)
- **End Node:** ['Floor'] (ID: 93)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Floor'] (ID: 93)
- **End Node:** ['Area'] (ID: 94)
- **Properties:**
```json
{}
```

#### CONTAINS_WASTE (Sample)

**Sample 1:**
- **Start Node:** ['WasteShipment'] (ID: 140)
- **End Node:** ['WasteItem'] (ID: 56)
- **Properties:**
```json
{
  "unit": "tons",
  "quantity": 15
}
```

**Sample 2:**
- **Start Node:** ['WasteShipment'] (ID: 80)
- **End Node:** ['WasteItem'] (ID: 6437)
- **Properties:**
```json
{
  "unit": "tons",
  "quantity": 15
}
```

#### DISPOSED_AT (Sample)

**Sample 1:**
- **Start Node:** ['WasteShipment'] (ID: 140)
- **End Node:** ['DisposalFacility'] (ID: 55)
- **Properties:**
```json
{
  "disposal_method": ""
}
```

**Sample 2:**
- **Start Node:** ['WasteShipment'] (ID: 80)
- **End Node:** ['DisposalFacility'] (ID: 55)
- **Properties:**
```json
{
  "disposal_method": ""
}
```

**Sample 3:**
- **Start Node:** ['WasteShipment'] (ID: 80)
- **End Node:** ['DisposalFacility'] (ID: 6431)
- **Properties:**
```json
{
  "disposal_method": ""
}
```

#### DOCUMENTS (Sample)

**Sample 1:**
- **Start Node:** ['WasteManifest'] (ID: 139)
- **End Node:** ['WasteShipment'] (ID: 140)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['WasteManifest'] (ID: 79)
- **End Node:** ['WasteShipment'] (ID: 80)
- **Properties:**
```json
{}
```

#### EXTRACTED_TO (Sample)

**Sample 1:**
- **Start Node:** ['Document', 'Waterbill'] (ID: 58)
- **End Node:** ['WaterBill'] (ID: 65)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Document', 'Electricitybill'] (ID: 6433)
- **End Node:** ['UtilityBill'] (ID: 6440)
- **Properties:**
```json
{
  "extraction_date": "2025-09-03T20:23:47.257201"
}
```

#### GENERATED_BY (Sample)

**Sample 1:**
- **Start Node:** ['WasteShipment'] (ID: 140)
- **End Node:** ['WasteGenerator'] (ID: 179)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['WasteShipment'] (ID: 80)
- **End Node:** ['WasteGenerator'] (ID: 179)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['WasteShipment'] (ID: 80)
- **End Node:** ['WasteGenerator'] (ID: 6419)
- **Properties:**
```json
{}
```

#### GENERATES_EMISSION (Sample)

**Sample 1:**
- **Start Node:** ['ElectricityConsumption'] (ID: 85)
- **End Node:** ['Emission'] (ID: 6505)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['ElectricityConsumption'] (ID: 86)
- **End Node:** ['Emission'] (ID: 6506)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['ElectricityConsumption'] (ID: 6448)
- **End Node:** ['Emission'] (ID: 6507)
- **Properties:**
```json
{}
```

#### GENERATES_WASTE (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['WasteGeneration'] (ID: 5371)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['WasteGeneration'] (ID: 5372)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['WasteGeneration'] (ID: 5373)
- **Properties:**
```json
{}
```

#### HAS_COMPLIANCE_RECORD (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['ComplianceRecord'] (ID: 5451)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['ComplianceRecord'] (ID: 5452)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['ComplianceRecord'] (ID: 5453)
- **Properties:**
```json
{}
```

#### HAS_ELECTRICITY_CONSUMPTION (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['ElectricityConsumption'] (ID: 5251)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['ElectricityConsumption'] (ID: 5252)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['ElectricityConsumption'] (ID: 5253)
- **Properties:**
```json
{}
```

#### HAS_EMISSION_FACTOR (Sample)

**Sample 1:**
- **Start Node:** ['Site'] (ID: 91)
- **End Node:** ['EmissionFactor'] (ID: 84)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Site'] (ID: 5461)
- **End Node:** ['EmissionFactor'] (ID: 84)
- **Properties:**
```json
{}
```

#### HAS_ENVIRONMENTAL_KPI (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalKPI'] (ID: 5431)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalKPI'] (ID: 5432)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalKPI'] (ID: 5433)
- **Properties:**
```json
{}
```

#### HAS_ENVIRONMENTAL_RISK (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalRisk'] (ID: 5411)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalRisk'] (ID: 5412)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalRisk'] (ID: 5413)
- **Properties:**
```json
{}
```

#### HAS_INCIDENT (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 180)
- **End Node:** ['Incident'] (ID: 187)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 180)
- **End Node:** ['Incident'] (ID: 72)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 74)
- **End Node:** ['Incident'] (ID: 77)
- **Properties:**
```json
{}
```

#### HAS_RECOMMENDATION (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalRecommendation'] (ID: 5423)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalRecommendation'] (ID: 5424)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['EnvironmentalRecommendation'] (ID: 5425)
- **Properties:**
```json
{}
```

#### HAS_RISK (Sample)

**Sample 1:**
- **Start Node:** ['Site'] (ID: 5461)
- **End Node:** ['RiskAssessment'] (ID: 5520)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Site'] (ID: 5462)
- **End Node:** ['RiskAssessment'] (ID: 5525)
- **Properties:**
```json
{}
```

#### HAS_TARGET (Sample)

**Sample 1:**
- **Start Node:** ['Site'] (ID: 5461)
- **End Node:** ['EnvironmentalTarget'] (ID: 6407)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Site'] (ID: 5461)
- **End Node:** ['EnvironmentalTarget'] (ID: 6408)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Site'] (ID: 5461)
- **End Node:** ['EnvironmentalTarget'] (ID: 6409)
- **Properties:**
```json
{}
```

#### HAS_WATER_CONSUMPTION (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['WaterConsumption'] (ID: 5311)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['WaterConsumption'] (ID: 5312)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 60)
- **End Node:** ['WaterConsumption'] (ID: 5313)
- **Properties:**
```json
{}
```

#### LOCATED_AT (Sample)

No sample data available.

#### LOCATED_IN (Sample)

**Sample 1:**
- **Start Node:** ['Facility'] (ID: 137)
- **End Node:** ['Area'] (ID: 129)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Facility'] (ID: 141)
- **End Node:** ['Area'] (ID: 138)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Facility'] (ID: 144)
- **End Node:** ['Area'] (ID: 143)
- **Properties:**
```json
{}
```

#### MEASURED_AT (Sample)

**Sample 1:**
- **Start Node:** ['EHSMetric'] (ID: 130)
- **End Node:** ['Area'] (ID: 163)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['EHSMetric'] (ID: 131)
- **End Node:** ['Area'] (ID: 163)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['EHSMetric'] (ID: 132)
- **End Node:** ['Area'] (ID: 163)
- **Properties:**
```json
{}
```

#### MONITORS (Sample)

**Sample 1:**
- **Start Node:** ['Meter'] (ID: 63)
- **End Node:** ['Facility'] (ID: 60)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Meter'] (ID: 63)
- **End Node:** ['Facility'] (ID: 60)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Meter'] (ID: 63)
- **End Node:** ['Facility'] (ID: 75)
- **Properties:**
```json
{}
```

#### OCCURRED_AT (Sample)

**Sample 1:**
- **Start Node:** ['Incident'] (ID: 5141)
- **End Node:** ['Area'] (ID: 163)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['Incident'] (ID: 5142)
- **End Node:** ['Area'] (ID: 162)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['Incident'] (ID: 5143)
- **End Node:** ['Area'] (ID: 162)
- **Properties:**
```json
{}
```

#### PERMITS (Sample)

No sample data available.

#### PROVIDED_BY (Sample)

**Sample 1:**
- **Start Node:** ['WaterBill'] (ID: 59)
- **End Node:** ['UtilityProvider'] (ID: 62)
- **Properties:**
```json
{}
```

**Sample 2:**
- **Start Node:** ['WaterBill'] (ID: 65)
- **End Node:** ['UtilityProvider'] (ID: 62)
- **Properties:**
```json
{}
```

**Sample 3:**
- **Start Node:** ['WaterBill'] (ID: 65)
- **End Node:** ['UtilityProvider'] (ID: 181)
- **Properties:**
```json
{}
```

#### RECORDED_IN (Sample)

**Sample 1:**
- **Start Node:** ['Meter'] (ID: 63)
- **End Node:** ['WaterBill'] (ID: 59)
- **Properties:**
```json
{
  "reading_date": "2025-06-30"
}
```

**Sample 2:**
- **Start Node:** ['Meter'] (ID: 68)
- **End Node:** ['UtilityBill'] (ID: 66)
- **Properties:**
```json
{
  "reading_date": "2025-06-30"
}
```

**Sample 3:**
- **Start Node:** ['Meter'] (ID: 69)
- **End Node:** ['UtilityBill'] (ID: 66)
- **Properties:**
```json
{
  "reading_date": "2025-06-30"
}
```

#### RESULTED_IN (Sample)

**Sample 1:**
- **Start Node:** ['WasteShipment'] (ID: 140)
- **End Node:** ['Emission'] (ID: 57)
- **Properties:**
```json
{
  "calculation_method": "waste_disposal_landfill"
}
```

**Sample 2:**
- **Start Node:** ['WaterBill'] (ID: 59)
- **End Node:** ['Emission'] (ID: 64)
- **Properties:**
```json
{
  "calculation_method": "water_treatment_distribution"
}
```

**Sample 3:**
- **Start Node:** ['UtilityBill'] (ID: 66)
- **End Node:** ['Emission'] (ID: 70)
- **Properties:**
```json
{
  "calculation_method": "grid_average"
}
```

#### TRACKS (Sample)

**Sample 1:**
- **Start Node:** ['Document', 'Wastemanifest'] (ID: 6447)
- **End Node:** ['WasteManifest'] (ID: 79)
- **Properties:**
```json
{
  "tracking_date": "2025-09-03T20:21:43.276279"
}
```

#### TRANSPORTED_BY (Sample)

**Sample 1:**
- **Start Node:** ['WasteShipment'] (ID: 140)
- **End Node:** ['Transporter'] (ID: 54)
- **Properties:**
```json
{
  "transport_date": "2025-07-15"
}
```

**Sample 2:**
- **Start Node:** ['WasteShipment'] (ID: 80)
- **End Node:** ['Transporter'] (ID: 54)
- **Properties:**
```json
{
  "transport_date": "2025-07-15"
}
```

**Sample 3:**
- **Start Node:** ['WasteShipment'] (ID: 80)
- **End Node:** ['Transporter'] (ID: 6425)
- **Properties:**
```json
{
  "transport_date": "2025-07-15"
}
```

