# Neo4j Data Transformation Plan

> Created: 2025-08-30
> Version: 1.0.0
> Status: Ready for Implementation

## Overview

This document outlines the complete transformation plan to restructure the current flat facility data structure into a hierarchical organizational structure required for the EHS dashboard. The transformation will create a proper geographic and organizational hierarchy while addressing data quality issues.

## Current Data State Analysis

### Existing Facilities Found
1. **Apex Manufacturing - Plant A**
   - Location: Mechanicsburg, CA
   - Status: Valid facility
   - Issues: None identified

2. **DEMO_FACILITY_001**
   - Status: Duplicate entries detected
   - Issues: Multiple nodes with same identifier
   - Action Required: Deduplication

3. **TEST_FACILITY_PDF_001**
   - Status: Test facility
   - Issues: May need to be preserved for testing

### Data Quality Issues
- Duplicate DEMO_FACILITY_001 entries
- Lack of hierarchical structure
- Missing geographic organization
- No parent-child relationships between locations

## Target Hierarchical Structure

```
Global View (Root)
├── North America (Region)
    ├── Illinois (State/Province)
    │   └── Algonquin Site (Site)
    │       ├── Building A (Building)
    │       ├── Building B (Building)
    │       └── Warehouse C (Building)
    └── Texas (State/Province)
        └── Houston (Site)
            ├── Refinery Complex (Building)
            ├── Storage Terminal (Building)
            └── Admin Building (Building)
```

## Transformation Plan

### Phase 1: Data Cleanup and Preparation

#### Step 1.1: Identify and Remove Duplicate Facilities
```cypher
// Query to identify duplicate DEMO_FACILITY_001 entries
MATCH (f:Facility {name: "DEMO_FACILITY_001"})
RETURN f, id(f) as node_id
ORDER BY node_id;

// Remove duplicate entries (keep the first one found)
MATCH (f:Facility {name: "DEMO_FACILITY_001"})
WITH f, id(f) as node_id
ORDER BY node_id
WITH collect(f)[1..] as duplicates
UNWIND duplicates as duplicate
DETACH DELETE duplicate;
```

#### Step 1.2: Backup Existing Data
```cypher
// Export current facility data for backup
MATCH (f:Facility)
RETURN f.name as facility_name, 
       f.location as location,
       f.address as address,
       f.city as city,
       f.state as state,
       f.country as country,
       labels(f) as labels,
       properties(f) as all_properties;
```

### Phase 2: Create Hierarchical Node Structure

#### Step 2.1: Create Root and Regional Nodes
```cypher
// Create Global View root node
CREATE (global:Organization:Root {
    id: "global-view-001",
    name: "Global View",
    type: "root",
    level: 0,
    created_at: datetime(),
    updated_at: datetime()
});

// Create North America region
CREATE (na:Organization:Region {
    id: "north-america-001",
    name: "North America",
    type: "region",
    level: 1,
    created_at: datetime(),
    updated_at: datetime()
});

// Create relationship between Global and North America
MATCH (global:Root {name: "Global View"})
MATCH (na:Region {name: "North America"})
CREATE (global)-[:CONTAINS]->(na);
```

#### Step 2.2: Create State/Province Level Nodes
```cypher
// Create Illinois state node
CREATE (illinois:Organization:State {
    id: "illinois-001",
    name: "Illinois",
    type: "state",
    country: "United States",
    level: 2,
    created_at: datetime(),
    updated_at: datetime()
});

// Create Texas state node
CREATE (texas:Organization:State {
    id: "texas-001",
    name: "Texas",
    type: "state",
    country: "United States",
    level: 2,
    created_at: datetime(),
    updated_at: datetime()
});

// Create relationships to North America
MATCH (na:Region {name: "North America"})
MATCH (illinois:State {name: "Illinois"})
MATCH (texas:State {name: "Texas"})
CREATE (na)-[:CONTAINS]->(illinois)
CREATE (na)-[:CONTAINS]->(texas);
```

#### Step 2.3: Create Site Level Nodes
```cypher
// Create Algonquin Site
CREATE (algonquin:Organization:Site {
    id: "algonquin-site-001",
    name: "Algonquin Site",
    type: "site",
    address: "2525 Algonquin Road",
    city: "Algonquin",
    state: "Illinois",
    country: "United States",
    postal_code: "60102",
    latitude: 42.1658,
    longitude: -88.2943,
    level: 3,
    created_at: datetime(),
    updated_at: datetime()
});

// Create Houston Site
CREATE (houston:Organization:Site {
    id: "houston-site-001",
    name: "Houston",
    type: "site",
    address: "1000 Louisiana Street",
    city: "Houston",
    state: "Texas",
    country: "United States",
    postal_code: "77002",
    latitude: 29.7604,
    longitude: -95.3698,
    level: 3,
    created_at: datetime(),
    updated_at: datetime()
});

// Create relationships to states
MATCH (illinois:State {name: "Illinois"})
MATCH (texas:State {name: "Texas"})
MATCH (algonquin:Site {name: "Algonquin Site"})
MATCH (houston:Site {name: "Houston"})
CREATE (illinois)-[:CONTAINS]->(algonquin)
CREATE (texas)-[:CONTAINS]->(houston);
```

#### Step 2.4: Create Building Level Nodes
```cypher
// Create buildings for Algonquin Site
CREATE (buildingA:Organization:Building {
    id: "algonquin-building-a-001",
    name: "Building A",
    type: "building",
    building_type: "manufacturing",
    floor_count: 3,
    square_footage: 50000,
    level: 4,
    created_at: datetime(),
    updated_at: datetime()
});

CREATE (buildingB:Organization:Building {
    id: "algonquin-building-b-001",
    name: "Building B",
    type: "building",
    building_type: "office",
    floor_count: 2,
    square_footage: 25000,
    level: 4,
    created_at: datetime(),
    updated_at: datetime()
});

CREATE (warehouseC:Organization:Building {
    id: "algonquin-warehouse-c-001",
    name: "Warehouse C",
    type: "building",
    building_type: "warehouse",
    floor_count: 1,
    square_footage: 75000,
    level: 4,
    created_at: datetime(),
    updated_at: datetime()
});

// Create buildings for Houston Site
CREATE (refinery:Organization:Building {
    id: "houston-refinery-001",
    name: "Refinery Complex",
    type: "building",
    building_type: "industrial",
    floor_count: 4,
    square_footage: 150000,
    level: 4,
    created_at: datetime(),
    updated_at: datetime()
});

CREATE (terminal:Organization:Building {
    id: "houston-terminal-001",
    name: "Storage Terminal",
    type: "building",
    building_type: "storage",
    floor_count: 2,
    square_footage: 100000,
    level: 4,
    created_at: datetime(),
    updated_at: datetime()
});

CREATE (admin:Organization:Building {
    id: "houston-admin-001",
    name: "Admin Building",
    type: "building",
    building_type: "office",
    floor_count: 5,
    square_footage: 30000,
    level: 4,
    created_at: datetime(),
    updated_at: datetime()
});

// Create relationships to sites
MATCH (algonquin:Site {name: "Algonquin Site"})
MATCH (houston:Site {name: "Houston"})
MATCH (buildingA:Building {name: "Building A"})
MATCH (buildingB:Building {name: "Building B"})
MATCH (warehouseC:Building {name: "Warehouse C"})
MATCH (refinery:Building {name: "Refinery Complex"})
MATCH (terminal:Building {name: "Storage Terminal"})
MATCH (admin:Building {name: "Admin Building"})
CREATE (algonquin)-[:CONTAINS]->(buildingA)
CREATE (algonquin)-[:CONTAINS]->(buildingB)
CREATE (algonquin)-[:CONTAINS]->(warehouseC)
CREATE (houston)-[:CONTAINS]->(refinery)
CREATE (houston)-[:CONTAINS]->(terminal)
CREATE (houston)-[:CONTAINS]->(admin);
```

### Phase 3: Handle Existing Facilities

#### Step 3.1: Map Apex Manufacturing to Hierarchy
```cypher
// Update Apex Manufacturing to fit new structure
MATCH (apex:Facility {name: "Apex Manufacturing - Plant A"})
SET apex:Organization:Building,
    apex.id = "apex-plant-a-001",
    apex.type = "building",
    apex.building_type = "manufacturing",
    apex.level = 4,
    apex.updated_at = datetime();

// Create California state if needed for Apex
CREATE (california:Organization:State {
    id: "california-001",
    name: "California",
    type: "state",
    country: "United States",
    level: 2,
    created_at: datetime(),
    updated_at: datetime()
});

// Create Mechanicsburg site for Apex
CREATE (mechanicsburg:Organization:Site {
    id: "mechanicsburg-site-001",
    name: "Mechanicsburg Site",
    type: "site",
    city: "Mechanicsburg",
    state: "California",
    country: "United States",
    level: 3,
    created_at: datetime(),
    updated_at: datetime()
});

// Create relationships
MATCH (na:Region {name: "North America"})
MATCH (california:State {name: "California"})
MATCH (mechanicsburg:Site {name: "Mechanicsburg Site"})
MATCH (apex:Building {name: "Apex Manufacturing - Plant A"})
CREATE (na)-[:CONTAINS]->(california)
CREATE (california)-[:CONTAINS]->(mechanicsburg)
CREATE (mechanicsburg)-[:CONTAINS]->(apex);
```

#### Step 3.2: Handle DEMO_FACILITY_001
```cypher
// Update remaining DEMO_FACILITY_001 to be a proper building
MATCH (demo:Facility {name: "DEMO_FACILITY_001"})
SET demo:Organization:Building,
    demo.id = "demo-facility-001",
    demo.name = "Demo Building",
    demo.type = "building",
    demo.building_type = "demo",
    demo.level = 4,
    demo.updated_at = datetime();

// Assign to Algonquin Site for demo purposes
MATCH (algonquin:Site {name: "Algonquin Site"})
MATCH (demo:Building {name: "Demo Building"})
CREATE (algonquin)-[:CONTAINS]->(demo);
```

#### Step 3.3: Handle TEST_FACILITY_PDF_001
```cypher
// Update test facility
MATCH (test:Facility {name: "TEST_FACILITY_PDF_001"})
SET test:Organization:Building,
    test.id = "test-facility-pdf-001",
    test.name = "PDF Test Building",
    test.type = "building",
    test.building_type = "test",
    test.level = 4,
    test.updated_at = datetime();

// Assign to Houston Site for testing
MATCH (houston:Site {name: "Houston"})
MATCH (test:Building {name: "PDF Test Building"})
CREATE (houston)-[:CONTAINS]->(test);
```

### Phase 4: Create Sample EHS Data

#### Step 4.1: Create Sample Incidents
```cypher
// Create sample incidents for Algonquin Site
MATCH (buildingA:Building {name: "Building A"})
CREATE (incident1:Incident {
    id: "INC-2025-001",
    title: "Minor Slip and Fall",
    description: "Employee slipped on wet floor in manufacturing area",
    severity: "Minor",
    status: "Closed",
    incident_date: date("2025-01-15"),
    reported_date: date("2025-01-15"),
    closed_date: date("2025-01-20"),
    created_at: datetime(),
    updated_at: datetime()
})
CREATE (buildingA)-[:HAS_INCIDENT]->(incident1);

MATCH (refinery:Building {name: "Refinery Complex"})
CREATE (incident2:Incident {
    id: "INC-2025-002",
    title: "Chemical Spill Containment",
    description: "Small chemical spill properly contained and cleaned",
    severity: "Major",
    status: "Under Investigation",
    incident_date: date("2025-02-10"),
    reported_date: date("2025-02-10"),
    created_at: datetime(),
    updated_at: datetime()
})
CREATE (refinery)-[:HAS_INCIDENT]->(incident2);
```

#### Step 4.2: Create Sample Compliance Records
```cypher
// Create compliance records
MATCH (algonquin:Site {name: "Algonquin Site"})
CREATE (compliance1:ComplianceRecord {
    id: "COMP-2025-001",
    regulation: "OSHA 29 CFR 1910",
    status: "Compliant",
    audit_date: date("2025-01-01"),
    next_audit_date: date("2025-07-01"),
    auditor: "EHS Team",
    created_at: datetime(),
    updated_at: datetime()
})
CREATE (algonquin)-[:HAS_COMPLIANCE]->(compliance1);

MATCH (houston:Site {name: "Houston"})
CREATE (compliance2:ComplianceRecord {
    id: "COMP-2025-002",
    regulation: "EPA Clean Air Act",
    status: "Under Review",
    audit_date: date("2025-02-01"),
    next_audit_date: date("2025-08-01"),
    auditor: "Environmental Consulting Group",
    created_at: datetime(),
    updated_at: datetime()
})
CREATE (houston)-[:HAS_COMPLIANCE]->(compliance2);
```

### Phase 5: Validation and Testing

#### Step 5.1: Validate Hierarchy Structure
```cypher
// Verify complete hierarchy paths
MATCH path = (root:Root)-[:CONTAINS*]->(leaf)
WHERE NOT (leaf)-[:CONTAINS]->()
RETURN [node in nodes(path) | node.name] as hierarchy_path,
       length(path) as depth
ORDER BY hierarchy_path;
```

#### Step 5.2: Validate Data Consistency
```cypher
// Check for orphaned nodes
MATCH (n:Organization)
WHERE NOT (n:Root) AND NOT ()-[:CONTAINS]->(n)
RETURN n.name as orphaned_node, labels(n) as node_labels;

// Verify all buildings have proper relationships
MATCH (b:Building)
OPTIONAL MATCH (site:Site)-[:CONTAINS]->(b)
RETURN b.name as building_name, 
       CASE WHEN site IS NULL THEN 'ORPHANED' ELSE site.name END as parent_site;
```

## Rollback Procedures

### Emergency Rollback (Complete Restoration)
```cypher
// Step 1: Remove all new hierarchy nodes
MATCH (n:Organization)
WHERE n.created_at >= datetime("2025-08-30T00:00:00Z")
DETACH DELETE n;

// Step 2: Restore original facility labels
MATCH (f)
WHERE f:Building AND NOT f:Facility
SET f:Facility
REMOVE f:Organization, f:Building;

// Step 3: Remove hierarchy-specific properties
MATCH (f:Facility)
REMOVE f.level, f.type, f.building_type, f.floor_count, f.square_footage;
```

### Partial Rollback (Specific Components)
```cypher
// Remove specific site and its children
MATCH (site:Site {name: "Algonquin Site"})-[:CONTAINS*0..]->(child)
DETACH DELETE site, child;

// Remove specific building from hierarchy
MATCH (building:Building {name: "Building A"})
DETACH DELETE building;
```

## Implementation Timeline

### Phase 1: Preparation (Day 1)
- Backup existing data
- Clean duplicate entries
- Validate current data state

### Phase 2: Core Hierarchy (Day 2)
- Create root, region, and state nodes
- Establish primary relationships
- Validate basic structure

### Phase 3: Sites and Buildings (Day 3)
- Create site-level nodes
- Create building-level nodes
- Map existing facilities

### Phase 4: Sample Data (Day 4)
- Create sample incidents
- Create sample compliance records
- Populate test data

### Phase 5: Validation (Day 5)
- Run all validation queries
- Test dashboard connectivity
- Performance testing

## Success Criteria

1. **Structure Completeness**
   - All hierarchy levels created (Root → Region → State → Site → Building)
   - All relationships properly established
   - No orphaned nodes

2. **Data Integrity**
   - No duplicate facilities
   - All existing facilities properly mapped
   - Sample data created for testing

3. **Dashboard Compatibility**
   - Dashboard can query hierarchical structure
   - All required data fields populated
   - Performance meets requirements

## Risk Mitigation

### Data Loss Prevention
- Complete backup before transformation
- Incremental validation at each step
- Rollback procedures tested

### Performance Considerations
- Index creation for frequently queried properties
- Batch operations for large datasets
- Monitor query performance

### Testing Strategy
- Validate each phase independently
- Test dashboard connectivity after each major change
- Performance benchmarking

## Post-Implementation Tasks

1. **Index Creation**
```cypher
CREATE INDEX FOR (o:Organization) ON (o.id);
CREATE INDEX FOR (o:Organization) ON (o.name);
CREATE INDEX FOR (o:Organization) ON (o.type);
CREATE INDEX FOR (o:Organization) ON (o.level);
```

2. **Performance Optimization**
```cypher
// Create composite index for hierarchy queries
CREATE INDEX FOR (o:Organization) ON (o.type, o.level);
```

3. **Dashboard Integration Testing**
- Test all dashboard queries
- Validate data aggregations
- Performance benchmarking

## Conclusion

This transformation plan provides a comprehensive approach to restructuring the Neo4j database from a flat facility structure to a proper hierarchical organization. The plan includes data cleanup, new structure creation, sample data generation, and comprehensive rollback procedures to ensure safe implementation.

The hierarchical structure will enable the dashboard to provide proper drill-down capabilities from global view to specific buildings, while maintaining data integrity and performance.