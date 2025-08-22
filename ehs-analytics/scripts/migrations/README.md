# Database Migrations

This directory contains Neo4j database migrations for the EHS Analytics system.

## Overview

These migrations add the missing Equipment and Permit entities to the Neo4j schema, along with their relationships to existing entities.

## Migration Files

| File | Description |
|------|-------------|
| `001_add_equipment_entity.py` | Creates Equipment entity with properties and constraints |
| `002_add_permit_entity.py` | Creates Permit entity with properties and constraints |
| `003_add_relationships.py` | Creates relationships between entities |

## Entity Schemas

### Equipment Entity
- **id**: Unique identifier (string)
- **equipment_type**: Type of equipment (string)
- **model**: Equipment model (string) 
- **efficiency_rating**: Efficiency rating value (float)
- **installation_date**: Installation date (date)

### Permit Entity
- **id**: Unique identifier (string)
- **permit_type**: Type of permit (string)
- **limit**: Permit limit value (float)
- **unit**: Unit of measurement (string)
- **expiration_date**: Expiration date (date)
- **regulatory_authority**: Issuing authority (string)

## Relationships

- `Equipment -> LOCATED_AT -> Facility`
- `Equipment -> AFFECTS_CONSUMPTION -> UtilityBill`
- `Permit -> PERMITS -> Facility`

## Running Migrations

### Prerequisites

1. Neo4j database running on `bolt://localhost:7687`
2. Environment variables set in `.env` file:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=EhsAI2024!
   ```

### Test Setup

```bash
python3 scripts/test_migrations.py
```

### Run Migrations

```bash
# Run all pending migrations
python3 scripts/run_migrations.py

# Check migration status
python3 scripts/run_migrations.py status
```

### Individual Migration

```bash
python3 scripts/migrations/001_add_equipment_entity.py
python3 scripts/migrations/002_add_permit_entity.py
python3 scripts/migrations/003_add_relationships.py
```

## Features

- **Idempotent**: Safe to run multiple times
- **Tracking**: Migrations are tracked in Neo4j
- **Logging**: Detailed logging of all operations
- **Verification**: Built-in schema verification
- **Error Handling**: Proper error handling and rollback

## Constraints and Indexes

Each entity gets:
- Unique constraint on `id` field
- Not-null constraint on `id` field
- Performance indexes on commonly queried fields

## Migration Tracking

Migrations are tracked using `Migration` nodes in Neo4j:
```cypher
(:Migration {
  name: "001_add_equipment_entity",
  applied_at: datetime(),
  description: "..."
})
```