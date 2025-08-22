# Neo4j Configuration for EHS AI Platform

## Docker Container Details

**Container Name**: `ehs-neo4j`  
**Image**: `neo4j:5.23.0`  
**Status**: Running

## Connection Details

- **Browser URL**: http://localhost:7474
- **Bolt URL**: bolt://localhost:7687
- **Username**: `neo4j`
- **Password**: `EhsAI2024!`

## Configuration

### Ports
- `7474`: HTTP/Browser interface
- `7687`: Bolt protocol

### Memory Settings
- Initial heap size: 2GB
- Maximum heap size: 4GB

### Plugins
- APOC (Awesome Procedures on Cypher)
- Graph Data Science Library

### Volume Mappings
- Data: `~/neo4j/ehs-data:/data`
- Logs: `~/neo4j/ehs-logs:/logs`
- Import: `~/neo4j/ehs-import:/var/lib/neo4j/import`
- Plugins: `~/neo4j/ehs-plugins:/plugins`

## Docker Commands

### Start Container
```bash
docker start ehs-neo4j
```

### Stop Container
```bash
docker stop ehs-neo4j
```

### View Logs
```bash
docker logs ehs-neo4j
```

### Access Neo4j Shell
```bash
docker exec -it ehs-neo4j cypher-shell -u neo4j -p 'EhsAI2024!'
```

## Initial Setup Commands

### Create Indexes for EHS Schema
```cypher
// Create indexes for better query performance
CREATE INDEX facility_id IF NOT EXISTS FOR (f:Facility) ON (f.id);
CREATE INDEX facility_name IF NOT EXISTS FOR (f:Facility) ON (f.name);
CREATE INDEX utility_bill_id IF NOT EXISTS FOR (u:UtilityBill) ON (u.id);
CREATE INDEX permit_id IF NOT EXISTS FOR (p:Permit) ON (p.id);
CREATE INDEX equipment_id IF NOT EXISTS FOR (e:Equipment) ON (e.id);
CREATE INDEX emission_id IF NOT EXISTS FOR (e:Emission) ON (e.id);

// Create composite indexes for time-series queries
CREATE INDEX utility_bill_time IF NOT EXISTS FOR (u:UtilityBill) ON (u.period_start, u.period_end);
CREATE INDEX emission_date IF NOT EXISTS FOR (e:Emission) ON (e.date);
```

### Vector Index for Semantic Search
```cypher
// Create vector index for document embeddings
CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
FOR (d:Document) 
ON (d.embedding) 
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}
```

## Integration with Data Foundation

The Neo4j instance should be configured in the data-foundation `.env` file:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=EhsAI2024!
```

## Backup Strategy

Weekly backups should be created using:
```bash
docker exec -t ehs-neo4j neo4j-admin database dump neo4j --to-stdout > ehs-neo4j-backup-$(date +%Y%m%d).dump
```

## Monitoring

Access Neo4j metrics at: http://localhost:7474/metrics

Last updated: 2025-08-17