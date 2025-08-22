# EHS AI Demo - Docker Setup

This Docker Compose setup provides a complete, self-sufficient environment for the EHS AI Demo application with no cloud dependencies.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │  EHS Analytics  │
│   (React/Nginx) │    │   (FastAPI)      │    │   (FastAPI)     │
│   Port: 8080    │    │   Port: 8000     │    │   Port: 8001    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────┬───────────┴────────────┬───────────┘
                      │                        │
              ┌───────▼────────┐      ┌────────▼────────┐
              │   Neo4j DB     │      │  Shared Storage │
              │   Port: 7474   │      │   (Volumes)     │
              │   Port: 7687   │      │                 │
              └────────────────┘      └─────────────────┘
```

## Services

### 1. Neo4j Database (`neo4j`)
- **Image**: `neo4j:5.13.0`
- **Ports**: 
  - `7474` - Neo4j Browser (HTTP)
  - `7687` - Bolt protocol
- **Features**: APOC and Graph Data Science plugins
- **Credentials**: `neo4j/password123` (change in production)

### 2. Data Foundation Backend (`data-foundation-backend`)
- **Build**: `./data-foundation/backend`
- **Port**: `8000`
- **Purpose**: Document processing, knowledge graph construction
- **Dependencies**: Neo4j

### 3. Data Foundation Frontend (`data-foundation-frontend`)
- **Build**: `./data-foundation/frontend`
- **Port**: `8080`
- **Purpose**: Web interface for document upload and management
- **Dependencies**: Data Foundation Backend

### 4. EHS Analytics (`ehs-analytics`)
- **Build**: `./ehs-analytics`
- **Port**: `8001`
- **Purpose**: AI-powered EHS analytics and natural language querying
- **Dependencies**: Neo4j, Data Foundation Backend

## Volume Structure

```
./volumes/
├── neo4j/
│   ├── data/          # Neo4j database files
│   ├── logs/          # Neo4j logs
│   ├── import/        # Import directory
│   └── plugins/       # Neo4j plugins
├── app-data/          # Shared application data
└── logs/
    ├── backend/       # Backend service logs
    └── analytics/     # Analytics service logs

./input-documents/     # Input documents (mounted read-only)
├── pdfs/
├── reports/
├── regulations/
└── permits/
```

## Quick Start

1. **Setup Environment**:
   ```bash
   ./setup-volumes.sh
   ```

2. **Configure API Keys** (optional but recommended):
   ```bash
   # Edit the .env file
   nano ehs-analytics/.env
   
   # Add your API keys:
   # OPENAI_API_KEY=your_key_here
   # ANTHROPIC_API_KEY=your_key_here
   ```

3. **Start Services**:
   ```bash
   docker-compose up -d
   ```

4. **Verify Health**:
   ```bash
   # Check all services are running
   docker-compose ps
   
   # Check service health
   curl http://localhost:8000/health  # Data Foundation
   curl http://localhost:8001/health  # EHS Analytics
   ```

5. **Access Applications**:
   - **Frontend UI**: http://localhost:8080
   - **Data Foundation API**: http://localhost:8000/docs
   - **EHS Analytics API**: http://localhost:8001/docs
   - **Neo4j Browser**: http://localhost:7474

## Development Mode

The setup includes development-friendly features:

- **Live Code Reloading**: Source code changes are reflected automatically
- **Debug Logging**: Enhanced logging for development
- **Development Overrides**: `docker-compose.override.yml` for dev-specific settings

### Development Commands

```bash
# Start in development mode (default)
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Restart a specific service
docker-compose restart [service-name]

# Rebuild a service
docker-compose up -d --build [service-name]

# Access service shell
docker-compose exec [service-name] /bin/bash
```

## Configuration

### Environment Variables

Each service can be configured via environment variables:

#### Data Foundation Backend
- `NEO4J_URI`: Neo4j connection string
- `NEO4J_USERNAME/PASSWORD`: Database credentials
- `LOG_LEVEL`: Logging level

#### EHS Analytics
- `NEO4J_URI`: Neo4j connection string
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `DATA_FOUNDATION_API_URL`: Data Foundation backend URL

#### Neo4j
- `NEO4J_AUTH`: Username/password
- `NEO4J_PLUGINS`: Enabled plugins
- Memory and performance settings

### Custom Configuration

1. **Override Environment Variables**:
   ```bash
   # Create a .env file in the root directory
   echo "NEO4J_PASSWORD=your_secure_password" > .env
   ```

2. **Custom Docker Compose Override**:
   ```yaml
   # docker-compose.override.yml
   version: '3.8'
   services:
     ehs-analytics:
       environment:
         - CUSTOM_SETTING=value
   ```

## Data Management

### Input Documents

Place documents in the `input-documents/` directory:

```bash
# Example: Add PDF reports
cp /path/to/your/reports/*.pdf input-documents/reports/

# The containers will have read-only access to these files
```

### Database Backup

```bash
# Backup Neo4j data
docker-compose exec neo4j neo4j-admin database dump neo4j

# Copy backup from container
docker-compose cp neo4j:/dumps/neo4j.dump ./backups/
```

### Logs Access

```bash
# View live logs
docker-compose logs -f

# Access log files directly
tail -f volumes/logs/backend/app.log
tail -f volumes/logs/analytics/app.log
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**:
   ```bash
   # Check if ports are in use
   netstat -tulpn | grep :8080
   
   # Modify ports in docker-compose.yml if needed
   ```

2. **Permission Issues**:
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER volumes/
   chmod -R 755 volumes/
   ```

3. **Neo4j Connection Issues**:
   ```bash
   # Check Neo4j logs
   docker-compose logs neo4j
   
   # Test connection
   docker-compose exec neo4j cypher-shell -u neo4j -p password123
   ```

4. **Service Won't Start**:
   ```bash
   # Check service logs
   docker-compose logs [service-name]
   
   # Rebuild service
   docker-compose up -d --build [service-name]
   ```

### Health Checks

All services include health checks:

```bash
# Check health status
docker-compose ps

# Service-specific health endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### Resource Requirements

**Minimum Requirements**:
- RAM: 8GB
- Storage: 10GB free space
- CPU: 4 cores recommended

**Recommended for Production**:
- RAM: 16GB+
- Storage: 50GB+ SSD
- CPU: 8+ cores

## Security Considerations

### Development vs Production

**Development** (default configuration):
- Uses default passwords
- Debug logging enabled
- Permissive CORS settings
- No SSL/TLS

**Production Recommendations**:
1. Change default passwords
2. Use environment variables for secrets
3. Enable SSL/TLS
4. Restrict CORS origins
5. Use proper firewall rules
6. Regular security updates

### Secrets Management

```bash
# Use Docker secrets or external secret management
# Never commit API keys to version control

# Example with .env file (not committed):
cat > .env << 'EOF'
NEO4J_PASSWORD=secure_password_here
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
EOF
```

## Monitoring

### Service Monitoring

```bash
# Resource usage
docker stats

# Service status
docker-compose ps

# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### Log Monitoring

```bash
# Aggregate logs
docker-compose logs -f

# Service-specific logs
docker-compose logs -f ehs-analytics
docker-compose logs -f data-foundation-backend
```

## Support

For issues and questions:

1. Check service logs: `docker-compose logs [service-name]`
2. Verify health endpoints
3. Review configuration files
4. Check resource usage: `docker stats`

## License

This Docker setup is part of the EHS AI Demo project and follows the same licensing terms.