# EHS Analytics Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the EHS Analytics platform in production environments. The platform consists of multiple components including the FastAPI application, Neo4j database, Redis cache, and various AI services.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Environment Configuration](#environment-configuration)
4. [Database Setup](#database-setup)
5. [Application Deployment](#application-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security Configuration](#security-configuration)
8. [Load Balancing and Scaling](#load-balancing-and-scaling)
9. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 100 GB SSD
- **Network**: 1 Gbps connection

#### Recommended Production Requirements
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: 500+ GB NVMe SSD
- **Network**: 10 Gbps connection

#### High Availability Setup
- **Load Balancer**: 2+ instances
- **Application Servers**: 3+ instances  
- **Database Cluster**: 3+ Neo4j nodes
- **Cache Cluster**: 3+ Redis nodes

### Software Dependencies

#### Required Software
- **Operating System**: Ubuntu 22.04 LTS, CentOS 8, or Amazon Linux 2
- **Python**: 3.9+ with pip
- **Docker**: 24.0+ with Docker Compose
- **Git**: 2.30+
- **Nginx**: 1.20+ (for reverse proxy)

#### Optional Components
- **Kubernetes**: 1.25+ (for container orchestration)
- **Terraform**: 1.5+ (for infrastructure as code)
- **Ansible**: 2.12+ (for configuration management)

### External Services

#### Required Services
- **Neo4j Database**: 5.13+ Enterprise
- **Redis**: 7.0+ (for caching and session storage)
- **OpenAI API**: GPT-4 access or Anthropic Claude
- **SSL Certificate**: Let's Encrypt or commercial certificate

#### Optional Services
- **Vector Stores**: Pinecone, Weaviate, or Qdrant
- **Monitoring**: Prometheus + Grafana
- **Log Aggregation**: ELK Stack or Fluentd
- **APM**: New Relic, DataDog, or Sentry

## Infrastructure Requirements

### Cloud Provider Recommendations

#### AWS Configuration
```yaml
# Recommended AWS Setup
EC2 Instances:
  - Application: t3.large (2 vCPU, 8 GB RAM) x 3
  - Database: r5.xlarge (4 vCPU, 32 GB RAM) x 3
  - Load Balancer: Application Load Balancer
  - Cache: ElastiCache Redis Cluster

Storage:
  - EBS GP3: 500 GB for application data
  - EBS GP3: 1 TB for database storage
  - S3: For backups and static assets

Network:
  - VPC with private/public subnets
  - NAT Gateway for outbound traffic
  - Security Groups for traffic control
```

#### Docker Compose (Single Server)
```yaml
version: '3.8'

services:
  ehs-analytics:
    image: ehs-analytics:latest
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
    depends_on:
      - neo4j
      - redis
    restart: unless-stopped
    
  neo4j:
    image: neo4j:5.13-enterprise
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/production_password
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    restart: unless-stopped
    
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    command: redis-server --requirepass redis_password
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - ehs-analytics
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  redis_data:
```

#### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ehs-analytics
  namespace: ehs
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ehs-analytics
  template:
    metadata:
      labels:
        app: ehs-analytics
    spec:
      containers:
      - name: ehs-analytics
        image: ehs-analytics:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: NEO4J_URI
          valueFrom:
            secretKeyRef:
              name: ehs-secrets
              key: neo4j-uri
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ehs-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Environment Configuration

### Environment Variables

Create a `.env.production` file with the following variables:

```bash
# Application Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
NEO4J_URI=bolt://neo4j-cluster:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_neo4j_password
NEO4J_DATABASE=ehs_production
NEO4J_MAX_CONNECTION_LIFETIME=3600
NEO4J_MAX_CONNECTION_POOL_SIZE=100

# Redis Configuration
REDIS_URL=redis://redis-cluster:6379
REDIS_PASSWORD=your_secure_redis_password
REDIS_TTL_SECONDS=3600

# AI Model Configuration
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
LLAMA_PARSE_API_KEY=your_llama_parse_api_key

# Vector Store Configuration (Optional)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
WEAVIATE_URL=https://your-weaviate-instance.com
WEAVIATE_API_KEY=your_weaviate_api_key
QDRANT_URL=https://your-qdrant-instance.com
QDRANT_API_KEY=your_qdrant_api_key

# Security Configuration
JWT_SECRET_KEY=your_jwt_secret_key_min_32_chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
CORS_ORIGINS=https://dashboard.ehs-analytics.company.com,https://admin.ehs-analytics.company.com

# Monitoring and Logging
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
PROMETHEUS_PORT=9090
ENABLE_METRICS=true

# Performance Configuration
MAX_WORKERS=4
WORKER_TIMEOUT=300
KEEP_ALIVE=2
MAX_REQUEST_SIZE=100000000
QUERY_TIMEOUT_SECONDS=600

# Email Configuration (for alerts)
SMTP_HOST=smtp.company.com
SMTP_PORT=587
SMTP_USERNAME=noreply@company.com
SMTP_PASSWORD=smtp_password
SMTP_USE_TLS=true
```

### Configuration File Override

Create `config/production.yaml`:

```yaml
# Production Configuration Override
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  max_request_size: 100_000_000
  cors_origins:
    - "https://dashboard.ehs-analytics.company.com"
    - "https://admin.ehs-analytics.company.com"

# Database optimizations for production
neo4j:
  max_connection_lifetime: 3600
  max_connection_pool_size: 100
  connection_timeout: 30
  max_retry_time: 30

# Caching configuration
cache:
  redis:
    ttl_seconds: 3600
    max_connections: 50
  
  in_memory:
    max_size: 5000
    ttl_seconds: 1800

# Logging configuration
logging:
  level: "INFO"
  format: "json"
  handlers:
    - type: "console"
      format: "json"
    - type: "file"
      filename: "/var/log/ehs_analytics/app.log"
      max_bytes: 50_000_000
      backup_count: 10
    - type: "syslog"
      facility: "local0"

# Monitoring configuration
monitoring:
  prometheus:
    enabled: true
    port: 9090
    metrics_path: "/metrics"
  
  health_check:
    enabled: true
    interval_seconds: 30
    timeout_seconds: 10

# Performance tuning
performance:
  max_workers: 4
  worker_timeout: 300
  keep_alive: 2
  max_concurrent_queries: 100
  query_cache_size: 1000

# Security settings
security:
  jwt_algorithm: "HS256"
  access_token_expire_minutes: 30
  refresh_token_expire_days: 7
  password_min_length: 12
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
```

## Database Setup

### Neo4j Production Configuration

#### Single Instance Setup
```bash
# Install Neo4j Enterprise
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j-enterprise=1:5.13.0

# Configure Neo4j
sudo systemctl stop neo4j
```

Neo4j configuration (`/etc/neo4j/neo4j.conf`):
```properties
# Network configuration
server.default_listen_address=0.0.0.0
server.bolt.listen_address=0.0.0.0:7687
server.http.listen_address=0.0.0.0:7474
server.https.listen_address=0.0.0.0:7473

# Security
dbms.security.auth_enabled=true
dbms.security.allow_csv_import_from_file_urls=false

# Performance tuning
server.memory.heap.initial_size=4g
server.memory.heap.max_size=4g
server.memory.pagecache.size=2g

# Transaction timeout
dbms.transaction.timeout=60s
dbms.transaction.concurrent.maximum=1000

# Logging
dbms.logs.query.enabled=INFO
dbms.logs.query.threshold=2s
dbms.logs.query.parameter_logging_enabled=true

# Backup
dbms.backup.enabled=true
dbms.backup.listen_address=0.0.0.0:6362

# Enterprise features
dbms.license.acceptlicense=true
dbms.mode=SINGLE
```

#### High Availability Cluster
```bash
# Core Server Configuration
dbms.mode=CORE
causal_clustering.minimum_core_cluster_size_at_formation=3
causal_clustering.minimum_core_cluster_size_at_runtime=3
causal_clustering.initial_discovery_members=neo4j-core-1:5000,neo4j-core-2:5000,neo4j-core-3:5000
causal_clustering.discovery_listen_address=0.0.0.0:5000
causal_clustering.transaction_listen_address=0.0.0.0:6000
causal_clustering.raft_listen_address=0.0.0.0:7000
```

### Database Initialization

```bash
#!/bin/bash
# scripts/init_production_db.sh

echo "Initializing EHS Analytics Production Database..."

# Start Neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to start..."
until cypher-shell -u neo4j -p neo4j "RETURN 1;" 2>/dev/null; do
  echo "Neo4j is unavailable - sleeping"
  sleep 5
done

echo "Neo4j is ready"

# Change default password
echo "Changing default password..."
cypher-shell -u neo4j -p neo4j "ALTER CURRENT USER SET PASSWORD '$NEO4J_PASSWORD';"

# Run migrations
echo "Running database migrations..."
python3 scripts/run_migrations.py --environment production

# Create indexes for performance
echo "Creating performance indexes..."
cypher-shell -u neo4j -p $NEO4J_PASSWORD << EOF
// Facility indexes
CREATE INDEX facility_name_idx IF NOT EXISTS FOR (f:Facility) ON (f.name);
CREATE INDEX facility_type_idx IF NOT EXISTS FOR (f:Facility) ON (f.type);

// Equipment indexes
CREATE INDEX equipment_facility_idx IF NOT EXISTS FOR (e:Equipment) ON (e.facility_id);
CREATE INDEX equipment_type_idx IF NOT EXISTS FOR (e:Equipment) ON (e.type);

// Permit indexes
CREATE INDEX permit_facility_idx IF NOT EXISTS FOR (p:Permit) ON (p.facility_id);
CREATE INDEX permit_expiration_idx IF NOT EXISTS FOR (p:Permit) ON (p.expiration_date);

// Document indexes
CREATE INDEX document_facility_idx IF NOT EXISTS FOR (d:Document) ON (d.facility_id);
CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON (d.type);
CREATE INDEX document_date_idx IF NOT EXISTS FOR (d:Document) ON (d.date);

// Vector index for embeddings
CREATE VECTOR INDEX ehs_embeddings_idx IF NOT EXISTS
FOR (d:Document) ON (d.embedding)
OPTIONS {indexConfig: {
  "vector.dimensions": 1536,
  "vector.similarity_function": "cosine"
}};

// Full-text indexes
CREATE FULLTEXT INDEX document_content_idx IF NOT EXISTS 
FOR (d:Document) ON EACH [d.content, d.title, d.description];
EOF

echo "Database initialization completed!"
```

### Redis Configuration

Redis configuration (`/etc/redis/redis.conf`):
```conf
# Network
bind 0.0.0.0
port 6379
protected-mode yes
requirepass your_secure_redis_password

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Performance
tcp-backlog 511
timeout 0
tcp-keepalive 300

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command SHUTDOWN SHUTDOWN_SECRET
```

## Application Deployment

### Docker Image Build

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r ehs && useradd -r -g ehs ehs

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directories
RUN mkdir -p /app/logs /app/config \
    && chown -R ehs:ehs /app

# Switch to non-root user
USER ehs
WORKDIR /app

# Copy application code
COPY --chown=ehs:ehs src/ ./src/
COPY --chown=ehs:ehs config/ ./config/
COPY --chown=ehs:ehs scripts/ ./scripts/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "src.ehs_analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Build and Deploy Script

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

# Configuration
IMAGE_NAME="ehs-analytics"
IMAGE_TAG="${CI_COMMIT_SHA:-latest}"
REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
ENVIRONMENT="${DEPLOY_ENV:-production}"

echo "Deploying EHS Analytics ${IMAGE_TAG} to ${ENVIRONMENT}..."

# Build Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# Push to registry
echo "Pushing to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# Deploy based on environment
case ${ENVIRONMENT} in
  "production")
    echo "Deploying to production..."
    # Update docker-compose file with new image tag
    sed -i "s|image: ${IMAGE_NAME}:.*|image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" docker-compose.prod.yml
    
    # Deploy with zero downtime
    docker-compose -f docker-compose.prod.yml up -d --force-recreate --no-deps ehs-analytics
    
    # Health check
    echo "Waiting for health check..."
    for i in {1..30}; do
      if curl -f http://localhost:8000/health; then
        echo "Deployment successful!"
        exit 0
      fi
      sleep 10
    done
    
    echo "Health check failed!"
    exit 1
    ;;
    
  "kubernetes")
    echo "Deploying to Kubernetes..."
    kubectl set image deployment/ehs-analytics ehs-analytics=${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} -n ehs
    kubectl rollout status deployment/ehs-analytics -n ehs
    ;;
    
  *)
    echo "Unknown environment: ${ENVIRONMENT}"
    exit 1
    ;;
esac
```

### Nginx Configuration

Create `/etc/nginx/sites-available/ehs-analytics`:
```nginx
upstream ehs_backend {
    server 127.0.0.1:8000;
    # Add more servers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
    
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=auth:10m rate=10r/m;

# SSL Configuration
ssl_certificate /etc/letsencrypt/live/api.ehs-analytics.company.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/api.ehs-analytics.company.com/privkey.pem;
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_timeout 10m;
ssl_session_cache shared:SSL:10m;

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name api.ehs-analytics.company.com;
    return 301 https://$server_name$request_uri;
}

# Main server block
server {
    listen 443 ssl http2;
    server_name api.ehs-analytics.company.com;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header Content-Security-Policy "default-src 'self'" always;
    
    # Logging
    access_log /var/log/nginx/ehs-analytics-access.log;
    error_log /var/log/nginx/ehs-analytics-error.log;
    
    # Basic configuration
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    keepalive_timeout 65s;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/xml application/rss+xml 
               application/atom+xml application/javascript;
    
    # API endpoints
    location /api/ {
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        
        # Proxy settings
        proxy_pass http://ehs_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://ehs_backend;
        proxy_set_header Host $host;
        access_log off;
    }
    
    # Documentation endpoints
    location ~ ^/(docs|redoc|openapi.json) {
        proxy_pass http://ehs_backend;
        proxy_set_header Host $host;
    }
    
    # Root endpoint
    location = / {
        proxy_pass http://ehs_backend;
        proxy_set_header Host $host;
    }
    
    # Deny all other locations
    location / {
        return 404;
    }
}
```

## Monitoring and Observability

### Prometheus Configuration

Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ehs_analytics_rules.yml"

scrape_configs:
  - job_name: 'ehs-analytics'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'neo4j'
    static_configs:
      - targets: ['localhost:2004']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
      
  - job_name: 'nginx'
    static_configs:
      - targets: ['localhost:9113']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

Create `ehs_analytics_rules.yml`:
```yaml
groups:
  - name: ehs-analytics
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 10% for 2 minutes"
      
      # High response time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time"
          description: "95th percentile response time is above 2 seconds"
      
      # Database connection issues
      - alert: DatabaseDown
        expr: up{job="neo4j"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Neo4j database is down"
          description: "Neo4j database has been down for more than 1 minute"
      
      # Cache issues
      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Redis cache is down"
          description: "Redis cache has been down for more than 1 minute"
      
      # High memory usage
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / process_virtual_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 90%"
      
      # Query queue backup
      - alert: QueryQueueBackup
        expr: query_queue_size > 100
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Query queue backup"
          description: "Query queue has more than 100 pending queries"
```

### Grafana Dashboard

Create `dashboards/ehs-analytics.json`:
```json
{
  "dashboard": {
    "title": "EHS Analytics Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])"
          }
        ]
      },
      {
        "title": "Active Queries",
        "type": "stat",
        "targets": [
          {
            "expr": "query_active_count"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "neo4j_database_pool_total_size"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

Create `config/logging.conf`:
```ini
[loggers]
keys=root,ehs_analytics

[handlers]
keys=consoleHandler,fileHandler,syslogHandler

[formatters]
keys=jsonFormatter,simpleFormatter

[logger_root]
level=INFO
handlers=consoleHandler,fileHandler

[logger_ehs_analytics]
level=INFO
handlers=consoleHandler,fileHandler,syslogHandler
qualname=ehs_analytics
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=jsonFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=handlers.RotatingFileHandler
level=INFO
formatter=jsonFormatter
args=('/var/log/ehs_analytics/app.log', 'a', 50000000, 10)

[handler_syslogHandler]
class=handlers.SysLogHandler
level=INFO
formatter=jsonFormatter
args=(('localhost', 514),)

[formatter_jsonFormatter]
format={"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s","module":"%(module)s","function":"%(funcName)s","line":%(lineno)d}
datefmt=%Y-%m-%dT%H:%M:%S%z

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Security Configuration

### SSL/TLS Setup

```bash
#!/bin/bash
# scripts/setup_ssl.sh

# Install Certbot
sudo apt update
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d api.ehs-analytics.company.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Firewall Configuration

```bash
#!/bin/bash
# scripts/setup_firewall.sh

# Install UFW
sudo apt install ufw

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH access
sudo ufw allow ssh

# HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Database access (only from app servers)
sudo ufw allow from 10.0.1.0/24 to any port 7687
sudo ufw allow from 10.0.1.0/24 to any port 6379

# Monitoring
sudo ufw allow from 10.0.1.0/24 to any port 9090

# Enable firewall
sudo ufw --force enable
```

### Security Headers

Add to Nginx configuration:
```nginx
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none';" always;
```

### JWT Configuration

In production configuration:
```yaml
security:
  jwt:
    secret_key: "your-256-bit-secret-key-here"
    algorithm: "HS256"
    access_token_expire_minutes: 30
    refresh_token_expire_days: 7
  
  password_policy:
    min_length: 12
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_special_chars: true
    
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
    whitelist_ips:
      - "10.0.0.0/8"
      - "172.16.0.0/12"
      - "192.168.0.0/16"
```

## Load Balancing and Scaling

### Horizontal Scaling

#### Docker Swarm Setup
```bash
# Initialize swarm
docker swarm init

# Create overlay network
docker network create -d overlay ehs-network

# Deploy stack
docker stack deploy -c docker-compose.prod.yml ehs-stack
```

#### Kubernetes HPA
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ehs-analytics-hpa
  namespace: ehs
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ehs-analytics
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling

#### Neo4j Cluster Scaling
```bash
# Add read replica
causal_clustering.read_replica_name=read-replica-1
causal_clustering.initial_discovery_members=core-1:5000,core-2:5000,core-3:5000

# Configure read routing
NEO4J_READ_REPLICA_ROUTING=true
```

#### Redis Cluster
```yaml
version: '3.8'
services:
  redis-node-1:
    image: redis:7.2-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    
  redis-node-2:
    image: redis:7.2-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    
  redis-node-3:
    image: redis:7.2-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
```

## Backup and Disaster Recovery

### Database Backup

```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_DIR="/backups/neo4j"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ehs_backup_$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Neo4j backup
neo4j-admin database backup \
  --to-path=$BACKUP_DIR \
  --database=ehs_production \
  --backup-name=$BACKUP_NAME \
  --compress

# Upload to S3
aws s3 cp $BACKUP_DIR/$BACKUP_NAME.tar.gz s3://ehs-backups/database/

# Cleanup old backups (keep 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_NAME"
```

### Application Backup

```bash
#!/bin/bash
# scripts/backup_application.sh

BACKUP_DIR="/backups/application"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /app/config/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /app/logs/

# Upload to S3
aws s3 sync $BACKUP_DIR s3://ehs-backups/application/

echo "Application backup completed"
```

### Disaster Recovery Plan

1. **Recovery Time Objective (RTO)**: 4 hours
2. **Recovery Point Objective (RPO)**: 1 hour

#### Recovery Steps:

1. **Assessment** (30 minutes)
   - Identify failed components
   - Assess data integrity
   - Determine recovery scope

2. **Infrastructure Recovery** (2 hours)
   - Provision new servers if needed
   - Restore network connectivity
   - Configure load balancers

3. **Database Recovery** (1 hour)
   - Restore Neo4j from backup
   - Verify data integrity
   - Update connection strings

4. **Application Recovery** (30 minutes)
   - Deploy application containers
   - Restore configuration
   - Perform health checks

5. **Verification** (30 minutes)
   - Run integration tests
   - Verify API endpoints
   - Monitor system health

### Automated Backup Cron Jobs

```bash
# /etc/cron.d/ehs-analytics-backup
# Database backup every 6 hours
0 */6 * * * root /opt/ehs-analytics/scripts/backup_database.sh

# Application backup daily at 2 AM
0 2 * * * root /opt/ehs-analytics/scripts/backup_application.sh

# Log rotation daily at 3 AM
0 3 * * * root /usr/sbin/logrotate /opt/ehs-analytics/config/logrotate.conf
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Failures

**Symptoms:**
- API returns 503 Service Unavailable
- Health check shows database as unhealthy
- Connection timeout errors in logs

**Diagnosis:**
```bash
# Check Neo4j status
systemctl status neo4j

# Test connection
cypher-shell -u neo4j -p $NEO4J_PASSWORD "RETURN 1;"

# Check logs
tail -f /var/log/neo4j/neo4j.log
```

**Solutions:**
- Restart Neo4j service
- Check firewall rules
- Verify credentials
- Increase connection pool size

#### 2. High Memory Usage

**Symptoms:**
- Application becomes slow
- Out of memory errors
- Container restarts

**Diagnosis:**
```bash
# Check memory usage
free -h
docker stats

# Check application metrics
curl http://localhost:8000/metrics | grep memory
```

**Solutions:**
- Increase container memory limits
- Tune garbage collection
- Implement query result caching
- Scale horizontally

#### 3. Slow Query Performance

**Symptoms:**
- High response times
- Query timeouts
- User complaints

**Diagnosis:**
```bash
# Check slow queries in Neo4j
cypher-shell -u neo4j -p $NEO4J_PASSWORD "SHOW FUNCTIONS YIELD name WHERE name CONTAINS 'slow';"

# Check API metrics
curl http://localhost:8000/metrics | grep http_request_duration
```

**Solutions:**
- Add database indexes
- Optimize Cypher queries
- Implement query caching
- Scale database cluster

### Log Analysis

```bash
# Search for errors
grep "ERROR" /var/log/ehs_analytics/app.log | tail -20

# Monitor real-time logs
tail -f /var/log/ehs_analytics/app.log | grep -E "(ERROR|WARN)"

# Analyze request patterns
awk '{print $7}' /var/log/nginx/ehs-analytics-access.log | sort | uniq -c | sort -nr
```

### Performance Tuning

#### Application Tuning

```python
# config/performance.py
UVICORN_CONFIG = {
    "workers": 4,
    "worker_class": "uvicorn.workers.UvicornWorker",
    "worker_connections": 1000,
    "max_requests": 1000,
    "max_requests_jitter": 50,
    "preload_app": True,
    "timeout": 300,
    "keepalive": 2,
}
```

#### Database Tuning

```cypher
// Neo4j performance tuning
// Increase page cache
CALL dbms.setConfigValue('dbms.memory.pagecache.size', '4G');

// Increase heap size
CALL dbms.setConfigValue('dbms.memory.heap.max_size', '4G');

// Query optimization
PROFILE MATCH (f:Facility)-[:HAS]->(e:Equipment) 
WHERE f.name = 'Apex Manufacturing' 
RETURN e;
```

### Emergency Response

#### Incident Response Plan

1. **Severity Levels:**
   - **Critical**: Complete service outage
   - **High**: Major feature unavailable
   - **Medium**: Performance degradation
   - **Low**: Minor issues

2. **Response Times:**
   - **Critical**: 15 minutes
   - **High**: 1 hour
   - **Medium**: 4 hours
   - **Low**: Next business day

3. **Escalation Path:**
   - L1: Operations team
   - L2: Development team
   - L3: Architecture team
   - L4: Management team

#### Emergency Contacts

```yaml
contacts:
  operations:
    primary: "ops-team@company.com"
    phone: "+1-555-0123"
  
  development:
    primary: "dev-team@company.com"
    phone: "+1-555-0124"
  
  management:
    primary: "management@company.com"
    phone: "+1-555-0125"
```

---

## Conclusion

This deployment guide provides comprehensive instructions for deploying the EHS Analytics platform in production environments. Follow these guidelines to ensure a secure, scalable, and maintainable deployment.

For additional support:
- **Documentation**: Review API documentation and implementation guides
- **Monitoring**: Use provided monitoring and alerting configurations
- **Support**: Contact the development team for technical issues

**Last Updated**: 2024-08-20  
**Version**: 1.0.0