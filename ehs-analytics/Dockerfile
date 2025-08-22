# EHS Analytics Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/input /app/forecast_cache

# Copy application source code
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY logging_config.py /app/

# Create a non-root user
RUN groupadd -r ehs && useradd -r -g ehs ehs && \
    chown -R ehs:ehs /app
USER ehs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "ehs_analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]