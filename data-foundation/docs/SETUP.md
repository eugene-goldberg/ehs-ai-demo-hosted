# EHS AI Data Foundation Setup Guide

## Virtual Environment Setup

This project uses `uv` for fast, reliable Python package management.

### Prerequisites

1. **uv** - Install from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
2. **Python 3.11** - Required for compatibility with all dependencies
3. **Neo4j** - See [neo4j-config.md](./neo4j-config.md) for database setup

### Quick Setup

```bash
# Run the setup script
./scripts/setup_venv.sh
```

This script will:
- Create a virtual environment with Python 3.11
- Install all required dependencies
- Test the Neo4j connection
- Create initial EHS schema indexes

### Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r backend/requirements.txt --index-strategy unsafe-best-match

# Test Neo4j connection
python scripts/test_neo4j_connection.py
```

### Environment Configuration

1. Copy the example environment file:
   ```bash
   cp backend/example.env backend/.env
   ```

2. Neo4j credentials are already configured if you followed [neo4j-config.md](./neo4j-config.md)

3. Add your LLM API keys as needed:
   ```env
   OPENAI_API_KEY="your-key-here"
   ANTHROPIC_API_KEY="your-key-here"
   # etc.
   ```

### Running the Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the backend
cd backend
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start the frontend
cd frontend
npm install  # First time only
npm run dev
```

### Troubleshooting

#### Neo4j Connection Issues
- Ensure Neo4j container is running: `docker ps | grep ehs-neo4j`
- Check credentials match in `.env` file
- Test connection: `python scripts/test_neo4j_connection.py`

#### Dependency Issues
- macOS ARM64: CPU-only torch versions aren't available; regular versions are used
- Clear cache if needed: `uv cache clean`
- Reinstall: `uv pip install -r backend/requirements.txt --reinstall`

#### Virtual Environment
- Always activate before running: `source .venv/bin/activate`
- Check activation: `which python` should show `.venv/bin/python`
- Deactivate when done: `deactivate`

### Development Notes

- Dependencies are managed via `requirements.txt` and `constraints.txt`
- A `pyproject.toml` is available for future migration to modern Python packaging
- All utility scripts should be placed in the `scripts/` directory