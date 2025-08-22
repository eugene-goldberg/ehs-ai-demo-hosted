# EHS AI Data Foundation

This is the data foundation component of the EHS AI Platform, formerly known as Neo4j LLM Knowledge Graph Builder.

## Documentation

All documentation has been organized in the `docs/` directory:

- [Main Documentation](docs/README.md) - Overview and features
- [Setup Guide](docs/SETUP.md) - Virtual environment and installation
- [Neo4j Configuration](docs/neo4j-config.md) - Database setup and connection
- [Backend Documentation](docs/backend-README.md) - Backend API details
- [Frontend Documentation](docs/frontend-README.md) - Frontend application guide
- [Scripts Documentation](docs/scripts-README.md) - Utility scripts reference
- [Experiments](docs/experiments-README.md) - Research and experimentation notes
- [Data Guide](docs/data-README.md) - Sample data information

## Quick Start

```bash
# Set up the environment
./scripts/setup_venv.sh

# Activate virtual environment
source .venv/bin/activate

# Start the application
cd backend && uvicorn src.main:app --reload
```

For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md).

## Project Structure

```
data-foundation/
├── backend/         # FastAPI backend application
├── frontend/        # React frontend application
├── scripts/         # Utility scripts
├── docs/           # All documentation
├── experiments/    # Research notebooks
├── data/          # Sample data
└── .venv/         # Python virtual environment
```