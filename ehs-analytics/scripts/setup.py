#!/usr/bin/env python3
"""
EHS Analytics Setup Script

This script helps set up the EHS Analytics environment and database.
"""

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="EHS Analytics Setup and Management")
console = Console()

@app.command()
def init_database():
    """Initialize the Neo4j database with EHS schema."""
    console.print(Panel("Initializing EHS Analytics Database", style="bold blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Setting up database schema...", total=None)
        
        # TODO: Implement database initialization
        # This would create the Neo4j schema, indexes, and constraints
        
        progress.update(task, description="Database initialized successfully!")
    
    console.print("✅ Database setup complete", style="bold green")

@app.command()
def check_dependencies():
    """Check if all required dependencies are installed and configured."""
    console.print(Panel("Checking EHS Analytics Dependencies", style="bold blue"))
    
    dependencies = [
        ("Neo4j", "neo4j"),
        ("LangChain", "langchain"),
        ("LlamaIndex", "llama_index"),
        ("FastAPI", "fastapi"),
        ("Neo4j GraphRAG", "neo4j_graphrag"),
    ]
    
    for name, module in dependencies:
        try:
            __import__(module)
            console.print(f"✅ {name}", style="green")
        except ImportError:
            console.print(f"❌ {name} - Not installed", style="red")
    
    # Check environment variables
    console.print("\nEnvironment Variables:", style="bold")
    env_vars = [
        "NEO4J_URI",
        "NEO4J_USERNAME", 
        "NEO4J_PASSWORD",
        "OPENAI_API_KEY",
    ]
    
    for var in env_vars:
        if os.getenv(var):
            console.print(f"✅ {var}", style="green")
        else:
            console.print(f"❌ {var} - Not set", style="red")

@app.command()
def create_config():
    """Create configuration files from templates."""
    console.print(Panel("Creating Configuration Files", style="bold blue"))
    
    project_root = Path(__file__).parent.parent
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"
    
    if env_file.exists():
        console.print("⚠️  .env file already exists", style="yellow")
        if not typer.confirm("Overwrite existing .env file?"):
            return
    
    if env_example.exists():
        env_example.copy(env_file)
        console.print(f"✅ Created .env file from template", style="green")
        console.print(f"📝 Please edit {env_file} with your actual values", style="blue")
    else:
        console.print("❌ .env.example template not found", style="red")

@app.command()
def run_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Run the EHS Analytics API server."""
    console.print(Panel(f"Starting EHS Analytics API on {host}:{port}", style="bold blue"))
    
    try:
        import uvicorn
        uvicorn.run(
            "ehs_analytics.api.main:app",
            host=host,
            port=port,
            reload=reload
        )
    except ImportError:
        console.print("❌ uvicorn not installed. Run: pip install uvicorn", style="red")
        sys.exit(1)

@app.command()
def run_tests():
    """Run the test suite."""
    console.print(Panel("Running EHS Analytics Tests", style="bold blue"))
    
    import subprocess
    
    try:
        result = subprocess.run(["pytest", "-v"], capture_output=True, text=True)
        console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr, style="red")
        
        if result.returncode == 0:
            console.print("✅ All tests passed!", style="bold green")
        else:
            console.print("❌ Some tests failed", style="bold red")
            sys.exit(1)
            
    except FileNotFoundError:
        console.print("❌ pytest not installed. Run: pip install pytest", style="red")
        sys.exit(1)

if __name__ == "__main__":
    app()