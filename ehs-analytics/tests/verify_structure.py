#!/usr/bin/env python3
"""
Project Structure Verification Test Script
Verifies that the EHS Analytics project structure was created correctly.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import tomllib
from datetime import datetime

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"structure_verification_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StructureVerifier:
    """Verifies project structure components."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def log_error(self, message: str):
        """Log an error and add to errors list."""
        logger.error(message)
        self.errors.append(message)
        
    def log_warning(self, message: str):
        """Log a warning and add to warnings list."""
        logger.warning(message)
        self.warnings.append(message)
        
    def log_success(self, message: str):
        """Log a success message."""
        logger.info(f"✅ {message}")
        
    def verify_directories(self) -> bool:
        """Verify all required directories exist."""
        logger.info("🔍 Verifying directory structure...")
        
        required_dirs = [
            "src",
            "src/ehs_analytics",
            "src/ehs_analytics/agents",
            "src/ehs_analytics/models", 
            "src/ehs_analytics/retrieval",
            "src/ehs_analytics/retrieval/strategies",
            "src/ehs_analytics/risk_assessment",
            "src/ehs_analytics/risk_assessment/algorithms",
            "src/ehs_analytics/recommendations",
            "src/ehs_analytics/workflows",
            "src/ehs_analytics/utils",
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "tests/fixtures",
            "config",
            "docs",
            "logs",
            "scripts"
        ]
        
        success = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.log_success(f"Directory exists: {dir_path}")
            else:
                self.log_error(f"Missing directory: {dir_path}")
                success = False
                
        return success
        
    def verify_init_files(self) -> bool:
        """Verify all required __init__.py files exist."""
        logger.info("🔍 Verifying __init__.py files...")
        
        required_init_files = [
            "src/__init__.py",
            "src/ehs_analytics/__init__.py",
            "src/ehs_analytics/agents/__init__.py",
            "src/ehs_analytics/models/__init__.py",
            "src/ehs_analytics/retrieval/__init__.py",
            "src/ehs_analytics/retrieval/strategies/__init__.py",
            "src/ehs_analytics/risk_assessment/__init__.py",
            "src/ehs_analytics/risk_assessment/algorithms/__init__.py",
            "src/ehs_analytics/recommendations/__init__.py",
            "src/ehs_analytics/workflows/__init__.py",
            "src/ehs_analytics/utils/__init__.py"
        ]
        
        success = True
        for init_file in required_init_files:
            full_path = self.project_root / init_file
            if full_path.exists() and full_path.is_file():
                self.log_success(f"Init file exists: {init_file}")
            else:
                self.log_error(f"Missing __init__.py file: {init_file}")
                success = False
                
        return success
        
    def verify_pyproject_toml(self) -> bool:
        """Verify pyproject.toml exists and contains required dependencies."""
        logger.info("🔍 Verifying pyproject.toml...")
        
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            self.log_error("pyproject.toml file missing")
            return False
            
        try:
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
                
            # Check for required dependencies
            dependencies = config.get("project", {}).get("dependencies", [])
            
            required_deps = [
                "neo4j-graphrag-python",
                "neo4j",
                "langchain",
                "langchain-openai", 
                "llama-index",
                "llama-parse",
                "langgraph",
                "pydantic",
                "fastapi",
                "pytest"
            ]
            
            success = True
            for dep in required_deps:
                found = any(dep in d for d in dependencies)
                if found:
                    self.log_success(f"Required dependency found: {dep}")
                else:
                    self.log_error(f"Missing required dependency: {dep}")
                    success = False
                    
            # Verify neo4j-graphrag-python specifically
            neo4j_graphrag_found = any("neo4j-graphrag-python" in d for d in dependencies)
            if neo4j_graphrag_found:
                self.log_success("neo4j-graphrag-python dependency verified")
            else:
                self.log_error("neo4j-graphrag-python dependency missing")
                success = False
                
            return success
            
        except Exception as e:
            self.log_error(f"Error reading pyproject.toml: {e}")
            return False
            
    def verify_core_files(self) -> bool:
        """Verify core project files exist."""
        logger.info("🔍 Verifying core files...")
        
        required_files = [
            "README.md",
            "pyproject.toml"
        ]
        
        success = True
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                self.log_success(f"Core file exists: {file_path}")
            else:
                self.log_error(f"Missing core file: {file_path}")
                success = False
                
        return success
        
    def verify_python_environment(self) -> bool:
        """Verify Python environment is set up correctly."""
        logger.info("🔍 Verifying Python environment...")
        
        try:
            # Check Python version
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            python_version = result.stdout.strip()
            self.log_success(f"Python version: {python_version}")
            
            # Check if we can import key dependencies (if installed)
            try:
                import pydantic
                self.log_success("Pydantic import successful")
            except ImportError:
                self.log_warning("Pydantic not installed (expected in fresh setup)")
                
            return True
            
        except Exception as e:
            self.log_error(f"Python environment check failed: {e}")
            return False
            
    def verify_package_structure(self) -> bool:
        """Verify the package can be imported properly."""
        logger.info("🔍 Verifying package structure...")
        
        # Add src to Python path for testing
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        try:
            # Try to import the main package
            import ehs_analytics
            self.log_success("Main package import successful")
            
            # Try to import subpackages
            subpackages = [
                "ehs_analytics.agents",
                "ehs_analytics.models",
                "ehs_analytics.retrieval",
                "ehs_analytics.risk_assessment", 
                "ehs_analytics.recommendations",
                "ehs_analytics.workflows",
                "ehs_analytics.utils"
            ]
            
            success = True
            for package in subpackages:
                try:
                    __import__(package)
                    self.log_success(f"Subpackage import successful: {package}")
                except ImportError as e:
                    self.log_error(f"Failed to import {package}: {e}")
                    success = False
                    
            return success
            
        except ImportError as e:
            self.log_error(f"Failed to import main package: {e}")
            return False
            
    def run_all_verifications(self) -> Dict[str, Any]:
        """Run all verification tests."""
        logger.info("🚀 Starting project structure verification...")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Log file: {log_file}")
        
        results = {
            "directories": self.verify_directories(),
            "init_files": self.verify_init_files(), 
            "pyproject_toml": self.verify_pyproject_toml(),
            "core_files": self.verify_core_files(),
            "python_env": self.verify_python_environment(),
            "package_structure": self.verify_package_structure()
        }
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        
        logger.info("\n" + "="*50)
        logger.info("📊 VERIFICATION SUMMARY")
        logger.info("="*50)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            logger.info(f"{test_name:20} {status}")
            
        logger.info("-"*50)
        logger.info(f"Overall: {passed}/{total} tests passed")
        
        if self.errors:
            logger.info(f"\n❌ {len(self.errors)} errors found:")
            for error in self.errors:
                logger.info(f"  • {error}")
                
        if self.warnings:
            logger.info(f"\n⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.info(f"  • {warning}")
                
        logger.info(f"\n📝 Full log saved to: {log_file}")
        
        return {
            "results": results,
            "passed": passed,
            "total": total, 
            "errors": self.errors,
            "warnings": self.warnings,
            "success": passed == total
        }

def main():
    """Main entry point."""
    verifier = StructureVerifier()
    results = verifier.run_all_verifications()
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)

if __name__ == "__main__":
    main()
