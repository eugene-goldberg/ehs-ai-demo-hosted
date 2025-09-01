#!/usr/bin/env python3
"""
Comprehensive Implementation Verification Script
Checks actual implementation status of all EHS components.
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime, timedelta
import subprocess
from pathlib import Path
from neo4j import GraphDatabase
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verification_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImplementationVerifier:
    def __init__(self):
        self.base_path = Path("/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "neo4j_data": {},
            "api_endpoints": {},
            "codebase_analysis": {},
            "missing_components": [],
            "summary": {}
        }
        
        # Load environment variables
        self.load_env_vars()
        
    def load_env_vars(self):
        """Load environment variables from .env file"""
        env_file = self.base_path / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    def verify_neo4j_data(self):
        """Check Neo4j for actual environmental data"""
        logger.info("Verifying Neo4j data...")
        
        try:
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            username = os.getenv('NEO4J_USERNAME', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD', 'password')
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                # Check for environmental data nodes
                queries = {
                    "ElectricityConsumption": "MATCH (n:ElectricityConsumption) RETURN count(n) as count, min(n.date) as min_date, max(n.date) as max_date",
                    "WaterConsumption": "MATCH (n:WaterConsumption) RETURN count(n) as count, min(n.date) as min_date, max(n.date) as max_date",
                    "WasteGeneration": "MATCH (n:WasteGeneration) RETURN count(n) as count, min(n.date) as min_date, max(n.date) as max_date",
                    "CO2Emissions": "MATCH (n:CO2Emissions) RETURN count(n) as count, min(n.date) as min_date, max(n.date) as max_date",
                    "EHSGoals": "MATCH (n:EHSGoal) RETURN count(n) as count, collect(n.type) as goal_types",
                    "Departments": "MATCH (n:Department) RETURN count(n) as count, collect(n.name) as departments",
                    "AllNodes": "MATCH (n) RETURN DISTINCT labels(n) as node_types, count(n) as count"
                }
                
                for query_name, query in queries.items():
                    try:
                        result = session.run(query)
                        records = list(result)
                        self.results["neo4j_data"][query_name] = records[0].data() if records else None
                        logger.info(f"Neo4j {query_name}: {self.results['neo4j_data'][query_name]}")
                    except Exception as e:
                        self.results["neo4j_data"][query_name] = f"Error: {str(e)}"
                        logger.error(f"Error querying {query_name}: {e}")
            
            driver.close()
            
        except Exception as e:
            self.results["neo4j_data"]["connection_error"] = str(e)
            logger.error(f"Neo4j connection error: {e}")
    
    def verify_api_endpoints(self):
        """Test all environmental API endpoints"""
        logger.info("Verifying API endpoints...")
        
        base_url = "http://localhost:8000"
        endpoints = [
            "/health",
            "/environmental/electricity",
            "/environmental/water", 
            "/environmental/waste",
            "/environmental/co2",
            "/environmental/summary",
            "/dashboard/executive",
            "/dashboard/kpis",
            "/goals/annual",
            "/risk-assessment/environmental"
        ]
        
        for endpoint in endpoints:
            try:
                url = f"{base_url}{endpoint}"
                response = requests.get(url, timeout=10)
                
                self.results["api_endpoints"][endpoint] = {
                    "status_code": response.status_code,
                    "response_size": len(response.text),
                    "headers": dict(response.headers),
                    "response_sample": response.text[:500] if response.text else None
                }
                
                if response.status_code == 200:
                    try:
                        json_data = response.json()
                        self.results["api_endpoints"][endpoint]["json_keys"] = list(json_data.keys()) if isinstance(json_data, dict) else None
                        self.results["api_endpoints"][endpoint]["data_count"] = len(json_data) if isinstance(json_data, list) else None
                    except:
                        pass
                
                logger.info(f"API {endpoint}: {response.status_code}")
                
            except Exception as e:
                self.results["api_endpoints"][endpoint] = {
                    "error": str(e),
                    "status": "unreachable"
                }
                logger.error(f"Error testing {endpoint}: {e}")
    
    def analyze_codebase(self):
        """Analyze codebase for implementation details"""
        logger.info("Analyzing codebase...")
        
        # Files to check
        important_files = [
            "main.py",
            "routers/environmental.py",
            "routers/dashboard.py", 
            "routers/goals.py",
            "models/environmental.py",
            "services/neo4j_service.py",
            "services/risk_assessment.py",
            "config/database.py",
            "requirements.txt"
        ]
        
        for file_path in important_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    self.results["codebase_analysis"][file_path] = {
                        "exists": True,
                        "size": len(content),
                        "lines": len(content.split('\n')),
                        "contains_co2": "co2" in content.lower() or "carbon" in content.lower(),
                        "contains_risk": "risk" in content.lower(),
                        "contains_goals": "goal" in content.lower(),
                        "functions": self.extract_functions(content, file_path)
                    }
                except Exception as e:
                    self.results["codebase_analysis"][file_path] = {
                        "exists": True,
                        "error": str(e)
                    }
            else:
                self.results["codebase_analysis"][file_path] = {
                    "exists": False
                }
    
    def extract_functions(self, content, file_path):
        """Extract function names from Python files"""
        functions = []
        if file_path.endswith('.py'):
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith('def ') or line.strip().startswith('async def '):
                    func_name = line.strip().split('(')[0].replace('def ', '').replace('async def ', '')
                    functions.append(func_name)
        return functions
    
    def check_directory_structure(self):
        """Check project directory structure"""
        logger.info("Checking directory structure...")
        
        expected_dirs = [
            "routers",
            "models", 
            "services",
            "config",
            "tests",
            "tmp",
            "data"
        ]
        
        structure = {}
        for dir_name in expected_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                structure[dir_name] = {
                    "exists": True,
                    "files": [f.name for f in files if f.is_file()],
                    "subdirs": [f.name for f in files if f.is_dir()]
                }
            else:
                structure[dir_name] = {"exists": False}
        
        self.results["codebase_analysis"]["directory_structure"] = structure
    
    def check_for_missing_components(self):
        """Identify missing components based on verification results"""
        logger.info("Identifying missing components...")
        
        missing = []
        
        # Check Neo4j data
        neo4j_data = self.results["neo4j_data"]
        if "connection_error" in neo4j_data:
            missing.append("Neo4j database connection")
        else:
            for data_type in ["ElectricityConsumption", "WaterConsumption", "WasteGeneration"]:
                if data_type in neo4j_data:
                    count = neo4j_data[data_type].get("count", 0) if neo4j_data[data_type] else 0
                    if count == 0:
                        missing.append(f"Neo4j {data_type} data")
            
            # Check for CO2 data
            co2_count = neo4j_data.get("CO2Emissions", {}).get("count", 0) if neo4j_data.get("CO2Emissions") else 0
            if co2_count == 0:
                missing.append("CO2 emissions data in Neo4j")
            
            # Check for goals
            goals_count = neo4j_data.get("EHSGoals", {}).get("count", 0) if neo4j_data.get("EHSGoals") else 0
            if goals_count == 0:
                missing.append("EHS goals data in Neo4j")
        
        # Check API endpoints
        for endpoint, result in self.results["api_endpoints"].items():
            if result.get("status_code") != 200:
                missing.append(f"Working API endpoint: {endpoint}")
        
        # Check code files
        for file_path, analysis in self.results["codebase_analysis"].items():
            if not file_path == "directory_structure" and not analysis.get("exists", False):
                missing.append(f"Code file: {file_path}")
        
        # Check for specific functionality
        codebase = self.results["codebase_analysis"]
        
        # CO2 conversion logic
        co2_found = any(analysis.get("contains_co2", False) for analysis in codebase.values() if isinstance(analysis, dict))
        if not co2_found:
            missing.append("CO2 conversion logic in codebase")
        
        # Risk assessment
        risk_found = any(analysis.get("contains_risk", False) for analysis in codebase.values() if isinstance(analysis, dict))
        if not risk_found:
            missing.append("Risk assessment functionality")
        
        # Goals functionality
        goals_found = any(analysis.get("contains_goals", False) for analysis in codebase.values() if isinstance(analysis, dict))
        if not goals_found:
            missing.append("Goals management functionality")
        
        self.results["missing_components"] = missing
    
    def generate_summary(self):
        """Generate implementation status summary"""
        logger.info("Generating summary...")
        
        # Count implemented vs missing
        total_components = 20  # Estimated total components needed
        missing_count = len(self.results["missing_components"])
        implemented_count = total_components - missing_count
        
        # Neo4j status
        neo4j_status = "Connected" if "connection_error" not in self.results["neo4j_data"] else "Disconnected"
        
        # API status
        working_apis = sum(1 for result in self.results["api_endpoints"].values() if result.get("status_code") == 200)
        total_apis = len(self.results["api_endpoints"])
        
        # Data status
        data_types = ["ElectricityConsumption", "WaterConsumption", "WasteGeneration", "CO2Emissions"]
        data_counts = {}
        for data_type in data_types:
            data_info = self.results["neo4j_data"].get(data_type, {})
            data_counts[data_type] = data_info.get("count", 0) if data_info else 0
        
        self.results["summary"] = {
            "overall_completion": f"{implemented_count}/{total_components} components ({(implemented_count/total_components)*100:.1f}%)",
            "neo4j_status": neo4j_status,
            "api_status": f"{working_apis}/{total_apis} endpoints working",
            "data_status": data_counts,
            "critical_missing": [comp for comp in self.results["missing_components"] if any(keyword in comp.lower() for keyword in ["neo4j", "api", "data"])],
            "implementation_priority": self.get_implementation_priority()
        }
    
    def get_implementation_priority(self):
        """Determine implementation priority based on missing components"""
        missing = self.results["missing_components"]
        
        priority = []
        
        # Critical infrastructure
        if any("Neo4j" in comp for comp in missing):
            priority.append("HIGH: Fix Neo4j connection and data")
        
        # Core APIs
        api_missing = [comp for comp in missing if "API endpoint" in comp]
        if api_missing:
            priority.append(f"HIGH: Fix {len(api_missing)} API endpoints")
        
        # Data
        data_missing = [comp for comp in missing if "data" in comp.lower()]
        if data_missing:
            priority.append(f"MEDIUM: Add missing data types ({len(data_missing)})")
        
        # Features
        feature_missing = [comp for comp in missing if any(keyword in comp.lower() for keyword in ["co2", "risk", "goals"])]
        if feature_missing:
            priority.append(f"MEDIUM: Implement missing features ({len(feature_missing)})")
        
        return priority
    
    def save_detailed_report(self):
        """Save detailed verification report"""
        report_path = self.base_path / "tmp" / "implementation_verification_report.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Detailed report saved to: {report_path}")
            
            # Also create a human-readable summary
            summary_path = self.base_path / "tmp" / "implementation_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("EHS IMPLEMENTATION VERIFICATION SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Verification Date: {self.results['timestamp']}\n\n")
                
                f.write("OVERALL STATUS:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.results["summary"].items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                
                f.write("\n\nMISSING COMPONENTS:\n")
                f.write("-" * 20 + "\n")
                for component in self.results["missing_components"]:
                    f.write(f"• {component}\n")
                
                f.write("\n\nNEO4J DATA STATUS:\n")
                f.write("-" * 20 + "\n")
                for data_type, info in self.results["neo4j_data"].items():
                    f.write(f"{data_type}: {info}\n")
                
                f.write("\n\nAPI ENDPOINTS STATUS:\n")
                f.write("-" * 25 + "\n")
                for endpoint, info in self.results["api_endpoints"].items():
                    status = info.get("status_code", "ERROR")
                    f.write(f"{endpoint}: {status}\n")
            
            logger.info(f"Summary report saved to: {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving reports: {e}")
    
    def run_verification(self):
        """Run complete verification process"""
        logger.info("Starting comprehensive implementation verification...")
        
        try:
            self.verify_neo4j_data()
            self.verify_api_endpoints()
            self.analyze_codebase()
            self.check_directory_structure()
            self.check_for_missing_components()
            self.generate_summary()
            self.save_detailed_report()
            
            # Print summary to console
            print("\n" + "=" * 60)
            print("EHS IMPLEMENTATION VERIFICATION COMPLETE")
            print("=" * 60)
            print(f"Overall Status: {self.results['summary']['overall_completion']}")
            print(f"Neo4j: {self.results['summary']['neo4j_status']}")
            print(f"APIs: {self.results['summary']['api_status']}")
            print(f"Missing Components: {len(self.results['missing_components'])}")
            
            if self.results["missing_components"]:
                print(f"\nTop Missing Components:")
                for i, component in enumerate(self.results["missing_components"][:5], 1):
                    print(f"{i}. {component}")
            
            print(f"\nDetailed reports saved in:")
            print(f"- {self.base_path}/tmp/implementation_verification_report.json")
            print(f"- {self.base_path}/tmp/implementation_summary.txt")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    verifier = ImplementationVerifier()
    success = verifier.run_verification()
    
    if success:
        print("\nVerification completed successfully!")
        return 0
    else:
        print("\nVerification failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())