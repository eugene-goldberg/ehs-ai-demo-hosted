#!/usr/bin/env python3
"""
Script to create all indexes (vector and fulltext) in Neo4j for EHS document search.
"""

import os
import sys
import logging
import subprocess
import json
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_script(script_path: str) -> Dict[str, Any]:
    """
    Run an index creation script and capture results.
    
    Args:
        script_path: Path to the script to run
        
    Returns:
        Dictionary with execution results
    """
    try:
        logger.info(f"Running script: {script_path}")
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        return {
            "script": script_path,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
        
    except Exception as e:
        logger.error(f"Failed to run script {script_path}: {str(e)}")
        return {
            "script": script_path,
            "success": False,
            "error": str(e),
            "return_code": -1
        }

def load_script_results(results_file: str) -> Dict[str, Any]:
    """
    Load results from a script's output file.
    
    Args:
        results_file: Path to the results JSON file
        
    Returns:
        Dictionary with results data
    """
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load results from {results_file}: {str(e)}")
    
    return {}

def main():
    """Main function to create all indexes."""
    logger.info("Starting comprehensive index creation process...")
    
    # Ensure output directory exists
    output_dir = "scripts/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Scripts to run in order
    scripts = [
        "create_vector_indexes.py",
        "create_fulltext_indexes.py"
    ]
    
    # Results tracking
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "scripts": {},
        "summary": {
            "total_scripts": len(scripts),
            "successful_scripts": 0,
            "failed_scripts": 0
        }
    }
    
    # Run each script
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            all_results["scripts"][script] = {
                "success": False,
                "error": "Script file not found"
            }
            all_results["summary"]["failed_scripts"] += 1
            continue
        
        # Execute script
        execution_result = run_script(script_path)
        all_results["scripts"][script] = execution_result
        
        if execution_result["success"]:
            all_results["summary"]["successful_scripts"] += 1
            logger.info(f"✅ Successfully completed: {script}")
        else:
            all_results["summary"]["failed_scripts"] += 1
            logger.error(f"❌ Failed to complete: {script}")
            logger.error(f"Error: {execution_result.get('stderr', 'Unknown error')}")
    
    # Load individual script results for detailed summary
    vector_results = load_script_results("scripts/output/vector_index_results.json")
    fulltext_results = load_script_results("scripts/output/fulltext_index_results.json")
    
    # Comprehensive summary
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE INDEX CREATION SUMMARY")
    logger.info("="*60)
    
    # Script execution summary
    logger.info(f"Scripts executed: {all_results['summary']['total_scripts']}")
    logger.info(f"Successful: {all_results['summary']['successful_scripts']}")
    logger.info(f"Failed: {all_results['summary']['failed_scripts']}")
    
    # Vector index summary
    if vector_results:
        vector_created = sum(vector_results.get("creation_results", {}).values())
        vector_online = sum(vector_results.get("verification_results", {}).values())
        logger.info(f"\nVector Indexes:")
        logger.info(f"  Created: {vector_created}")
        logger.info(f"  Online: {vector_online}")
        logger.info(f"  Sample data: {'✅' if vector_results.get('sample_data_created') else '❌'}")
    
    # Fulltext index summary
    if fulltext_results:
        fulltext_created = sum(fulltext_results.get("creation_results", {}).values())
        fulltext_online = sum(fulltext_results.get("verification_results", {}).values())
        logger.info(f"\nFulltext Indexes:")
        logger.info(f"  Created: {fulltext_created}")
        logger.info(f"  Online: {fulltext_online}")
        logger.info(f"  Sample data: {'✅' if fulltext_results.get('sample_data_created') else '❌'}")
    
    # Combine results
    all_results["vector_index_details"] = vector_results
    all_results["fulltext_index_details"] = fulltext_results
    
    # Overall status
    overall_success = all_results["summary"]["failed_scripts"] == 0
    logger.info(f"\nOverall Status: {'✅ SUCCESS' if overall_success else '❌ PARTIAL/FAILED'}")
    
    # Save comprehensive results
    results_file = "scripts/output/all_indexes_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nComprehensive results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main()
