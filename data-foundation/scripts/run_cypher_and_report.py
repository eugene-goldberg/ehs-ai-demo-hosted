#!/usr/bin/env python3
"""
Run Cypher queries against Neo4j and generate a comprehensive report.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "EhsAI2024!"


class Neo4jQueryRunner:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        self.driver.close()
    
    def run_query(self, query, parameters=None):
        """Run a single query and return results."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            # Convert result to list of records
            records = []
            for record in result:
                # Convert Neo4j objects to serializable format
                record_dict = {}
                for key, value in dict(record).items():
                    if hasattr(value, '_properties'):  # Neo4j Node
                        record_dict[key] = {
                            'id': value._properties.get('id', ''),
                            'labels': list(value.labels),
                            'properties': dict(value._properties),
                            'element_id': value.element_id
                        }
                    elif hasattr(value, 'type'):  # Neo4j Relationship
                        record_dict[key] = {
                            'type': value.type,
                            'properties': dict(value._properties) if hasattr(value, '_properties') else {},
                            'element_id': value.element_id
                        }
                    else:
                        record_dict[key] = value
                records.append(record_dict)
            return records
    
    def clear_test_data(self):
        """Clear any existing test data to ensure clean run."""
        queries = [
            "MATCH (n:Document) WHERE n.id STARTS WITH 'electric_bill_' DETACH DELETE n",
            "MATCH (n:UtilityBill) WHERE n.id STARTS WITH 'bill_electric_bill_' DETACH DELETE n",
            "MATCH (n:Emission) WHERE n.id STARTS WITH 'emission_electric_bill_' DETACH DELETE n",
            "MATCH (n:Facility) WHERE n.id = 'facility_apex_plant_a' DETACH DELETE n",
            "MATCH (n:Meter) WHERE n.id STARTS WITH 'MTR-' DETACH DELETE n"
        ]
        for query in queries:
            self.run_query(query)
        print("✓ Cleared existing test data")


def generate_report(queries_file, output_file):
    """Generate a comprehensive report of Cypher query execution."""
    
    # Load queries
    with open(queries_file, 'r') as f:
        queries = json.load(f)
    
    # Initialize Neo4j connection
    runner = Neo4jQueryRunner(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    # Start report
    report = {
        "report_metadata": {
            "title": "EHS AI Platform - Cypher Query Execution Report",
            "generated_at": datetime.now().isoformat(),
            "neo4j_connection": {
                "uri": NEO4J_URI,
                "username": NEO4J_USERNAME,
                "database": "neo4j"
            },
            "source_document": "electric_bill.pdf",
            "queries_file": queries_file
        },
        "execution_summary": {
            "total_queries": len(queries),
            "successful": 0,
            "failed": 0
        },
        "query_executions": []
    }
    
    try:
        # Clear existing test data
        runner.clear_test_data()
        
        # Execute each query
        print(f"\nExecuting {len(queries)} Cypher queries...")
        print("=" * 60)
        
        for i, query_info in enumerate(queries):
            query = query_info["query"]
            parameters = query_info["parameters"]
            
            execution_record = {
                "query_number": i + 1,
                "query": query,
                "parameters": parameters,
                "execution_time": datetime.now().isoformat()
            }
            
            try:
                # Execute query
                result = runner.run_query(query, parameters)
                
                execution_record["status"] = "SUCCESS"
                execution_record["result"] = result
                execution_record["records_affected"] = len(result)
                
                report["execution_summary"]["successful"] += 1
                
                print(f"✓ Query {i+1}: SUCCESS")
                if result:
                    first_result = result[0]
                    # Check if it's a node or relationship
                    if 'n' in first_result:
                        print(f"  Created: Node with ID: {first_result['n']['properties']['id']}")
                    elif 'r' in first_result:
                        print(f"  Created: Relationship of type: {first_result['r']['type']}")
                
            except Exception as e:
                execution_record["status"] = "FAILED"
                execution_record["error"] = str(e)
                
                report["execution_summary"]["failed"] += 1
                
                print(f"✗ Query {i+1}: FAILED - {str(e)}")
            
            report["query_executions"].append(execution_record)
        
        # Verify final graph state
        print("\n" + "=" * 60)
        print("VERIFYING FINAL GRAPH STATE")
        print("=" * 60)
        
        verification_queries = [
            ("Total Nodes", "MATCH (n) RETURN count(n) as count, labels(n) as labels"),
            ("Document Nodes", "MATCH (d:Document) RETURN d"),
            ("UtilityBill Nodes", "MATCH (u:UtilityBill) RETURN u"),
            ("Facility Nodes", "MATCH (f:Facility) RETURN f"),
            ("Emission Nodes", "MATCH (e:Emission) RETURN e"),
            ("Meter Nodes", "MATCH (m:Meter) RETURN m"),
            ("All Relationships", "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"),
            ("Full Graph Pattern", """
                MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)-[:BILLED_TO]->(f:Facility)
                OPTIONAL MATCH (b)-[:RESULTED_IN]->(e:Emission)
                OPTIONAL MATCH (m:Meter)-[:MONITORS]->(f)
                RETURN d, b, f, e, m
            """)
        ]
        
        report["graph_verification"] = {}
        
        for label, query in verification_queries:
            try:
                result = runner.run_query(query)
                report["graph_verification"][label] = {
                    "query": query,
                    "result": result,
                    "record_count": len(result)
                }
                print(f"\n{label}:")
                for record in result[:3]:  # Show first 3 records
                    print(f"  {record}")
                if len(result) > 3:
                    print(f"  ... and {len(result) - 3} more")
            except Exception as e:
                report["graph_verification"][label] = {
                    "query": query,
                    "error": str(e)
                }
                print(f"\n{label}: ERROR - {str(e)}")
        
    finally:
        runner.close()
    
    # Write report to file
    report_path = Path(output_file)
    
    # Write JSON version
    with open(report_path.with_suffix('.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Write human-readable version
    with open(report_path, 'w') as f:
        f.write("EHS AI PLATFORM - CYPHER QUERY EXECUTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {report['report_metadata']['generated_at']}\n")
        f.write(f"Neo4j URI: {report['report_metadata']['neo4j_connection']['uri']}\n")
        f.write(f"Source Document: {report['report_metadata']['source_document']}\n\n")
        
        f.write("EXECUTION SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Queries: {report['execution_summary']['total_queries']}\n")
        f.write(f"Successful: {report['execution_summary']['successful']}\n")
        f.write(f"Failed: {report['execution_summary']['failed']}\n\n")
        
        f.write("QUERY EXECUTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        for exec_record in report['query_executions']:
            f.write(f"Query #{exec_record['query_number']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Status: {exec_record['status']}\n")
            f.write(f"Query: {exec_record['query']}\n")
            f.write(f"Parameters: {json.dumps(exec_record['parameters'], indent=2)}\n")
            
            if exec_record['status'] == 'SUCCESS':
                f.write(f"Records Affected: {exec_record['records_affected']}\n")
                if exec_record['result']:
                    f.write(f"Result: {json.dumps(exec_record['result'], indent=2)}\n")
            else:
                f.write(f"Error: {exec_record.get('error', 'Unknown error')}\n")
            
            f.write("\n")
        
        f.write("\nGRAPH VERIFICATION\n")
        f.write("=" * 80 + "\n\n")
        
        for label, verification in report['graph_verification'].items():
            f.write(f"{label}:\n")
            f.write(f"Query: {verification['query']}\n")
            if 'error' in verification:
                f.write(f"Error: {verification['error']}\n")
            else:
                f.write(f"Record Count: {verification['record_count']}\n")
                if verification['result']:
                    f.write("Results:\n")
                    for record in verification['result'][:5]:
                        f.write(f"  {json.dumps(record, indent=2)}\n")
                    if len(verification['result']) > 5:
                        f.write(f"  ... and {len(verification['result']) - 5} more records\n")
            f.write("\n")
    
    print(f"\n✓ Report generated:")
    print(f"  - JSON: {report_path.with_suffix('.json')}")
    print(f"  - Text: {report_path}")
    
    return report


def main():
    queries_file = "cypher_queries.json"
    output_file = "cypher_execution_report.txt"
    
    print("EHS AI Platform - Cypher Query Runner")
    print("=" * 60)
    
    try:
        report = generate_report(queries_file, output_file)
        
        print("\n" + "=" * 60)
        print("EXECUTION COMPLETE")
        print("=" * 60)
        print(f"✓ Successfully executed {report['execution_summary']['successful']} queries")
        if report['execution_summary']['failed'] > 0:
            print(f"✗ Failed queries: {report['execution_summary']['failed']}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()