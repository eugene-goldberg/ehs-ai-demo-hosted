from src.database.neo4j_client import Neo4jClient, ConnectionConfig
import os
os.chdir("/home/azureuser/dev/ehs-ai-demo/data-foundation/neo4j-data")
config = ConnectionConfig.from_env()
client = Neo4jClient(config=config, enable_logging=True)
if client.connect():
    print("Connected to Neo4j successfully")
    # Test the current query with algonquin filter
    location_filter = "algonquin"
    result = client.execute_query("""
    MATCH (s:Site)-[:GENERATES_WASTE]->(w:WasteGeneration)
    WHERE ($location IS NULL OR s.id CONTAINS $location OR s.name CONTAINS $location)
    RETURN s.id as location, s.name as site_name, count(w) as waste_count
    """, parameters={"location": location_filter})
    print(f"Sites found with location filter {location_filter}:")
    for record in result:
        print(f"  {record[site_name]} (id: {record[location]}): {record[waste_count]} waste records")
    
    # Test with site_id filter  
    location_filter = "algonquin_il"
    result = client.execute_query("""
    MATCH (s:Site)-[:GENERATES_WASTE]->(w:WasteGeneration)
    WHERE ($location IS NULL OR s.id CONTAINS $location OR s.name CONTAINS $location OR s.site_id CONTAINS $location)
    RETURN s.id as location, s.site_id, s.name as site_name, count(w) as waste_count
    """, parameters={"location": location_filter})
    print(f"Sites found with location filter {location_filter} (including site_id):")
    for record in result:
        print(f"  {record[site_name]} (id: {record[location]}, site_id: {record[site_id]}): {record[waste_count]} waste records")
    
    client.close()
else:
    print("Failed to connect to Neo4j")
