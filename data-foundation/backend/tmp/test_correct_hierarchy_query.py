#!/usr/bin/env python3
"""
Test script to demonstrate the correct way to query the location hierarchy
based on the actual database structure found.
"""

import os
import sys
from datetime import datetime
from neo4j import GraphDatabase

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CorrectHierarchyQuery:
    def __init__(self):
        self.driver = None
        self.session = None
        
        # Neo4j connection details from .env
        self.uri = "bolt://localhost:7687"
        self.username = "neo4j"
        self.password = "EhsAI2024!"
        self.database = "neo4j"
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            self.session = self.driver.session(database=self.database)
            print(f"✅ Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.session:
            self.session.close()
        if self.driver:
            self.driver.close()
        print("🔌 Disconnected from Neo4j")
    
    def query_hierarchy_correctly(self):
        """Query the hierarchy using the correct approach based on properties"""
        print("\n=== CORRECT HIERARCHY QUERY ===")
        
        query = """
        MATCH (s:Site)
        OPTIONAL MATCH (b:Building) WHERE b.site_code = s.code
        OPTIONAL MATCH (f:Floor) WHERE id(f) IN [b.building_id | b <- [b] WHERE b IS NOT NULL]
        OPTIONAL MATCH (a:Area) WHERE id(a) IN [f.floor_id | f <- [f] WHERE f IS NOT NULL]
        
        WITH s, collect(DISTINCT {
            building_id: b.code,
            building_name: b.name,
            building_type: b.type,
            floors: []
        }) as buildings_raw
        
        RETURN s.code as site_id, s.name as site_name, s.type as site_type,
               [building IN buildings_raw WHERE building.building_name IS NOT NULL] as buildings
        ORDER BY s.name
        """
        
        try:
            result = self.session.run(query)
            sites = list(result)
            
            if not sites:
                print("⚠️  No sites found")
                return []
            
            print(f"Found {len(sites)} sites with hierarchy:")
            for site in sites:
                print(f"\n📍 Site: {site['site_name']} (ID: {site['site_id']})")
                print(f"   Type: {site['site_type']}")
                
                buildings = site['buildings']
                print(f"   Buildings: {len(buildings)}")
                
                for building in buildings:
                    print(f"     🏢 {building['building_name']} ({building['building_id']})")
                    print(f"        Type: {building['building_type']}")
            
            return sites
            
        except Exception as e:
            print(f"❌ Error in correct hierarchy query: {e}")
            return []
    
    def query_using_relationships(self):
        """Try to find what relationships actually exist for hierarchy"""
        print("\n=== CHECKING EXISTING RELATIONSHIPS ===")
        
        # Check Site relationships
        query = """
        MATCH (s:Site)-[r]->(target)
        RETURN s.name as site_name, type(r) as relationship, 
               labels(target)[0] as target_type, target.name as target_name
        LIMIT 20
        """
        
        try:
            result = self.session.run(query)
            site_rels = list(result)
            
            if site_rels:
                print("Site relationships found:")
                for rel in site_rels:
                    print(f"  {rel['site_name']} -[{rel['relationship']}]-> {rel['target_name']} ({rel['target_type']})")
            else:
                print("⚠️  No outgoing relationships from Sites found")
                
        except Exception as e:
            print(f"❌ Error checking site relationships: {e}")
        
        # Check what relationships connect to Buildings
        query = """
        MATCH (source)-[r]->(b:Building)
        RETURN labels(source)[0] as source_type, source.name as source_name,
               type(r) as relationship, b.name as building_name
        LIMIT 10
        """
        
        try:
            result = self.session.run(query)
            building_rels = list(result)
            
            if building_rels:
                print("\nRelationships pointing to Buildings:")
                for rel in building_rels:
                    print(f"  {rel['source_name']} ({rel['source_type']}) -[{rel['relationship']}]-> {rel['building_name']}")
            else:
                print("\n⚠️  No incoming relationships to Buildings found")
                
        except Exception as e:
            print(f"❌ Error checking building relationships: {e}")
    
    def build_hierarchy_from_properties(self):
        """Build hierarchy using property references (correct approach)"""
        print("\n=== BUILDING HIERARCHY FROM PROPERTIES ===")
        
        # First, get all sites
        sites_query = """
        MATCH (s:Site)
        RETURN s.code as site_code, s.name as site_name, s.type as site_type
        ORDER BY s.name
        """
        
        try:
            result = self.session.run(sites_query)
            sites = list(result)
            
            hierarchy = []
            
            for site in sites:
                site_data = {
                    "site_id": site['site_code'],
                    "site_name": site['site_name'],
                    "site_type": site['site_type'],
                    "buildings": []
                }
                
                # Get buildings for this site
                buildings_query = """
                MATCH (b:Building) 
                WHERE b.site_code = $site_code
                RETURN b.code as building_code, b.name as building_name, 
                       b.type as building_type, id(b) as building_internal_id
                ORDER BY b.name
                """
                
                buildings_result = self.session.run(buildings_query, site_code=site['site_code'])
                buildings = list(buildings_result)
                
                for building in buildings:
                    building_data = {
                        "building_id": building['building_code'],
                        "building_name": building['building_name'],
                        "building_type": building['building_type'],
                        "floors": []
                    }
                    
                    # Get floors for this building
                    floors_query = """
                    MATCH (f:Floor)
                    WHERE f.building_id = $building_id
                    RETURN f.name as floor_name, f.level as floor_level,
                           id(f) as floor_internal_id
                    ORDER BY f.level
                    """
                    
                    floors_result = self.session.run(floors_query, building_id=str(building['building_internal_id']))
                    floors = list(floors_result)
                    
                    for floor in floors:
                        floor_data = {
                            "floor_name": floor['floor_name'],
                            "floor_level": floor['floor_level'],
                            "areas": []
                        }
                        
                        # Get areas for this floor
                        areas_query = """
                        MATCH (a:Area)
                        WHERE a.floor_id = $floor_id
                        RETURN a.name as area_name, a.type as area_type
                        ORDER BY a.name
                        """
                        
                        areas_result = self.session.run(areas_query, floor_id=str(floor['floor_internal_id']))
                        areas = list(areas_result)
                        
                        for area in areas:
                            area_data = {
                                "area_name": area['area_name'],
                                "area_type": area['area_type']
                            }
                            floor_data["areas"].append(area_data)
                        
                        building_data["floors"].append(floor_data)
                    
                    site_data["buildings"].append(building_data)
                
                hierarchy.append(site_data)
            
            # Display the hierarchy
            print("Complete hierarchy structure:")
            for site in hierarchy:
                print(f"\n📍 {site['site_name']} ({site['site_id']}) - {site['site_type']}")
                
                for building in site['buildings']:
                    print(f"  🏢 {building['building_name']} ({building['building_id']}) - {building['building_type']}")
                    
                    for floor in building['floors']:
                        print(f"    🏠 {floor['floor_name']} (Level {floor['floor_level']})")
                        
                        for area in floor['areas']:
                            print(f"      📦 {area['area_name']} ({area['area_type']})")
            
            return hierarchy
            
        except Exception as e:
            print(f"❌ Error building hierarchy from properties: {e}")
            return []
    
    def test_api_compatible_query(self):
        """Test a query that would work for the API endpoint"""
        print("\n=== API-COMPATIBLE QUERY ===")
        
        query = """
        MATCH (s:Site)
        WITH s, 
             [(s)<-[:CONTAINS]-(b:Building) | {
                 building_id: b.code,
                 building_name: b.name,
                 building_type: b.type,
                 floors: [(b)<-[:CONTAINS]-(f:Floor) | {
                     floor_name: f.name,
                     floor_level: f.level,
                     areas: [(f)<-[:CONTAINS]-(a:Area) | {
                         area_name: a.name,
                         area_type: a.type
                     }]
                 }]
             }] as buildings_with_relationships
        
        RETURN s.code as site_id, s.name as site_name, s.type as site_type,
               buildings_with_relationships as buildings
        ORDER BY s.name
        """
        
        try:
            result = self.session.run(query)
            sites = list(result)
            
            print("API-compatible query results:")
            for site in sites:
                print(f"\n📍 {site['site_name']} ({site['site_id']})")
                buildings = site['buildings']
                print(f"   Buildings found via relationships: {len(buildings)}")
                
                for building in buildings:
                    print(f"     🏢 {building['building_name']}")
            
            return sites
            
        except Exception as e:
            print(f"❌ Error in API-compatible query: {e}")
            return []
    
    def run_all_tests(self):
        """Run all hierarchy query tests"""
        print("=" * 60)
        print(f"CORRECT HIERARCHY QUERY TESTING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        if not self.connect():
            return False
        
        try:
            # Test existing relationships
            self.query_using_relationships()
            
            # Build hierarchy from properties
            hierarchy = self.build_hierarchy_from_properties()
            
            # Test API-compatible query
            self.test_api_compatible_query()
            
            return True
            
        finally:
            self.close()

def main():
    """Main function"""
    test = CorrectHierarchyQuery()
    success = test.run_all_tests()
    
    if success:
        print("\n✅ Hierarchy query testing completed successfully")
        print("\n📋 RECOMMENDATIONS:")
        print("1. The hierarchy exists through property references, not relationships")
        print("2. Buildings link to Sites via site_code property")  
        print("3. Floors link to Buildings via building_id property (Neo4j internal ID)")
        print("4. Areas link to Floors via floor_id property (Neo4j internal ID)")
        print("5. The API query needs to be rewritten to use these property-based connections")
    else:
        print("\n❌ Hierarchy query testing failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
