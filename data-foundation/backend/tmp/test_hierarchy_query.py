#!/usr/bin/env python3
"""
Test script to query Neo4j database and examine the location hierarchy
that was created by our migration.
"""

import os
import sys
from datetime import datetime
from neo4j import GraphDatabase

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HierarchyQueryTest:
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
    
    def query_all_sites(self):
        """Query all Site nodes"""
        print("\n=== SITE NODES ===")
        query = """
        MATCH (s:Site)
        RETURN s.name as site_name, s.site_id as site_id, 
               s.address as address, s.city as city, s.state as state
        ORDER BY s.name
        """
        
        try:
            result = self.session.run(query)
            sites = list(result)
            
            if not sites:
                print("⚠️  No Site nodes found")
                return []
            
            print(f"Found {len(sites)} Site nodes:")
            for site in sites:
                print(f"  - {site['site_name']} (ID: {site['site_id']})")
                print(f"    Address: {site['address']}, {site['city']}, {site['state']}")
            
            return sites
            
        except Exception as e:
            print(f"❌ Error querying sites: {e}")
            return []
    
    def query_all_buildings(self):
        """Query all Building nodes"""
        print("\n=== BUILDING NODES ===")
        query = """
        MATCH (b:Building)
        RETURN b.name as building_name, b.building_id as building_id,
               b.site_name as site_name
        ORDER BY b.site_name, b.name
        """
        
        try:
            result = self.session.run(query)
            buildings = list(result)
            
            if not buildings:
                print("⚠️  No Building nodes found")
                return []
            
            print(f"Found {len(buildings)} Building nodes:")
            current_site = None
            for building in buildings:
                if current_site != building['site_name']:
                    current_site = building['site_name']
                    print(f"  Site: {current_site}")
                print(f"    - {building['building_name']} (ID: {building['building_id']})")
            
            return buildings
            
        except Exception as e:
            print(f"❌ Error querying buildings: {e}")
            return []
    
    def query_all_floors(self):
        """Query all Floor nodes"""
        print("\n=== FLOOR NODES ===")
        query = """
        MATCH (f:Floor)
        RETURN f.name as floor_name, f.floor_id as floor_id,
               f.building_name as building_name, f.site_name as site_name
        ORDER BY f.site_name, f.building_name, f.name
        """
        
        try:
            result = self.session.run(query)
            floors = list(result)
            
            if not floors:
                print("⚠️  No Floor nodes found")
                return []
            
            print(f"Found {len(floors)} Floor nodes:")
            current_site = None
            current_building = None
            for floor in floors:
                if current_site != floor['site_name']:
                    current_site = floor['site_name']
                    print(f"  Site: {current_site}")
                if current_building != floor['building_name']:
                    current_building = floor['building_name']
                    print(f"    Building: {current_building}")
                print(f"      - {floor['floor_name']} (ID: {floor['floor_id']})")
            
            return floors
            
        except Exception as e:
            print(f"❌ Error querying floors: {e}")
            return []
    
    def query_all_areas(self):
        """Query all Area nodes"""
        print("\n=== AREA NODES ===")
        query = """
        MATCH (a:Area)
        RETURN a.name as area_name, a.area_id as area_id,
               a.floor_name as floor_name, a.building_name as building_name, 
               a.site_name as site_name
        ORDER BY a.site_name, a.building_name, a.floor_name, a.name
        """
        
        try:
            result = self.session.run(query)
            areas = list(result)
            
            if not areas:
                print("⚠️  No Area nodes found")
                return []
            
            print(f"Found {len(areas)} Area nodes:")
            current_site = None
            current_building = None
            current_floor = None
            for area in areas:
                if current_site != area['site_name']:
                    current_site = area['site_name']
                    print(f"  Site: {current_site}")
                if current_building != area['building_name']:
                    current_building = area['building_name']
                    print(f"    Building: {current_building}")
                if current_floor != area['floor_name']:
                    current_floor = area['floor_name']
                    print(f"      Floor: {current_floor}")
                print(f"        - {area['area_name']} (ID: {area['area_id']})")
            
            return areas
            
        except Exception as e:
            print(f"❌ Error querying areas: {e}")
            return []
    
    def query_hierarchy_relationships(self):
        """Query relationships in the hierarchy"""
        print("\n=== HIERARCHY RELATIONSHIPS ===")
        
        # Query Site -> Building relationships
        print("\n--- Site -> Building relationships ---")
        query = """
        MATCH (s:Site)-[r:HAS_BUILDING]->(b:Building)
        RETURN s.name as site_name, type(r) as relationship, b.name as building_name
        ORDER BY s.name, b.name
        """
        
        try:
            result = self.session.run(query)
            site_building_rels = list(result)
            if site_building_rels:
                print(f"Found {len(site_building_rels)} Site->Building relationships:")
                for rel in site_building_rels:
                    print(f"  {rel['site_name']} -[{rel['relationship']}]-> {rel['building_name']}")
            else:
                print("⚠️  No Site->Building relationships found")
        except Exception as e:
            print(f"❌ Error querying Site->Building relationships: {e}")
        
        # Query Building -> Floor relationships
        print("\n--- Building -> Floor relationships ---")
        query = """
        MATCH (b:Building)-[r:HAS_FLOOR]->(f:Floor)
        RETURN b.name as building_name, type(r) as relationship, f.name as floor_name
        ORDER BY b.name, f.name
        """
        
        try:
            result = self.session.run(query)
            building_floor_rels = list(result)
            if building_floor_rels:
                print(f"Found {len(building_floor_rels)} Building->Floor relationships:")
                for rel in building_floor_rels:
                    print(f"  {rel['building_name']} -[{rel['relationship']}]-> {rel['floor_name']}")
            else:
                print("⚠️  No Building->Floor relationships found")
        except Exception as e:
            print(f"❌ Error querying Building->Floor relationships: {e}")
        
        # Query Floor -> Area relationships
        print("\n--- Floor -> Area relationships ---")
        query = """
        MATCH (f:Floor)-[r:HAS_AREA]->(a:Area)
        RETURN f.name as floor_name, type(r) as relationship, a.name as area_name
        ORDER BY f.name, a.name
        """
        
        try:
            result = self.session.run(query)
            floor_area_rels = list(result)
            if floor_area_rels:
                print(f"Found {len(floor_area_rels)} Floor->Area relationships:")
                for rel in floor_area_rels:
                    print(f"  {rel['floor_name']} -[{rel['relationship']}]-> {rel['area_name']}")
            else:
                print("⚠️  No Floor->Area relationships found")
        except Exception as e:
            print(f"❌ Error querying Floor->Area relationships: {e}")
    
    def query_complete_hierarchy(self):
        """Query the complete hierarchy with all levels"""
        print("\n=== COMPLETE HIERARCHY STRUCTURE ===")
        query = """
        MATCH path = (s:Site)-[:HAS_BUILDING]->(b:Building)-[:HAS_FLOOR]->(f:Floor)-[:HAS_AREA]->(a:Area)
        RETURN s.name as site_name, b.name as building_name, 
               f.name as floor_name, a.name as area_name
        ORDER BY s.name, b.name, f.name, a.name
        """
        
        try:
            result = self.session.run(query)
            hierarchies = list(result)
            
            if not hierarchies:
                print("⚠️  No complete Site->Building->Floor->Area paths found")
                return
            
            print(f"Found {len(hierarchies)} complete hierarchy paths:")
            current_site = None
            current_building = None
            current_floor = None
            
            for hierarchy in hierarchies:
                if current_site != hierarchy['site_name']:
                    current_site = hierarchy['site_name']
                    print(f"\n📍 Site: {current_site}")
                if current_building != hierarchy['building_name']:
                    current_building = hierarchy['building_name']
                    print(f"  🏢 Building: {current_building}")
                if current_floor != hierarchy['floor_name']:
                    current_floor = hierarchy['floor_name']
                    print(f"    🏠 Floor: {current_floor}")
                print(f"      📦 Area: {hierarchy['area_name']}")
                
        except Exception as e:
            print(f"❌ Error querying complete hierarchy: {e}")
    
    def query_sample_data_for_api(self):
        """Query sample data that would be returned by the API"""
        print("\n=== SAMPLE API DATA ===")
        
        # Get a sample site with all its hierarchy
        query = """
        MATCH (s:Site)
        OPTIONAL MATCH (s)-[:HAS_BUILDING]->(b:Building)
        OPTIONAL MATCH (b)-[:HAS_FLOOR]->(f:Floor)
        OPTIONAL MATCH (f)-[:HAS_AREA]->(a:Area)
        RETURN s.site_id as site_id, s.name as site_name,
               collect(DISTINCT {
                   building_id: b.building_id,
                   building_name: b.name,
                   floors: []
               }) as buildings
        LIMIT 1
        """
        
        try:
            result = self.session.run(query)
            sample_data = list(result)
            
            if sample_data:
                site = sample_data[0]
                print(f"Sample data for site '{site['site_name']}' (ID: {site['site_id']}):")
                print(f"Buildings: {len([b for b in site['buildings'] if b['building_name']])}")
                
                for building in site['buildings']:
                    if building['building_name']:
                        print(f"  - {building['building_name']} (ID: {building['building_id']})")
            else:
                print("⚠️  No sample data found")
                
        except Exception as e:
            print(f"❌ Error querying sample API data: {e}")
    
    def run_all_queries(self):
        """Run all hierarchy queries"""
        print("=" * 60)
        print(f"LOCATION HIERARCHY ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        if not self.connect():
            return False
        
        try:
            # Query all node types
            sites = self.query_all_sites()
            buildings = self.query_all_buildings()
            floors = self.query_all_floors()
            areas = self.query_all_areas()
            
            # Query relationships
            self.query_hierarchy_relationships()
            
            # Query complete hierarchy
            self.query_complete_hierarchy()
            
            # Query sample API data
            self.query_sample_data_for_api()
            
            # Summary
            print("\n=== SUMMARY ===")
            print(f"Total Sites: {len(sites)}")
            print(f"Total Buildings: {len(buildings)}")
            print(f"Total Floors: {len(floors)}")
            print(f"Total Areas: {len(areas)}")
            
            return True
            
        finally:
            self.close()

def main():
    """Main function"""
    test = HierarchyQueryTest()
    success = test.run_all_queries()
    
    if success:
        print("\n✅ Hierarchy analysis completed successfully")
    else:
        print("\n❌ Hierarchy analysis failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
