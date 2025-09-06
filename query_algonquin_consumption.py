#!/usr/bin/env python3
"""
Query Neo4j database to get 6 months of historical consumption data for the Algonquin site.
Looks for ElectricityConsumption, WaterConsumption, and WasteGeneration nodes.
"""
from neo4j import GraphDatabase
import json
from datetime import datetime, timedelta
import os

# Neo4j connection configuration
uri = 'bolt://localhost:7687'
username = 'neo4j'
password = 'EhsAI2024!'  # From the existing code

def format_consumption_data(records, data_type):
    """Format consumption records for better readability."""
    formatted_data = []
    for record in records:
        data_dict = dict(record)
        # Convert any datetime objects to ISO format
        for key, value in data_dict.items():
            if isinstance(value, datetime):
                data_dict[key] = value.isoformat()
        data_dict['data_type'] = data_type
        formatted_data.append(data_dict)
    return formatted_data

def query_algonquin_consumption():
    """Main function to query consumption data for Algonquin site."""
    print("=== Querying Neo4j for Algonquin Site Consumption Data ===")
    print(f"Connection: {uri}")
    print(f"Username: {username}")
    print(f"Looking for site_id: 'algonquin_il'")
    print("")
    
    # Calculate 6 months ago from today
    six_months_ago = datetime.now() - timedelta(days=180)
    print(f"Searching for data from: {six_months_ago.isoformat()}")
    print("")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        all_consumption_data = {
            'electricity_consumption': [],
            'water_consumption': [],
            'waste_generation': []
        }
        
        with driver.session() as session:
            
            # 1. Query ElectricityConsumption nodes
            print("=== 1. ELECTRICITY CONSUMPTION DATA ===")
            electricity_query = """
                MATCH (ec:ElectricityConsumption)
                WHERE ec.site_id = $site_id
                AND (ec.date IS NOT NULL OR ec.month IS NOT NULL OR ec.period_start IS NOT NULL)
                RETURN ec, 
                       ec.date as date,
                       ec.month as month,
                       ec.period_start as period_start,
                       ec.period_end as period_end,
                       ec.consumption as consumption_kwh,
                       ec.cost as cost,
                       ec.demand_kwh as demand_kwh,
                       ec.rate_schedule as rate_schedule,
                       properties(ec) as all_properties
                ORDER BY COALESCE(ec.date, ec.period_start, ec.month) DESC
            """
            
            electricity_result = session.run(electricity_query, site_id='algonquin_il')
            electricity_records = []
            
            for record in electricity_result:
                node_props = dict(record['ec'])
                all_props = record['all_properties']
                
                consumption_record = {
                    'node_id': record['ec'].id,
                    'site_id': node_props.get('site_id'),
                    'date': record['date'],
                    'month': record['month'],
                    'period_start': record['period_start'],
                    'period_end': record['period_end'],
                    'consumption_kwh': record['consumption_kwh'],
                    'cost': record['cost'],
                    'demand_kwh': record['demand_kwh'],
                    'rate_schedule': record['rate_schedule'],
                    'all_properties': all_props
                }
                electricity_records.append(consumption_record)
                print(f"  Found electricity record: {consumption_record}")
            
            all_consumption_data['electricity_consumption'] = format_consumption_data(electricity_records, 'electricity')
            print(f"  Total Electricity Records: {len(electricity_records)}")
            print("")
            
            # 2. Query WaterConsumption nodes
            print("=== 2. WATER CONSUMPTION DATA ===")
            water_query = """
                MATCH (wc:WaterConsumption)
                WHERE wc.site_id = $site_id
                AND (wc.date IS NOT NULL OR wc.month IS NOT NULL OR wc.period_start IS NOT NULL)
                RETURN wc,
                       wc.date as date,
                       wc.month as month,
                       wc.period_start as period_start,
                       wc.period_end as period_end,
                       wc.consumption as consumption_gallons,
                       wc.cost as cost,
                       wc.rate_per_gallon as rate_per_gallon,
                       properties(wc) as all_properties
                ORDER BY COALESCE(wc.date, wc.period_start, wc.month) DESC
            """
            
            water_result = session.run(water_query, site_id='algonquin_il')
            water_records = []
            
            for record in water_result:
                node_props = dict(record['wc'])
                all_props = record['all_properties']
                
                consumption_record = {
                    'node_id': record['wc'].id,
                    'site_id': node_props.get('site_id'),
                    'date': record['date'],
                    'month': record['month'],
                    'period_start': record['period_start'],
                    'period_end': record['period_end'],
                    'consumption_gallons': record['consumption_gallons'],
                    'cost': record['cost'],
                    'rate_per_gallon': record['rate_per_gallon'],
                    'all_properties': all_props
                }
                water_records.append(consumption_record)
                print(f"  Found water record: {consumption_record}")
            
            all_consumption_data['water_consumption'] = format_consumption_data(water_records, 'water')
            print(f"  Total Water Records: {len(water_records)}")
            print("")
            
            # 3. Query WasteGeneration nodes
            print("=== 3. WASTE GENERATION DATA ===")
            waste_query = """
                MATCH (wg:WasteGeneration)
                WHERE wg.site_id = $site_id
                AND (wg.date IS NOT NULL OR wg.month IS NOT NULL OR wg.generation_date IS NOT NULL)
                RETURN wg,
                       wg.date as date,
                       wg.month as month,
                       wg.generation_date as generation_date,
                       wg.disposal_date as disposal_date,
                       wg.quantity as waste_quantity,
                       wg.unit as waste_unit,
                       wg.waste_type as waste_type,
                       wg.disposal_method as disposal_method,
                       wg.cost as cost,
                       properties(wg) as all_properties
                ORDER BY COALESCE(wg.date, wg.generation_date, wg.month) DESC
            """
            
            waste_result = session.run(waste_query, site_id='algonquin_il')
            waste_records = []
            
            for record in waste_result:
                node_props = dict(record['wg'])
                all_props = record['all_properties']
                
                consumption_record = {
                    'node_id': record['wg'].id,
                    'site_id': node_props.get('site_id'),
                    'date': record['date'],
                    'month': record['month'],
                    'generation_date': record['generation_date'],
                    'disposal_date': record['disposal_date'],
                    'waste_quantity': record['waste_quantity'],
                    'waste_unit': record['waste_unit'],
                    'waste_type': record['waste_type'],
                    'disposal_method': record['disposal_method'],
                    'cost': record['cost'],
                    'all_properties': all_props
                }
                waste_records.append(consumption_record)
                print(f"  Found waste record: {consumption_record}")
            
            all_consumption_data['waste_generation'] = format_consumption_data(waste_records, 'waste')
            print(f"  Total Waste Records: {len(waste_records)}")
            print("")
            
            # 4. Alternative queries in case site_id format is different
            print("=== 4. ALTERNATIVE QUERIES (different site_id formats) ===")
            
            # Try different variations of site identifiers
            alternative_site_ids = [
                'algonquin',
                'Algonquin',
                'ALGONQUIN_IL',
                'algonquin-il',
                'Algonquin_IL',
                'AlgonquinIL'
            ]
            
            for alt_site_id in alternative_site_ids:
                print(f"  Trying site_id: '{alt_site_id}'")
                
                # Quick check for any consumption nodes with this site_id
                check_query = """
                    CALL {
                        MATCH (ec:ElectricityConsumption) WHERE ec.site_id = $site_id RETURN count(ec) as electricity_count
                        UNION
                        MATCH (wc:WaterConsumption) WHERE wc.site_id = $site_id RETURN count(wc) as water_count  
                        UNION
                        MATCH (wg:WasteGeneration) WHERE wg.site_id = $site_id RETURN count(wg) as waste_count
                    }
                    RETURN sum(electricity_count + water_count + waste_count) as total_records
                """
                
                check_result = session.run(check_query, site_id=alt_site_id)
                total_found = check_result.single()['total_records']
                
                if total_found > 0:
                    print(f"    Found {total_found} records with site_id '{alt_site_id}'")
                    # If we found data with alternative site_id, re-run the main queries
                    # (This would be implemented similar to above but with alt_site_id)
                else:
                    print(f"    No records found with site_id '{alt_site_id}'")
            
            print("")
            
            # 5. General exploration - show all available site_ids
            print("=== 5. AVAILABLE SITE IDs IN DATABASE ===")
            
            site_exploration_query = """
                CALL {
                    MATCH (ec:ElectricityConsumption) 
                    WHERE ec.site_id IS NOT NULL 
                    RETURN DISTINCT ec.site_id as site_id, 'ElectricityConsumption' as node_type
                    UNION
                    MATCH (wc:WaterConsumption) 
                    WHERE wc.site_id IS NOT NULL 
                    RETURN DISTINCT wc.site_id as site_id, 'WaterConsumption' as node_type
                    UNION  
                    MATCH (wg:WasteGeneration) 
                    WHERE wg.site_id IS NOT NULL 
                    RETURN DISTINCT wg.site_id as site_id, 'WasteGeneration' as node_type
                }
                RETURN site_id, collect(node_type) as node_types
                ORDER BY site_id
            """
            
            site_result = session.run(site_exploration_query)
            available_sites = []
            
            for record in site_result:
                site_info = {
                    'site_id': record['site_id'],
                    'node_types': record['node_types']
                }
                available_sites.append(site_info)
                print(f"  Site ID: '{record['site_id']}' has node types: {record['node_types']}")
            
            all_consumption_data['available_sites'] = available_sites
            print(f"  Total unique sites found: {len(available_sites)}")
            print("")
        
        driver.close()
        
        # Summary and output
        print("=== SUMMARY ===")
        total_electricity = len(all_consumption_data['electricity_consumption'])
        total_water = len(all_consumption_data['water_consumption'])
        total_waste = len(all_consumption_data['waste_generation'])
        total_records = total_electricity + total_water + total_waste
        
        print(f"Total records found for 'algonquin_il':")
        print(f"  - Electricity: {total_electricity}")
        print(f"  - Water: {total_water}")
        print(f"  - Waste: {total_waste}")
        print(f"  - TOTAL: {total_records}")
        print("")
        
        if total_records > 0:
            print("=== DETAILED DATA (JSON FORMAT) ===")
            print(json.dumps(all_consumption_data, indent=2, default=str))
        else:
            print("No consumption data found for site 'algonquin_il'.")
            print("Available sites are listed above. You may need to use a different site_id.")
        
        return all_consumption_data
        
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        print(f"URI: {uri}")
        print(f"Username: {username}")
        print("")
        print("Please ensure:")
        print("1. Neo4j is running on localhost:7687")
        print("2. Username and password are correct")
        print("3. Database contains the expected consumption data")
        return None

if __name__ == "__main__":
    result = query_algonquin_consumption()
    
    # Save results to file for later analysis
    if result:
        output_file = '/Users/eugene/dev/ai/agentos/ehs-ai-demo/algonquin_consumption_data.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"\nCould not save results to file: {e}")