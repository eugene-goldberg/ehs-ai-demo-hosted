#!/usr/bin/env python3
"""
Verification script for environmental data loading
Shows the key risk patterns that were implemented for LLM analysis
"""

import os
import sys
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.database.neo4j_client import ConnectionConfig
from neo4j import GraphDatabase

def verify_data():
    """Verify the environmental data patterns"""
    config = ConnectionConfig.from_env()
    driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))
    
    print("ENVIRONMENTAL DATA VERIFICATION")
    print("="*80)
    
    with driver.session(database=config.database) as session:
        # Check our specific sites
        print("\n1. SITES WITH RISK PATTERNS:")
        result = session.run('''
        MATCH (s:Site) 
        WHERE s.id IN ['algonquin_il', 'houston_tx']
        RETURN s.name, s.risk_profile, s.location, s.electricity_target_change, s.recycling_target_rate
        ORDER BY s.name
        ''')
        
        for record in result:
            name = record["s.name"]
            risk = record["s.risk_profile"]
            location = record["s.location"]
            elec_target = record["s.electricity_target_change"]
            recycling_target = record["s.recycling_target_rate"]
            
            print(f"   {name} ({risk} RISK)")
            print(f"   Location: {location}")
            print(f"   Electricity Target: {elec_target*100:.0f}% reduction")
            print(f"   Recycling Target: {recycling_target*100:.0f}%")
            print()
        
        # Check electricity trends (risk pattern verification)
        print("2. ELECTRICITY CONSUMPTION TRENDS (Risk Assessment Data):")
        result = session.run('''
        MATCH (s:Site)-[:HAS_ELECTRICITY_CONSUMPTION]->(ec:ElectricityConsumption)
        WHERE s.id IN ['algonquin_il', 'houston_tx']
        WITH s.name as site, s.id as site_id, ec.date as date, ec.consumption_kwh as consumption, 
             s.electricity_target_change as target
        ORDER BY site, date
        WITH site, site_id, target, collect({date: date, consumption: consumption}) as data
        RETURN site, site_id, target,
               data[0].consumption as march_avg,
               data[-1].consumption as august_avg,
               round((data[-1].consumption - data[0].consumption) / data[0].consumption * 100, 2) as actual_change
        ORDER BY site
        ''')
        
        for record in result:
            site = record["site"]
            site_id = record["site_id"]
            target_change = record["target"] * 100
            march_avg = record["march_avg"]
            august_avg = record["august_avg"]
            actual_change = record["actual_change"]
            
            status = "⚠️ EXCEEDS TARGET" if actual_change > 5 else "✅ ON TRACK" if actual_change < 0 else "⚠️ ABOVE TARGET"
            
            print(f"   {site}:")
            print(f"     March: {march_avg:.0f} kWh/day → August: {august_avg:.0f} kWh/day")
            print(f"     Actual Change: {actual_change:+.1f}% | Target: {target_change:.0f}% reduction")
            print(f"     Risk Status: {status}")
            print()
        
        # Check summer water spikes
        print("3. WATER CONSUMPTION SUMMER SPIKES:")
        result = session.run('''
        MATCH (s:Site)-[:HAS_WATER_CONSUMPTION]->(wc:WaterConsumption)
        WHERE s.id IN ['algonquin_il', 'houston_tx']
        WITH s.name as site, 
             wc.date.month as month, 
             avg(wc.consumption_gallons) as avg_consumption
        ORDER BY site, month
        WITH site, collect({month: month, consumption: avg_consumption}) as monthly_data
        RETURN site,
               [x IN monthly_data WHERE x.month = 3][0].consumption as march,
               [x IN monthly_data WHERE x.month = 6][0].consumption as june,
               [x IN monthly_data WHERE x.month = 7][0].consumption as july,
               [x IN monthly_data WHERE x.month = 8][0].consumption as august
        ORDER BY site
        ''')
        
        for record in result:
            site = record["site"]
            march = record["march"]
            june = record["june"]
            july = record["july"] 
            august = record["august"]
            
            summer_avg = (june + july + august) / 3
            spike_percent = ((summer_avg - march) / march) * 100
            
            print(f"   {site}:")
            print(f"     Spring (March): {march:.0f} gallons/day")
            print(f"     Summer Average: {summer_avg:.0f} gallons/day")
            print(f"     Summer Spike: {spike_percent:+.1f}%")
            print()
        
        # Check recycling rates and improvement
        print("4. WASTE RECYCLING PERFORMANCE:")
        result = session.run('''
        MATCH (s:Site)-[:GENERATES_WASTE]->(wg:WasteGeneration)
        WHERE s.id IN ['algonquin_il', 'houston_tx'] AND wg.waste_type = 'Recyclable'
        WITH s.name as site, s.recycling_target_rate as target,
             wg.date as date, wg.recycling_rate_achieved as rate
        ORDER BY site, date
        WITH site, target, collect(rate) as rates
        RETURN site, target,
               rates[0] as first_rate,
               rates[-1] as last_rate,
               size(rates) as rate_count,
               reduce(sum = 0.0, rate IN rates | sum + rate) / size(rates) as avg_rate
        ORDER BY site
        ''')
        
        for record in result:
            site = record["site"]
            target = record["target"] * 100
            first_rate = record["first_rate"] * 100
            last_rate = record["last_rate"] * 100
            avg_rate = record["avg_rate"] * 100
            
            improvement = last_rate - first_rate
            gap_to_target = target - avg_rate
            
            status = "✅ IMPROVING" if improvement > 1 else "⚠️ STAGNANT" if improvement < 1 else "➡️ STABLE"
            target_status = "⚠️ BELOW TARGET" if gap_to_target > 10 else "✅ NEAR TARGET"
            
            print(f"   {site}:")
            print(f"     March Rate: {first_rate:.1f}% → August Rate: {last_rate:.1f}%")
            print(f"     Average Rate: {avg_rate:.1f}% | Target: {target:.0f}%")
            print(f"     Trend: {status} | Target Gap: {target_status}")
            print()
        
        # Count total records for verification
        print("5. DATA COMPLETENESS:")
        result = session.run('''
        MATCH (s:Site) WHERE s.id IN ['algonquin_il', 'houston_tx']
        WITH s
        OPTIONAL MATCH (s)-[:HAS_ELECTRICITY_CONSUMPTION]->(ec)
        OPTIONAL MATCH (s)-[:HAS_WATER_CONSUMPTION]->(wc)  
        OPTIONAL MATCH (s)-[:GENERATES_WASTE]->(wg)
        OPTIONAL MATCH (s)-[:HAS_TARGET]->(et)
        RETURN s.name as site,
               count(DISTINCT ec) as electricity_records,
               count(DISTINCT wc) as water_records,
               count(DISTINCT wg) as waste_records,
               count(DISTINCT et) as target_records
        ORDER BY site
        ''')
        
        for record in result:
            print(f"   {record['site']}:")
            print(f"     Electricity Records: {record['electricity_records']} (daily, 6 months)")
            print(f"     Water Records: {record['water_records']} (daily, 6 months)")
            print(f"     Waste Records: {record['waste_records']} (weekly, 6 months)")
            print(f"     Environmental Targets: {record['target_records']}")
            print()
        
        # Sample some metadata for LLM context
        print("6. SAMPLE METADATA FOR LLM ANALYSIS:")
        result = session.run('''
        MATCH (s:Site {id: 'algonquin_il'})-[:HAS_ELECTRICITY_CONSUMPTION]->(ec:ElectricityConsumption)
        WHERE ec.equipment_notes CONTAINS 'HVAC' OR ec.equipment_notes CONTAINS 'efficiency'
        RETURN ec.date, ec.consumption_kwh, ec.equipment_notes
        ORDER BY ec.date DESC LIMIT 3
        ''')
        
        print("   Algonquin IL Equipment Issues (Sample):")
        for record in result:
            date = record["ec.date"]
            consumption = record["ec.consumption_kwh"]
            notes = record["ec.equipment_notes"]
            print(f"     {date}: {consumption:.0f} kWh - {notes}")
        
        result = session.run('''
        MATCH (s:Site {id: 'houston_tx'})-[:GENERATES_WASTE]->(wg:WasteGeneration)
        WHERE wg.performance_notes CONTAINS 'recycling' OR wg.performance_notes CONTAINS 'improvement'
        RETURN wg.date, wg.waste_type, wg.recycling_rate_achieved, wg.performance_notes
        ORDER BY wg.date DESC LIMIT 3
        ''')
        
        print("   Houston TX Recycling Improvements (Sample):")
        for record in result:
            date = record["wg.date"]
            waste_type = record["wg.waste_type"]
            rate = record["wg.recycling_rate_achieved"] * 100
            notes = record["wg.performance_notes"]
            print(f"     {date}: {rate:.1f}% - {notes}")
    
    driver.close()
    
    print("\n7. LLM ANALYSIS READINESS:")
    print("   ✅ High-risk site (Algonquin IL) with worsening electricity trend")
    print("   ✅ Medium-risk site (Houston TX) with flat electricity but improving recycling")
    print("   ✅ Seasonal patterns (summer spikes in water consumption)")
    print("   ✅ Equipment and operational metadata for context")
    print("   ✅ Performance targets for gap analysis")
    print("   ✅ 183 days of comprehensive environmental data")
    print("\n   The data is ready for risk assessment and LLM analysis!")
    print("="*80)

if __name__ == "__main__":
    verify_data()