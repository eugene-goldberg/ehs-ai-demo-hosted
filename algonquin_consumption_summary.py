#!/usr/bin/env python3
"""
Summary script to extract and analyze the Algonquin consumption data from Neo4j.
Creates a clean summary report of 6 months historical data.
"""
from neo4j import GraphDatabase
import json
from datetime import datetime, timedelta
from collections import defaultdict

# Neo4j connection configuration
uri = 'bolt://localhost:7687'
username = 'neo4j'
password = 'EhsAI2024!'

def create_algonquin_summary():
    """Create a clean summary of Algonquin consumption data."""
    print("=== ALGONQUIN SITE CONSUMPTION DATA SUMMARY ===")
    print(f"Generated on: {datetime.now().isoformat()}")
    print(f"Site ID: algonquin_il")
    print("")
    
    summary_data = {
        'site_id': 'algonquin_il',
        'generated_date': datetime.now().isoformat(),
        'electricity_consumption': [],
        'water_consumption_summary': {},
        'water_consumption_daily': [],
        'waste_generation': [],
        'trends_analysis': {}
    }
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            
            # 1. ELECTRICITY CONSUMPTION (Monthly Data)
            print("1. ELECTRICITY CONSUMPTION (Monthly)")
            print("-" * 40)
            
            electricity_query = """
                MATCH (ec:ElectricityConsumption)
                WHERE ec.site_id = 'algonquin_il'
                AND ec.consumption_kwh IS NOT NULL
                RETURN ec.month as month,
                       ec.year as year,
                       ec.consumption_kwh as consumption_kwh,
                       ec.cost_usd as cost_usd,
                       ec.date as date,
                       ec.data_source as data_source
                ORDER BY ec.year DESC, ec.date DESC
            """
            
            electricity_result = session.run(electricity_query)
            total_electricity_kwh = 0
            total_electricity_cost = 0
            
            for record in electricity_result:
                month = record['month']
                year = record['year']
                consumption = record['consumption_kwh']
                cost = record['cost_usd']
                date = record['date']
                data_source = record['data_source']
                
                electricity_record = {
                    'month': month,
                    'year': year,
                    'date': str(date),
                    'consumption_kwh': consumption,
                    'cost_usd': cost,
                    'data_source': data_source
                }
                
                summary_data['electricity_consumption'].append(electricity_record)
                
                if consumption:
                    total_electricity_kwh += consumption
                if cost:
                    total_electricity_cost += cost
                
                print(f"  {month} {year}: {consumption:,.2f} kWh, ${cost:,.2f}")
            
            print(f"  TOTAL: {total_electricity_kwh:,.2f} kWh, ${total_electricity_cost:,.2f}")
            print("")
            
            # 2. WATER CONSUMPTION SUMMARY
            print("2. WATER CONSUMPTION SUMMARY")
            print("-" * 40)
            
            water_summary_query = """
                MATCH (wc:WaterConsumption)
                WHERE wc.site_id = 'algonquin_il'
                AND wc.consumption_gallons IS NOT NULL
                RETURN count(wc) as total_days,
                       sum(wc.consumption_gallons) as total_gallons,
                       sum(wc.cost_usd) as total_cost,
                       avg(wc.consumption_gallons) as avg_daily_gallons,
                       min(wc.date) as earliest_date,
                       max(wc.date) as latest_date,
                       sum(wc.process_usage_gallons) as total_process,
                       sum(wc.cooling_usage_gallons) as total_cooling,
                       sum(wc.domestic_usage_gallons) as total_domestic
            """
            
            water_summary_result = session.run(water_summary_query)
            water_summary_record = water_summary_result.single()
            
            if water_summary_record:
                total_days = water_summary_record['total_days']
                total_gallons = water_summary_record['total_gallons']
                total_cost = water_summary_record['total_cost']
                avg_daily = water_summary_record['avg_daily_gallons']
                earliest = water_summary_record['earliest_date']
                latest = water_summary_record['latest_date']
                total_process = water_summary_record['total_process']
                total_cooling = water_summary_record['total_cooling']
                total_domestic = water_summary_record['total_domestic']
                
                water_summary = {
                    'total_days': total_days,
                    'date_range': f"{earliest} to {latest}",
                    'total_gallons': total_gallons,
                    'total_cost_usd': total_cost,
                    'average_daily_gallons': avg_daily,
                    'usage_breakdown': {
                        'process_usage_gallons': total_process,
                        'cooling_usage_gallons': total_cooling, 
                        'domestic_usage_gallons': total_domestic
                    }
                }
                
                summary_data['water_consumption_summary'] = water_summary
                
                print(f"  Data Period: {earliest} to {latest} ({total_days} days)")
                print(f"  Total Water Usage: {total_gallons:,.2f} gallons")
                print(f"  Total Cost: ${total_cost:,.2f}")
                print(f"  Average Daily Usage: {avg_daily:,.2f} gallons/day")
                print(f"  Usage Breakdown:")
                print(f"    - Process: {total_process:,.2f} gallons ({total_process/total_gallons*100:.1f}%)")
                print(f"    - Cooling: {total_cooling:,.2f} gallons ({total_cooling/total_gallons*100:.1f}%)")
                print(f"    - Domestic: {total_domestic:,.2f} gallons ({total_domestic/total_gallons*100:.1f}%)")
                print("")
            
            # 3. WATER SOURCE BREAKDOWN
            print("3. WATER SOURCE BREAKDOWN")
            print("-" * 40)
            
            water_source_query = """
                MATCH (wc:WaterConsumption)
                WHERE wc.site_id = 'algonquin_il'
                RETURN wc.source_type as source_type,
                       count(wc) as days_count,
                       sum(wc.consumption_gallons) as total_gallons,
                       avg(wc.consumption_gallons) as avg_daily_gallons
                ORDER BY total_gallons DESC
            """
            
            water_source_result = session.run(water_source_query)
            for record in water_source_result:
                source = record['source_type']
                days = record['days_count']
                gallons = record['total_gallons']
                avg_daily = record['avg_daily_gallons']
                
                print(f"  {source}: {gallons:,.2f} gallons over {days} days (avg {avg_daily:,.2f}/day)")
            
            print("")
            
            # 4. WASTE GENERATION DATA
            print("4. WASTE GENERATION DATA")
            print("-" * 40)
            
            waste_query = """
                MATCH (wg:WasteGeneration)
                WHERE wg.site_id = 'algonquin_il'
                RETURN count(wg) as record_count
            """
            
            waste_result = session.run(waste_query)
            waste_count = waste_result.single()['record_count']
            
            if waste_count > 0:
                # Get detailed waste data if available
                waste_detail_query = """
                    MATCH (wg:WasteGeneration)
                    WHERE wg.site_id = 'algonquin_il'
                    RETURN wg.date as date,
                           wg.quantity as quantity,
                           wg.unit as unit,
                           wg.waste_type as waste_type,
                           wg.disposal_method as disposal_method,
                           wg.cost as cost
                    ORDER BY wg.date DESC
                    LIMIT 10
                """
                
                waste_detail_result = session.run(waste_detail_query)
                for record in waste_detail_result:
                    waste_record = {
                        'date': str(record['date']),
                        'quantity': record['quantity'],
                        'unit': record['unit'],
                        'waste_type': record['waste_type'],
                        'disposal_method': record['disposal_method'],
                        'cost': record['cost']
                    }
                    summary_data['waste_generation'].append(waste_record)
                    print(f"  {record['date']}: {record['quantity']} {record['unit']} of {record['waste_type']}")
            else:
                print("  No waste generation records found")
            
            print("")
            
            # 5. MONTHLY TRENDS ANALYSIS
            print("5. MONTHLY TRENDS ANALYSIS")
            print("-" * 40)
            
            # Group water consumption by month for trends
            monthly_water_query = """
                MATCH (wc:WaterConsumption)
                WHERE wc.site_id = 'algonquin_il'
                WITH wc.date.year as year, wc.date.month as month,
                     sum(wc.consumption_gallons) as monthly_total,
                     avg(wc.consumption_gallons) as monthly_avg,
                     count(wc) as days_in_month
                ORDER BY year DESC, month DESC
                RETURN year, month, monthly_total, monthly_avg, days_in_month
            """
            
            monthly_water_result = session.run(monthly_water_query)
            monthly_trends = []
            
            print("  Water Consumption by Month:")
            for record in monthly_water_result:
                year = record['year']
                month = record['month']
                total = record['monthly_total']
                avg_daily = record['monthly_avg']
                days = record['days_in_month']
                
                month_name = datetime(year, month, 1).strftime('%B')
                trend_record = {
                    'year': year,
                    'month': month,
                    'month_name': month_name,
                    'total_gallons': total,
                    'average_daily_gallons': avg_daily,
                    'days_recorded': days
                }
                monthly_trends.append(trend_record)
                
                print(f"    {month_name} {year}: {total:,.2f} gallons ({days} days, avg {avg_daily:,.2f}/day)")
            
            summary_data['trends_analysis']['monthly_water_trends'] = monthly_trends
            
        driver.close()
        
        # 6. SAVE SUMMARY TO FILE
        output_file = '/Users/eugene/dev/ai/agentos/ehs-ai-demo/algonquin_summary_report.json'
        with open(output_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print("")
        print("=== SUMMARY STATISTICS ===")
        print(f"Electricity Records: {len(summary_data['electricity_consumption'])}")
        print(f"Water Summary Period: {summary_data['water_consumption_summary'].get('date_range', 'N/A')}")
        print(f"Waste Records: {len(summary_data['waste_generation'])}")
        print("")
        print(f"Summary report saved to: {output_file}")
        
        return summary_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    summary = create_algonquin_summary()
    
    if summary:
        print("\n=== DATA RETRIEVAL SUCCESSFUL ===")
        print("6 months of historical consumption data for Algonquin site has been successfully queried and summarized.")
    else:
        print("\n=== ERROR ===")
        print("Failed to retrieve data. Please check Neo4j connection and database contents.")