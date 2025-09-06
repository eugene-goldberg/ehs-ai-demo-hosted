#!/usr/bin/env python3
"""
Script to update Algonquin electricity consumption data in Neo4j
Updates consumption values and proportionally adjusts costs to achieve HIGH risk rating
"""

import os
from neo4j import GraphDatabase
from datetime import datetime
import sys

class Neo4jUpdater:
    def __init__(self):
        # Neo4j connection details
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
        self.driver = None
        
        # Data to update: month_name -> {consumption_kwh: value}
        self.updates = {
            "March": 4416,    # March 2025: 4,416 kWh (from 31,469)
            "April": 3975,    # April 2025: 3,975 kWh (from 45,839)
            "May": 3754,      # May 2025: 3,754 kWh (from 54,378)
            "June": 3975,     # June 2025: 3,975 kWh (from 69,555)
            "July": 4200,     # July 2025: 4,200 kWh (from 72,429)
            "August": 4500,   # August 2025: 4,500 kWh (from 93,527)
        }
        
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test = result.single()
                if test:
                    print(f"✅ Connected to Neo4j at {self.uri}")
                    return True
                else:
                    print(f"❌ Failed to test Neo4j connection")
                    return False
        except Exception as e:
            print(f"❌ Error connecting to Neo4j: {e}")
            return False
            
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed")
            
    def find_electricity_consumption_nodes(self):
        """Find ElectricityConsumption nodes for algonquin_il site"""
        try:
            with self.driver.session() as session:
                print("\n🔍 Searching for ElectricityConsumption nodes for algonquin_il...")
                
                query = "MATCH (n:ElectricityConsumption {site_id: 'algonquin_il'}) RETURN n ORDER BY n.date"
                result = session.run(query)
                nodes = [dict(record['n']) for record in result]
                
                if nodes:
                    print(f"\n📋 Found {len(nodes)} ElectricityConsumption nodes:")
                    for i, node in enumerate(nodes):
                        month = node.get('month', 'Unknown')
                        consumption = node.get('consumption_kwh', 'Unknown')
                        cost = node.get('cost_usd', 'Unknown')
                        print(f"  {i+1}. {month} 2025: {consumption} kWh, ${cost}")
                    return nodes
                else:
                    print("❌ No ElectricityConsumption nodes found")
                    return []
                    
        except Exception as e:
            print(f"❌ Error finding electricity consumption nodes: {e}")
            return []
            
    def update_consumption_data(self):
        """Update the electricity consumption data with new values"""
        try:
            with self.driver.session() as session:
                print("\n🔄 Starting data updates...")
                
                updated_count = 0
                
                for month_name, new_consumption in self.updates.items():
                    print(f"\n  Processing {month_name} (consumption: {new_consumption} kWh)...")
                    
                    # First, find the current record
                    find_query = """
                    MATCH (n:ElectricityConsumption {site_id: 'algonquin_il', month: $month_name, year: 2025})
                    RETURN n.consumption_kwh as old_consumption, n.cost_usd as old_cost, n
                    """
                    
                    result = session.run(find_query, month_name=month_name)
                    record = result.single()
                    
                    if record:
                        old_consumption = record['old_consumption']
                        old_cost = record['old_cost']
                        
                        print(f"    Found existing record: {old_consumption} kWh, ${old_cost}")
                        
                        # Calculate new cost proportionally (maintain same cost per kWh)
                        if old_consumption and old_consumption > 0 and old_cost:
                            cost_per_kwh = old_cost / old_consumption
                            new_cost = round(new_consumption * cost_per_kwh, 2)
                        else:
                            # Use a default cost per kWh if we can't calculate
                            new_cost = round(new_consumption * 0.12, 2)  # $0.12 per kWh default
                        
                        print(f"    Calculating new cost: ${old_cost} / {old_consumption} kWh = ${cost_per_kwh:.4f}/kWh")
                        print(f"    New cost: {new_consumption} kWh × ${cost_per_kwh:.4f}/kWh = ${new_cost}")
                        print(f"    Updating to: {new_consumption} kWh, ${new_cost}")
                        
                        # Update the record
                        update_query = """
                        MATCH (n:ElectricityConsumption {site_id: 'algonquin_il', month: $month_name, year: 2025})
                        SET n.consumption_kwh = $new_consumption, 
                            n.cost_usd = $new_cost,
                            n.updated_at = datetime(),
                            n.data_source = 'Updated for HIGH risk profile'
                        RETURN n.consumption_kwh as updated_consumption, n.cost_usd as updated_cost
                        """
                        
                        update_result = session.run(update_query, 
                                                   month_name=month_name, 
                                                   new_consumption=new_consumption,
                                                   new_cost=new_cost)
                        
                        update_record = update_result.single()
                        if update_record:
                            print(f"    ✅ Updated successfully: {update_record['updated_consumption']} kWh, ${update_record['updated_cost']}")
                            updated_count += 1
                        else:
                            print(f"    ❌ Update failed")
                    else:
                        print(f"    ❌ No record found for {month_name}")
                
                print(f"\n📈 Update Summary: {updated_count}/{len(self.updates)} records updated")
                return updated_count > 0
                
        except Exception as e:
            print(f"❌ Error updating consumption data: {e}")
            return False
            
    def verify_updates(self):
        """Verify that the updates were successful"""
        try:
            with self.driver.session() as session:
                print("\n✅ Verifying updates...")
                
                query = """
                MATCH (n:ElectricityConsumption {site_id: 'algonquin_il', year: 2025})
                WHERE n.month IN ['March', 'April', 'May', 'June', 'July', 'August']
                RETURN n.month, n.consumption_kwh, n.cost_usd
                ORDER BY 
                  CASE n.month
                    WHEN 'March' THEN 3
                    WHEN 'April' THEN 4
                    WHEN 'May' THEN 5
                    WHEN 'June' THEN 6
                    WHEN 'July' THEN 7
                    WHEN 'August' THEN 8
                  END
                """
                
                result = session.run(query)
                records = [(record['n.month'], record['n.consumption_kwh'], record['n.cost_usd']) 
                          for record in result]
                
                if records:
                    print("📊 Current values:")
                    print("   Month   | Consumption (kWh) | Cost ($)  | Status")
                    print("   --------|-------------------|-----------|-------")
                    total_reduction_kwh = 0
                    total_old_kwh = 0
                    
                    # Original values for comparison
                    original_values = {
                        'March': 31468.29,
                        'April': 45838.74,
                        'May': 51215.48,  # Using value from May data that was truncated
                        'June': 58339.62,  # Using value from June data that was truncated
                        'July': 68236.49,  # Using value from July data that was truncated
                        'August': 93527.34
                    }
                    
                    for month, consumption, cost in records:
                        expected_consumption = self.updates.get(month, "N/A")
                        original_consumption = original_values.get(month, 0)
                        status = "✅" if consumption == expected_consumption else "❌"
                        
                        if original_consumption > 0:
                            reduction = original_consumption - consumption
                            total_reduction_kwh += reduction
                            total_old_kwh += original_consumption
                            reduction_pct = (reduction / original_consumption) * 100
                            print(f"   {month:7} | {consumption:17.2f} | {cost:9.2f} | {status} (-{reduction:.0f} kWh, -{reduction_pct:.1f}%)")
                        else:
                            print(f"   {month:7} | {consumption:17.2f} | {cost:9.2f} | {status}")
                    
                    if total_old_kwh > 0:
                        total_reduction_pct = (total_reduction_kwh / total_old_kwh) * 100
                        print(f"\n📉 Total reduction: {total_reduction_kwh:.0f} kWh ({total_reduction_pct:.1f}%)")
                        print(f"🎯 This significant reduction should achieve HIGH risk status")
                    
                    return True
                else:
                    print("❌ No records found for verification")
                    return False
                    
        except Exception as e:
            print(f"❌ Error verifying updates: {e}")
            return False

def main():
    """Main function to execute the update process"""
    print("🔧 Neo4j Electricity Consumption Update Script")
    print("=" * 50)
    
    updater = Neo4jUpdater()
    
    try:
        # Connect to database
        if not updater.connect():
            print("❌ Failed to connect to database")
            return 1
            
        # Find existing nodes
        nodes = updater.find_electricity_consumption_nodes()
        if not nodes:
            print("❌ Could not find ElectricityConsumption nodes")
            return 1
            
        # Update data
        if not updater.update_consumption_data():
            print("❌ Updates failed")
            return 1
            
        # Verify updates
        if not updater.verify_updates():
            print("❌ Verification failed")
            return 1
            
        print("\n🎉 All updates completed successfully!")
        print("💡 The Algonquin site should now show HIGH risk due to significant consumption reductions")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        updater.close()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)