#!/usr/bin/env python3
"""
Check actual properties available in WaterConsumption nodes
"""

from context_retriever import ContextRetriever

def check_water_properties():
    """Check what properties are actually available in WaterConsumption nodes"""
    print("Checking actual WaterConsumption node properties")
    print("=" * 50)
    
    retriever = ContextRetriever()
    
    try:
        # Query to get actual properties of WaterConsumption nodes
        query = """
        MATCH (wc:WaterConsumption)
        WHERE wc.site_id = 'algonquin_il'
        RETURN properties(wc) as props
        LIMIT 5
        """
        
        records = retriever._execute_query(query)
        print(f"Found {len(records)} records")
        
        if records:
            for i, record in enumerate(records[:3]):
                if hasattr(record, 'data'):
                    props = record.data()['props']
                else:
                    props = record['props']
                    
                print(f"\nRecord {i+1} properties:")
                for key, value in props.items():
                    print(f"  {key}: {value} ({type(value).__name__})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        retriever.close()

if __name__ == "__main__":
    check_water_properties()
