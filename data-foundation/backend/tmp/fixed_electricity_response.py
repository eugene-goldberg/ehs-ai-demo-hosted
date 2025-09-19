def format_electricity_response(data: Dict, site: Optional[str], entities: Dict) -> str:
    """Format electricity consumption response"""
    if not data:
        return "No electricity consumption data available."
    
    # Handle string responses (e.g., formatted assessment data)
    if isinstance(data, str):
        return data
    
    # Handle non-dict responses
    if not isinstance(data, dict):
        return str(data)
    
    try:
        if isinstance(data, dict) and "algonquin_il" in data:
            # Multi-site data
            response_parts = ["Here's the electricity consumption data for both sites:"]
            
            for site_key, site_data in data.items():
                if site_data and "total_sum" in site_data:
                    site_name = "Algonquin, IL" if site_key == "algonquin_il" else "Houston, TX"
                    response_parts.append(f"\n{site_name}: {site_data['total_sum']:.2f} kWh total consumption")
                    response_parts.append(f"  Average: {site_data.get('average', 0):.2f} kWh")
                    response_parts.append(f"  Peak: {site_data.get('max_value', 0):.2f} kWh")
        else:
            # Single site data
            site_name = "Algonquin, IL" if site == "algonquin_il" else "Houston, TX" if site == "houston_tx" else "All sites"
            response_parts = [f"Electricity consumption data for {site_name}:"]
            response_parts.append(f"Total consumption: {data.get('total_sum', 0):.2f} kWh")
            response_parts.append(f"Average consumption: {data.get('average', 0):.2f} kWh")
            response_parts.append(f"Peak consumption: {data.get('max_value', 0):.2f} kWh")
            response_parts.append(f"Minimum consumption: {data.get('min_value', 0):.2f} kWh")
        
        return " ".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error formatting electricity response: {e}")
        return "Error formatting electricity consumption data."
