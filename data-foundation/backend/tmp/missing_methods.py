    def get_electricity_context(self, site: Optional[str], 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
        """Fetch electricity consumption data from Neo4j"""
        
        # Map site names to Neo4j site IDs
        site_mapping = {
            'houston_texas': 'houston_texas',
            'houston_tx': 'houston_texas',
            'houston': 'houston_texas',
            'algonquin_illinois': 'algonquin_il',
            'algonquin_il': 'algonquin_il',
            'algonquin': 'algonquin_il'
        }
        
        if site and site.lower() in site_mapping:
            site = site_mapping[site.lower()]
        
        # Build query based on available parameters
        if start_date and end_date:
            query = """
            MATCH (s:Site)-[:HAS_ELECTRICITY_CONSUMPTION]->(e:ElectricityConsumption)
            WHERE s.id = $site AND e.date >= $start_date AND e.date <= $end_date
            RETURN s.id as site_id, s.name as site_name, e.date as date,
                   e.consumption_kwh as consumption, e.cost_usd as cost,
                   e.co2_emissions as co2_emissions, e.peak_demand_kw as peak_demand
            ORDER BY e.date DESC
            LIMIT 100
            """
            params = {"site": site, "start_date": start_date, "end_date": end_date}
        elif start_date:
            query = """
            MATCH (s:Site)-[:HAS_ELECTRICITY_CONSUMPTION]->(e:ElectricityConsumption)
            WHERE s.id = $site AND e.date >= $start_date
            RETURN s.id as site_id, s.name as site_name, e.date as date,
                   e.consumption_kwh as consumption, e.cost_usd as cost,
                   e.co2_emissions as co2_emissions, e.peak_demand_kw as peak_demand
            ORDER BY e.date DESC
            LIMIT 100
            """
            params = {"site": site, "start_date": start_date}
        else:
            query = """
            MATCH (s:Site)-[:HAS_ELECTRICITY_CONSUMPTION]->(e:ElectricityConsumption)
            WHERE s.id = $site
            RETURN s.id as site_id, s.name as site_name, e.date as date,
                   e.consumption_kwh as consumption, e.cost_usd as cost,
                   e.co2_emissions as co2_emissions, e.peak_demand_kw as peak_demand
            ORDER BY e.date DESC
            LIMIT 100
            """
            params = {"site": site}
        
        try:
            records = self._execute_query(query, params)
            
            if not records:
                return {
                    "site": site,
                    "period": {"start": start_date, "end": end_date},
                    "message": "No electricity consumption data found",
                    "record_count": 0
                }
            
            # Convert records to dictionaries if needed
            if hasattr(records[0], 'data'):
                # Neo4j Record objects
                record_data = [record.data() for record in records]
            else:
                # Already dictionaries from Neo4j client
                record_data = records
            
            # Calculate aggregates
            consumption_values = [r.get('consumption', 0) for r in record_data if r.get('consumption')]
            cost_values = [r.get('cost', 0) for r in record_data if r.get('cost')]
            co2_values = [r.get('co2_emissions', 0) for r in record_data if r.get('co2_emissions')]
            
            return {
                "site": record_data[0]['site_name'] if record_data else site,
                "site_id": record_data[0]['site_id'] if record_data else site,
                "period": {
                    "start": start_date or (min(r['date'] for r in record_data) if record_data else None),
                    "end": end_date or (max(r['date'] for r in record_data) if record_data else None)
                },
                "record_count": len(record_data),
                "aggregates": {
                    "total": sum(consumption_values),
                    "average": sum(consumption_values) / len(consumption_values) if consumption_values else 0,
                    "min": min(consumption_values) if consumption_values else 0,
                    "max": max(consumption_values) if consumption_values else 0,
                    "total_cost": sum(cost_values),
                    "total_co2": sum(co2_values),
                    "avg_cost_per_kwh": sum(cost_values) / sum(consumption_values) if consumption_values and sum(consumption_values) > 0 else 0
                },
                "recent_data": [
                    {
                        "date": r['date'],
                        "consumption": r['consumption'],
                        "cost": r['cost'],
                        "co2": r['co2_emissions'],
                        "peak_demand": r['peak_demand']
                    }
                    for r in record_data[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error fetching electricity context: {e}")
            return {
                "site": site,
                "period": {"start": start_date, "end": end_date},
                "error": str(e),
                "record_count": 0
            }

    def get_water_context(self, site: Optional[str], 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, Any]:
        """Fetch water consumption data from Neo4j"""
        
        # Map site names to Neo4j site IDs
        site_mapping = {
            'houston_texas': 'houston_texas',
            'houston_tx': 'houston_texas',
            'houston': 'houston_texas',
            'algonquin_illinois': 'algonquin_il',
            'algonquin_il': 'algonquin_il',
            'algonquin': 'algonquin_il'
        }
        
        if site and site.lower() in site_mapping:
            site = site_mapping[site.lower()]
        
        # Build query based on available parameters
        if start_date and end_date:
            query = """
            MATCH (wc:WaterConsumption)
            WHERE wc.site_id = $site AND wc.date >= $start_date AND wc.date <= $end_date
            RETURN wc.site_id as site_id, wc.date as date,
                   wc.consumption_gallons as consumption_gallons, 
                   wc.cost_usd as cost_usd,
                   wc.cost_per_gallon as cost_per_gallon,
                   wc.cooling_usage_gallons as cooling_usage,
                   wc.domestic_usage_gallons as domestic_usage,
                   wc.process_usage_gallons as process_usage,
                   wc.source_type as source_type,
                   wc.quality_rating as quality_rating,
                   wc.seasonal_notes as seasonal_notes
            ORDER BY wc.date DESC
            LIMIT 100
            """
            params = {"site": site, "start_date": start_date, "end_date": end_date}
        elif start_date:
            query = """
            MATCH (wc:WaterConsumption)
            WHERE wc.site_id = $site AND wc.date >= $start_date
            RETURN wc.site_id as site_id, wc.date as date,
                   wc.consumption_gallons as consumption_gallons, 
                   wc.cost_usd as cost_usd,
                   wc.cost_per_gallon as cost_per_gallon,
                   wc.cooling_usage_gallons as cooling_usage,
                   wc.domestic_usage_gallons as domestic_usage,
                   wc.process_usage_gallons as process_usage,
                   wc.source_type as source_type,
                   wc.quality_rating as quality_rating,
                   wc.seasonal_notes as seasonal_notes
            ORDER BY wc.date DESC
            LIMIT 100
            """
            params = {"site": site, "start_date": start_date}
        else:
            query = """
            MATCH (wc:WaterConsumption)
            WHERE wc.site_id = $site
            RETURN wc.site_id as site_id, wc.date as date,
                   wc.consumption_gallons as consumption_gallons, 
                   wc.cost_usd as cost_usd,
                   wc.cost_per_gallon as cost_per_gallon,
                   wc.cooling_usage_gallons as cooling_usage,
                   wc.domestic_usage_gallons as domestic_usage,
                   wc.process_usage_gallons as process_usage,
                   wc.source_type as source_type,
                   wc.quality_rating as quality_rating,
                   wc.seasonal_notes as seasonal_notes
            ORDER BY wc.date DESC
            LIMIT 100
            """
            params = {"site": site}
        
        try:
            records = self._execute_query(query, params)
            
            if not records:
                return {
                    "site": site,
                    "period": {"start": start_date, "end": end_date},
                    "message": "No water consumption data found",
                    "record_count": 0
                }
            
            # Convert records to dictionaries if needed
            if hasattr(records[0], 'data'):
                # Neo4j Record objects
                record_data = [record.data() for record in records]
            else:
                # Already dictionaries from Neo4j client
                record_data = records
            
            # Calculate aggregates
            consumption_values = [r.get('consumption_gallons', 0) for r in record_data if r.get('consumption_gallons')]
            cost_values = [r.get('cost_usd', 0) for r in record_data if r.get('cost_usd')]
            cooling_values = [r.get('cooling_usage', 0) for r in record_data if r.get('cooling_usage')]
            domestic_values = [r.get('domestic_usage', 0) for r in record_data if r.get('domestic_usage')]
            process_values = [r.get('process_usage', 0) for r in record_data if r.get('process_usage')]
            
            return {
                "site": site,
                "site_id": site,
                "period": {
                    "start": start_date or (min(r.get('date') for r in record_data if r.get('date')) if record_data else None),
                    "end": end_date or (max(r.get('date') for r in record_data if r.get('date')) if record_data else None)
                },
                "record_count": len(record_data),
                "aggregates": {
                    "total_gallons": sum(consumption_values),
                    "average_gallons": sum(consumption_values) / len(consumption_values) if consumption_values else 0,
                    "min_gallons": min(consumption_values) if consumption_values else 0,
                    "max_gallons": max(consumption_values) if consumption_values else 0,
                    "total_cost": sum(cost_values),
                    "avg_cost_per_gallon": sum(cost_values) / sum(consumption_values) if consumption_values and sum(consumption_values) > 0 else 0,
                    "total_cooling_usage": sum(cooling_values),
                    "total_domestic_usage": sum(domestic_values),
                    "total_process_usage": sum(process_values)
                },
                "recent_data": [
                    {
                        "date": r.get('date'),
                        "consumption_gallons": r.get('consumption_gallons'),
                        "cost_usd": r.get('cost_usd'),
                        "cost_per_gallon": r.get('cost_per_gallon'),
                        "cooling_usage": r.get('cooling_usage'),
                        "domestic_usage": r.get('domestic_usage'),
                        "process_usage": r.get('process_usage'),
                        "source_type": r.get('source_type'),
                        "quality_rating": r.get('quality_rating'),
                        "seasonal_notes": r.get('seasonal_notes')
                    }
                    for r in record_data[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error fetching water context: {e}")
            return {
                "site": site,
                "period": {"start": start_date, "end": end_date},
                "error": str(e),
                "record_count": 0
            }
