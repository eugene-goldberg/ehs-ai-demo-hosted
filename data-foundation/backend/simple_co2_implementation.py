import re

# Read the original file
with open('src/services/context_retriever.py', 'r') as f:
    content = f.read()

# Simple, working CO2 method that follows the exact pattern of other methods
co2_method = '''
    def get_co2_goals_context(self, site: Optional[str], 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, Any]:
        """Fetch CO2 goals and emissions data from Neo4j for Algonquin IL and Houston TX only"""
        
        # Map site names to Neo4j site IDs - only support Algonquin IL and Houston TX
        site_mapping = {
            'houston_texas': 'houston_tx',
            'houston_tx': 'houston_tx', 
            'houston': 'houston_tx',
            'algonquin_illinois': 'algonquin_il',
            'algonquin_il': 'algonquin_il',
            'algonquin': 'algonquin_il'
        }
        
        if site and site.lower() in site_mapping:
            site = site_mapping[site.lower()]
        
        # Only allow supported sites
        if site not in ['algonquin_il', 'houston_tx']:
            return {
                "site": site,
                "period": {"start": start_date, "end": end_date},
                "error": f"CO2 goals data only available for Algonquin IL and Houston TX. Requested site: {site}",
                "record_count": 0
            }
        
        try:
            # Query for all Goal nodes for the site (including CO2-related ones)
            goals_query = """
            MATCH (g:Goal)
            WHERE g.site_id = 
            RETURN g.id as goal_id, g.category as category, g.description as description,
                   g.target_value as target_value, g.unit as unit, g.target_date as target_date,
                   g.baseline_year as baseline_year, g.target_year as target_year,
                   g.period as period, g.created_at as created_at, g.updated_at as updated_at
            ORDER BY g.target_date DESC
            """
            
            # Query for all Environmental Targets for the site
            targets_query = """
            MATCH (et:EnvironmentalTarget)
            WHERE et.site_id = 
            RETURN et.id as target_id, et.target_type as target_type, et.description as description,
                   et.target_value as target_value, et.target_unit as target_unit, 
                   et.deadline as deadline, et.status as status,
                   et.created_at as created_at, et.updated_at as updated_at
            ORDER BY et.deadline DESC
            """
            
            params = {"site": site}
            
            # Execute queries
            goals = self._execute_query(goals_query, params)
            targets = self._execute_query(targets_query, params)
            
            # Convert records to dictionaries if needed
            if goals and hasattr(goals[0], 'data'):
                goals_data = [record.data() for record in goals]
            else:
                goals_data = goals if goals else []
                
            if targets and hasattr(targets[0], 'data'):
                targets_data = [record.data() for record in targets]
            else:
                targets_data = targets if targets else []
            
            # Filter for CO2-related goals and targets
            co2_goals = []
            for goal in goals_data:
                desc = goal.get('description', '').lower()
                unit = goal.get('unit', '').lower()
                category = goal.get('category', '').lower()
                if ('co2' in desc or 'carbon' in desc or 'emission' in desc or 
                    'co2' in unit or category == 'electricity'):
                    co2_goals.append(goal)
            
            co2_targets = []
            for target in targets_data:
                desc = target.get('description', '').lower()
                ttype = target.get('target_type', '').lower()
                if ('co2' in desc or 'carbon' in desc or 'emission' in desc or 
                    'emission' in ttype):
                    co2_targets.append(target)
            
            # Filter by date range if provided
            if start_date or end_date:
                filtered_goals = []
                for goal in co2_goals:
                    goal_date = goal.get('target_date')
                    if goal_date:
                        if start_date and goal_date < start_date:
                            continue
                        if end_date and goal_date > end_date:
                            continue
                    filtered_goals.append(goal)
                co2_goals = filtered_goals
                
                filtered_targets = []
                for target in co2_targets:
                    target_date = target.get('deadline')
                    if target_date:
                        if start_date and target_date < start_date:
                            continue
                        if end_date and target_date > end_date:
                            continue
                    filtered_targets.append(target)
                co2_targets = filtered_targets
            
            if not co2_goals and not co2_targets:
                return {
                    "site": site,
                    "period": {"start": start_date, "end": end_date},
                    "message": "No CO2 goals or targets found",
                    "record_count": 0
                }
            
            # Calculate aggregates
            co2_reduction_targets = []
            electricity_reduction_targets = []
            
            for goal in co2_goals:
                if goal.get('unit') == 'tonnes CO2e':
                    co2_reduction_targets.append(goal.get('target_value', 0))
                elif goal.get('category') == 'electricity':
                    electricity_reduction_targets.append(goal.get('target_value', 0))
            
            # Group targets by type
            target_types = {}
            for target in co2_targets:
                ttype = target.get('target_type', 'Unknown')
                if ttype not in target_types:
                    target_types[ttype] = {'count': 0, 'in_progress': 0, 'completed': 0, 'planning': 0}
                target_types[ttype]['count'] += 1
                status = target.get('status', 'unknown')
                if status in target_types[ttype]:
                    target_types[ttype][status] += 1
            
            return {
                "site": site,
                "site_id": site,
                "period": {
                    "start": start_date,
                    "end": end_date
                },
                "record_count": len(co2_goals) + len(co2_targets),
                "summary": {
                    "co2_goals_count": len([g for g in co2_goals if g.get('unit') == 'tonnes CO2e']),
                    "electricity_goals_count": len([g for g in co2_goals if g.get('category') == 'electricity']),
                    "environmental_targets_count": len(co2_targets),
                    "total_co2_reduction_target": sum(co2_reduction_targets),
                    "total_electricity_reduction_target": sum(electricity_reduction_targets),
                    "target_types_breakdown": target_types
                },
                "co2_goals": [
                    {
                        "goal_id": g.get('goal_id'),
                        "category": g.get('category'),
                        "description": g.get('description'),
                        "target_value": g.get('target_value'),
                        "unit": g.get('unit'),
                        "target_date": g.get('target_date'),
                        "baseline_year": g.get('baseline_year'),
                        "target_year": g.get('target_year'),
                        "period": g.get('period'),
                        "created_at": g.get('created_at'),
                        "updated_at": g.get('updated_at')
                    }
                    for g in co2_goals
                ],
                "environmental_targets": [
                    {
                        "target_id": t.get('target_id'),
                        "target_type": t.get('target_type'),
                        "description": t.get('description'),
                        "target_value": t.get('target_value'),
                        "target_unit": t.get('target_unit'),
                        "deadline": t.get('deadline'),
                        "status": t.get('status'),
                        "created_at": t.get('created_at'),
                        "updated_at": t.get('updated_at')
                    }
                    for t in co2_targets
                ]
            }
            
        except Exception as e:
            logger.error(f"Error fetching CO2 goals context: {e}")
            return {
                "site": site,
                "period": {"start": start_date, "end": end_date},
                "error": str(e),
                "record_count": 0
            }
'''

# Insert the CO2 method before get_context_for_intent function
pattern = r'(\n\ndef get_context_for_intent)'
replacement = co2_method + r'\1'
content = re.sub(pattern, replacement, content)

# Update the get_context_for_intent function to handle co2_goals intent
old_intent = r"(elif intent_type\.lower\(\) == 'waste_generation':[\s]*context = retriever\.get_waste_context\(site_filter, start_date, end_date\))"
new_intent = r"\1\n        elif intent_type.lower() == 'co2_goals':\n            context = retriever.get_co2_goals_context(site_filter, start_date, end_date)"
content = re.sub(old_intent, new_intent, content)

# Write the updated content
with open('src/services/context_retriever.py', 'w') as f:
    f.write(content)

print('Successfully implemented simple CO2 goals method')
