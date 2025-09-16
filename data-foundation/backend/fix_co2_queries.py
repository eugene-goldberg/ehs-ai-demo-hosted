import re

# Read the file
with open('src/services/context_retriever.py', 'r') as f:
    content = f.read()

# Fix the CO2 goals query
old_query1 = """co2_goals_query = \"\"\"
            MATCH (g:Goal)
            WHERE g.site_id = \ 
            AND (toLower(g.description) CONTAINS 'co2' OR toLower(g.description) CONTAINS 'carbon' 
                 OR toLower(g.description) CONTAINS 'emission' OR g.unit CONTAINS 'CO2')
            RETURN g.id as goal_id, g.category as category, g.description as description,
                   g.target_value as target_value, g.unit as unit, g.target_date as target_date,
                   g.baseline_year as baseline_year, g.target_year as target_year,
                   g.period as period, g.created_at as created_at, g.updated_at as updated_at
            ORDER BY g.target_date DESC
            \"\"\""""

new_query1 = """co2_goals_query = \"\"\"
            MATCH (g:Goal)
            WHERE g.site_id = \ AND (toLower(g.description) CONTAINS 'co2' OR toLower(g.description) CONTAINS 'carbon' OR toLower(g.description) CONTAINS 'emission' OR g.unit CONTAINS 'CO2')
            RETURN g.id as goal_id, g.category as category, g.description as description,
                   g.target_value as target_value, g.unit as unit, g.target_date as target_date,
                   g.baseline_year as baseline_year, g.target_year as target_year,
                   g.period as period, g.created_at as created_at, g.updated_at as updated_at
            ORDER BY g.target_date DESC
            \"\"\""""

content = content.replace(old_query1, new_query1)

# Fix the environmental targets query
old_query2 = """env_targets_query = \"\"\"
            MATCH (et:EnvironmentalTarget)
            WHERE et.site_id = \ 
            AND (toLower(et.description) CONTAINS 'co2' OR toLower(et.description) CONTAINS 'carbon' 
                 OR toLower(et.description) CONTAINS 'emission' OR toLower(et.target_type) CONTAINS 'emission')
            RETURN et.id as target_id, et.target_type as target_type, et.description as description,
                   et.target_value as target_value, et.target_unit as target_unit, 
                   et.deadline as deadline, et.status as status,
                   et.created_at as created_at, et.updated_at as updated_at
            ORDER BY et.deadline DESC
            \"\"\""""

new_query2 = """env_targets_query = \"\"\"
            MATCH (et:EnvironmentalTarget)
            WHERE et.site_id = \ AND (toLower(et.description) CONTAINS 'co2' OR toLower(et.description) CONTAINS 'carbon' OR toLower(et.description) CONTAINS 'emission' OR toLower(et.target_type) CONTAINS 'emission')
            RETURN et.id as target_id, et.target_type as target_type, et.description as description,
                   et.target_value as target_value, et.target_unit as target_unit, 
                   et.deadline as deadline, et.status as status,
                   et.created_at as created_at, et.updated_at as updated_at
            ORDER BY et.deadline DESC
            \"\"\""""

content = content.replace(old_query2, new_query2)

# Write back
with open('src/services/context_retriever.py', 'w') as f:
    f.write(content)

print('Fixed CO2 query syntax')
