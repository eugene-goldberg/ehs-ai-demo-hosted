# This document outlines the goals and the logic of the Executve Dashboard

> Created: 2025-08-31
> Version: 1.0.0

## The golas of the Executive Dashboard

1   Clearly display the anuual Environmental Health & Safety (EHS) corporate goals
    Annual % reduction of CO2 emissions. These emissions are calculated based on the Electricity consumtion
    Annual % reduction of water consumtion
    Annual % reduction of waste generation
    These % number must be displayed prominently and clearly at the very top of the dashboard
2   For every category (Electricity Consulmption, Water Consumption, Waste Generation) this
     dashboard will display the following inforation: a) The facts (i.e. aggregated data 'to date'), b) Risk assessment (as calculated by the LLM through comparing the corresponding annual gol to the current facts, calculating the trends, and assessing the feasibility of reaching the annuals goals based on the current facts) c) Risk mitigation recommendations (proposed by the LLM based on the assessed risks)

## The logic which must be implemented to enable the goals

1   The Neo4j database is pre-populated with 6 month of historical data, which represents the 
    electricity consumption, water consumption and waste generation for each of the two sites: Algonquin Illinois and Houston Taxes
2   The Risk Assessment Agent evaluates the risks for each site / each category by doing the
    following:
        a) reading the corresponding Neo4j factual data (6 month of hostorical data) for a given
           site
        b) reading the corresponding annual goal for each site / each category
        c) converting the electricity consumption into CO2 emmissions (since the annual goals
           are expressed as CO2 emission reduction %)
3   The Risk Assessemnt Agent will record the assessed risks for each site / category into a
    corresponding record in Neo4j
4   Based on the assessed risks for each site / category the Risk Assessment Agent will provide
    risk mitigation recommendations and will record these recommendations into the corresponding records within Neo4j so the relationships between each site and their sorresponding risks and risk mitigating recommendations will be 100% clear
5   When the Executive Dashboard is being opened / displayed it will call the corresponding
    API / service, which will retrieve the facts, the risks and the recommendations for each site. The dashboard will display this data using the current layout