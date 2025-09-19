    def get_recommendations_context(self, site: Optional[str], 
                                  category: Optional[str] = None,
                                  priority: Optional[str] = None) -> Dict[str, Any]:
        """Fetch recommendations data from Neo4j for Algonquin IL and Houston TX only"""
        
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
                "filters": {"category": category, "priority": priority},
                "error": f"Recommendations data only available for Algonquin IL and Houston TX. Requested site: {site}",
                "record_count": 0
            }
        
        # Updated query to retrieve the correct fields from Neo4j
        query = """
        MATCH (r:Recommendation)
        WHERE r.site_id = $site
        RETURN r.recommendations as recommendations, 
               r.total_recommendations as total_recommendations,
               r.created_date as created_date, 
               r.site_id as site_id
        ORDER BY r.created_date DESC
        LIMIT 1
        """
        
        params = {"site": site}
        
        try:
            records = self._execute_query(query, params)
            
            if not records:
                return {
                    "site": site,
                    "site_id": site,
                    "filters": {"category": category, "priority": priority},
                    "message": "No recommendations found",
                    "record_count": 0
                }
            
            # Convert records to dictionaries if needed
            if hasattr(records[0], 'data'):
                # Neo4j Record objects
                record_data = records[0].data()
            else:
                # Already dictionaries from Neo4j client
                record_data = records[0]
            
            # Parse the JSON string to get the recommendations array
            recommendations_json = record_data.get('recommendations', '[]')
            try:
                recommendations_list = json.loads(recommendations_json)
                if not isinstance(recommendations_list, list):
                    logger.warning(f"Expected list but got {type(recommendations_list)}: {recommendations_list}")
                    recommendations_list = []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse recommendations JSON: {e}")
                logger.error(f"Raw JSON string: {recommendations_json}")
                return {
                    "site": site,
                    "site_id": site,
                    "filters": {"category": category, "priority": priority},
                    "error": f"Failed to parse recommendations data: {str(e)}",
                    "record_count": 0
                }
            
            # Apply filters to the recommendations
            filtered_recommendations = []
            for rec in recommendations_list:
                # Apply category filter if specified
                if category and rec.get('category', '').lower() != category.lower():
                    continue
                
                # Apply priority filter if specified  
                if priority and rec.get('priority', '').lower() != priority.lower():
                    continue
                
                filtered_recommendations.append(rec)
            
            # Calculate summary statistics
            all_categories = [rec.get('category', 'unknown') for rec in filtered_recommendations]
            all_priorities = [rec.get('priority', 'unknown') for rec in filtered_recommendations]
            all_timelines = [rec.get('timeline', rec.get('implementation_timeline', 'unknown')) for rec in filtered_recommendations]
            all_efforts = [rec.get('effort_level', 'unknown') for rec in filtered_recommendations]
            
            # Group by various attributes
            category_breakdown = {}
            for cat in all_categories:
                category_breakdown[cat] = category_breakdown.get(cat, 0) + 1
            
            priority_breakdown = {}
            for pri in all_priorities:
                priority_breakdown[pri] = priority_breakdown.get(pri, 0) + 1
            
            timeline_breakdown = {}
            for timeline in all_timelines:
                timeline_breakdown[timeline] = timeline_breakdown.get(timeline, 0) + 1
            
            effort_breakdown = {}
            for effort in all_efforts:
                effort_breakdown[effort] = effort_breakdown.get(effort, 0) + 1
            
            # Format recommendations for output
            formatted_recommendations = []
            for rec in filtered_recommendations:
                formatted_rec = {
                    "id": rec.get('id'),
                    "title": rec.get('title'),
                    "description": rec.get('description'),
                    "category": rec.get('category'),
                    "expected_impact": rec.get('expected_impact'),
                    "timeline": rec.get('timeline', rec.get('implementation_timeline')),
                    "effort_level": rec.get('effort_level'),
                    "priority": rec.get('priority'),
                    "estimated_cost": rec.get('estimated_cost'),
                    "roi_timeframe": rec.get('roi_timeframe'),
                    "industry_citation": rec.get('industry_citation', rec.get('citation'))
                }
                formatted_recommendations.append(formatted_rec)
            
            return {
                "site": site,
                "site_id": site,
                "filters": {
                    "category": category,
                    "priority": priority
                },
                "record_count": len(filtered_recommendations),
                "total_recommendations": record_data.get('total_recommendations', len(recommendations_list)),
                "created_date": record_data.get('created_date'),
                "summary": {
                    "total_recommendations": len(filtered_recommendations),
                    "category_breakdown": category_breakdown,
                    "priority_breakdown": priority_breakdown,
                    "timeline_breakdown": timeline_breakdown,
                    "effort_breakdown": effort_breakdown,
                    "high_priority_count": priority_breakdown.get('High', priority_breakdown.get('high', 0)),
                    "medium_priority_count": priority_breakdown.get('Medium', priority_breakdown.get('medium', 0)),
                    "low_priority_count": priority_breakdown.get('Low', priority_breakdown.get('low', 0))
                },
                "recommendations": formatted_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error fetching recommendations context: {e}")
            return {
                "site": site,
                "filters": {"category": category, "priority": priority},
                "error": str(e),
                "record_count": 0
            }
