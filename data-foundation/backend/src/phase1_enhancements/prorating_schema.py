import logging
import os
from datetime import datetime
from neo4j.exceptions import TransientError
from langchain_neo4j import Neo4jGraph
import uuid

class ProRatingSchema:
    """
    Schema management class for pro-rating enhancements to ProcessedDocument nodes.
    Creates MonthlyUsageAllocation nodes and relationships to track monthly usage allocations
    for documents spanning multiple months.
    """

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def create_constraints_and_indexes(self):
        """
        Create constraints and indexes for MonthlyUsageAllocation nodes.
        """
        try:
            logging.info("Creating constraints and indexes for pro-rating schema")
            
            # Create constraint for allocation_id if it doesn't exist
            constraint_query = """
                CREATE CONSTRAINT unique_allocation_id IF NOT EXISTS
                FOR (a:MonthlyUsageAllocation) REQUIRE a.allocation_id IS UNIQUE
            """
            self.graph.query(constraint_query, session_params={"database": self.graph._database})
            
            # Create composite index for year/month queries
            year_month_index_query = """
                CREATE INDEX idx_allocation_year_month IF NOT EXISTS
                FOR (a:MonthlyUsageAllocation) ON (a.usage_year, a.usage_month)
            """
            self.graph.query(year_month_index_query, session_params={"database": self.graph._database})
            
            # Create index for allocation percentage for analytics
            percentage_index_query = """
                CREATE INDEX idx_allocation_percentage IF NOT EXISTS
                FOR (a:MonthlyUsageAllocation) ON (a.allocation_percentage)
            """
            self.graph.query(percentage_index_query, session_params={"database": self.graph._database})
            
            # Create index for allocated usage for cost queries
            usage_index_query = """
                CREATE INDEX idx_allocated_usage IF NOT EXISTS
                FOR (a:MonthlyUsageAllocation) ON (a.allocated_usage)
            """
            self.graph.query(usage_index_query, session_params={"database": self.graph._database})
            
            logging.info("Constraints and indexes created successfully")
            
        except Exception as e:
            error_message = f"Error creating constraints and indexes: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def create_allocation_schema(self):
        """
        Create the complete pro-rating allocation schema including constraints and indexes.
        """
        try:
            logging.info("Creating complete pro-rating allocation schema")
            
            # Step 1: Create constraints and indexes
            self.create_constraints_and_indexes()
            
            # Step 2: Verify the schema changes
            validation_result = self.validate_allocation_schema()
            
            result = {
                "constraints_created": True,
                "indexes_created": True,
                "schema_validation_passed": validation_result
            }
            
            logging.info(f"Pro-rating schema creation completed: {result}")
            return result
            
        except Exception as e:
            error_message = f"Error creating pro-rating schema: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def validate_allocation_schema(self):
        """
        Validate that the pro-rating schema is properly created.
        Returns True if validation passes, False otherwise.
        """
        try:
            logging.info("Validating pro-rating schema")
            
            # Check if constraints exist
            constraint_check_query = """
                SHOW CONSTRAINTS
                YIELD name, type, entityType, labelsOrTypes, properties
                WHERE name CONTAINS 'allocation'
                RETURN count(*) as constraint_count
            """
            
            constraint_result = self.graph.query(constraint_check_query, session_params={"database": self.graph._database})
            constraint_count = constraint_result[0]['constraint_count'] if constraint_result else 0
            
            # Check if indexes exist
            index_check_query = """
                SHOW INDEXES
                YIELD name, type, entityType, labelsOrTypes, properties
                WHERE name CONTAINS 'allocation'
                RETURN count(*) as index_count
            """
            
            index_result = self.graph.query(index_check_query, session_params={"database": self.graph._database})
            index_count = index_result[0]['index_count'] if index_result else 0
            
            # Schema is valid if we have at least the constraint and some indexes
            validation_passed = constraint_count >= 1 and index_count >= 3
            
            logging.info(f"Schema validation - Constraints: {constraint_count}, Indexes: {index_count}, Passed: {validation_passed}")
            return validation_passed
                
        except Exception as e:
            error_message = f"Error validating pro-rating schema: {str(e)}"
            logging.error(error_message)
            return False

    def create_monthly_allocation(self, document_file_name, usage_year, usage_month, 
                                allocated_usage, allocated_cost, allocation_percentage,
                                days_in_month, days_allocated):
        """
        Create a MonthlyUsageAllocation node and link it to a ProcessedDocument.
        
        Args:
            document_file_name: The fileName of the ProcessedDocument
            usage_year: Year of the allocation
            usage_month: Month of the allocation (1-12)
            allocated_usage: The allocated usage amount for this month
            allocated_cost: The allocated cost for this month
            allocation_percentage: Percentage of total usage allocated to this month (0-100)
            days_in_month: Total days in the month
            days_allocated: Number of days allocated to this month
            
        Returns:
            Dictionary containing allocation information
        """
        try:
            logging.info(f"Creating monthly allocation for document: {document_file_name}, {usage_year}-{usage_month}")
            
            # Validate input parameters
            if not (1 <= usage_month <= 12):
                raise ValueError(f"Invalid month: {usage_month}. Must be between 1 and 12")
            
            if not (0 <= allocation_percentage <= 100):
                raise ValueError(f"Invalid allocation percentage: {allocation_percentage}. Must be between 0 and 100")
            
            if days_allocated > days_in_month:
                raise ValueError(f"Days allocated ({days_allocated}) cannot exceed days in month ({days_in_month})")
            
            # Generate unique allocation ID
            allocation_id = str(uuid.uuid4())
            current_time = datetime.now().isoformat()
            
            # Create allocation node and relationship
            create_query = """
                MATCH (d:Document {fileName: $document_file_name})
                CREATE (a:MonthlyUsageAllocation {
                    allocation_id: $allocation_id,
                    usage_year: $usage_year,
                    usage_month: $usage_month,
                    allocated_usage: $allocated_usage,
                    allocated_cost: $allocated_cost,
                    allocation_percentage: $allocation_percentage,
                    days_in_month: $days_in_month,
                    days_allocated: $days_allocated,
                    created_at: $created_at,
                    updated_at: $updated_at
                })
                CREATE (d)-[:HAS_MONTHLY_ALLOCATION]->(a)
                RETURN a.allocation_id as allocation_id,
                       a.usage_year as year,
                       a.usage_month as month,
                       a.allocated_usage as usage,
                       a.allocated_cost as cost
            """
            
            params = {
                "document_file_name": document_file_name,
                "allocation_id": allocation_id,
                "usage_year": usage_year,
                "usage_month": usage_month,
                "allocated_usage": allocated_usage,
                "allocated_cost": allocated_cost,
                "allocation_percentage": allocation_percentage,
                "days_in_month": days_in_month,
                "days_allocated": days_allocated,
                "created_at": current_time,
                "updated_at": current_time
            }
            
            result = self.graph.query(create_query, params, session_params={"database": self.graph._database})
            
            if result:
                allocation_info = result[0]
                logging.info(f"Successfully created allocation: {allocation_id}")
                return allocation_info
            else:
                raise Exception(f"Failed to create allocation - document may not exist: {document_file_name}")
                
        except Exception as e:
            error_message = f"Error creating monthly allocation for {document_file_name}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_allocations_by_document(self, document_file_name):
        """
        Retrieve all monthly allocations for a specific document.
        
        Args:
            document_file_name: The fileName of the ProcessedDocument
            
        Returns:
            List of allocation dictionaries
        """
        try:
            query = """
                MATCH (d:Document {fileName: $document_file_name})-[:HAS_MONTHLY_ALLOCATION]->(a:MonthlyUsageAllocation)
                RETURN a.allocation_id as allocation_id,
                       a.usage_year as usage_year,
                       a.usage_month as usage_month,
                       a.allocated_usage as allocated_usage,
                       a.allocated_cost as allocated_cost,
                       a.allocation_percentage as allocation_percentage,
                       a.days_in_month as days_in_month,
                       a.days_allocated as days_allocated,
                       a.created_at as created_at,
                       a.updated_at as updated_at
                ORDER BY a.usage_year DESC, a.usage_month DESC
            """
            
            param = {"document_file_name": document_file_name}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            return result
            
        except Exception as e:
            error_message = f"Error retrieving allocations for document {document_file_name}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_allocations_by_month(self, usage_year, usage_month):
        """
        Retrieve all monthly allocations for a specific year and month.
        
        Args:
            usage_year: Year to query
            usage_month: Month to query (1-12)
            
        Returns:
            List of allocation dictionaries with document information
        """
        try:
            if not (1 <= usage_month <= 12):
                raise ValueError(f"Invalid month: {usage_month}. Must be between 1 and 12")
            
            query = """
                MATCH (d:Document)-[:HAS_MONTHLY_ALLOCATION]->(a:MonthlyUsageAllocation)
                WHERE a.usage_year = $usage_year AND a.usage_month = $usage_month
                RETURN d.fileName as document_file_name,
                       d.status as document_status,
                       a.allocation_id as allocation_id,
                       a.allocated_usage as allocated_usage,
                       a.allocated_cost as allocated_cost,
                       a.allocation_percentage as allocation_percentage,
                       a.days_in_month as days_in_month,
                       a.days_allocated as days_allocated,
                       a.created_at as created_at
                ORDER BY a.allocated_cost DESC
            """
            
            params = {"usage_year": usage_year, "usage_month": usage_month}
            result = self.graph.query(query, params, session_params={"database": self.graph._database})
            
            return result
            
        except Exception as e:
            error_message = f"Error retrieving allocations for {usage_year}-{usage_month}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def update_monthly_allocation(self, allocation_id, allocated_usage=None, allocated_cost=None, 
                                allocation_percentage=None, days_allocated=None):
        """
        Update an existing monthly allocation.
        
        Args:
            allocation_id: The unique ID of the allocation to update
            allocated_usage: New allocated usage amount (optional)
            allocated_cost: New allocated cost (optional)
            allocation_percentage: New allocation percentage (optional)
            days_allocated: New days allocated count (optional)
        """
        try:
            logging.info(f"Updating monthly allocation: {allocation_id}")
            
            # Build dynamic SET clause based on provided parameters
            set_clauses = []
            params = {"allocation_id": allocation_id, "updated_at": datetime.now().isoformat()}
            
            if allocated_usage is not None:
                set_clauses.append("a.allocated_usage = $allocated_usage")
                params["allocated_usage"] = allocated_usage
                
            if allocated_cost is not None:
                set_clauses.append("a.allocated_cost = $allocated_cost")
                params["allocated_cost"] = allocated_cost
                
            if allocation_percentage is not None:
                if not (0 <= allocation_percentage <= 100):
                    raise ValueError(f"Invalid allocation percentage: {allocation_percentage}. Must be between 0 and 100")
                set_clauses.append("a.allocation_percentage = $allocation_percentage")
                params["allocation_percentage"] = allocation_percentage
                
            if days_allocated is not None:
                set_clauses.append("a.days_allocated = $days_allocated")
                params["days_allocated"] = days_allocated
            
            if not set_clauses:
                logging.warning("No properties to update")
                return
                
            set_clauses.append("a.updated_at = $updated_at")
            
            query = f"""
                MATCH (a:MonthlyUsageAllocation {{allocation_id: $allocation_id}})
                SET {', '.join(set_clauses)}
                RETURN a.allocation_id as updated_allocation
            """
            
            result = self.graph.query(query, params, session_params={"database": self.graph._database})
            
            if result:
                logging.info(f"Successfully updated allocation: {allocation_id}")
            else:
                logging.warning(f"Allocation not found: {allocation_id}")
                
        except Exception as e:
            error_message = f"Error updating allocation {allocation_id}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def delete_monthly_allocation(self, allocation_id):
        """
        Delete a monthly allocation and its relationship.
        
        Args:
            allocation_id: The unique ID of the allocation to delete
        """
        try:
            logging.info(f"Deleting monthly allocation: {allocation_id}")
            
            query = """
                MATCH (a:MonthlyUsageAllocation {allocation_id: $allocation_id})
                DETACH DELETE a
                RETURN count(a) as deleted_count
            """
            
            param = {"allocation_id": allocation_id}
            result = self.graph.query(query, param, session_params={"database": self.graph._database})
            
            deleted_count = result[0]['deleted_count'] if result else 0
            
            if deleted_count > 0:
                logging.info(f"Successfully deleted allocation: {allocation_id}")
            else:
                logging.warning(f"Allocation not found: {allocation_id}")
                
            return deleted_count
                
        except Exception as e:
            error_message = f"Error deleting allocation {allocation_id}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def get_monthly_usage_summary(self, usage_year, usage_month):
        """
        Get aggregated usage summary for a specific month.
        
        Args:
            usage_year: Year to query
            usage_month: Month to query (1-12)
            
        Returns:
            Dictionary with aggregated usage statistics
        """
        try:
            if not (1 <= usage_month <= 12):
                raise ValueError(f"Invalid month: {usage_month}. Must be between 1 and 12")
            
            query = """
                MATCH (a:MonthlyUsageAllocation)
                WHERE a.usage_year = $usage_year AND a.usage_month = $usage_month
                RETURN count(a) as total_allocations,
                       sum(a.allocated_usage) as total_allocated_usage,
                       sum(a.allocated_cost) as total_allocated_cost,
                       avg(a.allocation_percentage) as avg_allocation_percentage,
                       max(a.allocated_cost) as max_allocation_cost,
                       min(a.allocated_cost) as min_allocation_cost
            """
            
            params = {"usage_year": usage_year, "usage_month": usage_month}
            result = self.graph.query(query, params, session_params={"database": self.graph._database})
            
            if result:
                return result[0]
            else:
                return {
                    "total_allocations": 0,
                    "total_allocated_usage": 0,
                    "total_allocated_cost": 0,
                    "avg_allocation_percentage": 0,
                    "max_allocation_cost": 0,
                    "min_allocation_cost": 0
                }
            
        except Exception as e:
            error_message = f"Error getting usage summary for {usage_year}-{usage_month}: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)

    def migrate_existing_documents_to_allocations(self):
        """
        Migration helper to create monthly allocations for existing documents
        that don't have allocations but could benefit from pro-rating.
        This is a placeholder for future migration logic.
        """
        try:
            logging.info("Starting migration of existing documents to allocation schema")
            
            # This would contain logic to:
            # 1. Find documents spanning multiple months based on their date ranges
            # 2. Calculate appropriate allocations based on document metadata
            # 3. Create MonthlyUsageAllocation nodes for these documents
            
            # For now, just return a placeholder result
            result = {
                "migration_started": True,
                "documents_processed": 0,
                "allocations_created": 0,
                "errors": []
            }
            
            logging.info(f"Migration completed: {result}")
            return result
            
        except Exception as e:
            error_message = f"Error during migration: {str(e)}"
            logging.error(error_message)
            raise Exception(error_message)