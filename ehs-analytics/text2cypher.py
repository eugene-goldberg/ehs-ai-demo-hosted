
    def _enhance_query_with_context(self, query: str, query_type: QueryType) -> str:
        """Simple query enhancement without complex context."""
        return query

    def _execute_cypher_chain(self, query: str) -> Dict[str, Any]:
        """Execute the Cypher chain with proper error handling."""
        try:
            result = self.cypher_chain.invoke({"question": query})
            return result
        except Exception as e:
            logger.error("Cypher chain execution failed", error=str(e))
            raise
